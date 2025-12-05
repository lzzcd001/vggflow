import os
from collections import defaultdict
import contextlib
import datetime
import time
import wandb
from functools import partial
import tempfile
from PIL import Image
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
import logging
import copy
import pickle, gzip

import math

import diffusers
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers import UNet2DConditionModel





from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module

from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lib.distributed import init_distributed_singlenode, set_seed, setup_for_distributed

import lib.reward_func.prompts
import lib.reward_func.rewards
from lib.diffusion.sample_trajectory_sd3 import sample_trajectory
from lib.diffusion.inference_step import inference_step, predict_clean, get_alpha_prod_t


from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

from torch.nn.attention import SDPBackend, sdpa_kernel

import torch.nn.functional as F
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor2_0
from contextlib import ExitStack

# Optional: use the pure math SDPA (more robust for higher-order grads)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAVE_SDPA_BACKEND = True
except Exception:
    HAVE_SDPA_BACKEND = False

from functools import wraps

def softclamp(x, beta: float = 10.0, low: float = -1.0, high: float = 1.0):
    return low + F.softplus(beta*(x - low), beta=1.0) / beta \
               - F.softplus(beta*(x - high), beta=1.0) / beta

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_integer("seed", 0, "Seed.")


def latent_to_image(self, latents, clamp=True): # self is a pipeline
    latents = latents.to(torch.float32)
    if hasattr(self.vae.config, "shift_factor"):
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    else:  # AutoencoderTiny
        latents = latents / self.vae.config.scaling_factor
    # latents is ~ [-3, 3]
    image = self.vae.decode(latents, return_dict=False)[0] # in [-1, 1]
    if clamp:
        image = torch.clamp((image + 1) / 2, 0., 1.) # [-1, 1] to [0, 1]
    else:
        image = softclamp((image + 1) / 2, beta=10.0) # [-1, 1] to [0, 1]
    return image

def feedforward(config, autocast, transformer, latent, timestep, prompt_embeds, pooled_prompt_embeds):
    latent_model_input = torch.cat([latent] * 2) if config.sampling.guidance_scale > 1. else latent
    timestep = timestep.repeat(2) if config.sampling.guidance_scale > 1. else timestep
    with autocast():
        velocity = transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
    if config.sampling.guidance_scale > 1.:
        velocity_uncond, velocity_text = velocity.chunk(2)
        velocity = velocity_uncond + config.sampling.guidance_scale * (velocity_text - velocity_uncond)
    return velocity

def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def process_info_dict(info, result, logger, config, global_step, num_processes, epoch):
    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
    for k, v in info.items():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
    info = {k: v / num_processes for k, v in info.items()}
    for k, v in info.items():
        result[k][global_step] = v.item()

    info.update({"epoch": epoch})
    if get_local_rank() == 0:
        logger.info(f"global_step={global_step}  " +
                    " ".join([f"{k}={v:.3f}" for k, v in info.items()]))
        if config['wandb']:
            wandb.log(info, step=global_step)
    info = defaultdict(list)  # reset info dict
    return info, result

def main(args):
    if FLAGS.config.training.use_jvp:
        with sdpa_kernel(SDPBackend.MATH):
            train()
    else:
        train()

def setup(local_rank, is_local_main_process):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    config = FLAGS.config

    config.gpu_type = torch.cuda.get_device_name() \
                            if torch.cuda.is_available() else "CPU"
    if is_local_main_process:
        logger.info(f"GPU type: {config.gpu_type}")

    if FLAGS.seed is not None:
        config.seed = FLAGS.seed
    else:
        config.seed = 0

    wandb_name = f"{config.experiment.reward_fn.split('_')[0]}_vggflow_{FLAGS.exp_name}_seed{config.seed}"


    if config.logging.use_wandb:
        wandb_key = config.logging.wandb_key
        wandb.login(key=wandb_key)
        wandb.init(project=config.logging.proj_name, name=wandb_name, config=config.to_dict(),
           dir=config.logging.wandb_dir,
           save_code=True, mode="online" if is_local_main_process else "disabled")

    save_dir = os.path.join(config.saving.output_dir, wandb_name)
    os.makedirs(save_dir, exist_ok=True)

    if is_local_main_process:
        logger.info(f"\n{config}")
    set_seed(config.seed)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = torch.device(local_rank)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision, torch_dtype=weight_dtype,
    )

    if config.pretrained.autoencodertiny: # faster SD3 decoding
        if is_local_main_process:
            pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=weight_dtype)
        dist.barrier()
        pipeline.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=weight_dtype)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=weight_dtype)
    pipeline.text_encoder_2.to(device, dtype=weight_dtype)
    pipeline.text_encoder_3.to(device, dtype=weight_dtype)
    pipeline.to(device)
    pipeline.scheduler.set_timesteps(config.sampling.num_steps, device=device)


    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )


    transformer = pipeline.transformer
    transformer.requires_grad_(False)


    if config.model.use_value_net:
        if config.model.value_net_param == 'lora':
            value_net = copy.deepcopy(pipeline.transformer)
            value_net.train()
            for name, param in transformer.named_parameters():
                param.requires_grad_(False)
            value_net.to(device, dtype=weight_dtype)
            value_net_lora_config = LoraConfig(
                r=config.model.lora_rank, lora_alpha=config.model.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            value_net.add_adapter(value_net_lora_config, adapter_name="default") ## LoRA
            value_net.set_adapter("default")
            value_net.proj_out.weight.requires_grad_(True)
            value_net.proj_out.weight.data *= 1e-5
            value_net.to(device)
        elif config.model.value_net_param == 'copy':
            value_net = copy.deepcopy(pipeline.transformer)
            value_net.proj_out.weight.requires_grad_(True)
            value_net.proj_out.weight.data *= 1e-5
            value_net.to(device)
        elif config.model.value_net_param == 'small':
            value_net = UNet2DConditionModel(
                in_channels=16, out_channels=16, block_out_channels=config.model.flow_channel_width,
                layers_per_block=config.model.value_layers_per_block, 
                cross_attention_dim=4096,
            )
            value_net.conv_out.weight.requires_grad_(True)
            value_net.conv_out.weight.data *= 1e-3
            value_net.to(device, dtype=weight_dtype)
        else:
            raise NotImplementedError
    
        value_net_params = value_net.parameters()

        value_net = DDP(value_net, device_ids=[local_rank])
    else:
        value_net = None


    for name, param in transformer.named_parameters():
        param.requires_grad_(False)
    transformer.to(device, dtype=weight_dtype)
    transformer_lora_config = LoraConfig(
        r=config.model.lora_rank, lora_alpha=config.model.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config, adapter_name="default") ## LoRA

    transformer.set_adapter("default")
    if config.training.mixed_precision in ["fp16", "bf16"]:
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(transformer, dtype=torch.float32)
        if config.model.use_value_net:
            if config.model.value_net_param == 'lora' or config.model.value_net_param == 'copy':
                cast_training_params(value_net, dtype=torch.float32)

    pf_params = [param for name, param in transformer.named_parameters() if '.default.' in name]

    if config.training.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.cuda.amp.GradScaler(
            growth_interval=config.training.gradscaler_growth_interval
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 is True by default
        torch.backends.cudnn.benchmark = True

    optimizer_cls = torch.optim.AdamW

    transformer.to(device)
    transformer = DDP(transformer, device_ids=[local_rank])

    #######################################################
    params = [
        {"params": pf_params, "lr": config.training.lr},
    ]
    if config.model.use_value_net:
        params.append({
            "params": value_net_params, "lr": config.training.lr,
        })

    optimizer = optimizer_cls(
        params,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    return config, pipeline, optimizer, transformer, logger, scaler, value_net, save_dir


def train():
    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    config, pipeline, optimizer, transformer, logger, scaler, value_net, save_dir = setup(local_rank, is_local_main_process)

    device = torch.device(local_rank)

    # prepare prompt and reward fn
    prompt_fn = getattr(lib.reward_func.prompts, config.experiment.prompt_fn)
    reward_fn = getattr(lib.reward_func.rewards, config.experiment.reward_fn)(torch.float32, device)

    autocast = contextlib.nullcontext # LoRA weights are actually float32, but other part of SD are in bf16/fp16
    ref_compute_mode = torch.inference_mode

    result = defaultdict(dict)
    result["config"] = config.to_dict()
    start_time = time.time()

    #######################################################
    # Start!
    samples_per_epoch = (
        config.sampling.batch_size * num_processes
        * config.sampling.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.training.batch_size * num_processes
        * config.training.gradient_accumulation_steps
    )

    if is_local_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {config.training.num_epochs}")
        logger.info(f"  Sample batch size per device = {config.sampling.batch_size}")
        logger.info(f"  Train batch size per device = {config.training.batch_size}")
        logger.info(
            f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
        )
        logger.info("")
        logger.info(f"  Total number of samples per epoch = test_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
        )
        logger.info(
            f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
        )
        logger.info(f"  Number of inner epochs = {config.training.num_inner_epochs}")

    assert config.sampling.batch_size >= config.training.batch_size
    assert config.sampling.batch_size % config.training.batch_size == 0 # not necessary
    assert samples_per_epoch % total_train_batch_size == 0

    first_epoch = 0
    # first_epoch = -1 ## epoch -1 only to collect data; training starts from epoch 0
    global_step = 0
    curr_samples = None

    num_inference_steps = config.sampling.num_steps
    scheduler_dt = pipeline.scheduler.timesteps[0] - pipeline.scheduler.timesteps[1]
    num_train_timesteps = int(num_inference_steps * config.model.timestep_fraction)
    if num_train_timesteps != num_inference_steps:
        num_train_timesteps += 1
    accumulation_steps = config.training.gradient_accumulation_steps * num_train_timesteps

    for epoch in range(first_epoch, config.training.num_epochs):

        #################### SAMPLING ####################
        torch.cuda.empty_cache()
        transformer.eval()
        transformer.zero_grad()

        samples = []
        prompts = []
        with torch.inference_mode():
            # pipeline.transformer.set_attn_processor({name: proc for name, proc in orig_attn_processors.items()})
            for i in tqdm(
                range(config.sampling.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not is_local_main_process,
                position=0,
            ):
                # generate prompts
                prompts, prompt_metadata = zip(
                    *[
                        prompt_fn(**config.experiment.prompt_fn_kwargs)
                        for _ in range(config.sampling.batch_size)
                    ]
                )
                prompts = list(prompts)

                # sample
                with autocast():
                    images, latents, timesteps, prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, unet_outputs = sample_trajectory(
                        pipeline, transformer.module, prompt=prompts, negative_prompt=None,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=config.sampling.guidance_scale,
                        output_type="image",
                        return_output=config.model.unet_reg_scale > 0.,
                    )

                latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                if config.model.unet_reg_scale > 0.:
                    unet_outputs = torch.stack(unet_outputs, dim=1)
                timesteps = pipeline.scheduler.timesteps.repeat(
                    config.sampling.batch_size, 1
                )  # (bs, num_steps)  (981, 961, ..., 21, 1) corresponds to "next_latents"

                step_index = torch.arange(timesteps.size(1), device=timesteps.device, dtype=torch.int64).view(1, -1).expand(timesteps.size(0), -1)

                rewards = reward_fn(images.float(), prompts, prompt_metadata) # (reward, reward_metadata)
                samples.append(
                    {
                        "prompts": prompts, # tuple of strings
                        "prompt_metadata": prompt_metadata,
                        "prompt_embeds": prompt_embeds,
                        "pooled_prompt_embeds": pooled_prompt_embeds,
                        "negative_prompt_embeds": negative_prompt_embeds,
                        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents,
                        "rewards": rewards,
                        "step_index": step_index,
                    }
                )
                if config.model.unet_reg_scale > 0:
                    samples[-1]["unet_outputs"] = unet_outputs

            # wait for all rewards to be computed
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not is_local_main_process,
                position=0,
            ):
                rewards, reward_metadata = sample["rewards"]
                sample["rewards"] = torch.as_tensor(rewards, device=device)

            # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
            new_samples = {}
            for k in samples[0].keys():
                if k in ["prompts", "prompt_metadata"]:
                    # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                    new_samples[k] = [item for s in samples for item in s[k]]
                else:
                    new_samples[k] = torch.cat([s[k] for s in samples])
            samples = new_samples

            if epoch >= 0:
                # this is a hack to force wandb to log the images as JPEGs instead of PNGs
                with tempfile.TemporaryDirectory() as tmpdir:
                    for i, image in enumerate(images):
                        pil = Image.fromarray(
                            (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                        pil = pil.resize((256, 256))
                        pil.save(os.path.join(tmpdir, f"{i}.jpg"))
                    if config.logging.use_wandb and is_local_main_process:
                        wandb.log(
                            {
                                "images": [
                                    wandb.Image(
                                        os.path.join(tmpdir, f"{i}.jpg"),
                                        caption=f"{prompt} | {reward:.2f}",
                                    )
                                    for i, (prompt, reward) in enumerate(
                                        zip(prompts, rewards)
                                    )
                                ],
                            },
                            step=global_step,
                        )

                rewards = torch.zeros(world_size * len(samples["rewards"]),
                            dtype=samples["rewards"].dtype, device=device)
                dist.all_gather_into_tensor(rewards, samples["rewards"])
                rewards = rewards.detach().cpu().float().numpy()
                result["reward_mean"][global_step] = rewards.mean()
                result["reward_std"][global_step] = rewards.std()

                if is_local_main_process:
                    logger.info(f"global_step: {global_step}  rewards: {rewards.mean().item():.3f}")
                    if config.logging.use_wandb:
                        wandb.log(
                            {
                                "reward_mean": rewards.mean(),
                                "reward_std": rewards.std(),
                            },
                            step=global_step,
                        )

                total_batch_size, num_timesteps = samples["timesteps"].shape
                assert (
                    total_batch_size
                    == config.sampling.batch_size * config.sampling.num_batches_per_epoch
                )
                assert num_timesteps == num_inference_steps
            
        rgrad_threshold = 1.0

        ### No sampling for Epoch -1 
        if curr_samples is None:
            curr_samples = samples
            continue
        
        curr_samples = samples

        #################### TRAINING ####################
        for inner_epoch in range(config.training.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=device)
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    curr_samples[k] = [v[i] for i in perm]
                else:
                    curr_samples[k] = v[perm]

            if config.model.timestep_fraction < 1:
                if config.sampling.low_var_subsampling:
                    n_trunks = int(num_inference_steps * config.model.timestep_fraction)
                    assert n_trunks >= 1, "Must have at least one trunk"
                    assert num_inference_steps % n_trunks == 0, "num_inference_steps must be divisible by n_trunks"

                    trunk_size = num_inference_steps // n_trunks
                    step_indices = torch.arange(num_inference_steps, device=device)
                    trunks = step_indices.view(n_trunks, trunk_size)  # shape: (n_trunks, trunk_size)

                    # Precompute trunk access pattern (reversed order, repeated)
                    trunk_order = list(reversed(range(n_trunks))) * trunk_size  # len = num_inference_steps

                    perms_list = []
                    for _ in range(total_batch_size):
                        tmp = []
                        for i in trunk_order:
                            trunk = trunks[i]
                            index = torch.randint(0, trunk_size, (1,))
                            tmp.append(trunk[index])
                        interleaved = torch.cat(tmp)
                        perms_list.append(torch.cat([torch.tensor([num_inference_steps - 1], device=device), interleaved]))

                    perms = torch.stack(perms_list)  # shape: (batch_size, 1 + chunk_size * n_trunks)
                else:
                    perms = torch.stack(
                        [
                            torch.randperm(num_timesteps - 1, device=device)
                            for _ in range(total_batch_size)
                        ]
                    ) # (total_batch_size, num_steps)
                    perms = torch.cat([num_timesteps - 1 + torch.zeros_like(perms[:, :1]), perms], dim=1)
            else:
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps, device=device)
                        for _ in range(total_batch_size)
                    ]
                ) # (total_batch_size, num_steps)

            perms = perms.clamp(min=1)

            # "prompts" & "prompt_metadata" are constant along time dimension
            key_ls = ["timesteps", "latents", "step_index"]
            curr_samples['last_latent'] = curr_samples['latents'][:, -1]
            for key in key_ls:
                curr_samples[key] = curr_samples[key][torch.arange(total_batch_size, device=device)[:, None], perms]
            if config.model.unet_reg_scale > 0:
                curr_samples["unet_outputs"] = \
                    curr_samples["unet_outputs"][torch.arange(total_batch_size, device=device)[:, None], perms]

            ### rebatch for training
            samples_batched = {}
            for k, v in curr_samples.items():
                if k in ["prompts", "prompt_metadata"]:
                    samples_batched[k] = [v[i:i + config.training.batch_size]
                                for i in range(0, len(v), config.training.batch_size)]
                else:
                    samples_batched[k] = v.reshape(-1, config.training.batch_size, *v.shape[1:])

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_local_main_process,
            ):
                """
                sample: [
                ('prompts', list of strings, len=train_bs), ('prompt_metadata', list of dicts),
                (bf16) ('prompt_embeds', torch.Size([1, 77, 768])),
                (int64) ('timesteps', torch.Size([1, 50])),
                (bf16) ('latents', torch.Size([1, 50, 4, 64, 64])), ('next_latents', torch.Size([1, 50, 4, 64, 64])),
                ]
                """

                sigma_last = pipeline.scheduler.sigmas[-1]
                buffer = []
                for step in tqdm(range(num_train_timesteps), desc="Timestep", position=1, leave=False, disable=not is_local_main_process):
                    j = step
                    with autocast():

                        xt = sample["latents"][:, step].detach()

                        guidance_rexp = config.model.reward_scale
                        sigma = pipeline.scheduler.sigmas.gather(0, sample["step_index"][:, step]).view(-1, 1, 1, 1)
                        with torch.enable_grad():
                            xinp = xt.detach()
                            xinp.requires_grad_()
                            xinp_model_input = torch.cat([xinp] * 2) if pipeline.do_classifier_free_guidance else xinp

                            timestep = sample["timesteps"][:, step].repeat(2) if config.sampling.guidance_scale > 1. else sample["timesteps"][:, step]
                            prompt_embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]], dim=0)
                            pooled_prompt_embeds = torch.cat([sample["negative_pooled_prompt_embeds"], sample["pooled_prompt_embeds"]], dim=0)

                            # use the current transformer to predict x1
                            noise_pred_forguide = transformer(
                                hidden_states=xinp_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                return_dict=False,
                            )[0]

                            if pipeline.do_classifier_free_guidance:  # if self._guidance_scale > 1
                                noise_pred_uncond_forguide, noise_pred_text_forguide = noise_pred_forguide.chunk(2)
                                noise_pred_uncond_forguide = noise_pred_uncond_forguide.detach() # stop grad on unconditional bracnh
                                noise_pred_forguide = noise_pred_uncond_forguide + config.sampling.guidance_scale * (
                                            noise_pred_text_forguide - noise_pred_uncond_forguide)
                                noise_pred_uncond_forguide = noise_pred_text_forguide = None


                            latents_last = xinp + (sigma_last - sigma) * noise_pred_forguide.detach()
                            latents_last = latents_last.to(torch.float32)
                            bs, c, h, w = latents_last.size()
                            n_jitter = 1
                            std_jitter = 0.0
                            latents_last_jitter = (latents_last.unsqueeze(1)
                                + std_jitter * torch.randn(bs, n_jitter, c, h, w, device=latents_last.device)
                            ).view(-1, c, h, w)
                            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                                image_from_latent = latent_to_image(pipeline, latents_last_jitter)
                                latents_last = None
                                rewards, _ = reward_fn(image_from_latent.float(), sample["prompts"], sample["prompt_metadata"])
                                image_from_latent = None
                                rgrad = torch.autograd.grad(rewards.sum(), xinp, retain_graph=True)[0].detach().float() / n_jitter
                                with torch.no_grad():
                                    rgrad_norm = torch.linalg.norm(rgrad.view(rgrad.size(0), -1), dim=1)
                                    if config.training.quantile_clipping:
                                        rgrad = rgrad / (rgrad_norm.view(-1, 1, 1, 1) + 1e-8) * rgrad_norm.view(-1, 1, 1, 1).clamp(max=rgrad_threshold)
                                reward_mask = (rewards >= 0).float()
                                rewards = None
                                del rewards


                        with torch.inference_mode():
                            transformer.module.disable_adapters()  # Re-apply LoRA configuration
                            velocity_target = velocity_target_raw = feedforward(config, autocast,
                                                        transformer,
                                                xt, sample["timesteps"][:, step], prompt_embeds, pooled_prompt_embeds)

                            if config.model.eta_mode == 'linear':
                                velocity_target = velocity_target - (1 - sigma) * guidance_rexp * rgrad.float()  #
                            elif config.model.eta_mode == 'constant':
                                velocity_target = velocity_target - guidance_rexp * rgrad.float()  #
                            else:
                                velocity_target = velocity_target - (1 - sigma).pow(2) * guidance_rexp * rgrad.float() 

                        transformer.module.enable_adapters()  # Re-apply LoRA configuration
                        transformer.module.set_adapter("default")
                        if config.model.use_value_net:
                            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                                if config.training.use_jvp:
                                    timestep_var = sample["timesteps"][:, step].float().detach().clone().requires_grad_(True)
                                    if config.model.value_net_param == 'small':
                                        value_correction = value_net(
                                            sample=xt.float(),
                                            timestep=timestep_var,
                                            encoder_hidden_states=prompt_embeds[:xt.size(0)].float()
                                        ).sample
                                    else:
                                        value_correction = value_net(
                                            hidden_states=xt.float(),
                                            timestep=timestep_var,
                                            encoder_hidden_states=prompt_embeds[:xt.size(0)].float(),
                                            pooled_projections=pooled_prompt_embeds[:xt.size(0)].float(),
                                            joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                            return_dict=False,
                                        )[0]
                                else:
                                    if config.model.value_net_param == 'small':
                                        value_correction = value_net(
                                            sample=xt.float(),
                                            timestep=sample["timesteps"][:, step],
                                            encoder_hidden_states=prompt_embeds[:xt.size(0)].float()
                                        ).sample
                                    else:
                                        value_correction = value_net(
                                            hidden_states=xt.float(),
                                            timestep=sample["timesteps"][:, step],
                                            encoder_hidden_states=prompt_embeds[:xt.size(0)].float(),
                                            pooled_projections=pooled_prompt_embeds[:xt.size(0)].float(),
                                            joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                            return_dict=False,
                                        )[0]

                            velocity_target = velocity_target + value_correction
                            
                            


                        with torch.enable_grad():
                            velocity = noise_pred_forguide

                            if config.model.unet_reg_scale > 0:
                                unetdiff = (velocity - sample["unet_outputs"][:, j]).pow(2)
                                unetreg = torch.mean(unetdiff, dim=(1, 2, 3))
                                unetdiffnorm = unetdiff.sum(dim=[1,2,3]).sqrt()

                    with torch.autocast(dtype=torch.float32, device_type="cuda"):
                        if config.training.reward_masking:
                            loss_raw = ((velocity - velocity_target.detach()).float().pow(2).mean(dim=[1,2,3]) * reward_mask).sum() / (reward_mask.sum() + 1e-8)
                        else:
                            loss_raw = (velocity - velocity_target.detach()).float().pow(2).mean()
                        velocity = velocity_target = None

                        if config.model.unet_reg_scale > 0:
                            loss = loss_raw + config.model.unet_reg_scale * unetreg.mean()
                        else:
                            loss = loss_raw
                    
                    velocity = noise_pred_forguide = None
                        

                    if config.model.use_value_net:
                        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                            eps_step = 1e-4
                            eps_timestep = 1e-4

                            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                                # notice that "timestep" is in diffusion convention, i.e. from T to 0
                                if config.model.value_net_param == 'small':
                                    value_correction_tinc = value_net(
                                        sample=xt,
                                        timestep=sample["timesteps"][:, step] - eps_timestep * torch.ones_like(sample["timesteps"][:, step]),
                                        encoder_hidden_states=prompt_embeds[:xt.size(0)] 
                                    ).sample 
                                else:
                                    value_correction_tinc = value_net(
                                        hidden_states=xt,
                                        timestep=sample["timesteps"][:, step] - eps_timestep * torch.ones_like(sample["timesteps"][:, step]),
                                        encoder_hidden_states=prompt_embeds[:xt.size(0)] ,
                                        pooled_projections=pooled_prompt_embeds[:xt.size(0)] ,
                                        joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                        return_dict=False,
                                    )[0]

                            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                                value_correction = value_correction.float()
                                if config.model.eta_mode == 'linear':
                                    nabla_V = (value_correction / guidance_rexp - (1 - sigma) * rgrad)
                                elif config.model.eta_mode == 'constant':
                                    nabla_V = (value_correction / guidance_rexp - rgrad)
                                else:
                                    nabla_V = (value_correction / guidance_rexp - (1 - sigma).pow(2) * rgrad)
                                rgrad = None


                            xinp_copy = xt.detach()
                            xinp_copy.requires_grad_()
                            xinp_copy_model_input = torch.cat([xinp_copy] * 2) if pipeline.do_classifier_free_guidance else xinp

                            tinc = timestep - eps_timestep * torch.ones_like(timestep)
                            noise_pred_forguide_tinc = transformer(
                                hidden_states=xinp_copy_model_input,
                                timestep=tinc,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                return_dict=False,
                            )[0]

                            if pipeline.do_classifier_free_guidance:  # if self._guidance_scale > 1
                                noise_pred_uncond_forguide_tinc, noise_pred_text_forguide_tinc = noise_pred_forguide_tinc.chunk(2)
                                noise_pred_uncond_forguide_tinc = noise_pred_uncond_forguide_tinc.detach() # stop grad on unconditional bracnh
                                noise_pred_forguide_tinc = noise_pred_uncond_forguide_tinc + config.sampling.guidance_scale * (
                                            noise_pred_text_forguide_tinc - noise_pred_uncond_forguide_tinc)
                                noise_pred_uncond_forguide_tinc = noise_pred_text_forguide_tinc = None


                            tinc_scale = tinc[:xinp.size(0)].view(-1, 1, 1, 1) / 1000
                            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                                latents_last_tinc = xinp - tinc_scale * noise_pred_forguide_tinc
                                noise_pred_forguide_tinc = None
                                image_from_latent_tinc = latent_to_image(pipeline, latents_last_tinc.to(torch.float32))
                                latents_last_tinc = None
                                rewards_tinc, _ = reward_fn(image_from_latent_tinc.float(), sample["prompts"], sample["prompt_metadata"])
                                image_from_latent_tinc = None
                                rgrad_tinc = torch.autograd.grad(rewards_tinc.sum(), xinp_copy, retain_graph=True)[0].detach().float()
                                with torch.no_grad():
                                    rgrad_inc_norm = torch.linalg.norm(rgrad_tinc.view(rgrad_tinc.size(0), -1), dim=1)
                                    if config.training.quantile_clipping:
                                        rgrad_tinc = rgrad_tinc / (rgrad_inc_norm.view(-1, 1, 1, 1) + 1e-8) * rgrad_inc_norm.view(-1, 1, 1, 1).clamp(max=rgrad_threshold)
                                xinp_copy = xinp_copy_model_input = None
                                rewards_tinc = None
                                del rewards_tinc

                            if config.model.eta_mode == 'linear':
                                nabla_V_tinc = (value_correction_tinc / guidance_rexp - (1 - tinc_scale) * rgrad_tinc)
                            elif config.model.eta_mode == 'constant':
                                nabla_V_tinc = (value_correction_tinc / guidance_rexp - rgrad_tinc)
                            else:
                                nabla_V_tinc = (value_correction_tinc / guidance_rexp - (1 - tinc_scale).pow(2) * rgrad_tinc)


                            ## HACK: T = 1000
                            value_correction_tinc = rgrad_tinc = None
                            del_time = (nabla_V_tinc - nabla_V) / eps_timestep
                            nabla_V_tinc = None


                            if config.training.detach_dir:
                                g_dir = (velocity_target_raw / guidance_rexp + nabla_V).detach()
                            else:
                                g_dir = (velocity_target_raw / guidance_rexp + nabla_V)

                            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                                if config.model.value_net_param == 'small':
                                    value_correction_xinc = value_net(
                                        sample=xt + g_dir * eps_step,
                                        timestep=sample["timesteps"][:, step],
                                        encoder_hidden_states=prompt_embeds[:xt.size(0)]
                                    ).sample
                                else:
                                    value_correction_xinc = value_net(
                                        hidden_states=xt + g_dir * eps_step,
                                        timestep=sample["timesteps"][:, step],
                                        encoder_hidden_states=prompt_embeds[:xt.size(0)] ,
                                        pooled_projections=pooled_prompt_embeds[:xt.size(0)] ,
                                        joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                        return_dict=False,
                                    )[0]


                            xinc = (xt + g_dir * eps_step).detach()
                            xinc.requires_grad_()
                            xinc_model_input = torch.cat([xinc] * 2) if pipeline.do_classifier_free_guidance else xinp
                            noise_pred_forguide_xinc = transformer(
                                hidden_states=xinc_model_input,
                                timestep=timestep,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_prompt_embeds,
                                joint_attention_kwargs=pipeline.joint_attention_kwargs,
                                return_dict=False,
                            )[0]
                            xinc_model_input = None

                            if pipeline.do_classifier_free_guidance:  # if self._guidance_scale > 1
                                noise_pred_uncond_forguide_xinc, noise_pred_text_forguide_xinc = noise_pred_forguide_xinc.chunk(2)
                                noise_pred_uncond_forguide_xinc = noise_pred_uncond_forguide_xinc.detach() # stop grad on unconditional bracnh
                                noise_pred_forguide_xinc = noise_pred_uncond_forguide_xinc + config.sampling.guidance_scale * (
                                            noise_pred_text_forguide_xinc - noise_pred_uncond_forguide_xinc)
                                noise_pred_uncond_forguide_xinc = noise_pred_text_forguide_xinc = None


                            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                                latents_last_xinc = xinc - timestep[:xinc.size(0)].view(-1, 1, 1, 1) / 1000 * noise_pred_forguide_xinc
                                noise_pred_forguide_xinc = None
                                image_from_latent_xinc = latent_to_image(pipeline, latents_last_xinc.to(torch.float32))
                                latents_last_xinc = None
                                rewards_xinc, _ = reward_fn(image_from_latent_xinc.float(), sample["prompts"], sample["prompt_metadata"])
                                image_from_latent_xinc = None
                                rgrad_xinc = torch.autograd.grad(rewards_xinc.sum(), xinc, retain_graph=False)[0].detach().float()
                                with torch.no_grad():
                                    rgrad_xinc_norm = torch.linalg.norm(rgrad_xinc.view(rgrad_xinc.size(0), -1), dim=1)
                                    rgrad_xinc = rgrad_xinc / (rgrad_xinc_norm.view(-1, 1, 1, 1) + 1e-8) * rgrad_xinc_norm.view(-1, 1, 1, 1).clamp(max=rgrad_threshold)
                                xinc = None
                                rewards_xinc = None
                                del rewards_xinc


                            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                                if config.model.eta_mode == 'linear':
                                    nabla_V_xinc_scaled = (value_correction_xinc - (1 - sigma) * guidance_rexp * rgrad_xinc)
                                elif config.model.eta_mode == 'constant':
                                    nabla_V_xinc_scaled = (value_correction_xinc - guidance_rexp * rgrad_xinc)
                                else:
                                    nabla_V_xinc_scaled = (value_correction_xinc - (1 - sigma).pow(2) * guidance_rexp * rgrad_xinc)

                                value_correction_xinc = rgrad_xinc = None
                                fd2 = (nabla_V_xinc_scaled - nabla_V * guidance_rexp) / eps_step
                                nabla_V_xinc_scaled = None

                            transformer.module.disable_adapters()  # Re-apply LoRA configuration
                            vo_inc = feedforward(config, autocast,
                                    transformer,
                                    xt + nabla_V * eps_step, sample["timesteps"][:, step], prompt_embeds, pooled_prompt_embeds
                            )
                            transformer.module.enable_adapters()  # Re-apply LoRA configuration
                            transformer.module.set_adapter("default")
                            with torch.autocast(dtype=torch.float32, device_type="cuda"):
                                fd1 = (vo_inc.float() - velocity_target_raw) / eps_step
                            nabla_V = None

                            loss_consistency = ((del_time - fd1 - fd2) / guidance_rexp).float().pow(2).mean()
                            del_time = fd1 = fd2 = None

                        end_mask = sample["timesteps"][:, step] == pipeline.scheduler.timesteps[-1]
                        info["loss_consistency"].append(loss_consistency)

                        with torch.autocast(dtype=torch.float32, device_type="cuda"):
                            loss_terminal = (value_correction.float().pow(2).mean(dim=[1,2,3]) * end_mask.float()).sum() / (end_mask.float().sum() + 1e-6)
                        value_correction_terminal = xt = None
                        info["loss_terminal"].append(loss_terminal)
                        loss = loss + loss_consistency + loss_terminal * config.training.coeff_terminal

                    if config.model.unet_reg_scale > 0:
                        loss = loss + unetreg.mean()

                    loss = loss / accumulation_steps
                    if scaler:
                        # Backward passes under autocast are not recommended
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()


                    #### Log
                    info["loss_raw"].append(loss_raw)
                    info["loss"].append(loss)
                    with torch.inference_mode():
                        info["rgrad_mean"].append(rgrad_norm.mean())
                        info["rgrad_min"].append(rgrad_norm.min())
                        info["rgrad_max"].append(rgrad_norm.max())
                        info["rgrad_all_08quantile"].append(rgrad_norm)
                        info["rgrad_all_median"].append(rgrad_norm)
                        info["rgrad_all_std"].append(rgrad_norm)

                    if config.model.unet_reg_scale > 0:
                        try:
                            info["unetreg"].append(unetreg.mean().detach())
                        except:
                            pass


                    # prevent OOM
                    image = None
                    noise_pred_uncond = noise_pred_text = noise_pred = None
                    logr_next_tmp = logr_tmp = None
                    _ = log_pf = log_pb = None
                    unetreg = losses =  None
                    noise_pred_uncond_ref = noise_pred_text_ref = noise_pred_ref = None
                    score_pf_target = None
                    grad_norm_score_pf_target = grad_norm_score_pf_reverse_target = None



                if ((j == num_train_timesteps - 1) and
                        (i + 1) % config.training.gradient_accumulation_steps == 0):
                    if scaler:
                        scaler.unscale_(optimizer)
                        pf_update_grad = torch.nn.utils.clip_grad_norm_(transformer.parameters(), config.training.max_grad_norm)
                        if config.model.use_value_net:
                            vnet_grad = torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.training.max_grad_norm)

                        scaler.step(optimizer)
                        # optimizer.step()
                        scaler.update()
                    else:
                        pf_update_grad = torch.nn.utils.clip_grad_norm_([p for name, p in transformer.named_parameters() if '.default.' in name], config.training.max_grad_norm)
                        if config.model.use_value_net:
                            vnet_grad = torch.nn.utils.clip_grad_norm_(value_net.parameters(), config.training.max_grad_norm)

                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    ### avoid memory leak
                    for param in transformer.parameters():
                        param.grad = None

                    cache = {}
                    old_info = info
                    info = {}
                    for k, v in old_info.items():
                        if '_min' in k:
                            info[k] = torch.min(torch.stack(v))
                        elif '_max' in k:
                            info[k] = torch.max(torch.stack(v))
                        elif '_all' in k:
                            info[k] = torch.stack(v)
                            cache[k] = [torch.zeros_like(info[k])] * num_processes
                        else:
                            try:
                                info[k] = torch.mean(torch.stack(v))
                            except Exception as e:
                                print(k)
                                print(v)
                                raise e

                    new_info = {}
                    dist.barrier()
                    for k, v in info.items():
                        if '_min' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MIN)
                        elif '_max' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MAX)
                        elif '_median' in k:
                            dist.all_gather(cache[k], v)
                            new_info[k.replace('_all', '')] = torch.median(cache[k][local_rank])
                        elif '_08quantile' in k:
                            dist.all_gather(cache[k], v)
                            new_info[k.replace('_all', '')] = torch.quantile(cache[k][local_rank], 0.8)
                        elif '_std' in k:
                            dist.all_gather(cache[k], v)
                            new_info[k.replace('_all', '')] = torch.std(cache[k][local_rank])
                        else:
                            dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    
                    for k in list(info.keys()):
                        if '_all' in k:
                            info.pop(k, None)
                    for k in new_info.keys():
                        info[k] = new_info[k]
                    info = {k: v / num_processes if ('_min' not in k and '_max' not in k) else v for k, v in info.items()}
                    for k, v in info.items():
                        result[k][global_step] = v.item()
                    
                    rgrad_threshold = info['rgrad_08quantile'].item()

                    info.update({"epoch": epoch})
                    info.update({"global_step": global_step})
                    result["epoch"][global_step] = epoch
                    result["time"][global_step] = time.time() - start_time


                    if is_local_main_process:
                        if scaler:
                            info.update({"grad_scale": scaler.get_scale()})
                            result["grad_scale"] = scaler.get_scale()


                    if is_local_main_process:
                        if config.logging.use_wandb:
                            wandb.log(info, step=global_step)
                        logger.info(f"global_step={global_step}  " +
                              " ".join([f"{k}={v:.6f}" for k, v in info.items()]))
                    info = defaultdict(list) # reset info dict


        curr_samples = samples
        if is_local_main_process:
            pickle.dump(result, gzip.open(os.path.join(save_dir, f"result.json"), 'wb'))
        dist.barrier()

        if epoch % config.logging.save_freq == 0 or epoch == config.training.num_epochs - 1:
            if is_local_main_process:
                save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}")
                unwrapped_transformer = unwrap_model(transformer)
                transformer_lora_layers = get_peft_model_state_dict(unwrapped_transformer, adapter_name='default')
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=save_path, transformer_lora_layers=transformer_lora_layers,
                    is_main_process=is_local_main_process, safe_serialization=True,
                )
                logger.info(f"Saved state to {save_path}")

            dist.barrier()

    if is_local_main_process:
        save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}")
        unwrapped_transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(unwrapped_transformer, adapter_name='default')
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=save_path, transformer_lora_layers=transformer_lora_layers,
            is_main_process=is_local_main_process, safe_serialization=True,
        )
        logger.info(f"Saved state to {save_path}")
    dist.barrier()

    if config.logging.use_wandb and is_local_main_process:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == '__main__':
  app.run(main)