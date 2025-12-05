from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    rescale_noise_cfg,
)
try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
from .inference_step import inference_step
from ..utils import image_postprocess
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps


def latent_to_image(self, latents): # self is a pipeline
    if hasattr(self.vae.config, "shift_factor"):
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    else:  # AutoencoderTiny
        latents = latents / self.vae.config.scaling_factor
    # latents is ~ [-3, 3]
    image = self.vae.decode(latents, return_dict=False)[0] # in [-1, 1]
    image = torch.clamp((image + 1) / 2, 0., 1.) # [-1, 1] to [0, 1]
    return image

@torch.inference_mode()
def sample_trajectory(self, tsfm_model_to_use,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "image",
        # return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,

        guidance_rexp=-1.0,
        reward_fn=None,
        prompt_metadata=None,
        return_output=True,
 ):
    assert output_type in ["image", "latent", "pil"]
    height = height or self.default_sample_size * self.vae_scale_factor # 128 * 8
    width = width or self.default_sample_size * self.vae_scale_factor # 128 * 8

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    with torch.no_grad():
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip, # which is be self._clip_skip
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            # lora_scale=lora_scale, # old version does not have this input
        )

        if self.do_classifier_free_guidance:
            prompt_embeds_comb = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds_comb = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 4. Prepare timesteps
    """
    retrieve_timesteps:
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    
    timesteps = tensor([1000.0000, 987.3806, 974.1077, 960.1293, 945.3875, 929.8179,
            913.3490, 895.9003, 877.3819, 857.6923, 836.7167, 814.3248,
            790.3683, 764.6771, 737.0558, 707.2785, 675.0823, 640.1602,
            602.1506, 560.6250, 515.0721, 464.8760, 409.2888, 347.3926,
            278.0488, 199.8270, 110.9057, 8.9286], device='cuda:0')
    sigmas = concat(timesteps / 1000., "0.")
    """

    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0) # 0
    self._num_timesteps = len(timesteps)

    # 5. Prepare latent variables
    # num_channels_latents = self.transformer.config.in_channels
    num_channels_latents = tsfm_model_to_use.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    all_latents = [latents]
    all_outputs = []
    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            # the model output should actually be velocity
            with torch.inference_mode():
                # noise_pred = self.transformer(
                noise_pred = tsfm_model_to_use(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_comb,
                    pooled_projections=pooled_prompt_embeds_comb,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform cfg
                if self.do_classifier_free_guidance: # if self._guidance_scale > 1
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


            # compute the previous noisy sample x_t -> x_t-1
            """ what happens in scheduler.step():
            sigma = self.sigmas[self.step_index]
            sigma_next = self.sigmas[self.step_index + 1]
            prev_sample = sample + (sigma_next - sigma) * model_output
            -- In SD3, model output is negative velocity
            
            latent_model_input[:1] + (self.scheduler.sigmas[1] - self.scheduler.sigmas[0]) * noise_pred # == latents
            self.scheduler.sigmas[1] - self.scheduler.sigmas[0] is negative value
            """
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            all_latents.append(latents)
            if return_output:
                all_outputs.append(noise_pred)

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    if output_type == "latent":
        image = latents
    else:
        image = latent_to_image(self, latents).detach()
        if output_type == "pil":
            # tensor to list of PIL; do_denormalize is True by default
            image = self.image_processor.postprocess(image,
                output_type=output_type, do_denormalize=[False for _ in image])

    # Offload all models
    self.maybe_free_model_hooks()

    if return_output:
        return (image, all_latents, timesteps, prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, all_outputs)
    else:
        # if not return_dict:
        return (image, all_latents, timesteps, prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_prompt_embeds, None)
