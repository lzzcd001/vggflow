"""VGG-Flow training orchestration."""

import torch
import torch.distributed as dist
from collections import defaultdict
from functools import partial
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


class VGGFlowTrainer:
    """Orchestrates VGG-Flow training loop."""

    def __init__(
        self,
        vggflow_algorithm,
        transformer,
        value_net,
        optimizer,
        scaler,
        config,
        pipeline,
        logger,
        local_rank
    ):
        """
        Initialize VGG-Flow trainer.

        Args:
            vggflow_algorithm: VGGFlowAlgorithm instance
            transformer: Transformer model
            value_net: Value network (or None)
            optimizer: Optimizer
            scaler: Gradient scaler (for mixed precision)
            config: Configuration object
            pipeline: Diffusion pipeline
            logger: Logger instance
            local_rank: Local rank for distributed training
        """
        self.algorithm = vggflow_algorithm
        self.transformer = transformer
        self.value_net = value_net
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config
        self.pipeline = pipeline
        self.logger = logger
        self.local_rank = local_rank
        self.info = defaultdict(list)

    def train_inner_epoch(
        self,
        samples,
        epoch,
        inner_epoch,
        global_step,
        rgrad_threshold,
        num_processes
    ):
        """
        Run one inner training epoch on collected samples.

        Args:
            samples: Dict of sampled trajectories
            epoch: Current epoch number
            inner_epoch: Current inner epoch number
            global_step: Current global step
            rgrad_threshold: Gradient clipping threshold
            num_processes: Number of processes for distributed training

        Returns:
            global_step: Updated global step
            rgrad_threshold: Updated threshold
        """
        total_batch_size = len(samples["prompts"])

        # Shuffle samples
        samples = self._shuffle_samples(samples, total_batch_size)

        # Subsample timesteps if needed
        num_inference_steps = self.config.sampling.num_steps
        num_train_timesteps = self._compute_num_train_timesteps(num_inference_steps)

        if self.config.model.timestep_fraction < 1:
            samples = self._subsample_timesteps(
                samples, total_batch_size, num_inference_steps, num_train_timesteps
            )

        # Add last latent to samples
        samples['last_latent'] = samples['latents'][:, -1]

        # Create batches
        batches = self._create_batches(samples, total_batch_size)

        # Training loop
        self.transformer.train()
        self.info = defaultdict(list)

        accumulation_steps = self.config.training.gradient_accumulation_steps * num_train_timesteps

        for batch_idx, batch in enumerate(tqdm(
            batches,
            desc=f"Epoch {epoch}.{inner_epoch}: training",
            position=0,
            disable=not self._is_main_process()
        )):
            global_step, rgrad_threshold = self._train_on_batch(
                batch, batch_idx, global_step, rgrad_threshold,
                num_train_timesteps, accumulation_steps, num_processes
            )

        return global_step, rgrad_threshold

    def _train_on_batch(
        self,
        batch,
        batch_idx,
        global_step,
        rgrad_threshold,
        num_train_timesteps,
        accumulation_steps,
        num_processes
    ):
        """Train on a single batch across all timesteps."""
        for step_idx in tqdm(
            range(num_train_timesteps),
            desc="Timestep",
            position=1,
            leave=False,
            disable=not self._is_main_process()
        ):
            self._train_single_timestep(
                batch, step_idx, batch_idx, rgrad_threshold, accumulation_steps
            )

            # Update parameters after final timestep in accumulation window
            if ((step_idx == num_train_timesteps - 1) and
                (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0):

                global_step = self._update_parameters(global_step)
                rgrad_threshold = self._aggregate_and_log_metrics(
                    global_step, num_processes
                )

        return global_step, rgrad_threshold

    def _train_single_timestep(
        self, batch, step_idx, batch_idx, rgrad_threshold, accumulation_steps
    ):
        """
        Train on a single timestep.

        This calls the core VGG-Flow algorithm.
        """
        # Extract current state
        xt = batch["latents"][:, step_idx].detach()
        timestep = batch["timesteps"][:, step_idx]
        sigma = self._get_sigma(batch, step_idx)

        # Prepare prompt embeddings
        prompt_embeds = torch.cat(
            [batch["negative_prompt_embeds"], batch["prompt_embeds"]], dim=0
        )
        pooled_prompt_embeds = torch.cat(
            [batch["negative_pooled_prompt_embeds"], batch["pooled_prompt_embeds"]], dim=0
        )

        # Step 1: Compute target velocity using VGG-Flow algorithm
        velocity_target, loss_components = self.algorithm.compute_velocity_target(
            xt=xt,
            timestep=timestep,
            sigma=sigma,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompts=batch["prompts"],
            prompt_metadata=batch["prompt_metadata"],
            rgrad_threshold=rgrad_threshold
        )

        # Step 2: Reuse velocity from compute_velocity_target (matches main: single forward pass)
        velocity_current = loss_components['velocity_current']

        # Step 3: Velocity matching loss
        loss_velocity = self._compute_velocity_loss(
            velocity_current,
            velocity_target,
            loss_components['reward_mask']
        )

        # Step 4: Value consistency loss (if using value net)
        if self.config.model.use_value_net:
            loss_consistency, loss_terminal = (
                self.algorithm.compute_value_consistency_loss(
                    xt=xt,
                    timestep=timestep,
                    sigma=sigma,
                    value_correction=loss_components['value_correction'],
                    nabla_V=loss_components['nabla_V'],
                    velocity_target_raw=loss_components['velocity_target_raw'],
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompts=batch["prompts"],
                    prompt_metadata=batch["prompt_metadata"],
                    rgrad_threshold=rgrad_threshold
                )
            )

            self.info["loss_consistency"].append(loss_consistency.detach())
            self.info["loss_terminal"].append(loss_terminal.detach())

            loss = (
                loss_velocity +
                loss_consistency +
                loss_terminal * self.config.training.coeff_terminal
            )
        else:
            loss = loss_velocity

        # Step 5: UNet regularization (if enabled)
        if self.config.model.unet_reg_scale > 0:
            unet_diff = (velocity_current - batch["unet_outputs"][:, step_idx]).pow(2)
            unet_reg = unet_diff.mean(dim=(1, 2, 3))
            loss = loss + self.config.model.unet_reg_scale * unet_reg.mean()
            try:
                self.info["unetreg"].append(unet_reg.mean().detach())
            except AttributeError:
                pass

        # Step 6: Backward
        loss = loss / accumulation_steps
        self._backward(loss)

        # Step 7: Log metrics
        self._log_step_metrics(loss_velocity, loss, loss_components)

        # Step 8: Release memory to prevent OOM
        velocity_current = None
        velocity_target = None
        loss_components = None

    def _compute_velocity_loss(self, velocity, target, reward_mask):
        """Compute velocity matching loss."""
        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            diff = (velocity - target.detach()).float().pow(2)
            diff_per_sample = diff.mean(dim=[1, 2, 3])

            if self.config.training.reward_masking:
                loss = (diff_per_sample * reward_mask).sum() / (reward_mask.sum() + 1e-8)
            else:
                loss = diff_per_sample.mean()

        return loss

    def _backward(self, loss):
        """Backward pass with optional gradient scaling."""
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _update_parameters(self, global_step):
        """Update parameters and increment global step."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self._clip_gradients()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        # Clear gradients to avoid memory leak
        for param in self.transformer.parameters():
            param.grad = None

        global_step += 1
        return global_step

    def _clip_gradients(self):
        """Clip gradients."""
        torch.nn.utils.clip_grad_norm_(
            self.transformer.parameters(),
            self.config.training.max_grad_norm
        )
        if self.config.model.use_value_net:
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(),
                self.config.training.max_grad_norm
            )

    def _log_step_metrics(self, loss_velocity, loss_scaled, loss_components):
        """Log step-level metrics."""
        self.info["loss_raw"].append(loss_velocity.detach())
        self.info["loss"].append(loss_scaled.detach())

        # Log reward gradient statistics using unclipped norm
        # (matches main version: threshold adapts to actual gradient magnitudes)
        rgrad_norm = loss_components['rgrad_norm']
        with torch.inference_mode():
            self.info["rgrad_mean"].append(rgrad_norm.mean())
            self.info["rgrad_min"].append(rgrad_norm.min())
            self.info["rgrad_max"].append(rgrad_norm.max())
            self.info["rgrad_all_08quantile"].append(rgrad_norm)
            self.info["rgrad_all_median"].append(rgrad_norm)
            self.info["rgrad_all_std"].append(rgrad_norm)

    def _aggregate_and_log_metrics(self, global_step, num_processes):
        """Aggregate metrics across processes and log."""
        # Aggregate metrics
        cache = {}
        old_info = self.info
        aggregated_info = {}

        for k, v in old_info.items():
            if '_min' in k:
                aggregated_info[k] = torch.min(torch.stack(v))
            elif '_max' in k:
                aggregated_info[k] = torch.max(torch.stack(v))
            elif '_all' in k:
                aggregated_info[k] = torch.stack(v)
                cache[k] = [torch.zeros_like(aggregated_info[k]) for _ in range(num_processes)]
            else:
                aggregated_info[k] = torch.mean(torch.stack(v))

        # Distributed reduction
        new_info = {}
        dist.barrier()
        for k, v in aggregated_info.items():
            if '_min' in k:
                dist.all_reduce(v, op=dist.ReduceOp.MIN)
            elif '_max' in k:
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
            elif '_median' in k:
                dist.all_gather(cache[k], v)
                gathered = torch.cat([t.reshape(-1) for t in cache[k]], dim=0)
                new_info[k.replace('_all', '')] = torch.median(gathered)
            elif '_08quantile' in k:
                dist.all_gather(cache[k], v)
                gathered = torch.cat([t.reshape(-1) for t in cache[k]], dim=0)
                new_info[k.replace('_all', '')] = torch.quantile(gathered, 0.8)
            elif '_std' in k:
                dist.all_gather(cache[k], v)
                gathered = torch.cat([t.reshape(-1) for t in cache[k]], dim=0)
                new_info[k.replace('_all', '')] = torch.std(gathered)
            else:
                dist.all_reduce(v, op=dist.ReduceOp.SUM)

        # Average reduced means across processes; keep min/max as-is; drop raw "_all" tensors.
        final_info = {}
        for k, v in aggregated_info.items():
            if '_all' in k:
                continue
            if '_min' in k or '_max' in k:
                final_info[k] = v
            else:
                final_info[k] = v / num_processes

        # Add computed statistics (already global across all ranks; must not be divided)
        final_info.update(new_info)

        # Update rgrad threshold
        rgrad_threshold = final_info.get('rgrad_08quantile', 1.0).item()

        # Add epoch and step info
        if self._is_main_process():
            if self.scaler:
                final_info["grad_scale"] = self.scaler.get_scale()
            final_info["global_step"] = float(global_step)

            # Log to wandb
            if self.config.logging.use_wandb:
                import wandb
                wandb.log(final_info, step=global_step)

            # Log to console
            self.logger.info(f"global_step={global_step}  " +
                           " ".join([f"{k}={v:.6f}" for k, v in final_info.items()]))

        # Reset info dict
        self.info = defaultdict(list)

        return rgrad_threshold

    def _shuffle_samples(self, samples, total_batch_size):
        """Shuffle samples along batch dimension."""
        device = samples["timesteps"].device
        perm = torch.randperm(total_batch_size, device=device)

        shuffled = {}
        for k, v in samples.items():
            if k in ["prompts", "prompt_metadata"]:
                shuffled[k] = [v[i] for i in perm]
            else:
                shuffled[k] = v[perm]

        return shuffled

    def _subsample_timesteps(
        self, samples, total_batch_size, num_inference_steps, num_train_timesteps
    ):
        """Subsample timesteps for training."""
        num_timesteps = num_inference_steps
        device = samples["timesteps"].device

        if self.config.sampling.low_var_subsampling:
            # Low-variance subsampling
            n_trunks = int(num_inference_steps * self.config.model.timestep_fraction)
            assert n_trunks >= 1, "Must have at least one trunk"
            assert num_inference_steps % n_trunks == 0, (
                "num_inference_steps must be divisible by n_trunks"
            )
            trunk_size = num_inference_steps // n_trunks

            step_indices = torch.arange(num_inference_steps, device=device)
            trunks = step_indices.view(n_trunks, trunk_size)

            trunk_order = list(reversed(range(n_trunks))) * trunk_size

            perms_list = []
            for _ in range(total_batch_size):
                tmp = []
                for i in trunk_order:
                    trunk = trunks[i]
                    index = torch.randint(0, trunk_size, (1,))
                    tmp.append(trunk[index])
                interleaved = torch.cat(tmp)
                perms_list.append(
                    torch.cat([torch.tensor([num_inference_steps - 1], device=device), interleaved])
                )
            perms = torch.stack(perms_list)
        else:
            # Random subsampling
            perms = torch.stack([
                torch.randperm(num_timesteps - 1, device=device)
                for _ in range(total_batch_size)
            ])
            perms = torch.cat([
                num_timesteps - 1 + torch.zeros_like(perms[:, :1]),
                perms
            ], dim=1)

        perms = perms.clamp(min=1)

        # Apply permutation
        key_ls = ["timesteps", "latents", "step_index"]
        for key in key_ls:
            samples[key] = samples[key][
                torch.arange(total_batch_size, device=device)[:, None], perms
            ]

        if self.config.model.unet_reg_scale > 0:
            samples["unet_outputs"] = samples["unet_outputs"][
                torch.arange(total_batch_size, device=device)[:, None], perms
            ]

        return samples

    def _create_batches(self, samples, total_batch_size):
        """Create batches from samples."""
        samples_batched = {}

        for k, v in samples.items():
            if k in ["prompts", "prompt_metadata"]:
                samples_batched[k] = [
                    v[i:i + self.config.training.batch_size]
                    for i in range(0, len(v), self.config.training.batch_size)
                ]
            else:
                samples_batched[k] = v.reshape(
                    -1, self.config.training.batch_size, *v.shape[1:]
                )

        # Convert dict of lists to list of dicts
        batches = [
            dict(zip(samples_batched, x))
            for x in zip(*samples_batched.values())
        ]

        return batches

    def _get_sigma(self, batch, step_idx):
        """Get noise level sigma for current step."""
        sigma = self.pipeline.scheduler.sigmas.gather(
            0, batch["step_index"][:, step_idx]
        ).view(-1, 1, 1, 1)
        return sigma

    def _compute_num_train_timesteps(self, num_inference_steps):
        """Compute number of training timesteps."""
        num_train_timesteps = int(num_inference_steps * self.config.model.timestep_fraction)
        if num_train_timesteps != num_inference_steps:
            num_train_timesteps += 1
        return num_train_timesteps

    def _is_main_process(self):
        """Check if this is the main process."""
        return self.local_rank == 0
