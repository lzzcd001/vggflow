from typing import Optional, Tuple, Union

import math
import torch

try:
    from diffusers.utils import randn_tensor
except ImportError:
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput, DDIMScheduler
from diffusers import DPMSolverSinglestepScheduler


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)

def get_alpha_prod_t(self, timestep, sample):
    # 2. compute alphas, betas
    # self.alphas_cumprod  torch.Size([1000])
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    return alpha_prod_t


def predict_clean(
        self, 
        model_output, 
        sample: torch.FloatTensor, 
        timestep: int, 
        no_jacobian: bool = False,
        strength: float = 1.0
    ):
    with torch.no_grad():
        alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
        alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
        alpha_prod_t = alpha_prod_t.to(sample.dtype)
        beta_prod_t = 1 - alpha_prod_t

        beta_prod_t[timestep == 0] = 0
        alpha_prod_t[timestep == 0] = 1

    if self.config.prediction_type == "epsilon":
        pred_clean_sample = (
            sample - beta_prod_t ** (0.5) * model_output * strength
        ) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_clean_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_clean_sample = (alpha_prod_t ** 0.5) * sample - (
            beta_prod_t ** 0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
            " `v_prediction`"
        )
    return pred_clean_sample