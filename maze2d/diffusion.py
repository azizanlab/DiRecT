from typing import Any, Callable, Sequence, Dict, Union, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning.pytorch as pl


def pad_t_like_x(t: Union[float, int, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Reshape scalar or 1D time tensor to broadcast with x: (B,) -> (B, 1, ..., 1)."""
    if isinstance(t, (float, int)):
        t = torch.tensor(t, dtype=x.dtype, device=x.device)
    return t.reshape(-1, *([1] * (x.dim() - 1)))


@torch.no_grad()
def apply_conditioning(
    x: torch.Tensor,
    conditions: Dict[Any, Any],
    action_dim: int = 2,
    zero_conditioning: bool = False,
):
    """Replace observation entries at conditioned timesteps (start/goal).

    For each (timestep, value) pair in conditions, sets x[..., timestep, action_dim:]
    to the condition value (or zeros if zero_conditioning=True).
    """
    for t, val in conditions.items():
        x[..., t, action_dim:] = 0.0 if zero_conditioning else val.clone()
    return x


class CosineNoiseSchedule(nn.Module):
    """Cosine noise schedule: alpha(t) = sin(pi/2 * t/(1+s)), sigma(t) = cos(pi/2 * t/(1+s)).

    Convention: t=0 is the prior (pure noise), t=1 is the data distribution.

    Args:
        offset: Small offset s to avoid singularity at t=0.
    """

    def __init__(self, offset: float = 0.008):
        super().__init__()
        self.offset = offset

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi / 2.0 * t / (1.0 + self.offset))

    def sigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(torch.pi / 2.0 * t / (1.0 + self.offset))

    def dalpha_t(self, t: torch.Tensor) -> torch.Tensor:
        return (
            torch.pi
            / (2.0 * (self.offset + 1.0))
            * torch.cos(torch.pi / 2 * t / (1 + self.offset))
        )

    def dsigma_t(self, t: torch.Tensor) -> torch.Tensor:
        return (
            -torch.pi
            / (2.0 * (self.offset + 1.0))
            * torch.sin(torch.pi / 2 * t / (1 + self.offset))
        )

    def inverse_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return (2.0 / torch.pi) * (1.0 + self.offset) * torch.arccos(sigma)

    def inverse_log_snr(self, log_snr: torch.Tensor) -> torch.Tensor:
        return (
            (2.0 / torch.pi)
            * (1.0 + self.offset)
            * torch.arctan(torch.exp(log_snr / 2.0))
        )

    # dxt = f(t) xt + g(t) dwt
    def forward_drift(self, t: torch.Tensor) -> torch.Tensor:
        return self.dalpha_t(t) / self.alpha_t(t)

    # dxt = f(t) xt + g(t) dwt
    def forward_diffusion(self, t: torch.Tensor) -> torch.Tensor:
        sigma_t = self.sigma_t(t)
        dsigma_t = self.dsigma_t(t)
        alpha_t = self.alpha_t(t)
        dalpha_t = self.dalpha_t(t)
        return torch.sqrt(2 * sigma_t * (dalpha_t / alpha_t * sigma_t - dsigma_t))


class Diffusion(pl.LightningModule):
    """Continuous-time diffusion model with cosine noise schedule.

    Supports multiple prediction types (data, noise, score, velocity) and provides
    methods for estimating clean data, noise, score, and velocity from model output.
    Includes a built-in deterministic DDIM sampler for training-time rendering.

    The forward process is: x_t = sigma(t) * x_0 + alpha(t) * x_1,
    where x_0 is noise and x_1 is data.

    Args:
        neural_network: Denoising network (e.g. TemporalUnet).
        prediction_type: What the network predicts ("data", "noise", "score", "velocity").
        lr: Learning rate for Adam optimizer.
        scheduler_T_max: Period for cosine annealing LR scheduler.
        noise_schedule_kwargs: Passed to CosineNoiseSchedule (e.g. offset).
        antithetic_sampling: Use antithetic time sampling during training.
        clip_denoised: Whether to clip denoised predictions (unused currently).
        horizon: Planning horizon (number of timesteps per trajectory).
    """

    def __init__(
        self,
        neural_network: nn.Module,
        prediction_type: str,
        lr: float = 1e-3,
        scheduler_T_max: int = 10_000,
        noise_schedule_kwargs: Dict[str, Any] = {},
        antithetic_sampling: bool = False,
        clip_denoised: bool = True,
        horizon: int = 384,
    ):
        super().__init__()
        self.neural_network = neural_network
        self.prediction_type = prediction_type
        self.noise_schedule_kwargs = noise_schedule_kwargs
        self.antithetic_sampling = antithetic_sampling
        self.lr = lr
        self.scheduler_T_max = scheduler_T_max
        self.sigma_interp = 0.0
        self.clip_denoised = clip_denoised
        self.horizon = horizon

        assert self.prediction_type in {"data", "noise", "score", "velocity"}

        self.noise_schedule = CosineNoiseSchedule(**noise_schedule_kwargs)

    def sample_noise_like(self, x):
        return torch.randn_like(x)

    def compute_mu_t(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = pad_t_like_x(t, x0)
        return self.noise_schedule.sigma_t(t) * x0 + self.noise_schedule.alpha_t(t) * x1

    def sample_times_like(
        self, x: torch.Tensor, antithetic_sampling: Optional[bool] = None
    ) -> torch.Tensor:
        if antithetic_sampling is None:
            antithetic_sampling = self.antithetic_sampling

        batch_size = x.shape[0]
        unit_times = torch.rand(batch_size, dtype=x.dtype, device=x.device)
        unit_times = pad_t_like_x(unit_times, x)
        if not antithetic_sampling:
            return unit_times

        unit_times = (unit_times + torch.arange(start=0, stop=batch_size, dtype=x.dtype, device=x.device)) / batch_size  # type: ignore
        return unit_times

    def interpolation_noise_schedule(
        self, t: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        del s
        return self.sigma_interp * torch.ones_like(t)

    def sample_xt(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor
    ) -> torch.Tensor:
        mu_t = self.compute_mu_t(x0, x1, t)
        c_t0 = self.interpolation_noise_schedule(t, torch.zeros_like(t))
        c_t0 = pad_t_like_x(c_t0, x0)
        return mu_t + c_t0 * epsilon

    def estimate_data(
        self, model_out: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        t = pad_t_like_x(t, xt)
        alpha_t = self.noise_schedule.alpha_t(t)
        sigma_t = self.noise_schedule.sigma_t(t)
        match self.prediction_type:
            case "noise":
                x1_pred = (xt - sigma_t * model_out) / alpha_t
            case "data":
                x1_pred = model_out
            case "score":
                x1_pred = (xt + sigma_t**2 * model_out) / alpha_t
            case "velocity":
                dalpha_t = self.noise_schedule.dalpha_t(t)
                dsigma_t = self.noise_schedule.dsigma_t(t)
                x1_pred = (dsigma_t * xt - sigma_t * model_out) / (
                    alpha_t * dsigma_t - sigma_t * dalpha_t
                )
            case _:
                raise ValueError(
                    f"parametrization type {self.prediction_type} not valid"
                )
        return x1_pred

    def estimate_noise(
        self,
        model_out: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t = pad_t_like_x(t, xt)
        alpha_t = self.noise_schedule.alpha_t(t)
        sigma_t = self.noise_schedule.sigma_t(t)
        dalpha_t = self.noise_schedule.dalpha_t(t)
        dsigma_t = self.noise_schedule.dsigma_t(t)
        match self.prediction_type:
            case "noise":
                noise = model_out
            case "data":
                noise = (xt - alpha_t * model_out) / sigma_t
            case "score":
                noise = -sigma_t * model_out
            case "velocity":
                noise = (-dalpha_t * xt + alpha_t * model_out) / (
                    alpha_t * dsigma_t - sigma_t * dalpha_t
                )
            case _:
                raise ValueError(
                    f"parametrization type {self.prediction_type} not valid"
                )
        return noise

    def estimate_score(
        self,
        model_out: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t = pad_t_like_x(t, xt)
        sigma_t = self.noise_schedule.sigma_t(t)
        return -self.estimate_noise(model_out, xt, t) / sigma_t

    def estimate_velocity(
        self,
        model_out: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t = pad_t_like_x(t, xt)
        dalpha_t = self.noise_schedule.dalpha_t(t)
        dsigma_t = self.noise_schedule.dsigma_t(t)
        noise_pred = self.estimate_noise(model_out, xt, t)
        start_pred = self.estimate_data(model_out, xt, t)
        vel = dalpha_t * start_pred + dsigma_t * noise_pred
        return vel

    ## Pytorch Lightning methods ##
    def configure_optimizers(self):  # type: ignore
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.scheduler_T_max
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: Dict[Any, Any],
        condition_output: bool = True,
        **model_input_kwargs,
    ):
        x = apply_conditioning(x, conditions=conditions, zero_conditioning=False)
        model_out = self.neural_network.forward(x, t, **model_input_kwargs)
        zero_conditioning = self.prediction_type in {"velocity"}
        if condition_output:
            model_out = apply_conditioning(
                model_out, conditions=conditions, zero_conditioning=zero_conditioning
            )
        return model_out

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        del batch_idx
        x = batch.pop("x")
        conditions = batch.pop("conditions")
        x0 = apply_conditioning(self.sample_noise_like(x), conditions=conditions)
        t = self.sample_times_like(x)
        xt = self.compute_mu_t(x0, x, t)
        model_out = self.forward(
            x=xt,
            t=t,
            conditions=conditions,
            condition_output=False,
            **batch)

        match self.prediction_type:
            case "noise":
                noise = apply_conditioning(x0, conditions, zero_conditioning=False)
                loss = F.mse_loss(noise, model_out).mean()
            case "data":
                data = apply_conditioning(
                    x, conditions=conditions, zero_conditioning=False
                )
                loss = F.mse_loss(data, model_out).mean()
            case "score":
                sigma_t = self.noise_schedule.sigma_t(t)
                score = -x0 / sigma_t
                loss = F.mse_loss(score, model_out).mean()
            case "velocity":
                dalpha_t = self.noise_schedule.dalpha_t(t)
                dsigma_t = self.noise_schedule.dsigma_t(t)
                ut = dalpha_t * x + dsigma_t * x0
                ut = apply_conditioning(
                    ut, conditions=conditions, zero_conditioning=True
                )
                loss = F.mse_loss(ut, model_out).mean()
            case _:
                raise ValueError(f"Loss type f{self.prediction_type} not recognized")
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    ## Sampling
    def deterministic_ddim_default_sampler(
        self,
        x0: torch.Tensor,
        num_steps: int,
        conditions: Dict[Any, Any] = {},
        **extra_model_kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        batch_size = x0.shape[0]
        timesteps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=self.dtype, device=self.device)  # type: ignore
        alpha_t = self.noise_schedule.alpha_t(timesteps)
        sigma_t = self.noise_schedule.sigma_t(timesteps)

        x = apply_conditioning(x0, conditions=conditions, zero_conditioning=False)

        intermediates = torch.empty(
            batch_size, num_steps + 1, *x0.shape[1:], dtype=x0.dtype, device=x.device
        )
        intermediates[:, 0, ...] = x

        for i, t in enumerate(timesteps[:-1]):
            t = pad_t_like_x(t, x)
            model_out = self.forward(
                x=x,
                t=t,
                conditions=conditions,
                **extra_model_kwargs,
            )
            x1_pred = self.estimate_data(model_out, x, t)
            x = (sigma_t[i + 1] / sigma_t[i]) * x + (
                alpha_t[i + 1] - alpha_t[i] * sigma_t[i + 1] / sigma_t[i]
            ) * x1_pred
            x = apply_conditioning(x, conditions, zero_conditioning=False)
            intermediates[:, i + 1, ...] = x
        return x, intermediates
