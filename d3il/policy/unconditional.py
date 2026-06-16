import math
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model import CosineScheduleDiffusion, apply_conditioning
from utils.arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple("Trajectories", "actions observations values")


class UnconditionalPolicy(nn.Module):
    """DDPM sampling policy — raw model outputs, no guidance.

    Reverse-diffuses from t=1 (noise) to t=0 (data) using the cosine schedule
    and DDPM posterior steps.  The last step is always deterministic.

    Supports batch_size > 1 for parallel episode evaluation.
    """

    def __init__(
        self,
        model,
        normalizer,
        action_dim,
        state_dim,
        horizon,
        n_sampling_steps=10,
        device="cuda:0",
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_sampling_steps = n_sampling_steps
        self.device = device

        self.diffusion = CosineScheduleDiffusion(model, action_dim)

    def __call__(self, conditions, batch_size=1):
        """Sample trajectories via DDPM reverse process.

        Args:
            conditions: {0: obs} where obs is (state_dim,) or (batch_size, state_dim)
            batch_size: number of trajectories to sample in parallel

        Returns:
            actions: (batch_size, action_dim) — first action per trajectory
            trajectories: Trajectories namedtuple with (batch_size, horizon, ...) arrays
            denoising_chain: unnormalized chain (only for batch_size=1, else None)
            x0_estimates: unnormalized estimates (only for batch_size=1, else None)
            info: dict with computation_time
        """
        # normalise the observation in conditions
        conditions = apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        transition_dim = self.action_dim + self.state_dim

        # start from pure noise
        x = torch.randn(
            batch_size, self.horizon, transition_dim, device=self.device
        )
        x = apply_conditioning(x, to_torch(conditions, device=self.device), self.action_dim)

        # time grid: from t=1 (noisy) to t=0 (clean)
        timesteps = torch.linspace(1.0, 0.0, self.n_sampling_steps + 1, device=self.device)

        save_chain = (batch_size == 1)
        if save_chain:
            denoising_chain = [x.clone()]
            x0_estimates = []

        for i in range(self.n_sampling_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            is_last_step = (i == self.n_sampling_steps - 1)

            abar_now = self.diffusion.alpha_bar(t_now)
            abar_next = self.diffusion.alpha_bar(t_next)

            # model directly predicts x_0
            t_batch = t_now.expand(batch_size)
            x0_pred = self.model(x, t_batch)

            # clamp x0_pred at conditioned positions to the known values
            x0_pred = apply_conditioning(x0_pred, to_torch(conditions, device=self.device), self.action_dim)

            if save_chain:
                x0_estimates.append(x0_pred.detach().clone())

            # DDPM posterior: q(x_{t_next} | x_{t_now}, x_0)
            beta = 1.0 - abar_now / abar_next  # β_t = 1 - ᾱ_t / ᾱ_{t-1}

            coeff_x0 = math.sqrt(abar_next) * beta / (1.0 - abar_now)
            coeff_xt = math.sqrt(1.0 - beta) * (1.0 - abar_next) / (1.0 - abar_now)
            posterior_mean = coeff_x0 * x0_pred + coeff_xt * x

            if is_last_step:
                x = posterior_mean
            else:
                posterior_var = max((1.0 - abar_next) / (1.0 - abar_now) * beta, 0.0)
                noise = torch.randn_like(x)
                x = posterior_mean + math.sqrt(posterior_var) * noise

            # re-apply conditioning
            x = apply_conditioning(x, to_torch(conditions, device=self.device), self.action_dim)
            x = x.detach()

            if save_chain:
                denoising_chain.append(x.clone())

        # unnormalise
        normed_actions = x[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(to_np(normed_observations), "observations")

        trajectories = Trajectories(actions, observations, np.zeros((batch_size, 1)))

        t_end = time.time()
        info = {"computation_time": t_end - t_start}

        if save_chain:
            chain = torch.stack(denoising_chain, dim=1)
            chain = self._unnormalize_chain(chain)
            x0_est = torch.stack(x0_estimates, dim=1)
            x0_est = self._unnormalize_chain(x0_est)
            return actions[:, 0], trajectories, chain, x0_est, info
        else:
            return actions[:, 0], trajectories, None, None, info

    def _unnormalize_chain(self, chain):
        """Unnormalize a chain of shape (batch, steps, horizon, transition_dim)."""
        obs_chain = chain[..., self.action_dim:]
        obs_unnorm = self.normalizer.unnormalize(to_np(obs_chain), "observations")
        act_chain = chain[..., :self.action_dim]
        act_unnorm = self.normalizer.unnormalize(to_np(act_chain), "actions")
        return np.concatenate([act_unnorm, obs_unnorm], axis=-1)
