"""
ProjectedDiffusionPolicy: IPOPT constraint projection at each DDPM step.

At each denoising step, takes a standard DDPM step then projects the result
onto the feasible set via IPOPT (when t <= projection_start threshold).
DDPM is batched on GPU; IPOPT projection is per-sample (CasADi limitation).
"""

import math
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model import CosineScheduleDiffusion, apply_conditioning
from utils.arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple("Trajectories", "actions observations values")


class ProjectedDiffusionPolicy(nn.Module):
    """DDPM sampling with per-step IPOPT projection.

    Reverse-diffuses from t=1 to t=0, projecting the trajectory onto the
    feasible set at each step where t <= projection_start.
    """

    def __init__(
        self,
        model,
        normalizer,
        projector,
        action_dim,
        state_dim,
        horizon,
        n_sampling_steps=10,
        projection_start=0.5,
        n_projection_workers=1,
        device="cuda:0",
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.projector = projector
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_sampling_steps = n_sampling_steps
        self.projection_start = projection_start
        self.n_projection_workers = n_projection_workers
        self.device = device

        self.diffusion = CosineScheduleDiffusion(model, action_dim)

    def __call__(self, conditions, batch_size=1):
        """Sample trajectories with projection.

        Same interface as UnconditionalPolicy.__call__.
        """
        conditions = apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        transition_dim = self.action_dim + self.state_dim
        conditions_torch = to_torch(conditions, device=self.device)

        # start from pure noise
        x = torch.randn(
            batch_size, self.horizon, transition_dim, device=self.device
        )
        x = apply_conditioning(x, conditions_torch, self.action_dim)

        # time grid: t=1 (noisy) to t=0 (clean)
        timesteps = torch.linspace(1.0, 0.0, self.n_sampling_steps + 1, device=self.device)

        save_chain = (batch_size == 1)
        if save_chain:
            denoising_chain = [x.clone()]
            x0_estimates = []

        for i in range(self.n_sampling_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            is_last_step = (i == self.n_sampling_steps - 1)

            # --- standard DDPM step ---
            abar_now = self.diffusion.alpha_bar(t_now)
            abar_next = self.diffusion.alpha_bar(t_next)

            t_batch = t_now.expand(batch_size)
            x0_pred = self.model(x, t_batch)
            x0_pred = apply_conditioning(x0_pred, conditions_torch, self.action_dim)

            if save_chain:
                x0_estimates.append(x0_pred.detach().clone())

            # DDPM posterior: q(x_{t_next} | x_{t_now}, x_0)
            beta = 1.0 - abar_now / abar_next
            coeff_x0 = math.sqrt(abar_next) * beta / (1.0 - abar_now)
            coeff_xt = math.sqrt(1.0 - beta) * (1.0 - abar_next) / (1.0 - abar_now)
            posterior_mean = coeff_x0 * x0_pred + coeff_xt * x

            if is_last_step:
                x = posterior_mean
            else:
                posterior_var = max((1.0 - abar_next) / (1.0 - abar_now) * beta, 0.0)
                noise = torch.randn_like(x)
                x = posterior_mean + math.sqrt(posterior_var) * noise

            x = apply_conditioning(x, conditions_torch, self.action_dim)

            # --- IPOPT projection ---
            # projection_start is fraction of steps completed before projection begins
            # 0.0 = all steps, 0.5 = second half, 1.0 = never
            frac_done = i / self.n_sampling_steps
            if frac_done >= self.projection_start:
                x = self._project_batch(x, conditions_torch)

            x = x.detach()

            if save_chain:
                denoising_chain.append(x.clone())

        # unnormalize
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

    def _project_batch(self, x, conditions_torch):
        """Project each sample in the batch via IPOPT.

        Args:
            x: (B, H, transition_dim) normalized trajectory tensor
            conditions_torch: dict with conditioned observations

        Returns:
            x_proj: (B, H, transition_dim) projected trajectory tensor
        """
        B = x.shape[0]
        x_np = to_np(x)  # (B, H, transition_dim)

        # convert full trajectories to DOF vectors
        dofs = np.stack([self.projector.full_trajectory_to_dof(x_np[b]) for b in range(B)])
        s0s = x_np[:, 0, self.action_dim:]  # (B, state_dim)

        # solve in parallel across workers
        dofs_proj, statuses = self.projector.project_batch(dofs, s0s, n_workers=self.n_projection_workers)

        # convert back to full trajectories
        for b in range(B):
            x_np[b] = self.projector.dof_to_full_trajectory(dofs_proj[b], s0s[b])

        x_proj = torch.tensor(x_np, dtype=x.dtype, device=x.device)
        # re-apply conditioning to ensure initial state is exact
        x_proj = apply_conditioning(x_proj, conditions_torch, self.action_dim)
        return x_proj

    def _unnormalize_chain(self, chain):
        """Unnormalize a chain of shape (batch, steps, horizon, transition_dim)."""
        obs_chain = chain[..., self.action_dim:]
        obs_unnorm = self.normalizer.unnormalize(to_np(obs_chain), "observations")
        act_chain = chain[..., :self.action_dim]
        act_unnorm = self.normalizer.unnormalize(to_np(act_chain), "actions")
        return np.concatenate([act_unnorm, obs_unnorm], axis=-1)
