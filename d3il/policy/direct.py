"""
DiReCTPolicy: Diffusion with Real-time Constrained Trajectory optimization.

At each denoising step, predicts x_0 via the diffusion model, projects
the clean prediction through IPOPT (obstacle + dynamics + action constraints),
then maps the correction back to noisy space.
"""

import math
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model import CosineScheduleDiffusion, apply_conditioning
from obstacles import CONSTRAINT_SETS
from utils.arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple("Trajectories", "actions observations values")


class DiReCTPolicy(nn.Module):
    """DDPM sampling with IPOPT projection in clean x_0 space.

    At each denoising step where guidance is active, the model's x_0 prediction
    is projected onto the feasible set via IPOPT, and the correction is mapped
    back to noisy space via alpha_s * (x0_projected - x0_pred).
    """

    def __init__(
        self,
        model,
        normalizer,
        action_dim,
        state_dim,
        horizon,
        n_sampling_steps=10,
        guidance_start=0.0,
        guidance_last_steps=0,
        obstacle_margin=0.02,
        control_penalty_weight=1.0,
        prediction_objective="",
        prediction_objective_scale=100.0,
        max_action_delta=0.0,
        dynamics_model=None,
        dynamics_relaxation=0.0,
        device="cuda:0",
    ):
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_sampling_steps = n_sampling_steps
        self.guidance_start = guidance_start
        self.guidance_last_steps = guidance_last_steps
        self.obstacle_margin = obstacle_margin
        self.control_penalty_weight = control_penalty_weight
        self.prediction_objective = prediction_objective
        self.prediction_objective_scale = prediction_objective_scale
        self.max_action_delta = max_action_delta
        self.device = device

        self.diffusion = CosineScheduleDiffusion(model, action_dim)

        # Build IPOPT projector
        from projector import IpoptProjector

        circular_obs = CONSTRAINT_SETS["novel"]["circular"]
        planar_obs = CONSTRAINT_SETS["novel"]["planar"]

        objective_fn = None
        if prediction_objective == "distance":
            objective_fn = self._make_distance_objective(normalizer)

        self._projector = IpoptProjector(
            circular_obstacles=circular_obs,
            planar_obstacles=planar_obs,
            horizon=horizon,
            action_dim=action_dim,
            state_dim=state_dim,
            normalizer=normalizer,
            dynamics=dynamics_model,
            dynamics_relaxation=dynamics_relaxation,
            obstacle_margin=obstacle_margin,
            objective_fn=objective_fn,
            objective_scale=prediction_objective_scale,
            max_action_delta=max_action_delta,
        )

    def _make_distance_objective(self, normalizer):
        """Create CasADi objective: minimize distance to goal y-position."""
        import casadi as cs

        obs_mins = normalizer.normalizers["observations"].mins
        obs_maxs = normalizer.normalizers["observations"].maxs
        ymin = float(obs_mins[3])
        ymax = float(obs_maxs[3])
        y_goal = 0.35
        dof = self.horizon * (self.action_dim + self.state_dim) - self.state_dim

        def obj_fn(X_dof, s0):
            y_norm = X_dof[dof - self.state_dim + 3]
            y_real = (y_norm + 1.0) / 2.0 * (ymax - ymin) + ymin
            return (y_real - y_goal) ** 2
        return obj_fn

    def __call__(self, conditions, batch_size=1, a_prev=None):
        """Sample trajectories with IPOPT-projected DDPM reverse process."""
        conditions = apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        transition_dim = self.action_dim + self.state_dim
        conditions_torch = to_torch(conditions, device=self.device)

        x = torch.randn(
            batch_size, self.horizon, transition_dim, device=self.device
        )
        x = apply_conditioning(x, conditions_torch, self.action_dim)

        timesteps = torch.linspace(
            1.0, 0.0, self.n_sampling_steps + 1, device=self.device
        )

        save_chain = batch_size == 1
        if save_chain:
            denoising_chain = [x.clone()]
            x0_estimates = []

        for i in range(self.n_sampling_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]
            is_last_step = i == self.n_sampling_steps - 1
            dt = t_now - t_next

            abar_now = self.diffusion.alpha_bar(t_now)
            abar_next = self.diffusion.alpha_bar(t_next)
            g_t = self.diffusion.diffusion_coeff(t_now)

            t_batch = t_now.expand(batch_size)
            x0_pred = self.model(x, t_batch)
            x0_pred = apply_conditioning(x0_pred, conditions_torch, self.action_dim)

            if save_chain:
                x0_estimates.append(x0_pred.detach().clone())

            beta = 1.0 - abar_now / abar_next
            coeff_x0 = math.sqrt(abar_next) * beta / (1.0 - abar_now)
            coeff_xt = math.sqrt(1.0 - beta) * (1.0 - abar_next) / (1.0 - abar_now)
            posterior_mean = coeff_x0 * x0_pred + coeff_xt * x
            posterior_var = max((1.0 - abar_next) / (1.0 - abar_now) * beta, 0.0)

            if is_last_step:
                x_uncontrolled = posterior_mean
            else:
                noise = torch.randn_like(x)
                x_uncontrolled = posterior_mean + math.sqrt(posterior_var) * noise

            x_uncontrolled = apply_conditioning(
                x_uncontrolled, conditions_torch, self.action_dim
            )

            frac_done = i / self.n_sampling_steps
            if self.guidance_last_steps > 0:
                steps_remaining = self.n_sampling_steps - i
                apply_guidance = steps_remaining <= self.guidance_last_steps
            else:
                apply_guidance = frac_done >= self.guidance_start

            if apply_guidance:
                x_next = self._apply_prediction_guidance(
                    x, x0_pred, x_uncontrolled, dt,
                    abar_next, conditions_torch, batch_size,
                    a_prev=a_prev,
                    g_t=g_t,
                )
            else:
                x_next = x_uncontrolled

            x = apply_conditioning(x_next, conditions_torch, self.action_dim)
            x = x.detach()

            if save_chain:
                denoising_chain.append(x.clone())

        normed_actions = x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(
            to_np(normed_observations), "observations"
        )

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

    def _apply_prediction_guidance(
        self, x, x0_pred, x_uncontrolled, dt, abar_next,
        conditions_torch, batch_size, a_prev=None, g_t=None,
    ):
        """Project x0_pred in clean space via IPOPT, map correction back to noisy space."""
        B = batch_size
        x0_np = to_np(x0_pred)

        dofs = np.stack([
            self._projector.full_trajectory_to_dof(x0_np[b]) for b in range(B)
        ])
        s0s = x0_np[:, 0, self.action_dim:]

        # Time-dependent regularization: lambda * alpha_s^2 / dt
        # reg_weight = float(self.control_penalty_weight * float(abar_next) / float(dt))

        ## with g**2 scaling
        reg_weight = float(self.control_penalty_weight * float(abar_next) / float(dt) / float(g_t)**2)

        a_prevs = None
        if a_prev is not None and self.max_action_delta > 0:
            a_prevs = np.tile(to_np(a_prev).reshape(1, -1), (B, 1))

        dofs_proj, statuses = self._projector.project_batch(
            dofs, s0s, n_workers=B, reg_weight=reg_weight, a_prevs=a_prevs
        )

        x0_proj_np = np.stack([
            self._projector.dof_to_full_trajectory(dofs_proj[b], s0s[b])
            for b in range(B)
        ])
        x0_projected = torch.tensor(
            x0_proj_np, dtype=x.dtype, device=x.device
        )
        x0_projected = apply_conditioning(
            x0_projected, conditions_torch, self.action_dim
        )

        alpha_s = math.sqrt(float(abar_next))
        x_next = x_uncontrolled + alpha_s * (x0_projected - x0_pred)
        return x_next

    def _unnormalize_chain(self, chain):
        """Unnormalize a chain of shape (batch, steps, horizon, transition_dim)."""
        obs_chain = chain[..., self.action_dim :]
        obs_unnorm = self.normalizer.unnormalize(to_np(obs_chain), "observations")
        act_chain = chain[..., : self.action_dim]
        act_unnorm = self.normalizer.unnormalize(to_np(act_chain), "actions")
        return np.concatenate([act_unnorm, obs_unnorm], axis=-1)
