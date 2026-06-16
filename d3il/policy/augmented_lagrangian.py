"""
AugmentedLagrangianPolicy: Primal-dual and augmented Lagrangian constrained
DDPM sampling.

At each denoising step, computes gradient of the (augmented) Lagrangian
w.r.t. the trajectory and adds it to the DDPM update. Dual variables are
updated via gradient ascent.

Algorithms:
  primal-dual          — grad = nabla_x [sum lambda_j * g_j(x)]
  augmented-lagrangian — grad = nabla_x [sum (lambda_j + rho*(g_j - s_j)) * g_j(x)]
"""

import math
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from model import CosineScheduleDiffusion, apply_conditioning
from policy.safediffuser import CBFConstraintModule
from utils.arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple("Trajectories", "actions observations values")


class AugmentedLagrangianPolicy(nn.Module):
    """DDPM sampling with primal-dual or augmented Lagrangian constraint enforcement.

    Reverse-diffuses from t=1 (noise) to t=0 (clean), applying Lagrangian
    gradient corrections at each step.
    """

    def __init__(
        self,
        model,
        normalizer,
        action_dim,
        state_dim,
        horizon,
        n_sampling_steps=10,
        algorithm_type="augmented-lagrangian",
        guidance_start=0.0,
        guidance_last_steps=0,
        obstacle_margin=0.02,
        learning_rate=0.456,
        initial_penalty=2.5e-4,
        final_penalty=0.00154,
        use_cbfs=True,
        cbf_alpha=0.9,
        dynamics_model=None,
        device="cuda:0",
    ):
        super().__init__()
        assert algorithm_type in ("primal-dual", "augmented-lagrangian"), (
            f"AugmentedLagrangianPolicy supports primal-dual/augmented-lagrangian, got {algorithm_type!r}"
        )
        self.model = model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_sampling_steps = n_sampling_steps
        self.algorithm_type = algorithm_type
        self.guidance_start = guidance_start
        self.guidance_last_steps = guidance_last_steps
        self.obstacle_margin = obstacle_margin
        self.learning_rate = learning_rate
        self.initial_penalty = initial_penalty
        self.final_penalty = final_penalty
        self.use_cbfs = use_cbfs
        self.cbf_alpha = cbf_alpha
        self.device = device

        # Precompute penalty growth rate for augmented Lagrangian
        total_steps = n_sampling_steps
        if final_penalty > 0 and initial_penalty > 0 and total_steps > 0:
            self._penalty_multiplier = (final_penalty / initial_penalty) ** (
                1.0 / total_steps
            )
        else:
            self._penalty_multiplier = 1.0
        self._step_learning_rate = learning_rate / total_steps if total_steps > 0 else learning_rate

        self.diffusion = CosineScheduleDiffusion(model, action_dim)
        self.cbf_module = CBFConstraintModule(
            normalizer, action_dim, obstacle_margin
        ).to(device)

        # Dynamics model buffers
        self.use_dynamics = dynamics_model is not None
        if self.use_dynamics:
            self.register_buffer(
                "A_dyn",
                torch.tensor(dynamics_model["A"], dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "B_dyn",
                torch.tensor(dynamics_model["B"], dtype=torch.float32, device=device),
            )
            self.register_buffer(
                "c_dyn",
                torch.tensor(dynamics_model["c"], dtype=torch.float32, device=device).squeeze(),
            )

    def __call__(self, conditions, batch_size=1, **kwargs):
        """Sample trajectories with Lagrangian-guided DDPM reverse process."""
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

        # Initialize dual variables
        with torch.no_grad():
            ineq_init, dyn_init = self._compute_constraint_values(x)
        self._dual_vars = torch.zeros_like(ineq_init)
        self._slack_vars = torch.zeros_like(ineq_init)
        self._penalty = self.initial_penalty
        if dyn_init is not None:
            self._dual_vars_eq = torch.zeros_like(dyn_init)
        else:
            self._dual_vars_eq = None

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

            abar_now = self.diffusion.alpha_bar(t_now)
            abar_next = self.diffusion.alpha_bar(t_next)

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
                if self.algorithm_type == "primal-dual":
                    x_next = self._apply_primal_dual_guidance(
                        x, x_uncontrolled, conditions_torch
                    )
                else:
                    x_next = self._apply_augmented_lagrangian_guidance(
                        x, x_uncontrolled, conditions_torch
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

    # ------------------------------------------------------------------
    # Constraint computation
    # ------------------------------------------------------------------

    def _compute_constraint_values(self, x):
        """Compute normalized constraint values for primal-dual/AL methods.

        Uses dist/(r+margin) - 1 for circular obstacles (0 at boundary, O(1)
        scale, uniform gradient). Planar constraints are normalized by margin.
        """
        cbf_mod = self.cbf_module
        obs_norm = x[:, :, self.action_dim:]
        obs_real = cbf_mod._unnorm_obs(obs_norm)
        x_pos = obs_real[:, :, 2]
        y_pos = obs_real[:, :, 3]

        constraint_list = []

        # Circular: h = dist / (r + margin) - 1
        eps = 1e-6
        for i in range(cbf_mod.n_circular):
            cx, cy = cbf_mod.centers[i]
            r = cbf_mod.radii[i]
            dist = torch.sqrt((x_pos - cx) ** 2 + (y_pos - cy) ** 2 + eps)
            h = dist / (r + cbf_mod.obstacle_margin) - 1.0
            constraint_list.append(h)

        # Planar: h = (slope * x + intercept - y) / margin - 1
        for i in range(cbf_mod.n_planar):
            margin = cbf_mod.obstacle_margin if cbf_mod.obstacle_margin > 0 else 1.0
            h = (cbf_mod.slopes[i] * x_pos + cbf_mod.intercepts[i] - y_pos) / margin - 1.0
            constraint_list.append(h)

        raw = torch.stack(constraint_list, dim=-1)  # (B, H, NC)

        if self.use_cbfs:
            # CBF: g_{t+1} - (1 - alpha) * g_t >= 0
            cbf = raw[:, 1:, :] - (1 - self.cbf_alpha) * raw[:, :-1, :]
            ineq = cbf.reshape(x.shape[0], -1)
        else:
            ineq = raw.reshape(x.shape[0], -1)

        dyn = None
        if self.use_dynamics:
            s_t = x[:, :-1, self.action_dim :]
            a_t = x[:, :-1, : self.action_dim]
            s_next = x[:, 1:, self.action_dim :]
            residual = (
                s_t @ self.A_dyn.T + a_t @ self.B_dyn.T + self.c_dyn
            ) - s_next
            dyn = residual.reshape(x.shape[0], -1)

        return ineq, dyn

    # ------------------------------------------------------------------
    # Primal-dual guidance
    # ------------------------------------------------------------------

    def _apply_primal_dual_guidance(self, x, x_uncontrolled, conditions_torch):
        """Primal-dual constrained guidance."""
        with torch.enable_grad():
            x_in = x.clone().detach().requires_grad_(True)
            ineq, dyn = self._compute_constraint_values(x_in)

            lagrangian = (ineq * self._dual_vars).sum()
            if dyn is not None and self._dual_vars_eq is not None:
                lagrangian = lagrangian + (dyn * self._dual_vars_eq).sum()

            grad = torch.autograd.grad(lagrangian, x_in)[0]

        x_next = x_uncontrolled + grad

        # Update duals (inequality: clamp >= 0)
        self._dual_vars = (
            self._dual_vars - self._step_learning_rate * ineq.detach()
        ).clamp(min=0.0)

        # Update duals (equality: no clamping)
        if dyn is not None and self._dual_vars_eq is not None:
            self._dual_vars_eq = (
                self._dual_vars_eq + self._step_learning_rate * dyn.detach()
            )

        x_next = apply_conditioning(x_next, conditions_torch, self.action_dim)
        return x_next

    # ------------------------------------------------------------------
    # Augmented Lagrangian guidance
    # ------------------------------------------------------------------

    def _apply_augmented_lagrangian_guidance(self, x, x_uncontrolled, conditions_torch):
        """Augmented Lagrangian constrained guidance."""
        with torch.enable_grad():
            x_in = x.clone().detach().requires_grad_(True)
            ineq, dyn = self._compute_constraint_values(x_in)
            ineq_det = ineq.detach()

            # Augmented Lagrangian vector: lambda + rho * (g - s)
            vec = self._dual_vars + self._penalty * (ineq_det - self._slack_vars)

            lagrangian = (ineq * vec).sum()
            if dyn is not None and self._dual_vars_eq is not None:
                dyn_det = dyn.detach()
                lagrangian = lagrangian + (
                    dyn * (self._dual_vars_eq + self._penalty * dyn_det)
                ).sum()

            grad = torch.autograd.grad(lagrangian, x_in)[0]

        x_next = x_uncontrolled - grad  # NOTE: minus sign

        # Update slack (inequality only)
        self._slack_vars = (
            self._dual_vars / self._penalty + ineq_det
        ).clamp(min=0.0)

        # Update duals (inequality)
        self._dual_vars = self._dual_vars + self._penalty * (
            ineq_det - self._slack_vars
        )

        # Update duals (equality: no slack, no clamping)
        if dyn is not None and self._dual_vars_eq is not None:
            self._dual_vars_eq = self._dual_vars_eq + self._penalty * dyn_det

        # Increase penalty
        self._penalty *= self._penalty_multiplier

        x_next = apply_conditioning(x_next, conditions_torch, self.action_dim)
        return x_next

    def _unnormalize_chain(self, chain):
        """Unnormalize a chain of shape (batch, steps, horizon, transition_dim)."""
        obs_chain = chain[..., self.action_dim :]
        obs_unnorm = self.normalizer.unnormalize(to_np(obs_chain), "observations")
        act_chain = chain[..., : self.action_dim]
        act_unnorm = self.normalizer.unnormalize(to_np(act_chain), "actions")
        return np.concatenate([act_unnorm, obs_unnorm], axis=-1)
