"""
SafeDiffuserPolicy: CBF-guided DDPM sampling with RoS, ReS, and TVS algorithms.

At each denoising step, solves a batched QP (via qpth) to find the closest
velocity to the unconstrained DDPM update that satisfies Control Barrier
Function constraints for obstacle avoidance.

Algorithms:
  RoS — basic CBF:       min ||v - v_ref||^2  s.t. J^T v + k h >= 0
  ReS — slack relaxation: min ||[v;r]||^2      s.t. J^T v + k h - w r >= 0
  TVS — time-varying:     min ||v - v_ref||^2  s.t. J^T v + k(h - σ) + σ' >= 0

Fully batched on GPU. Supports optional dynamics equality constraints.
"""

import math
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from qpth.qp import QPFunction, QPSolvers

from model import CosineScheduleDiffusion, apply_conditioning
from obstacles import PILLARS, NOVEL_CIRCULAR, NOVEL_PLANAR, CONSTRAINT_SETS
from utils.arrays import to_np, to_torch, apply_dict

Trajectories = namedtuple("Trajectories", "actions observations values")


# ---------------------------------------------------------------------------
# Differentiable constraint evaluation
# ---------------------------------------------------------------------------

class CBFConstraintModule(nn.Module):
    """Differentiable obstacle constraint computation for QP-based guidance.

    h(x) >= 0 means feasible (outside obstacle).
    Circular: h = dist^2 - (r + margin)^2  (squared to avoid sqrt)
    Planar:   h = slope * x_pos + intercept - y_pos - margin
    """

    def __init__(self, normalizer, action_dim, obstacle_margin=0.02):
        super().__init__()
        self.action_dim = action_dim
        self.obstacle_margin = obstacle_margin

        # Normalizer bounds for observations
        obs_norm = normalizer.normalizers["observations"]
        self.register_buffer(
            "obs_mins", torch.tensor(obs_norm.mins, dtype=torch.float32)
        )
        self.register_buffer(
            "obs_maxs", torch.tensor(obs_norm.maxs, dtype=torch.float32)
        )

        # Circular obstacles
        circular = PILLARS + NOVEL_CIRCULAR
        centers = torch.tensor([obs.center for obs in circular], dtype=torch.float32)
        radii = torch.tensor([obs.radius for obs in circular], dtype=torch.float32)
        self.register_buffer("centers", centers)  # (N_circ, 2)
        self.register_buffer("radii", radii)      # (N_circ,)
        self.n_circular = len(circular)

        # Planar obstacles
        if NOVEL_PLANAR:
            slopes = torch.tensor(
                [obs.slope for obs in NOVEL_PLANAR], dtype=torch.float32
            )
            intercepts = torch.tensor(
                [obs.intercept for obs in NOVEL_PLANAR], dtype=torch.float32
            )
            self.register_buffer("slopes", slopes)          # (N_plan,)
            self.register_buffer("intercepts", intercepts)  # (N_plan,)
            self.n_planar = len(NOVEL_PLANAR)
        else:
            self.n_planar = 0

        self.n_constraints = self.n_circular + self.n_planar

    def _unnorm_obs(self, obs_norm):
        return (obs_norm + 1.0) / 2.0 * (self.obs_maxs - self.obs_mins) + self.obs_mins

    def forward(self, x):
        """Evaluate all constraints on a trajectory batch.

        Args:
            x: (B, H, D) normalized trajectory tensor
        Returns:
            constraints: (B, H, NC) where NC = n_constraints, positive = feasible
        """
        obs_norm = x[:, :, self.action_dim:]
        obs_real = self._unnorm_obs(obs_norm)
        x_pos = obs_real[:, :, 2]  # (B, H)
        y_pos = obs_real[:, :, 3]  # (B, H)

        constraint_list = []

        # Circular: h = dist^2 - (r + margin)^2
        for i in range(self.n_circular):
            cx, cy = self.centers[i]
            r = self.radii[i]
            dist_sq = (x_pos - cx) ** 2 + (y_pos - cy) ** 2
            h = dist_sq - (r + self.obstacle_margin) ** 2
            constraint_list.append(h)  # (B, H)

        # Planar: h = slope * x + intercept - y - margin
        for i in range(self.n_planar):
            h = self.slopes[i] * x_pos + self.intercepts[i] - y_pos - self.obstacle_margin
            constraint_list.append(h)  # (B, H)

        return torch.stack(constraint_list, dim=-1)  # (B, H, NC)


# ---------------------------------------------------------------------------
# Jacobian computation
# ---------------------------------------------------------------------------

def compute_constraints_and_jacobians(constraint_module, x):
    """Compute constraint values and their Jacobians via vjp + vmap.

    Args:
        constraint_module: CBFConstraintModule instance
        x: (B, H, D) normalized trajectory
    Returns:
        constraints: (B, H, NC) detached constraint values
        jacobians:   (B, H, NC, D) Jacobians dh/dx
    """
    B, H, D = x.shape
    x_var = x.detach().clone().requires_grad_(True)

    constraints, vjp_fn = torch.func.vjp(constraint_module, x_var)
    NC = constraints.shape[-1]

    E = torch.eye(NC, device=x.device, dtype=x.dtype)
    V = E[:, None, None, :].expand(NC, B, H, NC).contiguous()

    grads = torch.func.vmap(lambda v: vjp_fn(v)[0])(V)  # (NC, B, H, D)
    jacobians = grads.permute(1, 2, 0, 3).contiguous()   # (B, H, NC, D)

    return constraints.detach(), jacobians.detach()


# ---------------------------------------------------------------------------
# QP formulation
# ---------------------------------------------------------------------------

def formulate_qp_per_timestep(
    disp_ref, constraints, jacobians, algorithm_type, cbf_coefficient, **kwargs
):
    """Formulate B*H independent QPs (no dynamics coupling).

    Args:
        disp_ref:     (B, H, D) reference displacement (x_uncontrolled - x)
        constraints:  (B, H, NC) constraint values at current state
        jacobians:    (B, H, NC, D) constraint Jacobians
        algorithm_type: "RoS", "ReS", or "TVS"
        cbf_coefficient: k in the CBF condition
        **kwargs: algorithm-specific (relaxation_weight, barrier_margin, barrier_margin_diff)
    Returns:
        Q, q, G, h, A_eq, b_eq for QPFunction
    """
    B, H, D = disp_ref.shape
    NC = constraints.shape[-1]
    device = disp_ref.device
    dtype = disp_ref.dtype
    BH = B * H

    disp_ref_flat = disp_ref.reshape(BH, D)
    constraints_flat = constraints.reshape(BH, NC)
    jacobians_flat = jacobians.reshape(BH, NC, D)

    e = torch.empty(0, device=device, dtype=dtype)

    if algorithm_type == "RoS":
        Q = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(BH, D, D).contiguous()
        q = -disp_ref_flat
        G = -jacobians_flat
        h = cbf_coefficient * constraints_flat
        return Q, q, G, h, e, e

    elif algorithm_type == "ReS":
        w = kwargs["relaxation_weight"]
        nv = D + NC
        Q = torch.eye(nv, device=device, dtype=dtype).unsqueeze(0).expand(BH, nv, nv).contiguous()
        q = torch.cat(
            [-disp_ref_flat, torch.zeros(BH, NC, device=device, dtype=dtype)], dim=-1
        )
        G = torch.cat(
            [
                -jacobians_flat,
                w * torch.eye(NC, device=device, dtype=dtype).unsqueeze(0).expand(BH, NC, NC),
            ],
            dim=-1,
        )
        h = cbf_coefficient * constraints_flat
        return Q, q, G, h, e, e

    elif algorithm_type == "TVS":
        barrier_margin = kwargs["barrier_margin"]
        barrier_margin_diff = kwargs["barrier_margin_diff"]
        Q = torch.eye(D, device=device, dtype=dtype).unsqueeze(0).expand(BH, D, D).contiguous()
        q = -disp_ref_flat
        G = -jacobians_flat
        h = (
            cbf_coefficient * (constraints_flat - barrier_margin)
            + barrier_margin_diff
        )
        return Q, q, G, h, e, e

    else:
        raise ValueError(f"Unknown algorithm_type: {algorithm_type}")


def formulate_qp_full_trajectory(
    disp_ref, constraints, jacobians,
    algorithm_type, cbf_coefficient,
    action_dim, state_dim,
    A_dyn, B_dyn, c_dyn, x_current,
    **kwargs,
):
    """Formulate B QPs over the full trajectory with dynamics equality constraints.

    Decision variable: disp_flat (B, H*D) — displacement for entire trajectory.
    Inequality: CBF constraints for t=1..H-1 (skip t=0, conditioned).
    Equality: s_{t+1} = A s_t + B a_t + c for t=0..H-2.

    Args:
        disp_ref:     (B, H, D) reference displacement
        constraints:  (B, H, NC) constraint values
        jacobians:    (B, H, NC, D) constraint Jacobians
        algorithm_type, cbf_coefficient: as above
        action_dim, state_dim: dimension sizes
        A_dyn: (state_dim, state_dim) dynamics matrix
        B_dyn: (state_dim, action_dim) input matrix
        c_dyn: (state_dim,) bias vector
        x_current: (B, H, D) current trajectory state
        **kwargs: algorithm-specific parameters
    Returns:
        Q, q, G, h, A_eq, b_eq for QPFunction
    """
    B, H, D = disp_ref.shape
    NC = constraints.shape[-1]
    device = disp_ref.device
    dtype = disp_ref.dtype
    ad = action_dim
    sd = state_dim
    nv_base = H * D

    # --- Inequality constraints (t=1..H-1) ---
    n_ineq_timesteps = H - 1
    n_ineq = n_ineq_timesteps * NC

    # Build block-diagonal G from per-timestep Jacobians (skip t=0)
    if algorithm_type == "ReS":
        w = kwargs["relaxation_weight"]
        n_slack = n_ineq_timesteps * NC
        nv = nv_base + n_slack
    else:
        nv = nv_base

    G_full = torch.zeros(B, n_ineq, nv, device=device, dtype=dtype)
    h_full = torch.zeros(B, n_ineq, device=device, dtype=dtype)

    for t_idx in range(n_ineq_timesteps):
        t = t_idx + 1  # actual horizon timestep (skip t=0)
        row_start = t_idx * NC
        row_end = row_start + NC
        col_start = t * D
        col_end = col_start + D

        G_full[:, row_start:row_end, col_start:col_end] = -jacobians[:, t, :, :]

        if algorithm_type == "ReS":
            slack_col_start = nv_base + t_idx * NC
            slack_col_end = slack_col_start + NC
            G_full[:, row_start:row_end, slack_col_start:slack_col_end] = (
                w * torch.eye(NC, device=device, dtype=dtype).unsqueeze(0).expand(B, NC, NC)
            )

    # h values (algorithm-specific)
    constraints_skip0 = constraints[:, 1:, :]  # (B, H-1, NC)
    constraints_flat = constraints_skip0.reshape(B, n_ineq)

    if algorithm_type == "TVS":
        barrier_margin = kwargs["barrier_margin"]
        barrier_margin_diff = kwargs["barrier_margin_diff"]
        h_full = (
            cbf_coefficient * (constraints_flat - barrier_margin) + barrier_margin_diff
        )
    else:
        h_full = cbf_coefficient * constraints_flat

    # --- Objective ---
    disp_ref_flat = disp_ref.reshape(B, nv_base)
    Q_full = torch.eye(nv, device=device, dtype=dtype).unsqueeze(0).expand(B, nv, nv).contiguous()
    if algorithm_type == "ReS":
        q_full = torch.cat(
            [-disp_ref_flat, torch.zeros(B, n_slack, device=device, dtype=dtype)],
            dim=-1,
        )
    else:
        q_full = -disp_ref_flat

    # --- Equality constraints (dynamics) ---
    n_eq = (H - 1) * sd
    A_eq = torch.zeros(B, n_eq, nv, device=device, dtype=dtype)
    b_eq = torch.zeros(B, n_eq, device=device, dtype=dtype)

    for t in range(H - 1):
        eq_row_start = t * sd

        for d in range(sd):
            row = eq_row_start + d

            col_next = (t + 1) * D + ad + d
            A_eq[:, row, col_next] = 1.0

            for j in range(sd):
                col_cur_s = t * D + ad + j
                A_eq[:, row, col_cur_s] = -A_dyn[d, j]

            for j in range(ad):
                col_cur_a = t * D + j
                A_eq[:, row, col_cur_a] = -B_dyn[d, j]

        s_t = x_current[:, t, ad:]
        a_t = x_current[:, t, :ad]
        s_next = x_current[:, t + 1, ad:]

        residual = (
            torch.mm(s_t, A_dyn.T) + torch.mm(a_t, B_dyn.T) + c_dyn.unsqueeze(0)
            - s_next
        )
        b_eq[:, eq_row_start:eq_row_start + sd] = residual

    return Q_full, q_full, G_full, h_full, A_eq, b_eq


# ---------------------------------------------------------------------------
# SafeDiffuser Policy (RoS / ReS / TVS only)
# ---------------------------------------------------------------------------

class SafeDiffuserPolicy(nn.Module):
    """DDPM sampling with CBF-guided constraint enforcement via batched QP.

    Reverse-diffuses from t=1 (noise) to t=0 (clean), applying CBF guidance
    at each step to enforce obstacle avoidance constraints.
    """

    def __init__(
        self,
        model,
        normalizer,
        action_dim,
        state_dim,
        horizon,
        n_sampling_steps=10,
        algorithm_type="RoS",
        cbf_coefficient=1.0,
        guidance_start=0.0,
        guidance_last_steps=0,
        obstacle_margin=0.02,
        # ReS
        slack_weight=1000.0,
        relaxation_end_fraction=0.9,
        # TVS
        sigmoid_bias_fraction=0.9,
        sigmoid_input_scale=32.0,
        sigmoid_output_scale=1.0,
        # Dynamics
        dynamics_model=None,
        device="cuda:0",
    ):
        super().__init__()
        assert algorithm_type in ("RoS", "ReS", "TVS"), (
            f"SafeDiffuserPolicy only supports RoS/ReS/TVS, got {algorithm_type!r}"
        )
        self.model = model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_sampling_steps = n_sampling_steps
        self.algorithm_type = algorithm_type
        self.cbf_coefficient = cbf_coefficient
        self.guidance_start = guidance_start
        self.guidance_last_steps = guidance_last_steps
        self.obstacle_margin = obstacle_margin
        self.slack_weight = slack_weight
        self.relaxation_end_fraction = relaxation_end_fraction
        self.sigmoid_bias_fraction = sigmoid_bias_fraction
        self.sigmoid_input_scale = sigmoid_input_scale
        self.sigmoid_output_scale = sigmoid_output_scale
        self.device = device

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
        """Sample trajectories with CBF-guided DDPM reverse process."""
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
                x_next = self._apply_cbf_guidance(
                    x, x_uncontrolled, dt, frac_done, conditions_torch, batch_size
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

    def _apply_cbf_guidance(
        self, x, x_uncontrolled, dt, frac_done, conditions_torch, batch_size
    ):
        """Apply CBF-guided QP correction to the DDPM step."""
        D = self.action_dim + self.state_dim

        disp_ref = x_uncontrolled - x

        constraints, jacobians = compute_constraints_and_jacobians(
            self.cbf_module, x
        )

        qp_kwargs = {}
        if self.algorithm_type == "ReS":
            if frac_done < self.relaxation_end_fraction:
                qp_kwargs["relaxation_weight"] = float(self.slack_weight)
            else:
                qp_kwargs["relaxation_weight"] = 0.0
        elif self.algorithm_type == "TVS":
            sigma = self.sigmoid_output_scale * torch.sigmoid(
                torch.tensor(
                    self.sigmoid_input_scale * (frac_done - self.sigmoid_bias_fraction),
                    device=x.device,
                )
            )
            dsigma = (
                self.sigmoid_input_scale
                * sigma
                * (1.0 - sigma / self.sigmoid_output_scale)
            )
            qp_kwargs["barrier_margin"] = (sigma - 1.0).item()
            qp_kwargs["barrier_margin_diff"] = dsigma.item()

        if self.use_dynamics:
            Q, q, G, h_qp, A_eq, b_eq = formulate_qp_full_trajectory(
                disp_ref,
                constraints,
                jacobians,
                algorithm_type=self.algorithm_type,
                cbf_coefficient=self.cbf_coefficient,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
                A_dyn=self.A_dyn,
                B_dyn=self.B_dyn,
                c_dyn=self.c_dyn,
                x_current=x,
                **qp_kwargs,
            )
        else:
            Q, q, G, h_qp, A_eq, b_eq = formulate_qp_per_timestep(
                disp_ref,
                constraints,
                jacobians,
                algorithm_type=self.algorithm_type,
                cbf_coefficient=self.cbf_coefficient,
                **qp_kwargs,
            )

        try:
            disp_opt = QPFunction(verbose=-1, solver=QPSolvers.PDIPM_BATCHED)(
                Q, q, G, h_qp, A_eq, b_eq
            )
        except Exception as e:
            print(f"[SafeDiffuser] QP solver failed (frac_done={frac_done:.2f}): {e}")
            if self.use_dynamics:
                disp_opt = disp_ref.reshape(batch_size, -1)
            else:
                disp_opt = disp_ref.reshape(batch_size * self.horizon, -1)

        if self.use_dynamics:
            disp_opt = disp_opt[:, : self.horizon * D]
            disp_opt = disp_opt.reshape(batch_size, self.horizon, D)
        else:
            disp_opt = disp_opt[:, :D]
            disp_opt = disp_opt.reshape(batch_size, self.horizon, D)

        for t_cond in conditions_torch:
            disp_opt[:, t_cond, self.action_dim :] = 0.0

        x_next = x + disp_opt
        return x_next

    def _unnormalize_chain(self, chain):
        """Unnormalize a chain of shape (batch, steps, horizon, transition_dim)."""
        obs_chain = chain[..., self.action_dim :]
        obs_unnorm = self.normalizer.unnormalize(to_np(obs_chain), "observations")
        act_chain = chain[..., : self.action_dim]
        act_unnorm = self.normalizer.unnormalize(to_np(act_chain), "actions")
        return np.concatenate([act_unnorm, obs_unnorm], axis=-1)
