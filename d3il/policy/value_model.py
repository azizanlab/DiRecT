"""
ProxyValueModel: batched differentiable value function for gradient guidance.

value = objective_weight * objective
      - obstacle_weight * obstacle_penalty
      - dynamics_weight * dynamics_penalty

All operations are in PyTorch for autograd compatibility.
"""

import numpy as np
import torch
from torch import nn

from obstacles import PILLARS, NOVEL_CIRCULAR, NOVEL_PLANAR


class ProxyValueModel(nn.Module):
    """Differentiable value function for trajectory guidance.

    Evaluates on normalized trajectories of shape (B, H, action_dim + state_dim).
    Internally unnormalizes positions using the dataset normalizer bounds.
    """

    def __init__(
        self,
        horizon,
        action_dim,
        state_dim,
        normalizer,
        objective="",
        constraints=False,
        dynamics=False,
        objective_weight=1.0,
        obstacle_weight=100.0,
        dynamics_weight=0.0,
        obstacle_margin=0.02,
        dynamics_model=None,
        y_goal=0.35,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.objective = objective
        self.constraints = constraints
        self.dynamics = dynamics
        self.objective_weight = objective_weight
        self.obstacle_weight = obstacle_weight
        self.dynamics_weight = dynamics_weight
        self.obstacle_margin = obstacle_margin
        self.y_goal = y_goal

        # normalizer bounds for observations (x_pos=dim2, y_pos=dim3)
        obs_norm = normalizer.normalizers["observations"]
        self.register_buffer("obs_mins", torch.tensor(obs_norm.mins, dtype=torch.float32))
        self.register_buffer("obs_maxs", torch.tensor(obs_norm.maxs, dtype=torch.float32))

        # dynamics model (A, B, c) as buffers
        if dynamics and dynamics_model is not None:
            self.register_buffer("A", torch.tensor(dynamics_model["A"], dtype=torch.float32))
            self.register_buffer("B", torch.tensor(dynamics_model["B"], dtype=torch.float32))
            self.register_buffer("c", torch.tensor(dynamics_model["c"], dtype=torch.float32).squeeze())

        # precompute obstacle tensors
        if constraints:
            circular = PILLARS + NOVEL_CIRCULAR
            centers = torch.tensor([obs.center for obs in circular], dtype=torch.float32)
            radii = torch.tensor([obs.radius for obs in circular], dtype=torch.float32)
            self.register_buffer("obs_centers", centers)
            self.register_buffer("obs_radii", radii)

            if NOVEL_PLANAR:
                slopes = torch.tensor([obs.slope for obs in NOVEL_PLANAR], dtype=torch.float32)
                intercepts = torch.tensor([obs.intercept for obs in NOVEL_PLANAR], dtype=torch.float32)
                self.register_buffer("planar_slopes", slopes)
                self.register_buffer("planar_intercepts", intercepts)
            else:
                self.planar_slopes = None

    def _unnorm_obs(self, obs_norm):
        """Unnormalize observations from [-1,1] to real-world coords.

        Args:
            obs_norm: (..., state_dim) normalized observations
        Returns:
            (..., state_dim) real-world observations
        """
        return (obs_norm + 1.0) / 2.0 * (self.obs_maxs - self.obs_mins) + self.obs_mins

    def forward(self, x):
        """Compute value for a batch of trajectories.

        Args:
            x: (B, H, action_dim + state_dim) normalized trajectories

        Returns:
            value: (B,) scalar value per trajectory (higher is better)
        """
        B = x.shape[0]
        actions = x[:, :, :self.action_dim]          # (B, H, action_dim)
        obs_norm = x[:, :, self.action_dim:]          # (B, H, state_dim)
        obs_real = self._unnorm_obs(obs_norm)         # (B, H, state_dim)

        value = torch.zeros(B, device=x.device)

        # --- objective ---
        if self.objective == "distance":
            # terminal y-position cost: -(y_terminal - y_goal)^2
            y_terminal = obs_real[:, -1, 3]  # (B,) — y_pos is dim 3
            value = value + self.objective_weight * (-(y_terminal - self.y_goal) ** 2)

        # --- obstacle penalty ---
        if self.constraints and self.obstacle_weight > 0:
            value = value - self.obstacle_weight * self._obstacle_penalty(obs_real)

        # --- dynamics penalty ---
        if self.dynamics and self.dynamics_weight > 0:
            value = value - self.dynamics_weight * self._dynamics_penalty(actions, obs_norm)

        return value

    def _obstacle_penalty(self, obs_real):
        """Compute squared obstacle violation penalty.

        Args:
            obs_real: (B, H, state_dim) real-world observations
        Returns:
            penalty: (B,) sum of squared violations
        """
        x_pos = obs_real[:, :, 2]  # (B, H)
        y_pos = obs_real[:, :, 3]  # (B, H)

        penalty = torch.zeros(obs_real.shape[0], device=obs_real.device)

        # circular obstacles: violation = clamp(r + margin - dist, min=0)
        # obs_centers: (N_obs, 2), obs_radii: (N_obs,)
        for i in range(self.obs_centers.shape[0]):
            cx, cy = self.obs_centers[i]
            r = self.obs_radii[i]
            dist = torch.sqrt((x_pos - cx) ** 2 + (y_pos - cy) ** 2)  # (B, H)
            violation = torch.clamp(r + self.obstacle_margin - dist, min=0)  # (B, H)
            penalty = penalty + (violation ** 2).sum(dim=1)  # (B,)

        # planar obstacles: violation = clamp(-(slope * x + intercept - y) + margin, min=0)
        # feasible: slope * x + intercept - y >= 0
        if self.planar_slopes is not None:
            for i in range(self.planar_slopes.shape[0]):
                s = self.planar_slopes[i]
                b = self.planar_intercepts[i]
                feasibility = s * x_pos + b - y_pos  # (B, H) — positive = feasible
                violation = torch.clamp(self.obstacle_margin - feasibility, min=0)  # (B, H)
                penalty = penalty + (violation ** 2).sum(dim=1)

        return penalty

    def _dynamics_penalty(self, actions, obs_norm):
        """Compute squared dynamics residual penalty in normalized space.

        s_{t+1} = A @ s_t + B @ a_t + c  (fitted linear model)

        Args:
            actions: (B, H, action_dim) normalized actions
            obs_norm: (B, H, state_dim) normalized observations
        Returns:
            penalty: (B,) sum of squared residuals
        """
        # s_t for t=0..H-2, s_{t+1} for t=1..H-1
        s_t = obs_norm[:, :-1, :]      # (B, H-1, state_dim)
        s_next = obs_norm[:, 1:, :]    # (B, H-1, state_dim)
        a_t = actions[:, :-1, :]       # (B, H-1, action_dim)

        # predicted: A @ s_t + B @ a_t + c
        # A: (state_dim, state_dim), B: (state_dim, action_dim), c: (state_dim,)
        pred = (torch.einsum("ij,bhj->bhi", self.A, s_t)
                + torch.einsum("ij,bhj->bhi", self.B, a_t)
                + self.c)  # (B, H-1, state_dim)

        residual = pred - s_next  # (B, H-1, state_dim)
        penalty = (residual ** 2).sum(dim=(1, 2))  # (B,)
        return penalty
