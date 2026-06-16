import os
from dataclasses import dataclass
from typing import Any, List, Sequence, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import gymnasium as gym

import numpy as np
from numpy.typing import NDArray
import einops

import imageio
import matplotlib
import matplotlib.pyplot as plt

import casadi as cs

from d4rl.sequence import GoalDataset
from d4rl.maze_model import MazeEnv

matplotlib.use("Agg")


def plot2img(fig: Any, remove_margins: bool = True) -> NDArray:
    """Render a matplotlib figure to a numpy RGBA image array."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype="uint8").reshape((height, width, 4))  # type: ignore


@dataclass
class EllipseObstacle:
    """Ellipse or superellipse obstacle in maze coordinates.

    The constraint is: |x - cx|^p / rx^p + |y - cy|^p / ry^p >= 1
    where p=2 gives an ellipse and p=4 a superellipse (rounded rectangle).
    """

    center_x: float  # x position (column direction in maze)
    center_y: float  # y position (row direction in maze)
    radius_x: float  # radius in x direction
    radius_y: float  # radius in y direction
    power: int = 2  # 2 = ellipse, 4 = superellipse


MAZE2D_BROAD_OBSTACLES = [
    EllipseObstacle(center_x=5.1, center_y=4.3, radius_x=0.8, radius_y=0.8, power=2),
    EllipseObstacle(center_x=4.6, center_y=1.3, radius_x=0.8, radius_y=0.8, power=4),
]

MAZE2D_NARROW_OBSTACLES = [
    EllipseObstacle(center_x=4.8, center_y=1.3, radius_x=1.0, radius_y=1.0, power=4),
    EllipseObstacle(center_x=4.8, center_y=4.3, radius_x=1.0, radius_y=1.0, power=4),
    EllipseObstacle(center_x=2.3, center_y=1.8, radius_x=1.0, radius_y=0.5, power=4),
    EllipseObstacle(center_x=7.8, center_y=2.8, radius_x=1.8, radius_y=1.8, power=4),
    EllipseObstacle(center_x=6.9, center_y=6.3, radius_x=1.0, radius_y=1.0, power=4),
    EllipseObstacle(center_x=9.3, center_y=5.6, radius_x=1.0, radius_y=1.0, power=4),
]

OBSTACLES_DICT = {
    "broad": MAZE2D_BROAD_OBSTACLES,
    "narrow": MAZE2D_NARROW_OBSTACLES,
}


class Maze2dConstrainedEnv(gym.Env, nn.Module):
    """Maze2D environment with ellipse obstacle constraints.

    Wraps a MuJoCo pointmass maze with ellipse/superellipse obstacles and provides:
    - Constraint computation (ellipses, linear dynamics, input saturation)
    - CasADi symbolic constraint interface for optimization-based policies
    - Proxy value function (objective - constraint penalty) for guidance
    - Trajectory rendering and evaluation metrics

    Trajectory format: (batch, horizon, 6) where 6 = [action_x, action_y, obs_y, obs_x, vel_y, vel_x].
    All constraint computations operate in normalized [-1, 1] space.

    Args:
        env_dataset: GoalDataset with the loaded D4RL environment and normalizer.
        obstacle_types: Obstacle layout preset ("broad" or "narrow").
        objective: List of objective terms (e.g. ["distance"]).
        constraints: List of constraint types (e.g. ["ellipses", "dynamics", "input_saturation"]).
        constraint_margin: Global margin added to all obstacle constraints.
        dynamics_path: Path to fitted linear dynamics model (.npz).
        proxy_objective_value_weight: Weight for objective in proxy value.
        proxy_constraint_penalty_weight: Weight for constraint penalty in proxy value.
    """

    MAZE_BOUNDS = {
        "maze2d-large-v1": (0, 9, 0, 12),  # (y_min, y_max, x_min, x_max)
    }
    action_dim = 2
    state_dim = 4
    transition_dim = 6
    default_fixed_start = [0.94875744, 2.93648809, -0.01347715, 0.06358764]

    def __init__(
        self,
        env_dataset: GoalDataset,
        obstacle_types: str = "broad",
        objective: List[str] = ["distance"],
        constraints: List[str] = ["ellipses", "dynamics", "input_saturation"],
        constraint_margin: Optional[float] = None,
        dynamics_path: Optional[str] = None,
        proxy_objective_value_weight: float = 1.0,
        proxy_constraint_penalty_weight: float = 1.0,
    ):

        nn.Module.__init__(self)
        gym.Env.__init__(self)
        self.env_dataset = env_dataset
        self.env: MazeEnv = self.env_dataset.env  # type: ignore
        self.env_name = self.env.name  # type: ignore
        self.normalizer = env_dataset.normalizer
        self.bounds = self.MAZE_BOUNDS[self.env_name]
        self.obstacles = OBSTACLES_DICT[obstacle_types]
        self.obstacle_types = obstacle_types
        self.objectives = objective
        self.constraints = constraints
        self.horizon = self.env_dataset.horizon
        self.proxy_objective_value_weight = proxy_objective_value_weight
        self.proxy_constraint_penalty_weight = proxy_constraint_penalty_weight

        self.validate_configuration()

        self.constraint_margin = constraint_margin if constraint_margin is not None else 0.0

        # Normalize obstacles
        self._obs_norm_maxs: Optional[NDArray] = None
        self._obs_norm_mins: Optional[NDArray] = None
        if self.normalizer is not None:
            self._obs_norm_maxs = self.normalizer.normalizers["observations"].maxs
            self._obs_norm_mins = self.normalizer.normalizers["observations"].mins

        self._precompute_normalized_obstacles()
        self.dynamics_model = self.load_dynamics(dynamics_path)

    def validate_configuration(self):
        assert self.env_name in self.MAZE_BOUNDS
        assert self.obstacle_types in OBSTACLES_DICT
        assert all([x in {"distance"} for x in self.objectives])
        assert all([x in {"ellipses", "dynamics", "input_saturation"} for x in self.constraints])

    def _precompute_normalized_obstacles(self) -> None:
        n_obstacles = len(self.obstacles)

        if n_obstacles == 0:
            # Register empty buffers with correct shape for broadcasting
            self.register_buffer("_obs_rx", torch.zeros(1, 1, 0))
            self.register_buffer("_obs_ry", torch.zeros(1, 1, 0))
            self.register_buffer("_obs_cx", torch.zeros(1, 1, 0))
            self.register_buffer("_obs_cy", torch.zeros(1, 1, 0))
            self.register_buffer("_obs_power", torch.zeros(1, 1, 0))
            return

        if self._obs_norm_mins is None or self._obs_norm_maxs is None:
            obs_norm_min_0 = 0.0
            obs_norm_max_0 = 1.0
            obs_norm_min_1 = 0.0
            obs_norm_max_1 = 1.0
        else:
            obs_norm_min_0 = float(self._obs_norm_mins[0])
            obs_norm_max_0 = float(self._obs_norm_maxs[0])
            obs_norm_min_1 = float(self._obs_norm_mins[1])
            obs_norm_max_1 = float(self._obs_norm_maxs[1])

        rx = torch.zeros(n_obstacles)
        ry = torch.zeros(n_obstacles)
        cx = torch.zeros(n_obstacles)
        cy = torch.zeros(n_obstacles)
        power = torch.zeros(n_obstacles)

        for i, obs in enumerate(self.obstacles):
            rx[i] = 2.0 * obs.radius_x / (obs_norm_max_1 - obs_norm_min_1)
            ry[i] = 2.0 * obs.radius_y / (obs_norm_max_0 - obs_norm_min_0)
            cx[i] = 2.0 * (obs.center_x - obs_norm_min_1) / (obs_norm_max_1 - obs_norm_min_1) - 1.0
            cy[i] = 2.0 * (obs.center_y - obs_norm_min_0) / (obs_norm_max_0 - obs_norm_min_0) - 1.0
            power[i] = obs.power

        self.register_buffer("_obs_rx", rx.view(1, 1, -1))
        self.register_buffer("_obs_ry", ry.view(1, 1, -1))
        self.register_buffer("_obs_cx", cx.view(1, 1, -1))
        self.register_buffer("_obs_cy", cy.view(1, 1, -1))
        self.register_buffer("_obs_power", power.view(1, 1, -1))

    def load_dynamics(self, dynamics_path: Optional[str]) -> Any:
        if (
            dynamics_path is None
            or not os.path.exists(dynamics_path)
            or not "dynamics" in self.constraints
        ):
            print("Not using dynamic constraints")
            return None
        dynamics_data = np.load(dynamics_path, allow_pickle=True)
        dynamics_model = {
            **dynamics_data,
            "normalizer": dynamics_data["normalizer"].item(),
        }
        print("Using fitted dynamics data")
        return dynamics_model

    ## Render
    def render_maze(
        self,
        batch_observations: torch.Tensor,
        savepath: str,
        conditions: Dict[Any, Any] = {},
        plot_obstacles: bool = False,
        ncols: int = 1,
        normalized_obs: bool = True,
        style: str = "SafeDiffuser",
    ):
        del conditions

        assert style in {"SafeDiffuser", "trajectory"}

        _background = self.env.maze_arr == 10
        _remove_margins = False
        _extent = (0, 1, 1, 0)
        bounds = self.bounds
        if len(bounds) == 4:
            _, iscale, _, jscale = bounds
        else:
            iscale, jscale = 9, 12

        if normalized_obs:
            batch_observations = torch.as_tensor(
                self.normalizer.unnormalize(
                    batch_observations.cpu().detach().numpy(), "observations"
                ),
                dtype=batch_observations.dtype,
                device=batch_observations.device,
            )

        imgs = []
        assert batch_observations.dim() == 3
        for observations in batch_observations:

            observations = observations.detach().cpu().numpy()

            observations[..., 0] = observations[..., 0] + 0.7
            observations[..., 1] = observations[..., 1] + 0.7
            observations[..., 0] /= iscale
            observations[..., 1] /= jscale

            # Plot images
            plt.clf()
            fig = plt.gcf()
            fig.set_size_inches(5, 5)
            plt.imshow(
                _background * 0.5,
                extent=_extent,
                cmap=plt.cm.binary,  # type: ignore
                vmin=0,
                vmax=1,
            )

            path_length = len(observations)
            colors = plt.cm.jet(np.linspace(0, 1, path_length))  # type: ignore
            match style:
                case "SafeDiffuser":
                    plt.plot(observations[:, 1], observations[:, 0], c="black", zorder=10)
                    plt.scatter(observations[:, 1], observations[:, 0], c=colors, zorder=20)
                case "trajectory":
                    plt.plot(
                        observations[:, 1],
                        observations[:, 0],
                        c="blue",
                        linewidth=0.9,
                        zorder=12,
                    )

            if plot_obstacles:
                for obstacle in self.obstacles:

                    if isinstance(obstacle, EllipseObstacle):
                        theta = np.linspace(0, 2 * np.pi, 100)

                        # Scale obstacle coordinates to match the rendering extent
                        bounds = self.MAZE_BOUNDS.get(self.env.name, (0, 9, 0, 12))  # type: ignore
                        if len(bounds) == 4:
                            _, iscale, _, jscale = bounds
                        else:
                            iscale, jscale = 9, 12

                        # Convert obstacle center and radii to normalized coordinates
                        cx_norm = (obstacle.center_x + 0.7) / jscale
                        cy_norm = (obstacle.center_y + 0.7) / iscale
                        rx_norm = obstacle.radius_x / jscale
                        ry_norm = obstacle.radius_y / iscale

                        # Generate ellipse or superellipse points
                        if obstacle.power == 2:
                            # Regular ellipse
                            x = rx_norm * np.cos(theta) + cx_norm
                            y = ry_norm * np.sin(theta) + cy_norm
                        else:
                            # Superellipse (e.g., power=4)
                            x = (
                                rx_norm * np.sqrt(np.abs(np.cos(theta))) * np.sign(np.cos(theta))
                                + cx_norm
                            )
                            y = (
                                ry_norm * np.sqrt(np.abs(np.sin(theta))) * np.sign(np.sin(theta))
                                + cy_norm
                            )

                        match style:
                            case "SafeDiffuser":
                                plt.plot(x, y, c="red", zorder=10)
                            case "trajectory":
                                plt.fill(
                                    x,
                                    y,
                                    facecolor="red",
                                    edgecolor="none",
                                    alpha=0.35,
                                    zorder=5,
                                )

            plt.axis("off")
            img = plot2img(fig, remove_margins=_remove_margins)
            imgs.append(img)

        # Save composite image
        images = np.stack(imgs, axis=0)
        nrow = len(images) // ncols
        images = einops.rearrange(
            images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrow, ncol=ncols
        )
        imageio.imsave(savepath, images)

    ## Eval methods
    def get_conditions(self) -> Dict[Any, Any]:
        target = self.env._target  # type: ignore
        return {self.horizon - 1: np.array([*target, 0.0, 0.0])}

    def set_state(self, observation: NDArray):
        self.env.set_state(observation[0:2], observation[2:4])  # type: ignore

    ## Metrics
    def compute_metrics(self, trajectory_data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        metrics = {}
        assert "observation" in trajectory_data
        observations = trajectory_data["observation"].reshape(1, -1, self.state_dim)
        positions = observations[:, :, : self.action_dim]

        # Compute the segment vectors for curvature-based and acceleration-based smoothness
        w = torch.diff(positions, dim=1)
        a = torch.diff(w, dim=1)

        ## Curvature-based smoothness [CS]
        wnorm = torch.sum(w**2, dim=2, keepdim=True).sqrt()
        w_normalized = w / wnorm
        cos_theta = (w_normalized[:, :-1, :] * w_normalized[:, 1:, :]).sum(dim=2)
        cos_theta = cos_theta[cos_theta.isfinite()]
        CS = (1.0 - cos_theta).mean()
        metrics[f"{prefix}_CS"] = CS

        ## Acceleration-based smoothness [AS]
        anorm = torch.sum(a**2, dim=2).sqrt()
        AS = anorm.mean()
        metrics[f"{prefix}_AS"] = AS

        ## Violations
        unnormed_obs = trajectory_data["observation"].cpu().detach().numpy()
        unnormed_act = trajectory_data["actions"].cpu().detach().numpy()
        normed_obs = self.normalizer.normalize(unnormed_obs, "observations")
        normed_act = self.normalizer.normalize(unnormed_act, "actions")
        normed_obs = torch.as_tensor(
            normed_obs, dtype=observations.dtype, device=observations.device
        )
        normed_act = torch.as_tensor(
            normed_act, dtype=observations.dtype, device=observations.device
        )

        trajectory = torch.cat([normed_act, normed_obs], dim=-1).view(1, -1, self.transition_dim)
        total_violations = self.total_violations(trajectory).double()
        safety = (total_violations == 0).double()
        metrics[f"{prefix}_total_violations"] = total_violations
        metrics[f"{prefix}_safety"] = safety

        return metrics

    ## Environement interaction
    def get_normalized_score(self, total_rewards: float) -> float:
        return self.env.get_normalized_score(total_rewards)

    def step(self, action: torch.Tensor):
        return self.env.step(action.cpu().detach().numpy())

    def proxy_value(self, x: torch.Tensor) -> torch.Tensor:
        objective_value = self.objective_value(x)
        constraint_penalty = self.constraint_penalty(x)
        proxy_value = (
            objective_value * self.proxy_objective_value_weight
            - constraint_penalty * self.proxy_constraint_penalty_weight
        )
        return proxy_value

    def objective_value(self, x: torch.Tensor) -> torch.Tensor:
        objective_value = torch.zeros(
            x.shape[0], 1, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad
        )
        for objective in self.objectives:
            if objective == "distance":
                objective_value = objective_value + self.distance_objective_value(x)
        return objective_value

    def distance_objective_value(self, x: torch.Tensor) -> torch.Tensor:
        observations = x[:, :, self.action_dim :]
        step_distances = torch.diff(observations[..., : self.action_dim], dim=1)
        return -step_distances.square().sum(dim=1).sum(dim=1, keepdim=True)

    def get_constraints(
        self, x: torch.Tensor, normalize: bool = False, margins: bool = True
    ) -> Tuple[List[torch.Tensor], ...]:
        ineq_constraints = []
        eq_constraints = []
        for constraint in self.constraints:
            if constraint == "ellipses":
                ineq_constraints.append(
                    self.compute_constraint_ellipses(x, normalize=normalize, margins=margins)
                )
            elif constraint == "dynamics":
                eq_constraints.append(self.compute_constraint_dynamics(x))
            elif constraint == "input_saturation":
                ineq_constraints.append(self.compute_constraint_input_saturation(x))
        return ineq_constraints, eq_constraints

    def constraint_penalty(self, x: torch.Tensor) -> torch.Tensor:
        ineq_constraints, eq_constraints = self.get_constraints(x)
        constraint_violation = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        if ineq_constraints:
            ineq_constraints = torch.cat(ineq_constraints, dim=-1)
            constraint_violation = constraint_violation + ineq_constraints.clamp(
                max=0.0
            ).square().sum(dim=1, keepdim=True)
        if eq_constraints:
            eq_constraints = torch.cat(eq_constraints, dim=-1)
            constraint_violation = constraint_violation + eq_constraints.square().sum(
                dim=1, keepdim=True
            )
        return constraint_violation

    def compute_constraint_ellipses(
        self, x: torch.Tensor, normalize: bool = False, margins: bool = True
    ) -> torch.Tensor:
        del margins
        batch_size = x.shape[0]
        n_obstacles = self._obs_rx.shape[-1]  # type: ignore
        if n_obstacles == 0:
            return torch.zeros(batch_size, 0, device=x.device, dtype=x.dtype)

        observations = x[:, :, self.action_dim :]
        interior_obs = observations[:, :, :]  # (batch, horizon, state_dim)
        y_pos = interior_obs[:, :, 0:1]  # (batch, T, 1)
        x_pos = interior_obs[:, :, 1:2]  # (batch, T, 1)
        x_term = ((x_pos - self._obs_cx) / self._obs_rx).abs().pow(self._obs_power)  # type: ignore
        y_term = ((y_pos - self._obs_cy) / self._obs_ry).abs().pow(self._obs_power)  # type: ignore
        if normalize:
            constraint_values = (x_term + y_term).pow(1 / self._obs_power) - 1.0 # type: ignore
        else:
            constraint_values = x_term + y_term - 1.0
        return constraint_values.reshape(batch_size, -1)

    def compute_constraint_dynamics(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self.dynamics_model is None or (self.horizon <= 3):
            return torch.zeros(batch_size, 0, device=x.device, dtype=x.dtype)

        actions = x[:, :, : self.action_dim]  # (batch, horizon, action_dim)
        observations = x[:, :, self.action_dim :]  # (batch, horizon, state_dim)

        A = torch.tensor(
            self.dynamics_model["A"], device=x.device, dtype=x.dtype
        )  # (state_dim, state_dim)
        B = torch.tensor(
            self.dynamics_model["B"], device=x.device, dtype=x.dtype
        )  # (state_dim, action_dim)
        c = torch.tensor(self.dynamics_model["c"], device=x.device, dtype=x.dtype)  # (state_dim,)

        current_states = observations[:, :-1, :]  # (batch, horizon-1, state_dim)
        current_actions = actions[:, :-1, :]  # (batch, horizon-1, action_dim)
        next_states = observations[:, 1:, :]  # (batch, horizon-1, state_dim)

        # Vectorized dynamics: predicted = s @ A.T + a @ B.T + c
        predicted_next_states = current_states @ A.T + current_actions @ B.T + c.view(1, 1, -1)
        residuals = predicted_next_states - next_states  # (batch, horizon-3, state_dim)
        return residuals.reshape(batch_size, -1)

    def compute_constraint_input_saturation(self, x: torch.Tensor) -> torch.Tensor:
        actions = x[:, :, : self.action_dim].reshape(x.shape[0], -1)
        return 1.0 - actions.abs()

    # [Only for ellipse obstacles]
    def check_violation(self, observation: torch.Tensor) -> bool:
        y, x = observation[0].item(), observation[1].item()

        for obs in self.obstacles:
            if isinstance(obs, EllipseObstacle):
                val = ((x - obs.center_x) / obs.radius_x) ** obs.power + (
                    (y - obs.center_y) / obs.radius_y
                ) ** obs.power
                if val < 1.0:
                    return True
        return False

    # [Only for ellipse obstacles]
    def total_violations(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.view(1, *x.shape)
        constraints = self.compute_constraint_ellipses(x).reshape(x.shape[0], x.shape[1], -1)
        total_violations = torch.any(constraints < 0, dim=-1).sum(-1)
        return total_violations

    def unnormalize_chain(self, x_chain: torch.Tensor) -> torch.Tensor:
        normed_x_chain_observation = x_chain[:, :, :, self.action_dim :]
        x_chain_observation = self.normalizer.unnormalize(
            normed_x_chain_observation.cpu().detach().numpy(), "observations"
        )
        normed_x_chain_action = x_chain[:, :, :, : self.action_dim]
        x_chain_action = self.normalizer.unnormalize(
            normed_x_chain_action.cpu().detach().numpy(), "actions"
        )
        x_chain_np = np.concatenate((x_chain_action, x_chain_observation), axis=-1)
        return torch.as_tensor(x_chain_np, dtype=x_chain.dtype, device=x_chain.device)

    ## CASADI INTERFACE
    def number_dofs(self, conditions: Dict[Any, Any]) -> int:
        idx = {k for k in conditions if isinstance(k, int)}
        return self.horizon * self.transition_dim - len(idx) * self.state_dim

    def matrix_to_dofs(self, x: torch.Tensor, conditions: Dict[Any, Any] = {}) -> torch.Tensor:
        batch_dims = x.shape[:-2]
        dofs = []
        prev_idx = -1
        for i in sorted(conditions.keys()):
            if not isinstance(i, int):
                continue
            dofs.append(x[..., prev_idx + 1 : i, :].reshape(*batch_dims, -1))
            dofs.append(x[..., i, : self.action_dim].reshape(*batch_dims, -1))
            prev_idx = i
        dofs.append(x[..., prev_idx + 1 : -1, :].reshape(*batch_dims, -1))
        dofs = torch.concatenate(dofs, dim=-1)
        return dofs

    def dofs_to_matrix(self, dofs: torch.Tensor, conditions: Dict[Any, Any] = {}) -> torch.Tensor:
        batch_dims = dofs.shape[:-1]
        matrix = []
        prev_idx = 0
        offset = 0
        for i in sorted(conditions.keys()):
            if not isinstance(i, int):
                continue
            matrix.append(
                dofs[offset : offset + (i - prev_idx) * self.transition_dim].view(*batch_dims, -1)
            )
            offset += (i - prev_idx) * self.transition_dim
            prev_idx = i + 1
            matrix.append(dofs[offset : offset + self.action_dim].view(*batch_dims, -1))
            offset += self.action_dim
            matrix.append(conditions[i].view(*batch_dims, self.state_dim))
        matrix.append(
            dofs[offset : offset + (self.horizon - prev_idx) * self.transition_dim].view(
                *batch_dims, -1
            )
        )
        matrix = torch.concatenate(matrix, dim=-1).view(
            *batch_dims, self.horizon, self.transition_dim
        )
        return matrix

    def build_symbolic_matrix(self, dofs: Any, conditions_params: Dict[Any, Any]) -> Any:
        symbolic_matrix = []
        prev_idx = 0
        offset = 0
        for i in sorted(conditions_params.keys()):
            if not isinstance(i, int):
                continue
            symbolic_matrix.append(dofs[offset : offset + (i - prev_idx) * self.transition_dim])
            offset += (i - prev_idx) * self.transition_dim
            prev_idx = i + 1
            symbolic_matrix.append(dofs[offset : offset + self.action_dim])
            offset += self.action_dim
            symbolic_matrix.append(conditions_params[i])
        symbolic_matrix.append(
            dofs[offset : offset + (self.horizon - prev_idx) * self.transition_dim]
        )
        v = cs.vertcat(*symbolic_matrix)
        return cs.reshape(v, self.transition_dim, self.horizon).T

    def generate_casadi_constraints(
        self,
        dofs: Any,
        conditions_params: Dict[Any, Any] = {},
        normalize: bool = False,
        **kwargs,
    ) -> Tuple[List[Any], List[Any]]:
        ineq_constraints = []
        eq_constraints = []
        symbolic_matrix = self.build_symbolic_matrix(dofs, conditions_params)
        for constraint in self.constraints:
            match constraint:
                case "ellipses":
                    ineq_constraints.extend(
                        self.generate_ellipses_constraints(
                            symbolic_matrix, normalize=normalize, **kwargs
                        )
                    )
                case "dynamics":
                    eq_constraints.extend(
                        self.generate_dynamics_constraints(symbolic_matrix, **kwargs)
                    )
                case "input_saturation":
                    ineq_constraints.extend(
                        self.generate_input_saturation_constraints(symbolic_matrix, **kwargs)
                    )
        return ineq_constraints, eq_constraints

    def generate_ellipses_constraints(
        self, symbolic_matrix: cs.MX, normalize: bool = False, **kwargs
    ):
        del kwargs
        xpos_dofs = symbolic_matrix[:, self.action_dim + 1]
        ypos_dofs = symbolic_matrix[:, self.action_dim]

        constraints = []
        for obstacle in self.obstacles:
            if not isinstance(obstacle, EllipseObstacle):
                continue

            rx = (
                2.0 * obstacle.radius_x / (self._obs_norm_maxs[1] - self._obs_norm_mins[1])  # type: ignore
            )
            ry = 2.0 * obstacle.radius_y / (self._obs_norm_maxs[0] - self._obs_norm_mins[0])  # type: ignore
            cx = 2.0 * (obstacle.center_x - self._obs_norm_mins[1]) / (self._obs_norm_maxs[1] - self._obs_norm_mins[1]) - 1.0  # type: ignore
            cy = 2.0 * (obstacle.center_y - self._obs_norm_mins[0]) / (self._obs_norm_maxs[0] - self._obs_norm_mins[0]) - 1.0  # type: ignore

            # Constraint: (x-cx)^p/rx^p + (y-cy)^p/ry^p - 1 >= margin
            constraint = (
                cs.fabs(1.0 / rx * (xpos_dofs - cx)) ** obstacle.power
                + cs.fabs(1.0 / ry * (ypos_dofs - cy)) ** obstacle.power
            )
            if normalize:
                constraint = constraint ** (1.0 / obstacle.power)
            constraints.append(constraint - 1.0 - self.constraint_margin)
        return constraints

    def generate_dynamics_constraints(self, symbolic_matrix: cs.MX, **kwargs) -> List[Any]:
        del kwargs
        if self.dynamics_model is None:
            return []

        A = self.dynamics_model["A"]
        B = self.dynamics_model["B"]
        c = self.dynamics_model["c"]
        actions = symbolic_matrix[:, : self.action_dim]
        observations = symbolic_matrix[:, self.action_dim :]

        current_states = observations[:-1, :]
        current_actions = actions[:-1, :]
        next_states = observations[1:, :]
        predicted_next_states = (
            cs.mtimes(A, current_states.T) + cs.mtimes(B, current_actions.T) + c
        ).T
        return [predicted_next_states - next_states]

    def generate_input_saturation_constraints(self, symbolic_matrix: cs.MX, **kwargs) -> List[Any]:
        del kwargs
        constraints = []
        actions = symbolic_matrix[:, : self.action_dim]
        constraints.append(actions + 1.0)  # a >= -1.0
        constraints.append(-actions + 1.0)  # a <= 1.0
        return constraints

    def generate_casadi_objective(
        self, dofs: cs.MX, conditions_params: Dict[Any, Any] = {}, **kwargs
    ) -> Any:
        symbolic_matrix = self.build_symbolic_matrix(dofs, conditions_params)
        cs_objective = 0.0
        for objective in self.objectives:
            match objective:
                case "distance":
                    cs_objective = cs_objective + self.generate_distance_objective(
                        symbolic_matrix, **kwargs
                    )
        return cs_objective

    def generate_distance_objective(self, symbolic_matrix: cs.MX, **kwargs) -> Any:
        del kwargs
        positions = symbolic_matrix[:, self.action_dim : self.action_dim + 2]
        step_distances = cs.diff(positions, 1, 0)
        return -cs.sumsqr(step_distances)
