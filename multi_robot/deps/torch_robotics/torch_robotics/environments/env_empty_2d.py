from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from mmd.config.mmd_params import MMDParams as params


class EnvEmpty2D(EnvBase):

    def __init__(self,
                 name='EnvEmpty2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):

        obj_list = [
            MultiSphereField(
                np.array([]),  # (n, 2) array of sphere centers.
                np.array([]),  # (n, ) array of sphere radii.
                tensor_args=tensor_args
            ),
        ]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # Environments limits.
            obj_fixed_list=[ObjectField(obj_list, 'empty2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.05,
            n_pre_samples=50000,
            max_time=50
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params

        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params

        else:
            raise NotImplementedError

    def get_skill_pos_seq_l(self, robot=None, start_pos=None, goal_pos=None) -> List[torch.Tensor]:
        return None

    def compute_traj_data_adherence(self, path: torch.Tensor,
                                    fraction_of_length=params.data_adherence_linear_deviation_fraction) -> torch.Tensor:
        start_state_pos = path[0][:2]
        goal_state_pos = path[-1][:2]
        length = torch.norm(goal_state_pos - start_state_pos)
        path = path[:, :2]
        path = torch.stack([path[:, 0], path[:, 1], torch.zeros_like(path[:, 0])], dim=1)
        start_state_pos = torch.stack([start_state_pos[0], start_state_pos[1], torch.zeros_like(start_state_pos[0])]).unsqueeze(0)
        goal_state_pos = torch.stack([goal_state_pos[0], goal_state_pos[1], torch.zeros_like(goal_state_pos[0])]).unsqueeze(0)
        deviation_from_line = torch.norm(torch.cross(goal_state_pos - start_state_pos, path - start_state_pos),
                                         dim=1) / length
        return (deviation_from_line < fraction_of_length).float().mean().item()
    
    def compute_traj_data_adherence_batch(self, 
        paths: torch.Tensor,
        fraction_of_length=params.data_adherence_linear_deviation_fraction
    ) -> torch.Tensor:
        # The score is deviation of the path from a straight line. Cost in {0, 1}.
        # The score is 1 for each point on the path within a distance less than fraction_of_length * length from
        # the straight line. The computation is the average of the scores for all points in the path.
        # paths: (..., H, D)
        # returns: (...)
        start_state_pos = paths[..., 0, :2]  # (..., 2)
        goal_state_pos = paths[..., -1, :2]  # (..., 2)
        length = torch.norm(goal_state_pos - start_state_pos, dim=-1)  # (...)

        # Add a dimension for broadcasting over the horizon
        start_state_pos_h = start_state_pos.unsqueeze(-2)  # (..., 1, 2)
        goal_state_pos_h = goal_state_pos.unsqueeze(-2)  # (..., 1, 2)
        length_h = length.unsqueeze(-1)  # (..., 1)

        path_pos = paths[..., :2]  # (..., H, 2)

        # To use cross product for distance calculation, we embed in 3D
        zeros = torch.zeros_like(path_pos[..., :1])  # (..., H, 1)
        path_pos_3d = torch.cat((path_pos, zeros), dim=-1)  # (..., H, 3)

        zeros_start_goal = torch.zeros_like(start_state_pos_h[..., :1])  # (..., 1, 1)
        start_state_pos_3d = torch.cat((start_state_pos_h, zeros_start_goal), dim=-1)  # (..., 1, 3)
        goal_state_pos_3d = torch.cat((goal_state_pos_h, zeros_start_goal), dim=-1)  # (..., 1, 3)

        # Avoid division by zero for paths where start and goal are the same
        # Add a small epsilon to the length in the denominator
        length_h_safe = length_h + 1e-8
        # Avoid division by zero when start and goal coincide.

        deviation_from_line = torch.norm(
            torch.cross(goal_state_pos_3d - start_state_pos_3d, path_pos_3d - start_state_pos_3d, dim=-1),
            dim=-1
        ) / length_h_safe  # (..., H)

        # The threshold is a fraction of the total path length
        threshold = fraction_of_length * length_h  # (..., 1)

        # Score is the mean fraction of points within the threshold
        # For paths with zero length, all points are the same, so deviation is 0, score is 1.
        score = (deviation_from_line < threshold).float().mean(dim=-1)  # (...)
        return score


if __name__ == '__main__':
    env = EnvEmpty2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
