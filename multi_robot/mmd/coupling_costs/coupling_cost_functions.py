import einops
import numpy as np
import torch
from typing import List, Optional, Callable

from mp_baselines.planners.costs.cost_functions import Cost
from mmd.coupling_costs.cost_function_implementation import cost_registry

class DummyCost(Cost):
    """
    Zero-valued placeholder cost that preserves shape and gradient flow.
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def eval(self, trajs: torch.Tensor, **kwargs):
        _ = kwargs
        if len(trajs) == 0:
            raise ValueError("No input tensors provided to DummyCost.")
        if trajs.dim() < 3:
            raise ValueError(f"Input tensor trajs must be at least 3D (B, H, d). Got {trajs.dim()}D.")
        return torch.zeros_like(trajs) + trajs.sum() * 0
    
    def get_linear_system(self, trajs, **kwargs):
        pass


class RobotCollisionCost(Cost):
    """
    Wrapper that splits batched trajectories into grouped robot inputs before calling a registered cost.
    """
    def __init__(self, robot_radius: float, cost_func_key:str, **kwargs):
        self.robot_radius = robot_radius
        assert cost_func_key in cost_registry, f"Cost function key '{cost_func_key}' not found in registry."
        self.cost_func:Callable = cost_registry[cost_func_key]
        self.cost_func_key = cost_func_key
        self.cost_fn_kwargs = {}
        self.resolve_cost_fn_kwargs(**kwargs)

    def eval(self, trajs: torch.Tensor, **kwargs):
        """
        Evaluate the collision cost for the given trajectories.

        Args:
            trajs: Input tensor of shape (B, H, d) where B is the batch size, 
                H is the horizon, and d is the dimension of the trajectory; 
                by default assuming the 1st dim is the batch size and the 
                2nd last dim is horizon and the last dim is state dimension. 
            kwargs: Additional keyword arguments, such as 'margin_coeff' and 'group_size'. 

        Returns:
            A tensor of collision costs for the robot's trajectory, shape (B // group_size,).
        """
        if len(trajs) == 0:
            raise ValueError("No input tensors provided to RobotCollisionCost.")
        if trajs.dim() < 3:
            raise ValueError(f"Input tensor trajs must be at least 3D (B, H, d). Got {trajs.dim()}D.")

        grp_sz = kwargs.pop('group_size', 2)
        robot_trjs = torch.tensor_split(trajs[..., :2], grp_sz, dim=0)

        costs = self.cost_func(*robot_trjs, **self.cost_fn_kwargs)

        return costs
    
    def get_linear_system(self, trajs, **kwargs):
        pass

    def resolve_cost_fn_kwargs(self, **kwargs):
        if self.cost_func_key == 'sum_log_l2':
            eta = kwargs.get('margin_coeff', 1.95)
            self.cost_fn_kwargs = {
                'alpha': eta * self.robot_radius, 
            }
        elif self.cost_func_key == 'hinge_sqr_l2':
            eta = kwargs.get('radius_coeff', 4.0)
            self.cost_fn_kwargs = {
                'active_range': eta * self.robot_radius,
            }
        elif self.cost_func_key == 'hinge_cub_l2':
            eta = kwargs.get('radius_coeff', 4.0)
            self.cost_fn_kwargs = {
                'active_range': eta * self.robot_radius,
            }
        else:
            raise NotImplementedError(
                f"Cost function '{self.cost_func_key}' not implemented for RobotCollisionCost."
            )

        return self.cost_fn_kwargs
