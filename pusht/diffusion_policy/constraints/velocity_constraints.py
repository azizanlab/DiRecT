import numpy as np
import torch
import cvxpy as cp
from typing import Optional, Union, List, Dict, Tuple, Optional


def vel_proj_obj(var_dict, param_dict) -> cp.Expression:
    """
    Objective function for velocity projection, which minimizes the squared difference
    between the current position and the planned position.
    Args:
        var_dict (Dict[str, cp.Variable]): Dictionary containing the variable 'pos'.
        param_dict (Dict[str, Union[cp.Parameter, np.ndarray]]): Dictionary containing 
            parameters including 'pos'.
    Returns:
        cp.Expression: The objective function to be minimized.
    """
    assert 'pos' in var_dict, "Variable dictionary must contain 'pos'."
    assert 'pos' in param_dict, "Parameter dictionary must contain 'pos'."
    return cp.sum_squares(var_dict['pos'] - param_dict['pos'])

def vel_proj_constraints(var_dict, param_dict) -> List[cp.Expression]:
    """
    Constraints for velocity projection, ensuring that the position at each time step
    is within the maximum velocity constraint.
    Args:
        var_dict (Dict[str, cp.Variable]): Dictionary containing the variable 'pos'.
        param_dict (Dict[str, Union[cp.Parameter, np.ndarray]]): Dictionary containing 
            parameters including 'pos_init', 'vel_max', and 'delta_t'.
    Returns:
        List[cp.Expression]: List of constraints to be satisfied.
    """
    assert 'pos_init' in param_dict, "Parameter dictionary must contain 'pos_init'."

    cons_ls = []
    ## Anchor to the initial position
    cons_ls.append(cp.norm(var_dict['pos'][0] - param_dict['pos_init'], 2) <= param_dict['dx_max'])
    ## Ensure the position at each time step is within the maximum velocity constraint 
    for t in range(1, var_dict['pos'].shape[0]):
        cons_ls.append(cp.norm(var_dict['pos'][t] - var_dict['pos'][t-1], 2) <= param_dict['dx_max'])

    return cons_ls

def max_vel_constraint_bundle(horizon: int, ndim: int):
    """
    Creates a bundle of constraints for velocity projection in a trajectory optimization problem.
    Args:
        horizon (int): The number of time steps in the trajectory.
        ndim (int): The dimensionality of the position space.
    Returns:
        Dict: A dictionary containing the variable definitions, objective function, and constraints.
    """
    assert horizon > 0, "Horizon must be a positive integer."
    assert ndim > 0, "Dimension must be a positive integer."
    variables_def = {'pos': (horizon, ndim)}
    parameters_def = {
        'pos_init': (ndim,), 
        'pos': (horizon, ndim),
        'dx_max': ()
    }
    
    return {
        "variables": variables_def, 
        "parameters": parameters_def,
        "objective": vel_proj_obj, 
        "constraints": vel_proj_constraints
    }


vel_constraint_registry = {
    "max_vel": max_vel_constraint_bundle,
}
