import numpy as np
import torch
from typing import Union, Literal, List, Tuple

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments import *

def collision_detect(
    trajs: np.ndarray,
    safe_dist: float,
    norm_order: Union[int, str] = 2,
)->np.ndarray:
    """
    Check collision for batched trajectories and return a mask.

    Args:
        traj (np.ndarray): A batch of trajectories with shape (..., player, horizon, D),
            where D is the dimension of the state space.
        safe_dist (float): The safe distance threshold for collision detection.
        norm_order (Union[int, str]): The order of the norm to use for distance calculation.
            Can be a positive integer or 'inf'

    Returns:
        np.ndarray: A boolean array of shape (..., player, player, horizon) indicating pair-wise trajectory collision masks on all horizon steps.
    """

    trajs = np.asarray(trajs)

    assert trajs.ndim >= 3, "Trajectory must have at least 3 dimensions (player, horizon and state)."
    if trajs.shape[-1] < 1:
        raise ValueError("Trajectory must have at least 1 dimensions for collision detection.")
    assert trajs.shape[-3] >= 2, "There must be at least 2 players in the trajectory array for collision detection."

    n_players = trajs.shape[-3]
    horizon = trajs.shape[-2]
    state_dim = trajs.shape[-1]
    preceding_dims = trajs.shape[:-3]

    traj1 = trajs.reshape(preceding_dims + (n_players, 1, horizon, state_dim))
    traj2 = trajs.reshape(preceding_dims + (1, n_players, horizon, state_dim))

    distances = np.linalg.norm(traj1 - traj2, ord=norm_order, axis=-1)

    for p in range(n_players):
        distances[..., p, p, :] = safe_dist + 1

    collision_masks = distances < safe_dist

    expected_shape = trajs.shape[:-3] + (n_players, n_players, horizon)
    assert collision_masks.shape == expected_shape, (
        f"Expected collision mask shape {expected_shape}, got {collision_masks.shape}"
    )

    return collision_masks


def calc_data_adherence(
    trajs: np.ndarray,
    agent_model_ids_l: List[str],
    agent_model_transforms_l: List[List],
    start_time_l: List[int],
    horizon: int,
    tensor_args: dict,
) -> np.ndarray:
    """
    Calautate data adherence.

    Args:
        trajs (np.ndarray): A batch of trajectories with shape (..., n_agents, horizon, dimension).
        agent_model_ids_l (List[str]): List of model IDs for each agent.
        agent_model_transforms_l (List[List]): List of transforms for each agent's model.
        start_time_l (List[int]): List of start times for each agent.
        horizon (int): The horizon length of the trajectories.
        tensor_args (dict): Arguments for tensor creation, e.g., device and dtype.

    Returns:
        np.ndarray: Data adherence scores for each agent with shape (..., n_agents).
    """
    B, I, N, H, D = trajs.shape
    data_adherence = torch.zeros(*trajs.shape[:-2], **tensor_args)
    traj_tensor = torch.from_numpy(trajs).to(**tensor_args)
    for agent_id in range(N):
        agent_data_adherence = 0.0
        for skeleton_step, agent_model_id in enumerate(agent_model_ids_l[agent_id]):
            agent_model_transform = agent_model_transforms_l[agent_id][skeleton_step]
            agent_start_time = start_time_l[agent_id]
            single_tile_traj_len = horizon
            agent_path_in_model_frame = (
                traj_tensor[..., agent_id, :, :].clone()[
                    ...,
                    agent_start_time + skeleton_step * single_tile_traj_len:
                        agent_start_time + (skeleton_step + 1) * single_tile_traj_len,
                    :2
                ] - agent_model_transform
            )
            model_env_name = agent_model_id.split('-')[0]
            kwargs = {'tensor_args': tensor_args}
            import torch_robotics.environments as _env_module
            env_class = getattr(_env_module, model_env_name)
            env_object = env_class(**kwargs)
            assert hasattr(env_object, 'compute_traj_data_adherence_batch'), \
                "Environment object must implement `compute_traj_data_adherence_batch` method."
            agent_data_adherence += env_object.compute_traj_data_adherence_batch(agent_path_in_model_frame)
        agent_data_adherence /= len(agent_model_ids_l[agent_id])
        data_adherence[..., agent_id] = agent_data_adherence

    data_adherence = data_adherence.to('cpu').numpy() if isinstance(data_adherence, torch.Tensor) else data_adherence

    return data_adherence


def calc_velocity_vec(
    trajs: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Calculate the velocity of the trajectories.

    Args:
        trajs (np.ndarray): A batch of trajectories with shape (..., horizon, D).
        dt (float): The time step duration.

    Returns:
        np.ndarray: The velocity of the trajectories with shape (..., horizon-1, D).
    """
    trajs = np.asarray(trajs)
    assert trajs.ndim >= 2, "Trajectory must have at least 2 dimensions (horizon and state)."

    velocities_vec = np.diff(trajs, axis=-2) / dt

    return velocities_vec


def calc_velocity(
    trajs: np.ndarray,
    dt: float,
    norm_kwargs: dict = {}
) -> np.ndarray:
    """
    Calculate the velocity of the trajectories.

    Args:
        trajs (np.ndarray): A batch of trajectories with shape (..., horizon, D).
        dt (float): The time step duration.

    Returns:
        np.ndarray: The velocity of the trajectories with shape (..., horizon-1).
    """
    trajs = np.asarray(trajs)
    assert trajs.ndim >= 2, "Trajectory must have at least 2 dimensions (horizon and state)."

    default_norm_kwargs = {
        "ord": 2,
        "axis": -1,
        "keepdims": False,
    }
    default_norm_kwargs.update(norm_kwargs)

    v_vec = calc_velocity_vec(trajs, dt)
    velocities = np.linalg.norm(v_vec, **default_norm_kwargs)

    return velocities
