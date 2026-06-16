import cv2
import numpy as np
import torch as th
from typing import Union


def render_env_video(env, states, actions=None):
    observations = states
    imgs = list()
    for i in range(len(observations)):
        state = observations[i]
        env.set_state(state)
        if i == 0:
            env.set_state(state)
        img = env.render()
        # draw action
        if actions is not None:
            action = actions[i]
            coord = (action / 512 * 96).astype(np.int32)
            cv2.drawMarker(img, coord, 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=8, thickness=1)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


def rand_init_state(cfg):
    """
    Randomly initialize the environment state.
    """
    # seed = cfg.init_states_seed
    seed = cfg.training.seed
    n_init_states = cfg.n_init_states
    rng:np.random.Generator = np.random.default_rng(seed)
    init_states = []
    for _ in range(n_init_states):
        # Randomly generate a state within the environment bounds
        # Assuming the environment has bounds of 500x500
        # and the robot is initialized at a random position
        # with a random orientation.
        init_states.append(
            np.array([
                rng.integers(50, 450), rng.integers(50, 450),
                rng.integers(100, 400), rng.integers(100, 400),
                rng.random() * 2 * np.pi - np.pi
            ])
        )
    
    return init_states



def regroup_trajectories(
        trajectories: Union[th.Tensor, np.ndarray],
        tup_size: int,
        n_chunks: int = 1,
    ):
    """
    Regroups trajectories, preserving the tuple member dimension in the output.

    The output shape will be (tup_size, B, ...), where B is the total
    number of tuples across all chunks.

    Args:
        trajectories: The input tensor or array of shape (N, ...).
        tup_size: The number of trajectories in each tuple. This will be the
                  first dimension of the output array.
        n_chunks: The number of chunks to divide the data into.

    Returns:
        The regrouped tensor or array of shape (tup_size, B, ...).
    """
    shape = trajectories.shape
    N = shape[0]
    remaining_shape = shape[1:]

    # --- Input Validation ---
    if N <= 0:
        raise ValueError("The input trajectories must have a positive number of samples.")
    if tup_size <= 0 or n_chunks <= 0:
        raise ValueError("'tup_size' and 'n_chunks' must be positive integers.")
    if N % (n_chunks * tup_size) != 0:
        raise ValueError(
            f"Total batch size N ({N}) must be divisible by "
            f"(n_chunks * tup_size) ({n_chunks * tup_size})."
        )

    # 1. Calculate the number of tuples within each chunk.
    chunk_tup_B = N // (n_chunks * tup_size)

    # 2. Reshape into the logical 3D structure.
    logical_shape = (n_chunks, tup_size, chunk_tup_B, *remaining_shape)
    grouped_traj = trajectories.reshape(logical_shape)

    # 3. Permute to bring the 'tup_size' dimension to the front.
    permute_axes = (1, 0, 2) + tuple(range(3, grouped_traj.ndim))
    if isinstance(grouped_traj, np.ndarray):
        grouped_traj = grouped_traj.transpose(permute_axes)
    else: # assume torch.Tensor
        grouped_traj = grouped_traj.permute(*permute_axes)
    # At this point, the shape is (tup_size, n_chunks, chunk_tup_B, ...)

    # 4. Reshape to merge the chunk dimensions while keeping the tup_size dimension.
    # The '-1' automatically calculates B = n_chunks * chunk_tup_B.
    final_shape = (tup_size, -1, *remaining_shape)
    regrouped_traj = grouped_traj.reshape(final_shape)

    # --- Sanity check ---
    B = n_chunks * chunk_tup_B
    assert regrouped_traj.shape == (tup_size, B, *remaining_shape), \
        "The final shape of the regrouped array is incorrect."

    return regrouped_traj

