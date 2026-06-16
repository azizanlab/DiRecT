import numpy as np
from typing import Union, Literal
from itertools import combinations

def collision_detect(
    trajs: np.ndarray,
    safe_dist: float = 10.0,
    norm_order: Union[int, str] = 2,
):
    """
    Check collision for batched trajectories and return a mask.

    Args:
        traj (np.ndarray): A batch of trajectories with shape (..., player, horizon, D),
            where D is the dimension of the state space.
        safe_dist (float): The safe distance threshold for collision detection.
        norm_order (Union[int, str]): The order of the norm to use for distance calculation.
            Can be a positive integer or 'inf'

    Returns:
        np.ndarray: A boolean array of shape (..., player, horizon) indicating whether
            each player is in collision with any other player at each timestep.
    """

    # Ensure traj is a numpy array
    trajs = np.asarray(trajs)

    assert trajs.ndim >= 3, "Trajectory must have at least 3 dimensions (player, horizon and state)."
    if trajs.shape[-1] < 1:
        raise ValueError("Trajectory must have at least 1 dimension for collision detection.")
    P = trajs.shape[-3]
    assert P >= 2, f"Need at least 2 players for collision detection, got {P}."

    # Initialize per-player collision mask as all-False
    collision_masks = np.zeros(trajs.shape[:-1], dtype=bool)  # (..., P, H)

    for i, j in combinations(range(P), 2):
        dist_ij = np.linalg.norm(
            trajs[..., i, :, :2] - trajs[..., j, :, :2],
            ord=norm_order, axis=-1
        )  # (..., H)
        collide_ij = dist_ij < safe_dist
        collision_masks[..., i, :] |= collide_ij
        collision_masks[..., j, :] |= collide_ij

    # Sanity check
    assert collision_masks.shape == trajs.shape[:-1], \
        f"Expected collision mask shape {trajs.shape[:-1]}, got {collision_masks.shape}"

    return collision_masks


def calculate_per_segment_safety(safe_masks: np.ndarray, calc_horizon: int) -> np.ndarray:
    """
    Calculates the safe rate for each segment and each batch from a 4D safety mask.

    The function divides the time horizon (H) into segments of length `calc_horizon`
    and calculates whether ALL timesteps in each segment are safe (returns True only
    if all timesteps in the segment are safe, False otherwise). It does NOT average
    over the batch (N) or other dimensions.

    Args:
        safe_masks (np.ndarray): A boolean array of shape (..., H) where
                                 True indicates a safe state.
        calc_horizon (int): The desired length of each segment along the H dimension.

    Returns:
        np.ndarray: An array of shape (..., num_segments) where each value is
                    True if ALL timesteps in that segment are safe, False otherwise.
    """
    assert calc_horizon > 0, "`calc_horizon` must be a positive integer."

    # Get the total number of steps in the horizon from the last dimension.
    h_dim = safe_masks.shape[-1]

    # 1. Define the indices where the array will be split along the horizon axis.
    split_indices = range(calc_horizon, h_dim, calc_horizon)

    # 2. Split the safe_masks array into a list of segments along the H dimension (axis=-1).
    segments = np.split(safe_masks, split_indices, axis=-1)

    # 3. Calculate the safe rate for each segment, averaging ONLY over the horizon axis (-1).
    #    This preserves the I, P, and N dimensions.
    #    The result is a list of arrays, where each array has the shape (..., ).
    segment_rates_list = [np.all(s.astype(bool), axis=-1) for s in segments]

    # 4. Stack the list of (...,) arrays into a single array.
    #    The new dimension, axis=-1, represents the segments.
    #    The final shape is (..., num_segments).
    per_segment_batch_rates = np.stack(segment_rates_list, axis=-1)

    return per_segment_batch_rates


def _pairwise_distance_matrix(
        t1: np.ndarray, 
        t2: np.ndarray, 
        norm_ord=Union[float, Literal['fro', 'nuc'], None]
    ) -> np.ndarray:
    """Computes the pairwise distance matrix, normalized by trajectory length.
    
    Args: 
        t1 (np.ndarray): First batch of trajectories. Shape (..., H, d).
        t2 (np.ndarray): Second batch of trajectories. Must have the same shape as t1.
        norm_ord (Union[float, Literal['fro', 'nuc'], None]): The order of the norm to use for distance calculation.
            Can be a positive integer or 'fro' for Frobenius norm, 'nuc' for nuclear norm, or None for L2 norm.

    Returns:
        np.ndarray: A distance matrix of shape (..., H, H) where each entry (i, j) is the distance
            between the i-th point in t1 and the j-th point in t2.
    Raises:
        ValueError: If input shapes are not identical or have fewer than 2 dimensions.
    """
    t1_expanded = t1[..., :, np.newaxis, :]
    t2_expanded = t2[..., np.newaxis, :, :]
    diff = t1_expanded - t2_expanded
    # dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
    dist_matrix = np.linalg.norm(diff, ord=norm_ord, axis=-1)
    return dist_matrix


def frechet_distance(t1: np.ndarray, t2: np.ndarray, norm_ord=2) -> np.ndarray:
    """
    Computes the discrete Fréchet distance between two batches of trajectories.

    The Fréchet distance is the minimum required "leash length" to connect two
    points traversing their respective curves. It is a robust measure of
    trajectory shape similarity. This implementation is vectorized across
    any number of leading batch dimensions.

    Args:
        t1 (np.ndarray): First batch of trajectories. Shape (..., H, d), where
            ... is any number of batch dimensions, H is the horizon, and d is
            the spatial dimension.
        t2 (np.ndarray): Second batch of trajectories. Must have the same shape
            as t1.

    Returns:
        np.ndarray: A scalar distance for each pair in the batch.
            Shape (...).

    Raises:
        ValueError: If input shapes are not identical or have fewer than 2 dimensions.
    """
    # --- Input Validation ---
    if t1.shape != t2.shape:
        raise ValueError(f"Input trajectory shapes must be identical, but got "
                         f"{t1.shape} and {t2.shape}")
    if t1.ndim < 2:
        raise ValueError(f"Input trajectories must have at least 2 dimensions "
                         f"(H, d), but got {t1.ndim}")

    # --- Initialization ---
    H = t1.shape[-2]
    batch_shape = t1.shape[:-2]

    # Calculate the pairwise distance matrix between all points
    cost_matrix = _pairwise_distance_matrix(t1, t2, norm_ord=norm_ord)

    # Initialize the dynamic programming matrix for Fréchet distance
    frechet_matrix = np.full(cost_matrix.shape, -1.0, dtype=np.float64)

    # --- Dynamic Programming Recurrence ---
    # The loops over i and j are necessary for the DP algorithm.
    # However, all operations inside the loops are vectorized over the batch dimensions.

    # Initialize the first cell
    frechet_matrix[..., 0, 0] = cost_matrix[..., 0, 0]

    # Fill the first column
    for i in range(1, H):
        frechet_matrix[..., i, 0] = np.maximum(
            frechet_matrix[..., i-1, 0], cost_matrix[..., i, 0]
        )

    # Fill the first row
    for j in range(1, H):
        frechet_matrix[..., 0, j] = np.maximum(
            frechet_matrix[..., 0, j-1], cost_matrix[..., 0, j]
        )

    # Fill the rest of the matrix
    for i in range(1, H):
        for j in range(1, H):
            min_prev = np.minimum(
                frechet_matrix[..., i-1, j],
                np.minimum(frechet_matrix[..., i, j-1], frechet_matrix[..., i-1, j-1])
            )
            frechet_matrix[..., i, j] = np.maximum(cost_matrix[..., i, j], min_prev)

    # The Fréchet distance is the value in the top-right cell
    return frechet_matrix[..., H-1, H-1]


def dtw_distance(t1: np.ndarray, t2: np.ndarray, dist_norm_ord=2, norm_type='path') -> np.ndarray:
    """
    Computes the Dynamic Time Warping (DTW) distance between two batches of trajectories.

    DTW finds the optimal non-linear alignment between two time series,
    minimizing the cumulative distance. It is useful for comparing trajectories
    with similar shapes but different speeds or temporal alignments. This
    implementation is vectorized across any number of leading batch dimensions.

    Args:
        t1 (np.ndarray): First batch of trajectories. Shape (..., H, d), where
            `...` is any number of batch dimensions, H is the horizon, and d is
            the spatial dimension.
        t2 (np.ndarray): Second batch of trajectories. Must have the same shape
            as t1.
        dist_norm_ord (int or str): The order of the norm to use for distance calculation.
            Can be a positive integer or 'fro' for Frobenius norm, 'nuc' for nuclear norm, or None for L2 norm.
        norm_type (str): The type of normalization to apply to the DTW distance.
            Can be 'raw' (no normalization), 'path' (normalized by path length), or 'symmetric'
            (normalized by the sum of the lengths of both trajectories). 

    Returns:
        np.ndarray: A scalar distance for each pair in the batch.
            Shape (...).

    Raises:
        ValueError: If input shapes are not identical or have fewer than 2 dimensions.
    """
    # --- Input Validation ---
    if t1.shape != t2.shape:
        raise ValueError(f"Input trajectory shapes must be identical, but got "
                         f"{t1.shape} and {t2.shape}")
    if t1.ndim < 2:
        raise ValueError(f"Input trajectories must have at least 2 dimensions "
                         f"(H, d), but got {t1.ndim}")

    assert norm_type in ['raw', 'path', 'symmetric'], \
        f"Invalid norm_type '{norm_type}'. Must be one of 'raw', 'path', or 'symmetric'."

    # --- Initialization ---
    H = t1.shape[-2]
    batch_shape = t1.shape[:-2]

    cost_matrix = _pairwise_distance_matrix(t1, t2, norm_ord=dist_norm_ord)
    dtw_matrix = np.full(cost_matrix.shape, np.inf, dtype=np.float64)
    path_len_matrix = np.zeros(cost_matrix.shape, dtype=np.int32)

    dtw_matrix[..., 0, 0] = cost_matrix[..., 0, 0]
    path_len_matrix[..., 0, 0] = 1

    for i in range(1, H):
        dtw_matrix[..., i, 0] = cost_matrix[..., i, 0] + dtw_matrix[..., i-1, 0]
        path_len_matrix[..., i, 0] = 1 + path_len_matrix[..., i-1, 0]
    for j in range(1, H):
        dtw_matrix[..., 0, j] = cost_matrix[..., 0, j] + dtw_matrix[..., 0, j-1]
        path_len_matrix[..., 0, j] = 1 + path_len_matrix[..., 0, j-1]

    for i in range(1, H):
        for j in range(1, H):
            prev_costs_stack = np.stack(
                [dtw_matrix[..., i-1, j-1], dtw_matrix[..., i-1, j], dtw_matrix[..., i, j-1]],
                axis=0
            )
            prev_path_lens_stack = np.stack(
                [path_len_matrix[..., i-1, j-1], path_len_matrix[..., i-1, j], path_len_matrix[..., i, j-1]],
                axis=0
            )
            
            # 1. Find the index of the minimum cost neighbor ONCE.
            min_indices = np.argmin(prev_costs_stack, axis=0)
            
            # 2. Use these indices to gather the minimum costs.
            min_costs = np.take_along_axis(
                prev_costs_stack, min_indices[np.newaxis, ...], axis=0
            ).squeeze(axis=0)
            
            # 3. Use the same indices to gather the corresponding path lengths.
            chosen_prev_len = np.take_along_axis(
                prev_path_lens_stack, min_indices[np.newaxis, ...], axis=0
            ).squeeze(axis=0)

            # 4. Update the matrices.
            dtw_matrix[..., i, j] = cost_matrix[..., i, j] + min_costs
            path_len_matrix[..., i, j] = 1 + chosen_prev_len

    raw_score = dtw_matrix[..., -1, -1]
    
    if norm_type == 'raw':
        return raw_score
    elif norm_type == 'path':
        path_len = path_len_matrix[..., -1, -1]
        return np.divide(raw_score, path_len, out=np.zeros_like(raw_score), where=path_len!=0)
    elif norm_type == 'symmetric':
        return raw_score / (H + t2.shape[-2])


