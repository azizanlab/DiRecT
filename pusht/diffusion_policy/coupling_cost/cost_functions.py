import torch
import numpy as np
import math
from itertools import combinations


def _pairwise_log_l2_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute log-barrier L2 cost between a single pair of trajectories."""
    l2 = torch.linalg.norm(x - y, dim=-1)  # (..., H)
    per_step = -torch.log(alpha + l2)
    return per_step.sum(dim=-1)  # (...)


def _pairwise_dpp_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute DPP diversity cost between a single pair of trajectories."""
    phi_x = torch.flatten(x, start_dim=-2)  # (..., H*d)
    phi_y = torch.flatten(y, start_dim=-2)
    phi = torch.stack([phi_x, phi_y], dim=-2)  # (..., 2, H*d)
    phi = torch.nn.functional.normalize(phi, p=2, dim=-1)
    S = phi @ phi.transpose(-1, -2)  # (..., 2, 2)
    det = torch.linalg.det(S)
    det = torch.clamp(torch.abs(det), min=eps)
    return -torch.log(det)  # (...)


def sum_log_l2_cost(
    *trajectories: torch.Tensor,
    alpha: float,
    **kwargs,
) -> torch.Tensor:
    """
    Computes a summed, negative-logarithmic penalty on L2 distances
    for groups of N trajectories. The total cost is the sum over all
    C(N,2) pairwise costs:  c(X^1,...,X^N) = sum_{i<j} c(X^i, X^j).

    For each pair at each time-step, the cost is -log(alpha + ||x_h - y_h||).

    Args:
        *trajectories: N tensors of shape (..., H, d). All must have the same shape.
        alpha (float): Small constant added inside the log to avoid
                       singularity when distance is zero.

    Returns:
        torch.Tensor: A tensor of shape (...), where each element is
                      the total log-penalty cost for one trajectory group.
    """
    N = len(trajectories)
    if N < 2:
        raise ValueError(f"Need at least 2 trajectories, got {N}.")
    ref_shape = trajectories[0].shape
    for t in trajectories:
        if t.shape != ref_shape or t.dim() < 3:
            raise ValueError("Expected all trajectories of shape (..., H, d).")

    total_cost = torch.zeros(ref_shape[:-2], device=trajectories[0].device,
                             dtype=trajectories[0].dtype)
    for i, j in combinations(range(N), 2):
        total_cost = total_cost + _pairwise_log_l2_cost(
            trajectories[i], trajectories[j], alpha)
    return total_cost


def dpp_cost(
    *trajectories: torch.Tensor,
    eps: float = 1e-7,
    **kwargs,
) -> torch.Tensor:
    """
    Computes a diversity cost based on the Determinantal Point Process (DPP)
    for groups of N trajectories. The total cost is the sum over all C(N,2)
    pairwise DPP costs:  c(X^1,...,X^N) = sum_{i<j} c_DPP(X^i, X^j).

    Each pairwise cost is -log(det(S)) where S is the 2x2 kernel matrix
    of L2-normalized, flattened trajectory features. Minimizing this
    encourages diversity between trajectories.

    Args:
        *trajectories: N tensors of shape (..., H, d). All must have the same shape.
        eps (float): A small epsilon for numerical stability, used to clamp
                    the determinant before taking the logarithm.

    Returns:
        torch.Tensor: A tensor of costs with shape (...), where each element is the
                    total pairwise DPP cost for the corresponding group.
    """
    N = len(trajectories)
    if N < 2:
        raise ValueError(f"Need at least 2 trajectories, got {N}.")
    ref_shape = trajectories[0].shape
    for t in trajectories:
        if t.shape != ref_shape:
            raise ValueError("All input tensors must have the same shape.")
        if t.dim() < 3:
            raise ValueError(f"Input tensors must be at least 3D (..., H, d). Got {t.dim()}-D.")

    total_cost = torch.zeros(ref_shape[:-2], device=trajectories[0].device,
                             dtype=trajectories[0].dtype)
    for i, j in combinations(range(N), 2):
        total_cost = total_cost + _pairwise_dpp_cost(
            trajectories[i], trajectories[j], eps)
    return total_cost


# Registry for cost functions
cost_registry = {
    "sum_log_l2": sum_log_l2_cost,
    "dpp": dpp_cost,
}
