import torch
import torch.nn.functional as F
import numpy as np
import math

def dummy_cost(*inputs, **kwargs):
    """
    A dummy cost function that returns a tensor of zeros with the same shape as the input.
    This is useful for testing or as a placeholder when no cost is needed.
    
    Args:
        inputs: Input tensor, expected shape (B, H, d).
    
    Returns:
        A tensor of zeros with the same shape as the input.
    """
    ## Return a zero tensor of the same batch size as the first input 
    ## but keep the gradient flow.
    if len(inputs) == 0:
        raise ValueError("No input tensors provided to dummy_cost.")
    x:torch.Tensor = inputs[0]
    if x.dim() < 3:
        raise ValueError(f"Input tensor x must be at least 3D (B, H, d). Got {x.dim()}D.")
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device, requires_grad=True)


def exp_decay_l2_sqr_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    decay_rate: float = 10.0,
) -> torch.Tensor:
    """
    Calculates a cost that penalizes when x and y are close (in squared L2 norm)
    at any time step, using an exponential decay function.

    The penalty is higher when the L2 distance is small, and decays
    exponentially as the distance increases.

    Args:
        x: Input tensor, expected shape (B, H, d).
        y: Target tensor, expected shape (B, H, d).
        decay_rate: The decay rate (B) of the exponential function, determining
                    how sharply the penalty drops off with distance.

    Returns:
        A tensor of computed costs, shape (B,).
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors x and y must have the same shape.")
    if x.dim() < 3:
        raise ValueError(f"Input tensors x and y must be 3D (B, H, d). Got {x.dim()}D.")

    sq_diff = (x - y) ** 2

    sum_l2_sqr = sq_diff.sum(dim=(-2, -1))

    total_penalty_cost = torch.exp(-decay_rate * sum_l2_sqr)

    return total_penalty_cost


def exp_decay_l2_cost(
    x: torch.Tensor,
    y: torch.Tensor,
    decay_rate: float = 10.0,
) -> torch.Tensor:
    """
    Calculates a cost that penalizes when x and y are close (in squared L2 norm)
    at any time step, using an exponential decay function.

    The penalty is higher when the L2 distance is small, and decays
    exponentially as the distance increases.

    Args:
        x: Input tensor, expected shape (B, H, d).
        y: Target tensor, expected shape (B, H, d).
        decay_rate: The decay rate (B) of the exponential function, determining
                    how sharply the penalty drops off with distance.

    Returns:
        A tensor of computed costs, shape (B,).
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors x and y must have the same shape.")
    if x.dim() < 3:
        raise ValueError(f"Input tensors x and y must be 3D (B, H, d). Got {x.dim()}D.")

    l2 = torch.linalg.norm(x - y, dim=-1)

    total_penalty_cost = torch.exp(-decay_rate * l2.sum(dim=-1))

    return total_penalty_cost


def sum_log_l2_cost(
    *inputs: torch.Tensor,
    alpha: float,
    **kwargs,
) -> torch.Tensor:
    """
    Vectorized version of sum_log_l2_cost.

    Args:
        inputs: >=2 tensors, each of shape (B, H, d)
        alpha:  >0, small constant inside the log

    Returns:
        (B,) tensor: summed log-penalty per batch element.
    """
    assert len(inputs) >= 2, "At least two input tensors are required."
    assert alpha > 0, "Alpha must be > 0."
    x0 = inputs[0]
    assert x0.dim() >= 3, f"Input tensor must be at least 3D (B, H, d). Got {x0.dim()}D."

    for y in inputs[1:]:
        if y.shape != x0.shape or y.dim() < 3:
            raise ValueError("Expected inputs of shape (..., H, d).")

    # Stack: (B, H, N, d)
    X = torch.stack(inputs, dim=-2)  # same as before but no cdist
    N = len(inputs)

    # Upper-triangular pair indices (i<j)
    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=X.device)

    # Gather pairs: (B, H, P, d), where P = N*(N-1)//2
    Xi = X[..., i_idx, :]   # (B, H, P, d)
    Xj = X[..., j_idx, :]   # (B, H, P, d)

    # Distances for all P pairs at once
    D = torch.linalg.norm(Xi - Xj, dim=-1)  # (B, H, P)

    # Apply penalty and sum over pairs then time
    pair_costs = -torch.log(alpha + D)      # (B, H, P)
    return pair_costs.sum(dim=-1).sum(dim=-1)  # (B,)

def sum_log_l2_cost_loop(
    *inputs: torch.Tensor,
    alpha: float,
    **kwargs,
) -> torch.Tensor:
    """
    Computes a summed, negative-logarithmic penalty on squared L2 distances
    for batches of trajectories. At each time-step, the cost is
        -log( sum_d (x_hd - y_hd) + 1 )
    which strongly penalizes small distances and grows more gently as
    distance increases.

    Args:
        inputs (torch.Tensor): Tensor of shape (B, H, d).
        epsilon (float): Small constant added inside the log to avoid
                         singularity when distance is zero.

    Returns:
        torch.Tensor: A tensor of shape (B,), where each element is
                      the total log-penalty cost for one trajectory pair.
    """
    assert len(inputs) >= 2, "At least two input tensors are required for sum_log_l2_cost."
    assert alpha > 0, "Alpha must be a positive constant to avoid singularity in log."
    x = inputs[0]
    assert x.dim() >= 3, f"Input tensor x must be at least 3D (B, H, d). Got {x.dim()}D."
    for i, y in enumerate(inputs[1:]):
        if x.shape != y.shape or y.dim() < 3:
            raise ValueError("Expected inputs of shape (..., H, d).")

    # L2 norm per time-step -> (B, H) over every pair of inputs
    per_step_costs = 0.0
    for i in range(len(inputs) - 1):
        for j in range(i + 1, len(inputs)):
            per_step_costs += -torch.log(alpha + torch.linalg.norm(inputs[i] - inputs[j], dim=-1))
    
    # sum over time -> (B,)
    return per_step_costs.sum(dim=-1)


def hinge_cub_l2_cost(
    *inputs: torch.Tensor,
    active_range: float = 1.0,  # Radius for the collision cost
    **kwargs,
) -> torch.Tensor:
    assert len(inputs) >= 2, "At least two input tensors are required for sum_log_l2_cost."
    assert active_range > 0, "Alpha must be a positive constant to avoid singularity in log."
    x = inputs[0]
    assert x.dim() >= 3, f"Input tensor x must be at least 3D (B, H, d). Got {x.dim()}D."
    for i, y in enumerate(inputs[1:]):
        if x.shape != y.shape or y.dim() < 3:
            raise ValueError("Expected inputs of shape (..., H, d).")

    # L2 norm per time-step -> (B, H) over every pair of inputs
    per_step_costs = 0.0
    for i in range(len(inputs) - 1):
        for j in range(i + 1, len(inputs)): 
            l2_norm = torch.linalg.norm(inputs[i] - inputs[j], dim=-1)
            active_mask = l2_norm <= active_range
            per_step_costs += torch.where(
                active_mask,
                (active_range - l2_norm) ** 3, 
                torch.zeros_like(l2_norm)  # No cost outside the active range
            )

    return per_step_costs.sum(dim=-1)


def hinge_sqr_l2_cost(
    *inputs: torch.Tensor,
    active_range: float = 1.0,  # Radius for the collision cost
    **kwargs,
) -> torch.Tensor:
    """
    Vectorized hinge squared L2 cost.

    Args:
        inputs: >=2 tensors, each of shape (B, H, d)
        active_range: radius below which the hinge penalty is applied

    Returns:
        (B,) tensor: summed hinge-squared penalty per batch element.
    """
    assert len(inputs) >= 2, "At least two input tensors are required."
    assert active_range > 0, "active_range must be > 0."
    x0 = inputs[0]
    assert x0.dim() >= 3, f"Input tensor must be at least 3D (B, H, d). Got {x0.dim()}D."

    for y in inputs[1:]:
        if y.shape != x0.shape or y.dim() < 3:
            raise ValueError("Expected inputs of shape (..., H, d).")

    X = torch.stack(inputs, dim=-2)  # (..., H, N, d)
    N = len(inputs)

    i_idx, j_idx = torch.triu_indices(N, N, offset=1, device=X.device)
    Xi = X[..., i_idx, :]   # (..., H, P, d)
    Xj = X[..., j_idx, :]   # (..., H, P, d)

    D = torch.linalg.norm(Xi - Xj, dim=-1)
    pair_costs = F.relu(active_range - D) ** 2
    return pair_costs.sum(dim=-1).sum(dim=-1)


def hinge_sqr_l2_cost_loop(
    *inputs: torch.Tensor,
    active_range: float = 1.0,
    **kwargs,
) -> torch.Tensor:
    assert len(inputs) >= 2, "At least two input tensors are required for sum_log_l2_cost."
    assert active_range > 0, "Alpha must be a positive constant to avoid singularity in log."
    x = inputs[0]
    assert x.dim() >= 3, f"Input tensor x must be at least 3D (B, H, d). Got {x.dim()}D."
    for i, y in enumerate(inputs[1:]):
        if x.shape != y.shape or y.dim() < 3:
            raise ValueError("Expected inputs of shape (..., H, d).")

    # L2 norm per time-step -> (B, H) over every pair of inputs
    per_step_costs = 0.0
    for i in range(len(inputs) - 1):
        for j in range(i + 1, len(inputs)): 
            l2_norm = torch.linalg.norm(inputs[i] - inputs[j], dim=-1)
            active_mask = l2_norm <= active_range
            per_step_costs += torch.where(
                active_mask,
                (active_range - l2_norm) ** 2, 
                torch.zeros_like(l2_norm)  # No cost outside the active range
            )

    return per_step_costs.sum(dim=-1)


def dpp_cost(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7, **kwargs) -> torch.Tensor:
    """
    Computes a diversity score based on the Determinantal Point Process (DPP)
    for pairs of trajectories (x_i, y_i). The score is the log-determinant
    of the kernel matrix formed by each pair. A higher score corresponds to
    more diverse pairs of trajectories.

    This function handles tensors with 3 or more dimensions, applying the DPP
    calculation to the last two dimensions while preserving all leading dimensions.

    The group size is fixed to 2, with the groups being (x_i, y_i) for each
    item i in the batch dimensions.

    Args:
        x (torch.Tensor): The first batch of trajectories, shape (..., H, d).
        y (torch.Tensor): The second batch of trajectories, shape (..., H, d).
        eps (float): A small epsilon for numerical stability, used to clamp
                    the determinant before taking the logarithm.

    Returns:
        torch.Tensor: A tensor of costs with shape (...), where each element is the
                    log-determinant for the corresponding pair in the batch.
                    This cost should be minimized for diversity.
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors x and y must have the same shape.")
    if x.dim() < 3:
        raise ValueError(f"Input tensors must be at least 3D (..., H, d). Got {x.dim()}-D.")

    phi_x = torch.flatten(x, start_dim=-2)
    phi_y = torch.flatten(y, start_dim=-2)

    # Stack to form groups of 2 for each item in the batch dimensions.
    # phi will have a shape of (..., 2, H*d).
    phi = torch.stack([phi_x, phi_y], dim=-2)

    # Normalize feature vectors along the feature dimension (the last dimension).
    phi = torch.nn.functional.normalize(phi, p=2, dim=-1)

    S = phi @ phi.transpose(-1, -2)
    assert S.shape[-1] == S.shape[-2] == 2, "Kernel matrix S should be of shape (..., 2, 2)."

    # Compute the determinant for each 2x2 kernel matrix.
    # The result will have a shape of (...).
    det = torch.linalg.det(S)

    det = torch.clamp(torch.abs(det), min=eps)

    # The cost is the log determinant. Minimizing this cost encourages
    # diversity between the x and y trajectories.
    cost = -torch.log(det)

    return cost


# Registry for cost functions
cost_registry = {
    # "sum_recip_l2_sqr": sum_recip_l2_sqr_cost, 
    # "sum_recip_l2": sum_recip_l2_cost,
    # "exp_decay_l2_sqr": exp_decay_l2_sqr_cost, 
    # "exp_decay_l2": exp_decay_l2_cost,
    "dummy": dummy_cost,
    "sum_log_l2": sum_log_l2_cost,
    "hinge_cub_l2": hinge_cub_l2_cost,
    "hinge_sqr_l2": hinge_sqr_l2_cost,
    # "dpp": dpp_cost,
    # "sum_l2": sum_l2_cost,
}
