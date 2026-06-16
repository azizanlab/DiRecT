"""
Batched JAX SCP+ADMM Collision Avoidance Projector.

A fully JAX-jittable, GPU-accelerated trajectory projector using
Successive Convexification (SCP) with ADMM inner solves for multi-robot
collision avoidance. Drop-in replacement for IpoptProjector with the same
BaseProjector.project(x, params) interface.

Internal sections:
  1. Data Structures
  2. Torch-JAX Bridge
  3. Obstacle Oracle (SDF margin + normal)
  4. Active Constraint Selection
  5. Tridiagonal Solver (Thomas algorithm)
  6. Velocity Projection (Z update)
  7. Local Collision QP (C, s update)
  8. ADMM Inner Loop
  9. SCP Outer Loop
 10. JaxScpAdmmProjector class
"""

import os
import logging
import functools
from typing import List, Dict, Optional, Tuple, NamedTuple

import numpy as np
import torch

import jax
import jax.numpy as jnp
from jax import lax

from mmd.models.projection.projectors import BaseProjector
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    DistanceField,
    CollisionObjectDistanceField,
    CollisionWorkspaceBoundariesDistanceField,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class SphereObstacles(NamedTuple):
    centers: jnp.ndarray   # (N_sph, 2)
    radii: jnp.ndarray     # (N_sph,)

class BoxObstacles(NamedTuple):
    centers: jnp.ndarray       # (N_box, 2)
    half_sizes: jnp.ndarray    # (N_box, 2)
    corner_radii: jnp.ndarray  # (N_box,)

class ObstacleSet(NamedTuple):
    # Unified flat arrays for vectorized oracle.
    # All obstacles are stored as "generalized rounded boxes":
    #   spheres  -> half_sizes=0, corner_radius=radius
    #   boxes    -> corner_radius=0
    #   rounded  -> as-is
    centers: jnp.ndarray       # (N_obs, 2)
    half_sizes: jnp.ndarray    # (N_obs, 2)
    corner_radii: jnp.ndarray  # (N_obs,)
    n_total: int

class ActiveConstraintDescriptors(NamedTuple):
    # Robot-robot: (B, H, K_rr, ...)
    rr_i_idx: jnp.ndarray     # (B, H, K_rr) int
    rr_j_idx: jnp.ndarray     # (B, H, K_rr) int
    rr_normal: jnp.ndarray    # (B, H, K_rr, 2)
    rr_rhs: jnp.ndarray       # (B, H, K_rr)
    rr_mask: jnp.ndarray      # (B, H, K_rr) bool
    # Obstacle: (B, H, K_obs, ...)
    obs_robot_idx: jnp.ndarray  # (B, H, K_obs) int
    obs_obs_idx: jnp.ndarray    # (B, H, K_obs) int
    obs_normal: jnp.ndarray     # (B, H, K_obs, 2)
    obs_rhs: jnp.ndarray        # (B, H, K_obs)
    obs_mask: jnp.ndarray       # (B, H, K_obs) bool

class AdmmState(NamedTuple):
    X: jnp.ndarray       # (B, H, N_r, 2)
    V: jnp.ndarray       # (B, H, N_r, 2)
    Z: jnp.ndarray       # (B, H-1, N_r, 2)
    C: jnp.ndarray       # (B, H, N_r, 2)
    s: jnp.ndarray       # (B, H, K_rr + K_obs)
    U_xv: jnp.ndarray    # (B, H, N_r, 2)
    U_xc: jnp.ndarray    # (B, H, N_r, 2)
    U_z: jnp.ndarray     # (B, H-1, N_r, 2)

class ScpState(NamedTuple):
    X: jnp.ndarray
    admm_state: AdmmState
    mu: jnp.ndarray       # scalar
    tau: jnp.ndarray       # scalar


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Torch-JAX Bridge
# ═══════════════════════════════════════════════════════════════════════════════

def _torch_to_jax(t: torch.Tensor) -> jnp.ndarray:
    return jnp.asarray(t.detach().cpu().numpy())


def _jax_to_torch(a: jnp.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=dtype, device=device)


def _repack_batch_to_grouped(
    x: jnp.ndarray,
    pos_init: jnp.ndarray,
    pos_final: Optional[jnp.ndarray],
    n_agents: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Convert from IpoptProjector batch layout to grouped time-major layout.

    IPOPT layout: batch[a * group_size + g] = agent a, sample g.
    Output:  (group_size, H, n_agents, 2) trajectories
             (group_size, n_agents, 2) positions
    """
    B, H, dim = x.shape
    group_size = B // n_agents

    # x: (B, H, dim) -> (n_agents, group_size, H, dim) -> (group_size, H, n_agents, dim)
    x_grouped = x.reshape(n_agents, group_size, H, dim)
    x_grouped = jnp.transpose(x_grouped, (1, 2, 0, 3))

    pos_init_grouped = pos_init.reshape(n_agents, group_size, dim)
    pos_init_grouped = jnp.transpose(pos_init_grouped, (1, 0, 2))

    pos_final_grouped = None
    if pos_final is not None:
        pos_final_grouped = pos_final.reshape(n_agents, group_size, dim)
        pos_final_grouped = jnp.transpose(pos_final_grouped, (1, 0, 2))

    return x_grouped, pos_init_grouped, pos_final_grouped


def _unpack_grouped_to_batch(
    X: jnp.ndarray, n_agents: int, B: int
) -> jnp.ndarray:
    """Inverse of _repack_batch_to_grouped."""
    group_size, H, _n, dim = X.shape
    # (group_size, H, n_agents, dim) -> (n_agents, group_size, H, dim) -> (B, H, dim)
    X_t = jnp.transpose(X, (2, 0, 1, 3))
    return X_t.reshape(B, H, dim)


def _extract_obstacle_params_from_fields(
    collision_fields: List[DistanceField],
    ws_bounds: Optional[torch.Tensor],
    robot_radius: float,
) -> Tuple[ObstacleSet, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Walk collision_fields exactly as IpoptProjector.__init__ does and extract
    obstacle geometry as JAX arrays in a unified "generalized rounded box" format.
    """
    from torch_robotics.environments.primitives import (
        ObjectField, MultiSphereField, MultiBoxField, MultiRoundedBoxField,
    )
    from torch_robotics.environments.grid_map_sdf import GridMapSDF

    ws_min_np = None
    ws_max_np = None
    all_centers = []
    all_half_sizes = []
    all_corner_radii = []

    for field in collision_fields:
        if isinstance(field, CollisionWorkspaceBoundariesDistanceField):
            ws_min_np = field.ws_min.cpu().numpy()
            ws_max_np = field.ws_max.cpu().numpy()
        elif isinstance(field, CollisionObjectDistanceField):
            if field.df_obj_list_fn is None:
                continue
            df_obj_list = field.df_obj_list_fn()
            # Filter out grid SDFs (not supported in this version)
            analytic_objs = [o for o in df_obj_list if not isinstance(o, GridMapSDF)]

            for obj in analytic_objs:
                if isinstance(obj, ObjectField):
                    pos = obj.pos.cpu().numpy()[:2]
                    for prim in obj.fields:
                        _extract_prim(prim, pos, all_centers, all_half_sizes, all_corner_radii)
                else:
                    _extract_prim(obj, np.zeros(2), all_centers, all_half_sizes, all_corner_radii)

    if ws_bounds is not None:
        ws_min_np = ws_bounds[0].cpu().numpy()
        ws_max_np = ws_bounds[1].cpu().numpy()

    # Return numpy arrays -- defer JAX conversion to first solve call
    if len(all_centers) == 0:
        obs_np = {
            'centers': np.zeros((0, 2), dtype=np.float32),
            'half_sizes': np.zeros((0, 2), dtype=np.float32),
            'corner_radii': np.zeros((0,), dtype=np.float32),
            'n_total': 0,
        }
    else:
        obs_np = {
            'centers': np.array(all_centers, dtype=np.float32),
            'half_sizes': np.array(all_half_sizes, dtype=np.float32),
            'corner_radii': np.array(all_corner_radii, dtype=np.float32),
            'n_total': len(all_centers),
        }

    return obs_np, ws_min_np, ws_max_np


def _extract_prim(prim, pos_offset, all_centers, all_half_sizes, all_corner_radii):
    """Extract individual shape instances from a primitive into flat lists."""
    from torch_robotics.environments.primitives import (
        MultiSphereField, MultiBoxField, MultiRoundedBoxField,
    )

    if isinstance(prim, MultiSphereField):
        c_np = prim.centers.cpu().numpy()
        if c_np.ndim < 2 or c_np.shape[0] == 0:
            return
        centers = c_np[:, :2]
        radii = prim.radii.cpu().numpy()
        for i in range(len(centers)):
            r = float(radii[i])
            all_centers.append(centers[i] + pos_offset)
            # Sphere = rounded box with half_sizes=r, corner_radius=r.
            # This makes the rounded-box SDF correctly reduce to ||p-c|| - r.
            all_half_sizes.append(np.array([r, r]))
            all_corner_radii.append(r)

    elif isinstance(prim, (MultiRoundedBoxField, MultiBoxField)):
        c_np = prim.centers.cpu().numpy()
        if c_np.ndim < 2 or c_np.shape[0] == 0:
            return
        centers = c_np[:, :2]
        half_sizes = prim.half_sizes.cpu().numpy()[:, :2]
        has_radius = hasattr(prim, 'radius') and prim.radius is not None
        radii = prim.radius.cpu().numpy() if has_radius else None
        for i in range(len(centers)):
            r = float(radii[i]) if radii is not None else 0.0
            all_centers.append(centers[i] + pos_offset)
            all_half_sizes.append(half_sizes[i])
            all_corner_radii.append(r)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Obstacle Oracle (SDF margin + normal)
# ═══════════════════════════════════════════════════════════════════════════════

_SDF_EPS = 1e-12


def _rounded_box_sdf(p, center, half_size, corner_radius):
    """
    Signed distance from point p to a rounded box.
    Matches ipopt.py _primitive_sdf_exprs (lines 384-402).

    For spheres stored as half_size=0, corner_radius=r, this reduces to:
      sqrt(||p-c||^2 + eps) - r, which matches the sphere SDF.
    """
    q = jnp.abs(p - center) - half_size + corner_radius
    q_pos = jnp.maximum(q, 0.0)
    outer = jnp.sqrt(jnp.dot(q_pos, q_pos) + _SDF_EPS)
    inner = jnp.minimum(jnp.max(q), 0.0)
    return outer + inner - corner_radius


def _obstacle_margin_single(p, center, half_size, corner_radius, robot_radius):
    """Signed margin for a single robot position vs a single obstacle."""
    sdf_val = _rounded_box_sdf(p, center, half_size, corner_radius)
    return sdf_val - robot_radius


def _obstacle_margin_and_normal_single(p, center, half_size, corner_radius, robot_radius):
    """Margin and its gradient (normal) w.r.t. p."""
    margin_fn = lambda pp: _obstacle_margin_single(pp, center, half_size, corner_radius, robot_radius)
    margin = margin_fn(p)
    normal = jax.grad(margin_fn)(p)
    return margin, normal


def _all_obstacle_margins_single_robot(p, obs_set, robot_radius):
    """Margins for one robot position against all obstacles. Returns (N_obs,)."""
    return jax.vmap(
        lambda c, hs, cr: _obstacle_margin_single(p, c, hs, cr, robot_radius)
    )(obs_set.centers, obs_set.half_sizes, obs_set.corner_radii)


def _all_obstacle_margins(positions, obs_set, robot_radius):
    """
    Margins for all robots at one timestep against all obstacles.
    positions: (N_robots, 2)
    Returns: (N_robots, N_obs)
    """
    return jax.vmap(
        lambda p: _all_obstacle_margins_single_robot(p, obs_set, robot_radius)
    )(positions)


def _obstacle_margin_and_normal_for_pair(p, obs_idx, obs_set, robot_radius):
    """Margin and normal for one (robot_position, obstacle_index) pair."""
    center = obs_set.centers[obs_idx]
    half_size = obs_set.half_sizes[obs_idx]
    corner_radius = obs_set.corner_radii[obs_idx]
    return _obstacle_margin_and_normal_single(p, center, half_size, corner_radius, robot_radius)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Active Constraint Selection
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_robot_robot_margins_at_timestep(positions, pair_i, pair_j, min_sep):
    """
    Compute robot-robot separation margins for all pre-computed pairs at one timestep.
    positions: (N_robots, 2)
    pair_i, pair_j: (N_pairs,) int arrays of robot index pairs
    Returns: (N_pairs,) margins = ||p_i - p_j|| - min_sep
    """
    pi = positions[pair_i]   # (N_pairs, 2)
    pj = positions[pair_j]   # (N_pairs, 2)
    diff = pi - pj
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + _SDF_EPS)
    return dist - min_sep


def _compute_robot_robot_margins(X, pair_i, pair_j, min_sep):
    """
    X: (B, H, N_robots, 2)
    Returns: (B, H, N_pairs)
    """
    return jax.vmap(jax.vmap(
        lambda pos: _compute_robot_robot_margins_at_timestep(pos, pair_i, pair_j, min_sep)
    ))(X)


def _compute_obstacle_margins(X, obs_set, robot_radius):
    """
    X: (B, H, N_robots, 2)
    Returns: (B, H, N_robots * N_obs)
    """
    def per_timestep(positions):
        # (N_robots, N_obs) -> flatten to (N_robots * N_obs,)
        m = _all_obstacle_margins(positions, obs_set, robot_radius)
        return m.ravel()

    return jax.vmap(jax.vmap(per_timestep))(X)


def _select_active_constraints(
    X: jnp.ndarray,
    obs_set: ObstacleSet,
    robot_radius: float,
    min_sep: float,
    K_rr: int,
    K_obs: int,
    pair_i: jnp.ndarray,
    pair_j: jnp.ndarray,
    N_robots: int,
    margin_threshold: float = 0.0,
    eps_normal: float = 1e-8,
) -> ActiveConstraintDescriptors:
    """
    Evaluate all margins, select those below margin_threshold, build sparse
    linearization descriptors with masks.  Capped at K_rr / K_obs for fixed
    shapes; if more than K violate, the worst K are kept.

    margin_threshold: only constraints with margin < threshold are active.
        Set to 0.0 to activate only penetrating/touching constraints, or
        to a positive value to include a safety buffer.

    X: (B, H, N_robots, 2)
    """
    B, H, Nr, _ = X.shape
    N_obs = obs_set.n_total
    N_pairs = pair_i.shape[0]

    # --- Robot-robot ---
    if N_pairs > 0 and K_rr > 0:
        rr_margins = _compute_robot_robot_margins(X, pair_i, pair_j, min_sep)  # (B, H, N_pairs)

        def select_rr_per_bh(margins_bh, positions_bh):
            # Set margins above threshold to +inf so they sort to the end
            thresholded = jnp.where(margins_bh < margin_threshold, margins_bh, jnp.inf)
            # Pad if N_pairs < K_rr
            padded = jnp.concatenate([thresholded, jnp.full(max(K_rr - N_pairs, 0), jnp.inf)])[:K_rr]
            _, top_indices = lax.top_k(-padded, K_rr)

            top_indices = jnp.clip(top_indices, 0, N_pairs - 1)
            # Mask: valid index AND margin was below threshold
            selected_margins = margins_bh[top_indices]
            mask = (jnp.arange(K_rr) < N_pairs) & (selected_margins < margin_threshold)

            sel_i = pair_i[top_indices]
            sel_j = pair_j[top_indices]

            pi = positions_bh[sel_i]
            pj = positions_bh[sel_j]
            diff = pi - pj
            dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1, keepdims=True) + _SDF_EPS)
            normal = diff / jnp.maximum(dist, eps_normal)
            rhs = jnp.full(K_rr, min_sep)

            return sel_i, sel_j, normal, rhs, mask

        sel_i, sel_j, normals, rhs, masks = jax.vmap(jax.vmap(select_rr_per_bh))(
            rr_margins, X
        )
    else:
        sel_i = jnp.zeros((B, H, K_rr), dtype=jnp.int32)
        sel_j = jnp.zeros((B, H, K_rr), dtype=jnp.int32)
        normals = jnp.zeros((B, H, K_rr, 2))
        rhs = jnp.zeros((B, H, K_rr))
        masks = jnp.zeros((B, H, K_rr), dtype=jnp.bool_)

    # --- Obstacle ---
    if N_obs > 0 and K_obs > 0:
        obs_margins = _compute_obstacle_margins(X, obs_set, robot_radius)  # (B, H, N_robots*N_obs)
        N_cand = Nr * N_obs

        robot_indices_flat = jnp.repeat(jnp.arange(Nr), N_obs)
        obs_indices_flat = jnp.tile(jnp.arange(N_obs), Nr)

        def select_obs_per_bh(margins_bh, positions_bh):
            thresholded = jnp.where(margins_bh < margin_threshold, margins_bh, jnp.inf)
            padded = jnp.concatenate([thresholded, jnp.full(max(K_obs - N_cand, 0), jnp.inf)])[:K_obs]
            _, top_indices = lax.top_k(-padded, K_obs)
            top_indices = jnp.clip(top_indices, 0, N_cand - 1)

            selected_margins = margins_bh[top_indices]
            mask = (jnp.arange(K_obs) < N_cand) & (selected_margins < margin_threshold)

            sel_robot = robot_indices_flat[top_indices]
            sel_obs = obs_indices_flat[top_indices]

            def get_normal_rhs(r_idx, o_idx):
                p = positions_bh[r_idx]
                margin, normal = _obstacle_margin_and_normal_for_pair(
                    p, o_idx, obs_set, robot_radius
                )
                rhs_val = jnp.dot(normal, p) - margin
                return normal, rhs_val

            obs_normals, obs_rhs = jax.vmap(get_normal_rhs)(sel_robot, sel_obs)
            return sel_robot, sel_obs, obs_normals, obs_rhs, mask

        obs_sel_robot, obs_sel_obs, obs_normals_out, obs_rhs_out, obs_masks = jax.vmap(jax.vmap(
            select_obs_per_bh
        ))(obs_margins, X)
    else:
        obs_sel_robot = jnp.zeros((B, H, K_obs), dtype=jnp.int32)
        obs_sel_obs = jnp.zeros((B, H, K_obs), dtype=jnp.int32)
        obs_normals_out = jnp.zeros((B, H, K_obs, 2))
        obs_rhs_out = jnp.zeros((B, H, K_obs))
        obs_masks = jnp.zeros((B, H, K_obs), dtype=jnp.bool_)

    return ActiveConstraintDescriptors(
        rr_i_idx=sel_i,
        rr_j_idx=sel_j,
        rr_normal=normals,
        rr_rhs=rhs,
        rr_mask=masks,
        obs_robot_idx=obs_sel_robot,
        obs_obs_idx=obs_sel_obs,
        obs_normal=obs_normals_out,
        obs_rhs=obs_rhs_out,
        obs_mask=obs_masks,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Tridiagonal Solver (Thomas Algorithm)
# ═══════════════════════════════════════════════════════════════════════════════

def _thomas_factorize(diag_main, diag_sub, diag_sup):
    """
    Precompute Thomas algorithm factorization for a tridiagonal system.

    diag_main: (H,)   main diagonal
    diag_sub:  (H-1,) sub-diagonal (below main)
    diag_sup:  (H-1,) super-diagonal (above main)

    Returns: (c_prime (H-1,), d_denom (H,)) for forward/backward sweeps.
    """
    H = diag_main.shape[0]

    def forward_step(carry, i):
        c_prev, d_prev = carry
        # c_prime[i] = diag_sup[i] / (diag_main[i+1] - diag_sub[i] * c_prev)
        # But we index from 0: for step i (0-indexed), we process row i+1
        denom = diag_main[i + 1] - diag_sub[i] * c_prev
        c_new = jnp.where(i < H - 2, diag_sup[i + 1] / denom, 0.0)
        d_new = denom
        return (c_new, d_new), (c_new, denom)

    # First row
    c0 = diag_sup[0] / diag_main[0]
    d0 = diag_main[0]

    if H == 1:
        return jnp.array([]), jnp.array([d0])

    (_, _), (c_primes, d_denoms) = lax.scan(
        forward_step,
        (c0, d0),
        jnp.arange(H - 1),
    )

    c_prime = jnp.concatenate([jnp.array([c0]), c_primes[:-1]])
    d_denom = jnp.concatenate([jnp.array([d0]), d_denoms])

    return c_prime, d_denom


def _thomas_solve(c_prime, d_denom, diag_sub, rhs):
    """
    Batched tridiagonal solve using pre-factored coefficients.

    c_prime: (H-1,)
    d_denom: (H,)
    diag_sub: (H-1,)
    rhs: (..., H) -- arbitrary leading batch dims

    Returns: (..., H) solution.
    """
    H = d_denom.shape[0]
    batch_shape = rhs.shape[:-1]

    # Flatten batch dims
    rhs_flat = rhs.reshape(-1, H)  # (batch, H)
    Nb = rhs_flat.shape[0]

    # Forward sweep: compute modified RHS
    def forward_step(d_prev, i):
        # d'[i] = (rhs[i] - diag_sub[i-1] * d'[i-1]) / d_denom[i]
        d_new = (rhs_flat[:, i] - diag_sub[i - 1] * d_prev) / d_denom[i]
        return d_new, d_new

    d0 = rhs_flat[:, 0] / d_denom[0]

    if H == 1:
        return d0.reshape(*batch_shape, H)

    _, d_primes = lax.scan(
        forward_step,
        d0,
        jnp.arange(1, H),
    )
    # d_primes: (H-1, batch)
    d_all = jnp.concatenate([d0[None], d_primes], axis=0)  # (H, batch)

    # Backward sweep
    def backward_step(x_next, i):
        # x[i] = d'[i] - c'[i] * x[i+1]
        x_i = d_all[i] - c_prime[i] * x_next
        return x_i, x_i

    x_last = d_all[-1]
    _, x_rev = lax.scan(
        backward_step,
        x_last,
        jnp.arange(H - 2, -1, -1),
    )
    # x_rev: (H-1, batch), in reverse order
    x_all = jnp.concatenate([x_rev[::-1], x_last[None]], axis=0)  # (H, batch)

    return x_all.T.reshape(*batch_shape, H)


def _build_time_laplacian_coeffs(H):
    """
    Build the 1D time Laplacian L_H = D^T D tridiagonal coefficients.
    L_H main: [1, 2, 2, ..., 2, 1], off: [-1, -1, ..., -1]
    """
    main = jnp.concatenate([jnp.array([1.0]),
                            jnp.full(H - 2, 2.0),
                            jnp.array([1.0])])
    off = jnp.full(H - 1, -1.0)
    return main, off


def _build_x_tridiag_coeffs(smoothness_weight, tau, rho_xv, rho_xc, H,
                            fix_start=True, fix_final=False):
    """H_X = (1 + tau + rho_xv + rho_xc)I + lambda * L_H, with endpoint pinning."""
    L_main, L_off = _build_time_laplacian_coeffs(H)
    diag_shift = 1.0 + tau + rho_xv + rho_xc
    main = diag_shift + smoothness_weight * L_main
    sub = smoothness_weight * L_off
    sup = smoothness_weight * L_off

    # Pin endpoints: replace rows with identity so X[0]=b[0], X[-1]=b[-1]
    if fix_start:
        main = main.at[0].set(1.0)
        sup = sup.at[0].set(0.0)
    if fix_final:
        main = main.at[-1].set(1.0)
        sub = sub.at[-1].set(0.0)

    return main, sub, sup


def _build_v_tridiag_coeffs(rho_xv, rho_z, H, fix_start=True, fix_final=False):
    """
    H_V = rho_xv * I + rho_z * L_H, with endpoint rows replaced by identity
    to enforce V[0] = pos_init (and V[-1] = pos_final if fix_final).
    """
    L_main, L_off = _build_time_laplacian_coeffs(H)
    main = rho_xv + rho_z * L_main
    sub = rho_z * L_off
    sup = rho_z * L_off

    # Pin start: row 0 becomes identity (main[0]=1, sup[0]=0)
    if fix_start:
        main = main.at[0].set(1.0)
        sup = sup.at[0].set(0.0)

    # Pin end: row H-1 becomes identity (main[-1]=1, sub[-1]=0)
    if fix_final:
        main = main.at[-1].set(1.0)
        sub = sub.at[-1].set(0.0)

    return main, sub, sup


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Velocity Projection (Z Update)
# ═══════════════════════════════════════════════════════════════════════════════

def _project_to_velocity_balls(y, dx_max, eps=1e-10):
    """
    Project each velocity increment onto the L2 ball of radius dx_max.
    y: (B, H-1, N_robots, 2)
    Returns: same shape, with each (robot, timestep) vector clipped.
    """
    norm = jnp.sqrt(jnp.sum(y ** 2, axis=-1, keepdims=True) + eps)
    scale = jnp.minimum(1.0, dx_max / norm)
    return y * scale


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Local Collision QP (C, s Update)
# ═══════════════════════════════════════════════════════════════════════════════

def _sparse_At_lambda(lam_rr, lam_obs, descriptors, Nr, K_rr, B, H):
    """
    Compute A^T @ lambda from sparse descriptors.
    Returns (B, H, Nr, 2) -- gradient contribution to c from dual variables.

    For robot-robot constraint l: a_l = [0..0, +n_l at i, 0..0, -n_l at j, 0..0]
    For obstacle constraint l:    a_l = [0..0, +g_l at r, 0..0]
    """
    b_idx = jnp.arange(B)[:, None, None]
    h_idx = jnp.arange(H)[None, :, None]

    result = jnp.zeros((B, H, Nr, 2))

    # Robot-robot: scatter +lambda*n to robot i, -lambda*n to robot j
    rr_contrib = lam_rr[:, :, :, None] * descriptors.rr_normal  # (B,H,K_rr,2)
    result = result.at[b_idx, h_idx, descriptors.rr_i_idx].add(rr_contrib)
    result = result.at[b_idx, h_idx, descriptors.rr_j_idx].add(-rr_contrib)

    # Obstacle: scatter +lambda*g to robot r
    obs_contrib = lam_obs[:, :, :, None] * descriptors.obs_normal  # (B,H,K_obs,2)
    result = result.at[b_idx, h_idx, descriptors.obs_robot_idx].add(obs_contrib)

    return result


def _sparse_A_c(C, descriptors, K_rr, K_obs, B, H):
    """
    Compute A @ c from sparse descriptors.
    Returns (rr_vals (B,H,K_rr), obs_vals (B,H,K_obs)).

    Robot-robot: a_l^T c = n_l^T (c_i - c_j)
    Obstacle:    a_l^T c = g_l^T c_r
    """
    b_idx = jnp.arange(B)[:, None, None]
    h_idx = jnp.arange(H)[None, :, None]

    ci_rr = C[b_idx, h_idx, descriptors.rr_i_idx]  # (B,H,K_rr,2)
    cj_rr = C[b_idx, h_idx, descriptors.rr_j_idx]
    rr_vals = jnp.sum(descriptors.rr_normal * (ci_rr - cj_rr), axis=-1)

    cr_obs = C[b_idx, h_idx, descriptors.obs_robot_idx]  # (B,H,K_obs,2)
    obs_vals = jnp.sum(descriptors.obs_normal * cr_obs, axis=-1)

    return rr_vals, obs_vals


def _solve_local_collision_qp_dual(
    q_h: jnp.ndarray,
    descriptors: ActiveConstraintDescriptors,
    rho_xc: float,
    mu: jnp.ndarray,
    N_robots: int,
    K_rr: int,
    K_obs: int,
    maxiter: int,
):
    """
    Solve the local collision QP per (batch, time) via dual projected gradient ascent.

    Primal problem:
      min_{c, s}  (rho_xc/2)||c - q||^2 + mu * 1^T s
      s.t.  a_l^T c + s_l >= b_l   for each constraint l
            s >= 0

    Dual (with 0 <= lambda <= mu):
      max_lambda  -(1/(2*rho_xc)) * ||A^T lambda||^2 + lambda^T (b - A q)

    Primal recovery: c* = q + (1/rho_xc) * A^T lambda*
                     s_l* = max(0, b_l - a_l^T c*)

    q_h: (B, H, N_robots, 2)
    Returns: (C_new, s_new)
    """
    B, H, Nr, _ = q_h.shape
    K_total = K_rr + K_obs

    # Compute A @ q (constant)
    Aq_rr, Aq_obs = _sparse_A_c(q_h, descriptors, K_rr, K_obs, B, H)

    # Residual: b - A @ q (positive = q violates constraint)
    resid_rr = (descriptors.rr_rhs - Aq_rr) * descriptors.rr_mask    # (B,H,K_rr)
    resid_obs = (descriptors.obs_rhs - Aq_obs) * descriptors.obs_mask  # (B,H,K_obs)

    # Initialize dual variables to 0
    lam_rr = jnp.zeros((B, H, K_rr))
    lam_obs = jnp.zeros((B, H, K_obs))

    # Step size for dual projected gradient.
    # Each constraint row has ||a||^2 ~ 1. With overlap on same robot,
    # spectral_norm(A A^T) can reach ~max_per_robot. Use rho_xc / 4 as default;
    # for many constraints on few robots, the caller should increase qp_maxiter.
    alpha = rho_xc / 4.0

    def dual_step(carry, _):
        lam_rr, lam_obs = carry

        # Gradient of dual: dq/dlambda = -(1/rho_xc) * A @ (A^T @ lambda) + (b - A @ q)
        At_lam = _sparse_At_lambda(lam_rr, lam_obs, descriptors, Nr, K_rr, B, H)
        AAt_lam_rr, AAt_lam_obs = _sparse_A_c(At_lam, descriptors, K_rr, K_obs, B, H)

        grad_rr = -(1.0 / rho_xc) * AAt_lam_rr + resid_rr
        grad_obs = -(1.0 / rho_xc) * AAt_lam_obs + resid_obs

        # Projected gradient ascent: lambda <- clip(lambda + alpha * grad, 0, mu)
        lam_rr_new = jnp.clip(lam_rr + alpha * grad_rr, 0.0, mu)
        lam_obs_new = jnp.clip(lam_obs + alpha * grad_obs, 0.0, mu)

        # Mask out inactive constraints
        lam_rr_new = lam_rr_new * descriptors.rr_mask
        lam_obs_new = lam_obs_new * descriptors.obs_mask

        return (lam_rr_new, lam_obs_new), None

    (lam_rr_out, lam_obs_out), _ = lax.scan(dual_step, (lam_rr, lam_obs), None, length=maxiter)

    # Primal recovery: c = q + (1/rho_xc) * A^T @ lambda
    At_lam = _sparse_At_lambda(lam_rr_out, lam_obs_out, descriptors, Nr, K_rr, B, H)
    C_out = q_h + (1.0 / rho_xc) * At_lam

    # Slack recovery: s_l = max(0, b_l - a_l^T c)
    Ac_rr, Ac_obs = _sparse_A_c(C_out, descriptors, K_rr, K_obs, B, H)
    s_rr = jnp.maximum(descriptors.rr_rhs - Ac_rr, 0.0) * descriptors.rr_mask
    s_obs = jnp.maximum(descriptors.obs_rhs - Ac_obs, 0.0) * descriptors.obs_mask
    s_out = jnp.concatenate([s_rr, s_obs], axis=-1)

    return C_out, s_out


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: ADMM Inner Loop
# ═══════════════════════════════════════════════════════════════════════════════

def _diff_operator(X):
    """First difference along time: DX[t] = X[t+1] - X[t]. Returns (..., H-1, ...)."""
    return X[:, 1:] - X[:, :-1]


def _x_solve(
    V, C, U_xv, U_xc, X_pred, X_scp,
    tau, rho_xv, rho_xc, x_c_prime, x_d_denom, x_diag_sub,
    pos_init, pos_final, fix_final_point,
):
    """
    X update: solve H_X @ X = b_X per channel.
    H_X has identity rows at endpoints so X[0]=pos_init, X[-1]=pos_final.

    b_X = X_pred + tau*X_scp + rho_xv*(V - U_xv) + rho_xc*(C - U_xc)
    with b[0] = pos_init, b[-1] = pos_final (matching identity rows).
    """
    B, H, Nr, two = V.shape
    b = X_pred + tau * X_scp + rho_xv * (V - U_xv) + rho_xc * (C - U_xc)

    # Pin endpoints in RHS (tridiag has identity rows there)
    b = b.at[:, 0].set(pos_init)
    if fix_final_point:
        b = b.at[:, -1].set(pos_final)

    # Reshape to (B * Nr * 2, H) for batched tridiag solve
    b_flat = b.transpose(0, 2, 3, 1).reshape(-1, H)
    x_flat = _thomas_solve(x_c_prime, x_d_denom, x_diag_sub, b_flat)
    return x_flat.reshape(B, Nr, two, H).transpose(0, 3, 1, 2)


def _v_solve(
    X, Z, U_xv, U_z,
    rho_xv, rho_z, v_c_prime, v_d_denom, v_diag_sub,
    pos_init, pos_final, fix_final_point,
):
    """
    V update: solve H_V @ V = b_V with fixed endpoints.
    H_V = rho_xv * I + rho_z * L_H  (tridiag in time)

    b_V = rho_xv * (X + U_xv) + rho_z * D^T (Z - U_z)

    For fixed endpoints, we directly set V[0] = pos_init, V[-1] = pos_final,
    and solve only interior points, folding endpoint contributions into the RHS.
    """
    B, H, Nr, two = X.shape

    # b_consensus = rho_xv * (X + U_xv)
    b = rho_xv * (X + U_xv)

    # D^T applied to (Z - U_z): spread increments back to positions
    # D^T y[t] = -y[t] + y[t-1], with boundary adjustments
    z_term = Z - U_z  # (B, H-1, Nr, 2)
    dt_z = jnp.zeros_like(X)
    dt_z = dt_z.at[:, 0].set(-z_term[:, 0])
    dt_z = dt_z.at[:, 1:-1].set(z_term[:, :-1] - z_term[:, 1:])
    dt_z = dt_z.at[:, -1].set(z_term[:, -1])
    b = b + rho_z * dt_z

    # Pin endpoints in RHS (tridiag system has identity rows at endpoints)
    b = b.at[:, 0].set(pos_init)
    if fix_final_point:
        b = b.at[:, -1].set(pos_final)

    # Solve -- endpoints are built into the tridiag, no overwrite needed
    b_flat = b.transpose(0, 2, 3, 1).reshape(-1, H)
    v_flat = _thomas_solve(v_c_prime, v_d_denom, v_diag_sub, b_flat)
    V_new = v_flat.reshape(B, Nr, two, H).transpose(0, 3, 1, 2)

    return V_new


def _admm_step(
    state: AdmmState,
    X_pred: jnp.ndarray,
    X_scp: jnp.ndarray,
    descriptors: ActiveConstraintDescriptors,
    x_c_prime, x_d_denom, x_diag_sub,
    v_c_prime, v_d_denom, v_diag_sub,
    tau, rho_xv, rho_xc, rho_z, mu, dx_max,
    pos_init, pos_final, fix_final_point,
    N_robots, K_rr, K_obs, qp_maxiter,
):
    """Single ADMM iteration."""

    # 1. X solve (endpoints pinned as non-DOFs)
    X_new = _x_solve(
        state.V, state.C, state.U_xv, state.U_xc, X_pred, X_scp,
        tau, rho_xv, rho_xc, x_c_prime, x_d_denom, x_diag_sub,
        pos_init, pos_final, fix_final_point,
    )

    # 2. V solve
    V_new = _v_solve(
        X_new, state.Z, state.U_xv, state.U_z,
        rho_xv, rho_z, v_c_prime, v_d_denom, v_diag_sub,
        pos_init, pos_final, fix_final_point,
    )

    # 3. Z update (velocity projection)
    DV = _diff_operator(V_new)  # (B, H-1, Nr, 2)
    y = DV + state.U_z
    Z_new = _project_to_velocity_balls(y, dx_max)

    # 4. C, s update (local collision QP)
    q_h = X_new + state.U_xc
    C_new, s_new = _solve_local_collision_qp_dual(
        q_h, descriptors, rho_xc, mu, N_robots, K_rr, K_obs, qp_maxiter,
    )

    # 5. Dual updates
    U_xv_new = state.U_xv + X_new - V_new
    U_xc_new = state.U_xc + X_new - C_new
    U_z_new = state.U_z + DV - Z_new

    return AdmmState(
        X=X_new, V=V_new, Z=Z_new, C=C_new, s=s_new,
        U_xv=U_xv_new, U_xc=U_xc_new, U_z=U_z_new,
    )


def _admm_loop(
    state: AdmmState,
    X_pred, X_scp, descriptors,
    x_c_prime, x_d_denom, x_diag_sub,
    v_c_prime, v_d_denom, v_diag_sub,
    tau, rho_xv, rho_xc, rho_z, mu, dx_max,
    pos_init, pos_final, fix_final_point,
    N_robots, K_rr, K_obs, qp_maxiter,
    K_admm,
):
    """Run K_admm ADMM iterations via lax.scan."""

    def step_fn(state, _):
        new_state = _admm_step(
            state, X_pred, X_scp, descriptors,
            x_c_prime, x_d_denom, x_diag_sub,
            v_c_prime, v_d_denom, v_diag_sub,
            tau, rho_xv, rho_xc, rho_z, mu, dx_max,
            pos_init, pos_final, fix_final_point,
            N_robots, K_rr, K_obs, qp_maxiter,
        )
        return new_state, None

    final_state, _ = lax.scan(step_fn, state, None, length=K_admm)
    return final_state


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: SCP Outer Loop
# ═══════════════════════════════════════════════════════════════════════════════

def _scp_solve(
    X_pred: jnp.ndarray,
    obs_set: ObstacleSet,
    robot_radius: float,
    min_sep: float,
    pair_i: jnp.ndarray,
    pair_j: jnp.ndarray,
    pos_init: jnp.ndarray,
    pos_final: Optional[jnp.ndarray],
    fix_final_point: bool,
    dx_max: float,
    N_robots: int,
    smoothness_weight: float,
    M: int,
    K_admm: int,
    K_rr: int,
    K_obs: int,
    rho_xv: float,
    rho_xc: float,
    rho_z: float,
    mu_init: float,
    mu_max: float,
    mu_mult: float,
    tau_init: float,
    tau_max: float,
    tau_mult: float,
    qp_maxiter: int,
    margin_threshold: float,
    safety_buffer: float,
):
    """
    Full SCP solve: outer successive convexification with ADMM inner loop.

    X_pred: (B, H, N_robots, 2) -- reference trajectory
    Returns: (X_solution, diagnostics_dict)
    """
    B, H, Nr, _ = X_pred.shape

    # Precompute V tridiagonal factorization (constant across all iterations)
    v_main, v_sub, v_sup = _build_v_tridiag_coeffs(
        rho_xv, rho_z, H, fix_start=True, fix_final=fix_final_point
    )
    v_c_prime, v_d_denom = _thomas_factorize(v_main, v_sub, v_sup)

    # pos_final fallback
    if pos_final is None:
        pos_final = X_pred[:, -1]  # (B, Nr, 2)

    # Initialize SCP state
    K_total = K_rr + K_obs
    init_admm = AdmmState(
        X=X_pred,
        V=X_pred,
        Z=_diff_operator(X_pred),
        C=X_pred,
        s=jnp.zeros((B, H, K_total)),
        U_xv=jnp.zeros_like(X_pred),
        U_xc=jnp.zeros_like(X_pred),
        U_z=jnp.zeros((B, H - 1, Nr, 2)),
    )

    init_scp = ScpState(
        X=X_pred,
        admm_state=init_admm,
        mu=jnp.array(mu_init),
        tau=jnp.array(tau_init),
    )

    def scp_step(scp_state, _):
        X_current = scp_state.X
        mu = scp_state.mu
        tau = scp_state.tau

        # 1. Select active constraints at current linearization point
        #    Use inflated radius/separation for safety buffer against SCP gap
        eff_robot_radius = robot_radius + safety_buffer
        eff_min_sep = min_sep + 2.0 * safety_buffer
        descriptors = _select_active_constraints(
            X_current, obs_set, eff_robot_radius, eff_min_sep,
            K_rr, K_obs, pair_i, pair_j, N_robots,
            margin_threshold=margin_threshold,
        )

        # 2. Build X tridiag factorization for current tau
        x_main, x_sub, x_sup = _build_x_tridiag_coeffs(
            smoothness_weight, tau, rho_xv, rho_xc, H,
            fix_start=True, fix_final=fix_final_point,
        )
        x_c_prime, x_d_denom = _thomas_factorize(x_main, x_sub, x_sup)

        # 3. Reset collision-related duals (constraint linearization changed)
        #    Keep X, V, Z, U_xv, U_z warm-started from previous SCP iteration.
        prev = scp_state.admm_state
        admm_init = AdmmState(
            X=prev.X, V=prev.V, Z=prev.Z,
            C=prev.X,  # reset C to current X
            s=jnp.zeros_like(prev.s),
            U_xv=prev.U_xv, U_xc=jnp.zeros_like(prev.U_xc), U_z=prev.U_z,
        )

        # Run ADMM inner loop
        admm_out = _admm_loop(
            admm_init,
            X_pred, X_current, descriptors,
            x_c_prime, x_d_denom, x_sub,
            v_c_prime, v_d_denom, v_sub,
            tau, rho_xv, rho_xc, rho_z, mu, dx_max,
            pos_init, pos_final, fix_final_point,
            N_robots, K_rr, K_obs, qp_maxiter,
            K_admm,
        )

        # 4. Update SCP state
        X_new = admm_out.X
        mu_new = jnp.minimum(mu * mu_mult, mu_max)
        tau_new = jnp.minimum(tau * tau_mult, tau_max)

        return ScpState(
            X=X_new,
            admm_state=admm_out,
            mu=mu_new,
            tau=tau_new,
        ), None

    final_scp, _ = lax.scan(scp_step, init_scp, None, length=M)

    X_out = final_scp.X

    # Final velocity projection: clamp consecutive steps to dx_max.
    # ADMM consensus between X and V may not be exact, so X can have
    # small velocity violations.  Walk forward from the anchor, clamping
    # each displacement to the L2 ball of radius dx_max.
    def _clamp_velocities(X, pos_init, dx_max):
        # X: (group_size, H, N_robots, 2), pos_init: (group_size, N_robots, 2)
        def _step(prev, x_t):
            delta = x_t - prev
            norm = jnp.sqrt(jnp.sum(delta ** 2, axis=-1, keepdims=True) + 1e-10)
            scale = jnp.minimum(1.0, dx_max / norm)
            clamped = prev + delta * scale
            return clamped, clamped
        _, X_clamped = lax.scan(_step, pos_init, jnp.moveaxis(X, 1, 0))
        return jnp.moveaxis(X_clamped, 0, 1)

    X_out = _clamp_velocities(X_out, pos_init, dx_max)

    # Diagnostics
    diagnostics = {
        'final_mu': final_scp.mu,
        'final_tau': final_scp.tau,
        'final_slack_sum': jnp.sum(final_scp.admm_state.s),
    }

    return X_out, diagnostics


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: JaxScpAdmmProjector Class
# ═══════════════════════════════════════════════════════════════════════════════

class JaxScpAdmmProjector(BaseProjector):
    """Batched JAX SCP+ADMM trajectory projector (GPU-accelerated).

    Drop-in replacement for IpoptProjector with the same
    ``BaseProjector.project(x, params)`` interface.

    Solves the same constrained projection as IpoptProjector (minimum-
    deviation from reference subject to velocity, inter-robot collision,
    obstacle, and workspace constraints) but uses a two-level iterative
    algorithm that is fully JAX-jittable and runs on GPU:

    **Outer loop — Successive Convexification (SCP), M iterations:**
        1. Linearize nonlinear collision constraints around current X.
        2. Select the K worst-violating constraints (K_rr inter-robot,
           K_obs obstacle) for a sparse active set.
        3. Solve the resulting convex sub-problem via ADMM.
        4. Update SCP penalty (mu *= mu_mult) and trust region (tau *= tau_mult).

    **Inner loop — ADMM consensus, K_admm iterations per SCP step:**
        Variables are split into consensus groups:
          - X : trajectory (coupled to reference via quadratic cost)
          - V : trajectory copy for velocity consensus
          - Z : velocity increments, projected onto L2 balls (||z_t|| <= dx_max)
          - C : trajectory copy for collision consensus
          - s : slack variables for collision constraints

        Each ADMM iteration:
          1. X-update: tridiagonal solve (Thomas algorithm, O(H) per agent)
             incorporating smoothness, reference tracking, and consensus terms.
          2. V-update: tridiagonal solve for velocity consensus.
          3. Z-update: per-timestep L2 ball projection for velocity limits.
          4. (C, s)-update: dual projected gradient ascent for the local
             collision QP with sparse constraint Jacobians.
          5. Dual variable updates to enforce consensus.

    A final forward velocity-clamp pass walks from the anchor point and
    projects each step to enforce hard velocity constraints post-SCP.

    Parameters
    ----------
    collision_fields : list of DistanceField
        Obstacle geometry extracted from the planning task.
    robot_radius : float
        Collision radius per robot (in the caller's coordinate frame).
    n_agents : int
        Number of robots whose trajectories are jointly projected.
    ws_bounds : torch.Tensor, optional
        Workspace limits, shape (2, dim).  Row 0 = lower, row 1 = upper.
    M : int
        Number of SCP outer iterations (default 5).
    K_admm : int
        Number of ADMM inner iterations per SCP step (default 50).
    K_rr, K_obs : int
        Maximum active inter-robot / obstacle constraints per timestep.
        Auto-sized to cover all candidates if smaller than needed.
    rho_xv, rho_xc, rho_z : float
        ADMM penalty weights for the (X,V), (X,C), and (V,Z) consensus
        constraints respectively.
    mu_init, mu_max, mu_mult : float
        SCP collision penalty schedule: mu starts at mu_init, multiplied
        by mu_mult each SCP iteration, capped at mu_max.
    tau_init, tau_max, tau_mult : float
        SCP trust-region (safety buffer) schedule.
    margin_threshold : float
        Constraints with margin above this value are ignored (not active).
    safety_buffer : float
        Extra inflation of robot radius during SCP to compensate for
        linearization error.
    """

    def __init__(
        self,
        collision_fields: List[DistanceField],
        robot_radius: float = 0.0,
        n_agents: int = 1,
        ws_bounds: Optional[torch.Tensor] = None,
        use_obstacle_constraints: bool = True,
        use_inter_robot_constraints: bool = True,
        fix_final_point: bool = False,
        smoothness_weight: float = 0.0,
        # --- SCP parameters ---
        M: int = 5,
        K_admm: int = 50,
        K_rr: int = 16,
        K_obs: int = 16,
        # --- ADMM penalty parameters ---
        rho_xv: float = 1.0,
        rho_xc: float = 1.0,
        rho_z: float = 1.0,
        # --- SCP schedule parameters ---
        mu_init: float = 1.0,
        mu_max: float = 1000.0,
        mu_mult: float = 2.0,
        tau_init: float = 0.1,
        tau_max: float = 10.0,
        tau_mult: float = 1.5,
        # --- QP solver ---
        qp_maxiter: int = 100,
        # --- Constraint selection ---
        margin_threshold: float = 0.1,
        safety_buffer: float = 0.01,
        # --- Misc ---
        verbose: bool = False,
    ):
        super().__init__()

        self.robot_radius = robot_radius
        self.n_agents = n_agents
        self.fix_final_point = fix_final_point
        self.smoothness_weight = smoothness_weight
        self.verbose = verbose

        # Extract obstacle geometry as numpy (defer JAX conversion to first call)
        self._obs_np, ws_min_np, ws_max_np = _extract_obstacle_params_from_fields(
            collision_fields, ws_bounds, robot_radius
        )
        self.ws_min = ws_min_np
        self.ws_max = ws_max_np

        # SCP / ADMM parameters
        self.M = M
        self.K_admm = K_admm

        # Auto-size K_rr and K_obs to cover all candidate constraints
        n_pairs = n_agents * (n_agents - 1) // 2 if (use_inter_robot_constraints and n_agents > 1) else 0
        n_obs_total = self._obs_np['n_total'] if use_obstacle_constraints else 0
        self.K_rr = max(K_rr, n_pairs) if n_pairs > 0 else 0
        self.K_obs = max(K_obs, n_agents * n_obs_total) if n_obs_total > 0 else 0
        self.rho_xv = rho_xv
        self.rho_xc = rho_xc
        self.rho_z = rho_z
        self.mu_init = mu_init
        self.mu_max = mu_max
        self.mu_mult = mu_mult
        self.tau_init = tau_init
        self.tau_max = tau_max
        self.tau_mult = tau_mult
        self.qp_maxiter = qp_maxiter
        self.margin_threshold = margin_threshold
        self.safety_buffer = safety_buffer

        # Minimum separation distance for inter-robot constraints
        self.min_separation = 2.0 * robot_radius

        # Pre-compute inter-robot pair indices (numpy -- defer JAX to first call)
        pairs = []
        if use_inter_robot_constraints and n_agents > 1:
            for a in range(n_agents):
                for b in range(a + 1, n_agents):
                    pairs.append((a, b))
        if len(pairs) > 0:
            self._pair_i_np = np.array([p[0] for p in pairs], dtype=np.int32)
            self._pair_j_np = np.array([p[1] for p in pairs], dtype=np.int32)
        else:
            self._pair_i_np = np.zeros(0, dtype=np.int32)
            self._pair_j_np = np.zeros(0, dtype=np.int32)

        # JIT-compiled solver (lazy, built on first call)
        self._jit_solve = None
        self._cached_H = None

        logger.info(
            f'JaxScpAdmmProjector: {n_agents} agents, '
            f'{self._obs_np["n_total"]} obstacles, '
            f'M={M}, K_admm={K_admm}, K_rr={self.K_rr}, K_obs={self.K_obs}'
        )

    def _build_jit_solver(self, H: int):
        """Build and cache the JIT-compiled solver for a given horizon."""

        # Convert numpy arrays to JAX (first time touching JAX backend)
        obs_np = self._obs_np
        obs_set = ObstacleSet(
            centers=jnp.array(obs_np['centers']),
            half_sizes=jnp.array(obs_np['half_sizes']),
            corner_radii=jnp.array(obs_np['corner_radii']),
            n_total=obs_np['n_total'],
        )
        pair_i = jnp.array(self._pair_i_np)
        pair_j = jnp.array(self._pair_j_np)

        @jax.jit
        def solve(X_pred, pos_init, pos_final, dx_max_arr, smoothness_weight):
            return _scp_solve(
                X_pred=X_pred,
                obs_set=obs_set,
                robot_radius=self.robot_radius,
                min_sep=self.min_separation,
                pair_i=pair_i,
                pair_j=pair_j,
                pos_init=pos_init,
                pos_final=pos_final,
                fix_final_point=self.fix_final_point,
                dx_max=dx_max_arr,
                N_robots=self.n_agents,
                smoothness_weight=smoothness_weight,
                M=self.M,
                K_admm=self.K_admm,
                K_rr=self.K_rr,
                K_obs=self.K_obs,
                rho_xv=self.rho_xv,
                rho_xc=self.rho_xc,
                rho_z=self.rho_z,
                mu_init=self.mu_init,
                mu_max=self.mu_max,
                mu_mult=self.mu_mult,
                tau_init=self.tau_init,
                tau_max=self.tau_max,
                tau_mult=self.tau_mult,
                qp_maxiter=self.qp_maxiter,
                margin_threshold=self.margin_threshold,
                safety_buffer=self.safety_buffer,
            )

        self._jit_solve = solve
        self._cached_H = H
        if self.verbose:
            logger.info(f'JaxScpAdmmProjector: JIT-compiling solver for H={H}')

    def project(
        self,
        x: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Project trajectories onto the feasible set via SCP+ADMM.

        Args:
            x: Trajectory tensor, shape (B, H, dim). Agent a owns indices
                [a*group_size : (a+1)*group_size].
            params: Dict with:
                'dx_max' (float): max displacement per timestep.
                'pos_init' (Tensor): shape (B, dim), initial positions.
                'pos_final' (Tensor, optional): shape (B, dim), goal positions.

        Returns:
            Projected trajectories, same shape as x.
        """
        B, H, dim = x.shape
        n = self.n_agents
        group_size = B // n

        assert B % n == 0, f"Batch {B} not divisible by n_agents {n}."

        dx_max = float(params['dx_max'])
        pos_init = params['pos_init']
        pos_final = params.get('pos_final', None)

        # Convert to JAX
        x_jax = _torch_to_jax(x)
        pos_init_jax = _torch_to_jax(pos_init)
        pos_final_jax = _torch_to_jax(pos_final) if pos_final is not None else None

        # Repack batch layout: (B, H, dim) -> (group_size, H, n_agents, 2)
        x_grouped, pi_grouped, pf_grouped = _repack_batch_to_grouped(
            x_jax, pos_init_jax, pos_final_jax, n
        )

        # Build/rebuild JIT solver if horizon changed
        if self._jit_solve is None or self._cached_H != H:
            self._build_jit_solver(H)

        # Solve
        if pf_grouped is None:
            pf_grouped = x_grouped[:, -1]  # use last waypoint as fallback

        dx_max_arr = jnp.array(dx_max)
        sw_arr = jnp.array(self.smoothness_weight)

        X_sol, diagnostics = self._jit_solve(x_grouped, pi_grouped, pf_grouped, dx_max_arr, sw_arr)

        if self.verbose:
            for k, v in diagnostics.items():
                logger.info(f'  {k}: {v}')

        # Unpack back to (B, H, dim)
        result_jax = _unpack_grouped_to_batch(X_sol, n, B)

        return _jax_to_torch(result_jax, x.dtype, x.device)
