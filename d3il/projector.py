"""
IpoptProjector: projects a reference trajectory onto a feasible set
defined by circular/planar obstacles and optional linear dynamics.

CasADi uses column-major (Fortran) order for matrices.
All constraint geometry is in real-world (unnormalized) coordinates;
the decision variable X is in normalized space.
"""

from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import casadi as cs
import numpy as np

from obstacles import CircularObstacle, PlanarObstacle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_and_solve(
    config: dict,
    X_ref: np.ndarray,
    s0: np.ndarray,
    reg_weight: float = None,
    a_prev: np.ndarray = None,
) -> Tuple[np.ndarray, str, int]:
    """Standalone function for parallel workers (must be picklable)."""
    proj = IpoptProjector(**config)
    return proj.project(X_ref, s0, reg_weight=reg_weight, a_prev=a_prev)


# ---------------------------------------------------------------------------
# IpoptProjector
# ---------------------------------------------------------------------------

class IpoptProjector:

    def __init__(
        self,
        circular_obstacles: List[CircularObstacle],
        planar_obstacles: List[PlanarObstacle],
        horizon: int,
        action_dim: int,
        state_dim: int,
        normalizer,
        dynamics: Optional[Dict[str, np.ndarray]] = None,
        dynamics_relaxation: float = 0.0,
        obstacle_margin: float = 0.0,
        input_saturation: bool = False,
        objective_fn: Optional[Callable] = None,
        objective_scale: float = 1.0,
        max_action_delta: float = 0.0,
        solver_opts: Optional[dict] = None,
    ):
        self.circular_obstacles = circular_obstacles
        self.planar_obstacles = planar_obstacles
        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.transition_dim = action_dim + state_dim
        self.normalizer = normalizer
        self.dynamics = dynamics
        self.dynamics_relaxation = dynamics_relaxation
        self.obstacle_margin = obstacle_margin
        self.input_saturation = input_saturation
        self.objective_fn = objective_fn
        self.objective_scale = objective_scale
        self.max_action_delta = max_action_delta

        # DOF: full trajectory minus the fixed initial state
        # layout: [a_0, | a_1, s_1, | a_2, s_2, | ... | a_{H-1}, s_{H-1}]
        self.dof = horizon * self.transition_dim - state_dim

        # normalizer bounds for x_pos (dim 2) and y_pos (dim 3) of observations
        obs_mins = normalizer.normalizers["observations"].mins
        obs_maxs = normalizer.normalizers["observations"].maxs
        self._xmin = float(obs_mins[2])
        self._xmax = float(obs_maxs[2])
        self._ymin = float(obs_mins[3])
        self._ymax = float(obs_maxs[3])

        # action bounds in normalized space ([-1, 1] for LimitsNormalizer)
        act_mins = normalizer.normalizers["actions"].mins
        act_maxs = normalizer.normalizers["actions"].maxs
        self._act_norm_lo = np.array([
            (lo - lo) / (hi - lo) * 2 - 1 for lo, hi in zip(act_mins, act_maxs)
        ])  # always -1
        self._act_norm_hi = np.array([
            (hi - lo) / (hi - lo) * 2 - 1 for lo, hi in zip(act_mins, act_maxs)
        ])  # always +1

        # precompute index arrays into the flat DOF vector
        self._precompute_indices()

        # solver options (IPOPT defaults, output silencing only)
        self._solver_opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }
        if solver_opts is not None:
            self._solver_opts.update(solver_opts)

        # build the CasADi problem
        self._formulate()

    # ------------------------------------------------------------------
    # Index precomputation
    # ------------------------------------------------------------------

    def _precompute_indices(self):
        """Compute indices into the flat DOF vector for states and actions.

        DOF layout:
          [a_0(action_dim) | a_1(action_dim), s_1(state_dim) | ... | a_{H-1}(action_dim), s_{H-1}(state_dim)]

        s_0 is NOT in the DOF — it is a parameter.
        """
        ad = self.action_dim
        sd = self.state_dim
        td = self.transition_dim

        # state indices for s_1 .. s_{H-1}  (H-1 states in DOF)
        # first state s_1 starts at offset ad (for a_0) + ad (for a_1) = 2*ad
        self._state_indices = []      # list of (start, end) for each state block
        self._xpos_indices = []       # index of x_pos (obs dim 2) for each state
        self._ypos_indices = []       # index of y_pos (obs dim 3) for each state

        for t in range(1, self.horizon):
            # block t: starts at ad + (t-1)*td + ad = ad + (t-1)*td + ad
            # actually: block 0 is just a_0 (ad elements)
            #           block t (t>=1) is a_t (ad) + s_t (sd), starting at ad + (t-1)*td
            block_start = ad + (t - 1) * td
            s_start = block_start + ad  # state starts after action
            s_end = s_start + sd

            self._state_indices.append((s_start, s_end))
            self._xpos_indices.append(s_start + 2)  # obs dim 2 = x_pos
            self._ypos_indices.append(s_start + 3)  # obs dim 3 = y_pos

        # action indices for a_0 .. a_{H-1}
        self._action_indices = []
        # a_0 at offset 0
        self._action_indices.append((0, ad))
        for t in range(1, self.horizon):
            block_start = ad + (t - 1) * td
            self._action_indices.append((block_start, block_start + ad))

    # ------------------------------------------------------------------
    # Unnormalization (normalized [-1,1] -> real-world)
    # ------------------------------------------------------------------

    def _unnorm_x(self, x_norm):
        return (x_norm + 1.0) / 2.0 * (self._xmax - self._xmin) + self._xmin

    def _unnorm_y(self, y_norm):
        return (y_norm + 1.0) / 2.0 * (self._ymax - self._ymin) + self._ymin

    # ------------------------------------------------------------------
    # Problem formulation
    # ------------------------------------------------------------------

    def _formulate(self):
        opti = cs.Opti()

        # decision variable (column vector, CasADi column-major)
        X = opti.variable(self.dof, 1)
        X_ref = opti.parameter(self.dof, 1)
        s0 = opti.parameter(self.state_dim, 1)
        reg_weight = opti.parameter(1, 1)
        opti.set_value(reg_weight, 1.0)

        # --- objective: projection ---
        cost = 0.5 * reg_weight * cs.sumsqr(X - X_ref)

        if self.objective_fn is not None:
            cost = cost + self.objective_scale * self.objective_fn(X, s0)

        opti.minimize(cost)

        # --- circular obstacle constraints (squared distance) ---
        for obs in self.circular_obstacles:
            cx, cy = obs.center
            r_eff_sq = (obs.radius + self.obstacle_margin) ** 2

            for xi, yi in zip(self._xpos_indices, self._ypos_indices):
                xu = self._unnorm_x(X[xi])
                yu = self._unnorm_y(X[yi])
                opti.subject_to((xu - cx) ** 2 + (yu - cy) ** 2 >= r_eff_sq)

        # --- planar obstacle constraints ---
        for pobs in self.planar_obstacles:
            slope = pobs.slope
            intercept = pobs.intercept

            for xi, yi in zip(self._xpos_indices, self._ypos_indices):
                xu = self._unnorm_x(X[xi])
                yu = self._unnorm_y(X[yi])
                opti.subject_to(slope * xu + intercept - yu >= self.obstacle_margin)

        # --- input saturation (action box constraints in normalized space) ---
        if self.input_saturation:
            for a_start, a_end in self._action_indices:
                for d in range(self.action_dim):
                    opti.subject_to(X[a_start + d] >= self._act_norm_lo[d])
                    opti.subject_to(X[a_start + d] <= self._act_norm_hi[d])

        # --- dynamics constraints (optional) ---
        if self.dynamics is not None:
            A = self.dynamics["A"]  # (state_dim, state_dim)
            B = self.dynamics["B"]  # (state_dim, action_dim)
            c_vec = self.dynamics["c"]  # (state_dim,)

            # CasADi is column-major: use DM for constant matrices
            A_cs = cs.DM(A)    # (state_dim x state_dim)
            B_cs = cs.DM(B)    # (state_dim x action_dim)
            c_cs = cs.DM(c_vec.reshape(-1, 1))  # (state_dim x 1)

            relax = self.dynamics_relaxation

            # t=0: s_1 = A @ s_0 + B @ a_0 + c  (s_0 is parameter)
            a0_start, a0_end = self._action_indices[0]
            s1_start, s1_end = self._state_indices[0]
            a_0 = X[a0_start:a0_end]
            s_1 = X[s1_start:s1_end]
            pred_0 = cs.mtimes(A_cs, s0) + cs.mtimes(B_cs, a_0) + c_cs
            if relax > 0:
                opti.subject_to(s_1 <= pred_0 + relax)
                opti.subject_to(s_1 >= pred_0 - relax)
            else:
                opti.subject_to(s_1 == pred_0)

            # t=1..H-2: s_{t+1} = A @ s_t + B @ a_t + c
            for t in range(1, self.horizon - 1):
                s_t_start, s_t_end = self._state_indices[t - 1]
                a_t_start, a_t_end = self._action_indices[t]

                if t >= len(self._state_indices):
                    break

                s_t = X[s_t_start:s_t_end]
                a_t = X[a_t_start:a_t_end]
                sn_start, sn_end = self._state_indices[t]
                s_next = X[sn_start:sn_end]
                pred_t = cs.mtimes(A_cs, s_t) + cs.mtimes(B_cs, a_t) + c_cs
                if relax > 0:
                    opti.subject_to(s_next <= pred_t + relax)
                    opti.subject_to(s_next >= pred_t - relax)
                else:
                    opti.subject_to(s_next == pred_t)

        self._a_prev = None

        # --- solver ---
        opti.solver("ipopt", self._solver_opts)

        # store references
        self._opti = opti
        self._X = X
        self._X_ref = X_ref
        self._s0 = s0
        self._reg_weight = reg_weight

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def set_regularization_weight(self, w: float):
        """Set the regularization weight for the tracking cost."""
        self._opti.set_value(self._reg_weight, w)

    def project(self, X_ref: np.ndarray, s0: np.ndarray, reg_weight: float = None, a_prev: np.ndarray = None) -> Tuple[np.ndarray, str, int]:
        """Project a single reference trajectory onto the feasible set.

        Args:
            X_ref: (dof,) flat DOF vector in normalized space.
            s0: (state_dim,) initial state in normalized space.
            reg_weight: optional regularization weight (overrides default 1.0).
            a_prev: (action_dim,) last executed action in normalized space (for action rate constraint).

        Returns:
            X_proj: (dof,) projected DOF vector.
            status: "optimal" if the solver converged, "failed" otherwise.
            n_iters: number of IPOPT iterations spent on the solve.
        """
        X_ref = np.asarray(X_ref).flatten()
        s0 = np.asarray(s0).flatten()

        assert X_ref.shape == (self.dof,), f"X_ref shape {X_ref.shape} != ({self.dof},)"
        assert s0.shape == (self.state_dim,), f"s0 shape {s0.shape} != ({self.state_dim},)"

        # CasADi parameters are column vectors
        self._opti.set_value(self._X_ref, X_ref.reshape(-1, 1))
        self._opti.set_value(self._s0, s0.reshape(-1, 1))
        if reg_weight is not None:
            self._opti.set_value(self._reg_weight, reg_weight)
        if a_prev is not None and self._a_prev is not None:
            self._opti.set_value(self._a_prev, np.asarray(a_prev).reshape(-1, 1))
        self._opti.set_initial(self._X, X_ref.reshape(-1, 1))

        try:
            sol = self._opti.solve()
            X_proj = np.array(sol.value(self._X)).flatten()
            status = "optimal"
            stats_src = self._opti
        except RuntimeError:
            X_proj = np.array(self._opti.debug.value(self._X)).flatten()
            status = "failed"
            stats_src = self._opti.debug

        try:
            n_iters = int(stats_src.stats()["iter_count"])
        except Exception:
            n_iters = 0

        return X_proj, status, n_iters

    def project_batch(
        self,
        X_refs: np.ndarray,
        s0s: np.ndarray,
        n_workers: int = 1,
        reg_weight: float = None,
        a_prevs: np.ndarray = None,
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        """Project a batch of trajectories in parallel.

        Args:
            X_refs: (batch, dof)
            s0s: (batch, state_dim)
            n_workers: number of parallel IPOPT solves.
            reg_weight: optional regularization weight (passed to each solve).
            a_prevs: (batch, action_dim) last executed actions for action rate constraint.

        Returns:
            X_projs: (batch, dof)
            statuses: list of per-solve statuses ("optimal"/"failed").
            iters: list of per-solve IPOPT iteration counts.
        """
        batch_size = X_refs.shape[0]
        assert X_refs.shape == (batch_size, self.dof)
        assert s0s.shape == (batch_size, self.state_dim)

        if n_workers <= 1:
            projs = []
            statuses = []
            iters = []
            for i in range(batch_size):
                ap = a_prevs[i] if a_prevs is not None else None
                x_proj, status, n_iters = self.project(X_refs[i], s0s[i], reg_weight=reg_weight, a_prev=ap)
                projs.append(x_proj)
                statuses.append(status)
                iters.append(n_iters)
            return np.stack(projs), statuses, iters

        # parallel: build config dict for picklable worker function
        config = {
            "circular_obstacles": self.circular_obstacles,
            "planar_obstacles": self.planar_obstacles,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "normalizer": self.normalizer,
            "dynamics": self.dynamics,
            "dynamics_relaxation": self.dynamics_relaxation,
            "obstacle_margin": self.obstacle_margin,
            "input_saturation": self.input_saturation,
            "objective_fn": self.objective_fn,
            "objective_scale": self.objective_scale,
            "max_action_delta": self.max_action_delta,
            "solver_opts": self._solver_opts,
        }

        results = [None] * batch_size
        statuses = [None] * batch_size
        iters = [0] * batch_size
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _build_and_solve, config, X_refs[i], s0s[i], reg_weight,
                    a_prevs[i] if a_prevs is not None else None,
                ): i
                for i in range(batch_size)
            }
            for future in as_completed(futures):
                idx = futures[future]
                x_proj, status, n_iters = future.result()
                results[idx] = x_proj
                statuses[idx] = status
                iters[idx] = n_iters

        return np.stack(results), statuses, iters

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def dof_to_full_trajectory(self, X_dof: np.ndarray, s0: np.ndarray) -> np.ndarray:
        """Reconstruct the full (horizon, transition_dim) trajectory from DOF + s0.

        Returns:
            traj: (horizon, action_dim + state_dim)
        """
        X_dof = np.asarray(X_dof).flatten()
        s0 = np.asarray(s0).flatten()
        ad = self.action_dim
        sd = self.state_dim

        traj = np.zeros((self.horizon, ad + sd))

        # timestep 0: action from DOF, state = s0
        a0_s, a0_e = self._action_indices[0]
        traj[0, :ad] = X_dof[a0_s:a0_e]
        traj[0, ad:] = s0

        # timesteps 1..H-1: action + state from DOF
        for t in range(1, self.horizon):
            a_s, a_e = self._action_indices[t]
            s_s, s_e = self._state_indices[t - 1]
            traj[t, :ad] = X_dof[a_s:a_e]
            traj[t, ad:] = X_dof[s_s:s_e]

        return traj

    def full_trajectory_to_dof(self, traj: np.ndarray) -> np.ndarray:
        """Extract the DOF vector from a full (horizon, transition_dim) trajectory.

        s_0 is dropped (it's a parameter, not a DOF).
        """
        ad = self.action_dim
        sd = self.state_dim
        X_dof = np.zeros(self.dof)

        # a_0
        a0_s, a0_e = self._action_indices[0]
        X_dof[a0_s:a0_e] = traj[0, :ad]

        # a_t, s_t for t >= 1
        for t in range(1, self.horizon):
            a_s, a_e = self._action_indices[t]
            s_s, s_e = self._state_indices[t - 1]
            X_dof[a_s:a_e] = traj[t, :ad]
            X_dof[s_s:s_e] = traj[t, ad:]

        return X_dof
