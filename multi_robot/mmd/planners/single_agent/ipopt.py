import os
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
import logging
from mmd.models.projection.projectors import BaseProjector, ADMMProjectionOperatorCUDAGraph
from torch_robotics.torch_planning_objectives.fields.distance_fields import (
    DistanceField, CollisionObjectDistanceField, CollisionWorkspaceBoundariesDistanceField
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

import casadi as ca

### IPOPT PROJECTOR ###
class IpoptProjector(BaseProjector):
    """Nonlinear-programming trajectory projector using CasADi/IPOPT.

    Solves the following NLP independently for each (agent, sample) pair:

        min_{X}  (1/2)||X - X_ref||^2  +  smoothness_weight * sum_t ||x_{t+1} - x_t||^2

        s.t.  ||x_{t+1} - x_t||_2  <= dx_max           for all t   (velocity)
              sdf_k(x_t)            >= robot_radius      for all t,k (obstacle avoidance)
              ||x_a(t) - x_b(t)||_2 >= 2*robot_radius   for all pairs (a,b), t  (inter-robot)
              x_0 = pos_init                             (start pinning)
              x_{H-1} = pos_final   (optional)           (goal pinning)
              ws_min + r <= x_t <= ws_max - r            (workspace bounds)

    All quantities are in the caller's coordinate frame (typically normalized
    [-1, 1] space — the caller normalizes before calling ``project()``).

    Uses ``casadi.nlpsol`` with IPOPT and ``casadi.Function.map()`` for
    thread-parallel solves across the batch dimension.
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
        max_iter: int = 100,
        tol: float = 1e-4,
        verbose: bool = False,
        linear_solver: str = 'mumps',
        warm_start: bool = True,
    ):
        """
        Args:
            collision_fields: List of DistanceField objects from task.get_collision_fields().
                Handles CollisionObjectDistanceField, CollisionWorkspaceBoundariesDistanceField,
                and CollisionSelfField (ignored — not relevant for single-body robots).
            robot_radius: Collision radius of the robot body. Raw SDF values do not
                account for the robot volume, so obstacle constraints are enforced as
                sdf(x) >= robot_radius. Inter-robot constraints use 2 * robot_radius
                as the minimum separation distance.
            n_agents: Number of robots. The batch dimension B is split into n_agents
                groups of equal size; inter-robot constraints are imposed between
                corresponding samples across groups at each timestep.
            ws_bounds: Workspace bounds tensor of shape (2, dim), i.e. [[x_min, y_min], [x_max, y_max]].
                If None, extracted from CollisionWorkspaceBoundariesDistanceField if present.
            use_obstacle_constraints: If True, enforce sdf(x) >= robot_radius for
                every waypoint against each obstacle primitive.
            use_inter_robot_constraints: If True, enforce pairwise minimum separation
                ||x_a(t) - x_b(t)||_2 >= 2 * robot_radius between all robot pairs
                at every timestep.
            fix_final_point: If True, pin the last waypoint of each agent to a
                goal position supplied via params['pos_final'] at project() time.
                Enforces x_{H-1} == pos_final as an equality constraint.
            smoothness_weight: Weight for the smoothness regularisation term
                sum_t ||x_{t+1} - x_t||^2 added to the objective. Encourages
                shorter, smoother trajectories. 0.0 disables it.
            max_iter: Maximum number of IPOPT iterations per solve.
            tol: Convergence tolerance for IPOPT.
            verbose: If True, set IPOPT print_level to 5; otherwise 0.
            linear_solver: IPOPT linear solver backend (e.g. 'mumps', 'ma27').
            warm_start: Enable warm-starting between consecutive solves.
        """
        self.ca = ca
        self.robot_radius = robot_radius
        self.n_agents = n_agents
        self.use_obstacle_constraints = use_obstacle_constraints
        self.use_inter_robot_constraints = use_inter_robot_constraints
        self.fix_final_point = fix_final_point
        self.smoothness_weight = smoothness_weight
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.ipopt_opts = {
            'ipopt.max_iter': max_iter,
            'ipopt.tol': tol,
            'ipopt.print_level': 5 if verbose else 0,
            'ipopt.linear_solver': linear_solver,
            'ipopt.warm_start_init_point': 'yes' if warm_start else 'no',
            'print_time': 1 if verbose else 0,
        }

        self.sdf_fns = []  # List of CasADi Functions, one per primitive
        self.ws_min = None
        self.ws_max = None

        for field in collision_fields:
            if isinstance(field, CollisionWorkspaceBoundariesDistanceField):
                # The task may store relaxed bounds (e.g. * 1.08 hack).
                # We only use these as a fallback; prefer explicit ws_bounds.
                self.ws_min = field.ws_min.cpu().numpy()
                self.ws_max = field.ws_max.cpu().numpy()
            elif isinstance(field, CollisionObjectDistanceField):
                sdf_fn_list = self._build_casadi_sdf(field)
                self.sdf_fns.extend(sdf_fn_list)

        # Explicit ws_bounds override anything extracted from the boundary
        # field (which may carry the 1.08x relaxation hack from PlanningTask).
        if ws_bounds is not None:
            self.ws_min = ws_bounds[0].cpu().numpy()
            self.ws_max = ws_bounds[1].cpu().numpy()

        # Minimum separation distance for inter-robot constraints
        self.min_separation = 2.0 * robot_radius

        # Pre-compute inter-robot pair indices: all (a, b) with a < b
        self.agent_pairs = []
        if use_inter_robot_constraints and n_agents > 1:
            for a in range(n_agents):
                for b in range(a + 1, n_agents):
                    self.agent_pairs.append((a, b))

        n_obstacle = len(self.sdf_fns) if use_obstacle_constraints else 0
        n_inter = len(self.agent_pairs)
        logger.info(
            f'IpoptProjector: {n_obstacle} obstacle constraint(s), '
            f'{n_inter} inter-robot pair(s), '
            f'robot_radius={robot_radius}, '
            f'workspace bounds: {self.ws_min is not None}')

    ### VELOCITY CONSTRAINTS ###
    @staticmethod
    def velocity_constraints(
        ca_module,
        wpts: List['ca.MX'],
        pos_init: 'ca.MX',
        dx_max: 'ca.MX',
        H: int,
        dim: int,
    ) -> Tuple[List, List, List]:
        """
        Build CasADi expressions for the velocity (max-displacement) constraints.

        Enforces ||x_{t+1} - x_t||_2 <= dx_max for consecutive waypoints, plus
        ||x_0 - pos_init||_2 <= dx_max for the anchor.

        Args:
            ca_module: The casadi module.
            wpts: List of H column vectors, each (dim, 1).
            pos_init: Initial position column vector, (dim, 1).
            dx_max: Maximum displacement per step (symbolic scalar).
            H: Number of waypoints.
            dim: State dimension.

        Returns:
            (g, lbg, ubg): constraint expressions, lower bounds, upper bounds.
        """
        g = []
        lbg = []
        ubg = []
        dx_max_sq = dx_max ** 2

        # Anchor: x_0 is now pinned via variable bounds (lbx=ubx=pos_init),
        # so we only need the velocity constraint from x_0 to x_1.

        # Consecutive: ||x_{t+1} - x_t||^2 - dx_max^2 <= 0
        for t in range(H - 1):
            diff = wpts[t + 1] - wpts[t]
            g.append(ca_module.dot(diff, diff) - dx_max_sq)
            lbg.append(-float('inf'))
            ubg.append(0.0)

        return g, lbg, ubg

    ### INTER ROBOT COLLISIONS ###
    def inter_robot_constraints(
        self,
        X_agents: List[List['ca.MX']],
        H: int,
    ) -> Tuple[List, List, List]:
        """
        Build CasADi expressions for inter-robot collision avoidance.

        For every pair of agents (a, b) and every timestep t, enforces:
            ||x_a(t) - x_b(t)||^2 >= (2 * robot_radius)^2

        Args:
            X_agents: List of waypoint lists, one per agent.
                X_agents[a][t] is a (dim, 1) column vector.
            H: Number of waypoints.

        Returns:
            (g, lbg, ubg): constraint expressions, lower bounds, upper bounds.
        """
        ca = self.ca
        g = []
        lbg = []
        ubg = []

        min_sep_sq = self.min_separation ** 2

        for (a, b) in self.agent_pairs:
            for t in range(H):
                diff = X_agents[a][t] - X_agents[b][t]
                g.append(ca.dot(diff, diff))
                lbg.append(min_sep_sq)
                ubg.append(float('inf'))

        return g, lbg, ubg

    ### OBSTACLE COLLISIONS ###
    def obstacle_constraints(
        self,
        wpts: List['ca.MX'],
        H: int,
    ) -> Tuple[List, List, List]:
        """
        Build CasADi expressions for obstacle avoidance constraints.

        For every waypoint t and every obstacle primitive, enforces:
            sdf(x_t) >= robot_radius

        Args:
            wpts: List of H column vectors, each (dim, 1).
            H: Number of waypoints.

        Returns:
            (g, lbg, ubg): constraint expressions, lower bounds, upper bounds.
        """
        g = []
        lbg = []
        ubg = []

        for sdf_fn in self.sdf_fns:
            for t in range(H):
                g.append(sdf_fn(wpts[t]))
                lbg.append(self.robot_radius)
                ubg.append(float('inf'))

        return g, lbg, ubg

    ### CASADI CONSTRUCTION OF OBSTACLES ###
    def _build_casadi_sdf(
        self, field: CollisionObjectDistanceField
    ) -> List['ca.Function']:
        """
        Extract obstacle SDFs from a CollisionObjectDistanceField and build
        individual CasADi functions — one per primitive instance.

        Returns:
            List of CasADi Function f(x) -> sdf_value where x has shape (dim,).
            Each function corresponds to a single primitive (one sphere, one box, etc.).
        """
        if field.df_obj_list_fn is None:
            return []

        df_obj_list = field.df_obj_list_fn()
        if len(df_obj_list) == 0:
            return []

        from torch_robotics.environments.grid_map_sdf import GridMapSDF
        grid_sdfs = [o for o in df_obj_list if isinstance(o, GridMapSDF)]
        analytic_objs = [o for o in df_obj_list if not isinstance(o, GridMapSDF)]

        fns = []
        # Grid SDF is already C2-smooth (bspline), keep as single function
        for grid_sdf in grid_sdfs:
            fns.append(self._build_interpolant_from_grid(grid_sdf))
        # Analytic primitives: one function per individual shape instance
        fns.extend(self._build_analytical_sdfs(analytic_objs))

        return fns

    def _build_interpolant_from_grid(self, grid_sdf) -> 'ca.Function':
        """
        Build a CasADi bspline interpolant from a precomputed GridMapSDF.

        The bspline interpolant is C2-smooth, which is better for IPOPT
        convergence than the piecewise-linear surrogate used by the torch grid.
        """
        ca = self.ca

        limits = grid_sdf.limits.cpu()
        cmap_dim = grid_sdf.cmap_dim.cpu()
        sdf_np = grid_sdf.sdf_tensor.cpu().numpy()
        dim = grid_sdf.dim

        grid_axes = []
        for d in range(dim):
            axis = np.linspace(
                float(limits[0, d]), float(limits[1, d]), int(cmap_dim[d])
            )
            grid_axes.append(axis)

        # CasADi interpolant expects data in column-major (Fortran) order
        sdf_flat = sdf_np.ravel(order='F').tolist()

        sdf_interp = ca.interpolant(
            'sdf_obstacle', 'bspline', grid_axes, sdf_flat
        )

        x_sym = ca.MX.sym('x', dim)
        sdf_val = sdf_interp(x_sym)
        return ca.Function('sdf_obstacle', [x_sym], [sdf_val], ['x'], ['sdf'])

    def _build_analytical_sdfs(self, obj_list) -> List['ca.Function']:
        """
        Build individual CasADi SDF functions from a list of ObjectField /
        primitive objects — one function per primitive instance (per sphere,
        per box, etc.).  Each constraint is smooth, which is better for IPOPT
        convergence than a single min(sdf_i) union.
        """
        ca = self.ca
        from torch_robotics.environments.primitives import (
            ObjectField, MultiSphereField, MultiBoxField, MultiRoundedBoxField
        )

        if len(obj_list) == 0:
            return []

        # Infer dim from the first primitive that actually has data,
        # since some fields (e.g. empty MultiSphereField) report dim=0.
        dim = 0
        for obj in obj_list:
            if isinstance(obj, ObjectField):
                for prim in obj.fields:
                    if prim.dim > 0:
                        dim = prim.dim
                        break
            elif obj.dim > 0:
                dim = obj.dim
            if dim > 0:
                break
        if dim == 0:
            return []

        x_sym = ca.MX.sym('x', dim)
        fns = []

        for obj in obj_list:
            if isinstance(obj, ObjectField):
                pos = obj.pos.cpu().numpy()[:dim]
                x_local = x_sym - pos
                for prim in obj.fields:
                    if prim.dim == 0:
                        continue  # skip empty primitives
                    exprs = self._primitive_sdf_exprs(prim, x_local)
                    for k, expr in enumerate(exprs):
                        name = f'sdf_{type(prim).__name__}_{len(fns)}'
                        fns.append(ca.Function(name, [x_sym], [expr], ['x'], ['sdf']))
            else:
                if obj.dim == 0:
                    continue
                exprs = self._primitive_sdf_exprs(obj, x_sym)
                for k, expr in enumerate(exprs):
                    name = f'sdf_{type(obj).__name__}_{len(fns)}'
                    fns.append(ca.Function(name, [x_sym], [expr], ['x'], ['sdf']))

        return fns

    def _primitive_sdf_exprs(self, prim, x_sym) -> List:
        """
        Return a list of CasADi scalar expressions — one per individual shape
        instance in the primitive (one per sphere, one per box, etc.).

        Supports MultiSphereField and MultiBoxField / MultiRoundedBoxField.
        """
        ca = self.ca
        from torch_robotics.environments.primitives import (
            MultiSphereField, MultiBoxField, MultiRoundedBoxField
        )

        # Small epsilon to avoid sqrt(0) gradient singularity in norm
        _eps = 1e-12

        if isinstance(prim, MultiSphereField):
            centers = prim.centers.cpu().numpy()
            radii = prim.radii.cpu().numpy()
            exprs = []
            for i in range(len(centers)):
                diff = x_sym - centers[i]
                d = ca.sqrt(ca.dot(diff, diff) + _eps) - float(radii[i])
                exprs.append(d)
            return exprs

        elif isinstance(prim, (MultiRoundedBoxField, MultiBoxField)):
            centers = prim.centers.cpu().numpy()
            half_sizes = prim.half_sizes.cpu().numpy()
            has_radius = hasattr(prim, 'radius') and prim.radius is not None
            radii = prim.radius.cpu().numpy() if has_radius else None

            dim = x_sym.shape[0]
            exprs = []
            for i in range(len(centers)):
                r = float(radii[i]) if radii is not None else 0.0
                q = ca.fabs(x_sym - centers[i]) - half_sizes[i] + r
                q_pos = ca.fmax(q, 0)
                outer = ca.sqrt(ca.dot(q_pos, q_pos) + _eps)
                q_max = q[0]
                for d_idx in range(1, dim):
                    q_max = ca.fmax(q_max, q[d_idx])
                inner = ca.fmin(q_max, 0)
                d = outer + inner - r
                exprs.append(d)
            return exprs

        else:
            logger.warning(
                f"Unsupported primitive type {type(prim).__name__} for "
                f"CasADi SDF conversion, skipping."
            )
            return []

    ### SOLVER CONSTRUCTION AND CACHING ###
    def _build_solver(self, H: int, dim: int):
        """
        Build the CasADi NLP for one sample group (n_agents coupled
        trajectories) and cache the solver.

        Decision variables:  X_a (H x dim) for each agent a, flattened
            into a single vector of length n_agents * H * dim.
        Parameters (passed at solve time, not optimised):
            p = [pos_init_0, ..., pos_init_{n-1},
                 (pos_final_0, ..., pos_final_{n-1}),   # only if fix_final_point
                 dx_max, X_ref_flat]
            length = n_agents * dim * (1 + fix_final_point) + 1 + n_agents * H * dim.
        """
        ca = self.ca
        n = self.n_agents
        n_vars = n * H * dim
        n_anchor_params = n * dim  # pos_inits
        n_goal_params = n * dim if self.fix_final_point else 0
        n_params = n_anchor_params + n_goal_params + 1 + n_vars

        # Decision variables — flat vector, row-major: [x0_0, x0_1, x1_0, x1_1, ...]
        X_flat = ca.MX.sym('X', n_vars)

        # Extract per-agent waypoint lists: X_agents[a][t] is (dim, 1) column vector
        X_agents = []
        for a in range(n):
            base = a * H * dim
            wpts = []
            for t in range(H):
                wpts.append(X_flat[base + t * dim: base + (t + 1) * dim])
            X_agents.append(wpts)

        # Parameters
        p = ca.MX.sym('p', n_params)
        idx = 0
        pos_inits = []
        for a in range(n):
            pos_inits.append(p[idx:idx + dim])
            idx += dim
        pos_finals = []
        if self.fix_final_point:
            for a in range(n):
                pos_finals.append(p[idx:idx + dim])
                idx += dim
        dx_max_sym = p[idx]; idx += 1
        X_ref = p[idx:]

        # Objective: projection – minimise distance to reference
        f = ca.dot(X_flat - X_ref, X_flat - X_ref)

        # Smoothness regularisation: sum of squared consecutive displacements
        if self.smoothness_weight > 0:
            for a in range(n):
                for t in range(H - 1):
                    diff = X_agents[a][t + 1] - X_agents[a][t]
                    f = f + self.smoothness_weight * ca.dot(diff, diff)

        # Constraints
        g, lbg, ubg = [], [], []

        for a in range(n):
            gv, lv, uv = self.velocity_constraints(
                ca, X_agents[a], pos_inits[a], dx_max_sym, H, dim)
            g.extend(gv); lbg.extend(lv); ubg.extend(uv)

            if self.use_obstacle_constraints and len(self.sdf_fns) > 0:
                go, lo, uo = self.obstacle_constraints(X_agents[a], H)
                g.extend(go); lbg.extend(lo); ubg.extend(uo)

        if self.use_inter_robot_constraints and len(self.agent_pairs) > 0:
            gr, lr, ur = self.inter_robot_constraints(X_agents, H)
            g.extend(gr); lbg.extend(lr); ubg.extend(ur)

        # Goal-pinning is now done via variable bounds (lbx=ubx=pos_final),
        # not equality constraints. This eliminates endpoints as DOFs.

        # Variable bounds (workspace, inset by robot_radius so the disk
        # doesn't overlap the boundary)
        lbx = np.full(n_vars, -np.inf)
        ubx = np.full(n_vars, np.inf)
        if self.ws_min is not None and self.ws_max is not None:
            for a in range(n):
                for t in range(H):
                    for d in range(dim):
                        idx = a * H * dim + t * dim + d
                        lbx[idx] = float(self.ws_min[d]) + self.robot_radius
                        ubx[idx] = float(self.ws_max[d]) - self.robot_radius

        g_cat = ca.vertcat(*g) if g else ca.MX(0, 1)
        nlp = {'x': X_flat, 'f': f, 'g': g_cat, 'p': p}
        solver = ca.nlpsol('ipopt_proj', 'ipopt', nlp, self.ipopt_opts)

        # Cache everything
        self._solver = solver
        self._H = H
        self._dim = dim
        self._n_vars = n_vars
        self._n_params = n_params
        self._lbg = np.array(lbg, dtype=np.float64) if lbg else np.array([])
        self._ubg = np.array(ubg, dtype=np.float64) if ubg else np.array([])
        self._lbx = lbx
        self._ubx = ubx
        self._solver_map = None
        self._cached_batch_groups = None

        logger.info(
            f'IpoptProjector solver built: {n_vars} vars, '
            f'{len(lbg)} constraints, {n} agent(s), H={H}, dim={dim}')

    ### PROJECT ###
    def project(
        self,
        x: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Project trajectories onto the feasible set via IPOPT.

        Args:
            x: Trajectory tensor, shape (B, H, dim).  Agent a owns indices
                [a*group_size : (a+1)*group_size] where group_size = B // n_agents.
            params: Dict with:
                'dx_max' (float): max displacement per timestep.
                'pos_init' (Tensor): shape (B, dim), initial positions.
                'pos_final' (Tensor, optional): shape (B, dim), goal positions.
                    Required when fix_final_point=True.

        Returns:
            Projected trajectories, same shape as x.
        """
        B, H, dim = x.shape
        n = self.n_agents
        group_size = B // n

        assert B % n == 0, (
            f"Batch size {B} must be divisible by n_agents {n}.")

        dx_max = float(params['dx_max'])
        pos_init = params['pos_init']
        assert pos_init.shape == (B, dim)

        pos_final = None
        if self.fix_final_point:
            pos_final = params['pos_final']
            assert pos_final.shape == (B, dim)

        # Build / rebuild solver if dimensions changed
        if not hasattr(self, '_solver') or self._H != H or self._dim != dim:
            self._build_solver(H, dim)

        # Build / rebuild mapped solver if batch size changed
        if (self._solver_map is None
                or self._cached_batch_groups != group_size):
            n_threads = min(group_size, os.cpu_count() or 8)
            # Force MUMPS/OpenMP to single-threaded so CasADi's thread
            # pool can run N solves truly in parallel.
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            self._solver_map = self._solver.map(
                group_size, 'thread', n_threads)
            self._cached_batch_groups = group_size
            logger.info(
                f'IpoptProjector: mapped solver for {group_size} groups, '
                f'{n_threads} threads')

        # To numpy
        x_np = x.detach().cpu().numpy()
        pos_init_np = pos_init.detach().cpu().numpy()
        pos_final_np = pos_final.detach().cpu().numpy() if pos_final is not None else None
        n_vars = self._n_vars

        # Pack x0 (initial guess) and p (parameters) per group.
        # For group g, agent a's trajectory is x_np[a * group_size + g].
        x0_all = np.zeros((n_vars, group_size))
        p_all = np.zeros((self._n_params, group_size))

        for g in range(group_size):
            x0_parts = []
            p_init_parts = []
            p_final_parts = []
            for a in range(n):
                batch_idx = a * group_size + g
                x0_parts.append(x_np[batch_idx].ravel())
                p_init_parts.append(pos_init_np[batch_idx])
                if pos_final_np is not None:
                    p_final_parts.append(pos_final_np[batch_idx])

            x0_flat = np.concatenate(x0_parts)
            p_pieces = p_init_parts + p_final_parts + [[dx_max]] + [x0_flat]
            x0_all[:, g] = x0_flat
            p_all[:, g] = np.concatenate(p_pieces)

        # Tile bounds across groups
        lbx_tiled = np.tile(self._lbx[:, None], (1, group_size))
        ubx_tiled = np.tile(self._ubx[:, None], (1, group_size))

        # Pin start (t=0) as non-DOF: lbx = ubx = pos_init
        for g_idx in range(group_size):
            for a in range(n):
                batch_idx = a * group_size + g_idx
                offset = a * H * dim
                for d in range(dim):
                    idx = offset + d  # t=0
                    lbx_tiled[idx, g_idx] = pos_init_np[batch_idx, d]
                    ubx_tiled[idx, g_idx] = pos_init_np[batch_idx, d]
                # Pin end (t=H-1) if fix_final_point
                if self.fix_final_point and pos_final_np is not None:
                    for d in range(dim):
                        idx = offset + (H - 1) * dim + d
                        lbx_tiled[idx, g_idx] = pos_final_np[batch_idx, d]
                        ubx_tiled[idx, g_idx] = pos_final_np[batch_idx, d]

        if len(self._lbg) > 0:
            lbg_tiled = np.tile(self._lbg[:, None], (1, group_size))
            ubg_tiled = np.tile(self._ubg[:, None], (1, group_size))
        else:
            lbg_tiled = np.zeros((0, group_size))
            ubg_tiled = np.zeros((0, group_size))

        # Solve all groups in parallel
        sol = self._solver_map(
            x0=x0_all,
            lbx=lbx_tiled,
            ubx=ubx_tiled,
            lbg=lbg_tiled,
            ubg=ubg_tiled,
            p=p_all,
        )

        x_sol = np.array(sol['x'])  # (n_vars, group_size)

        # Unpack into (B, H, dim)
        result = np.empty_like(x_np)
        for g in range(group_size):
            for a in range(n):
                batch_idx = a * group_size + g
                offset = a * H * dim
                result[batch_idx] = (
                    x_sol[offset:offset + H * dim, g].reshape(H, dim))

        return torch.tensor(result, dtype=x.dtype, device=x.device)

### END IPOPT PROJECTOR ###
