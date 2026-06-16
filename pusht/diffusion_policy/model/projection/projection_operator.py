import torch
import numpy as np
import logging
from itertools import repeat
from typing import Optional, Union, Callable, Dict, List, Tuple

from diffusion_policy.model.common.tensor_util import to_numpy, to_torch
from diffusion_policy.constraints.velocity_constraints import vel_constraint_registry
import time

logger = logging.getLogger(__name__)

class BaseProjectionOperator:
    """Base class for projection operators."""
    def __init__(self):
        pass

    def project(self, 
        x: torch.Tensor, 
        params: Optional[Dict[str, torch.Tensor]], 
    ) -> torch.Tensor:
        """Projects a tensor x onto a constraint set."""
        raise NotImplementedError("The project method must be implemented by subclasses.")



class ADMMProjectionOperator(BaseProjectionOperator):
    """
    ADMM Projection Operator for convex constraints.
    
    Args:
        constraint_fn: A callable that defines the convex constraint.
        rho: Penalty parameter for ADMM.
        max_iter: Maximum number of iterations for ADMM.
        tol: Tolerance for convergence.
    """
    def __init__(self, 
                rho: float = 1.0, 
                max_iter: int = 100, 
                tol: float = 1e-4, 
                convergence_check: bool = True,
                verbose: bool = False
        ):
        super().__init__()
        self.rho = rho
        self.max_iter = max_iter
        assert max_iter > 1, "max_iter must be greater than 1."
        self.tol = tol
        self.tol_sq = tol**2  # Pre-compute the squared tolerance
        logger.info(
            f"ADMM projector initialized with" 
            f"rho={rho}, max_iter={max_iter}, tol={tol}"
        )
        # Cached LU factorization info
        self._cached_n = None
        self._cached_dtype = None
        self._cached_lu = None
        self._cached_pivots = None

        self.cvg_chk = convergence_check
        self.verbose = verbose

    def _ensure_factorization(self, n: int, dtype, device) -> bool:
        """
        Ensure LU factorization of A = I + rho*D^T D is cached for given n, dtype, device.
        Recompute if n or dtype changed.
        Also cache the difference matrix D for dtu calculations.
        """
        if self._cached_n == n and self._cached_dtype == dtype and self._cached_lu is not None:
            return True

        # Build and cache (n-1 x n) difference operator D
        D = torch.zeros(n - 1, n, device=device, dtype=dtype)
        idx = torch.arange(n - 1, device=device)
        D[idx, idx]     = -1.0
        D[idx, idx + 1] = +1.0
        self._cached_D_T_expand = D.T.unsqueeze(0)  # (1, n, n-1)

        # Assemble system matrix A = I + rho * D^T D
        I_n = torch.eye(n, device=device, dtype=dtype)
        A = I_n + self.rho * (D.T @ D)

        # Compute and cache LU factorization
        lu, pivots = torch.linalg.lu_factor(A)
        self._cached_lu = lu
        self._cached_pivots = pivots
        self._cached_n = n
        self._cached_dtype = dtype

        return False

    def project(self,
                X: torch.Tensor,
                params: dict,
                timeit: bool = False
    ) -> torch.Tensor:
        """
        Batched ADMM projection for X ∈ R^{B x H x D}, enforcing
        ||x_{i+1} - x_i||_2 ≤ dx_max. Optionally includes an initial
        position `pos_init` before X[:,0,:].
        """
        assert 'dx_max' in params, "Parameter 'dx_max' must be provided."
        alpha = params['dx_max']
        x0 = params.get('pos_init', None)

        B, H, d = X.shape
        device = X.device
        dtype = X.dtype

        # Timing placeholders
        if timeit:
            t_init = t_linalg = t_loop = t_x_update = t_z_update = t_u_update = t_conv_chk = 0.0
            t_init_start = time.perf_counter()

        with torch.no_grad():
            # Prepare augmented trajectory
            x = X.detach().clone()
            n = H
            if x0 is not None:
                assert x0.shape == (B, d), "pos_init must be (B, D)"
                x = torch.cat([x0.detach().clone().to(device).unsqueeze(1), x], dim=1)
                n += 1
            X_aug = x.detach().clone()

            # Ensure factorization and D are cached
            self._ensure_factorization(n, dtype, device)
            lu, pivots = self._cached_lu, self._cached_pivots
            D = self._cached_D_T_expand

            # Initialize ADMM variables
            z = torch.zeros((B, n - 1, d), device=device, dtype=dtype)
            u = torch.zeros((B, n - 1, d), device=device, dtype=dtype)

            if timeit:
                t_init = time.perf_counter() - t_init_start
                t_loop_start = time.perf_counter()

            # ADMM iterations
            for _ in repeat(None, self.max_iter):
                if timeit:
                    t_x_update_start = time.perf_counter()

                # x-update: compute b = X_aug + rho * D^T (z - u)
                b = X_aug + self.rho * D @ (z - u)              # (B, n, d)

                if timeit:
                    toc = time.perf_counter()
                    t_linalg_start = toc
                    t_x_update += toc - t_x_update_start

                # Solve A x = b via cached LU
                x_new = torch.lu_solve(b, lu, pivots)
                
                if timeit:
                    toc = time.perf_counter()
                    t_linalg += toc - t_linalg_start
                    t_z_update_start = toc

                # z-update: project onto L2 balls
                diff = x_new[:, 1:] - x_new[:, :-1] + u
                norms = diff.norm(dim=2, keepdim=True)
                scale = torch.clamp(alpha / (norms + 1e-10), max=1.0)
                z_new = diff * scale

                if timeit:
                    toc = time.perf_counter()
                    t_z_update += toc - t_z_update_start
                    t_u_update_start = toc

                # u-update
                u += (x_new[:, 1:] - x_new[:, :-1]) - z_new

                if timeit:
                    toc = time.perf_counter()
                    t_u_update += toc - t_u_update_start
                    t_conv_chk_start = toc

                # --- OPTIMIZED CONVERGENCE CHECK ---
                # Check convergence using squared norms to avoid sqrt
                if self.cvg_chk:
                    r_norm_sq = (x_new[:,1:] - x_new[:,:-1] - z_new).pow(2).sum(dim=(1, 2))
                    # Note: z changes in this loop, so we compute the dual residual
                    # with the new and old z.
                    s_norm_sq = self.rho * (z_new - z).pow(2).sum(dim=(1, 2))
                    converged = torch.all(r_norm_sq < self.tol_sq) and torch.all(s_norm_sq < self.tol_sq) 

                if timeit:
                    t_conv_chk += time.perf_counter() - t_conv_chk_start

                if self.cvg_chk and converged:
                    break
                
                z = z_new
            #end for 

            if timeit:
                if X.is_cuda: torch.cuda.synchronize()
                t_loop = time.perf_counter() - t_loop_start


            # Drop initial state if added
            result = x_new[:, 1:] if x0 is not None else x_new
            assert result.shape == X.shape

        if timeit:
            t_total = time.perf_counter() - t_init_start
            return result, {
                'init': np.array([t_init]),
                'linalg': np.array([t_linalg]),
                'x_update': np.array([t_x_update]),
                'z_update': np.array([t_z_update]),
                'u_update': np.array([t_u_update]),
                'conv_chk': np.array([t_conv_chk]),
                'loop': np.array([t_loop]),
                'total': np.array([t_total])
            }
        return result



class ADMMProjectionOperatorCUDAGraph(ADMMProjectionOperator):
    """
    ADMM projection with CUDA-graph acceleration and anchor constraint.
    """
    def __init__(self, rho: float = 1.0, max_iter: int = 100, tol: float = 1e-4, 
                decomp: str = 'cholesky',
                convergence_check: bool = True, verbose: bool = False):
        super().__init__(rho=rho, max_iter=max_iter, tol=tol, convergence_check=convergence_check, verbose=verbose)
        
        assert decomp in ['cholesky', 'lu'], "`decomp` only supports 'cholesky' or 'lu' for now."
        self.decomp:str = decomp.lower()
        if decomp == 'lu':
            self._solve_primal = self._solve_primal_lu
        elif decomp == 'cholesky':
            self._solve_primal = self._solve_primal_cholesky
        else:
            raise ValueError(f"Unknown decomp: {decomp}")
        
        self._graph_cached_D_T_expand = None
        self._graph_cached_lu = None
        self._graph_cached_pivots = None
        self._graph_cached_cholesky = None
        self._graph_cached_cholesky_T = None
        self._cached_shape = None
        self.graph:torch.cuda.CUDAGraph = None
        self.stream: torch.cuda.Stream = None
        self.static_buffers_ready:bool = False
        self.static_x_penultimate = None
        self.static_z_penultimate = None
        self.static_b_anchor = None

    def _build_difference_operator(self, n: int, dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Create main diagonal (all 1s)
        D = torch.diag(torch.ones(n, dtype=dtype, device=device))
        # Create sub-diagonal (all -1s, one position below main diagonal)
        D += torch.diag(-1.0 * torch.ones(n-1, dtype=dtype, device=device), diagonal=-1)
        D_T_expand = D.t().unsqueeze(0)
        return D, D_T_expand
    
    def _ensure_factorization(self, n, dtype, device):
        matched = (self._cached_n == n and
            self._cached_dtype == dtype and
            self._graph_cached_D_T_expand is not None)
        
        if not matched:
            D, D_T_expand = self._build_difference_operator(n, dtype, device)
            self._graph_cached_D_T_expand = D_T_expand.contiguous()
            

            # Assemble system matrix A = I + rho * D^T D
            I_n = torch.eye(n, device=device, dtype=dtype)
            A = I_n + self.rho * (D.T @ D)
            if self.decomp == 'lu':
                # Compute and cache LU factorization
                lu, pivots = torch.linalg.lu_factor(A)
                self._graph_cached_lu = lu.contiguous()
                self._graph_cached_pivots = pivots.contiguous()
            elif self.decomp == 'cholesky':
                # Use banded Cholesky: pack banded matrix or dense
                self._graph_cached_cholesky = torch.linalg.cholesky(A).contiguous()
                self._graph_cached_cholesky_T = self._graph_cached_cholesky.transpose(-2, -1).contiguous()
            else:
                raise ValueError(f"Unknown decomp: {self.decomp}")

            self._cached_n = n
            self._cached_dtype = dtype


    def _capture_graph(self, B: int, n: int, d: int, alpha: float, device: Union[torch.device, str]):
        # Ensure max_iter is sufficient for penultimate capture
        if self.max_iter < 2:
            raise ValueError("max_iter must be at least 2 to capture penultimate iteration")

        dtype = self._cached_dtype
        # Static buffers
        self.static_X = torch.empty((B, n, d), device=device, dtype=dtype)
        self.static_z = torch.empty((B, n, d), device=device, dtype=dtype)
        self.static_u = torch.empty_like(self.static_z)
        self.static_b_anchor = torch.empty_like(self.static_z)
        self.static_rhs = torch.empty((B, n, d), device=device, dtype=dtype)
        self.static_x = torch.empty((B, n, d), device=device, dtype=dtype)
        self.static_x_penultimate = torch.empty_like(self.static_x)
        self.static_z_penultimate = torch.empty_like(self.static_z)

        self.static_diff_z = torch.empty_like(self.static_z)
        self.static_dtu = torch.empty((B, n, d), device=device, dtype=dtype)
        self.static_rho_dtu = torch.empty_like(self.static_dtu)
        self.static_Dx = torch.empty_like(self.static_z)
        self.static_norm = torch.empty((B, n, 1), device=device, dtype=dtype)
        self.static_scale = torch.empty_like(self.static_norm)
        self.static_temp_z = torch.empty_like(self.static_z)
        self.static_temp_upd = torch.empty_like(self.static_z)
        # Allocate static y buffer for Cholesky solve
        self.static_y = torch.empty_like(self.static_x)

        self.static_alpha = torch.tensor([alpha], device=device, dtype=dtype)
        self.static_rho = torch.tensor([self.rho], device=device, dtype=dtype)
        self.static_eps = torch.tensor([1e-10], device=device, dtype=dtype)

        self.graph = torch.cuda.CUDAGraph()
        self.stream = torch.cuda.Stream(device=device)
        torch.cuda.synchronize()

        # Warmup
        with torch.cuda.stream(self.stream), torch.no_grad():
            for _ in range(3):
                self._admm_iteration_static()

            # Begin CUDA graph capture
            self.graph.capture_begin()
            # Capture main iterations up to penultimate
            for _ in range(self.max_iter - 2):
                self._admm_iteration_static()
            # Penultimate iteration and snapshot
            self._admm_iteration_static()
            self.static_x_penultimate.copy_(self.static_x)
            self.static_z_penultimate.copy_(self.static_z)
            # Final iteration
            self._admm_iteration_static()
            self.graph.capture_end()

        torch.cuda.synchronize()
        self.static_buffers_ready = True
        self._cached_shape = (B, n, d)

    def _solve_primal_lu(self):
        torch.linalg.lu_solve(
            self._graph_cached_lu,
            self._graph_cached_pivots,
            self.static_rhs,
            out=self.static_x
        )

    def _solve_primal_cholesky(self):
        # Solve L L^T x = rhs, L is lower triangular 
        ## Solve L y = rhs
        torch.linalg.solve_triangular(
            self._graph_cached_cholesky,
            self.static_rhs,
            left=True,
            upper=False,
            out=self.static_y
        )
        ## Solve L^T x = y
        torch.linalg.solve_triangular(
            self._graph_cached_cholesky_T,
            self.static_y,
            left=True,
            upper=True,
            out=self.static_x
        )

    def _admm_iteration_static(self):
        # x-update with correct sign: z + b_anchor - u
        torch.add(self.static_z, self.static_b_anchor, out=self.static_diff_z)
        torch.sub(self.static_diff_z, self.static_u, out=self.static_diff_z)
        torch.matmul(self._graph_cached_D_T_expand, self.static_diff_z, out=self.static_dtu)
        torch.mul(self.static_dtu, self.static_rho, out=self.static_rho_dtu)
        torch.add(self.static_X, self.static_rho_dtu, out=self.static_rhs)
        self._solve_primal()

        # z-update (proximal Euclidean-ball projection)
        self.static_Dx[:, 0, :] = self.static_x[:, 0, :]
        torch.sub(self.static_x[:, 1:], self.static_x[:, :-1], out=self.static_Dx[:, 1:])
        torch.sub(self.static_Dx, self.static_b_anchor, out=self.static_temp_z)
        torch.add(self.static_temp_z, self.static_u, out=self.static_temp_z)
        torch.norm(self.static_temp_z, dim=2, keepdim=True, out=self.static_norm)
        torch.add(self.static_norm, self.static_eps, out=self.static_scale)
        self.static_scale.reciprocal_()
        torch.mul(self.static_scale, self.static_alpha, out=self.static_scale)
        torch.clamp_max_(self.static_scale, 1.0)
        torch.mul(self.static_temp_z, self.static_scale, out=self.static_z)

        # u-update
        torch.sub(self.static_Dx, self.static_z, out=self.static_temp_upd)
        torch.sub(self.static_temp_upd, self.static_b_anchor, out=self.static_temp_upd)
        torch.add(self.static_u, self.static_temp_upd, out=self.static_u)

    def project(self, X: torch.Tensor, params: dict, timeit: bool = False) -> torch.Tensor:
        alpha:float = params['dx_max']
        x0:torch.Tensor = params['pos_init']
        B, n, d = X.shape
        device, dtype = X.device, X.dtype

        self._ensure_factorization(n, dtype, device)
        need_new = (
            not self.static_buffers_ready or
            self._cached_shape != (B, n, d) or
            self._cached_dtype != dtype
        )
        if need_new:
            self._capture_graph(B, n, d, alpha, device)
            self._cached_dtype = dtype

        cur_stream = torch.cuda.current_stream(device)
        self.stream.wait_stream(cur_stream)
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.static_X.copy_(X)
            self.static_alpha.fill_(alpha)
            self.static_z.zero_()
            self.static_u.zero_()
            self.static_b_anchor.zero_()
            self.static_b_anchor[:, 0, :] = x0.to(dtype=dtype, device=device)
            self.graph.replay()
            out = self.static_x.detach().clone()

        cur_stream.wait_stream(self.stream)

        # --------------------- optional convergence log -------------------------
        if self.cvg_chk:
            assert self.max_iter >= 2, "max_iter must be at least 2 for convergence check."
            with torch.no_grad():
                diff_x   = self.static_x - self.static_x_penultimate
                prim_res = self.static_Dx - self.static_z - self.static_b_anchor
                diff_z   = self.static_z - self.static_z_penultimate
                dual_res = self.rho * torch.matmul(
                    self._graph_cached_D_T_expand, diff_z)
                max_xchg = torch.sqrt(diff_x.pow(2).sum((-1, -2)).max()).item()
                max_pr   = torch.sqrt(prim_res.pow(2).sum((-1, -2)).max()).item()
                max_dr   = torch.sqrt(dual_res.pow(2).sum((-1, -2)).max()).item()
                max_zn   = self.static_z.norm(dim=2).max().item()
                if self.verbose:
                    logger.info(f"[ADMM Convergence Check]:\n" 
                                f"\tΔx       {max_xchg:.3e} \n"
                                f"\tprimal:  {max_pr:.3e} \n"
                                f"\tdual:    {max_dr:.3e} \n"
                                f"\t||z||_∞: {max_zn:.3e} (≤ {alpha:3e})")
            
            if (max_xchg > self.tol or 
                max_pr > self.tol or 
                max_dr > self.tol or 
                max_zn >= alpha + self.tol):
                logger.warning(
                    f"ADMM projection did NOT converge within {self.max_iter} iterations:\n"
                    f"\tΔx       {max_xchg:.3e} \n"
                    f"\tprimal:  {max_pr:.3e} \n"
                    f"\tdual:    {max_dr:.3e} \n"
                    f"\t||z||_∞: {max_zn:.3e} (≤ {alpha:3e}) "
                )

        feasible = self.feasibility_check(out, params, eps=1e-5)
        if not feasible.all():
            logger.warning(
                f"ADMM projection operator returned infeasible trajectories: "
                f"{torch.logical_not(feasible).sum().item()} / {feasible.numel()} NOT feasible."
            )

        return out
    

    def feasibility_check(self, 
        x: torch.Tensor, 
        params: Dict[str, Union[torch.Tensor, float]],
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Check feasibility of the trajectory X under the projection operator.
        Returns a tensor indicating whether each trajectory is feasible.
        """
        alpha:float = params['dx_max']
        x0:torch.Tensor = params['pos_init']

        with torch.no_grad():
            ## Check x0 feasibility
            diff_x0 = x[..., 0, :] - x0 
            norm = torch.linalg.norm(diff_x0, dim=-1)
            feasible_x0 = norm <= alpha + eps

        return feasible_x0


class CoupledADMMProjectionOperator:
    """
    ADMM projector that jointly optimizes a pair of trajectories:
        min_{X,Y}  ||X - X̂||² + ||Y - Ŷ||² - λ||X - Y||²
        s.t.  velocity constraints on X and Y independently

    x0 is a fixed boundary condition (NOT a DOF), encoded via b_anchor.
    Uses an n×n difference operator D where:
        Dx[0] = x[0],  Dx[h] = x[h] - x[h-1]  for h>0
    and b_anchor = [x0, 0, ..., 0], so the constraint ||Dx - b_anchor|| <= dx_max
    gives ||x[0] - x0|| <= dx_max and ||x[h] - x[h-1]|| <= dx_max.

    Uses a sum/difference decomposition for the x-update:
        U = X + Y  →  (I + ρDᵀD) U = rhs_U
        V = X - Y  →  ((1-2λ)I + ρDᵀD) V = rhs_V
    Z-update (velocity ball projection) stays per-trajectory in X,Y space.
    """

    def __init__(self, lam: float = 0.0, rho: float = 6.0,
                 max_iter: int = 250, tol: float = 2e-4,
                 convergence_check: bool = True, verbose: bool = False):
        if lam >= 0.5:
            logger.warning(f"lambda={lam} >= 0.5: V-system matrix may be indefinite")
        self.lam = lam
        self.rho = rho
        self.max_iter = max_iter
        self.tol = tol
        self.tol_sq = tol ** 2
        self.cvg_chk = convergence_check
        self.verbose = verbose
        self._cached_n = None
        self._cached_D_T = None
        self._cached_lu_U = None
        self._cached_pivots_U = None
        self._cached_lu_V = None
        self._cached_pivots_V = None

    def _ensure_factorization(self, n: int, dtype, device):
        """Build n×n D and cache LU of A_U and A_V."""
        if self._cached_n == n and self._cached_lu_U is not None:
            return
        D = torch.eye(n, device=device, dtype=dtype)
        D += torch.diag(-torch.ones(n - 1, device=device, dtype=dtype), diagonal=-1)
        self._cached_D_T = D.T.unsqueeze(0).contiguous()
        I_n = torch.eye(n, device=device, dtype=dtype)
        DtD = D.T @ D
        A_U = I_n + self.rho * DtD
        lu_U, piv_U = torch.linalg.lu_factor(A_U)
        self._cached_lu_U = lu_U
        self._cached_pivots_U = piv_U
        A_V = (1 - 2 * self.lam) * I_n + self.rho * DtD
        lu_V, piv_V = torch.linalg.lu_factor(A_V)
        self._cached_lu_V = lu_V
        self._cached_pivots_V = piv_V
        self._cached_n = n

    def project_coupled(self, X_pair: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Jointly project a pair of trajectories with coupling.

        Args:
            X_pair: (2, B, H, D) — stacked trajectory pair (x0 NOT included)
            params: dict with 'dx_max' (float) and 'pos_init' (2, B, D)
        Returns:
            (2, B, H, D) — projected pair (x0 NOT included)
        """
        assert X_pair.shape[0] == 2
        X_hat, Y_hat = X_pair[0], X_pair[1]
        alpha = params['dx_max']
        x0_X, x0_Y = params['pos_init'][0], params['pos_init'][1]
        B, H, d = X_hat.shape
        device, dtype = X_hat.device, X_hat.dtype

        with torch.no_grad():
            self._ensure_factorization(H, dtype, device)
            D_T = self._cached_D_T
            b_X = torch.zeros(B, H, d, device=device, dtype=dtype)
            b_X[:, 0, :] = x0_X
            b_Y = torch.zeros(B, H, d, device=device, dtype=dtype)
            b_Y[:, 0, :] = x0_Y
            z_X = torch.zeros(B, H, d, device=device, dtype=dtype)
            u_X = torch.zeros(B, H, d, device=device, dtype=dtype)
            z_Y = torch.zeros(B, H, d, device=device, dtype=dtype)
            u_Y = torch.zeros(B, H, d, device=device, dtype=dtype)

            for _ in repeat(None, self.max_iter):
                rhs_X = X_hat + self.rho * (D_T @ (z_X + b_X - u_X))
                rhs_Y = Y_hat + self.rho * (D_T @ (z_Y + b_Y - u_Y))
                U_new = torch.linalg.lu_solve(self._cached_lu_U, self._cached_pivots_U, rhs_X + rhs_Y)
                V_new = torch.linalg.lu_solve(self._cached_lu_V, self._cached_pivots_V, rhs_X - rhs_Y)
                x_X_new = (U_new + V_new) / 2
                x_Y_new = (U_new - V_new) / 2

                Dx_X = torch.empty_like(x_X_new)
                Dx_X[:, 0, :] = x_X_new[:, 0, :]
                Dx_X[:, 1:, :] = x_X_new[:, 1:] - x_X_new[:, :-1]
                Dx_Y = torch.empty_like(x_Y_new)
                Dx_Y[:, 0, :] = x_Y_new[:, 0, :]
                Dx_Y[:, 1:, :] = x_Y_new[:, 1:] - x_Y_new[:, :-1]

                temp_X = Dx_X - b_X + u_X
                norms_X = temp_X.norm(dim=2, keepdim=True)
                z_X_new = temp_X * torch.clamp(alpha / (norms_X + 1e-10), max=1.0)
                temp_Y = Dx_Y - b_Y + u_Y
                norms_Y = temp_Y.norm(dim=2, keepdim=True)
                z_Y_new = temp_Y * torch.clamp(alpha / (norms_Y + 1e-10), max=1.0)

                u_X = u_X + (Dx_X - z_X_new - b_X)
                u_Y = u_Y + (Dx_Y - z_Y_new - b_Y)

                if self.cvg_chk:
                    r_sq_X = (Dx_X - z_X_new - b_X).pow(2).sum(dim=(1, 2))
                    s_sq_X = self.rho * (z_X_new - z_X).pow(2).sum(dim=(1, 2))
                    r_sq_Y = (Dx_Y - z_Y_new - b_Y).pow(2).sum(dim=(1, 2))
                    s_sq_Y = self.rho * (z_Y_new - z_Y).pow(2).sum(dim=(1, 2))
                    if (torch.all(r_sq_X < self.tol_sq) and torch.all(s_sq_X < self.tol_sq)
                        and torch.all(r_sq_Y < self.tol_sq) and torch.all(s_sq_Y < self.tol_sq)):
                        break
                z_X = z_X_new
                z_Y = z_Y_new

        return torch.stack([x_X_new, x_Y_new], dim=0)

