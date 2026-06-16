import os
os.environ.setdefault("OMP_NUM_THREADS",  "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS",   "1")

import cvxpy as cp
import numpy as np
import torch
import logging
from typing import Optional, Dict, Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from diffusion_policy.model.common.tensor_util import to_numpy
from diffusion_policy.model.projection.projection_operator import BaseProjectionOperator
from diffusion_policy.constraints.velocity_constraints import vel_constraint_registry

logger = logging.getLogger(__name__)

# --- Worker globals (one per process) ---
_PROBLEM = None
_VARIABLES = None
_PARAMETERS = None
_INPUT_KEY = None
_SOLVER = None
_SOLVER_KWARGS = None


def _init_worker(
    variables_def: Dict[str, Tuple],
    parameters_def: Dict[str, Tuple],
    objective_fn: Callable,
    constraints_fn: Optional[Callable],
    input_key: str,
    solver: str,
    solver_kwargs: Dict,
):
    """
    Initializer for each worker process: builds the CVXPY Problem & Parameters once.
    """

    global _PROBLEM, _PARAMETERS, _INPUT_KEY, _SOLVER, _SOLVER_KWARGS, _VARIABLES
    _INPUT_KEY = input_key
    _SOLVER = solver
    _SOLVER_KWARGS = solver_kwargs

    # Build CVXPY variables and parameters
    vars_ = {name: cp.Variable(shape, name=name)
            for name, shape in variables_def.items()}
    pars = {name: cp.Parameter(shape, name=name)
            for name, shape in parameters_def.items()}
    # Input parameter
    pars[input_key] = cp.Parameter(variables_def[input_key], name=f'in_{input_key}')

    # store variable mapping for warm-start
    _VARIABLES = vars_

    # Objective and constraints
    obj = objective_fn(vars_, pars)
    cons = constraints_fn(vars_, pars) if constraints_fn else []

    _PROBLEM = cp.Problem(cp.Minimize(obj), cons)
    _PARAMETERS = pars


def _solve_worker(x_i: np.ndarray, param_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Called inside each process: updates parameters, solves, returns variable values.
    """
    # update mandatory input
    _PARAMETERS[_INPUT_KEY].value = x_i
    # update other parameters
    for k, v in param_dict.items():
        if k in _PARAMETERS:
            _PARAMETERS[k].value = v

    _VARIABLES[_INPUT_KEY].value = x_i

    _PROBLEM.solve(solver=_SOLVER, **_SOLVER_KWARGS)
    if _PROBLEM.status not in ('optimal', ):
        logger.warning(
            f"CVXPY problem status: {_PROBLEM.status}. \n"
            f"Input: {_INPUT_KEY}={x_i}, Params: {param_dict}"
        )

    return {var.name(): var.value for var in _PROBLEM.variables()}


class CvxPyProjector(BaseProjectionOperator):
    """
    CVXPY-based projector with persistent process pool for true parallelism.
    """
    _pool: ProcessPoolExecutor = None

    def __init__(
        self,
        variables: Dict[str, Tuple],
        parameters: Optional[Dict[str, Tuple]] = None,
        objective: Callable[..., cp.Expression] = None,
        constraints: Optional[Callable[..., List[cp.Expression]]] = None, 
        parallel: bool = True,
        n_jobs: int = -1,
        solver: str = 'ECOS',
        verbose: bool = False,
        **solver_kwargs
    ):
        super().__init__()
        if not variables or not isinstance(variables, dict):
            raise ValueError("'variables' must be a non-empty dict")
        if objective is None:
            raise ValueError("'objective' must be provided")

        self._variables_def = variables
        self._parameters_def = parameters or {}
        self.objective_fn = objective
        self.constraints_fn = constraints
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.solver_kwargs['verbose'] = verbose
        self._input_key = next(iter(self._variables_def))

        # Cache for sequential fallback
        self._sequential_bundle = None

        self.parallel = parallel
        if parallel: 
            cores = os.cpu_count() or 1
            self.n_jobs = cores if n_jobs == -1 else min(n_jobs, cores)
            logger.info(f"Using {self.n_jobs} workers for parallel projection.") 
        else:
            self.n_jobs = 1
            self._sequential_inited = False
            logger.info(f"Running in sequential mode, no parallelization.") 


    def _build_single_problem_bundle(self):
        vars_ = {n: cp.Variable(s, name=n) for n, s in self._variables_def.items()}
        pars = {n: cp.Parameter(s, name=n) for n, s in self._parameters_def.items()}
        pars[self._input_key] = cp.Parameter(
            self._variables_def[self._input_key],
            name=f'in_{self._input_key}'
        )
        obj = self.objective_fn(vars_, pars)
        cons = self.constraints_fn(vars_, pars) if self.constraints_fn else []
        prob = cp.Problem(cp.Minimize(obj), cons)
        return prob, pars

    def project(
        self,
        x: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        is_batched = x.ndim > 1
        batch_size = x.shape[0] if is_batched else 1

        x_np = x.detach().cpu().numpy()
        if not is_batched:
            x_np = x_np[np.newaxis]

        params_np = to_numpy(params) if params else {}
        param_list = [
            {k: (v[i] if v.ndim and v.shape[0] == batch_size else v)
             for k, v in params_np.items()}
            for i in range(batch_size)
        ]

        # Parallel path
        if self.parallel and batch_size > 1:

            # init pool once
            if self._pool is None:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.n_jobs,
                    initializer=_init_worker,
                    initargs=(
                        self._variables_def,
                        self._parameters_def,
                        self.objective_fn,
                        self.constraints_fn,
                        self._input_key,
                        self.solver,
                        self.solver_kwargs,
                    )
                )

            # submit jobs
            futures = [
                self._pool.submit(_solve_worker, xi, pi)
                for xi, pi in zip(x_np, param_list)
            ]
            results = [f.result() for f in futures]

        # Sequential fallback
        else:
            if not self._sequential_inited:
                # initialize solver state in main process
                _init_worker(
                    self._variables_def,
                    self._parameters_def,
                    self.objective_fn,
                    self.constraints_fn,
                    self._input_key,
                    self.solver,
                    self.solver_kwargs
                )
                self._sequential_inited = True

            results = [
                _solve_worker(xi, pi)
                for xi, pi in zip(x_np, param_list)
            ]

        out = np.stack([r[self._input_key] for r in results])
        if not is_batched:
            out = out.squeeze(0)
        return torch.from_numpy(out).to(dtype=dtype, device=device)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __del__(self):
        # best‐effort cleanup
        self.close()


class MaxVelocityCvxPyProjector(CvxPyProjector):
    """A specialized projector for enforcing maximum velocity constraints.

    This projector is designed to project trajectories onto a set defined by
    maximum velocity constraints, ensuring that the change in position between
    consecutive time steps does not exceed a specified limit.
    """
    def __init__(self, horizon: int, ndim: int, proj_profile_key:str, **kwargs):
        """Initializes the MaxVelocityCvxPyProjector.

        Args:
            horizon: The number of time steps in the trajectory.
            ndim: The dimensionality of the position space.
            **kwargs: Additional keyword arguments passed to the base class.
        """
        
        constraint_kwargs = vel_constraint_registry[proj_profile_key](horizon, ndim)
        super().__init__(
            **constraint_kwargs, 
            **kwargs
        )
