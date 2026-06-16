import os
from os.path import join, dirname, abspath
import sys
import logging
import click
import hydra
import datetime
import numpy as np
from tqdm import tqdm
from itertools import product
import importlib.util

from diffusion_policy.coupling_cost.cost_functions import cost_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = str(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = str(dirname(dirname(abspath(__file__))))
LOG_DIR = str(join(dirname(dirname(abspath(__file__))), "logs/tests/event_logs"))
GUIDERS = ['vanilla', 'coupling', 'coupling_ps']
PROJECTORS = ['none', 'max_vel_cvxpy', 'max_vel_admm']
COST_FUNCS = list(cost_registry.keys())

PROJECTOR_PARAMS = {
    'max_vel_cvxpy': {
        '_target_': 'diffusion_policy.model.projection.cvxpy_projector_multiprocess.MaxVelocityCvxPyProjector',
        'horizon': "${eval:'${horizon} - ${n_obs_steps} + ${policy.oa_step_convention}'}",
        'ndim': "${action_dim}" ,
        'proj_profile_key': "${proj_profile_key}",
        'parallel': True,
        'n_jobs': 50,
        'solver': 'ECOS',
        'abstol': 1.0e-6,
        'reltol': 1.0e-5,
        'feastol': 1.0e-5,
        'verbose': False,
        'warm_start': True
    },
    'max_vel_admm': {
        '_target_': 'diffusion_policy.model.projection.projection_operator.ADMMProjectionOperatorCUDAGraph',
        'rho': 6.0,
        'max_iter': 250,
        'tol': 2e-4,
        'decomp': 'lu',
        'convergence_check': True,
        'verbose': False,
    }
}

# Sweep parameters
n_init_states = 50
trial = 200
group_size = 2
max_steps = 360
n_envs = None
n_diffusion_steps = 32
proj_profile_key= "max_vel"
default_cost_fn_key = "dpp"
steps = [1,]
scales = [0.2]
vmaxs = [8.4]
skips = []

def import_module_from_path(module_name: str, file_path: str):
    """ Imports a module dynamically from a full file path. """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{module_name}' at {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logger.info(f"Successfully imported and cached module '{module_name}' from {file_path}")
    return module

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--work-dir", type=str, default=BASE_DIR)
@click.option("--gpu-id", type=int, default=0)
@click.option("--guider", '-g', type=str, required=True)
@click.option("--projector", '-p', type=str, required=True)
@click.option("--cost-func-key", '-c', type=str, default=default_cost_fn_key)
@click.option("--product-params", '-r', is_flag=True)
@click.option("--group-size", "group_size_arg", '-n', type=int, default=None, help="Number of coupled trajectories per group (overrides script default)")
@click.option("--direct", is_flag=True, help="Use DiRecT policy (PGD mode) instead of PCD")
@click.option("--comment", '-m', type=str, default="")
@click.pass_context
def main(
    ctx,
    work_dir: str,
    gpu_id: int,
    guider: str,
    projector: str,
    cost_func_key: str,
    product_params: bool,
    group_size_arg: int,
    direct: bool,
    comment: str = "",
    ):
    global group_size
    if group_size_arg is not None:
        group_size = group_size_arg
    overrides = list(ctx.args)

    assert os.path.exists(work_dir), f"work_dir {work_dir} does not exist"
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        logger.info(f"Created log directory at {LOG_DIR}")
    log_file_nm = f"eval_runs-{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    if comment:
        log_file_nm += f"-{comment}"
    log_file_nm += ".log"
    log_pth = join(LOG_DIR, log_file_nm)
    logger.info(f"Logging all experiment directories to '{log_pth}'.")
    if overrides:
        logger.info(f"Applying ad-hoc overrides to all runs: '{overrides}'")

    ## check guider, projector, cost function
    assert guider in GUIDERS
    assert projector in PROJECTORS
    assert cost_func_key in COST_FUNCS
    assert trial % group_size == 0, \
        f"trial ({trial}) must be divisible by group_size ({group_size})"

    # Set the environment variable BEFORE any other modules (especially
    ## those that import torch/jax/tensorflow) are loaded.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # This must be the RELATIVE path from the project root to the config dir
    config_rel_path = os.path.relpath(
        join(PROJECT_ROOT, 'diffusion_policy', 'config'),
        start=BASE_DIR
    )
    logger.info(f"Initializing Hydra with config path: '{config_rel_path}' relative to root: '{PROJECT_ROOT}'")

    # Initialize Hydra.
    hydra.initialize(config_path=config_rel_path, job_name="seq_eval", version_base=None)

    # This caches their code in memory for the entire run.
    logger.info("Importing sub-evaluation scripts to create a snapshot...")
    guide_script_path = os.path.join(work_dir, "eval_H16_pusht_guided_parallel.py")
    baseline_script_path = os.path.join(work_dir, "eval_H16_pusht_fix_init.py")

    assert os.path.exists(guide_script_path), f"Sub-script not found: {guide_script_path}"
    assert os.path.exists(baseline_script_path), f"Sub-script not found: {baseline_script_path}"

    eval_guide_module = import_module_from_path("eval_guide", guide_script_path)
    eval_baseline_module = import_module_from_path("eval_baseline", baseline_script_path)
    logger.info("Sub-scripts imported. Execution will use these in-memory versions.")

    ## run
    ### Baseline
    if guider == 'vanilla' and projector.lower() == 'none':
        ### Baseline: no coupling, no projection
        config_name = "eval_H16_pusht_fix_init_state.yaml"
        print("".join(['*']*70))
        logger.info(f"GPU {gpu_id}: Running '{guider}' for {trial} trials...")

        args = [
            f'n_init_states={n_init_states}',
            f'trial={trial}',
            f'task.env_runner.max_steps={max_steps}',
            f'task.env_runner.n_envs={n_envs if n_envs is not None else "null"}',
            f"log_pth='{log_pth}'",
            f'n_diffusion_steps={n_diffusion_steps}',
            f'group_size={group_size}',
        ]

        args.extend(overrides)

        try:
            # Use hydra.compose to load the config and apply overrides
            cfg = hydra.compose(config_name=config_name, overrides=args)
            # Call the raw logic function directly
            eval_baseline_module.run_eval(cfg)
        except Exception as e:
            logger.error(f"Execution of baseline script failed: {e}", exc_info=True)
        logger.info(f"Finished running '{guider}'-'{projector}'.")

    elif guider == 'vanilla' and projector.lower() != 'none':
        ### Projection only, no coupling
        config_name = "eval_H16_pusht_proj_admm.yaml" if projector.lower() == 'max_vel_admm' else "eval_H16_pusht_projection.yaml"

        for v_max in tqdm(vmaxs, desc="Projected Runs"):
            print("".join(['*']*70))
            logger.info(f"GPU {gpu_id}: Running '{projector.split('_')[-1]}' with v_max={v_max} for {trial} trials...")

            args = [
                f'projector={projector}',
                f'proj_profile_key={proj_profile_key}',
                f'v_max={v_max}',
                f'n_init_states={n_init_states}',
                f'trial={trial}',
                f'group_size={group_size}',
                f'task.env_runner.max_steps={max_steps}',
                f'task.env_runner.n_envs={n_envs if n_envs is not None else "null"}',
                f"log_pth='{log_pth}'",
                f'n_diffusion_steps={n_diffusion_steps}',
            ]

            # Append projector params
            for k, v in PROJECTOR_PARAMS[projector].items():
                args.append(f'policy.projector.{k}={v}')

            args.extend(overrides)

            try:
                cfg = hydra.compose(config_name=config_name, overrides=args)
                # Call the raw logic function directly
                eval_baseline_module.run_eval(cfg)
            except Exception as e:
                logger.error(f"Execution of baseline script failed: {e}", exc_info=True)

            logger.info(f"Finished running '{guider}'-'{projector}' with v_max={v_max}.")
        #end for v_max

    elif guider in ['coupling', 'coupling_ps'] and projector.lower() == 'none':
        ### Coupling only, no projection
        config_name = "eval_H16_pusht_cost_coupling.yaml"

        ## check param list
        if product_params:
            params = product(steps, scales)
            params_len = len(steps) * len(scales)
        else:
            assert len(steps) == len(scales)
            params = zip(steps, scales)
            params_len = len(steps)

        for stp, scl in tqdm(params, total=params_len, desc="Coupling Runs"):
            # skip logic remains the same
            param_np = np.array([stp, scl])
            is_skipped = any(np.linalg.norm(param_np - np.array(skip)) < 1e-6 for skip in skips)
            if is_skipped:
                logger.info(f"Skipping step={stp}, scale={scl}")
                continue

            print("".join(['*']*70))
            logger.info(f"GPU {gpu_id}: Running '{cost_func_key}'-'{guider}' with stp={stp}, scl={scl}, for {trial} trials...")

            # Build argument list for this iteration
            args = [
                f'guider={guider}',
                f'cost_func_key={cost_func_key}',
                f'stp={stp}',
                f'scl={scl}',
                f'n_init_states={n_init_states}',
                f'trial={trial}',
                f'group_size={group_size}',
                f'task.env_runner.max_steps={max_steps}',
                f'task.env_runner.n_envs={n_envs if n_envs is not None else "null"}',
                f"log_pth='{log_pth}'",
                f'n_diffusion_steps={n_diffusion_steps}',
            ]

            args.extend(overrides)

            # Call the imported function with the arguments for this iteration
            logger.info(f"Calling eval_guide_module.main with args:\n{args}")
            try:
                cfg = hydra.compose(config_name=config_name, overrides=args)
                # Call the raw logic function directly
                eval_guide_module.run_eval(cfg)
            except Exception as e:
                logger.error(f"Execution of guided script failed for stp={stp}, scl={scl}: {e}", exc_info=True)

            logger.info(f"Finished running '{cost_func_key}'-'{guider}'-'{projector}' with stp={stp}, scl={scl}.")
        #end for stp, scl

    ### Guided
    elif guider in ['coupling', 'coupling_ps'] and projector.lower() != 'none':
        config_name = "eval_H16_pusht_direct.yaml" if direct else "eval_H16_pusht_proj_coup.yaml"

        ## check param list
        if product_params:
            params = product(steps, scales, vmaxs)
            params_len = len(steps) * len(scales) * len(vmaxs)
        else:
            assert len(steps) == len(scales) == len(vmaxs)
            params = zip(steps, scales, vmaxs)
            params_len = len(steps)

        for stp, scl, v_max in tqdm(params, total=params_len, desc="PCDiff Runs"):
            # skip logic remains the same
            param_np = np.array([stp, scl])
            is_skipped = any(np.linalg.norm(param_np - np.array(skip)) < 1e-6 for skip in skips)
            if is_skipped:
                logger.info(f"Skipping step={stp}, scale={scl}")
                continue

            print("".join(['*']*70))
            logger.info(f"GPU {gpu_id}: Running '{projector.split('_')[-1]}'-'{cost_func_key}'-'{guider}' with v_max={v_max}, stp={stp}, scl={scl}, for {trial} trials...")

            # Build argument list for this iteration
            args = [
                f'guider={guider}',
                f'projector={projector}',
                f'proj_profile_key={proj_profile_key}',
                f'v_max={v_max}',
                f'cost_func_key={cost_func_key}',
                f'stp={stp}',
                f'scl={scl}',
                f'n_init_states={n_init_states}',
                f'trial={trial}',
                f'group_size={group_size}',
                f'task.env_runner.max_steps={max_steps}',
                f'task.env_runner.n_envs={n_envs if n_envs is not None else "null"}',
                f"log_pth='{log_pth}'",
                f'n_diffusion_steps={n_diffusion_steps}',
            ]

            # Append projector params
            for k, v in PROJECTOR_PARAMS[projector].items():
                args.append(f'policy.projector.{k}={v}')

            args.extend(overrides)

            # Call the imported function with the arguments for this iteration
            logger.info(f"Calling eval_guide_module.main with args:\n{args}")
            try:
                cfg = hydra.compose(config_name=config_name, overrides=args)
                # Call the raw logic function directly
                eval_guide_module.run_eval(cfg)
            except Exception as e:
                logger.error(f"Execution of guided script failed for stp={stp}, scl={scl}: {e}", exc_info=True)

            logger.info(f"Finished running '{guider}'-'{projector}' with v_max={v_max}, stp={stp}, scl={scl}.")
    else:
        raise ValueError(f"Unknown guider type '{guider}'; guider must be one of {GUIDERS}")

if __name__ == '__main__':
    main()
