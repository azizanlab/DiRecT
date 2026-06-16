import os
import sys
import logging
import datetime
import click
import pdb
import numpy as np
import importlib.util

from tqdm import tqdm
from itertools import product
from os.path import join, dirname

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR    = dirname(os.path.abspath(__file__))
PROJECT_ROOT= dirname(dirname(BASE_DIR))
CONFIG_DIR = join(PROJECT_ROOT, "configs")
LOG_DIR     = os.path.join(PROJECT_ROOT, "results/000event_logs")
GUIDERS     = ['vanilla', 'coupling', 'coupling_ps', 'base', 'final']
PROJECTORS  = ['none', 'max_vel_admm', 'ipopt', 'scp_admm']

PROJECTOR_PARAMS = {
    'max_vel_admm': {
        '_target_': 'ADMMProjectionOperatorCUDAGraph',
        'rho': 10.0,
        'max_iter': 1000,
        'tol': 3.0e-6,
        'decomp': 'lu',
        'convergence_check': True,
        'check_period': 100,
        'verbose': False,
    }
}


# default sweep params
# n_init_states     = 20  # Mini
n_init_states     = 100
# num_agents = 2
num_agents = 4
batch_size        = 128
# batch_size       = 4
default_cost_fn_key = "hinge_sqr_l2"
steps  = [1]
#----------------------- Highway -------------------------------
# vmaxs = [0.647, 0.781, 0.878]  # 80%, 90%, 95% of max action
vmaxs = [0.647]


#----------------------- Empty -----------------------------------------------
# vmaxs = [0.675, 0.692, 0.703]  # # 75%, 90%, 95% of max action
# vmaxs = [0.675, ]

#----------------------- Conveyor --------------------------------------------
# vmaxs = [1.21, 1.46, 1.76]  # 85%, 90%, 95% of max action



#----------------------- DropRegion --------------------------------------------
# vmaxs = [0.928, 1.13, 1.34]  # 85%, 90%, 95% of max action


scales = [1e-1, 2e-1,]
skips  = []

# Helpers
def import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore
    logger.info(f"Imported module '{module_name}' from '{file_path}'")
    return module

# CLI
@click.command(context_settings={"ignore_unknown_options": True,
                                 "allow_extra_args": True})
@click.option("--config", type=str, required=True,
              help="Path to the configuration YAML file.")
@click.option("--work-dir",    type=str, default=BASE_DIR)
@click.option("--gpu-id",      type=int, default=0)
@click.option("--guider", "-g",       type=click.Choice(GUIDERS), required=True)
@click.option("--projector","-p",      type=click.Choice(PROJECTORS), required=True)
@click.option("--cost-func-key","-c",  type=str, default=default_cost_fn_key)
@click.option("--product-params","-r", is_flag=True)
@click.option("--comments", "-m",        type=str, default="")
@click.option("--n-inits", "-n",        type=int, default=None,
              help="Number of initial states (overrides the default in the script).")
@click.option("--vmax", "-v",           type=float, multiple=True,
              help="Velocity cap(s) for projection. Repeatable, e.g. -v 0.675 -v 0.703.")
@click.option("--agents", "-a",        type=int, default=None,
              help="Number of agents (overrides the default in the script).")
@click.pass_context
def main(ctx, config, work_dir, gpu_id, guider, projector,
         cost_func_key, product_params, comments, n_inits, vmax, agents):
    global n_init_states, vmaxs, num_agents
    if n_inits is not None:
        n_init_states = n_inits
    if vmax:
        vmaxs = list(vmax)
    if agents is not None:
        num_agents = agents

    # grab any --foo=bar or foo=bar tokens
    raw_overrides = list(ctx.args)

    # sanity
    assert os.path.isdir(LOG_DIR), f"LOG_DIR not found: {LOG_DIR}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    ### Now import GPU-specific modules
    from mmd.common.argparse import load_config, parse_overrides

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_nm = f"eval_runs-{timestamp}" + (f"-{comments}" if comments else "") + ".log"
    log_pth = os.path.join(LOG_DIR, log_nm)
    logger.info(f"Logging all runs to {log_pth}")
    if raw_overrides:
        logger.info(f"Ad-hoc overrides: {raw_overrides}")

    # import sub-scripts
    guide_py    = os.path.join(work_dir, "inference_pcdiff.py")
    base_py     = os.path.join(work_dir, "inference_pcdiff.py")
    eval_guide  = import_module_from_path("eval_guide", guide_py)
    eval_base   = import_module_from_path("eval_guide", base_py)

    cfg_dir = CONFIG_DIR

    # --- 4a) Baseline ---
    if guider=="vanilla" and projector=="none":
        logger.info(f"Running baseline (vanilla/none) for {n_init_states} trials…")

        args_list = [
                f"--system.log_pth={log_pth!r}",
                f"--experiment.num_agents={num_agents}",
                f"--experiment.n_inits={n_init_states}",
                f"--planner.single_agent.planner_alg='vanilla'",
                f"--planner.single_agent.cost_func_key={cost_func_key}",
                f"--planner.single_agent.n_guide_steps={0}",
                f"--planner.single_agent.weight_grad_cost_constraints={0}",
                f"--planner.single_agent.n_samples={batch_size}",
            ] + raw_overrides

        overrides = parse_overrides(args_list)
        cfg = load_config(join(cfg_dir, config), overrides)
        eval_base.run_planning_experiment(cfg, comments=comments)
        logger.info("Baseline done.")

    # --- 4b) Projection-only ---
    elif guider=="vanilla" and projector!="none":
        for v_max in tqdm(vmaxs, desc="Projected runs", ncols=80):
            logger.info(f"Running projector={projector} v_max={v_max} for {n_init_states} trials…")
            args_list = [
                f"--system.log_pth={log_pth!r}",
                f"--experiment.num_agents={num_agents}",
                f"--experiment.n_inits={n_init_states}",
                f"--planner.projection.project_params.vel_max={v_max}",
                f"--planner.single_agent.planner_alg='projected'",
                f"--planner.single_agent.cost_func_key={cost_func_key}",
                f"--planner.single_agent.n_guide_steps={0}",
                f"--planner.single_agent.weight_grad_cost_constraints={0}",
                f"--planner.single_agent.n_samples={batch_size}",
            ]
            # add projector defaults
            for k, v in PROJECTOR_PARAMS[projector].items():
                args_list.append(f"--planner.projection.projector.{k}={v}")

            args_list += raw_overrides
            overrides = parse_overrides(args_list)
            cfg = load_config(join(cfg_dir, config), overrides)
            eval_base.run_planning_experiment(cfg, comments=comments)
            logger.info(f"Finished v_max={v_max}.")

    # --- 4c) Coupling-only ---
    elif guider in ("coupling","coupling_ps") and projector=="none":
        grid = (product(steps, scales) if product_params
                else zip(steps, scales))
        total = (len(steps)*len(scales)) if product_params else len(steps)

        for stp, scl in tqdm(grid, total=total, desc="Coupling runs", ncols=80):
            if any(np.linalg.norm(np.array([stp,scl]) - np.array(skip))<1e-6
                   for skip in skips):
                logger.info(f"Skip stp={stp}, scl={scl}")
                continue

            logger.info(f"Running coupling {cost_func_key} stp={stp}, scl={scl} for {n_init_states} trials…")
            args_list = [
                f"--system.log_pth={log_pth!r}",
                f"--experiment.num_agents={num_agents}",
                f"--experiment.n_inits={n_init_states}",
                f"--planner.single_agent.planner_alg='coupling'",
                f"--planner.single_agent.cost_func_key={cost_func_key}",
                f"--planner.single_agent.n_guide_steps={stp}",
                f"--planner.single_agent.weight_grad_cost_constraints={scl}",
                f"--planner.single_agent.n_samples={batch_size}",
            ] + raw_overrides

            overrides = parse_overrides(args_list)
            cfg = load_config(join(cfg_dir, config), overrides)
            eval_guide.run_planning_experiment(cfg, comments=comments)
            logger.info(f"Done stp={stp}, scl={scl}.")

    # --- 4d) Coupling + Projection ---
    elif guider in ("coupling","coupling_ps") and projector!="none":
        grid = (product(steps, scales, vmaxs) if product_params
                else zip(steps, scales, vmaxs))
        total = (len(steps)*len(scales)*len(vmaxs)
                 if product_params else len(steps))

        for trial, (stp, scl, v_max) in enumerate(tqdm(grid, total=total, desc="PCDiff runs", ncols=80)):
            if any(np.linalg.norm(np.array([stp,scl]) - np.array(skip))<1e-6
                   for skip in skips):
                logger.info(f"Skip stp={stp}, scl={scl}")
                continue

            logger.info(f"Running coupling+proj {cost_func_key} stp={stp}, scl={scl}, v_max={v_max}…")
            args_list = [
                f"--system.log_pth={log_pth!r}",
                f"--experiment.num_agents={num_agents}",
                f"--experiment.n_inits={n_init_states}",
                f"--planner.projection.project_params.vel_max={v_max}",
                f"--planner.single_agent.planner_alg='pcd'",
                f"--planner.single_agent.cost_func_key={cost_func_key}",
                f"--planner.single_agent.n_guide_steps={stp}",
                f"--planner.single_agent.weight_grad_cost_constraints={scl}",
                f"--planner.single_agent.n_samples={batch_size}",
            ]
            for k, v in PROJECTOR_PARAMS[projector].items():
                args_list.append(f"--planner.projection.projector.{k}={v}")
            args_list += raw_overrides

            overrides = parse_overrides(args_list)
            cfg = load_config(join(cfg_dir, config), overrides)
            eval_guide.run_planning_experiment(cfg, comments=comments)
            logger.info(f"Done stp={stp}, scl={scl}, v_max={v_max}.")

    # --- 4e) DiRecT base + projection (ipopt or scp_admm) ---
    elif guider=="base" and projector in ("ipopt", "scp_admm"):
        projector_type_map = {"ipopt": "Ipopt", "scp_admm": "JaxScpADMM"}
        proj_type = projector_type_map[projector]

        for v_max in tqdm(vmaxs, desc=f"DiRecT+{projector} runs", ncols=80):
            logger.info(f"Running DiRecT base + {projector} v_max={v_max} for {n_init_states} trials…")
            args_list = [
                f"--system.log_pth={log_pth!r}",
                f"--experiment.num_agents={num_agents}",
                f"--experiment.n_inits={n_init_states}",
                f"--planner.projection.project_params.vel_max={v_max}",
                f"--planner.single_agent.planner_alg='base'",
                f"--planner.single_agent.start_guide_steps_fraction={0.0}",
                f"--planner.single_agent.n_guide_steps={0}",
                f"--planner.single_agent.weight_grad_cost_constraints={0}",
                f"--planner.single_agent.n_samples={batch_size}",
                f"--planner.single_agent.projector_type={proj_type!r}",
            ] + raw_overrides

            overrides = parse_overrides(args_list)
            cfg = load_config(join(cfg_dir, config), overrides)
            eval_base.run_planning_experiment(cfg, comments=comments)
            logger.info(f"Done v_max={v_max}.")

    # --- 4g) DiRecT final (unguided diffusion + project at last step only) ---
    elif guider == "final" and projector != "none":
        projector_type_map = {
            "ipopt": "Ipopt",
            "scp_admm": "JaxScpADMM",
            "max_vel_admm": "ADMMVelOnly",
        }
        proj_type = projector_type_map[projector]

        for v_max in tqdm(vmaxs, desc=f"DiRecT final+{projector} runs", ncols=80):
            logger.info(f"Running DiRecT final + {projector} v_max={v_max} for {n_init_states} trials…")
            args_list = [
                f"--system.log_pth={log_pth!r}",
                f"--experiment.num_agents={num_agents}",
                f"--experiment.n_inits={n_init_states}",
                f"--planner.projection.project_params.vel_max={v_max}",
                f"--planner.single_agent.planner_alg='final'",
                f"--planner.single_agent.start_guide_steps_fraction={0.0}",
                f"--planner.single_agent.n_guide_steps={0}",
                f"--planner.single_agent.weight_grad_cost_constraints={0}",
                f"--planner.single_agent.n_samples={batch_size}",
                f"--planner.single_agent.projector_type={proj_type!r}",
            ]
            if projector in PROJECTOR_PARAMS:
                for k, v in PROJECTOR_PARAMS[projector].items():
                    args_list.append(f"--planner.projection.projector.{k}={v}")
            args_list += raw_overrides

            overrides = parse_overrides(args_list)
            cfg = load_config(join(cfg_dir, config), overrides)
            eval_base.run_planning_experiment(cfg, comments=comments)
            logger.info(f"Done v_max={v_max}.")

    else:
        raise click.UsageError(f"Unknown combination: guider={guider}, projector={projector}")

if __name__ == "__main__":
    main()
