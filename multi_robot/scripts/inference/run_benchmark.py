#!/usr/bin/env python
"""
Benchmark comparison script for multi-robot diffusion planning.

Runs all method variants across environments, loss functions, velocity
constraints, and agent counts.  Every method call explicitly sets ALL key
parameters so you know exactly what is running.

Usage:
    # Full benchmark
    python scripts/inference/run_benchmark.py --gpu-id 0

    # Quick test: one env, one agent count
    python scripts/inference/run_benchmark.py --gpu-id 0 \
        --envs empty --agents 4 --vmaxs 0.675 --costs hinge_sqr_l2

    # Only specific methods
    python scripts/inference/run_benchmark.py --gpu-id 0 --only pcd direct

    # Dry run (print what would run, don't execute)
    python scripts/inference/run_benchmark.py --gpu-id 0 --dry-run
"""

import os
import sys
import logging
import datetime
import argparse
import importlib.util
from os.path import join, dirname

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = dirname(os.path.abspath(__file__))
PROJECT_ROOT = dirname(dirname(BASE_DIR))
CONFIG_DIR = join(PROJECT_ROOT, "configs")
LOG_DIR = os.path.join(PROJECT_ROOT, "results/000event_logs")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        GLOBAL PARAMETERS                                  ║
# ║            Edit these to control the full benchmark sweep.                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ─── ENVIRONMENTS ───
ENVS = {
    'empty': {
        'pcd_config': 'pcdiff_empty.yaml',
        'direct_config': 'direct_empty.yaml',
        'vel_maxs': [0.675, 0.692, 0.703],       # 75%, 90%, 95% of max action
        'guide_weight_coll': 0.1,                 # weight_grad_cost_collision
        'admm_rho': 10.0,                         # ADMM penalty (from original config)
        'direct_smoothness_weight': 0.0,          # projection smoothness for DiRecT
    },
    'highway': {
        'pcd_config': 'pcdiff_highway.yaml',
        'direct_config': 'direct_highway.yaml',
        'vel_maxs': [0.647, 0.781, 0.878],       # 80%, 90%, 95% of max action
        'guide_weight_coll': 0.1,
        'admm_rho': 6.0,
        'direct_smoothness_weight': 0.0,
    },
    'conveyor': {
        'pcd_config': 'pcdiff_conveyor.yaml',
        'direct_config': 'direct_conveyor.yaml',
        'vel_maxs': [1.21, 1.46, 1.76],          # 85%, 90%, 95% of max action
        'guide_weight_coll': 0.05,
        'admm_rho': 6.0,
        'direct_smoothness_weight': 0.0,
    },
    'dropregion': {
        'pcd_config': 'pcdiff_dropregion.yaml',
        'direct_config': 'direct_dropregion.yaml',
        'vel_maxs': [0.928, 1.13, 1.34],         # 85%, 90%, 95% of max action
        'guide_weight_coll': 0.05,
        'admm_rho': 6.0,
        'direct_smoothness_weight': 0.0,
    },
}

# ─── SWEEP AXES ───
N_AGENTS_LIST       = [4, 8, 12, 16, 20]
COST_FUNC_KEYS      = ['hinge_sqr_l2', 'sum_log_l2']

# ─── EXPERIMENT SETTINGS ───
N_INITS             = 100        # number of random initializations
BATCH_SIZE          = 128        # n_samples per agent
SEED                = 18
TIMEIT              = False

# ─── GUIDANCE DEFAULTS (for PCD / CD) ───
N_GUIDE_STEPS       = 1
GUIDE_WEIGHT_SMOOTH = 0.0       # weight_grad_cost_smoothness
GUIDE_FRACTION      = 1.0       # start_guide_steps_fraction

# ─── PER-COST γ (coupling strength) ───
COST_FUNC_DEFAULTS = {
    'hinge_sqr_l2': {'guide_weight_constr': 0.2},    # SHD
    'sum_log_l2':   {'guide_weight_constr': 0.02},   # LB
}

# ─── PROJECTION START FRACTION (per method) ───
# PCD projects at every step (no t_start_projection param in PCD class)
PROJ_FRACTION_DIRECT  = 0.5     # DiRecT: JaxScpADMM on Tweedie
PROJ_FRACTION_FINAL   = 0.5     # final: ignored (overridden to t=0 only in __init__)

# ─── COST FUNCTION PARAMETERS ───
COST_MARGIN_COEFF   = 1.9
COST_RADIUS_COEFF   = 6.0

# ─── ADMM PROJECTOR DEFAULTS ───
ADMM_RHO            = 6.0
ADMM_MAX_ITER       = 500
ADMM_TOL            = 1e-5
ADMM_DECOMP         = 'lu'
ADMM_CHECK_PERIOD   = 200

# ─── MMD-CBS SETTINGS ───
MMD_RUNTIME_LIMIT   = 60          # CBS runtime limit in seconds
MMD_MULTI_AGENT_PLANNER = "CBS"
MMD_SINGLE_AGENT_PLANNER = "MPD"
MMD_OBSTACLE_MARGIN = 0.2         # margin for random start/goal sampling
MMD_ENV_INSTANCE_MAP = {
    'empty':      'EnvEmpty2DRobotPlanarDiskRandom',
    'highway':    'EnvHighways2DRobotPlanarDiskRandom',
    'conveyor':   'EnvConveyor2DRobotPlanarDiskRandom',
    'dropregion': 'EnvDropRegion2DRobotPlanarDiskRandom',
}

# ─── WHICH METHODS TO RUN ───
ALL_METHODS = [
        'diffusion_policy',
        'cd',
        'pcd',
        'direct',
        'final_projection',
        'mmd_cbs',
]

# ═════════════════════════════════════════════════════════════════════════════


def import_inference_module():
    """Import inference_pcdiff.py as a module."""
    path = os.path.join(BASE_DIR, "inference_pcdiff.py")
    spec = importlib.util.spec_from_file_location("inference_pcdiff", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inference_pcdiff"] = mod
    spec.loader.exec_module(mod)
    return mod


def import_mmd_modules():
    """Import inference_mmd.py and launch_mmd.py as modules."""
    path = os.path.join(BASE_DIR, "inference_mmd.py")
    spec = importlib.util.spec_from_file_location("inference_mmd", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["inference_mmd"] = mod
    spec.loader.exec_module(mod)

    path2 = os.path.join(BASE_DIR, "launch_mmd.py")
    spec2 = importlib.util.spec_from_file_location("launch_mmd", path2)
    mod2 = importlib.util.module_from_spec(spec2)
    sys.modules["launch_mmd"] = mod2
    spec2.loader.exec_module(mod2)
    return mod2


def _run_experiment(config_path, args_list, log_pth, label, dry_run=False):
    """Load config with overrides and run a single experiment."""
    from mmd.common.argparse import load_config, parse_overrides

    logger.info(f"\n{'─'*70}")
    logger.info(f"  {label}")
    logger.info(f"  config: {config_path}")
    logger.info(f"  overrides ({len(args_list)}):")
    for a in args_list:
        logger.info(f"    {a}")
    logger.info(f"{'─'*70}")

    if dry_run:
        logger.info("  [DRY RUN] skipping execution")
        return

    overrides = parse_overrides(args_list)
    cfg = load_config(config_path, overrides)

    inference_mod = import_inference_module()
    inference_mod.run_planning_experiment(cfg, comments=label)


def _admm_projector_args(rho=None):
    """ADMM projector override args."""
    return [
        f"--planner.projection.projector._target_=ADMMProjectionOperatorCUDAGraph",
        f"--planner.projection.projector.rho={rho if rho is not None else ADMM_RHO}",
        f"--planner.projection.projector.max_iter={ADMM_MAX_ITER}",
        f"--planner.projection.projector.tol={ADMM_TOL}",
        f"--planner.projection.projector.decomp={ADMM_DECOMP}",
        f"--planner.projection.projector.convergence_check=True",
        f"--planner.projection.projector.check_period={ADMM_CHECK_PERIOD}",
        f"--planner.projection.projector.verbose=False",
    ]


def _common_args(n_agents, log_pth):
    """Args shared by every method."""
    return [
        f"--system.log_pth={log_pth!r}",
        f"--experiment.num_agents={n_agents}",
        f"--experiment.n_inits={N_INITS}",
        f"--experiment.seed={SEED}",
        f"--planner.single_agent.n_samples={BATCH_SIZE}",
        f"--planner.single_agent.timeit={TIMEIT}",
    ]


# ═══════════════════════════════════════════════════════════════════════════
# METHOD RUNNERS — each one explicitly sets ALL relevant parameters
# ═══════════════════════════════════════════════════════════════════════════

def run_diffusion_policy(env_key, n_agents, log_pth, dry_run=False):
    """Diffusion Policy baseline: vanilla diffusion, no guidance, no projection."""
    config_path = join(CONFIG_DIR, ENVS[env_key]['pcd_config'])
    env = ENVS[env_key]
    label = f"DIFFUSION_POLICY | env={env_key} agents={n_agents}"

    args_list = _common_args(n_agents, log_pth) + [
        # ── Planner identity ──
        f"--planner.single_agent.class=PCD",
        f"--planner.single_agent.planner_alg=vanilla",
        # ── No guidance ──
        f"--planner.single_agent.start_guide_steps_fraction=0.0",
        f"--planner.single_agent.n_guide_steps=0",
        f"--planner.single_agent.weight_grad_cost_constraints=0",
        f"--planner.single_agent.weight_grad_cost_collision=0",
        f"--planner.single_agent.weight_grad_cost_smoothness={GUIDE_WEIGHT_SMOOTH}",
        # ── Cost function (unused but set for consistency) ──
        f"--planner.single_agent.cost_func_key=hinge_sqr_l2",
    ]

    _run_experiment(config_path, args_list, log_pth, label, dry_run)


def run_cd(env_key, n_agents, cost_func_key, guide_weight_constr, log_pth, dry_run=False):
    """Coupled Diffusion: coupling guidance only, no velocity projection."""
    env = ENVS[env_key]
    config_path = join(CONFIG_DIR, env['pcd_config'])
    label = f"CD | env={env_key} agents={n_agents} cost={cost_func_key} scl={guide_weight_constr}"

    args_list = _common_args(n_agents, log_pth) + [
        # ── Planner identity ──
        f"--planner.single_agent.class=PCD",
        f"--planner.single_agent.planner_alg=coupling",
        # ── Guidance ──
        f"--planner.single_agent.start_guide_steps_fraction={GUIDE_FRACTION}",
        f"--planner.single_agent.n_guide_steps={N_GUIDE_STEPS}",
        f"--planner.single_agent.weight_grad_cost_constraints={guide_weight_constr}",
        f"--planner.single_agent.weight_grad_cost_collision={env['guide_weight_coll']}",
        f"--planner.single_agent.weight_grad_cost_smoothness={GUIDE_WEIGHT_SMOOTH}",
        # ── Cost function ──
        f"--planner.single_agent.cost_func_key={cost_func_key}",
        f"--planner.single_agent.cost_func_param.margin_coeff={COST_MARGIN_COEFF}",
        f"--planner.single_agent.cost_func_param.radius_coeff={COST_RADIUS_COEFF}",
    ]

    _run_experiment(config_path, args_list, log_pth, label, dry_run)


def run_pcd(env_key, n_agents, vel_max, cost_func_key, guide_weight_constr, log_pth, dry_run=False):
    """PCD (full): coupling guidance + ADMM velocity projection."""
    env = ENVS[env_key]
    config_path = join(CONFIG_DIR, env['pcd_config'])
    label = f"PCD | env={env_key} agents={n_agents} vmax={vel_max} cost={cost_func_key} scl={guide_weight_constr}"

    args_list = _common_args(n_agents, log_pth) + [
        # ── Planner identity ──
        f"--planner.single_agent.class=PCD",
        f"--planner.single_agent.planner_alg=pcd",
        # ── Guidance ──
        f"--planner.single_agent.start_guide_steps_fraction={GUIDE_FRACTION}",
        f"--planner.single_agent.n_guide_steps={N_GUIDE_STEPS}",
        f"--planner.single_agent.weight_grad_cost_constraints={guide_weight_constr}",
        f"--planner.single_agent.weight_grad_cost_collision={env['guide_weight_coll']}",
        f"--planner.single_agent.weight_grad_cost_smoothness={GUIDE_WEIGHT_SMOOTH}",
        # ── Cost function ──
        f"--planner.single_agent.cost_func_key={cost_func_key}",
        f"--planner.single_agent.cost_func_param.margin_coeff={COST_MARGIN_COEFF}",
        f"--planner.single_agent.cost_func_param.radius_coeff={COST_RADIUS_COEFF}",
        # ── Velocity projection ──
        f"--planner.projection.project_params.vel_max={vel_max}",
    ] + _admm_projector_args(rho=env['admm_rho'])

    _run_experiment(config_path, args_list, log_pth, label, dry_run)


def run_direct(env_key, n_agents, vel_max, log_pth, dry_run=False):
    """DiRecT: no guidance, JaxScpADMM Tweedie projection at every step."""
    env = ENVS[env_key]
    config_path = join(CONFIG_DIR, env['direct_config'])
    label = f"DiRecT | env={env_key} agents={n_agents} vmax={vel_max}"

    args_list = _common_args(n_agents, log_pth) + [
        # ── Planner identity ──
        f"--planner.single_agent.class=DiRecT",
        f"--planner.single_agent.planner_alg=base",
        f"--planner.single_agent.projector_type=JaxScpADMM",
        # ── No guidance ──
        f"--planner.single_agent.start_guide_steps_fraction=0.0",
        f"--planner.single_agent.n_guide_steps=0",
        f"--planner.single_agent.weight_grad_cost_constraints=0",
        f"--planner.single_agent.weight_grad_cost_collision=0",
        f"--planner.single_agent.weight_grad_cost_smoothness={GUIDE_WEIGHT_SMOOTH}",
        # ── Projection ──
        f"--planner.single_agent.start_projection_guidance={PROJ_FRACTION_DIRECT}",
        f"--planner.projection.project_params.vel_max={vel_max}",
        f"--planner.projection.project_params.smoothness_weight={env['direct_smoothness_weight']}",
    ]

    _run_experiment(config_path, args_list, log_pth, label, dry_run)


def run_final_projection(env_key, n_agents, vel_max, log_pth, dry_run=False):
    """Final Projection: unguided diffusion + JaxScpADMM projection at last step only."""
    env = ENVS[env_key]
    config_path = join(CONFIG_DIR, env['direct_config'])
    label = f"FINAL_PROJ | env={env_key} agents={n_agents} vmax={vel_max}"

    args_list = _common_args(n_agents, log_pth) + [
        # ── Planner identity ──
        f"--planner.single_agent.class=DiRecT",
        f"--planner.single_agent.planner_alg=final",
        f"--planner.single_agent.projector_type=JaxScpADMM",
        # ── No guidance ──
        f"--planner.single_agent.start_guide_steps_fraction=0.0",
        f"--planner.single_agent.n_guide_steps=0",
        f"--planner.single_agent.weight_grad_cost_constraints=0",
        f"--planner.single_agent.weight_grad_cost_collision=0",
        f"--planner.single_agent.weight_grad_cost_smoothness={GUIDE_WEIGHT_SMOOTH}",
        # ── Projection (t=0 only, enforced by planner_alg=final) ──
        f"--planner.single_agent.start_projection_guidance={PROJ_FRACTION_FINAL}",
        f"--planner.projection.project_params.vel_max={vel_max}",
        f"--planner.projection.project_params.smoothness_weight={env['direct_smoothness_weight']}",
    ]

    _run_experiment(config_path, args_list, log_pth, label, dry_run)


def run_mmd_cbs(env_key, n_agents, log_pth, dry_run=False):
    """MMD-CBS baseline: CBS multi-agent planner with MPD low-level, no velocity constraints."""
    from mmd.common.experiments.experiments import MultiAgentPlanningExperimentConfig

    instance_name = MMD_ENV_INSTANCE_MAP.get(env_key)
    if instance_name is None:
        logger.warning(f"MMD: no instance mapping for env '{env_key}', skipping.")
        return

    label = f"MMD-CBS | env={env_key} agents={n_agents}"
    logger.info(f"\n{'─'*70}")
    logger.info(f"  {label}")
    logger.info(f"  instance: {instance_name}")
    logger.info(f"{'─'*70}")

    if dry_run:
        logger.info("  [DRY RUN] skipping execution")
        return

    launch_mmd_mod = import_mmd_modules()

    experiment_config = MultiAgentPlanningExperimentConfig()
    experiment_config.seed = SEED
    experiment_config.num_agents_l = [n_agents]
    experiment_config.instance_name = instance_name
    experiment_config.stagger_start_time_dt = 0
    experiment_config.multi_agent_planner_class_l = [MMD_MULTI_AGENT_PLANNER]
    experiment_config.single_agent_planner_class = MMD_SINGLE_AGENT_PLANNER
    experiment_config.runtime_limit = MMD_RUNTIME_LIMIT
    experiment_config.num_trials_per_combination = N_INITS
    experiment_config.render_animation = False

    launch_mmd_mod.run_multi_agent_experiment(experiment_config)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global GUIDE_WEIGHT_SMOOTH

    parser = argparse.ArgumentParser(
        description="Benchmark comparison across methods, envs, agents, vmaxs, costs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--envs", nargs="+", default=list(ENVS.keys()),
                        choices=list(ENVS.keys()),
                        help="Environments to run (default: all)")
    parser.add_argument("--agents", nargs="+", type=int, default=N_AGENTS_LIST,
                        help=f"Agent counts (default: {N_AGENTS_LIST})")
    parser.add_argument("--vmaxs", nargs="+", type=float, default=None,
                        help="Velocity limits (default: per-environment from ENVS)")
    parser.add_argument("--guide-weight-smooth", type=float, default=GUIDE_WEIGHT_SMOOTH,
                        help=f"weight_grad_cost_smoothness (default: {GUIDE_WEIGHT_SMOOTH})")
    parser.add_argument("--costs", nargs="+", default=COST_FUNC_KEYS,
                        choices=COST_FUNC_KEYS,
                        help=f"Cost functions (default: {COST_FUNC_KEYS})")
    parser.add_argument("--only", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help=f"Methods to run (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    parser.add_argument("-m", "--comments", type=str, default="benchmark",
                        help="Comment tag for log file")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_pth = os.path.join(LOG_DIR,
                           f"benchmark-{timestamp}-{args.comments}.log")
    logger.info(f"Logging to {log_pth}")

    # Update module-level defaults from CLI flags
    GUIDE_WEIGHT_SMOOTH = args.guide_weight_smooth

    methods = set(args.only)
    envs = args.envs
    agents_list = args.agents
    costs = args.costs

    # Methods that don't sweep over vel_max / cost
    NO_SWEEP_METHODS = {'diffusion_policy', 'mmd_cbs'}
    # Methods that sweep over cost functions (LB / SHD)
    COST_SWEEP_METHODS = {'pcd', 'cd'}

    # Count total runs for progress tracking
    n_runs = 0
    for env in envs:
        vmaxs = args.vmaxs if args.vmaxs is not None else ENVS[env]['vel_maxs']
        for n_ag in agents_list:
            for m in ALL_METHODS:
                if m not in methods:
                    continue
                if m in NO_SWEEP_METHODS:
                    n_runs += 1
                elif m in COST_SWEEP_METHODS:
                    if m == 'cd':
                        n_runs += len(costs)  # CD doesn't sweep vel_max
                    else:
                        n_runs += len(vmaxs) * len(costs)
                else:
                    n_runs += len(vmaxs)

    logger.info(f"Total experiment runs: {n_runs}")
    logger.info(f"  Environments: {envs}")
    logger.info(f"  Agent counts: {agents_list}")
    logger.info(f"  Vel maxs:     {{'per-env' if args.vmaxs is None else args.vmaxs}}")
    logger.info(f"  Cost funcs:   {costs}")
    logger.info(f"  Cost γ:       {COST_FUNC_DEFAULTS}")
    logger.info(f"  Methods:      {[m for m in ALL_METHODS if m in methods]}")

    run_i = 0
    for env in envs:
        vmaxs = args.vmaxs if args.vmaxs is not None else ENVS[env]['vel_maxs']
        logger.info(f"\n  env={env}  vel_maxs={vmaxs}")

        for n_agents in agents_list:
            for method in ALL_METHODS:
                if method not in methods:
                    continue

                if method == 'diffusion_policy':
                    run_i += 1
                    logger.info(f"\n[{run_i}/{n_runs}]")
                    run_diffusion_policy(env, n_agents, log_pth, dry_run=args.dry_run)

                elif method == 'cd':
                    for cost_func_key in costs:
                        gwc = COST_FUNC_DEFAULTS[cost_func_key]['guide_weight_constr']
                        run_i += 1
                        logger.info(f"\n[{run_i}/{n_runs}]")
                        run_cd(env, n_agents, cost_func_key,
                               gwc, log_pth, dry_run=args.dry_run)

                elif method == 'pcd':
                    for vel_max in vmaxs:
                        for cost_func_key in costs:
                            gwc = COST_FUNC_DEFAULTS[cost_func_key]['guide_weight_constr']
                            run_i += 1
                            logger.info(f"\n[{run_i}/{n_runs}]")
                            run_pcd(env, n_agents, vel_max, cost_func_key,
                                    gwc, log_pth, dry_run=args.dry_run)

                elif method == 'direct':
                    for vel_max in vmaxs:
                        run_i += 1
                        logger.info(f"\n[{run_i}/{n_runs}]")
                        run_direct(env, n_agents, vel_max, log_pth,
                                   dry_run=args.dry_run)

                elif method == 'final_projection':
                    for vel_max in vmaxs:
                        run_i += 1
                        logger.info(f"\n[{run_i}/{n_runs}]")
                        run_final_projection(env, n_agents, vel_max, log_pth,
                                             dry_run=args.dry_run)

                elif method == 'mmd_cbs':
                    run_i += 1
                    logger.info(f"\n[{run_i}/{n_runs}]")
                    run_mmd_cbs(env, n_agents, log_pth, dry_run=args.dry_run)

    logger.info(f"\n{'═'*70}")
    logger.info(f"Benchmark complete. {run_i} runs. Log: {log_pth}")
    logger.info(f"{'═'*70}")


if __name__ == "__main__":
    main()
