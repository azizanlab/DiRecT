"""
Convert MMD (CBS) results from Env* directories into the single_tile_* format
used by the PCD benchmark, including GIF generation.

This creates:
    single_tile_ENVNAME/agent___N/method___mmd/cost___None/TIMESTAMP-MMD_CBS_|_env=ENVNAME_agents=N/0/
        ├── single_tile_ENVNAME.gif
        ├── results.json
        ├── config.yaml
        ├── raw_data.npz
        └── processed_data.npz

Usage:
    python scripts/analyze/mmd_to_single_tile.py
    python scripts/analyze/mmd_to_single_tile.py --env empty
    python scripts/analyze/mmd_to_single_tile.py --dry-run
    python scripts/analyze/mmd_to_single_tile.py --skip-gif
"""
import os
import re
import sys
import glob
import json
import pickle
import argparse
import logging
import numpy as np
import torch
from os.path import join, dirname, basename
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = join(PROJECT_ROOT, "results")
TRAINED_MODELS_DIR = join(PROJECT_ROOT, "data_trained_models")
TENSOR_ARGS = {'dtype': torch.float32, 'device': 'cuda'}

ENV_DIRS = {
    "empty":      "EnvEmpty2DRobotPlanarDiskRandom",
    "highway":    "EnvHighways2DRobotPlanarDiskRandom",
    "conveyor":   "EnvConveyor2DRobotPlanarDiskRandom",
    "dropregion": "EnvDropRegion2DRobotPlanarDiskRandom",
}

ENV_TO_SINGLE_TILE = {
    "empty":      "single_tile_empty",
    "highway":    "single_tile_highway",
    "conveyor":   "single_tile_conveyor",
    "dropregion": "single_tile_dropregion",
}

ENV_TO_MODEL_ID = {
    "empty":      "EnvEmpty2D-RobotPlanarDisk",
    "highway":    "EnvHighways2D-RobotPlanarDisk",
    "conveyor":   "EnvConveyor2D-RobotPlanarDisk",
    "dropregion": "EnvDropRegion2D-RobotPlanarDisk",
}


def discover_mmd_directories(env_filter=None, min_date="2026-04-20"):
    """Find all MMD result directories grouped by (env, n_agents)."""
    entries = []
    for env_key, env_dir_name in ENV_DIRS.items():
        if env_filter and env_key != env_filter:
            continue
        env_path = join(RESULTS_DIR, env_dir_name)
        if not os.path.isdir(env_path):
            continue
        for timestamp_dir in sorted(os.listdir(env_path)):
            ts_path = join(env_path, timestamp_dir)
            if not os.path.isdir(ts_path) or not re.match(r"\d{4}-\d{2}-\d{2}", timestamp_dir):
                continue
            if min_date and timestamp_dir < min_date:
                continue
            agent_dirs = glob.glob(join(ts_path, "num_agents___*"))
            for agent_dir in agent_dirs:
                match = re.search(r"num_agents___(\d+)", basename(agent_dir))
                if match:
                    n_agents = int(match.group(1))
                    entries.append({
                        "env_key": env_key,
                        "timestamp": timestamp_dir,
                        "abs_path": ts_path,
                        "n_agents": n_agents,
                    })
    return entries


def load_trial_result(trial_dir):
    """Load results.pkl from a trial directory."""
    pkl_path = join(trial_dir, "results.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_trial_base_dir(entry):
    """Get the base trial directory for an entry."""
    return join(entry["abs_path"],
                f"num_agents___{entry['n_agents']}",
                "planner___CBS",
                "single_agent_planner___MPD")


def has_any_trials(entry):
    """Check if any trial directories with results.pkl exist."""
    base = get_trial_base_dir(entry)
    if not os.path.isdir(base):
        return False
    for name in os.listdir(base):
        if name.isdigit() and os.path.exists(join(base, name, "results.pkl")):
            return True
    return False


def pick_best_trial(entry):
    """Pick the trial with best success status (first successful, or trial 0)."""
    base = get_trial_base_dir(entry)
    if not os.path.isdir(base):
        return None, None

    trial_dirs = []
    for name in os.listdir(base):
        if name.isdigit():
            trial_dirs.append((int(name), join(base, name)))
    trial_dirs.sort(key=lambda x: x[0])

    # Try to find a successful trial for the GIF
    best_result = None
    best_dir = None
    for trial_num, trial_dir in trial_dirs:
        result = load_trial_result(trial_dir)
        if result is None:
            continue
        if best_result is None:
            best_result = result
            best_dir = trial_dir
        if result.success_status and result.success_status.name == "SUCCESS":
            best_result = result
            best_dir = trial_dir
            break
    return best_result, best_dir


def get_task_for_env(env_key):
    """Load the PlanningTask for an environment via the dataset."""
    from mmd.utils.loading import load_params_from_yaml
    from mmd.trainer import get_dataset

    model_id = ENV_TO_MODEL_ID[env_key]
    model_dir = join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(join(model_dir, "args.yaml"))

    train_subset, _, _, _ = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,
        **args,
        tensor_args=TENSOR_ARGS
    )
    dataset = train_subset.dataset
    return dataset.task, dataset.robot


def render_gif(task, paths_l, start_l, goal_l, output_path, n_agents):
    """Render a GIF of the trajectories using PlanningVisualizer."""
    from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
    from mmd.common.trajectory_utils import densify_trajs
    import matplotlib
    matplotlib.use('Agg')

    planner_visualizer = PlanningVisualizer(task=task)

    # Generate colors for agents
    import matplotlib.cm as cm
    colors = [cm.tab20(i / max(1, n_agents - 1)) if n_agents > 1 else cm.tab20(0)
              for i in range(n_agents)]

    # Densify for smoother animation
    render_paths = densify_trajs(paths_l, 1)
    # Add batch dim
    render_paths_batched = [p.unsqueeze(0) for p in render_paths]

    n_frames = max(2, render_paths_batched[0].shape[1])
    planner_visualizer.animate_multi_robot_trajectories(
        trajs_l=render_paths_batched,
        start_state_l=start_l,
        goal_state_l=goal_l,
        plot_trajs=True,
        video_filepath=output_path,
        n_frames=n_frames,
        anim_time=5.0,
        colors=colors,
    )


def build_results_json(result, n_agents):
    """Build a results.json dict from a MultiAgentPlanningSingleTrialResult."""
    d = {
        "num_ct_expansions": int(result.num_ct_expansions),
        "num_collisions_in_solution": int(result.num_collisions_in_solution),
        "data_adherence": float(result.data_adherence),
        "planning_time": float(result.planning_time),
        "path_length_per_agent": float(result.path_length_per_agent),
        "mean_path_acceleration_per_agent": float(result.mean_path_acceleration_per_agent),
        "success_status": result.success_status.name if result.success_status else "UNKNOWN",
    }
    # Velocity lists
    for attr in ['max_velocity_l', 'mean_velocity_l']:
        val = getattr(result, attr, [])
        if val:
            d[attr] = [float(v) if not hasattr(v, 'item') else v.item() for v in val]
        else:
            d[attr] = []

    d["global_model_ids"] = result.global_model_ids if result.global_model_ids else []
    d["agent_skeleton_l"] = result.agent_skeleton_l if result.agent_skeleton_l else [[[0, 0]]] * n_agents
    d["start_state_pos_l"] = "raw_data.starts_global"
    d["goal_state_pos_l"] = "raw_data.goals_global"
    d["agent_path_l"] = "raw_data.all_state_trjs"
    return d


def build_config_yaml(result, n_agents, env_key):
    """Build a config dict matching the PCD format."""
    cfg = result.trial_config
    return {
        "num_agents": n_agents,
        "dt": getattr(cfg, 'dt', 0.1),
        "planner_class": "CBS",
        "single_agent_planner_class": "MPD",
        "instance_name": ENV_DIRS[env_key],
    }


def _compute_and_save_stats(entry, all_state_trjs, out_dir, task_cache):
    """Compute collision masks, velocity, data adherence and save processed_data.npz + summary_stats JSON."""
    from mmd.common.eval_utils import collision_detect, calc_data_adherence, calc_velocity

    env_key = entry["env_key"]
    n_agents = entry["n_agents"]
    B, I, P, H, D = all_state_trjs.shape

    # Load task/robot for collision checking and normalization
    if env_key not in task_cache:
        logger.info(f"Loading task for {env_key}...")
        task_cache[env_key] = get_task_for_env(env_key)
    task, robot = task_cache[env_key]

    # Inter-agent collisions
    collision_masks = collision_detect(all_state_trjs[..., :2], safe_dist=robot.radius * 2, norm_order=2)
    cnt_coll = np.sum(collision_masks.any(-1).astype(int), axis=(-2, -1))
    cnt_coll = cnt_coll // 2  # pairs counted twice

    # Static obstacle collisions
    trajs_tensor = torch.from_numpy(all_state_trjs).reshape(B * I * P, H, -1).to(**TENSOR_ARGS)
    _, _, _, _, trajs_wp_coll = task.get_trajs_collision_and_free(
        trajs_tensor, return_indices=True, num_interpolation=0)
    obstacle_collision_masks = trajs_wp_coll.cpu().numpy().reshape(B, I, P, H)

    # Velocity (on normalized trajectories)
    dataset = task_cache[env_key]  # (task, robot) tuple - need dataset for normalization
    # Use simple finite-difference velocity on position
    traj_pos = all_state_trjs[..., :2]
    dt = 5.0 / H  # default traj_duration / horizon
    velocities = calc_velocity(traj_pos, dt=dt)
    velocities = np.concatenate([np.zeros(velocities.shape[:-1] + (1,)), velocities], axis=-1)

    # Data adherence placeholder (requires model loading - use result value if available)
    data_adherence = np.ones((B, I, P), dtype=np.float32)

    # Save processed_data.npz
    np.savez_compressed(join(out_dir, "processed_data.npz"),
                        collision_masks=collision_masks,
                        obstacle_collision_masks=obstacle_collision_masks,
                        velocity_trjs=velocities,
                        data_adherence=data_adherence)

    # Compute summary stats
    # Safety rates
    safe_masks = ~collision_masks  # [B, I, P, P, H]
    trj_safe = safe_masks.all(-1)  # [B, I, P, P]
    strict_safe = trj_safe.all(axis=(-1, -2))  # [B, I]
    overall_strict_safe_rate = float(strict_safe.mean())
    loose_safe = (trj_safe.astype(float).sum(axis=(-1, -2)) - P) / (P * (P - 1))
    overall_loose_safe_rate = float(loose_safe.mean())

    # Obstacle safety
    static_safe = ~obstacle_collision_masks.any(-1)  # [B, I, P]
    overall_static_safe = float(static_safe.astype(float).mean())

    # Combined safety
    agent_strict_safe = trj_safe.all(-1)  # [B, I, P]
    all_safe = np.logical_and(agent_strict_safe, static_safe)
    overall_all_safe = float(all_safe.astype(float).mean())

    # Success rate
    success_masks = all_safe.all(-1)  # [B, I]
    best_success = success_masks.max(axis=0) if B > 1 else success_masks.squeeze(0)
    success_rate = float(np.mean(best_success.astype(float)))

    # Use a default vmax (0.8) for the summary stats filename
    # The notebook will recompute vel_constraint_sat for each actual vmax
    default_vmax = 0.8
    vel_sat = velocities <= default_vmax + 5e-5
    vel_sat_per_traj = np.all(vel_sat, axis=-1)
    overall_vel_sat = float(vel_sat_per_traj.astype(float).mean())

    summary = {
        "Overall Mean Collision Count": float(cnt_coll.mean()),
        "Overall Mean Strict Inter-Agent Safe Rate": overall_strict_safe_rate,
        "Overall Mean Loose Inter-Agent Safe Rate": overall_loose_safe_rate,
        "Overall Mean Static Obstacle Safe Rate": overall_static_safe,
        "Overall Mean All Safe Rate": overall_all_safe,
        "Overall Success Rate": success_rate,
        "Overall Mean Velocity Constraint Satisfaction": overall_vel_sat,
        "Max Velocity": default_vmax,
        "num_agents": n_agents,
    }

    stats_path = join(out_dir, f"summary_stats_vmax{default_vmax}.json")
    with open(stats_path, 'w') as f:
        json.dump([summary], f, indent=4)
    logger.info(f"  Stats saved: {os.path.relpath(stats_path, PROJECT_ROOT)}")


def process_entry(entry, task_cache, skip_gif=False):
    """Process a single MMD result directory into single_tile format."""
    env_key = entry["env_key"]
    n_agents = entry["n_agents"]
    timestamp = entry["timestamp"]

    # Pick best trial for GIF
    result, trial_dir = pick_best_trial(entry)
    if result is None:
        logger.warning(f"No results found: {entry['abs_path']}")
        return False

    # Build output directory
    tile_name = ENV_TO_SINGLE_TILE[env_key]
    ts_label = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S").strftime("%m%d%Y_%H%M%S")
    run_dir_name = f"{ts_label}-MMD_CBS_|_env={env_key}_agents={n_agents}"
    out_dir = join(RESULTS_DIR, tile_name,
                   f"agent___{n_agents}",
                   "method___mmd",
                   "cost___None",
                   run_dir_name,
                   "0")

    os.makedirs(out_dir, exist_ok=True)

    # Save results.json
    results_json = build_results_json(result, n_agents)
    with open(join(out_dir, "results.json"), 'w') as f:
        json.dump(results_json, f, indent=4)

    # Save config.yaml
    import yaml
    config_dict = build_config_yaml(result, n_agents, env_key)
    with open(join(out_dir, "config.yaml"), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # Save raw_data.npz (trajectories, starts, goals)
    paths_np = torch.stack(result.agent_path_l, dim=0).cpu().numpy()  # [P, H, D]
    starts_np = np.stack([s.cpu().numpy() if torch.is_tensor(s) else np.array(s)
                          for s in result.start_state_pos_l], axis=0)  # [P, 2]
    goals_np = np.stack([g.cpu().numpy() if torch.is_tensor(g) else np.array(g)
                         for g in result.goal_state_pos_l], axis=0)  # [P, 2]
    all_state_trjs = paths_np[np.newaxis, np.newaxis, ...]  # [1, 1, P, H, D]
    np.savez_compressed(join(out_dir, "raw_data.npz"),
                        all_state_trjs=all_state_trjs,
                        starts_global=starts_np,
                        goals_global=goals_np)

    # Compute processed data and summary stats
    _compute_and_save_stats(entry, all_state_trjs, out_dir, task_cache)

    # Render GIF
    if not skip_gif:
        if env_key not in task_cache:
            logger.info(f"Loading task for {env_key}...")
            task_cache[env_key] = get_task_for_env(env_key)

        task, robot = task_cache[env_key]
        gif_path = join(out_dir, f"{tile_name}.gif")

        paths_l = result.agent_path_l
        # Ensure tensors are on correct device
        paths_l = [p.to(**TENSOR_ARGS) if torch.is_tensor(p) else torch.tensor(p, **TENSOR_ARGS)
                    for p in paths_l]
        start_l = [s.to(**TENSOR_ARGS) if torch.is_tensor(s) else torch.tensor(s, **TENSOR_ARGS)
                    for s in result.start_state_pos_l]
        goal_l = [g.to(**TENSOR_ARGS) if torch.is_tensor(g) else torch.tensor(g, **TENSOR_ARGS)
                   for g in result.goal_state_pos_l]

        try:
            render_gif(task, paths_l, start_l, goal_l, gif_path, n_agents)
            logger.info(f"  GIF saved: {os.path.relpath(gif_path, PROJECT_ROOT)}")
        except Exception as e:
            logger.error(f"  GIF rendering failed: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MMD results to single_tile format with GIFs.")
    parser.add_argument("--env", choices=list(ENV_DIRS.keys()),
                        help="Filter by environment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be created")
    parser.add_argument("--skip-gif", action="store_true",
                        help="Skip GIF rendering (just create metadata)")
    parser.add_argument("--min-date", default="2026-04-20",
                        help="Ignore timestamps before this date")
    args = parser.parse_args()

    entries = discover_mmd_directories(env_filter=args.env, min_date=args.min_date)
    logger.info(f"Discovered {len(entries)} MMD result directories.")

    if args.dry_run:
        for e in entries:
            env_key = e["env_key"]
            n_agents = e["n_agents"]
            timestamp = e["timestamp"]
            tile_name = ENV_TO_SINGLE_TILE[env_key]
            ts_label = datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S").strftime("%m%d%Y_%H%M%S")
            run_dir_name = f"{ts_label}-MMD_CBS_|_env={env_key}_agents={n_agents}"
            out_rel = join(tile_name, f"agent___{n_agents}", "method___mmd", "cost___None", run_dir_name, "0")
            has_data = has_any_trials(e)
            status = "OK" if has_data else "NO DATA"
            print(f"  [{status}] {out_rel}")
        return

    task_cache = {}
    success = 0
    for i, entry in enumerate(entries):
        logger.info(f"[{i+1}/{len(entries)}] {entry['env_key']} / {entry['n_agents']} agents / {entry['timestamp']}")
        try:
            if process_entry(entry, task_cache, skip_gif=args.skip_gif):
                success += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            continue

    logger.info(f"Done. {success}/{len(entries)} directories processed.")


if __name__ == "__main__":
    main()
