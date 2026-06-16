"""
Auto-discover and post-process all MMD result directories.

Step 1: process_mmd.py  -> generates raw_data_*.npz + processed_data_*.npz
Step 2: analyzing_mmd.py -> generates summary stats JSON

Usage:
    python scripts/analyze/run_all_mmd_postprocessing.py
    python scripts/analyze/run_all_mmd_postprocessing.py --dry-run
    python scripts/analyze/run_all_mmd_postprocessing.py --env conveyor
    python scripts/analyze/run_all_mmd_postprocessing.py --skip-process
    python scripts/analyze/run_all_mmd_postprocessing.py --force
"""
import os
import re
import glob
import argparse
import logging
from os.path import join, dirname, basename

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = dirname(dirname(dirname(os.path.abspath(__file__))))
RESULTS_DIR = join(PROJECT_ROOT, "results")

ENV_DIRS = {
    "empty":      "EnvEmpty2DRobotPlanarDiskRandom",
    "highways":   "EnvHighways2DRobotPlanarDiskRandom",
    "conveyor":   "EnvConveyor2DRobotPlanarDiskRandom",
    "dropregion": "EnvDropRegion2DRobotPlanarDiskRandom",
}

DEFAULT_MAX_VEL = {
    "empty":      0.703,
    "highways":   0.878,
    "conveyor":   1.76,
    "dropregion": 1.34,
}


def discover_directories(env_filter=None, min_date="2026-04-20"):
    """Auto-discover all result directories and their agent counts."""
    entries = []
    for env_key, env_dir_name in ENV_DIRS.items():
        if env_filter and env_key != env_filter:
            continue
        env_path = join(RESULTS_DIR, env_dir_name)
        if not os.path.isdir(env_path):
            logger.warning(f"Environment directory not found: {env_path}")
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
                    rel_path = os.path.relpath(ts_path, PROJECT_ROOT)
                    entries.append({
                        "env_key": env_key,
                        "abs_path": ts_path,
                        "rel_path": rel_path,
                        "n_agents": n_agents,
                    })
    return entries


def has_npz_files(abs_path, n_agents):
    raw = join(abs_path, f"raw_data_{n_agents}_agents.npz")
    processed = join(abs_path, f"processed_data_{n_agents}_agents.npz")
    return os.path.exists(raw) and os.path.exists(processed)


def run_process_step(entries, force=False, dry_run=False):
    """Step 1: Run process_mmd.main() for each directory."""
    from process_mmd import main as process_main

    to_process = []
    for entry in entries:
        if not force and has_npz_files(entry["abs_path"], entry["n_agents"]):
            logger.info(f"SKIP (already processed): {entry['rel_path']} [{entry['n_agents']} agents]")
            continue
        to_process.append(entry)

    if dry_run:
        logger.info(f"Would process {len(to_process)} directories:")
        for e in to_process:
            print(f"  {e['rel_path']}  [{e['n_agents']} agents]")
        return

    logger.info(f"Processing {len(to_process)} directories...")
    for i, entry in enumerate(to_process):
        logger.info(f"[{i+1}/{len(to_process)}] Processing: {entry['rel_path']} [{entry['n_agents']} agents]")
        try:
            process_main([entry["n_agents"]], entry["rel_path"])
        except Exception as e:
            logger.error(f"FAILED: {entry['rel_path']}: {e}")
            continue


def run_analyze_step(entries, dry_run=False):
    """Step 2: Run analyzing_mmd.run_analysis_for_directory() for each directory."""
    from analyzing_mmd import run_analysis_for_directory

    to_analyze = []
    for entry in entries:
        if not has_npz_files(entry["abs_path"], entry["n_agents"]):
            logger.warning(f"SKIP (no npz files): {entry['rel_path']} [{entry['n_agents']} agents]")
            continue
        to_analyze.append(entry)

    if dry_run:
        logger.info(f"Would analyze {len(to_analyze)} directories:")
        for e in to_analyze:
            print(f"  {e['rel_path']}  [{e['n_agents']} agents]")
        return

    logger.info(f"Analyzing {len(to_analyze)} directories...")
    for i, entry in enumerate(to_analyze):
        max_vel = DEFAULT_MAX_VEL[entry["env_key"]]
        config = {
            "switches": {
                "plot_trajectories": False,
                "plot_velocity_stats": False,
                "save_plots": False,
                "save_stats_json": True,
            },
            "constants": {
                "EPS": 1e-6,
                "VEL_TOL": 5e-5,
                "DEFAULT_MAX_VEL": max_vel,
                "VEL_HIST_QUANTILES": [0.75, 0.8, 0.85, 0.9, 0.95, 0.99],
            },
            "plotting": {
                "num_inits_plot": 10,
                "num_trj_tups_plot": 64,
                "traj_plot_xlim": [-1, 1],
                "traj_plot_ylim": [-1, 1],
                "traj_plot_h_end": 64,
                "vel_hist_bins": 100,
            },
            "data": {
                "default_trained_models_dir": "default_trained_models_dir",
            },
        }
        logger.info(f"[{i+1}/{len(to_analyze)}] Analyzing: {entry['rel_path']} [{entry['n_agents']} agents, max_vel={max_vel}]")
        try:
            run_analysis_for_directory(entry["rel_path"], config, base_dir=PROJECT_ROOT)
        except Exception as e:
            logger.error(f"FAILED: {entry['rel_path']}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Auto post-process all MMD result directories.")
    parser.add_argument("--env", choices=list(ENV_DIRS.keys()), help="Filter by environment")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed without running")
    parser.add_argument("--skip-process", action="store_true", help="Skip step 1 (process_mmd)")
    parser.add_argument("--skip-analyze", action="store_true", help="Skip step 2 (analyzing_mmd)")
    parser.add_argument("--force", action="store_true", help="Reprocess directories that already have .npz files")
    parser.add_argument("--min-date", default="2026-04-20", help="Ignore timestamp dirs before this date (default: 2026-04-20)")
    args = parser.parse_args()

    entries = discover_directories(env_filter=args.env, min_date=args.min_date)
    logger.info(f"Discovered {len(entries)} result directories.")

    if args.dry_run:
        for e in entries:
            status = "DONE" if has_npz_files(e["abs_path"], e["n_agents"]) else "TODO"
            print(f"  [{status}] {e['rel_path']}  [{e['n_agents']} agents]")
        print(f"\nTotal: {len(entries)} directories")
        return

    if not args.skip_process:
        run_process_step(entries, force=args.force, dry_run=False)

    if not args.skip_analyze:
        run_analyze_step(entries, dry_run=False)

    logger.info("Done.")


if __name__ == "__main__":
    main()
