import numpy as np
import torch
import os
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt
import logging
import re
from tqdm import tqdm
from os.path import join, dirname
from typing import Dict, Any, Tuple

from torch_robotics.tasks.tasks import PlanningTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from helpers import (
    plot_velocity_stats, plot_velocity_histogram,
)

def load_data(data_abs_dir: str) -> Tuple[Dict[str, np.ndarray]]:
    """Loads raw data from the .npz file."""
    raw_data_pth_ls = []
    processed_data_pth_ls = []
    n_agents_ls = []
    for root, _, files in os.walk(data_abs_dir):
        for file in files:
            if file.endswith('.npz'):
                if 'raw_data' not in file:
                    continue
                
                file_path = os.path.join(root, file)
                raw_data_pth_ls.append(file_path)
                match = re.search(r'(\d+)_agents', file)
                try:
                    n_agents = int(match.group(1))
                except (AttributeError, ValueError):
                    logger.warning(f"Could not extract number of agents from filename: {file}. Using default value 2.")
                    n_agents = 2
                n_agents_ls.append(n_agents)
                
                processed_data_pth = os.path.join(root, file.replace('raw_data', 'processed_data'))
                assert os.path.exists(processed_data_pth), \
                    f"Processed data file not found: \'{processed_data_pth}\'. Ensure it exists."
                processed_data_pth_ls.append(processed_data_pth)
    
    if not raw_data_pth_ls or not processed_data_pth_ls:
        raise FileNotFoundError(
            f"No raw or processed data files found in directory: {data_abs_dir}. "
            "Ensure the directory contains .npz files with 'raw_data' or 'processed_data' in their names."
        )
    
    raw_data_ls = []
    processed_data_ls = []
    for raw_data_path, processed_data_path in zip(raw_data_pth_ls, processed_data_pth_ls):
        print(f"Loading data from: {dirname(raw_data_path)}")
        raw_data = np.load(raw_data_path, allow_pickle=True)
        processed_data = np.load(processed_data_path, allow_pickle=True)
        raw_data_ls.append(raw_data)
        processed_data_ls.append(processed_data)

    return n_agents_ls, raw_data_ls, processed_data_ls


def analyze_collisions(
    robot_collision_masks: np.ndarray, 
    config: Dict, 
    obstacle_collision_masks: np.ndarray = None
) -> Dict[str, Any]:
    """Performs collision detection and calculates safety statistics."""
    print("Analyzing collisions...")

    N, I, P, _, H = robot_collision_masks.shape
    safe_masks = np.logical_not(robot_collision_masks)  # (N, I, P, P, H)
    trj_safe_masks:np.ndarray = safe_masks.all(-1)  # (N, I, P, P)

    strict_safe_masks = trj_safe_masks.all(axis=(-1, -2))      # (N, I,), whether all agents are collision-free with each other
    assert strict_safe_masks.shape == (N, I), \
        f"Expected strict safe masks shape [N, I]({N, I}), got {strict_safe_masks.shape}."
    overall_std_strict_safe_rt = strict_safe_masks.astype(float).std()  # Overall pooled std
    
    loose_safe_rates = (trj_safe_masks.astype(float).sum(axis=(-1, -2)) - P) \
        / (P * (P - 1))  # (N, I) 
    loose_safe_rates_per_init = loose_safe_rates.mean(axis=0)  # (I,)
    strict_safe_masks = trj_safe_masks.all(-1)      # (N, I, P), whether this agent is collision-free with all other agents
    strict_safe_rates_per_init = np.mean(strict_safe_masks.all(-1).astype(float), axis=(0,))  # (I,)
    overall_loose_safe_rate = loose_safe_rates_per_init.mean()  # Overall safe rate across all players
    overall_strict_safe_rate = strict_safe_rates_per_init.mean()  # Overall safe rate across all players
    collision_counts = np.sum(robot_collision_masks.astype(int), axis=(-1, -2, -3)) // 2  # (N, I)

    print(f"Overall strict safe rate: {overall_strict_safe_rate:.2%}")
    print(f"Overall loose safe rate: {overall_loose_safe_rate:.2%}")

    if obstacle_collision_masks is None or not isinstance(obstacle_collision_masks, np.ndarray):
        raise ValueError(
            "Obstacle collision masks must be provided as a numpy array. "
            "Ensure 'obstacle_collision_masks' is passed correctly."
        )

    static_obstacle_safe_rates_per_agent = np.logical_not(obstacle_collision_masks.any(-1))  # (N, I, P)
    static_obstacle_safe_rates = static_obstacle_safe_rates_per_agent.astype(float).mean(axis=-1)  # (N, I)
    overall_static_obstacle_safe_rate = static_obstacle_safe_rates.mean()  # Overall safe rate across all batches & inits
    print(f"Overall static obstacle safe rate: {overall_static_obstacle_safe_rate:.2%}")

    all_safe_masks = np.logical_and(
        strict_safe_masks,        # (N, I, P) whether this agent is collision-free with all other agents 
        static_obstacle_safe_rates_per_agent,  # (N, I, P) whether this agent is collision-free with all obstacles
    )  # (N, I, P)
    all_safe_rates_per_agent = all_safe_masks.mean(axis=(0, 1))  # (P,) mean safe rate per player across all inits and batches 
    overal_safe_rates = all_safe_rates_per_agent.mean()  # Overall safe rate across all batches & inits
    print(f"Overall safe rate: {overal_safe_rates:.2%}")

    success_masks = all_safe_masks.all(-1)  # (N, I) whether all agents are collision-free with all other agents and obstacles
    best_success_masks = success_masks.max(axis=0)  # (I,) whether at least one batch is collision-free for each init 
    success_rate = best_success_masks.mean()  # Overall success rate across all inits
    std_success_rate = best_success_masks.std()  # Overall pooled std
    print(f"Overall success rate: {success_rate:.2%}")

    return {
        'Overall Mean Collision Count': collision_counts.mean(),
        'Overall Mean Strict Inter-Agent Safe Rate': overall_strict_safe_rate,
        'Overall Std Strict Inter-Agent Safe Rate': overall_std_strict_safe_rt,
        'Overall Mean Loose Inter-Agent Safe Rate': overall_loose_safe_rate,
        'Overall Mean Static Obstacle Safe Rate': overall_static_obstacle_safe_rate,
        'Overall Mean All Safe Rate': overal_safe_rates, 
        'Overall Success Rate': success_rate,
        'Overall Std Success Rate': std_success_rate,
    }


def analyze_rewards(data_adherences:np.ndarray) -> Dict[str, Any]:
    """Calculates and returns data adherence statistics."""
    print("Analyzing Data Adherence...")
    assert data_adherences.ndim == 3, "Data adherence should be a 3D array (N, I, P)."
    N, I, P = data_adherences.shape
    mean_rewards_per_player = np.mean(data_adherences, axis=(0, 1)) # (P, )
    std_rewards_per_player = data_adherences.std(axis=(0, 1))  # (P, )
    overall_mean_rewards = np.mean(mean_rewards_per_player)
    
    print(f"Mean data adherence per player: {np.round(mean_rewards_per_player, 4)}")
    print(f"Overall mean data adherence: {overall_mean_rewards:.4f}")
    
    return {
        'Overall Agent Data Adherence': tuple(mean_rewards_per_player.tolist()),
        'Std Agent Data Adherence': tuple(std_rewards_per_player.tolist()),
        'Overall Mean Data Adherence': overall_mean_rewards,
    }

def analyze_velocity(vel: np.ndarray, config: Dict, expt_params: Dict, save_dir: str):
    """Calculates velocity, profiles it, and plots statistics."""
    print("Analyzing velocity...")
    constants = config['constants']

    N, I, P, H = vel.shape
    v_eps = constants['VEL_TOL']

    max_vel = expt_params.get('max_vel')
    if max_vel is None:
        max_vel = constants['DEFAULT_MAX_VEL']
        logger.warning(f"Max velocity not found in experiment setting, using default max velocity: {max_vel}.")
    if abs(max_vel - constants['DEFAULT_MAX_VEL']) > 1e-6:
        logger.warning(f"Max velocity ({max_vel}) differs from default ({constants['DEFAULT_MAX_VEL']}).")

    vel_constr_sat = vel <= max_vel + v_eps
    vel_constr_sat_seg:np.ndarray = np.all(vel_constr_sat, axis=-1)  # (N, I, P)
    mean_vel_constr_sat_per_player = np.mean(vel_constr_sat_seg.astype(float), axis=(0, 1))  # (P, )
    overall_mean_vel_constr_sat = np.mean(mean_vel_constr_sat_per_player)
    overall_std_vel_constr_sat_per_player = vel_constr_sat_seg.std(axis=(0, 1))

    print("Mean velocity constraint satisfaction per player: " + '(' +
        ", ".join([f"{v:.2%}" for v in mean_vel_constr_sat_per_player])
        + ')')
    print(f"Overall mean velocity constraint satisfaction: {overall_mean_vel_constr_sat:.2%}")

    v = vel.reshape(N*I*P, H) # (I*N*P, H)
    if config['switches']['plot_velocity_stats']:
        hard_deck = max_vel
        proj_method = expt_params.get('projection_method')
        if proj_method is None:
            proj_method = "N/A"
        title = f"(Normalized) Velocity Stats (Proj: {proj_method})"
        subtitle = f"(Horizon: {H}, Max Vel.: {max_vel:.4f}, Num_Agents: {P}, Samples/Agent: {expt_params.get('n_samples', N)})"
        
        fig1, _ = plot_velocity_stats(np.arange(v.shape[-1]), v.mean(0), v.std(0), v.max(0), v.min(0), v.mean(), hard_deck, title, subtitle)
        fig2, _ = plot_velocity_histogram(v, bins=config['plotting']['vel_hist_bins'], quantiles=constants['VEL_HIST_QUANTILES'], title=f"Velocity Histogram (Proj: {proj_method})", subtitle=subtitle)

        if config['switches']['save_plots']:
            logger.info("Saving velocity plots...")
            fig1.savefig(os.path.join(save_dir, f"vel_stats_{proj_method.replace('/', '')}_max_vel{max_vel}.png"), bbox_inches='tight', dpi=300)
            fig2.savefig(os.path.join(save_dir, f"vel_hist_{proj_method.replace('/', '')}_max_vel{max_vel}.png"), bbox_inches='tight', dpi=300)
        plt.close(fig1)
        plt.close(fig2)
    
    return {
        'Overall Mean Velocity Constraint Satisfaction': overall_mean_vel_constr_sat,
        'Mean Velocity Constraint Satisfaction per Player': tuple(mean_vel_constr_sat_per_player.tolist()),
        'Std Velocity Constraint Satisfaction per Player': tuple(overall_std_vel_constr_sat_per_player.tolist()),
        'Max Velocity': max_vel,
    }

def aggregate_and_save_stats(num_agents:int, stats: Dict, data_dir: str, config: Dict):
    """Aggregates all computed stats into a DataFrame and saves to JSON."""
    df = pd.DataFrame([stats])
    df.index = [os.path.basename(data_dir)]
    
    if config['switches']['save_stats_json']:
        save_path = os.path.join(data_dir, f"summary_stats_ag{num_agents}_vmax{stats['Max Velocity']}.json")
        print(f"Saving summary stats to: {save_path}")
        df.to_json(save_path, orient='records', indent=4)
    
    return df

def run_analysis_for_directory(data_dir: str, config: Dict, base_dir:str) -> pd.DataFrame:
    """Main pipeline for processing a single data directory."""
    data_abs_dir = data_dir if os.path.isabs(data_dir) or data_dir.startswith('~') else\
        os.path.join(base_dir, data_dir)
    save_dir = os.path.join(data_abs_dir, "figs")
    os.makedirs(save_dir, exist_ok=True)
    
    expt_params = {}  # This path skips run-config metadata and uses config/default fallbacks for velocity/projection fields.

    success_cnt = 0
    n_agents_ls, raw_data_ls, processed_data_ls = load_data(data_abs_dir)
    for n_agents, raw_data, processed_data in zip(n_agents_ls, raw_data_ls, processed_data_ls):
        all_states:np.ndarray = raw_data['all_state_trjs']  # (N, I, P, H, D)
        num_agents = all_states.shape[-3]
        assert num_agents == n_agents, f"Number of agents in data ({num_agents}) does not match expected ({n_agents})."
        collision_masks:np.ndarray = processed_data['collision_masks']  # (N, I, P, P, H)
        assert collision_masks.shape[-2] == num_agents, \
            f"Expected number of agents in collision masks ({collision_masks.shape[-2]}) to match number of agents in data ({num_agents})."
        all_vel = processed_data['velocity_trjs']  # (N, I, P, H)
        assert all_vel.shape[-2] == num_agents, \
            f"Expected number of agents in velocity trajectories ({all_vel.shape[-2]}) to match number of agents in data ({num_agents})."
        data_adherence:np.ndarray = processed_data['data_adherence']  # (N, I, P) 
        assert data_adherence.shape[-1] == num_agents, \
            f"Expected number of agents in data adherence ({data_adherence.shape[-1]}) to match number of agents in data ({num_agents})."
        obst_coll_msks:np.ndarray = processed_data['obstacle_collision_masks'] # (N, I, P, H)
        assert obst_coll_msks.shape[-2] == num_agents, \
            f"Expected number of agents in obstacle collision masks ({obst_coll_msks.shape[-2]}) to match number of agents in data ({num_agents})."
        
        if 'missing_ids' in raw_data.files: 
            missing_ids = raw_data['missing_ids']
            n_missing = len(missing_ids)
            n_total_inits = raw_data['all_state_trjs'].shape[1] + n_missing
            if n_missing > 0:
                logger.warning(f"Found {n_missing} missing data entries out of {n_total_inits} Inits. Filling with default values.")
                sorted_missing_ids = sorted(missing_ids)
                
                # Prepare default value arrays for bulk insertion
                n_missing = len(sorted_missing_ids)
                bulk_default_coll_msk = np.ones((collision_masks.shape[0], n_missing, num_agents, num_agents, collision_masks.shape[-1]), dtype=collision_masks.dtype)
                bulk_default_data_adh = np.zeros((data_adherence.shape[0], n_missing, num_agents), dtype=data_adherence.dtype)
                bulk_default_obst_coll_msk = np.ones((obst_coll_msks.shape[0], n_missing, num_agents, obst_coll_msks.shape[-1]), dtype=obst_coll_msks.dtype)

                # Insert all missing entries in a single operation for each array
                collision_masks = np.insert(collision_masks, sorted_missing_ids, bulk_default_coll_msk, axis=1)
                data_adherence = np.insert(data_adherence, sorted_missing_ids, bulk_default_data_adh, axis=1)
                obst_coll_msks = np.insert(obst_coll_msks, sorted_missing_ids, bulk_default_obst_coll_msk, axis=1)
                assert collision_masks.shape[1] == data_adherence.shape[1] == obst_coll_msks.shape[1] == n_total_inits, \
                    f"After inserting missing data, inconsistent Init dimension sizes: " \
                    f"collision_masks {collision_masks.shape[1]}, data_adherence {data_adherence.shape[1]}, " \
                    f"obst_coll_msks {obst_coll_msks.shape[1]}, all_state_trjs {n_total_inits}." 

        summary_stats = {"Directory": data_dir, **expt_params}        
        summary_stats.update(
            analyze_collisions(
                collision_masks, 
                config, 
                obstacle_collision_masks=obst_coll_msks
            )
        )
        summary_stats.update(analyze_rewards(data_adherence))
        summary_stats.update(analyze_velocity(all_vel, config, expt_params, save_dir))
        
        final_df = aggregate_and_save_stats(num_agents, summary_stats, data_abs_dir, config)
        success_cnt += 1

    return success_cnt


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Run trajectory analysis from a config file.")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = dirname(dirname(script_dir))
    cfg_dir = join(base_dir, 'configs')

    cfg_pth  = os.path.join(cfg_dir, args.config) 
    assert os.path.exists(cfg_pth), f"Configuration file not found: {cfg_pth}"

    with open(cfg_pth, 'r') as f:
        config = yaml.safe_load(f)

    data_dirs = []
    for log_pth in tqdm(config['log_paths'], desc="Processing Directories", unit="dir"):
        if not os.path.isabs(log_pth) and not log_pth.startswith('~'):
            log_pth = os.path.join(base_dir, log_pth)
        with open(log_pth, 'r') as f:
            logged_dirs = [line.strip() for line in f if line.strip()]
            data_dirs.extend(logged_dirs)
    
    success_cnt = 0
    for data_dir in tqdm(data_dirs, desc="Processing Directories", unit="dir"):
        print("\n" + "="*80)
        logger.info(f"Processing Directory: {data_dir}")
        print("="*80)
        if not os.path.isabs(data_dir) and not data_dir.startswith('~'):
            data_dir = os.path.join(base_dir, data_dir)
        try:
            success_cnt = run_analysis_for_directory(data_dir, config, base_dir=base_dir)
        except Exception as e:
            logger.error(f"\nERROR processing directory '{data_dir}': \n{e}")
            logger.info("Skipping to next directory.\n")
            continue

    if success_cnt == 0:
        logger.warning("No successful analyses were performed. Check the data directories and configuration.")
    else:
        logger.info(f"Successfully processed {success_cnt} directories.")

if __name__ == '__main__':
    main()
