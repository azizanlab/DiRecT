import numpy as np
import os
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Local imports from our new helper files
from analysis_helpers import extract_params_from_path
from plotting_utils import (
    plot_trajectory_tuples, plot_collision_masks, 
    plot_velocity_stats, plot_velocity_histogram
)
# Assuming diffusion_policy is in the python path
from diffusion_policy.common.eval_util import (
    collision_detect, calculate_per_segment_safety,
    dtw_distance, frechet_distance
)

def load_data(data_abs_dir: str) -> Dict[str, np.ndarray]:
    """Loads raw data from the .npz file."""
    raw_data_path = os.path.join(data_abs_dir, "raw_data.npz")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at: {raw_data_path}")
    logger.info(f"Loading data from: {raw_data_path}")
    return np.load(raw_data_path)

def analyze_and_plot_trajectories(all_actions: np.ndarray, config: Dict, save_dir: str, title_args: Dict, group_size: int = 2):
    """Plots trajectory tuples based on config settings."""
    if not config['switches']['plot_trajectories']:
        return

    logger.info("Plotting trajectory tuples...")
    figs_axs = plot_trajectory_tuples(
        all_actions[..., :10],
        tup_size=group_size,
        h_start=0, h_end=config['plotting']['traj_plot_h_end'],
        title="Action Trajectory Tuples",
        title_args=title_args,
        cmap_name='plasma',
        xlim=tuple(config['plotting']['traj_plot_xlim']),
        ylim=tuple(config['plotting']['traj_plot_ylim']),
        figsize=(10, 10)
    )

    if config['switches']['save_plots']:
        logger.info(f"Saving {len(figs_axs)} trajectory plots to '{save_dir}'...")
        for i, (fig, _) in tqdm(enumerate(figs_axs), total=len(figs_axs)):
            fig.savefig(os.path.join(save_dir, f"init_st{i:03d}_action_trj_tups.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)

def analyze_collisions(roll_actions: np.ndarray, config: Dict, save_dir: str) -> Dict[str, Any]:
    """Performs collision detection and calculates safety statistics."""
    logger.info("Analyzing collisions...")
    constants = config['constants']
    collision_masks = collision_detect(
        roll_actions, safe_dist=constants['SAFE_DIST'], norm_order=2
    ).transpose(0, 2, 1, 3)  # (I, P, N, H)

    safe_masks = np.logical_not(collision_masks)
    per_segment_mask = calculate_per_segment_safety(safe_masks, constants['ACTION_HORIZON'])
    
    final_segment_safe_rate = np.mean(per_segment_mask.astype(float), axis=(-1, -2))
    overall_safe_rate = final_segment_safe_rate.mean()
    collision_counts = np.sum(collision_masks.astype(int), axis=-1)

    logger.info(f"Overall safe rate: {overall_safe_rate:.2%}")

    if config['switches']['plot_trajectories']:
        fig, _ = plot_collision_masks(collision_masks[:10, ...], constants['SAFE_DIST'], safe_rates=final_segment_safe_rate)
        if config['switches']['save_plots']:
            fig.savefig(os.path.join(save_dir, "collision_masks.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)
        
    return {
        'Overall Mean Collision Count': collision_counts.mean(),
        'Overall Mean Safe Rate': overall_safe_rate,
    }

def compute_distance_metrics(roll_actions: np.ndarray, config: Dict, save_dir: str) -> Dict[str, Any]:
    """Computes DTW and Frechet distances, plots histograms, and returns stats."""
    logger.info("Computing DTW and Frechet distances...")
    constants = config['constants']
    action_horizon = constants['ACTION_HORIZON']
    I, N, P, H, _ = roll_actions.shape
    n_seg = (H + action_horizon - 1) // action_horizon # Number of segments
    split_indices = range(action_horizon, roll_actions.shape[-2], action_horizon)
    action_segments = np.array(np.split(roll_actions[:, :, ...], split_indices, axis=-2))  # [(I, N, P, a_h, 2)]*n_seg
    action_segments = np.stack(action_segments, axis=-3).transpose(2, 0, 1, 3, 4, 5) # [I, N, P, n_seg, a_h, 2] -> (P, I, N, n_seg, a_h, 2)
    assert action_segments.shape == (P, I, N, n_seg, action_horizon, 2), \
        f"Action segments shape mismatch: expect [P, I, N, n_seg, a_h, 2] ({(P, I, N, n_seg, action_horizon, 2)}) but got {action_segments.shape}"

    a_max = np.array(constants['MAX_ACTION'])
    a_min = np.array(constants['MIN_ACTION'])
    norm_segments = (action_segments - a_min) / (a_max - a_min + constants['EPS']) * 2 - 1.0

    # Compute pairwise DTW and Frechet distances for all C(P,2) player pairs
    from itertools import combinations
    all_dtw = []
    all_frechet = []
    for pi, pj in combinations(range(P), 2):
        dtw_ij = dtw_distance(norm_segments[pi], norm_segments[pj], norm_type='raw')  # [I, N, n_seg]
        fre_ij = frechet_distance(norm_segments[pi], norm_segments[pj])               # [I, N, n_seg]
        all_dtw.append(dtw_ij)
        all_frechet.append(fre_ij)
    all_dtw = np.stack(all_dtw, axis=0)       # (n_pairs, I, N, n_seg)
    all_frechet = np.stack(all_frechet, axis=0)

    # Aggregate across segments -> (n_pairs, I, N)
    dtw_full_per_pair = all_dtw.mean(axis=-1)
    frechet_full_per_pair = all_frechet.mean(axis=-1)

    # Min and mean across pairs -> (I, N)
    dtw_full_min = dtw_full_per_pair.min(axis=0)
    dtw_full_mean = dtw_full_per_pair.mean(axis=0)
    frechet_full_min = frechet_full_per_pair.min(axis=0)
    frechet_full_mean = frechet_full_per_pair.mean(axis=0)

    # Use mean-across-pairs as the primary metric (identical to old behavior for P=2)
    dtw_full = dtw_full_mean
    frechet_full = frechet_full_mean
    # For histograms, use mean-across-pairs per-segment
    dtw_dists = all_dtw.mean(axis=0)       # (I, N, n_seg)
    frechet_dists = all_frechet.mean(axis=0)

    overall_mean_dtwd = np.mean(dtw_full)
    overall_mean_fd = np.mean(frechet_full)
    overall_dtw_std = dtw_full.std()
    within_init_dtw_std = dtw_full.std(axis=1).mean()
    between_init_dtw_std = dtw_full.mean(axis=1).std()
    overall_fd_std = frechet_full.std()
    within_init_fd_std = frechet_full.std(axis=1).mean()
    between_init_fd_std = frechet_full.mean(axis=1).std()
    logger.info(f"Overall Mean DTW: {overall_mean_dtwd:.4f}\u00b1{overall_dtw_std:.4f}, Overall Mean Frechet: {overall_mean_fd:.4f}\u00b1{overall_fd_std:.4f}")

    if config['switches']['plot_trajectories']:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle("Distance Metrics Between Paired Trajectories")
        bins = np.linspace(0, np.sqrt(2) * 2, 142)
        
        axes[0].hist(dtw_dists.flatten(), bins=bins, color='skyblue', alpha=0.8, density=True)
        axes[0].axvline(overall_mean_dtwd, color='blue', linestyle='--', label=f'Mean DTW={overall_mean_dtwd:.3f}')
        axes[0].set(title="Histogram of DTW Distances", xlabel="DTW", ylabel="Density")
        axes[0].legend()
        axes[0].grid(True, linestyle='--')
        
        axes[1].hist(frechet_dists.flatten(), bins=bins, color='salmon', alpha=0.8, density=True)
        axes[1].axvline(overall_mean_fd, color='red', linestyle='--', label=f'Mean DFD={overall_mean_fd:.3f}')
        axes[1].set(title="Histogram of Discrete Fréchet Distances", xlabel="DFD", ylabel="Density")
        axes[1].legend()
        axes[1].grid(True, linestyle='--')
        
        if config['switches']['save_plots']:
            fig.savefig(os.path.join(save_dir, "dtw_frechet_histogram.png"), bbox_inches='tight', dpi=400)
        plt.close(fig)

    result = {
        'Overall Mean DTW Distance': overall_mean_dtwd,
        'Overall Mean Frechet Distance': overall_mean_fd,
        'Overall Std DTW Distance': overall_dtw_std,
        'Overall Std Frechet Distance': overall_fd_std,
        'Within Init Std DTW Distance': within_init_dtw_std,
        'Within Init Std Frechet Distance': within_init_fd_std,
        'Between Init Std DTW Distance': between_init_dtw_std,
        'Between Init Std Frechet Distance': between_init_fd_std,
    }
    # Add min-across-pairs metrics (relevant when group_size > 2)
    if P > 2:
        result['Overall Min-Pair Mean DTW Distance'] = float(np.mean(dtw_full_min))
        result['Overall Min-Pair Mean Frechet Distance'] = float(np.mean(frechet_full_min))
    return result

def analyze_rewards(raw_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Calculates and returns reward statistics."""
    logger.info("Analyzing rewards...")
    all_rewards = raw_data['all_rewards'].transpose(0, 3, 1, 2)  # (P, I, N, H)
    P, I, N, H = all_rewards.shape
    max_rewards = np.max(all_rewards, axis=-1)
    assert max_rewards.shape == (P, I, N), \
        f"Max rewards shape mismatch: expect [P, I, N] ({(P, I, N)}) but got {max_rewards.shape}"
    mean_rewards_per_player = np.mean(max_rewards, axis=(-1, -2))
    overall_mean_rewards = np.mean(mean_rewards_per_player)
    std_rewards_per_player = np.std(max_rewards, axis=(-1, -2))
    overall_std_rewards = np.std(max_rewards)
    within_init_std_rewards_per_player = max_rewards.std(axis=2).mean(1)
    assert within_init_std_rewards_per_player.shape == (P,), \
        f"Within init std rewards shape mismatch: expect [P] ({(P,)}) but got {within_init_std_rewards_per_player.shape}"
    ##############################################################################
    between_init_std_rewards_per_player = max_rewards.mean(axis=2).std(1)
    assert between_init_std_rewards_per_player.shape == (P,), \
        f"Between init std rewards shape mismatch: expect [P] ({(P,)}) but got {between_init_std_rewards_per_player.shape}"

    max_rewards_all = max_rewards.mean(0)  # (I, N)
    assert max_rewards_all.shape == (I, N), \
        f"Max rewards all shape mismatch: expect [I, N] ({(I, N)}) but got {max_rewards_all.shape}"
    overall_within_init_std_rewards = max_rewards_all.std(axis=1).mean()
    overall_between_init_std_rewards = max_rewards_all.mean(axis=1).std()
    
    logger.info(f"Mean rewards per player: {np.round(mean_rewards_per_player, 4)}\u00b1{np.round(std_rewards_per_player, 4)}")
    logger.info(f"Overall mean reward: {overall_mean_rewards:.4f}\u00b1{overall_std_rewards:.4f}")
    
    return {
        'Overall Mean Rewards': overall_mean_rewards,
        'Overall Std Rewards': overall_std_rewards, 
        'Within Init Std Rewards': overall_within_init_std_rewards,
        'Between Init Std Rewards': overall_between_init_std_rewards,
        'Mean Paired Rewards': tuple(mean_rewards_per_player.tolist()),
        'Std Paired Rewards': tuple(std_rewards_per_player.tolist()),
        'Within Init Std Paired Rewards': tuple(within_init_std_rewards_per_player.tolist()),
        'Between Init Std Paired Rewards': tuple(between_init_std_rewards_per_player.tolist()),
        'Episode Length': all_rewards.shape[-1],
    }

def analyze_velocity(raw_data: Dict[str, np.ndarray], config: Dict, expt_params: Dict, save_dir: str):
    """Calculates velocity, profiles it, and plots statistics."""
    logger.info("Analyzing velocity...")
    constants = config['constants']
    all_actions = raw_data['all_roll_action_trjs']
    all_state_trjs = raw_data['all_state_trjs']

    P, N, H, _, I = all_actions.shape
    v_eps = constants['VEL_TOL']
    dt = constants['DT']
    a_max = np.array(constants['MAX_ACTION'])
    a_min = np.array(constants['MIN_ACTION'])
    action_horizon = constants['ACTION_HORIZON']
    n_seg = (H + action_horizon - 1) // action_horizon  # Number of segments

    # Observations are the positions from the state trajectories
    obs_trjs = all_state_trjs[..., :-1, :2, :].transpose(4, 1, 0, 2, 3) # (I,N,P,H,2)
    obs_trjs = obs_trjs.reshape(I*N*P, H, 2)
    obs_segments = np.stack(np.array_split(obs_trjs, H // action_horizon, axis=1), axis=1) # (I*N*P, n_seg, a_h, 2)
    norm_obs_segments = 2 * (obs_segments - a_min) / (a_max - a_min + constants['EPS']) - 1
    assert norm_obs_segments.shape == (I*N*P, n_seg, action_horizon, 2), \
        f"Observation segments shape mismatch: expect [I*N*P, n_seg, a_h, 2] ({(I*N*P, n_seg, action_horizon, 2)}) but got {norm_obs_segments.shape}"
    
    # Actions
    norm_actions = all_actions.transpose(4,1,0,2,3).reshape(I*N*P, H, 2)
    norm_actions = 2 * (norm_actions - a_min) / (a_max - a_min + constants['EPS']) - 1
    norm_action_segments = np.stack(np.array_split(norm_actions, H // action_horizon, axis=1), axis=1) # (I*N*P, n_seg, a_h, 2)
    assert norm_action_segments.shape == (I*N*P, n_seg, action_horizon, 2), \
        f"Action segments shape mismatch: expect [I*N*P, n_seg, a_h, 2] ({(I*N*P, n_seg, action_horizon, 2)}) but got {norm_action_segments.shape}"

    # Displacements
    norm_displacement_segments = np.concatenate([norm_obs_segments[:, :, :1, :], norm_action_segments], axis=-2)
    dx = np.diff(norm_displacement_segments, axis=-2) # (I*N*P, n_seg, a_h, 2)
    # Calculate velocity
    v = np.linalg.norm(dx, axis=-1) / dt  # (I*N*P, n_seg, a_h)
    assert v.shape == (I*N*P, n_seg, action_horizon), \
        f"Velocity shape mismatch: expect [I*N*P, n_seg, a_h] ({(I*N*P, n_seg, action_horizon)}) but got {v.shape}"
    
    ## Check if velocity exceeds max threshold segment-wise
    max_vel = expt_params.get('max_vel')
    if max_vel is None:
        max_vel = constants['DEFAULT_MAX_VEL']
        logger.warning(f"Max velocity not found in experiment setting, using default max velocity: {max_vel}.")
    if abs(max_vel - constants['DEFAULT_MAX_VEL']) > 1e-6:
        logger.warning(f"Max velocity ({max_vel}) differs from default ({constants['DEFAULT_MAX_VEL']}).")

    ## Velocity constraint satisfaction
    vel_constr_sat = v <= max_vel + v_eps
    vel_constr_sat_seg:np.ndarray = np.all(vel_constr_sat, axis=-1)  # (I*N*P, n_seg)
    vel_constr_sat_seg = vel_constr_sat_seg.reshape(I*N, P, -1) # (I*N, P, n_seg)
    mean_vel_constr_sat_per_player = np.mean(vel_constr_sat_seg.astype(float), axis=-1)  # (I*N, P)
    overall_mean_vel_constr_sat = np.mean(mean_vel_constr_sat_per_player)
    overall_std_vel_constr_sat_per_player = np.std(mean_vel_constr_sat_per_player, axis=0)  # (P,)
    
    vel_constr_sat_full_per_player = vel_constr_sat_seg.reshape(I, N, P, -1).mean(-1)  # (I, N, P, n_seg) -> (I, N, P)
    assert vel_constr_sat_full_per_player.shape == (I, N, P), \
        f"Velocity constraint satisfaction full shape mismatch: expect [I, N, P] ({(I, N, P)}) but got {vel_constr_sat_full_per_player.shape}"
    within_init_std_vel_constr_sat_per_player = vel_constr_sat_full_per_player.std(axis=1).mean(0) # (P,)
    between_init_std_vel_constr_sat_per_player = vel_constr_sat_full_per_player.mean(axis=1).std(0) # (P,)
    assert within_init_std_vel_constr_sat_per_player.shape == (P,), \
        f"Mean velocity constraint satisfaction shape mismatch: expect [P] ({(P)}) but got {within_init_std_vel_constr_sat_per_player.shape}"
    assert between_init_std_vel_constr_sat_per_player.shape == (P,), \
        f"Between init std velocity constraint satisfaction shape mismatch: expect [P] ({(P,)}) but got {between_init_std_vel_constr_sat_per_player.shape}"
    
    vel_constr_sat_full = vel_constr_sat_full_per_player.mean(axis=-1)  # (I, N, P) -> (I, N)
    assert vel_constr_sat_full.shape == (I, N), \
        f"Velocity constraint satisfaction full shape mismatch: expect [I, N] ({(I, N)}) but got {vel_constr_sat_full.shape}"
    overall_std_vel_constr_sat_full = vel_constr_sat_full.std()
    within_init_std_vel_constr_sat_full = vel_constr_sat_full.std(axis=1).mean()
    between_init_std_vel_constr_sat_full = vel_constr_sat_full.mean(axis=1).std()
    


    # Plotting
    v = v.reshape(-1, v.shape[-1] * v.shape[-2]) # (I*N*P, H // action_horizon * 2)
    if config['switches']['plot_velocity_stats']:
        hard_deck = max_vel
        proj_method = expt_params.get('projection_method')
        if proj_method is None:
            proj_method = "N/A"
        title = f"(Normalized) Velocity Stats (Proj: {proj_method})"
        subtitle = f"cfg: {expt_params.get('diffusion_config', 'N/A')}; ep_len={expt_params.get('episode_length', 'N/A')}"
        
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
        'Overall Std Velocity Constraint Satisfaction': overall_std_vel_constr_sat_full, 
        'Mean Velocity Constraint Satisfaction per Player': tuple(mean_vel_constr_sat_per_player.mean(axis=0).tolist()),
        'Std Velocity Constraint Satisfaction per Player': tuple(overall_std_vel_constr_sat_per_player.tolist()),
        'Within Init Std Velocity Constraint Satisfaction per Player': tuple(within_init_std_vel_constr_sat_per_player.tolist()),
        'Between Init Std Velocity Constraint Satisfaction per Player': tuple(between_init_std_vel_constr_sat_per_player.tolist()),
        'Within Init Std Velocity Constraint Satisfaction': within_init_std_vel_constr_sat_full, 
        'Between Init Std Velocity Constraint Satisfaction': between_init_std_vel_constr_sat_full,
        'Max Velocity': max_vel, 
    }

def aggregate_and_save_stats(stats: Dict, data_dir: str, config: Dict):
    """Aggregates all computed stats into a DataFrame and saves to JSON."""
    df = pd.DataFrame([stats])
    df.index = [os.path.basename(data_dir)] # Use folder name as index
    
    if config['switches']['save_stats_json']:
        save_path = os.path.join(data_dir, f"summary_stats_vmax{stats['Max Velocity']}.json")
        logger.info(f"Saving summary stats to: {save_path}")
        df.to_json(save_path, orient='records', indent=4)
    
    return df

def run_analysis_for_directory(data_dir: str, config: Dict) -> pd.DataFrame:
    """Main pipeline for processing a single data directory."""
    basedir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_abs_dir = os.path.join(basedir, data_dir)
    save_dir = os.path.join(data_abs_dir, "figs")
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract parameters from path
    expt_params = extract_params_from_path(data_dir)
    title_args = {'stp': expt_params.get('stp'), 'scl': expt_params.get('scl')}

    # Load Data
    raw_data = load_data(data_abs_dir)
    all_actions = raw_data['all_roll_action_trjs']  # (P, N, H, 2, I)
    group_size = all_actions.shape[0]  # P = group_size, inferred from data
    roll_actions = all_actions.transpose(4, 1, 0, 2, 3)  # (I, N, P, H, 2)

    # --- Run Analysis Modules ---
    summary_stats = {"Directory": data_dir, "group_size": group_size, **expt_params}
    analyze_and_plot_trajectories(all_actions, config, save_dir, title_args, group_size=group_size)
    summary_stats.update(compute_distance_metrics(roll_actions, config, save_dir))
    summary_stats.update(analyze_rewards(raw_data))
    summary_stats.update(analyze_velocity(raw_data, config, expt_params, save_dir))

    # --- Policy Timing ---
    if 'policy_time_mean' in raw_data:
        summary_stats['Policy Time Mean (s)'] = float(raw_data['policy_time_mean'])
        summary_stats['Policy Time Std (s)'] = float(raw_data['policy_time_std'])
        summary_stats['Policy Calls'] = int(len(raw_data['policy_times']))
        logger.info(f"Policy call time: {raw_data['policy_time_mean']:.4f} ± {raw_data['policy_time_std']:.4f} s")

    # --- Aggregate and Save ---
    final_df = aggregate_and_save_stats(summary_stats, data_abs_dir, config)
    return final_df


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

    # Determine base_dir as the grandparent of this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    cfg_pth  = os.path.join(base_dir, args.config) 
    assert os.path.exists(cfg_pth), f"Configuration file not found: {cfg_pth}"

    # Load configuration
    with open(cfg_pth, 'r') as f:
        config = yaml.safe_load(f)

    # Process each data directory specified in the config
    ## Read log paths and data directories from config
    data_dirs = []
    for log_pth in tqdm(config['log_paths'], desc="Processing Directories", unit="dir"):
        with open(os.path.join(base_dir, log_pth), 'r') as f:
            logged_dirs = [line.strip() for line in f if line.strip()]
            data_dirs.extend(logged_dirs)
    
    all_stats_df = []
    for data_dir in tqdm(data_dirs, desc="Processing Directories", unit="dir"):
        print("\n" + "="*80)
        logger.info(f"Processing Directory: {data_dir}")
        print("="*80)
        data_dir = os.path.join(base_dir, data_dir)  # Ensure absolute path
        try:
            stats_df = run_analysis_for_directory(data_dir, config)
            all_stats_df.append(stats_df)
        except Exception as e:
            logger.error(f"\nERROR processing directory '{data_dir}': \n{e}")
            logger.info("Skipping to next directory.\n")
            continue
            
    # Display final aggregated stats
    if all_stats_df:
        final_summary_df = pd.concat(all_stats_df)
        print("\n" + "="*80)
        print("          FINAL AGGREGATED STATISTICS         ")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(final_summary_df)
    else:
        print("\nNo data was processed successfully.")

if __name__ == '__main__':
    main()
