import numpy as np
import torch
import os
import pandas as pd
import yaml
import argparse
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from os.path import join, dirname
from typing import Dict, Any, Tuple, Callable

from torch_robotics.tasks.tasks import PlanningTask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from helpers import (
    extract_params_from_config,
    plot_trajectory_tuples, plot_collision_masks,
    plot_velocity_stats, plot_velocity_histogram,
)
from mmd.common.argparse import load_config

TENSOR_ARGS = {
        'dtype': torch.float32,
        'device': 'cpu'
    }

def load_data(data_abs_dir: str) -> Tuple[Dict[str, np.ndarray]]:
    """Loads raw data from the .npz file."""
    raw_data_path = os.path.join(data_abs_dir, "raw_data.npz")
    processed_data_path = os.path.join(data_abs_dir, "processed_data.npz")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at: \'{raw_data_path}\'")
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at: \'{processed_data_path}\'")

    print(f"Loading data from: {raw_data_path}")
    raw_data = np.load(raw_data_path, allow_pickle=True)
    processed_data = np.load(processed_data_path, allow_pickle=True)

    return raw_data, processed_data

def get_planning_dataset(expt_params: Dict[str, Any], config: Dict[str, Any]) -> Tuple[Any, Any]:
    from mmd.utils.loading import load_params_from_yaml
    from mmd.trainer import get_dataset

    trained_models_dir = expt_params.get('data_trained_models_dir')
    if not trained_models_dir:
        trained_models_dir = config['data']['default_trained_models_dir']
        logger.warning(
            "No trained models directory specified in experiment parameters. "
            f"Using default: \'{trained_models_dir}\'."
        )

    args = load_params_from_yaml(os.path.join(trained_models_dir, "args.yaml"))
    trainset, _, _, _ = get_dataset(
            dataset_class='TrajectoryDataset',
            use_extra_objects=False,
            obstacle_cutoff_margin=0.05,
            **args,
            tensor_args=TENSOR_ARGS
        )
    return trainset.dataset

def get_planning_task(expt_params: Dict[str, Any], config:Dict[str, Any]) -> PlanningTask:
    trainset = get_planning_dataset(expt_params, config)
    task:PlanningTask = trainset.dataset.task
    return task



def analyze_and_plot_trajectories(all_actions: np.ndarray, config: Dict, save_dir: str, title_args: Dict):
    """Plots trajectory tuples based on config settings."""
    if not config['switches']['plot_trajectories']:
        return

    print("Plotting trajectory tuples...")
    n_trj_tups_plot = config['plotting']['num_trj_tups_plot']
    n_inits_plot = config['plotting']['num_inits_plot']
    figs_axs = plot_trajectory_tuples(
        all_actions[:n_trj_tups_plot, :n_inits_plot, ...],
        tup_size=2,
        h_start=0, h_end=config['plotting']['traj_plot_h_end'],
        title="Robot Trajectory Tuples",
        title_args=title_args,
        cmap_name='plasma',
        xlim=tuple(config['plotting']['traj_plot_xlim']),
        ylim=tuple(config['plotting']['traj_plot_ylim']),
        figsize=(10, 10)
    )

    if config['switches']['save_plots']:
        print(f"Saving {len(figs_axs)} trajectory plots to '{save_dir}'...")
        for i, (fig, _) in tqdm(enumerate(figs_axs), total=len(figs_axs)):
            fig.savefig(os.path.join(save_dir, f"init_st{i:03d}_action_trj_tups.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)

def analyze_costs(
    all_pos_trjs:np.ndarray,
    expt_params:Dict,
):
    print("Analyzing costs...")
    from mmd.coupling_costs.cost_function_implementation import cost_registry
    from mmd.coupling_costs.coupling_cost_functions import RobotCollisionCost

    cost_fn_name = expt_params["cost_function"]
    if cost_fn_name == 'vanilla':
        logger.info("Using 'vanilla' cost function, which is a no-op. Returning NaN.")
        return {
            'Overall Mean Cost': np.nan,
            'Overall Std Cost': np.nan,
            'Within Init Std Cost': np.nan,
            'Between Init Std Cost': np.nan,
        }
    else:
        assert cost_fn_name in cost_registry, f"Cost function '{cost_fn_name}' not found in registry."
    robot_radius = expt_params['robot_radius']
    cost_func = RobotCollisionCost(
        robot_radius=robot_radius,
        cost_func_key=cost_fn_name,
        **expt_params.get('cost_func_param', {})
    )

    N, I, P, H, D = all_pos_trjs.shape
    # Reshape all_pos_trjs so that dim 0 is N*P, preserving I and stacking P
    pos_trjs_reshape = all_pos_trjs.transpose(0, 2, 1, 3, 4).reshape(N * P, I, H, D)
    costs:torch.Tensor = cost_func.eval(
        torch.from_numpy(pos_trjs_reshape).to(**TENSOR_ARGS),
        group_size=P,
    ).to('cpu').numpy() # (N, I)
    assert costs.ndim == 2
    assert costs.shape == (N, I), f"Expected costs shape (N, I), got {costs.shape}."

    # Intra-group variance E[Var(cost | init)]
    cost_within_init_std = costs.std(axis=0).mean()  # Mean std across inits
    cost_between_inits_std = costs.mean(axis=0).std()  # Std of means across inits

    return {
        'Overall Mean Cost': costs.mean(),
        'Overall Std Cost': costs.std(),
        'Within Init Std Cost': cost_within_init_std,
        'Between Init Std Cost': cost_between_inits_std,
    }

def analyze_collisions(
    all_pos_trjs:np.ndarray, robot_collision_masks: np.ndarray,
    config: Dict,
    expt_params:Dict,
    save_dir: str,
    obstacle_collision_masks: np.ndarray = None
) -> Dict[str, Any]:
    """Performs collision detection and calculates safety statistics."""
    print("Analyzing collisions...")
    constants = config['constants']
    # collision_masks = collision_detect(
    #     roll_actions, safe_dist=constants['SAFE_DIST'], norm_order=2
    # ).transpose(0, 2, 1, 3)  # (I, P, N, H)

    ## ----------------------- Inter-robot Collision Detection --------------------------------
    N, I, P, _, H = robot_collision_masks.shape
    safe_masks = np.logical_not(robot_collision_masks)  # (N, I, P, P, H)
    trj_safe_masks:np.ndarray = safe_masks.all(-1)  # (N, I, P, P)

    loose_safe_rates = (trj_safe_masks.astype(float).sum(axis=(-1, -2)) - P) \
        / (P * (P - 1))  # (N, I)
    assert loose_safe_rates.shape == (N, I), \
        f"Expected loose safe rates shape [N, I]({N, I}) , got {loose_safe_rates.shape}."
    overall_loose_safe_rate = loose_safe_rates.mean()
    overall_std_loose_safe_rt = loose_safe_rates.std()  # Overall pooled std
    within_init_std_loose_safe_rt = loose_safe_rates.std(axis=0).mean()  # Mean std across inits
    between_inits_std_loose_safe_rt = loose_safe_rates.mean(axis=0).std()  # Std of means across inits

    strict_safe_masks = trj_safe_masks.all(axis=(-1, -2))      # (N, I,), whether all agents are collision-free with each other
    assert strict_safe_masks.shape == (N, I), \
        f"Expected strict safe masks shape [N, I]({N, I}), got {strict_safe_masks.shape}."
    overall_strict_safe_rate = strict_safe_masks.astype(float).mean()  # Overall safe rate
    overall_std_strict_safe_rt = strict_safe_masks.astype(float).std()  # Overall pooled std
    within_init_std_strict_safe_rt = strict_safe_masks.astype(float).std(axis=0).mean()  # Mean std across inits
    between_inits_std_strict_safe_rt = strict_safe_masks.astype(float).mean(axis=0).std()  # Std of means across inits

    collision_counts:np.ndarray = np.sum(robot_collision_masks.astype(int), axis=(-1, -2, -3)) // 2  # (N, I)
    assert collision_counts.shape == (N, I), \
        f"Expected collision counts shape [N, I]({N, I}), got {collision_counts.shape}."
    collision_counts_std = np.std(collision_counts.flatten())
    coll_within_init_std = collision_counts.std(axis=0).mean()  # Mean std across inits
    coll_between_inits_std = collision_counts.mean(axis=0).std()  # Std of means across inits

    print(f"Overall strict safe rate: {overall_strict_safe_rate:.2%}\u00b1{overall_std_strict_safe_rt:.2%}")
    print(f"Overall loose safe rate: {overall_loose_safe_rate:.2%}\u00b1{overall_std_loose_safe_rt:.2%}")

    ## ----------------------- Static Obstacle Collision Detection --------------------------------
    if obstacle_collision_masks is None or not isinstance(obstacle_collision_masks, np.ndarray):
        N, I, P, H, _ = all_pos_trjs.shape
        task:PlanningTask = get_planning_task(expt_params, config)
        trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_collisions = \
            task.get_trajs_collision_and_free(
                torch.from_numpy(all_pos_trjs).reshape(N*I*P, H, -1),  # Reshape to (N*I*P, H, D)
                return_indices=True,
                num_interpolation=0,
            )
        obstacle_collision_masks:np.ndarray = trajs_waypoints_collisions.to('cpu').numpy().reshape(N, I, P, H)  # (N, I, P, H)

    static_obstacle_safe_rates_per_agent:np.ndarray = np.logical_not(obstacle_collision_masks.any(-1))  # (N, I, P)
    static_obstacle_safe_rates:np.ndarray = static_obstacle_safe_rates_per_agent.astype(float).mean(axis=-1)  # (N, I)
    overall_static_obstacle_safe_rate = static_obstacle_safe_rates.mean()  # Overall safe rate across all batches & inits
    overall_std_static_obst_safe_rt = static_obstacle_safe_rates.std()  # Overall pooled std
    within_init_std_static_obst_safe_rt = static_obstacle_safe_rates.std(axis=0).mean()  # Mean std across inits
    between_inits_std_static_obst_safe_rt = static_obstacle_safe_rates.mean(axis=0).std()  # Std of means across inits

    print(f"Overall static obstacle safe rate: {overall_static_obstacle_safe_rate:.2%}\u00b1{overall_std_static_obst_safe_rt:.2%}")

    ## ----------------------- All Safe rate   --------------------------------
    obstacle_safe_masks = np.logical_not(obstacle_collision_masks.any(axis=(-1, -2)))  # (N, I,)
    assert obstacle_safe_masks.shape == (N, I), \
        f"Expected obstacle safe masks shape [N, I]({N, I}), got {obstacle_safe_masks.shape}."
    all_safe_masks = np.logical_and(
        strict_safe_masks,        # (N, I,) whether this set of trjs is collision-free with all other agents
        obstacle_safe_masks,      # (N, I) whether this set of trjs is collision-free with all obstacles
    )  # (N, I, )
    assert all_safe_masks.shape == (N, I), \
        f"Expected all safe masks shape [N, I]({N, I}), got {all_safe_masks.shape}."
    overal_safe_rates = all_safe_masks.astype(float).mean()  # Overall safe rate across all batches & inits
    overall_std_safe_rt = all_safe_masks.astype(float).std()  # Overall pooled std
    within_init_std_safe_rt = all_safe_masks.astype(float).std(axis=0).mean()  # Mean std across inits
    between_inits_std_safe_rt = all_safe_masks.astype(float).mean(axis=0).std()  # Std of means across inits
    print(f"Overall safe rate: {overal_safe_rates:.2%}")

    best_success_masks = all_safe_masks.astype(float).max(axis=0)  # (I,) whether at least one batch is collision-free for each init
    success_rate = best_success_masks.mean()  # Overall success rate across all inits
    std_success_rate = best_success_masks.std()  # Overall pooled std
    print(f"Overall success rate: {success_rate:.2%}\u00b1{std_success_rate:.2%}")

    ## ----------------------- Plotting Collision Masks --------------------------------
    if config['switches']['plot_trajectories']:
        # n_init_plots = config['plotting']['num_inits_plot']
        # # n_trj_tups = config['plotting']['num_trj_tups_plot']
        # fig, _ = plot_collision_masks(collision_masks[:n_trj_tups, :n_init_plots, ...], 2*expt_params['robot_radius'], safe_rates=strict_safe_rates)
        # if config['switches']['save_plots']:
        #     fig.savefig(os.path.join(save_dir, "collision_masks.png"), bbox_inches='tight', dpi=300)
        # plt.close(fig)
        pass
    return {
        'Overall Mean Collision Count': collision_counts.mean(),
        'Overall Std Collision Count': collision_counts_std,
        'Within Init Std Collision Count': coll_within_init_std,
        'Between Init Std Collision Count': coll_between_inits_std,
        'Overall Mean Strict Inter-Agent Safe Rate': overall_strict_safe_rate,
        'Overall Std Strict Inter-Agent Safe Rate': overall_std_strict_safe_rt,
        'Within Init Std Strict Inter-Agent Safe Rate': within_init_std_strict_safe_rt,
        'Between Init Std Strict Inter-Agent Safe Rate': between_inits_std_strict_safe_rt,
        'Overall Mean Loose Inter-Agent Safe Rate': overall_loose_safe_rate,
        'Overall Std Loose Inter-Agent Safe Rate': overall_std_loose_safe_rt,
        'Within Init Std Loose Inter-Agent Safe Rate': within_init_std_loose_safe_rt,
        'Between Init Std Loose Inter-Agent Safe Rate': between_inits_std_loose_safe_rt,
        'Overall Mean Static Obstacle Safe Rate': overall_static_obstacle_safe_rate,
        'Overall Std Static Obstacle Safe Rate': overall_std_static_obst_safe_rt,
        'Within Init Std Static Obstacle Safe Rate': within_init_std_static_obst_safe_rt,
        'Between Init Std Static Obstacle Safe Rate': between_inits_std_static_obst_safe_rt,
        'Overall Mean All Safe Rate': overal_safe_rates,
        'Overall Std All Safe Rate': overall_std_safe_rt,
        'Within Init Std All Safe Rate': within_init_std_safe_rt,
        'Between Init Std All Safe Rate': between_inits_std_safe_rt,
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

    avg_mean_rewards = data_adherences.mean(-1) # (N, I)
    assert avg_mean_rewards.shape == (N, I), \
        f"Expected average mean rewards shape (N, I), got {avg_mean_rewards.shape}."
    avg_mean_rewards_std = avg_mean_rewards.std()  # Overall pooled std
    within_init_rewards_std = avg_mean_rewards.std(axis=0).mean()  # Mean std across inits
    between_inits_rewards_std = avg_mean_rewards.mean(axis=0).std()  # Std of means across inits
    print(f"Overall mean data adherence: {overall_mean_rewards:.4f}\u00b1{avg_mean_rewards_std:.4f}")


    return {
        'Overall Agent Data Adherence': tuple(mean_rewards_per_player.tolist()),
        'Std Agent Data Adherence': tuple(std_rewards_per_player.tolist()),
        'Overall Mean Data Adherence': overall_mean_rewards,
        'Overall Std Data Adherence': avg_mean_rewards_std,
        'Within Init Std Data Adherence': within_init_rewards_std,
        'Between Init Std Data Adherence': between_inits_rewards_std,
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

    ## Velocity constraint satisfaction
    vel_constr_sat = vel <= max_vel + v_eps
    vel_constr_sat_seg:np.ndarray = np.all(vel_constr_sat, axis=-1)  # (N, I, P)
    mean_vel_constr_sat_per_player = np.mean(vel_constr_sat_seg.astype(float), axis=(0, 1))  # (P, )
    overall_mean_vel_constr_sat = np.mean(mean_vel_constr_sat_per_player)
    overall_std_vel_constr_sat_per_player = vel_constr_sat_seg.std(axis=(0, 1))

    trj_vel_constr_sat: np.ndarray = vel_constr_sat_seg.mean(-1)  # (N, I)
    assert trj_vel_constr_sat.shape == (N, I), \
        f"Expected trajectory velocity constraint satisfaction shape (N, I), got {trj_vel_constr_sat.shape}."
    overall_vel_constr_sat_std = trj_vel_constr_sat.std()  # Overall pooled std
    within_init_vel_constr_sat_std = trj_vel_constr_sat.std(axis=0).mean()  # Mean std across inits
    between_inits_vel_constr_sat_std = trj_vel_constr_sat.mean(axis=0).std()  # Std of means across inits

    print("Mean velocity constraint satisfaction per player: " + '(' +
        ", ".join([f"{v:.2%}" for v in mean_vel_constr_sat_per_player])
        + ')')
    print(f"Overall mean velocity constraint satisfaction: {overall_mean_vel_constr_sat:.2%}")

    # Plotting
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
        'Overall Std Velocity Constraint Satisfaction': overall_vel_constr_sat_std,
        'Within Init Std Velocity Constraint Satisfaction': within_init_vel_constr_sat_std,
        'Between Init Std Velocity Constraint Satisfaction': between_inits_vel_constr_sat_std,
        'Max Velocity': max_vel,
    }

def aggregate_and_save_stats(stats: Dict, data_dir: str, config: Dict):
    """Aggregates all computed stats into a DataFrame and saves to JSON."""
    df = pd.DataFrame([stats])
    df.index = [os.path.basename(data_dir)]

    if config['switches']['save_stats_json']:
        save_path = os.path.join(data_dir, f"summary_stats_vmax{stats['Max Velocity']}.json")
        print(f"Saving summary stats to: {save_path}")
        df.to_json(save_path, orient='records', indent=4)

    return df

def run_analysis_for_directory(data_dir: str, config: Dict, base_dir:str) -> pd.DataFrame:
    """Main pipeline for processing a single data directory."""
    data_abs_dir = data_dir if os.path.isabs(data_dir) or data_dir.startswith('~') else\
        os.path.join(base_dir, data_dir)
    save_dir = os.path.join(data_abs_dir, "figs")
    os.makedirs(save_dir, exist_ok=True)

    expt_params = extract_params_from_config(join(data_abs_dir, "run_config_full.yaml"), base_dir=base_dir)
    title_args = {'stp': expt_params.get('stp'), 'scl': expt_params.get('scl')}

    # Load Data
    raw_data, processed_data = load_data(data_abs_dir)
    all_states:np.ndarray = raw_data['all_state_trjs']  # (N, I, P, H, D)
    all_pos = all_states[..., :2]  # (N, I, P, H, 2)
    collision_masks:np.ndarray = processed_data['collision_masks']  # (N, I, P, P, H)
    # collision_masks = collision_masks.swapaxes(0, 1)  # (I, N, P, P, H)
    all_vel = processed_data['velocity_trjs']  # (N, I, P, H)
    data_adherence = processed_data['data_adherence']  # (N, I, P)
    obst_coll_msks = processed_data['obstacle_collision_masks'] if \
        'obstacle_collision_masks' in processed_data else \
        None  # (N, I, P, H)

    # --- Run Analysis Modules ---
    summary_stats = {"Directory": data_dir, **expt_params}
    analyze_and_plot_trajectories(all_pos, config, save_dir, title_args)
    summary_stats.update(
        analyze_collisions(
            all_pos,
            collision_masks,
            config, expt_params, save_dir,
            obstacle_collision_masks=obst_coll_msks
        )
    )
    summary_stats.update(analyze_rewards(data_adherence))
    summary_stats.update(analyze_velocity(all_vel, config, expt_params, save_dir))
    summary_stats.update(analyze_costs(all_pos, expt_params))

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

    all_stats_df = []
    for data_dir in tqdm(data_dirs, desc="Processing Directories", unit="dir"):
        print("\n" + "="*80)
        logger.info(f"Processing Directory: {data_dir}")
        print("="*80)
        if not os.path.isabs(data_dir) and not data_dir.startswith('~'):
            data_dir = os.path.join(base_dir, data_dir)
        try:
            stats_df = run_analysis_for_directory(data_dir, config, base_dir=base_dir)
            all_stats_df.append(stats_df)
        except Exception as e:
            logger.error(f"\nERROR processing directory '{data_dir}': \n{e}")
            logger.info("Skipping to next directory.\n")
            continue

    if all_stats_df:
        final_summary_df = pd.concat(all_stats_df)
        print("\n" + "="*80)
        print("          FINAL AGGREGATED STATISTICS         ")
        print("="*80)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(final_summary_df)

        # Save panda as csv
        final_summary_df.to_csv("logs/summary_analysis.csv")

    else:
        print("\nNo data was processed successfully.")

if __name__ == '__main__':
    main()
