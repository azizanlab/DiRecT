"""
Modified from
- https://github.com/yoraish/mmd

MIT License

Copyright (c) 2024 Yorai Shaoul
Copyright (c) 2025 Hao Luan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import argparse
import click
import yaml
from datetime import datetime
import time
import logging
import einops
import torch
import numpy as np
from tqdm import tqdm
from math import ceil
from pathlib import Path
from einops._torch_specific import allow_ops_in_compiled_graph
from typing import List, Dict, Any, Tuple, Union
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.metrics import (
    compute_path_length_from_pos, compute_average_velocity_from_pos, compute_max_velocity_from_pos, compute_average_acceleration_from_pos_vel)
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from torch_robotics.environments import *
from torch_robotics.tasks.tasks import PlanningTask
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory, CostConstraint

from mmd.models import TemporalUnet, UNET_DIM_MULTS
from mmd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mmd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mmd.trainer import get_dataset, get_model
from mmd.planners.single_agent import PCD, DiRecT
from mmd.planners.multi_agent import DummyPlanning
from mmd.common.constraints import MultiPointConstraint
from mmd.common.conflicts import PointConflict
from mmd.common.trajectory_utils import densify_trajs
from mmd.common.argparse import load_config, parse_overrides
from mmd.common.eval_utils import collision_detect, calc_data_adherence, calc_velocity
from mmd.common import (get_start_goal_pos_circle,
                        get_start_goal_pos_boundary, get_start_goal_pos_batch_random_in_env)
from mmd.common.pretty_print import *
from mmd.common.experiments import (
    MultiAgentPlanningSingleTrialConfig, MultiAgentPlanningSingleTrialResult, get_result_dir_from_trial_config,
    TrialSuccessStatus,
    get_result_dir_pcdiff,
)

# Logger setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

allow_ops_in_compiled_graph()


def generate_start_goal_positions(config: dict, tensor_args: dict, batch_size=1, method: str = None) -> List[List[torch.Tensor]]:
    """Generates start and goal positions based on the configuration."""
    cfg = config['environment']['start_goal_config']
    num_agents = config['experiment']['num_agents']
    if not method:
        method = cfg.get('method', 'random_in_env')

    if method == "random_in_env":
        env_class = eval(cfg['env_class_for_random'])
        return get_start_goal_pos_batch_random_in_env(
            num_agents,
            batch_size,
            env_class,
            tensor_args,
            margin=cfg.get('margin', 0.2),
            obstacle_margin=cfg.get('obstacle_margin', 0.11)
        )
    else:
        raise ValueError(f"Unknown start/goal generation method: {method}")


def transform_start_goal_positions(
    num_agents: int,
    start_l: List[torch.Tensor],
    goal_l: List[torch.Tensor],
    global_model_transforms: List[List[torch.Tensor]],
    agent_skeleton_l: List[List[List[int]]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    start_l_global = [start_l[i] + global_model_transforms[agent_skeleton_l[i]
                                                           [0][0]][agent_skeleton_l[i][0][1]] for i in range(num_agents)]
    goal_l_global = [goal_l[i] + global_model_transforms[agent_skeleton_l[i]
                                                         [-1][0]][agent_skeleton_l[i][-1][1]] for i in range(num_agents)]

    return start_l_global, goal_l_global


def evaluate_and_save_results(
    config: Dict[str, Any],
    mock_trial_cfg: MultiAgentPlanningSingleTrialConfig,
    results_dir: str,
    all_paths: torch.Tensor,
    planning_time: float,
    trial_success_status: TrialSuccessStatus,
    num_ct_expansions: int,
    num_collisions_in_solution: int,
    timing_profiles_l: List[Dict[str, float]],
    start_l_global: List[List[torch.Tensor]],
    goal_l_global: List[List[torch.Tensor]],
    global_model_ids: List[List[str]],
    agent_skeleton_l: List[List[List[int]]],
    agent_model_ids_l: List[List[str]],
    agent_model_transforms_l: List[Dict[int, torch.Tensor]],
    # Or a more specific base planner class if available
    low_level_planner_l: List[Any],
    reference_robot: RobotBase,      # Using a generic base class for the robot
    tensor_args: Dict[str, Any]
) -> Tuple[List[torch.Tensor], TrialSuccessStatus]:
    """
    Processes the planner's output, calculates detailed metrics, and saves all results and configs.

    Args:
        config: The main configuration dictionary from the YAML file.
        mock_trial_cfg: A simplified config object used for saving and directory naming.
        results_dir: The path to the directory where results will be saved.
        all_paths_l: The raw output from the planner (a list of all sampled trajectories).
        planning_time: The total time taken for planning.
        trial_success_status: The success status returned by the planner.
        num_ct_expansions: The number of node expansions in the conflict tree (for CBS-like planners).
        num_collisions_in_solution: The initial number of collisions reported by the planner.
        timing_profiles_l: List of timing profiles for each initialization.
        start_l_global: A list of global start position tensors for each agent.
        goal_l_global: A list of global goal position tensors for each agent.
        global_model_ids: The grid of environment model IDs.
        agent_skeleton_l: The path skeleton for each agent across the model grid.
        agent_model_ids_l: The list of model IDs encountered by each agent.
        agent_model_transforms_l: The transformations for each model in an agent's skeleton.
        low_level_planner_l: The list of instantiated low-level planners.
        reference_robot: The instantiated reference robot object for accessing properties like radius.
        tensor_args: Dictionary specifying the torch device and dtype.

    Returns:
        A tuple containing:
        - paths_l (List[torch.Tensor]): The selected final path for each agent.
        - trial_success_status (TrialSuccessStatus): The final success status after post-evaluation.
    """
    exp_cfg = config['experiment']
    sa_planner_cfg = config['planner']['single_agent']
    robot_cfg = config['robot']
    num_agents = exp_cfg['num_agents']

    all_paths_array = all_paths.to('cpu').numpy()
    all_paths_array: np.ndarray = einops.rearrange(
        all_paths_array, '(p b) i h d -> b i p h d', p=num_agents)

    starts_array = torch.stack([torch.stack(inner)
                               for inner in start_l_global]).to('cpu').numpy()
    goals_array = torch.stack([torch.stack(inner)
                              for inner in goal_l_global]).to('cpu').numpy()

    # Make sure all_paths_l containts paths for all agents' path batches:
    # [agent_0_path_batch, agent_1_path_batch, ..., agent_N_path_batch]
    # paths_l = all_paths_l

    # ============================
    # Gather Primary Stats
    # ============================
    single_trial_result = MultiAgentPlanningSingleTrialResult()
    single_trial_result.trial_config = mock_trial_cfg
    single_trial_result.start_state_pos_l = "raw_data.starts_global"
    single_trial_result.goal_state_pos_l = "raw_data.goals_global"
    single_trial_result.global_model_ids = global_model_ids
    single_trial_result.agent_skeleton_l = agent_skeleton_l
    single_trial_result.agent_path_l = "raw_data.all_state_trjs"
    single_trial_result.num_collisions_in_solution = num_collisions_in_solution
    single_trial_result.planning_time = planning_time
    single_trial_result.num_ct_expansions = num_ct_expansions

    # ============================
    # Detailed Metric Calculation
    # ============================
    B, I, N, H, D = all_paths_array.shape

    # Check for inter-agent collisions in the final solution
    collision_masks = collision_detect(
        # [B, I, N, N, H]
        all_paths_array[..., :2], safe_dist=reference_robot.radius * 2, norm_order=2)
    # Count collisions per batch per init [B, I,]
    cnt_coll_trjs = np.sum(collision_masks.any(-1).astype(int), axis=(-2, -1))
    assert cnt_coll_trjs.shape == (B, I), f"Expected collision count shape {(B,)}, got {cnt_coll_trjs.shape}"
    assert np.all(cnt_coll_trjs %
                  2 == 0), "Collision count should be even (pairs of agents)."
    cnt_coll_trjs //= 2  # Each collision was counted twice
    single_trial_result.num_collisions_in_solution = np.sum(
        cnt_coll_trjs)  # Total collisions across all batches, all inits
    if np.all(cnt_coll_trjs == 0):
        single_trial_result.success_status = TrialSuccessStatus.SUCCESS
    elif np.any(cnt_coll_trjs == 0):
        single_trial_result.success_status = TrialSuccessStatus.PARTIAL_SUCCESS
    else:
        single_trial_result.success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS
    trial_success_status = single_trial_result.success_status  # Update status

    # Check for static obstacle collisions in the final solution
    task: PlanningTask = low_level_planner_l[0].dataset.task
    n_support_points = low_level_planner_l[0].dataset.n_support_points
    n_interpolated_points = ceil(
        n_support_points * sa_planner_cfg['factor_num_interpolated_points_for_collision'])
    trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_collisions = \
        task.get_trajs_collision_and_free(
            torch.from_numpy(all_paths_array).reshape(
                B*I*N, H, -1).to(**tensor_args),  # Reshape to (B*I*N, H, D)
            return_indices=True,
            num_interpolation=n_interpolated_points,
        )
    obstacle_collision_masks: np.ndarray = trajs_waypoints_collisions.to(
        'cpu').numpy().reshape(B, I, N, H)  # (N, I, P, H)

    if True:
        # Data Adherence Metric
        single_trial_result.data_adherence = 0.0
        ma_planner_cfg = config['planner']['multi_agent']
        start_time_l = [i * ma_planner_cfg['stagger_start_time_dt']
                        for i in range(num_agents)]
        data_adherence_batch = calc_data_adherence(
            all_paths_array,
            agent_model_ids_l,
            agent_model_transforms_l,
            start_time_l,
            H,
            tensor_args
        )  # [B, I, N]
        single_trial_result.data_adherence = data_adherence_batch.mean()
        # Acceleration and Velocity
        accel_sum = 0
        max_vel_l, mean_vel_l = [], []
        # normalize all paths to get the normalized velocities
        normed_trjs = low_level_planner_l[0].dataset.normalize_trajectories(
            torch.from_numpy(all_paths_array).to(**tensor_args),
        ).to('cpu').numpy()  # [B, I, N, H, D]
        traj_pos = normed_trjs[..., :2]  # [B, I, N, H, 2]
        velocities = calc_velocity(
            traj_pos, dt=low_level_planner_l[0].robot.dt)  # [B, I, N, H-1]
        # pad zero in front
        velocities = np.concatenate(
            [np.zeros(velocities.shape[:-1]+(1,)), velocities],
            axis=-1
        )
        dims_except_agent = tuple(
            range(velocities.ndim - 2)) + (velocities.ndim-1, )
        # Max velocity per agent
        max_vel_l = velocities.max(dims_except_agent).tolist()
        mean_vel_l = velocities.mean(
            dims_except_agent).tolist()  # Mean velocity per agent

        # single_trial_result.mean_path_acceleration_per_agent = accel_sum / num_agents
        single_trial_result.mean_path_acceleration_per_agent = None
        single_trial_result.max_velocity_l = max_vel_l
        single_trial_result.mean_velocity_l = mean_vel_l

    if timing_profiles_l[0] is not None:
        timing_profiles_merged = {
            'project': np.zeros((I,), dtype=float),
            'coupling': np.zeros((I,), dtype=float),
            'diffuse': np.zeros((I,), dtype=float),
            'others': np.zeros((I,), dtype=float),
        }
        for i, t_profile in enumerate(timing_profiles_l):
            for k in ['project', 'coupling', 'diffuse', 'others']:
                timing_profiles_merged[k][i] = t_profile.get(k, 0.0)

    # ============================
    # Save All Data
    # ============================
    # Prepare raw data dictionary for .npz saving
    raw_data = {
        "all_state_trjs": all_paths_array,  # [B, I, N, H, D]
        "starts_global": starts_array,      # [B, I, N, D]
        "goals_global": goals_array,        # [B, I, N, D]
    }
    processed_data = {
        # [B, I, N, H, D - 2]
        "velocity_trjs": velocities,
        # [B, I, N, N, H]
        "collision_masks": collision_masks,
        "data_adherence": data_adherence_batch,                # [B, I, N]
        "obstacle_collision_masks": obstacle_collision_masks,  # [B, I, N, H]
    }

    # Update mock config with final dt
    mock_trial_cfg.dt = low_level_planner_l[0].robot.dt

    print(GREEN, "--- FINAL RESULTS ---", RESET)
    print(single_trial_result)

    # Save results, config, and raw data
    single_trial_result.save_json(results_dir)
    mock_trial_cfg.save(results_dir)  # Saves the simplified trial config
    with open(os.path.join(results_dir, "run_config_full.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    np.savez_compressed(os.path.join(results_dir, "raw_data.npz"), **raw_data)
    np.savez_compressed(os.path.join(
        results_dir, "processed_data.npz"), **processed_data)
    if timing_profiles_l[0] is not None:
        np.savez_compressed(os.path.join(
            results_dir, "timing_profiles.npz"), **timing_profiles_merged)

    logger.info(f"All results, configs, raw & processed data are saved to '{results_dir}'.")

    return trial_success_status, processed_data


def run_planning_experiment(config: dict, comments: str = '', trial_number=0):
    """
    Runs a single multi-agent planning experiment based on the provided configuration.
    """
    exp_cfg = config['experiment']
    sys_cfg = config['system']
    planner_cfg = config['planner']
    env_cfg = config['environment']

    # Setup System and Directories
    device = sys_cfg['device']
    tensor_args = {'device': device, 'dtype': torch.float32}
    log_pth = sys_cfg.get('log_pth', None)

    base_dir = Path(__file__).resolve(
    ).parent.parent.parent if sys_cfg['base_dir'] == 'auto' else Path(sys_cfg['base_dir'])
    base_dir_str = base_dir.as_posix()
    trained_models_dir = base_dir.joinpath(sys_cfg['trained_models_dir_name'])

    time_str = datetime.now().strftime(
        "%m%d%Y_%H%M%S") if exp_cfg['time_str'] == 'auto' else exp_cfg['time_str']
    # Append comments to the results timestamp when provided.
    if comments:
        time_str += f"-{comments.replace(' ', '_')}"

    # Create results directory path correctly
    mock_trial_cfg = MultiAgentPlanningSingleTrialConfig()
    mock_trial_cfg.instance_name = exp_cfg['name']
    mock_trial_cfg.num_agents = exp_cfg['num_agents']
    mock_trial_cfg.multi_agent_planner_class = planner_cfg['multi_agent']['class']
    mock_trial_cfg.single_agent_planner_class = planner_cfg['single_agent']['class']
    results_dir = get_result_dir_pcdiff(
        mock_trial_cfg, planner_cfg, base_dir_str, time_str, trial_number)
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Results will be saved to: {results_dir}")

    # Setup Planning Problem
    num_agents = exp_cfg['num_agents']
    global_model_ids = env_cfg['global_model_ids']
    agent_skeleton_l = [
        [[0, 0]]] * num_agents if env_cfg['agent_skeleton_l'] == 'auto' else env_cfg['agent_skeleton_l']
    dummy_start_l, dummy_goal_l = generate_start_goal_positions(
        config, tensor_args)
    dummy_start_l = dummy_start_l[0]
    dummy_goal_l = dummy_goal_l[0]
    dummy_start = torch.stack(dummy_start_l)
    dummy_goal = torch.stack(dummy_goal_l)

    # ============================
    # Setup Planners with all keys
    # ============================
    sa_planner_cfg = planner_cfg['single_agent']
    ma_planner_cfg = planner_cfg['multi_agent']
    projection_args = planner_cfg.get('projection', None)
    cost_func_param = sa_planner_cfg.get('cost_func_param')
    if cost_func_param is None:
        cost_func_param = {}

    low_level_planner_model_args = {
        'planner_alg': sa_planner_cfg['planner_alg'],
        'use_guide_on_extra_objects_only': sa_planner_cfg['use_guide_on_extra_objects_only'],
        # Total samples across all agents
        'n_samples': sa_planner_cfg['n_samples'] * num_agents,
        'n_local_inference_noising_steps': sa_planner_cfg['n_local_inference_noising_steps'],
        'n_local_inference_denoising_steps': sa_planner_cfg['n_local_inference_denoising_steps'],
        'start_guide_steps_fraction': sa_planner_cfg['start_guide_steps_fraction'],
        'cost_func_key': sa_planner_cfg['cost_func_key'],
        'n_guide_steps': sa_planner_cfg['n_guide_steps'],
        'n_diffusion_steps_without_noise': sa_planner_cfg['n_diffusion_steps_without_noise'],
        'weight_grad_cost_collision': sa_planner_cfg['weight_grad_cost_collision'],
        'weight_grad_cost_smoothness': sa_planner_cfg['weight_grad_cost_smoothness'],
        'weight_grad_cost_constraints': sa_planner_cfg['weight_grad_cost_constraints'],
        'weight_grad_cost_soft_constraints': sa_planner_cfg['weight_grad_cost_soft_constraints'],
        'factor_num_interpolated_points_for_collision': sa_planner_cfg['factor_num_interpolated_points_for_collision'],
        'trajectory_duration': sa_planner_cfg['trajectory_duration'],
        'device': device,
        'debug': sa_planner_cfg['debug'],
        'seed': exp_cfg['seed'],
        'results_dir': results_dir,
        'trained_models_dir': trained_models_dir,
        'projection_args': projection_args,
        'cost_func_param': cost_func_param,
        'timeit': sa_planner_cfg.get('timeit', False),
        'n_agents': num_agents,
    }

    # DiRecT-specific args
    if sa_planner_cfg['class'] == 'DiRecT':
        low_level_planner_model_args['projector_type'] = sa_planner_cfg.get('projector_type', 'Ipopt')
        low_level_planner_model_args['start_projection_guidance'] = sa_planner_cfg.get('start_projection_guidance', 0.0)

    high_level_planner_model_args = {
        'is_xcbs': ma_planner_cfg.get('is_xcbs', False),
        'is_ecbs': ma_planner_cfg.get('is_ecbs', False),
        'start_time_l': [i * ma_planner_cfg['stagger_start_time_dt'] for i in range(num_agents)],
        'runtime_limit': exp_cfg['runtime_limit'],
        'conflict_type_to_constraint_types': {
            eval(k): {eval(v_i) for v_i in v} for k, v in ma_planner_cfg['conflict_type_to_constraint_types'].items()
        }
    }

    # ============================
    # Create Reference Agent and Planner
    # ============================
    tile_width, tile_height = 2.0, 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **tensor_args)
                                for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]

    low_level_planner_class = eval(sa_planner_cfg['class'])

    reference_agent_skeleton = [[r, c] for r in range(
        len(global_model_ids)) for c in range(len(global_model_ids[0]))]
    reference_agent_transforms = {
        sk_step: global_model_transforms[sk_coord[0]][sk_coord[1]]
        for sk_step, sk_coord in enumerate(reference_agent_skeleton)
    }
    reference_agent_model_ids = [
        global_model_ids[sk_coord[0]][sk_coord[1]]
        for sk_coord in reference_agent_skeleton
    ]

    logger.info("Creating reference agent and planner.")
    ref_args = low_level_planner_model_args.copy()
    ref_args.update({
        'start_state_pos': dummy_start, 'goal_state_pos': dummy_goal,
        'model_ids': reference_agent_model_ids,
        'transforms': reference_agent_transforms,
    })
    if sa_planner_cfg['class'] in {"PCD", "DiRecT"}:
        ref_args['model_id'] = reference_agent_model_ids[0]

    reference_low_level_planner = low_level_planner_class(**ref_args)
    reference_task = reference_low_level_planner.task
    reference_robot = reference_low_level_planner.robot

    # ============================
    # Create Agent-Specific Models and Final Planners
    # ============================

    agent_model_transforms_l, agent_model_ids_l = [], []
    for agent_id in range(num_agents):
        agent_transforms = {
            sk_step: global_model_transforms[sk_coord[0]][sk_coord[1]]
            for sk_step, sk_coord in enumerate(agent_skeleton_l[agent_id])
        }
        agent_ids = [
            global_model_ids[sk_coord[0]][sk_coord[1]]
            for sk_coord in agent_skeleton_l[agent_id]
        ]
        agent_model_transforms_l.append(agent_transforms)
        agent_model_ids_l.append(agent_ids)

    planners_creation_start_time = time.time()
    low_level_planner_args = low_level_planner_model_args.copy()
    low_level_planner_args.update({
        'start_state_pos': dummy_start,  # this does not matter
        'goal_state_pos': dummy_goal,  # this does not matter
        'model_ids': agent_model_ids_l[0],
        'transforms': agent_model_transforms_l[0]
    })
    if sa_planner_cfg['class'] in {"PCD", "DiRecT"}:
        low_level_planner_args['model_id'] = agent_model_ids_l[0][0]

    low_level_planner_l = [low_level_planner_class(**low_level_planner_args)]
    logger.info(f"Planner creation time: {time.time() - planners_creation_start_time:.4f}s\n" + '-' * 80)

    # ============================
    # Create the Multi-Agent Planner
    # ============================
    multi_agent_planner_class = eval(ma_planner_cfg['class'])
    planner: DummyPlanning
    planner = multi_agent_planner_class(
        low_level_planner_l,
        dummy_start_l, dummy_goal_l,
        reference_task=reference_task,
        reference_robot=reference_robot,
        **high_level_planner_model_args
    )

    # ============================
    # Start Planning
    # ============================

    n_inits = exp_cfg['n_inits']
    all_paths_l = []
    logger.info(f"Start planning for {n_inits} trials with {num_agents} agents.")

    # Reset seed for reproducibility (initialization of planner used random state generator)
    fix_random_seed(exp_cfg['seed'])
    # Precompute starts and goals for all inits
    starts = []
    goals = []
    start_ls, goal_ls = generate_start_goal_positions(
        config, tensor_args, n_inits)
    for s, g in zip(start_ls, goal_ls):
        start_l_global, goal_l_global = transform_start_goal_positions(
            num_agents,
            s, g,
            global_model_transforms, agent_skeleton_l
        )
        starts.append(start_l_global)
        goals.append(goal_l_global)

    pbar = tqdm(zip(starts, goals), desc="Inits", total=n_inits, ncols=80)
    timing_profiles_l = []
    for s_glb, g_glb in pbar:
        planner.update_start_goal_states(s_glb, g_glb)
        start_time = time.time()
        paths_this_init, t_profile, num_ct_expansions_this_init, trial_success_status, num_collisions_in_solution = \
        planner.plan(runtime_limit=exp_cfg['runtime_limit'], )
        planning_time = time.time() - start_time
        # List[ [N*B, H, D] ], move to CPU for saving
        all_paths_l.append(torch.stack(paths_this_init, dim=0).to('cpu'))
        timing_profiles_l.append(t_profile)
        pbar.set_postfix({
            'time': f"{planning_time:.3f}s"
        })
    # end for [init_idx]
    all_paths_cpu = torch.stack(all_paths_l, dim=1).to('cpu')  # [N*B, I, H, D]

    # ============================
    #  Process, Evaluate, and Save Results
    # ============================
    trial_success_status, processed_data = evaluate_and_save_results(
        config, mock_trial_cfg, results_dir,
        all_paths_cpu, planning_time, trial_success_status, num_ct_expansions_this_init, num_collisions_in_solution,
        timing_profiles_l,
        starts, goals,
        global_model_ids, agent_skeleton_l,
        agent_model_ids_l, agent_model_transforms_l,
        low_level_planner_l, reference_robot, tensor_args
    )

    # Write the output directory to a .log file
    if log_pth:
        outer_log_file = log_pth
        with open(outer_log_file, 'a' if os.path.exists(outer_log_file) else 'w') as f:
            f.write(f'{results_dir}\n')

    # ============================
    # Render Animation
    # ============================
    # Find a feasible path to render
    collisions: np.ndarray = processed_data['collision_masks'].any(
        axis=(-1, -2, -3))  # [B, I, N, N, H] -> [B, I]
    coll_free_path_mask = ~collisions  # [B, I]
    # Find all (b, i) indices where the path is collision-free
    # shape [num_free, 2], each row is [b, i]
    coll_free_indices = np.argwhere(coll_free_path_mask)
    # If there are any collision-free paths, pick the first one; otherwise, default to (0, 0)
    if coll_free_indices.shape[0] > 0:
        b_idx, i_idx = coll_free_indices[0]
    else:
        b_idx, i_idx = 0, 0

    grp_sz = sa_planner_cfg['n_samples']
    paths_l = [all_paths_cpu[n * grp_sz + b_idx, i_idx].to(**tensor_args)
               for n in range(num_agents)]
    if len(paths_l) > 0 and exp_cfg['render_animation']:
        logger.info("Rendering final animation...")
        # Restore the corresponding start and goal positions for rendering
        s_glb, g_glb = starts[i_idx], goals[i_idx]
        planner.update_start_goal_states(s_glb, g_glb)
        # Denser trajectory for smoother GIF
        render_paths = densify_trajs(paths_l, 1)
        planner.render_paths(
            render_paths,
            output_fpath=os.path.join(results_dir, f"{exp_cfg['name']}.gif"),
            plot_trajs=True,
            animation_duration=5.0
        )
        logger.info("Rendering complete.")


@ click.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@ click.option("--config", type=click.Path(exists=True), required=True,
               help="Path to the configuration YAML file.")
@ click.option("--log_pth", type=click.Path(), default=None,
               help="Optional path to save logs. If not provided, logs will be saved in the results directory.")
@ click.option("--trial_number", type=int, default=0,
               help="Trial number for saving results. Default is 0.")
@ click.option("--comments", "-m", default="", help="Optional comments for the results directory.")
@ click.pass_context
def main(ctx, config, log_pth, trial_number, comments):
    # ctx.args contains all the unknown args (our overrides)
    overrides = parse_overrides(ctx.args)

    # load & override
    cfg = load_config(config, overrides)
    cfg['log_pth'] = log_pth

    run_planning_experiment(cfg, comments, trial_number)


if __name__ == '__main__':
    main()
