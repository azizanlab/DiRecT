import pickle
import torch
import numpy as np
import os
import yaml
from tqdm import tqdm
from datetime import datetime
from os.path import join, dirname

from torch_robotics.tasks.tasks import PlanningTask
from mmd.utils.loading import load_params_from_yaml
from mmd.trainer import get_dataset
from mmd.datasets.trajectories import TrajectoryDataset
from mmd.config.mmd_params import MMDParams
from mmd.common.eval_utils import collision_detect, calc_data_adherence, calc_velocity
from mmd.common.experiments.experiments import MultiAgentPlanningSingleTrialResult, MultiAgentPlanningSingleTrialConfig

PROJECT_ROOT = dirname(dirname(dirname(__file__)))
TRAINED_MODELS_DIR = join(PROJECT_ROOT, "data_trained_models")
TENSOR_ARGS = {'dtype': torch.float32, 'device': 'cuda'}


def get_dataset_from_cfg(cfg:MultiAgentPlanningSingleTrialConfig):
    model_id = cfg.global_model_ids[0][0]
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)

    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    ####################################
    # Load dataset with env, robot, task. The TrajectoryDataset type is used here.
    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class='TrajectoryDataset',
        use_extra_objects=True,
        obstacle_cutoff_margin=0.05,
        **args,
        tensor_args=TENSOR_ARGS
    )
    # Extract objects from the dataset.
    dataset = train_subset.dataset
    return dataset

def get_args_from_cfg(num_agents:int, cfg:MultiAgentPlanningSingleTrialConfig):
    start_time_l = [i * cfg.stagger_start_time_dt for i in range(num_agents)]
    global_model_ids = cfg.global_model_ids
    tile_width, tile_height = 2.0, 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **TENSOR_ARGS) for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]
    agent_skeleton_l = [[[0, 0]]] * num_agents
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
    return start_time_l, agent_model_transforms_l, agent_model_ids_l


def main(n_agents_ls:list, data_base_dir:str):
    print(f"Project root: {PROJECT_ROOT}")
    if not os.path.isabs(data_base_dir):
        data_base_dir = join(PROJECT_ROOT, data_base_dir)

    for num_agents in n_agents_ls:
        print(f"Number of agents: {num_agents}")
        path_plate = f"num_agents___{num_agents}/planner___CBS/single_agent_planner___MPD"
        data_dir = join(data_base_dir, path_plate)
        print(f"Data directory: {data_dir}")

        all_paths = []
        single_trial_configs = []
        traj_duration_ls = []
        all_starts_l = []
        all_goals_l = []

        # get all the directories in the data_dir
        trial_dirs= []
        for root, dirs, files in os.walk(data_dir):
            for dir_name in dirs:
                if dir_name.isdigit():
                    trial_dirs.append(dir_name)
        if not trial_dirs:
            print(f"Warning: No trial directories found in \'{data_dir}\'. ")
            continue
        trial_dirs.sort(key=int)
        missing_ids = []
        for dir_name in tqdm(trial_dirs):
            trial_dir = join(data_dir, dir_name)
            pkl_pth = join(trial_dir, "results.pkl")
            if os.path.exists(pkl_pth):
                with open(pkl_pth, 'rb') as f:
                    trial_result: MultiAgentPlanningSingleTrialResult = pickle.load(f)
                    all_paths.append(torch.stack(trial_result.agent_path_l, dim=0).to('cpu').numpy())  # [P, H, D]
                    all_starts_l.append(np.stack(trial_result.start_state_pos_l, axis=0)) # [P, D]
                    all_goals_l.append(np.stack(trial_result.goal_state_pos_l, axis=0)) # [P, D]
                    assert all_starts_l[-1].shape == (num_agents, 2), f"Unexpected start shape: {all_starts_l[-1].shape}"
                    assert all_goals_l[-1].shape == (num_agents, 2), f"Unexpected goal shape: {all_goals_l[-1].shape}"
                    single_trial_configs.append(trial_result.trial_config)
            else:
                print(f"Warning: 'results.pkl' not found in {trial_dir}. Skipping this trial.")
                missing_ids.append(int(dir_name))
                continue
            mmd_params_pth = join(trial_dir, "mmd_params.yaml")
            if os.path.exists(mmd_params_pth):
                with open(mmd_params_pth, 'r') as f:
                    mmd_params = yaml.safe_load(f)
                    traj_duration = mmd_params.get('traj_duration', 5.0)
                    traj_duration_ls.append(float(traj_duration))
            else:
                print(f"Warning: mmd_params.yaml not found in {trial_dir}")
                traj_duration_ls.append(5.0)

        missing_id_array = None
        if missing_ids:
            print(f"Warning: Missing trial directories for IDs: {missing_ids}")
            missing_id_array = np.array(missing_ids)

        all_paths_array = np.stack(all_paths, axis=0)[np.newaxis, ...]  # [B, I, P, H, D]
        print(f"All paths shape: {all_paths_array.shape}")
        print(f"(Shape for refrence: [Batch, Init, Player, Horizon, Dim])")
        all_starts_array = np.stack(all_starts_l, axis=0)  # [I, P, D]
        all_goals_array = np.stack(all_goals_l, axis=0)    # [I, P, D]
        assert all_starts_array.shape[0] == all_paths_array.shape[1] == all_goals_array.shape[0], \
            f"Mismatch in number of initializations: starts {all_starts_array.shape[0]}, paths {all_paths_array.shape[1]}, goals {all_goals_array.shape[0]}"

        # check whether all traj_durations are the same
        traj_duration_set = set(traj_duration_ls)
        if len(traj_duration_set) > 1:
            print(f"Warning: Different trajectory durations found: {traj_duration_set}. Using the first one.")
        traj_duration = traj_duration_ls[0]

        B, I, P, H, D = all_paths_array.shape
        assert P == num_agents, f"Expected P (number of players) to be equal to num_agents ({num_agents}), got {P}."

        dataset:TrajectoryDataset = get_dataset_from_cfg(single_trial_configs[0])
        robot = dataset.robot
        task:PlanningTask = dataset.task
        robot.dt = traj_duration / H

        # Check for inter-agent collisions in the final solution
        collision_masks = collision_detect(all_paths_array[..., :2], safe_dist = robot.radius * 2, norm_order=2) # [B, I, N, N, H]
        cnt_coll_trjs = np.sum(collision_masks.any(-1).astype(int), axis=(-2, -1)) # Count collisions per batch per init [B, I,]
        assert cnt_coll_trjs.shape == (B, I), f"Expected collision count shape {(B,)}, got {cnt_coll_trjs.shape}"
        assert np.all(cnt_coll_trjs % 2 == 0), "Collision count should be even (pairs of agents)."
        cnt_coll_trjs //= 2  # Each collision was counted twice
        print(f"Count of collisions shape: {cnt_coll_trjs.shape}")

        # Check for static obstacle collisions in the final solution
        n_interpolated_points = 0
        trajs_coll, trajs_coll_idxs, trajs_free, trajs_free_idxs, trajs_waypoints_collisions = \
            task.get_trajs_collision_and_free(
                torch.from_numpy(all_paths_array).reshape(B*I*P, H, -1).to(**TENSOR_ARGS),  # Reshape to (B*I*P, H, D)
                return_indices=True,
                num_interpolation=n_interpolated_points,
            )
        obstacle_collision_masks:np.ndarray = trajs_waypoints_collisions.to('cpu').numpy().reshape(B, I, P, H)  # (B, I, P, H)
        print(f"Obstacle collision masks shape: {obstacle_collision_masks.shape}")

        ## Get data adherence
        start_time_l, agent_model_transforms_l, agent_model_ids_l = get_args_from_cfg(num_agents, single_trial_configs[0])
        data_adherence_batch = calc_data_adherence(
            all_paths_array,
            agent_model_ids_l,
            agent_model_transforms_l,
            start_time_l,
            H,
            TENSOR_ARGS
        )  # [B, I, P]
        print(f"Data adherence shape: {data_adherence_batch.shape}")

        ### normalize all paths to get the normalized velocities
        normed_trjs = dataset.normalize_trajectories(
            torch.from_numpy(all_paths_array).to(**TENSOR_ARGS),
        ).to('cpu').numpy()  # [B, I, P, H, D]
        traj_pos = normed_trjs[..., :2]  # [B, I, P, H, 2]
        velocities = calc_velocity(traj_pos, dt=robot.dt) # [B, I, P, H-1]
        velocities = np.concatenate(
            [np.zeros(velocities.shape[:-1]+(1,)), velocities],
            axis=-1
        )
        print(f"Velocities shape: {velocities.shape}")  # [B, I, P, H]

        raw_data = {
            "all_state_trjs": all_paths_array,  # [B, I, P, H, D]
            "starts_global": all_starts_array,  # [I, P, D]
            "goals_global": all_goals_array,  # [I, P, D]
        }
        if missing_id_array is not None:
            raw_data["missing_ids"] = missing_id_array  # [M,]
        processed_data = {
            "collision_masks": collision_masks,  # [B, I, N, N, H]
            "cnt_coll_trjs": cnt_coll_trjs,  # [B, I]
            "obstacle_collision_masks": obstacle_collision_masks,  # [B, I, P, H]
            "data_adherence": data_adherence_batch,  # [B, I, P]
            "velocity_trjs": velocities  # [B, I, P, H]
        }

        raw_data_pth = join(data_base_dir, f"raw_data_{num_agents}_agents.npz")
        np.savez_compressed(
            raw_data_pth,
            **raw_data,
        )
        print(f"Raw data saved to \'{raw_data_pth}\'.")

        processed_data_pth = join(data_base_dir, f"processed_data_{num_agents}_agents.npz")
        np.savez_compressed(
            processed_data_pth,
            **processed_data,
        )
        print(f"Processed data saved to \'{processed_data_pth}\'.")
    #end for [num_agents]


if __name__ == "__main__":
    list_n_agents_ls = [
        [4],
        # [2],
    ]
    list_data_base_dirs = [
        "results/EnvConveyor2DRobotPlanarDiskRandom/2026-04-14-19-26-50",
        # "results/EnvConveyor2DRobotPlanarDiskRandom/some_other_run_dir",
    ]
    assert len(list_n_agents_ls) == len(list_data_base_dirs), \
        "Number of agent lists must match number of data base directories."

    # Create a dummy log file and write the directories into it for record-keeping
    time_str = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_file_pth = join(PROJECT_ROOT, dirname(list_data_base_dirs[0]), f"dummy_logs-{time_str}.log")
    with open(log_file_pth, "w") as f:
        f.write("")
    print(f"Dummy log file created at: {log_file_pth}")

    for n_agents_ls, data_base_dir in zip(list_n_agents_ls, list_data_base_dirs):
        main(n_agents_ls, data_base_dir)
        with open(log_file_pth, "a") as f:
            f.write(f"{data_base_dir}\n")

    print(f"Log file at '{log_file_pth}' updated with data directories.")
