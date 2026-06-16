import os
import shutil
import csv
from typing import Any, Sequence, Optional, Tuple, Dict, Callable, List

import torch
import numpy as np
import random
import yaml

import warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*")

from config import Config, load_config, parse_args, apply_overrides
from config import save_config as save_config_yaml
from d4rl.sequence import GoalDataset
from d4rl.d4rl import suppress_output
from env import Maze2dConstrainedEnv
from policy import POLICY_DICT, Policy, apply_dict
from diffusion import Diffusion
from unet import TemporalUnet
from tqdm import tqdm


## UTILS
EXCLUDE_CSV_KEYS = {
    "rollout",
    "initial_observation",
    "conditions",
    "intermediates",
    "data_estimates",
    "observations",
    "action",
    "actions",
    "observation",
}


def save_result_to_csv(save_path: str, results: List[Dict[str, Any]]):

    # Find all keys
    keys = set()
    for result_dict in results:
        keys.update(result_dict.keys())

    # Do not save keys contained in `EXCLUDE_SAVE_KEYS` and have `run_id` first
    keys = keys - EXCLUDE_CSV_KEYS
    keys = sorted(keys, key=lambda x: (x != "run_id", x))

    # Add None to all missing keys and convert numpy to single float
    save_dicts = []
    for result_dict in results:
        save_dict = {}
        for key in keys:
            if key not in result_dict:
                save_dict[key] = None
            elif isinstance(result_dict[key], np.ndarray):
                save_dict[key] = result_dict[key].mean().item()
            elif isinstance(result_dict[key], torch.Tensor):
                save_dict[key] = result_dict[key].mean().cpu().detach().item()
            else:
                save_dict[key] = result_dict[key]
        save_dicts.append(save_dict)

    # Save
    fieldnames = list(keys)
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        for data in save_dicts:
            writer.writerow(data)


EXCLUDE_SAVE_KEYS = {
    "run_id",
    "rollout",
    "initial_observation",
    "conditions",
    "observation",
    "fixed_start",
    "intermediates",
    "data_estimates",
    "action",
    "actions",
    "observation",
}


def compute_and_save_statistics(
    summary_path: str, trajectory_data: List[Dict[str, Any]], verbose: bool = False
):
    if not trajectory_data:
        if verbose:
            print("No trajectory data to analyze")
        return

    # Find fields present in all trajectories
    all_keys = set(trajectory_data[0].keys())
    for traj in trajectory_data[1:]:
        all_keys &= set(traj.keys())
    common_keys = all_keys - EXCLUDE_SAVE_KEYS

    # Compute statistics for each common field
    statistics = {}
    total_runs = len(trajectory_data)

    for key in sorted(common_keys):
        values = []
        for traj in trajectory_data:
            val = traj[key]
            if val is None:
                continue

            # Convert to numpy if possible
            try:
                if isinstance(val, (list, tuple)):
                    val = np.array(val)
                elif isinstance(val, bool):
                    val = float(val)
                elif isinstance(val, np.ndarray):
                    val = val.mean()
                elif isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy().mean()

                # Check if numeric
                if isinstance(val, (int, float, np.number)):
                    values.append(float(val))
            except (ValueError, TypeError):
                continue

        if values:
            statistics[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values),
            }

    # Print summary statistics
    if verbose:
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total runs: {total_runs}\n")

        for key in sorted(statistics.keys()):
            stats = statistics[key]
            print(f"{key}:")
            print(f"  Mean: {stats['mean']:.6f}")
            print(f"  Std:  {stats['std']:.6f}")
            print(f"  Min:  {stats['min']:.6f}")
            print(f"  Max:  {stats['max']:.6f}")
            if stats["count"] < total_runs:
                print(f"  Count: {stats['count']}/{total_runs}")
            print()

        print("=" * 80 + "\n")

    # Save summary statistics to file
    with open(summary_path, "w") as f:
        f.write("EVALUATION SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total runs: {total_runs}\n\n")

        for key in sorted(statistics.keys()):
            stats = statistics[key]
            f.write(f"{key}:\n")
            f.write(f"  Mean: {stats['mean']:.6f}\n")
            f.write(f"  Std:  {stats['std']:.6f}\n")
            f.write(f"  Min:  {stats['min']:.6f}\n")
            f.write(f"  Max:  {stats['max']:.6f}\n")
            if stats["count"] < total_runs:
                f.write(f"  Count: {stats['count']}/{total_runs}\n")
            f.write("\n")
        f.write("=" * 80 + "\n")


## Rollout controller
def pd_controller(
    observation: torch.Tensor,
    state_sequence: torch.Tensor,
    idx_t: torch.Tensor,
    constants: Tuple[float, ...] = (5.0, 5.0, 1.0, 1.0),
) -> torch.Tensor:
    err = state_sequence[idx_t] - observation
    control_actions = err * torch.tensor(constants, dtype=err.dtype, device=err.device)
    action = control_actions.view(*control_actions.shape[:-1], 2, 2).sum(-2)
    return action


class Evaluator:
    def __init__(
        self,
        env: Maze2dConstrainedEnv,
        policy: Policy,
        eval_folder: str,
        fixed_start: bool = True,
        fixed_start_value: Optional[Tuple[float, ...]] = None,
        device: str = "cpu",
        num_random_repeats: int = 100,
        use_float64: bool = True,
        verbose_saving: bool = False,
        save_generation: bool = False,
        save_trajectories: bool = False,
        batch_size: int = 1,
    ):
        self.env = env
        self.policy = policy
        self.fixed_start = fixed_start
        self.fixed_start_value = fixed_start_value
        self.device = device
        self.num_random_repeats = num_random_repeats
        self.use_float64 = use_float64
        self.eval_folder = eval_folder
        self.verbose_saving = verbose_saving
        self.save_generation = save_generation
        self.save_trajectories = save_trajectories
        self.batch_size = batch_size

        os.makedirs(self.eval_folder, exist_ok=True)

        if fixed_start:
            if self.fixed_start_value is not None:
                self.fixed_start_state = self.fixed_start_value
            else:
                self.fixed_start_state = self.env.default_fixed_start
        self.conditions = self.env.get_conditions()

        self.device = device
        self.dtype = torch.float64 if use_float64 else torch.float32
        self.np_dtype = np.float64 if use_float64 else np.float32
        policy.to(device=self.device, dtype=self.dtype)

    def run_env(self, run_id: int) -> Dict[str, Any]:
        log_folder = os.path.join(self.eval_folder, f"{run_id}")
        os.makedirs(log_folder, exist_ok=True)

        run_dict = {}

        observation = self.env.reset()
        if self.fixed_start:
            observation = np.asarray(self.fixed_start_state)
            self.env.set_state(observation)

        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        conditions = self.conditions
        conditions[0] = observation
        apply_dict(
            conditions,
            lambda x: (
                torch.as_tensor(x, dtype=self.dtype, device=self.device)
                if isinstance(x, np.ndarray)
                else x
            ),
        )

        run_dict.update(
            {
                "initial_observation": observation,
                "conditions": conditions,
                "fixed_start": self.fixed_start,
            }
        )

        ## Generate Trajectory
        trajectory_data = self.policy.generate(conditions)
        run_dict.update(trajectory_data)
        metrics = self.env.compute_metrics(
            trajectory_data=trajectory_data, prefix="generated"
        )
        run_dict.update(metrics)

        ## Render
        if "observation" in trajectory_data:
            with suppress_output(enabled=not self.verbose_saving):
                self.env.render_maze(
                    trajectory_data["observation"].clone().view(1, -1, 4),
                    savepath=os.path.join(log_folder, "predicted.png"),
                    plot_obstacles=True,
                    normalized_obs=False,
                )

        if "intermediates" in trajectory_data and self.save_generation:
            intermediates_folder = os.path.join(log_folder, "intermediates")
            os.makedirs(intermediates_folder, exist_ok=True)
            intermediates = (
                trajectory_data["intermediates"]
                .clone()
                .view(1, -1, self.env.horizon, 6)[0]
            )
            with suppress_output(enabled=not self.verbose_saving):
                for idx, intermediate_sample in enumerate(intermediates):
                    self.env.render_maze(
                        intermediate_sample[:, 2:].unsqueeze(0),
                        savepath=os.path.join(intermediates_folder, f"{idx}.png"),
                        plot_obstacles=True,
                        normalized_obs=False,
                    )

        if "data_estimates" in trajectory_data and self.save_generation:
            estimates_folder = os.path.join(log_folder, "data_estimates")
            os.makedirs(estimates_folder, exist_ok=True)
            data_estimates = (
                trajectory_data["data_estimates"]
                .clone()
                .view(1, -1, self.env.horizon, 6)[0]
            )
            with suppress_output(enabled=not self.verbose_saving):
                for idx, estimate_sample in enumerate(data_estimates):
                    self.env.render_maze(
                        estimate_sample[:, 2:].unsqueeze(0),
                        savepath=os.path.join(estimates_folder, f"{idx}.png"),
                        plot_obstacles=True,
                        normalized_obs=False,
                    )

        ## Perform env rollout by PD tracking
        state_sequence = trajectory_data["observation"].view(1, self.env.horizon, 4)[0]

        rollout_observations = [observation.clone()]
        rollout_actions = []
        score = total_rewards = total_violation = 0.0
        for step in range(self.env.env.max_episode_steps):  # type: ignore
            action = pd_controller(
                observation=observation,
                state_sequence=state_sequence,
                idx_t=torch.as_tensor(
                    min(step, len(state_sequence) - 1),
                    device=self.device,
                    dtype=torch.long,
                ),
            ).clamp(min=-1.0, max=1.0)

            next_observation, reward, _, _, _ = self.env.step(action)
            next_observation = torch.as_tensor(
                next_observation, device=self.device, dtype=self.dtype
            )
            rollout_observations.append(next_observation.clone())
            rollout_actions.append(action.clone())
            total_rewards += float(reward)
            score = self.env.get_normalized_score(total_rewards)
            violation = self.env.check_violation(next_observation)
            total_violation += violation
            observation = next_observation
        rollout_actions.append(torch.zeros_like(rollout_actions[-1]))

        run_dict.update(
            {
                "score": score,
                "reward": total_rewards,
                "total_violation": total_violation,
                "rollout": torch.stack(rollout_observations, dim=0),
                "rollout_actions": torch.stack(rollout_actions, dim=0),
                "safety": total_violation == 0,
                "steps": len(rollout_observations) - 1,
                "run_id": run_id,
            }
        )

        ## Compute metrics over the rollout
        metrics = self.env.compute_metrics(
            trajectory_data={
                "observation": run_dict["rollout"],
                "actions": run_dict["rollout_actions"],
            },
            prefix="rollout",
        )
        run_dict.update(metrics)

        ## Plot final path
        with suppress_output(enabled=not self.verbose_saving):
            self.env.render_maze(
                run_dict["rollout"].view(1, -1, 4),
                savepath=os.path.join(log_folder, "rollout.png"),
                plot_obstacles=True,
                normalized_obs=False,
                style="trajectory",
            )

        ## Save trajectories as npz for post-analysis
        if self.save_trajectories:
            np.savez(
                os.path.join(log_folder, "trajectories.npz"),
                generated_observations=run_dict["observation"].cpu().detach().numpy(),
                generated_actions=run_dict["actions"].cpu().detach().numpy(),
                rollout_observations=run_dict["rollout"].cpu().detach().numpy(),
                rollout_actions=run_dict["rollout_actions"].cpu().detach().numpy(),
            )

        return run_dict

    def _setup_conditions(self):
        """Set up the observation and conditions for generation."""
        observation = self.env.reset()
        if self.fixed_start:
            observation = np.asarray(self.fixed_start_state)
            self.env.set_state(observation)

        observation = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        conditions = self.conditions
        conditions[0] = observation
        apply_dict(
            conditions,
            lambda x: (
                torch.as_tensor(x, dtype=self.dtype, device=self.device)
                if isinstance(x, np.ndarray)
                else x
            ),
        )
        return observation, conditions

    def _rollout_single(self, observation, state_sequence, run_id):
        """PD controller rollout + rendering + saving for a single trajectory."""
        log_folder = os.path.join(self.eval_folder, f"{run_id}")
        os.makedirs(log_folder, exist_ok=True)

        # Reset env to fixed start for each rollout
        if self.fixed_start:
            obs_np = np.asarray(self.fixed_start_state)
            self.env.set_state(obs_np)
            observation = torch.as_tensor(obs_np, dtype=self.dtype, device=self.device)

        rollout_observations = [observation.clone()]
        rollout_actions = []
        score = total_rewards = total_violation = 0.0
        for step in range(self.env.env.max_episode_steps):  # type: ignore
            action = pd_controller(
                observation=observation,
                state_sequence=state_sequence,
                idx_t=torch.as_tensor(
                    min(step, len(state_sequence) - 1),
                    device=self.device,
                    dtype=torch.long,
                ),
            ).clamp(min=-1.0, max=1.0)

            next_observation, reward, _, _, _ = self.env.step(action)
            next_observation = torch.as_tensor(
                next_observation, device=self.device, dtype=self.dtype
            )
            rollout_observations.append(next_observation.clone())
            rollout_actions.append(action.clone())
            total_rewards += float(reward)
            score = self.env.get_normalized_score(total_rewards)
            violation = self.env.check_violation(next_observation)
            total_violation += violation
            observation = next_observation
        rollout_actions.append(torch.zeros_like(rollout_actions[-1]))

        rollout = torch.stack(rollout_observations, dim=0)
        rollout_act = torch.stack(rollout_actions, dim=0)

        ## Render rollout
        with suppress_output(enabled=not self.verbose_saving):
            self.env.render_maze(
                rollout.view(1, -1, 4),
                savepath=os.path.join(log_folder, "rollout.png"),
                plot_obstacles=True,
                normalized_obs=False,
                style="trajectory",
            )

        return {
            "score": score,
            "reward": total_rewards,
            "total_violation": total_violation,
            "rollout": rollout,
            "rollout_actions": rollout_act,
            "safety": total_violation == 0,
            "steps": len(rollout_observations) - 1,
        }

    def run_batch(self, run_ids: List[int]) -> List[Dict[str, Any]]:
        """Generate a batch of trajectories and rollout each sequentially."""
        batch_size = len(run_ids)
        observation, conditions = self._setup_conditions()

        ## Generate batch of trajectories
        trajectory_data = self.policy.generate(conditions, batch_size=batch_size)

        results = []
        for b, run_id in enumerate(run_ids):
            log_folder = os.path.join(self.eval_folder, f"{run_id}")
            os.makedirs(log_folder, exist_ok=True)

            run_dict = {
                "initial_observation": observation,
                "conditions": conditions,
                "fixed_start": self.fixed_start,
                "run_id": run_id,
            }

            # Extract single trajectory from batch
            single_obs = trajectory_data["observation"][b : b + 1]  # [1, H, 4]
            single_act = trajectory_data["actions"][b : b + 1]  # [1, H, 2]
            run_dict["observation"] = single_obs
            run_dict["actions"] = single_act

            if "proxy_value" in trajectory_data:
                run_dict["proxy_value"] = trajectory_data["proxy_value"][b : b + 1]
            if "computation_time" in trajectory_data:
                run_dict["computation_time"] = trajectory_data["computation_time"] / batch_size
            if "total_control" in trajectory_data:
                tc = trajectory_data["total_control"]
                run_dict["total_control"] = tc[b] if isinstance(tc, torch.Tensor) and tc.dim() > 0 else tc
            if "intermediates" in trajectory_data:
                run_dict["intermediates"] = trajectory_data["intermediates"][b : b + 1]
            if "data_estimates" in trajectory_data:
                run_dict["data_estimates"] = trajectory_data["data_estimates"][b : b + 1]

            ## Generated metrics
            gen_metrics = self.env.compute_metrics(
                trajectory_data={"observation": single_obs, "actions": single_act},
                prefix="generated",
            )
            run_dict.update(gen_metrics)

            ## Render predicted
            with suppress_output(enabled=not self.verbose_saving):
                self.env.render_maze(
                    single_obs.clone().view(1, -1, 4),
                    savepath=os.path.join(log_folder, "predicted.png"),
                    plot_obstacles=True,
                    normalized_obs=False,
                )

            ## PD rollout (sequential per trajectory)
            state_sequence = single_obs.view(self.env.horizon, 4)
            rollout_result = self._rollout_single(observation, state_sequence, run_id)
            run_dict.update(rollout_result)

            ## Rollout metrics
            rollout_metrics = self.env.compute_metrics(
                trajectory_data={
                    "observation": run_dict["rollout"],
                    "actions": run_dict["rollout_actions"],
                },
                prefix="rollout",
            )
            run_dict.update(rollout_metrics)

            ## Save trajectories as npz
            if self.save_trajectories:
                np.savez(
                    os.path.join(log_folder, "trajectories.npz"),
                    generated_observations=run_dict["observation"].cpu().detach().numpy(),
                    generated_actions=run_dict["actions"].cpu().detach().numpy(),
                    rollout_observations=run_dict["rollout"].cpu().detach().numpy(),
                    rollout_actions=run_dict["rollout_actions"].cpu().detach().numpy(),
                )

            results.append(run_dict)

        return results

    def eval(self):
        trajectory_data = []
        batch_size = self.batch_size

        run_ids = list(range(self.num_random_repeats))
        for batch_start in tqdm(range(0, len(run_ids), batch_size)):
            batch = run_ids[batch_start : batch_start + batch_size]
            trajectory_data.extend(self.run_batch(batch))

        csv_path = os.path.join(self.eval_folder, "statistics.csv")
        save_result_to_csv(csv_path, trajectory_data)

        summary_path = os.path.join(self.eval_folder, "summary_statistics.txt")
        compute_and_save_statistics(summary_path, trajectory_data, verbose=True)


## Main
def deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args(description="Evaluate diffusion model")
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    print(yaml.dump(cfg.__dict__, default_flow_style=False))
    deterministic(cfg.seed)
    log_path = os.path.join(cfg.log_folder, cfg.exp_name, "eval")
    if cfg.overwrite and os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    save_config_yaml(cfg, os.path.join(log_path, "config.yaml"))

    dataset = GoalDataset(
        env=cfg.dataset.env_name,
        horizon=cfg.dataset.horizon,
        normalizer=cfg.dataset.normalizer,
        preprocess_fns=cfg.dataset.preprocess_fns,
        max_path_length=cfg.dataset.max_path_length,
        max_n_episodes=cfg.dataset.max_n_episodes,
        termination_penalty=cfg.dataset.termination_penalty,
        seed=cfg.seed,
        cheap_load=True
    )

    dynamics_path = cfg.eval.dynamics_path
    if dynamics_path is None:
        dynamics_path = os.path.join(cfg.log_folder, "dynamics", "linear_model.npz")

    env = Maze2dConstrainedEnv(
        env_dataset=dataset,
        obstacle_types=cfg.policy.obstacle_type,
        objective=cfg.policy.env_objective,
        constraints=cfg.policy.env_constraints,
        dynamics_path=dynamics_path,
        proxy_objective_value_weight=cfg.policy.proxy_objective_value_weight,
        proxy_constraint_penalty_weight=cfg.policy.proxy_constraint_penalty_weight,
        constraint_margin=cfg.policy.constraint_margin,
    )

    nn = TemporalUnet(
        horizon=cfg.dataset.horizon,
        transition_dim=6,
        dim=cfg.diffusion.dim,
        dim_mults=cfg.diffusion.dim_mults,
    )

    diffuser = Diffusion(
        neural_network=nn,
        prediction_type=cfg.diffusion.prediction_type,
        lr=cfg.diffusion.lr,
        scheduler_T_max=cfg.diffusion.scheduler_T_max,
        noise_schedule_kwargs=cfg.diffusion.noise_schedule_kwargs,
        antithetic_sampling=cfg.diffusion.antithetic_sampling,
        clip_denoised=cfg.diffusion.clip_denoised,
        horizon=cfg.dataset.horizon,
    )

    # Load weights
    if cfg.eval.weights_path is not None:
        weights_path = str(cfg.eval.weights_path)
        state_dict = torch.load(weights_path, map_location=diffuser.device)
        if weights_path.endswith(".ckpt"):
            state_dict = state_dict["state_dict"]
        diffuser.load_state_dict(state_dict, strict=True)

    policy_name = args.policy if args.policy is not None else "no-guidance"
    policy_cls = POLICY_DICT[policy_name]
    policy = policy_cls(
        env=env,
        diffuser=diffuser,
        sampling_steps=cfg.policy.sampling_steps,
        sampler_interpolation_coeff=cfg.policy.sampler_interpolation_coeff,
        time_discretization=cfg.policy.time_discretization,
        time_discretization_kwargs=cfg.policy.time_discretization_kwargs,
        policy_kwargs=cfg.policy.policy_kwargs,
    )

    evaluator = Evaluator(
        env=env,
        policy=policy,
        eval_folder=log_path,
        fixed_start=cfg.eval.fixed_start,
        device=cfg.eval.device,
        num_random_repeats=cfg.eval.num_random_repeats,
        use_float64=cfg.eval.use_float64,
        verbose_saving=cfg.eval.verbose_saving,
        save_generation=cfg.eval.save_generation,
        save_trajectories=cfg.eval.save_trajectories,
        batch_size=cfg.eval.batch_size,
    )

    evaluator.eval()


if __name__ == "__main__":
    main()
