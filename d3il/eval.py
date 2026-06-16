import os
import csv
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils
import tqdm
import tyro
import gym

import numpy as np
import torch
from dataclasses import dataclass, field

from datasets.sequence import SequenceDataset
from model import TemporalUnet
from obstacles import (
    CircularObstacle, PlanarObstacle,
    PILLARS, NOVEL_CIRCULAR, NOVEL_PLANAR, CONSTRAINT_SETS,
    count_constraints, check_violation,
)
from policy.unconditional import UnconditionalPolicy
from policy.gradient_guidance import GradientGuidancePolicy
from policy.classifier_guidance import ClassifierGuidancePolicy
from policy.projected_diffusion import ProjectedDiffusionPolicy
from policy.value_model import ProxyValueModel
from utils.training import deterministic, save_config, set_cuda_visible_device

import d3il


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvaluationConfig:
    seed: int = 0
    random_repeat: int = 50
    device: str = "cuda:0"
    log_folder: str = "logs"
    exp_name: str = "diffusion"

    render: bool = False

    controller: str = "rh"
    replan_steps: int = 8

    env: str = "avoiding-v0"
    horizon: int = 16
    normalizer: str = "LimitsNormalizer"
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 200
    max_n_episodes: int = 500
    termination_penalty: float = 0
    max_episode_length: int = 100

    state_dim: int = 4
    action_dim: int = 2

    diffusion_exp_name: str = "diffusion"
    diffusion_cp: str = "0"

    batch_size: int = 1
    n_sampling_steps: int = 10

    constraints: bool = False
    obstacle_margin: float = 0.02
    draw_obstacle_margin: bool = False

    # guidance
    guidance_method: str = "no"  # "no", "gradient", "classifier", "projection", "safediffuser", "primal-dual", "augmented-lagrangian", "direct"
    guidance_lr: float = 0.05
    guidance_steps: int = 5
    gradient_start: float = 0.0
    projection_start: float = 0.5

    # value model weights
    value_objective: str = ""  # "" or "distance"
    objective_weight: float = 1.0
    obstacle_weight: float = 100.0
    dynamics_weight: float = 0.0

    # safediffuser CBF
    cbf_algorithm: str = "RoS"  # "RoS", "ReS", "TVS"
    cbf_coefficient: float = 1.0
    guidance_start_fraction: float = 0.0  # fraction of steps before guidance begins
    guidance_last_steps: int = 0  # apply guidance only to last N steps (0 = use guidance_start_fraction)
    cbf_slack_weight: float = 1000.0
    cbf_relaxation_end: float = 0.9
    cbf_sigmoid_bias: float = 0.9
    cbf_sigmoid_input: float = 32.0
    cbf_sigmoid_output: float = 1.0

    # primal-dual / augmented Lagrangian
    pd_learning_rate: float = 0.456
    pd_initial_penalty: float = 2.5e-4
    pd_final_penalty: float = 0.00154
    pd_use_cbfs: bool = True
    pd_cbf_alpha: float = 0.9

    # DiRecT (prediction in clean space)
    pred_control_weight: float = 1.0
    pred_objective: str = ""           # "" or "distance"
    pred_objective_scale: float = 100.0
    max_action_delta: float = 0.01  # trajectory curvature limit (second difference on positions, 0 = disabled)

    # dynamics (always loaded; relaxation controls equality vs inequality tolerance)
    dynamics_relaxation: float = 0.0  # 0 = equality, >0 = inequality with tolerance

    # PID tracking controller
    pd_kp: float = 5.0  # proportional gain on position error
    pd_kd: float = 1.0  # derivative gain on velocity error
    pd_ki: float = 0.0  # integral gain on accumulated position error

    no_images: bool = False
    no_trajectories: bool = False
    no_early_stop: bool = False




# ---------------------------------------------------------------------------
# Batched episode runner
# ---------------------------------------------------------------------------

def run_batched_eval(envs, policy, cfg, normalizer=None, save_path=None):
    """Run random_repeat episodes using len(envs) env slots, batching inference.

    Slots are reused: when an episode finishes, the env is reset for the next
    episode until all random_repeat episodes are complete.
    """
    B = len(envs)  # number of env slots (= batch_size)
    total_episodes = cfg.random_repeat
    constraint_name = "novel" if cfg.constraints else ""

    assert (
        cfg.replan_steps < cfg.horizon
    ), f"replan steps ({cfg.replan_steps}) must be smaller than horizon ({cfg.horizon})"

    # results indexed by episode id
    all_rollouts = [None] * total_episodes
    all_successes = [False] * total_episodes
    all_violations = [0.0] * total_episodes
    all_rewards = [0.0] * total_episodes
    all_comp_times = [[] for _ in range(total_episodes)]
    all_planned_obs = [[] for _ in range(total_episodes)]  # list of predicted obs per replan
    all_plan_feasibility = [[] for _ in range(total_episodes)]  # True/False per replan

    # per-slot state
    slot_episode = list(range(min(B, total_episodes)))  # which episode each slot runs
    next_episode = len(slot_episode)  # next episode id to assign
    slot_active = [True] * B
    observations = [None] * B
    rollouts = [None] * B
    planned_actions = [None] * B
    planned_obs = [None] * B  # predicted observations for PD tracking
    action_indices = [0] * B
    slot_steps = [0] * B  # track steps per slot for max_episode_length
    integral_error = [np.zeros(2) for _ in range(B)]  # accumulated position error for I term
    last_action_norm = [None] * B  # last executed action in normalized space (for action rate constraint)
    use_pd = cfg.pd_kp > 0 or cfg.pd_kd > 0 or cfg.pd_ki > 0

    # deactivate unused slots if random_repeat < batch_size
    for i in range(B):
        if i >= total_episodes:
            slot_active[i] = False
        else:
            obs = envs[i].reset()
            observations[i] = obs
            rollouts[i] = [obs.copy()]

    episodes_done = 0
    pbar = tqdm.tqdm(desc=f"Eval (0/{total_episodes} episodes done)")
    while episodes_done < total_episodes:
        # 1. find slots that need replanning
        need_replan = [
            i for i in range(B)
            if slot_active[i] and (planned_actions[i] is None or action_indices[i] >= cfg.replan_steps)
        ]

        if need_replan:
            # 2. batch observations through policy
            batch_obs = np.stack([observations[i] for i in need_replan])
            batch_conditions = {0: batch_obs}

            # Pass last executed action for action rate constraint
            policy_kwargs = {}
            if hasattr(policy, 'max_action_delta') and policy.max_action_delta > 0:
                ap = last_action_norm[need_replan[0]]
                if ap is not None:
                    policy_kwargs["a_prev"] = ap

            _, batch_trajs, _, _, info = policy(
                batch_conditions, batch_size=len(need_replan),
                **policy_kwargs,
            )

            for j, i in enumerate(need_replan):
                planned_actions[i] = batch_trajs.actions[j]
                planned_obs[i] = batch_trajs.observations[j]
                action_indices[i] = 0
                integral_error[i] = np.zeros(2)  # reset I term at replan
                ep = slot_episode[i]
                all_planned_obs[ep].append(batch_trajs.observations[j])

                # Check feasibility of planned trajectory
                obs_plan = batch_trajs.observations[j]  # (H, 4)
                n_viol = sum(1 for t in range(len(obs_plan)) if check_violation(obs_plan[t], constraint_name) > 0)
                all_plan_feasibility[ep].append(n_viol == 0)
                if "computation_time" in info:
                    all_comp_times[ep].append(info["computation_time"] / len(need_replan))

        # 3. step each active slot
        for i in range(B):
            if not slot_active[i]:
                continue

            ep = slot_episode[i]
            action = planned_actions[i][action_indices[i]].copy()

            # PID tracking correction (in normalized space for comparable scales)
            if use_pd and planned_obs[i] is not None and normalizer is not None:
                obs_actual_n = normalizer.normalize(
                    observations[i].reshape(1, -1), "observations"
                ).flatten()
                obs_planned_n = normalizer.normalize(
                    planned_obs[i][action_indices[i]].reshape(1, -1), "observations"
                ).flatten()
                action_n = normalizer.normalize(
                    action.reshape(1, -1), "actions"
                ).flatten()
                pos_error = obs_planned_n[2:4] - obs_actual_n[2:4]
                vel_error = obs_planned_n[0:2] - obs_actual_n[0:2]
                integral_error[i] += pos_error
                action_n += (cfg.pd_kp * pos_error
                           + cfg.pd_kd * vel_error
                           + cfg.pd_ki * integral_error[i])
                action = normalizer.unnormalize(
                    action_n.reshape(1, -1), "actions"
                ).flatten()

            action_indices[i] += 1
            slot_steps[i] += 1

            obs, reward, terminated, info_env = envs[i].step(action)
            observations[i] = obs
            rollouts[i].append(obs.copy())
            all_rewards[ep] += reward

            # Track last action in normalized space for action rate constraint
            if normalizer is not None:
                last_action_norm[i] = normalizer.normalize(
                    action.reshape(1, -1), "actions"
                ).flatten()

            success = info_env[1]
            violation = check_violation(obs, constraint_name)
            if violation > 0 or (terminated and not success):
                violation = 1.0
            else:
                violation = 0.0
            all_violations[ep] += violation

            episode_done = False
            if terminated or success:
                episode_done = True
            elif violation > 0 and not cfg.no_early_stop:
                episode_done = True
            elif slot_steps[i] >= cfg.max_episode_length:
                episode_done = True

            if episode_done:
                # save episode results
                all_rollouts[ep] = np.array(rollouts[i])
                all_successes[ep] = success
                episodes_done += 1

                # Print episode feasibility summary
                feas = all_plan_feasibility[ep]
                n_feasible = sum(feas) if feas else 0
                n_total = len(feas) if feas else 0
                feas_str = f"plans: {n_feasible}/{n_total} feasible" if n_total > 0 else ""
                status_str = "OK" if success else "FAIL"
                safe_str = "safe" if all_violations[ep] == 0 else f"violations={int(all_violations[ep])}"
                pbar.write(f"  ep {ep}: {status_str} | {safe_str} | {feas_str} | {slot_steps[i]} steps")

                # Save per-episode prediction overlay plot
                if save_path is not None and not cfg.no_images and all_planned_obs[ep]:
                    pred_dir = os.path.join(save_path, "predictions")
                    os.makedirs(pred_dir, exist_ok=True)
                    ep_plotter = utils.AvoidingTrajectoryPlotter(
                        constraint=("novel" if cfg.constraints else ""),
                        obstacle_margin=cfg.obstacle_margin,
                        draw_obstacle_margin=cfg.draw_obstacle_margin,
                    )
                    ep_fig, ep_ax = ep_plotter.plot_overlaid_predictions_and_real(
                        predicted_trajectories=all_planned_obs[ep],
                        real_trajectory=all_rollouts[ep],
                        title=f"Episode {ep}",
                    )
                    ep_plotter.save_figure(ep_fig, os.path.join(pred_dir, f"ep_{ep:03d}.png"))
                    plt.close(ep_fig)

                pbar.update(1)
                pbar.set_description(f"Eval ({episodes_done}/{total_episodes} episodes done)")

                # reuse slot for next episode
                if next_episode < total_episodes:
                    slot_episode[i] = next_episode
                    next_episode += 1
                    obs = envs[i].reset()
                    observations[i] = obs
                    rollouts[i] = [obs.copy()]
                    planned_actions[i] = None
                    planned_obs[i] = None
                    action_indices[i] = 0
                    integral_error[i] = np.zeros(2)
                    last_action_norm[i] = None
                    slot_steps[i] = 0
                else:
                    slot_active[i] = False

    pbar.close()
    return all_rollouts, all_successes, all_violations, all_rewards, all_comp_times, all_planned_obs, all_plan_feasibility


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(cfg: EvaluationConfig):

    # get dataset (only for its normalizer)
    dataset = SequenceDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
    )
    normalizer = dataset.normalizer

    model = TemporalUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
        attention=False,
    ).to(cfg.device)
    model.load_state_dict(
        torch.load(
            os.path.join(
                cfg.log_folder,
                cfg.env,
                "diffusion",
                cfg.diffusion_exp_name,
                f"model_ema_{cfg.diffusion_cp}.pth",
            )
        )
    )
    model.eval()

    # --- load dynamics model (always on) ---
    dynamics_path = os.path.join(cfg.log_folder, cfg.env, "dynamics", "linear_model.npz")
    print(f"Loading dynamics model from: {dynamics_path}")
    dynamics_model = dict(np.load(dynamics_path, allow_pickle=True))

    # --- build policy ---
    if cfg.guidance_method == "gradient":
        value_model = ProxyValueModel(
            horizon=cfg.horizon,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            normalizer=normalizer,
            objective=cfg.value_objective,
            constraints=cfg.constraints,
            dynamics=True,
            objective_weight=cfg.objective_weight,
            obstacle_weight=cfg.obstacle_weight,
            dynamics_weight=cfg.dynamics_weight,
            obstacle_margin=cfg.obstacle_margin,
            dynamics_model=dynamics_model,
        ).to(cfg.device)

        policy = GradientGuidancePolicy(
            model=model,
            normalizer=normalizer,
            value_model=value_model,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            guidance_lr=cfg.guidance_lr,
            guidance_steps=cfg.guidance_steps,
            gradient_start=cfg.gradient_start,
            device=cfg.device,
        )
    elif cfg.guidance_method == "classifier":
        value_model = ProxyValueModel(
            horizon=cfg.horizon,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            normalizer=normalizer,
            objective=cfg.value_objective,
            constraints=cfg.constraints,
            dynamics=True,
            objective_weight=cfg.objective_weight,
            obstacle_weight=cfg.obstacle_weight,
            dynamics_weight=cfg.dynamics_weight,
            obstacle_margin=cfg.obstacle_margin,
            dynamics_model=dynamics_model,
        ).to(cfg.device)

        policy = ClassifierGuidancePolicy(
            model=model,
            normalizer=normalizer,
            value_model=value_model,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            guidance_lr=cfg.guidance_lr,
            gradient_start=cfg.gradient_start,
            device=cfg.device,
        )
    elif cfg.guidance_method == "projection":
        from projector import IpoptProjector
        constraint_name = "novel" if cfg.constraints else ""
        circular_obs = CONSTRAINT_SETS[constraint_name]["circular"] if constraint_name else []
        planar_obs = CONSTRAINT_SETS[constraint_name]["planar"] if constraint_name else []

        projector = IpoptProjector(
            circular_obstacles=circular_obs,
            planar_obstacles=planar_obs,
            horizon=cfg.horizon,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            normalizer=normalizer,
            dynamics=dynamics_model,
            dynamics_relaxation=cfg.dynamics_relaxation,
            obstacle_margin=cfg.obstacle_margin,
            max_action_delta=cfg.max_action_delta,
        )

        policy = ProjectedDiffusionPolicy(
            model=model,
            normalizer=normalizer,
            projector=projector,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            projection_start=cfg.projection_start,
            n_projection_workers=cfg.batch_size,
            device=cfg.device,
        )
    elif cfg.guidance_method == "safediffuser":
        from policy.safediffuser import SafeDiffuserPolicy

        policy = SafeDiffuserPolicy(
            model=model,
            normalizer=normalizer,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            algorithm_type=cfg.cbf_algorithm,
            cbf_coefficient=cfg.cbf_coefficient,
            guidance_start=cfg.guidance_start_fraction,
            guidance_last_steps=cfg.guidance_last_steps,
            obstacle_margin=cfg.obstacle_margin,
            slack_weight=cfg.cbf_slack_weight,
            relaxation_end_fraction=cfg.cbf_relaxation_end,
            sigmoid_bias_fraction=cfg.cbf_sigmoid_bias,
            sigmoid_input_scale=cfg.cbf_sigmoid_input,
            sigmoid_output_scale=cfg.cbf_sigmoid_output,
            dynamics_model=dynamics_model,
            device=cfg.device,
        )
    elif cfg.guidance_method in ("primal-dual", "augmented-lagrangian"):
        from policy.augmented_lagrangian import AugmentedLagrangianPolicy

        policy = AugmentedLagrangianPolicy(
            model=model,
            normalizer=normalizer,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            algorithm_type=cfg.guidance_method,
            guidance_start=cfg.guidance_start_fraction,
            guidance_last_steps=cfg.guidance_last_steps,
            obstacle_margin=cfg.obstacle_margin,
            learning_rate=cfg.pd_learning_rate,
            initial_penalty=cfg.pd_initial_penalty,
            final_penalty=cfg.pd_final_penalty,
            use_cbfs=cfg.pd_use_cbfs,
            cbf_alpha=cfg.pd_cbf_alpha,
            dynamics_model=dynamics_model,
            device=cfg.device,
        )
    elif cfg.guidance_method == "direct":
        from policy.direct import DiReCTPolicy

        policy = DiReCTPolicy(
            model=model,
            normalizer=normalizer,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            guidance_start=cfg.guidance_start_fraction,
            guidance_last_steps=cfg.guidance_last_steps,
            obstacle_margin=cfg.obstacle_margin,
            control_penalty_weight=cfg.pred_control_weight,
            prediction_objective=cfg.pred_objective,
            prediction_objective_scale=cfg.pred_objective_scale,
            max_action_delta=cfg.max_action_delta,
            dynamics_model=dynamics_model,
            dynamics_relaxation=cfg.dynamics_relaxation,
            device=cfg.device,
        )
    else:
        policy = UnconditionalPolicy(
            model=model,
            normalizer=normalizer,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
            n_sampling_steps=cfg.n_sampling_steps,
            device=cfg.device,
        )

    # create batch_size env instances (reused across episodes)
    n_envs = min(cfg.batch_size, cfg.random_repeat)
    print(f"Creating {n_envs} environment instances for {cfg.random_repeat} episodes...")
    envs = []
    for _ in range(n_envs):
        env = gym.make("avoiding-v0", render=cfg.render)
        env.set_seed(cfg.seed)
        env.start()
        envs.append(env)

    save_path = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # run batched evaluation
    (
        real_trajectories,
        successes,
        total_violations,
        total_rewards,
        computation_times,
        planned_observations,
        all_plan_feasibility,
    ) = run_batched_eval(envs, policy, cfg, normalizer=normalizer, save_path=save_path)

    # --- per-run metrics ---
    n_constraints = count_constraints(("novel" if cfg.constraints else ""))
    trajectory_data = []

    for i in range(cfg.random_repeat):
        steps = len(real_trajectories[i]) - 1
        safety = total_violations[i] == 0
        comp_times = computation_times[i]

        trajectory_data.append({
            "run_id": i,
            "steps": steps,
            "total_violations": float(total_violations[i]),
            "total_rewards": float(total_rewards[i]),
            "success": bool(successes[i]),
            "safety": bool(safety),
            "n_constraints": n_constraints,
            "constraint_set": ("novel" if cfg.constraints else ""),
            "average_computation_time": (
                float(np.mean(comp_times)) if comp_times else None
            ),
        })

    # --- always save per-run metrics CSV ---
    csv_path = os.path.join(save_path, "runs.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = list(trajectory_data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in trajectory_data:
            writer.writerow(data)
    print(f"Saved per-run metrics to: {csv_path}")

    # --- always save experiment summary ---
    n_runs = len(trajectory_data)
    n_success = sum(1 for d in trajectory_data if d["success"])
    n_safe = sum(1 for d in trajectory_data if d["safety"])
    avg_steps = np.mean([d["steps"] for d in trajectory_data])
    avg_violations = np.mean([d["total_violations"] for d in trajectory_data])
    avg_rewards = np.mean([d["total_rewards"] for d in trajectory_data])
    comp_times_all = [d["average_computation_time"] for d in trajectory_data if d["average_computation_time"] is not None]

    summary = {
        "exp_name": cfg.exp_name,
        "constraint_set": ("novel" if cfg.constraints else ""),
        "n_constraints": n_constraints,
        "n_runs": n_runs,
        "success_rate": n_success / n_runs,
        "safety_rate": n_safe / n_runs,
        "avg_steps": float(avg_steps),
        "avg_violations": float(avg_violations),
        "avg_rewards": float(avg_rewards),
        "avg_computation_time": float(np.mean(comp_times_all)) if comp_times_all else None,
    }

    summary_path = os.path.join(save_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary to: {summary_path}")

    # --- always save all-trajectories overlay ---
    plotter = utils.AvoidingTrajectoryPlotter(
        constraint=("novel" if cfg.constraints else ""),
        obstacle_margin=cfg.obstacle_margin,
        draw_obstacle_margin=cfg.draw_obstacle_margin,
    )
    fig, ax = plotter.plot_multiple_real_trajectories(real_trajectories)
    plotter.save_figure(fig, os.path.join(save_path, "all_real.png"))

    # --- save trajectory data (unless --no-trajectories) ---
    if not cfg.no_trajectories:
        traj_path = os.path.join(save_path, "trajectories.npz")
        np.savez(
            traj_path,
            **{f"run_{i}": traj for i, traj in enumerate(real_trajectories)},
        )
        print(f"Saved trajectory data to: {traj_path}")

    # print summary
    print(f"\n{'='*50}")
    print(f"Success rate: {summary['success_rate']:.1%} ({n_success}/{n_runs})")
    print(f"Safety rate:  {summary['safety_rate']:.1%} ({n_safe}/{n_runs})")
    print(f"Avg steps:    {summary['avg_steps']:.1f}")

    # Plan feasibility summary
    total_plans = sum(len(f) for f in all_plan_feasibility)
    feasible_plans = sum(sum(f) for f in all_plan_feasibility)
    if total_plans > 0:
        print(f"Plan feasibility: {feasible_plans}/{total_plans} ({feasible_plans/total_plans:.1%})")
        for ep_i, feas in enumerate(all_plan_feasibility):
            if feas and not all(feas):
                n_infeas = sum(1 for f in feas if not f)
                print(f"  ep {ep_i}: {n_infeas}/{len(feas)} infeasible plans")
    print(f"{'='*50}")


if __name__ == "__main__":
    cfg = tyro.cli(EvaluationConfig)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)
    save_config(cfg, log_subfolder)

    evaluate(cfg)
