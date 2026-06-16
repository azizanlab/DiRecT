import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import d3il
import gym
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from datasets.sequence import SequenceDataset
from obstacles import PILLARS

matplotlib.use("Agg")


class LinearDynamicsAnalyzer:
    def __init__(self, env_name: str, horizon: int = 16, max_episodes: int = 500):
        self.env_name = env_name
        self.horizon = horizon
        self.max_episodes = max_episodes

        self.output_dir = Path(f"logs/{self.env_name}/dynamics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output will be saved to: {self.output_dir}")

        print(f"Loading data from {self.env_name}...")
        (
            self.X,
            self.y,
            self.actions,
            self.metadata,
            self.dataset,
        ) = self._collect_data()

        self.A: np.ndarray = None
        self.B: np.ndarray = None
        self.c: np.ndarray = None
        self.metrics: Dict[str, Any] = {}
        self.y_pred: np.ndarray = None
        self.verification_metrics: Dict[str, Any] = None

    def _collect_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], SequenceDataset]:
        dataset = SequenceDataset(
            env=self.env_name,
            horizon=self.horizon,
            normalizer="LimitsNormalizer",
            preprocess_fns=[],
            max_path_length=200,
            max_n_episodes=self.max_episodes,
            termination_penalty=0,
            seed=0,
        )

        buffer = dataset.fields
        x_list, y_list, actions_list = [], [], []

        for path_idx in range(buffer.n_episodes):
            path_length = buffer.path_lengths[path_idx]
            observations = buffer.normed_observations[path_idx, :path_length]
            actions = buffer.normed_actions[path_idx, :path_length]

            for t in range(path_length - 1):
                x_list.append(observations[t])
                y_list.append(observations[t + 1])
                actions_list.append(actions[t])

        x = np.array(x_list)
        y = np.array(y_list)
        actions = np.array(actions_list)

        metadata = {
            "env_name": self.env_name,
            "num_transitions": len(x),
            "obs_dim": x.shape[1],
            "action_dim": actions.shape[1] if len(actions) > 0 else 0,
            "num_episodes": buffer.n_episodes,
        }
        return x, y, actions, metadata, dataset

    def fit_model(self):
        print("Fitting linear dynamics model...")
        features = np.hstack([self.X, self.actions])
        models = []
        predictions = []

        for i in range(self.y.shape[1]):
            model = LinearRegression()
            model.fit(features, self.y[:, i])
            models.append(model)
            predictions.append(model.predict(features))

        self.y_pred = np.column_stack(predictions)

        obs_dim = self.X.shape[1]
        self.A = np.array([m.coef_[:obs_dim] for m in models])
        self.B = np.array([m.coef_[obs_dim:] for m in models])
        self.c = np.array([m.intercept_ for m in models])

        self.metrics = {
            "overall_mse": mean_squared_error(self.y, self.y_pred),
            "overall_r2": r2_score(self.y, self.y_pred),
            "dim_mse": [
                mean_squared_error(self.y[:, i], self.y_pred[:, i])
                for i in range(self.y.shape[1])
            ],
            "dim_r2": [
                r2_score(self.y[:, i], self.y_pred[:, i])
                for i in range(self.y.shape[1])
            ],
        }
        self.metrics["rmse"] = np.sqrt(self.metrics["overall_mse"])

    def analyze_model(self):
        if self.A is None:
            print("Model not fitted yet. Call fit_model() first.")
            return

        print(f"\n=== Linear Dynamics Model Analysis for {self.env_name} ===")
        print(f"Dataset: {self.metadata['num_transitions']} transitions")
        print(
            f"Obs dim: {self.metadata['obs_dim']}, Action dim: {self.metadata['action_dim']}"
        )

        print("\n=== Model Performance ===")
        print(f"Overall R²: {self.metrics['overall_r2']:.4f}")
        print(f"Overall RMSE: {self.metrics['rmse']:.4f}")

        print("\n=== Per-dimension Performance ===")
        for i, (r2, mse) in enumerate(
            zip(self.metrics["dim_r2"], self.metrics["dim_mse"])
        ):
            print(f"Dimension {i}: R² = {r2:.4f}, MSE = {mse:.4f}")

        print("\n=== State Transition Matrix A (s_{t+1} dependence on s_t) ===")
        print("Shape:", self.A.shape)
        eigenvals = np.linalg.eigvals(self.A)
        max_eig = np.max(np.abs(eigenvals))
        print(f"Spectral radius (max eigenvalue magnitude): {max_eig:.4f}")

        if max_eig < 1.0:
            print("System appears stable (spectral radius < 1.0)")
        else:
            print("System may be unstable (spectral radius >= 1.0)")

        print("\n=== Action Influence Matrix B (s_{t+1} dependence on a_t) ===")
        print("Shape:", self.B.shape)
        print(f"Frobenius norm: {np.linalg.norm(self.B, 'fro'):.4f}")

    def verify_with_env(self, n_test_cases: int = 1000):
        print("\n=== Verifying Linear Dynamics with Real Environment (Single-Step) ===")
        env = gym.make(self.env_name)
        env.set_seed(42)
        env.start()

        test_indices = np.random.choice(len(self.X), n_test_cases, replace=True)
        env_next_states, model_next_states = [], []

        for test_idx in test_indices:
            try:
                current_state_norm = self.X[test_idx]
                action_norm = self.actions[test_idx]

                current_state_unnorm = self.dataset.normalizer.unnormalize(
                    current_state_norm.reshape(1, -1), "observations"
                ).flatten()
                action_unnorm = self.dataset.normalizer.unnormalize(
                    action_norm.reshape(1, -1), "actions"
                ).flatten()

                env.reset()
                env.set_state(current_state_unnorm)
                if not np.allclose(
                    env.get_observation(), current_state_unnorm, atol=1e-2
                ):
                    continue
                next_obs_unnorm, _, _, _ = env.step(action_unnorm)

                next_state_model_norm = (
                    self.A @ current_state_norm + self.B @ action_norm + self.c
                )

                next_state_env_norm = self.dataset.normalizer.normalize(
                    next_obs_unnorm.reshape(1, -1), "observations"
                ).flatten()

                env_next_states.append(next_state_env_norm)
                model_next_states.append(next_state_model_norm)
            except Exception:
                continue

        if not model_next_states:
            print("Warning: No successful verification tests completed!")
            return

        env_next = np.array(env_next_states)
        model_next = np.array(model_next_states)
        errors = env_next - model_next

        self.verification_metrics = {
            "overall_rmse": np.sqrt(np.mean(errors**2)),
            "rmse_per_dim": np.sqrt(np.mean(errors**2, axis=0)),
            "overall_r2": r2_score(env_next, model_next),
            "r2_per_dim": [
                r2_score(env_next[:, i], model_next[:, i])
                for i in range(env_next.shape[1])
            ],
            "n_comparisons": len(env_next),
        }

        print(f"Successful comparisons: {len(env_next)}/{n_test_cases}")
        print(
            f"Overall R² (normalized space): {self.verification_metrics['overall_r2']:.4f}"
        )
        print(
            f"Overall RMSE (normalized space): {self.verification_metrics['overall_rmse']:.6f}"
        )
        self.visualize_verification_results(env_next, model_next)

    def visualize(self):
        print("\nGenerating visualizations...")
        self._visualize_all_trajectories()
        self._visualize_model_predictions()

    def _visualize_all_trajectories(self):
        fig, ax = plt.subplots(figsize=(9, 10))
        buffer = self.dataset.fields
        for path_idx in range(buffer.n_episodes):
            path_length = buffer.path_lengths[path_idx]
            obs_norm = buffer.normed_observations[path_idx, :path_length]
            obs_unnorm = self.dataset.normalizer.unnormalize(obs_norm, "observations")
            ax.plot(obs_unnorm[:, 2], obs_unnorm[:, 3], alpha=0.6, lw=1)
            ax.plot(obs_unnorm[0, 2], obs_unnorm[0, 3], "o", markersize=3, alpha=0.7)

        ax.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], lw=5, label="Goal")
        for pillar in PILLARS:
            ax.add_patch(patches.Circle(list(pillar.center), pillar.radius, color="r", alpha=0.7))

        ax.set_xlim([0.2, 0.8])
        ax.set_ylim([-0.3, 0.4])
        ax.set_title("All Collected Trajectories")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True, alpha=0.3)
        ax.legend()

        save_path = self.output_dir / "all_trajectories.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"All trajectories visualization saved to: {save_path}")

    def _visualize_model_predictions(self):
        if self.y_pred is None:
            return
        obs_dim = self.y.shape[1]
        n_samples = min(10000, len(self.y))
        idx = np.random.choice(len(self.y), n_samples, replace=False)

        fig, axes = plt.subplots(2, (obs_dim + 1) // 2, figsize=(12, 8), squeeze=False)
        axes = axes.flatten()

        for i in range(obs_dim):
            ax = axes[i]
            ax.scatter(self.y[idx, i], self.y_pred[idx, i], alpha=0.6, s=1)
            min_val, max_val = self.y[idx, i].min(), self.y[idx, i].max()
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
            ax.set_xlabel(f"True Next Obs Dim {i}")
            ax.set_ylabel(f"Predicted Next Obs Dim {i}")
            ax.set_title(
                f"Dim {i}: R² = {r2_score(self.y[:, i], self.y_pred[:, i]):.3f}"
            )
            ax.grid(True, alpha=0.3)

        for i in range(obs_dim, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"Linear Model Predictions vs True Values\n{self.env_name}")
        plt.tight_layout()

        save_path = self.output_dir / "linear_model_predictions.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Prediction visualization saved to: {save_path}")

    def visualize_verification_results(self, env_states_all, model_states_all):
        if len(env_states_all) == 0 or self.verification_metrics is None:
            print("No verification data to visualize.")
            return

        obs_dim = env_states_all.shape[1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
        axes = axes.flatten()

        for dim in range(min(obs_dim, 4)):
            ax = axes[dim]
            ax.scatter(env_states_all[:, dim], model_states_all[:, dim], alpha=0.6, s=1)
            min_val = min(env_states_all[:, dim].min(), model_states_all[:, dim].min())
            max_val = max(env_states_all[:, dim].max(), model_states_all[:, dim].max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
            rmse = self.verification_metrics["rmse_per_dim"][dim]
            r2 = self.verification_metrics["r2_per_dim"][dim]
            ax.set_xlabel(f"Environment State Dim {dim}")
            ax.set_ylabel(f"Model Predicted Dim {dim}")
            ax.set_title(f"Dim {dim}: R²={r2:.3f}, RMSE={rmse:.4f}")
            ax.grid(True, alpha=0.3)

        for dim in range(obs_dim, 4):
            axes[dim].set_visible(False)

        plt.suptitle(
            f"Linear Model vs Environment Verification\n"
            f"Overall R²: {self.verification_metrics['overall_r2']:.4f}, "
            f"Overall RMSE: {self.verification_metrics['overall_rmse']:.4f} "
            f"({self.verification_metrics['n_comparisons']} comparisons)"
        )
        plt.tight_layout()

        save_path = self.output_dir / "dynamics_verification.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Verification visualization saved to: {save_path}")

    def save_model(self):
        if self.A is None:
            print("Model not fitted yet. Cannot save.")
            return

        save_data = {
            "A": self.A,
            "B": self.B,
            "c": self.c,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "normalizer": self.dataset.normalizer,
        }
        if self.verification_metrics is not None:
            save_data["verification_metrics"] = self.verification_metrics

        model_path = self.output_dir / "linear_model.npz"
        np.savez(model_path, **save_data)
        print(f"Model saved to: {model_path}")

    def run_analysis(self):
        self.visualize()
        self.fit_model()
        self.analyze_model()
        self._visualize_model_predictions()
        self.verify_with_env()
        self.save_model()

        print(f"\n=== Summary ===")
        print(f"Fitted linear dynamics model for {self.env_name}")
        print(f"Model form: s_{{t+1}} = A*s_t + B*a_t + c")
        print(
            f"Overall performance: R² = {self.metrics['overall_r2']:.4f}, RMSE = {self.metrics['rmse']:.4f}"
        )
        if self.verification_metrics is not None:
            print(
                f"Environment verification R²: {self.verification_metrics['overall_r2']:.4f}, "
                f"RMSE: {self.verification_metrics['overall_rmse']:.6f}"
            )
        print(f"Files saved to: {self.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fit, analyze, and verify a linear dynamics model for a gym environment."
    )
    parser.add_argument("--env", default="avoiding-v0", help="Environment name")
    args = parser.parse_args()

    analyzer = LinearDynamicsAnalyzer(env_name=args.env)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
