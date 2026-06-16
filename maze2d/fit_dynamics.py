import os
from typing import Any, Tuple, Dict, Optional

import numpy as np
from numpy.typing import NDArray
import yaml

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

import warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*")

from config import Config, load_config, parse_args, apply_overrides
from d4rl.sequence import GoalDataset


def collect_maze2d_data(
    cfg: Config,
) -> Tuple[NDArray, NDArray, NDArray, Dict[str, Any], GoalDataset]:

    dataset = GoalDataset(
        env=cfg.dataset.env_name,
        horizon=cfg.dataset.horizon,
        normalizer=cfg.dataset.normalizer,
        preprocess_fns=cfg.dataset.preprocess_fns,
        max_path_length=cfg.dataset.max_path_length,
        max_n_episodes=cfg.dataset.max_n_episodes,
        termination_penalty=cfg.dataset.termination_penalty,
        seed=cfg.seed,
    )

    buffer = dataset.fields

    x_list = []
    y_list = []
    actions_list = []

    for path_idx in range(buffer.n_episodes):
        path_length = buffer.path_lengths[path_idx]  # type: ignore
        observations = buffer.normed_observations[path_idx, :path_length]  # type: ignore
        actions = buffer.normed_actions[path_idx, :path_length]  # type: ignore

        for t in range(path_length - 1):
            x_list.append(observations[t])
            y_list.append(observations[t + 1])
            actions_list.append(actions[t])

    x = np.array(x_list)
    y = np.array(y_list)
    actions = np.array(actions_list)

    metadata = {
        "env_name": cfg.dataset.env_name,
        "num_transitions": len(x),
        "obs_dim": x.shape[1],
        "action_dim": actions.shape[1] if len(actions) > 0 else 0,
        "num_episodes": buffer.n_episodes,
    }

    return x, y, actions, metadata, dataset


# s_(i+1) = A * s_i + B * a_i + c
def fit_linear_dynamics_model(
    X: NDArray, y: NDArray, actions: NDArray
) -> Tuple[Any, NDArray, NDArray, NDArray, Dict[str, Any], NDArray]:
    features = np.hstack([X, actions])

    models = []
    predictions = []
    for i in range(y.shape[1]):
        model = LinearRegression()
        model.fit(features, y[:, i])
        models.append(model)
        predictions.append(model.predict(features))

    y_pred = np.column_stack(predictions)

    obs_dim = X.shape[1]
    action_dim = actions.shape[1]

    A = np.array([model.coef_[:obs_dim] for model in models])
    B = np.array([model.coef_[obs_dim:] for model in models])
    c = np.array([model.intercept_ for model in models])

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    dim_mse = [mean_squared_error(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]
    dim_r2 = [r2_score(y[:, i], y_pred[:, i]) for i in range(y.shape[1])]

    metrics = {
        "overall_mse": mse,
        "overall_r2": r2,
        "dim_mse": dim_mse,
        "dim_r2": dim_r2,
        "rmse": np.sqrt(mse),
    }

    return models, A, B, c, metrics, y_pred


def analyze_linear_model(
    A: NDArray,
    B: NDArray,
    c: NDArray,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
):

    print(f"\n=== Linear Dynamics Model Analysis for {metadata['env_name']} ===")
    print(f"Dataset: {metadata['num_transitions']} transitions")
    print(
        f"Observation dim: {metadata['obs_dim']}, Action dim: {metadata['action_dim']}"
    )

    print(f"\n=== Model Performance ===")
    print(f"Overall R²: {metrics['overall_r2']:.4f}")
    print(f"Overall RMSE: {metrics['rmse']:.4f}")

    print(f"\n=== Per-dimension Performance ===")
    for i, (r2, mse) in enumerate(zip(metrics["dim_r2"], metrics["dim_mse"])):
        print(f"Dimension {i}: R² = {r2:.4f}, MSE = {mse:.4f}")

    print(f"\n=== State Transition Matrix A (s_{{t+1}} dependence on s_t) ===")
    print("Shape:", A.shape)
    eigenvals = np.linalg.eigvals(A)
    print("Max absolute eigenvalue:", np.max(np.abs(eigenvals)))
    print("Spectral radius:", np.max(np.abs(eigenvals)))
    print("Frobenius norm:", np.linalg.norm(A, "fro"))

    print(f"\n=== Action Influence Matrix B (s_{{t+1}} dependence on a_t) ===")
    print("Shape:", B.shape)
    print("Frobenius norm:", np.linalg.norm(B, "fro"))

    print(f"\n=== Bias Vector c ===")
    print("Shape:", c.shape)
    print("Norm:", np.linalg.norm(c))

    max_eig = np.max(np.abs(eigenvals))
    if max_eig < 1.0:
        print(
            f"\n✓ System appears stable (max eigenvalue magnitude: {max_eig:.4f} < 1.0)"
        )
    else:
        print(
            f"\n⚠ System may be unstable (max eigenvalue magnitude: {max_eig:.4f} >= 1.0)"
        )


def visualize_predictions(
    y_true: NDArray,
    y_pred: NDArray,
    metadata: Dict[str, Any],
    save_path: Optional[str] = None,
):

    obs_dim = y_true.shape[1]
    n_samples = min(10000, len(y_true))
    idx = np.random.choice(len(y_true), n_samples, replace=False)

    y_true_sub = y_true[idx]
    y_pred_sub = y_pred[idx]

    _, axes = plt.subplots(2, (obs_dim + 1) // 2, figsize=(12, 8))
    if obs_dim == 1:
        axes = [axes]
    elif obs_dim == 2:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i in range(obs_dim):
        ax = axes[i]
        ax.scatter(y_true_sub[:, i], y_pred_sub[:, i], alpha=0.6, s=1)
        ax.plot(
            [y_true_sub[:, i].min(), y_true_sub[:, i].max()],
            [y_true_sub[:, i].min(), y_true_sub[:, i].max()],
            "r--",
            lw=2,
        )
        ax.set_xlabel(f"True Next Obs Dim {i}")
        ax.set_ylabel(f"Predicted Next Obs Dim {i}")
        ax.set_title(f"Dim {i}: R² = {r2_score(y_true[:, i], y_pred[:, i]):.3f}")
        ax.grid(True, alpha=0.3)

    for i in range(obs_dim, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Linear Model Predictions vs True Values\n{metadata["env_name"]}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()


def fit_maze2d(cfg: Config):
    output_dir = os.path.join(cfg.log_folder, "dynamics")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {cfg.dataset.env_name}...")
    X, y, actions, metadata, dataset = collect_maze2d_data(cfg)

    print(f"Fitting linear dynamics model...")
    models, A, B, c, metrics, y_pred = fit_linear_dynamics_model(X, y, actions)

    analyze_linear_model(A, B, c, metadata, metrics)

    viz_path = os.path.join(output_dir, "linear_model_predictions.png")
    visualize_predictions(y, y_pred, metadata, viz_path)

    model_path = os.path.join(output_dir, "linear_model.npz")
    np.savez(
        model_path,
        A=A,
        B=B,
        c=c,
        metadata=metadata,  # type: ignore
        metrics=metrics,  # type: ignore
        normalizer=dataset.normalizer,  # type: ignore
    )
    print(f"Model saved to: {model_path}")

    print(f"\n=== Summary ===")
    print(f"Fitted linear dynamics model for {metadata['env_name']}")
    print(f"Model form: s_{{t+1}} = A·s_t + B·a_t + c")
    print(
        f"Overall performance: R² = {metrics['overall_r2']:.4f}, RMSE = {metrics['rmse']:.4f}"
    )
    print(f"Files saved to: {output_dir}/")


def main():
    args = parse_args(description="Fit linear dynamics model")
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    print(yaml.dump(cfg.__dict__, default_flow_style=False))
    fit_maze2d(cfg)


if __name__ == "__main__":
    main()
