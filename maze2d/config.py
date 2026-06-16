"""Dataclass-based configuration system replacing Hydra."""

import argparse
import copy
import os
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class DatasetConfig:
    env_name: str = "maze2d-large-v1"
    horizon: int = 384
    normalizer: str = "LimitsNormalizer"
    max_path_length: int = 40000
    max_n_episodes: int = 1200
    termination_penalty: float = 0.0
    preprocess_fns: List[str] = field(default_factory=lambda: ["maze2d_set_terminals"])


@dataclass
class DiffusionConfig:
    prediction_type: str = "data"
    lr: float = 1e-3
    scheduler_T_max: int = 10000
    noise_schedule_kwargs: Dict[str, Any] = field(default_factory=dict)
    antithetic_sampling: bool = False
    clip_denoised: bool = True
    dim: int = 32
    dim_mults: Tuple[int, ...] = (1, 4, 8)


@dataclass
class TrainConfig:
    accelerator: str = "auto"
    devices: int = 1
    n_train_steps: int = 2_000_000
    gradient_accumulation_steps: int = 1
    log_every_n_steps: int = 10
    batch_size: int = 32
    num_workers: int = 0
    enable_checkpointing: bool = True
    ckpt_save_freq: int = 50000
    ema_decay: float = 0.995
    render_freq: int = 10000
    render_n_samples_per_condition: int = 5
    render_n_batches: int = 3
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    compile_disable: bool = False


@dataclass
class EvalConfig:
    num_random_repeats: int = 10
    device: str = "cuda"
    use_float64: bool = True
    fixed_start: bool = True
    verbose_saving: bool = False
    save_generation: bool = False
    save_trajectories: bool = False
    batch_size: int = 1
    dynamics_path: Optional[str] = None
    weights_path: Optional[str] = None


@dataclass
class PolicyConfig:
    obstacle_type: str = "broad"
    sampling_steps: int = 32
    time_discretization: str = "uniform"
    time_discretization_kwargs: Dict[str, Any] = field(default_factory=lambda: {"prior_time_offset": 0.001})
    sampler_interpolation_coeff: float = 1.0
    env_objective: List[str] = field(default_factory=lambda: ["distance"])
    env_constraints: List[str] = field(
        default_factory=lambda: ["ellipses", "dynamics", "input_saturation"]
    )
    proxy_objective_value_weight: float = 2.0
    proxy_constraint_penalty_weight: float = 0.1
    constraint_margin: float = 0.0
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    seed: int = 0
    log_folder: str = "logs"
    exp_name: str = "debug"
    overwrite: bool = True
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)


# Maps nested dataclass field names to their types
_NESTED_CONFIGS = {
    "dataset": DatasetConfig,
    "diffusion": DiffusionConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "policy": PolicyConfig,
}


def _dict_to_dataclass(cls, d: dict):
    """Recursively construct a dataclass from a dict, ignoring unknown keys."""
    if not isinstance(d, dict):
        return d
    field_names = {f.name for f in fields(cls)}
    filtered = {}
    for k, v in d.items():
        if k not in field_names:
            continue
        if k in _NESTED_CONFIGS:
            filtered[k] = _dict_to_dataclass(_NESTED_CONFIGS[k], v)
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(yaml_path: str = "configs/config.yaml") -> Config:
    """Load config from YAML file, falling back to dataclass defaults for missing keys."""
    if os.path.exists(yaml_path):
        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    return _dict_to_dataclass(Config, raw)


def parse_args(description: str = "") -> argparse.Namespace:
    """Parse common CLI arguments that override YAML config values."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--log-folder", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--no-overwrite", action="store_false", dest="overwrite")
    # Eval
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--dynamics-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-random-repeats", type=int, default=None)
    # Policy
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--obstacle-type", type=str, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    # Train
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-train-steps", type=int, default=None)
    return parser.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI argument overrides to the config (mutates cfg)."""
    if args.seed is not None:
        cfg.seed = args.seed
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.log_folder is not None:
        cfg.log_folder = args.log_folder
    if args.overwrite is not None:
        cfg.overwrite = args.overwrite
    # Eval
    if args.weights_path is not None:
        cfg.eval.weights_path = args.weights_path
    if args.dynamics_path is not None:
        cfg.eval.dynamics_path = args.dynamics_path
    if args.device is not None:
        cfg.eval.device = args.device
    if args.num_random_repeats is not None:
        cfg.eval.num_random_repeats = args.num_random_repeats
    # Policy (--policy loads a policy YAML that overrides policy section)
    if hasattr(args, 'policy') and args.policy is not None:
        load_policy_config(args.policy, cfg)
    if args.obstacle_type is not None:
        cfg.policy.obstacle_type = args.obstacle_type
    if args.sampling_steps is not None:
        cfg.policy.sampling_steps = args.sampling_steps
    # Train
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.n_train_steps is not None:
        cfg.train.n_train_steps = args.n_train_steps
    return cfg


def load_policy_config(policy_name: str, cfg: Config) -> Config:
    """Load a policy YAML and merge its values into cfg.policy."""
    policy_path = os.path.join(os.path.dirname(__file__), "configs", "policies", f"{policy_name}.yaml")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy config not found: {policy_path}")
    with open(policy_path) as f:
        policy_raw = yaml.safe_load(f) or {}
    # Merge into cfg.policy: policy YAML values override main config
    for k, v in policy_raw.items():
        if hasattr(cfg.policy, k):
            setattr(cfg.policy, k, v)
    return cfg


def save_config(cfg: Config, path: str):
    """Save config to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False, sort_keys=False)
