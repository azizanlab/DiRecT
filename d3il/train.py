from itertools import cycle
import math
import os
import torch
import tyro
import numpy as np
from dataclasses import dataclass, field

from datasets.sequence import SequenceDataset
from model import TemporalUnet, CosineScheduleDiffusion, apply_conditioning
from utils.arrays import batch_to_device, to_np, to_torch, apply_dict
from utils.training import deterministic, set_cuda_visible_device, save_config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import tqdm
import wandb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class TrainingConfig:
    seed: int = 0
    device: str = "cuda:0"
    log_folder: str = "logs"
    exp_name: str = "diffusion"

    env: str = "avoiding-v0"
    horizon: int = 16
    normalizer: str = "LimitsNormalizer"
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 200
    max_n_episodes: int = 500
    termination_penalty: float = 0

    state_dim: int = 4
    action_dim: int = 2

    # training
    n_train_steps: int = 1000001
    save_freq: int = 50000
    batch_size: int = 32
    learning_rate: float = 2e-4

    ema_decay: float = 0.995

    sample_freq: int = 5000
    n_sample_steps: int = 20
    n_samples: int = 8

    resume: bool = False

    wandb_project: str = "direct-d3il"
    wandb_entity: str = ""


@torch.no_grad()
def generate_samples(ema_model, diffusion, normalizer, dataset, cfg):
    """Generate DDIM samples from the EMA model for visualization (no conditioning)."""
    ema_model.eval()
    action_dim = cfg.action_dim
    state_dim = cfg.state_dim
    horizon = cfg.horizon
    transition_dim = action_dim + state_dim
    device = cfg.device

    # grab a real initial observation from the dataset
    sample = dataset[0]
    obs_0 = sample.conditions[0]  # already normalized
    conditions = {0: obs_0}

    x = torch.randn(cfg.n_samples, horizon, transition_dim, device=device)
    x = apply_conditioning(x, to_torch(conditions, device=device), action_dim)

    timesteps = torch.linspace(1.0, 0.0, cfg.n_sample_steps + 1, device=device)

    for i in range(cfg.n_sample_steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        abar_now = diffusion.alpha_bar(t_now)
        abar_next = diffusion.alpha_bar(t_next)

        t_batch = t_now.expand(cfg.n_samples)
        x0_pred = ema_model(x, t_batch)

        # DDIM (deterministic): x_{t_next} = √ᾱ_{t_next} * x0_pred + √(1-ᾱ_{t_next}) * ε_pred
        # where ε_pred = (x - √ᾱ_t * x0_pred) / √(1-ᾱ_t)
        eps_pred = (x - math.sqrt(abar_now) * x0_pred) / math.sqrt(1.0 - abar_now)
        x = math.sqrt(abar_next) * x0_pred + math.sqrt(1.0 - abar_next) * eps_pred

        # only condition the start point (not the endpoint)
        x = apply_conditioning(x, to_torch(conditions, device=device), action_dim)

    # unnormalize observations for plotting
    obs = normalizer.unnormalize(to_np(x[:, :, action_dim:]), "observations")
    return obs  # (n_samples, horizon, state_dim)


def plot_samples(observations, save_path):
    """Plot sampled trajectories on the obstacle field."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim([0.2, 0.8])
    ax.set_ylim([-0.3, 0.4])
    ax.set_facecolor([1, 1, 0.9])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # target
    ax.add_patch(patches.Rectangle((0.2, 0.35), 0.6, 0.05,
                                    facecolor=[0.4, 1, 0.4], alpha=0.35))
    # pillars
    for c, r in [([0.5,-0.1],0.03), ([0.425,0.08],0.025), ([0.575,0.08],0.025),
                  ([0.35,0.26],0.025), ([0.5,0.26],0.025), ([0.65,0.26],0.025)]:
        ax.add_patch(patches.Circle(c, r, color="r", alpha=0.7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(observations)))
    for i, obs in enumerate(observations):
        x_pos = obs[:, 2]  # actual_x
        y_pos = obs[:, 3]  # actual_y
        ax.plot(x_pos, y_pos, "-o", color=colors[i], linewidth=1.5, markersize=3, alpha=0.8)
        ax.plot(x_pos[0], y_pos[0], "s", color="green", markersize=5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def train(cfg: TrainingConfig, log_subfolder: str):
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

    unet = TemporalUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
    ).to(cfg.device)

    diffusion = CosineScheduleDiffusion(
        model=unet,
        action_dim=cfg.action_dim,
    )

    train_loader = cycle(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True))

    optimizer = torch.optim.Adam(unet.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.save_freq * 2
    )
    ema = torch.optim.swa_utils.AveragedModel(
        unet,
        avg_fn=lambda avg, new, num: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new,
    )

    # resume from latest checkpoint
    start_step = 0
    if cfg.resume:
        import glob
        ckpts = sorted(glob.glob(os.path.join(log_subfolder, "model_*.pth")))
        ckpts = [c for c in ckpts if "ema" not in c]
        if ckpts:
            latest = ckpts[-1]
            cp_idx = int(latest.split("model_")[-1].replace(".pth", ""))
            start_step = cp_idx * cfg.save_freq

            unet.load_state_dict(torch.load(latest))
            ema_path = latest.replace("model_", "model_ema_")
            if os.path.exists(ema_path):
                ema.module.load_state_dict(torch.load(ema_path))

            opt_path = os.path.join(log_subfolder, "optimizer.pth")
            if os.path.exists(opt_path):
                opt_state = torch.load(opt_path)
                optimizer.load_state_dict(opt_state["optimizer"])
                scheduler.load_state_dict(opt_state["scheduler"])

            print(f"Resumed from checkpoint {cp_idx} (step {start_step})")
        else:
            print("No checkpoints found, starting from scratch")

    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, "tensorboard_logs"))

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity or None,
        name=cfg.exp_name,
        config=vars(cfg),
        resume="allow" if cfg.resume else None,
    )

    samples_dir = os.path.join(log_subfolder, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    # skip dataloader to the right position
    if start_step > 0:
        print(f"Skipping dataloader to step {start_step}...")
        for _ in tqdm.tqdm(range(start_step), desc="Fast-forwarding"):
            next(train_loader)

    for i in tqdm.tqdm(range(start_step, cfg.n_train_steps)):
        batch = batch_to_device(next(train_loader), cfg.device)
        loss, infos = diffusion.loss(*batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        ema.update_parameters(unet)

        if i % cfg.save_freq == 0:
            torch.save(
                unet.state_dict(),
                os.path.join(log_subfolder, f"model_{i // cfg.save_freq}.pth"),
            )
            torch.save(
                ema.module.state_dict(),
                os.path.join(log_subfolder, f"model_ema_{i // cfg.save_freq}.pth"),
            )
            torch.save(
                {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
                os.path.join(log_subfolder, "optimizer.pth"),
            )

        if i % cfg.sample_freq == 0:
            obs = generate_samples(ema.module, diffusion, dataset.normalizer, dataset, cfg)
            sample_path = os.path.join(samples_dir, f"step_{i:07d}.png")
            plot_samples(obs, sample_path)
            wandb.log({"samples": wandb.Image(sample_path)}, step=i)

        writer.add_scalar("loss", loss, i)
        wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=i)

    wandb.finish()


if __name__ == "__main__":
    cfg = tyro.cli(TrainingConfig)

    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "diffusion", cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
