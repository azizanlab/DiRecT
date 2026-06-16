import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict

import numpy
import random
import torch
import yaml
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

import warnings
warnings.filterwarnings("ignore", message=".*Overriding environment.*")

from config import Config, load_config, parse_args, apply_overrides, save_config
from diffusion import Diffusion, apply_conditioning
from unet import TemporalUnet
from dataset import Maze2dGoalDataModule
from d4rl.sequence import GoalDataset
from env import Maze2dConstrainedEnv


@dataclass
class RenderCallback(pl.Callback):
    env: Any
    every_n_steps: int
    out_dir: str
    n_samples_per_condition: int = 2
    n_batches: int = 2

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del trainer, pl_module
        os.makedirs(self.out_dir, exist_ok=True)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        del batch_idx
        step = trainer.global_step
        if self.every_n_steps <= 0 or (step % self.every_n_steps) != 0:
            return
        if trainer.sanity_checking:
            return

        x = batch["x"].to(pl_module.device)  # (B, H, T)
        conditions: Dict[Any, torch.Tensor] = {
            k: v.to(pl_module.device) for k, v in batch["conditions"].items()
        }

        B = x.shape[0]
        take = min(self.n_batches, B)

        for i in range(take):
            cond_i = {k: v[i : i + 1] for k, v in conditions.items()}
            cond_rep = {
                k: v.repeat(self.n_samples_per_condition, 1) for k, v in cond_i.items()
            }

            # Build prior-like noise of shape (n_samples, H, T)
            H, T = x.shape[1], x.shape[2]
            prior = torch.empty(
                (self.n_samples_per_condition, H, T),
                device=pl_module.device,
                dtype=x.dtype,
            )

            x0 = pl_module.sample_noise_like(prior)  # type: ignore
            samples, _ = pl_module.deterministic_ddim_default_sampler(  # type: ignore
                x0=x0,
                num_steps=100,
                conditions=cond_rep,
            )  # (n_samples, H, T)

            savepath = os.path.join(self.out_dir, f"sample-step{step}-i{i}.png")
            self.env.render_maze(
                batch_observations=samples[:, :, 2:],
                savepath=savepath,
                conditions=conditions,
                plot_obstacles=False,
                ncols=samples.shape[0],
                normalized_obs=True
            )


def train_maze2d(cfg: Config):

    log_path = os.path.join(cfg.log_folder, cfg.exp_name, "train")

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

    datamodule = Maze2dGoalDataModule(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    env = Maze2dConstrainedEnv(
        env_dataset=dataset, obstacle_types="broad", objective=[], constraints=[]
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

    # Pytorch compilation
    # diffuser = torch.compile(
    #     diffuser,
    #     fullgraph=cfg.train.compile_fullgraph,
    #     dynamic=cfg.train.compile_dynamic,
    #     disable=cfg.train.compile_disable,
    # )

    wandb_logger = WandbLogger(
        name=cfg.exp_name,
        save_dir=log_path,
        project="DiRecT-maze2d",
        offline=False
    )

    # callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(log_path, "checkpoints"),
        filename="ema-{step}",
        every_n_train_steps=cfg.train.ckpt_save_freq,
        save_top_k=-1,
        save_last=True,
    )
    ema_cb = WeightAveraging(avg_fn=get_ema_avg_fn(cfg.train.ema_decay))
    render_cb = RenderCallback(
        env=env,
        every_n_steps=cfg.train.render_freq,
        out_dir=os.path.join(log_path, "render"),
        n_samples_per_condition=cfg.train.render_n_samples_per_condition,
        n_batches=cfg.train.render_n_batches,
    )

    callbacks = [ckpt_cb, ema_cb, render_cb]

    # train
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_steps=cfg.train.n_train_steps,
        accumulate_grad_batches=cfg.train.gradient_accumulation_steps,
        logger=[wandb_logger],
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_every_n_steps,
        enable_checkpointing=cfg.train.enable_checkpointing,
    )

    trainer.fit(diffuser, datamodule=datamodule)


## Main ##

def deterministic(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args(description="Train diffusion model")
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    print(yaml.dump(cfg.__dict__, default_flow_style=False))
    deterministic(cfg.seed)
    log_path = os.path.join(cfg.log_folder, cfg.exp_name, "train")
    if cfg.overwrite and os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    save_config(cfg, os.path.join(log_path, "config.yaml"))
    train_maze2d(cfg)


if __name__ == "__main__":
    main()


