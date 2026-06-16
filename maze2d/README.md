# Safe maze navigation in Maze2D

This folder presents the benchmarks for _DiRecT_ and relevant baselines on the Maze2D navigation with dynamics constraints.
![Maze2D visualization](assets/maze2d-comparison.gif)

**Table of Contents**:
- [Installation](#installation)
- [Fit the linearized dynamics](#fit-the-linearized-dynamics)
- [Train the model](#train-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Code structure](#code-structure)
- [References](#references)

## Installation
> [!NOTE]
> The `maze2d-large-v1` env has been ported from the old mujoco to the newer version, as well as gym to gymnasium, therefore a local mujoco installation is not required. All mujoco packages are installed through the conda environment.

1. Change the CUDA and torch versions in `environment.yml` to ones compatible with your system requirements.
2. Create and activate the conda environment (works with `conda`, `mamba`, or `miniconda` / `micromamba`):
```bash
# with conda (or replace `conda` by `mamba` / `micromamba` for faster solves)
conda env create -f environment.yml
conda activate direct-maze2d
```
> [!TIP]
> If you have packages installed in your user site-packages (e.g. `transformers` in `~/.local/lib/pythonX.Y/`), they can conflict with the conda environment. To prevent this, run `export PYTHONNOUSERSITE=1` before executing any commands below.

## Fit the linearized dynamics
Fit a linear dynamics model `s_{t+1} = A·s_t + B·a_t + c` from the D4RL `maze2d-large-v1` trajectories. The fitted matrices are later imposed as linear constraints during planning.

```bash
# base YAML config (dataset / seed / log_folder)
# --seed: RNG seed for dataset construction
# --log-folder: output root; model is written to <log-folder>/dynamics/
python fit_dynamics.py \
    --config configs/config.yaml \
    --seed 0 \
    --log-folder logs
```
Outputs:
- `logs/dynamics/linear_model.npz` — fitted `A`, `B`, `c`, metadata, and normalizer (consumed by `eval.py`).
- `logs/dynamics/linear_model_predictions.png` — per-dimension true-vs-predicted scatter plot.

## Train the model
Train the unconstrained diffusion planner (Temporal U-Net) on the same D4RL dataset. Weights & Biases is used for logging (project `DiRecT-maze2d`).

```bash
# base config (diffusion + train sections)
# --exp-name: subfolder under <log-folder> for this run
# --batch-size: overrides train.batch_size
# --n-train-steps: overrides train.n_train_steps
python train.py \
    --config configs/config.yaml \
    --exp-name maze2d-diffuser \
    --seed 0 \
    --batch-size 256 \
    --n-train-steps 2000000
```
Checkpoints (EMA weights), render previews, and the resolved config are written to `logs/<exp-name>/train/`:
- `checkpoints/ema-step=*.ckpt`, `checkpoints/last.ckpt` — saved every `train.ckpt_save_freq` steps.
- `render/sample-step*-i*.png` — periodic sample previews from the current model.
- `config.yaml` — snapshot of the fully-resolved config.

## Evaluation
Evaluate any of the planning policies on the constrained Maze2D task. The trained diffusion weights and fitted dynamics are both required.

```bash
# --policy: one of the policies listed below
# --obstacle-type: `broad` or `narrow`
# --dynamics-path: defaults to <log-folder>/dynamics/linear_model.npz
# --num-random-repeats: number of evaluation rollouts
python eval.py \
    --config configs/config.yaml \
    --policy direct \
    --obstacle-type broad \
    --weights-path logs/maze2d-diffuser/train/checkpoints/last.ckpt \
    --dynamics-path logs/dynamics/linear_model.npz \
    --exp-name eval/direct-broad \
    --num-random-repeats 100 \
    --seed 0 \
    --device cuda
```

Available policies (each is a YAML in `configs/policies/`): `no-guidance`, `classifier-guidance`, `gradient-guidance`, `projection`, `primal-dual`, `augmented-lagrangian`, `safediffuser-ros`, `safediffuser-res`, `safediffuser-tvs`, `direct`.

To run the full sweep over every policy × `{broad, narrow}` obstacle setting, use the provided SLURM script (edit the `#SBATCH --partition`, conda env name, and `WEIGHTS` path first):
```bash
sbatch benchmark.sh
```

## Results
For each evaluation, results are written under `logs/<exp-name>/eval/`:
- `config.yaml` — resolved config used for the run.
- `statistics.csv` — one row per rollout with per-run metrics (score, reward, total constraint violation, safety flag, generated/rollout metrics, …).
- `summary_statistics.txt` — mean / std / min / max across all `num_random_repeats` rollouts.
- `<run_id>/predicted.png` — diffused plan in the maze.
- `<run_id>/rollout.png` — closed-loop PD-tracked trajectory through the constrained environment.
- `<run_id>/intermediates/`, `<run_id>/data_estimates/` — per-denoising-step renders (only when `eval.save_generation: true`).
- `<run_id>/trajectories.npz` — raw generated and rollout state/action arrays (only when `eval.save_trajectories: true`).

## Code structure
```
maze2d/
├── configs/
│   ├── config.yaml              # base config (dataset / diffusion / train / eval / policy)
│   └── policies/                # per-policy overrides (loaded via `--policy <name>`)
│       ├── direct.yaml
│       ├── no-guidance.yaml
│       ├── classifier-guidance.yaml
│       ├── gradient-guidance.yaml
│       ├── projection.yaml
│       ├── primal-dual.yaml
│       ├── augmented-lagrangian.yaml
│       └── safediffuser-{ros,res,tvs}.yaml
├── d4rl/                        # ported D4RL maze2d env + dataset utilities (gymnasium / mujoco 3.x)
├── config.py                    # dataclass-based config + CLI argument parser
├── dataset.py                   # `Maze2dGoalDataModule` (Lightning datamodule)
├── diffusion.py                 # `Diffusion` Lightning module + samplers (DDIM, etc.)
├── unet.py                      # `TemporalUnet` score network
├── env.py                       # `Maze2dConstrainedEnv` — obstacles, dynamics & saturation constraints, metrics, rendering
├── policy.py                    # all planning policies (`POLICY_DICT`): DiRecT + baselines
├── fit_dynamics.py              # fit `s_{t+1} = A s_t + B a_t + c` from offline data
├── train.py                     # train the diffusion planner
├── eval.py                      # rollout + metric aggregation for a chosen policy
├── benchmark.sh                 # SLURM script sweeping all policies × obstacle types
└── environment.yml              # conda environment spec
```

## ✏️ Citation 
If you find our code or paper useful for your research, please consider citing our work:
```
@misc{giaretta2026directsafediffusionbasedplanning,
      title={DiRecT: Safe Diffusion-Based Planning via Receding-Horizon Denoising}, 
      author={Paolo Giaretta and Zeyang Li and Navid Azizan},
      year={2026},
      eprint={2606.15359},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.15359}, 
}
```

## References
- Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 9902–9915, 2022.
- Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, volume 34, pages 8780–8794, 2021.
- Jacob K Christopher, Stephen Baek, and Ferdinando Fioretto. Constrained synthesis with projected diffusion models. Advances in Neural Information Processing Systems, 37:89307–89333, 2024.
- Wei Xiao, Tsun-Hsuan Wang, Chuang Gan, Ramin Hasani, Mathias Lechner, and Daniela Rus. Safediffuser: Safe planning with diffusion probabilistic models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=ig2wk7kK9J
- Jichen Zhang, Liqun Zhao, Antonis Papachristodoulou, and Jack Umenberger. Constrained diffusers for safe planning and control, 2025. URL https://arxiv.org/abs/2506.12544.
- Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4rl: Datasets for deep data-driven reinforcement learning, 2020.
