# Safe multi-robot motion planning in MMD

This folder presents the benchmarks for _DiRecT_, _PCD_, and relevant baselines on the multi-robot motion planning task. The codebase is forked from the [**Projected Coupled Diffusion (PCD)**](https://github.com/EdmundLuan/pcd) [1], which itself extends [MMD](https://github.com/yoraish/mmd) [2]: the planner stack, projection operators, coupling costs, and most of the inference / analysis scripts already exist in the PCD parent. What is contributed here is the **DiRecT** planner on top.

![Multi-robot grid visualization](assets/mrmp-grid.gif)

**Table of Contents**
- [Installation](#installation)
- [Get Datasets](#get-datasets)
- [Generating missing Drop Region dataset](#generating-missing-drop-region-dataset)
- [Planning Inference](#planning-inference)
- [Analyzing Results](#analyzing-results)
- [What we modified/added](#what-we-modifiedadded)
- [Optimizers](#optimizers)
- [Code structure](#code-structure)
- [References](#references)

---

## Installation

Follow the setup steps from the PCD parent ([EdmundLuan/pcd](https://github.com/EdmundLuan/pcd)) as below. In summary:

### Requirements
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

### Steps

1. Create a conda environment and activate it (the env is named `direct-mmd` in `environment.yml`; this installs Python, PyTorch, JAX, and all pip dependencies):
    ```sh
    conda env create -f environment.yml
    conda activate direct-mmd
    ```
2. Install the local (patched) versions of [2, 3]:
    ```sh
    cd deps/torch_robotics      && pip install -e . && cd ../..
    cd deps/experiment_launcher && pip install -e . && cd ../..
    cd deps/motion_planning_baselines && pip install -e . && cd ../..
    ```
3. Install the `mmd` package:
    ```sh
    pip install -e .
    ```

## Get Datasets

Download the sample datasets and trained models provided by the PCD parent:

```sh
conda activate direct-mmd

gdown --id 1Onw0s1pDsMLDfJVOAqmNme4eVVoAkkjz
tar -xJvf data_trajectories.tar.xz

gdown --id 1WO3tpvg-HU0m9RyDvGyfDamo7roBYMud
tar -xJvf data_trained_models.tar.xz
```

> If the gdown IDs above have expired, the current dataset locations are tracked in the PCD parent repo: https://github.com/EdmundLuan/pcd.

## Generating missing Drop Region dataset

The gdown bundle ships the **DropRegion trained model** but not its **trajectory dataset**. The dataset is still required at inference time — only for the normalizer stats (min/max for `[-1, 1]` scaling), not for retraining.

> [!WARNING]
> To run any DropRegion experiment you must first generate this dataset, otherwise inference will fail to load the normalizer. Before running, edit the SLURM partition, GPU resources, and other system-specific settings in the script to match your cluster. Generation may take a while depending on your hardware.

```sh
conda activate direct-mmd
python scripts/generate_data/launch_generate_trajectories.py
```

The launcher has no CLI; edit `env_id` at the top of the file to switch environments (default `'EnvDropRegion2D'`). Output goes to `logs/<timestamp>/...` — move or symlink it into `data_trajectories/EnvDropRegion2D-RobotPlanarDisk/` to match the layout of the other environments.

---

## Planning Inference

All single-agent methods (PCD, DiRecT, baselines) go through `scripts/inference/launch_pcdiff.py`; MMD-CBS uses its own `launch_mmd.py`. Pick a config from `configs/` (`pcdiff_*.yaml` for PCD / baselines, `direct_*.yaml` for DiRecT) and one of the six commands below.

Before the first run, create the results directory:
```sh
mkdir -p results/000event_logs
```

```sh
conda activate direct-mmd

# Unconstrained — vanilla diffusion, no projection
python scripts/inference/launch_pcdiff.py --config pcdiff_empty.yaml -g vanilla -p none -m unconstrained

# PCD-SHD — coupling guidance + ADMM velocity projection, squared-hinge cost
python scripts/inference/launch_pcdiff.py --config pcdiff_empty.yaml -g coupling -p max_vel_admm -c hinge_sqr_l2 -r -m pcd_shd

# PCD-LB — coupling guidance + ADMM velocity projection, log-barrier cost
python scripts/inference/launch_pcdiff.py --config pcdiff_empty.yaml -g coupling -p max_vel_admm -c sum_log_l2 -r -m pcd_lb

# Final projection — unguided diffusion, project only at the last sampling step
python scripts/inference/launch_pcdiff.py --config direct_empty.yaml -g final -p scp_admm -m final_projection

# DiRecT — unguided diffusion + projection every step (use -p scp_admm or -p ipopt)
python scripts/inference/launch_pcdiff.py --config direct_empty.yaml -g base -p scp_admm -m direct

# MMD-CBS — multi-agent baseline (no CLI; edit MultiAgentPlanningExperimentConfig in the file)
python scripts/inference/launch_mmd.py
```

**Optional CLI overrides** (apply to any method above):
- `-n <int>` — number of initial states (default 100), e.g. `-n 2` for a quick test.
- `-v <float>` — velocity cap for projection. Repeatable for sweeps, e.g. `-v 0.675 -v 0.703`.
- `-a <int>` — number of agents (default 4).

Swap `pcdiff_empty.yaml` / `direct_empty.yaml` for the `highway`, `conveyor`, or `dropregion` variants to reproduce the other environments. To launch the full sweep across all methods × environments × agent counts in one go, use `python scripts/inference/run_benchmark.py --help`, or one of the shell wrappers in `scripts/inference/` (`run_full_sweep.sh`, `run_single_method.sh`, `run_all_methods.sh`, `sbatch_benchmark.sh`).

Results land under `results/<EnvRobot>Random/...` (PCD / DiRecT under `agent___N/method___.../cost___...`, MMD-CBS under `<timestamp>/num_agents___N/planner___CBS/...`); event logs are written to `results/000event_logs/`.

---

## Analyzing Results

PCD/DiRecT and MMD-CBS produce different output layouts, so they have separate analyzers:

```sh
# PCD / DiRecT results (any run produced by launch_pcdiff.py)
python scripts/analyze/analyzing_pcd.py --config analysis_config.yaml

# MMD-CBS results (auto-discover + post-process + analyze)
python scripts/analyze/run_all_mmd_postprocessing.py
```

Point the analyzer at the runs you want by editing `log_paths` in `configs/analysis_config.yaml` (PCD/DiRecT) or `configs/analyze_mmd.yaml` (MMD-CBS).

---

## What we modified/added

**Inherited from PCD (and from MMD before it):**
- `mmd/models/projection/` — ADMM-based state-space projection operators.
- `mmd/coupling_costs/` — pair-wise inter-robot cost functions.
- `mmd/models/diffusion_models/sample_functions.py::projected_coupling_ddpm_sample_fn` — projection-aware DDPM sampling hook.
- `mmd/models/diffusion_models/guides.py` — `GuideManager` framework for gradient-based coupling guidance.
- `mmd/planners/single_agent/` — `MPD`, `MPDEnsemble`, `PCD` planners (upstream MMD + PCD).
- `mmd/planners/multi_agent/` — `CBS`, `PrioritizedPlanning` multi-agent planners (upstream MMD).
- `mmd/common/` — conflict bookkeeping (`conflict_conversion.py`, `multi_agent_utils.py`, `constraints.py`, `experiences.py`).
- `configs/pcdiff_*.yaml`, `launch_pcdiff.py` / `launch_mmd.py`, analysis scripts under `scripts/analyze/`, and data-generation scripts under `scripts/generate_data/`.

**Added by this fork (DiRecT):**
- `mmd/planners/single_agent/direct.py` — new `DiRecT` (unguided diffusion + projection) planner class.
- `mmd/planners/single_agent/` — new `IpoptProjector` and `JaxScpAdmmProjector` projector classes.
- `mmd/models/diffusion_models/` — additional sampling-loop wiring to call the new projectors from the `-g base/final` branches.
- `mmd/config/mmd_params.py` — new entries for the DiRecT hyperparameters.
- New DiRecT branches in `scripts/inference/launch_pcdiff.py` — the `-g {base, final}` guiders and the `-p {ipopt, scp_admm}` projectors.
- New configs `configs/direct_{empty,highway,conveyor,dropregion}.yaml`.
- New benchmark methods `direct` and `final_projection` in `scripts/inference/run_benchmark.py`, plus shell helpers (`run_all_methods.sh`, `run_full_sweep.sh`, `run_single_method.sh`, and SLURM wrappers).
- Local patches to `deps/torch_robotics/` and other vendored deps for JAX SCP-ADMM and IPOPT projector compatibility.

## Optimizers

The planning stack exposes three families of hard-projection optimizers, picked at the command line via `-p, --projector`:

- **ADMM velocity projection** *(from the PCD parent)* — implemented in `mmd/models/projection/projectors.py` as `ADMMProjectionOperator` and the GPU-accelerated `ADMMProjectionOperatorCUDAGraph`. These solve the convex problem of projecting a candidate trajectory onto the bounded-velocity set (`‖x_{i+1} − x_i‖₂ ≤ v_max·Δt`) by alternating an x-update (linear solve with cached LU / Cholesky factorization), a z-update (Euclidean L2-ball projection), and a u-update (dual-variable ascent), with primal/dual residual convergence checks. Selected via `-p max_vel_admm`.
- **JAX Sequential Convex Programming + ADMM** *(DiRecT)* — implemented in `JaxScpAdmmProjector` (`mmd/planners/single_agent/`). It linearizes harder constraints (obstacles / dynamics) via SCP and runs ADMM on each linearized subproblem in JAX, JIT-compiled for batched trajectories. Selected via `-p scp_admm`; this is the default projector for DiRecT and for the `final_projection` method.
- **IPOPT interior-point** *(DiRecT)* — implemented in `IpoptProjector` as a general-purpose alternative when ADMM-based projectors are not flexible enough for a given constraint set. Selected via `-p ipopt`.

---

## Code structure

Only DiRecT-added and implementation-critical files are listed; everything else inherited from PCD/MMD is omitted for brevity.

```
multi_robot/
├── configs/
│   ├── direct_{empty,highway,conveyor,dropregion}.yaml      # NEW: DiRecT configs
│   └── pcdiff_{empty,highway,conveyor,dropregion}.yaml      # PCD / CD configs (inherited)
├── mmd/
│   ├── planners/single_agent/
│   │   └── direct.py                                        # NEW: DiRecT, IpoptProjector, JaxScpAdmmProjector
│   ├── models/projection/projectors.py                      # ADMM projection operators (inherited)
│   └── coupling_costs/                                      # inter-robot coupling costs (inherited)
├── scripts/
│   ├── inference/
│   │   ├── launch_pcdiff.py                                 # main entrypoint (PCD / DiRecT / baselines)
│   │   ├── launch_mmd.py                                    # MMD-CBS entrypoint (edit-the-file)
│   │   └── run_benchmark.py                                 # full sweep across methods × envs × #agents
│   ├── generate_data/launch_generate_trajectories.py        # generate DropRegion (or other) datasets
│   └── analyze/{analyzing_pcd.py, run_all_mmd_postprocessing.py}
├── deps/torch_robotics/                                     # local fork (patched for SCP-ADMM / IPOPT)
├── environment.yml                                          # conda env: direct-mmd
└── setup.py                                                 # `pip install -e .`  →  package `mmd`
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
- [1] Hao Luan. Projected Coupled Diffusion (PCD) for multi-robot motion planning. Implementation extending MMD. https://github.com/EdmundLuan/pcd
- [2] Y. Shaoul, I. Mishani, S. Vats, J. Li, M. Likhachev. Multi-Robot Motion Planning with Diffusion Models. ICLR 2025. https://github.com/yoraish/mmd
