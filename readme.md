# Vision-Based Sim-to-Real Manipulation Pipeline

## Overview

Inspired by the likes of [skild.ai](https://www.youtube.com/watch?v=JQAfxp-FB0I), [NVIDIA](https://www.youtube.com/watch?v=S4tvirlG8sQ), and [Tesla](https://www.youtube.com/watch?v=g6bOwQdCJrc), this project is an investigation into what can be achieved by learning vision-conditioned RL policies on consumer hardware, with the eventual goal of zero-shot transfer to a physical [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robotic arm.

The project is built around two questions:

1. **Can a policy trained purely from a wrist-mounted RGB camera in simulation learn to find and lift an object?**
2. **What kind of vision encoder works best for this kind of task on consumer hardware** — a frozen, off-the-shelf ImageNet backbone (ResNet18), or a small task-specific CNN pretrained from synthetic telemetry?

To answer these reproducibly, the repo provides an end-to-end pipeline (`train → collect → curate → train-cnn`) and a sweep runner that executes Cartesian-product ablations over rewards, vision backbones, and PPO hyperparameters, with automatic per-experiment evaluation and Markdown summaries.

The project enforces experimental rigor and reproducibility throughout: every configuration value (rewards, seeds, network shapes, augmentations, …) is set explicitly in YAML — no silent defaults. Every run writes a self-contained, timestamped artifact directory containing the resolved configs, seed, checkpoints, evaluation metrics, and TensorBoard logs.

## Problem

**Task.** The SO-101 6-DOF arm, equipped with a single [wrist-mounted](https://github.com/TheRobotStudio/SO-ARM100/blob/main/media/UVC_cam_mount_so101.jpg) [camera](https://www.amazon.com/dp/B07ZRJDTBQ), must locate a small [cube](assets/3dprint/black_cube.3mf) on its work surface and lift it to a height of 10 cm within 10 seconds.

**Observations.** Vision features (1024-D, from the chosen vision encoder) and the 6-D joint position vector, normalized to `[0.0, 1.0]`. Privileged state (e.g. cube pose) is hidden from the policy.

**Actions.** 6-D joint position commands, normalized to `[0.0, 1.0]`.

**Episodes.** 10 s per episode (`episode_length_s`). Physics runs at 120 Hz; observations are taken every 2 ticks (`decimation`), giving `10 × 120 / 2 = 600` policy steps per episode.

**Reward.** A phased reward shapes the policy through three stages — *approach*, *grasp*, and *lift* — with bonuses gated on the previous phase reaching saturation, plus out-of-reach and table-touch termination penalties. Reward terms are declarative entries in [`configs/policy_train.yaml`](configs/policy_train.yaml) (see `rewards:`), so individual terms can be enabled, scaled, or ablated entirely from configuration.

## Approach

### Vision encoders

Two vision encoders are supported (selected via `vision_encoder.type` in the env config). Both produce a 1024-D feature vector consumed by the PPO actor and critic; both are **frozen** during PPO training so the policy only optimizes the MLP head on top of stable features.

| `vision_encoder.type` | Description                                                                                                                                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `frozen_resnet18`     | ImageNet-pretrained ResNet18 truncated before the final pooling layer, followed by a SpatialSoftmax → 1024-D feature vector. Used as the project baseline.                                                                                                         |
| `frozen_cnn`          | Small task-specific CNN (4 conv layers + MLP projection) pretrained from synthetic telemetry, also producing a 1024-D feature vector via SpatialSoftmax. Architecture is defined in `configs/cnn_pretrain.yaml` and embedded in the resulting `cnn_checkpoint.pt`. |

> An earlier iteration trained the CNN end-to-end inside the PPO loop. As documented in the experiment journal, the latent representation was unstable (sudden mean/std drift, non-collapsing policy entropy), so that path was removed. The CNN is now exclusively trained via supervised pretraining from synthetic telemetry and used as a frozen feature extractor.

### Synthetic CNN pretraining

The `frozen_cnn` backbone is bootstrapped from a working `frozen_resnet18` policy:

1. **`train`** a policy with the ResNet18 baseline.
2. **`collect`** synthetic (image, telemetry) samples while playing back the trained policy in sim, sampling uniformly across episode timesteps to avoid early-episode bias.
3. **`curate`** rebalances the dataset along two axes — episode timestep and grip-zone distance — to prevent the dominant mid-approach hover regime from drowning out near-contact frames, and produces episode-level train/val/test splits.
4. **`train-cnn`** trains a small CNN with five regression / classification heads on the shared backbone latent:
   - cube position in grip-zone frame (`cube_pos_gz`)
   - gripper-cube alignment scalar
   - cube orientation as 6-D rotation (`cube_rot6d_gz`)
   - cube height above the table (`cube_height_w`)
   - cube visibility (`cube_in_camera_frame`)
5. The resulting `cnn_checkpoint.pt` is then used as the vision encoder in a fresh `train` run with `vision_encoder.type: frozen_cnn`.

The whole loop is driven by a single `pipeline` command (see Usage).

### Domain randomization

To support eventual sim-to-real transfer, the environment randomizes (per-episode, per-env): scene lighting; camera pose; per-frame image augmentation (gaussian noise, brightness, contrast, motion blur, JPEG compression); cube color, size, and starting position; arm starting joint positions; and a configurable set of distractor objects of varied geometry, size, and color.

### Reproducibility

Every run records:

- the fully resolved env / agent / pipeline / sweep YAML actually used,
- the explicit seed (set on `torch`, `numpy`, `random` before any framework initialization),
- a `run_manifest.json` and `cnn_checkpoint_provenance.json` linking checkpoints to the configs and source data that produced them,
- TensorBoard logs and (optionally) per-episode evaluation videos.

If a required configuration field is missing or `null`, validation fails at startup before any expensive work begins.

## Repository layout

```
configs/                YAML configs (env, CNN pretrain, pipeline, sweeps)
scripts/                Top-level CLI entry points (run.py, sweep.py, pipeline.py)
so101/                  Shared library (CNN training, curation, image utils, viz)
so101_rl/               Isaac Lab task package (So101-LiftCube-v0)
vision_backbone_demo/   Standalone visualizer for backbone activations / keypoints
artifacts/              Default output dir (timestamped per-run subdirs)
notes/experiment_journal/  Running log of experiments and findings
```

## System Requirements

- **NVIDIA Isaac Sim** (https://developer.nvidia.com/isaac-sim)
- **Isaac Lab** (https://isaac-sim.github.io/IsaacLab/)
- CUDA GPU with ≥16 GB VRAM (24 GB recommended for sweeps with on-screen video capture)

## Installation

```bash
pip install -e .   # installs the so101 library (image processing, CNN training, visualizers)
```

Set the `ISAAC_LAB_PATH` environment variable to your Isaac Lab installation before running Isaac-dependent commands (`train`, `collect`, `eval`, `play`, `export`, `pipeline`, `sweep`).

## Configuration

### Env config (`configs/policy_train.yaml`)

The single source of truth for everything that affects the observation space and the simulation: physics, joints, gripper geometry, cameras, image pipeline, distractors, domain randomization, declarative reward terms, termination conditions, and `vision_encoder.type`. Validated at startup against a typed dataclass hierarchy; unknown or missing keys raise immediately.

### CNN pretrain config (`configs/cnn_pretrain.yaml`)

Drives the supervised CNN pipeline and defines the `frozen_cnn` backbone architecture used at both pretraining and PPO time:

- **Curation**: histogram rebalancing across episode timesteps and grip-zone distance bins, episode-level train/val/test split fractions, per-episode sample caps.
- **Backbone**: layer shapes (channels, kernel sizes, strides, MLP head dims, output dim). Embedded in the resulting `cnn_checkpoint.pt` so PPO `train` runs reconstruct the architecture directly from the checkpoint.
- **Heads**: per-head MLP shapes and target normalization constants.
- **Image normalization**: explicit mean/std or `null` (the from-scratch CNN trains on un-normalized sim images by design — *do not* use ImageNet stats here).

### PPO / network config (`so101_rl/.../agents/skrl_ppo_cfg.yaml`)

PPO hyperparameters, training schedule, and network architecture. The `seed` here propagates to all RNGs; override at the command line with `--seed N`.

The split between env and agent configs follows one principle: **anything that affects the observation space lives in the env config; everything about the network lives in the skrl config.**

### Pipeline config (`configs/pipeline.yaml`)

A single YAML that drives the full `train → collect → curate → train-cnn` sequence. Each step's output dir is automatically wired as the next step's input.

### Sweep config (`configs/sweep.yaml`, `configs/sweeps/*.yaml`)

Defines a Cartesian-product grid of experiments. Each `config_set` is one dimension; entries within a `config_set` are merged onto the base env / agent configs. The `sweep` command executes every combination as a `train` + `eval` pair, then writes a comparison `summary.md` / `summary.json`.

## Usage

All commands are accessed through `./scripts/run.py`. Use `--help` on any subcommand for full options.

```bash
# ── RL training ────────────────────────────────────────────────────────────────

./scripts/run.py train \
    --task So101-LiftCube-v0 \
    --config configs/policy_train.yaml \
    --envs 256 \
    --cameras --headless

# Train with a pretrained CNN backbone (env config must set vision_encoder.type: frozen_cnn)
./scripts/run.py train \
    --task So101-LiftCube-v0 \
    --config configs/policy_train.yaml \
    --cnn-checkpoint artifacts/2026-04-22_10-36-02/cnn_checkpoint.pt \
    --cameras --headless

# Resume from a checkpoint
./scripts/run.py train \
    --task So101-LiftCube-v0 \
    --config configs/policy_train.yaml \
    --checkpoint artifacts/2026-04-22_14-14-05/skrl/so101_lift_cube/checkpoints/best_agent.pt \
    --cameras --headless

# ── Evaluation ─────────────────────────────────────────────────────────────────

./scripts/run.py eval \
    --experiment artifacts/2026-04-22_14-14-05 \
    --episodes 100 \
    --videos 5 \
    --envs 64 \
    --headless

# ── Telemetry collection (for CNN pretraining) ────────────────────────────────

./scripts/run.py collect \
    --task So101-LiftCube-v0 \
    --experiment artifacts/2026-04-22_14-14-05 \
    --sample-interval 8 \
    --episodes 10000 \
    --seed 42 \
    --output artifacts/ \
    --cameras --headless

# ── Dataset curation ───────────────────────────────────────────────────────────

./scripts/run.py curate \
    --input artifacts/<collect_run> \
    --output artifacts/ \
    --config configs/cnn_pretrain.yaml \
    --seed 42

# ── CNN backbone training ──────────────────────────────────────────────────────

./scripts/run.py train-cnn \
    --input artifacts/<curate_run> \
    --output artifacts/ \
    --config configs/cnn_pretrain.yaml \
    --device cuda:0

# ── Full pipeline (train → collect → curate → train-cnn) ──────────────────────

./scripts/run.py pipeline --config configs/pipeline.yaml

# Resume an interrupted pipeline from the curate step
./scripts/run.py pipeline \
    --pipeline-dir artifacts/pipeline_2026-04-18_16-25-11 \
    --from curate

# Start mid-way using an existing trained policy
./scripts/run.py pipeline \
    --config configs/pipeline.yaml \
    --from collect \
    --experiment artifacts/2026-04-22_14-14-05

# Dry run (print resolved commands without executing)
./scripts/run.py pipeline --config configs/pipeline.yaml --dry-run

# ── Sweeps (Cartesian-product ablations) ──────────────────────────────────────

./scripts/run.py sweep --sweep configs/sweep.yaml
./scripts/run.py sweep --sweep configs/sweep.yaml --dry-run
./scripts/run.py sweep --resume sweeps/sweep_<name>_<timestamp>/

# ── Vision backbone visualizer ────────────────────────────────────────────────
# Renders per-channel feature maps, SpatialSoftmax keypoints, and activation
# heatmaps over a recorded eval video — useful for sanity-checking that
# vision features remain meaningful throughout training.
# (`--backbone-cfg` points at a small standalone YAML mirroring the backbone
#  shape; see `configs/frozen_cnn_backbone.yaml`.)

python -m vision_backbone_demo \
    --input  artifacts/<eval_run>/evaluation/wrist_cam_env_000_ep_000.mp4 \
    --output artifacts/<eval_run>/evaluation/wrist_cam_env_000_ep_000_vis.mp4 \
    --backbone frozen_cnn \
    --cnn-checkpoint artifacts/<train_run>/cnn_checkpoint.pt \
    --backbone-cfg configs/frozen_cnn_backbone.yaml \
    --device cuda --seed 42 --max-channels 16 --fps 20
```

All commands that write output accept `--output PATH` as a base directory. Outputs are always written to `<output>/<timestamp>/` (or `<output>/pipeline_<timestamp>/`, `<output>/sweep_<name>_<timestamp>/`), defaulting to `artifacts/`.

**Common flags (most subcommands):**
| Flag                    | Description                                             |
| ----------------------- | ------------------------------------------------------- |
| `--output PATH`         | Base output dir; timestamped subdir created inside      |
| `--seed N`              | RNG seed for reproducibility                            |
| `--headless`            | Run Isaac Sim without a GUI window                      |
| `--cameras`             | Enable Isaac cameras (required for vision tasks)        |
| `--envs N`              | Override number of parallel environments                |
| `--checkpoint PATH`     | Resume from a PPO checkpoint                            |
| `--cnn-checkpoint PATH` | Pretrained CNN backbone `.pt` (for `frozen_cnn`)        |
| `--display N`           | X11 display socket number (e.g. `2` for `DISPLAY=:2`)   |
| `--dry-run`             | (`pipeline` / `sweep`) Print commands without executing |

## Domain Randomization

Lighting, camera feed augmentation (noise, brightness, contrast, motion blur, JPEG compression), camera pose, cube color/size/position, arm starting joint positions, and configurable distractor objects.

## Sim-to-Real Status

This project currently runs **only in simulation** (NVIDIA Isaac Sim / Isaac Lab). Physical-hardware deployment on the SO-101 is the next planned milestone; the design choices throughout — wrist-only camera, joint-position control normalized to `[0, 1]`, frozen vision encoders, aggressive domain randomization, declarative reward terms — are all in service of that eventual transfer.

## Credits

- SO-101 URDF: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- This project traces its origins to a [class project](https://github.com/utd-fall-25-cs-6341-robotics/cs6341-robotics-project-direct) created by myself and Kiran Hegde.
