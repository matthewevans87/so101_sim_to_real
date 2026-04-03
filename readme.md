# Vision-Based Manipulation via Sim-to-Real RL

## Overview

Inspired by the likes of [skild.ai](https://www.youtube.com/watch?v=JQAfxp-FB0I), [NVIDIA](https://www.youtube.com/watch?v=S4tvirlG8sQ), and [Tesla](https://www.youtube.com/watch?v=g6bOwQdCJrc), this project is an introductory investigation into what can be achieved by learning RL policies on consumer hardware for zero-shot transfer to physical robotic devices.

The project aims for experimental rigor and reproducibility. All configurations (rewards, seeds, network, etc.) are made via YAML configs and saved along with the results of each experiment. An evaluation script enables comparing the results of different experiments and supports a data driven approach to tuning and improvement. 

## Problem
**Task Definition**. The agent, an [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robotic arm with a single [wrist-mounted](https://github.com/TheRobotStudio/SO-ARM100/blob/main/media/UVC_cam_mount_so101.jpg) [camera](https://www.amazon.com/dp/B07ZRJDTBQ), is tasked with finding a small [cube](https://developer.nvidia.com/blog/reinforcing-the-value-of-simulation-by-teaching-dexterity-to-a-real-robot-hand) on its work surface and lifting it to a height of 10 cm within 10 seconds. 

**Observations**. The agent vision features (1024D) and joint positions (6D) at each time step, normalized to `[0.0, 1.0]`. All other information (e.g., explicit cube position, etc.) is hidden.

**Actions**. The agent can issue joint position commands, a 6D vector normalized to `[0.0, 1.0]`.

**Episodes**. Each episode is set to 10 seconds (see `episode_length_s` config). Physics steps are calculated 120 times per second, and observations are taken every 2 ticks (see `decimation` config) for a total of `10*120/2` steps.

## Environment


## RL Setup

## Baseline

## System Requirements

- **NVIDIA Isaac Sim** (https://developer.nvidia.com/isaac-sim)
- **Isaac Lab** (https://isaac-sim.github.io/IsaacLab/)
- CUDA GPU with ≥16 GB VRAM

## Installation

```bash
pip install -e .   # installs the so101 library (image processing, CNN training, visualizers)
```

Set the `ISAAC_LAB_PATH` environment variable to your Isaac Lab installation before running Isaac-dependent commands (`train`, `collect`, `eval`, `play`, `export`, `pipeline`).

## Configuration

### RL environment config (`configs/<experiment>.yaml`)

Controls simulation and environment parameters: physics, rewards, domain randomization, sensors, and `vision_encoder.type`. Validated at startup against a typed dataclass hierarchy (`so101_env_params.py`); unknown or missing keys raise immediately.

Current configs:
- `configs/baseline.yaml` — frozen ResNet18 + SpatialSoftmax (1024-D) vision features
- `configs/pretrained_cnn.yaml` — pretrained lightweight CNN backbone (weights loaded at train time via `--backbone`)
- `configs/trainable_cnn.yaml` — lightweight CNN trained end-to-end within the PPO loop

### PPO / network config (`so101_rl/.../agents/skrl_ppo_cfg.yaml`)

PPO hyperparameters, training schedule, and network architecture. The `seed` here propagates to all RNGs (`torch`, `numpy`, `random`); override at the command line with `--seed N`.

Network architecture is split by path within `models:`:
- `models.policy.network` / `models.value.network` — MLP layers used by the `resnet18` path
- `models.policy.cnn`, `models.policy.head_dims`, `models.value.hidden_dims` — CNN backbone and head used by the `trainable_cnn` / `pretrained_cnn` paths

The key principle: `vision_encoder.type` and image dimensions live in the env config (they affect observation space); everything about the network architecture lives in the skrl config.

### CNN pretrain config (`configs/cnn_pretrain.yaml`)

Controls the curation stage (histogram rebalancing, train/val/test splits) and the CNN supervised training stage (backbone architecture, loss weights, training schedule). The backbone architecture defined here must match `models.policy.cnn` in the skrl YAML so that pretrained weights load cleanly into the RL actor.

### Pipeline config (`configs/pipeline.yaml`)

A single YAML that drives the full `train → collect → curate → train-cnn` pipeline. All required fields are validated at startup — no silent defaults.

## Usage

All commands are accessed through `./scripts/run.py`. Use `--help` on any subcommand for full options.

```bash
# ── RL training ────────────────────────────────────────────────────────────────

./scripts/run.py train \
    --task So101-LiftCube-v0 \
    --config configs/baseline.yaml \
    --envs 16 \
    --cameras \
    --headless

# Resume from a checkpoint
./scripts/run.py train \
    --task So101-LiftCube-v0 \
    --config configs/pretrained_cnn.yaml \
    --backbone artifacts/2026-03-12_09-52-10/cnn/best_backbone.pt \
    --checkpoint artifacts/2026-03-12_09-52-10/skrl/so101_lift_cube/checkpoints/best_agent.pt \
    --cameras --headless

# ── Evaluation ─────────────────────────────────────────────────────────────────

./scripts/run.py eval \
    --experiment artifacts/2026-03-12_09-52-10 \
    --episodes 100 \
    --videos 5 \
    --envs 10 \
    --headless

# ── Telemetry collection ───────────────────────────────────────────────────────

./scripts/run.py collect \
    --task So101-LiftCube-v0 \
    --experiment artifacts/2026-03-12_09-52-10 \
    --sample-interval 8 \
    --episodes 1000 \
    --seed 42 \
    --output artifacts/ \
    --cameras --headless

# ── Dataset curation ───────────────────────────────────────────────────────────

./scripts/run.py curate \
    --input artifacts/2026-03-26_10-00-00 \
    --output artifacts/ \
    --config configs/cnn_pretrain.yaml \
    --seed 42

# ── CNN backbone training ──────────────────────────────────────────────────────

./scripts/run.py train-cnn \
    --input artifacts/2026-03-26_10-00-00 \
    --output artifacts/ \
    --config configs/cnn_pretrain.yaml \
    --device cuda:0

# ── Full pipeline (train → collect → curate → train-cnn) ──────────────────────

./scripts/run.py pipeline --config configs/pipeline.yaml

# Resume an interrupted pipeline from the curate step
./scripts/run.py pipeline \
    --pipeline-dir artifacts/pipeline_2026-03-26_10-00-00 \
    --from curate

# Start pipeline mid-way with an existing experiment
./scripts/run.py pipeline \
    --config configs/pipeline.yaml \
    --from collect \
    --experiment artifacts/2026-03-12_09-52-10

# Dry run (print resolved commands without executing)
./scripts/run.py pipeline --config configs/pipeline.yaml --dry-run
```

All commands that write output accept `--output PATH` as a base directory. Outputs are always written to `<output>/<timestamp>/` (or `<output>/pipeline_<timestamp>/` for `pipeline`), defaulting to `artifacts/`.

**Common flags (most subcommands):**
| Flag                    | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `--output PATH`         | Base output dir; timestamped subdir created inside     |
| `--seed N`              | RNG seed for reproducibility                           |
| `--headless`            | Run Isaac Sim without a GUI window                     |
| `--cameras`             | Enable Isaac cameras (required for vision tasks)       |
| `--envs N`              | Override number of parallel environments               |
| `--checkpoint PATH`     | Resume from a checkpoint                               |
| `--display N`           | X11 display socket number (e.g. `2` for `DISPLAY=:2`) |
| `--dry-run`             | (`pipeline` only) Print commands without executing     |

**`train` flags:**
| Flag             | Description                          |
| ---------------- | ------------------------------------ |
| `--config PATH`  | Env config YAML (required)           |
| `--iters N`      | Override max training iterations     |
| `--backbone PATH`| Pretrained CNN backbone `.pt` file   |

**`collect` flags:**
| Flag                  | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `--experiment PATH`   | RL experiment directory (required)                      |
| `--sample-interval N` | Collect one sample every N env steps (required)         |
| `--episodes N`        | Stop after N complete episodes (required)               |
| `--shard-size N`      | Samples per NPZ shard                                   |

**`eval` / `play` flags:**
| Flag          | Description                                             |
| ------------- | ------------------------------------------------------- |
| `--episodes N`| Number of evaluation episodes                           |
| `--videos N`  | Number of episodes to record video for                  |

## Domain Randomization

Lighting, camera feed augmentation (noise, brightness, contrast, motion blur, JPEG compression), camera pose, cube color/size/position, distractor objects.

## Credits

- SO-101 URDF: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- This project traces its origins to a [class project](https://github.com/utd-fall-25-cs-6341-robotics/cs6341-robotics-project-direct) created by myself and Kiran Hegde.

<!-- 
### Related Papers
| Title                                                                               | Authors                                                             | Date |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ---- |
| [End-to-End Training of Deep Visuomotor Policies](https://arxiv.org/pdf/1504.00702) | Sergey Levine and Chelsea Finn and Trevor Darrell and Pieter Abbeel | 2016 | --> |
