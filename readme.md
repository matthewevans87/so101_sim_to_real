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

## Usage

## System Requirements

- **NVIDIA Isaac Sim** (https://developer.nvidia.com/isaac-sim)
- **Isaac Lab** (https://isaac-sim.github.io/IsaacLab/)
- CUDA GPU with ≥16 GB VRAM

## Installation

```bash
pip install -e .   # installs so101_utils (shared image processing)
```

## Configuration

Configuration is made in two YAML files: **env config** and **SKRL config**:

**env config** — environment parameters (physics, rewards, domain randomization, sensors, etc.). `run.sh` passes this automatically via `SO101_ENV_CONFIG`; defaults to `configs/baseline.yaml`. Override with:
```bash
./scripts/run.sh train ... --env-config configs/my_config.yaml
```
The YAML is validated against a typed dataclass hierarchy (`so101_env_params.py`) at startup.
Current options are:
- `configs/baseline.yaml` - uses frozen weights of ResNet18 model for vision feature extraction
- `configs/trainable_cnn` - trains a CNN feature extractor in the PPO training loop.

**`so101_rl/.../agents/skrl_ppo_cfg.yaml`** — PPO hyperparameters, network architecture, and training schedule. Also holds the `seed`, which propagates to all RNGs (`torch`, `numpy`, `random`). Override the seed at the command line with `--seed N` (use `-1` for a random seed).
> Note: if `vision_encoder.type` is `trainable_cnn`, then the `models`, and `memory` sections of `skrl_ppo_cfg` are ignored, and `agent` and `trainer` have some of their properties overwritten. 

## Usage

```bash
# Train
./scripts/run.sh train 
    --task So101-LiftCube-v0 \
    --num-envs 10 \ # num of parallel envs to simulate
    --enable-cameras \ # required for vision features
    --headless # run a headless instance of Isaac Sim

# Evaluate
./scripts/run.sh evaluate \
    --experiment-path artifacts/2026-03-12_09-52-10 \
    --num-episodes 100 \ # the number of episodes to evaluate
    --num-videos 5 \ # the number of episodes to record video for
    --num-envs 10 \
    --headless
```

**Useful flags:**
| Flag                 | Description                              |
| -------------------- | ---------------------------------------- |
| `--num-envs N`       | Parallel environments                    |
| `--max-iterations N` | Training iterations                      |
| `--seed N`           | RNG seed (overrides YAML; `-1` = random) |
| `--checkpoint PATH`  | Resume from checkpoint                   |
| `--headless`         | No GUI                                   |
| `--enable-cameras`   | Required for vision tasks                |
| `--video`            | Record evaluation video                  |
| `--display N`        | X11 display (useful over SSH)            |

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
