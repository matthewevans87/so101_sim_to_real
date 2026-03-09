# Vision-Based Manipulation via Sim-to-Real RL

Training a vision-conditioned RL policy for the SO-101 robotic arm using NVIDIA Isaac Sim + Isaac Lab, targeting zero-shot sim-to-real transfer.

## Task

**`So101-LiftCube-v0`**: Locate and lift a cube using a wrist-mounted RGB camera.
- **Actor obs**: SpatialSoftmax features (1024-D) + joint positions (6-D)
- **Critic obs**: Privileged state (joints, cube pose, contact forces)
- **Actions**: 6-D joint position deltas

## Prerequisites

- **NVIDIA Isaac Sim** (https://developer.nvidia.com/isaac-sim)
- **Isaac Lab** (https://isaac-sim.github.io/IsaacLab/)
- CUDA GPU with ≥16 GB VRAM

## Installation

```bash
pip install -e .   # installs so101_utils (shared image processing)
```

## Configuration

There are two YAML configuration files:

**`configs/baseline.yaml`** — environment parameters (physics, rewards, domain randomisation, sensors, etc.). `run.sh` passes this automatically via `SO101_ENV_CONFIG`; defaults to `configs/baseline.yaml`. Override with:
```bash
./scripts/run.sh train ... --env-config configs/my_config.yaml
```
The YAML is validated against a typed dataclass hierarchy (`so101_env_params.py`) at startup.

**`so101_rl/.../agents/skrl_ppo_cfg.yaml`** — PPO hyperparameters, network architecture, and training schedule. Also holds the `seed`, which propagates to all RNGs (`torch`, `numpy`, `random`). Override the seed at the command line with `--seed N` (use `-1` for a random seed).

## Usage

```bash
# Train
./scripts/run.sh train --task So101-LiftCube-v0 \
    --num-envs 64 --enable-cameras --headless

# Evaluate
./scripts/run.sh play --task So101-LiftCube-v0 \
    --checkpoint logs/skrl/.../checkpoints/agent.pt \
    --enable-cameras --video

# Export
./scripts/run.sh export --task So101-LiftCube-v0 \
    --checkpoint logs/skrl/.../checkpoints/agent.pt --enable-cameras
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

## Architecture

- **Vision encoder**: ResNet18 → SpatialSoftmax → 1024-D (frozen)
- **Actor MLP**: `[256, 128, 64]` with ELU activations
- **Critic MLP**: `[256, 128, 64]` with ELU activations (privileged obs)
- **RL algorithm**: PPO with KL-adaptive LR (via skrl)

## Domain Randomisation

Lighting, camera feed augmentation (noise, brightness, contrast, motion blur, JPEG compression), camera pose, cube colour/size/position, distractor objects.

## Credits

- Simulator: [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) + [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- RL library: [skrl](https://skrl.readthedocs.io/)
- SO-101 URDF: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)

