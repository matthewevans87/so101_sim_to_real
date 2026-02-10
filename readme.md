# Vision-Based Manipulation via Sim-to-Real RL

Training vision-conditioned RL policies for the SO-101 robotic arm using NVIDIA Isaac Sim + Isaac Lab with zero-shot sim-to-real transfer.

## Overview

This project implements two key RL tasks:
- **Proprioception-Only** (`So101-JointPosGoUp-v0`): Maximize end-effector height using joint states (12-D obs)
- **Vision-Based** (`So101-LiftCube-v0`): Locate and interact with target cube using wrist-mounted RGB camera (256×192 + 6-D joints)

**Key Features:**
- On-policy PPO training with asymmetric actor-critic
- ResNet18 + Spatial Softmax vision encoding (frozen)
- Extensive domain randomization (lighting, camera augmentation, object placement, physics)
- Joint-space position control via LeRobot API

## Prerequisites

1. **NVIDIA Isaac Sim**: Install to `$ISAAC_SIM_PATH` (https://developer.nvidia.com/isaac-sim)
2. **Isaac Lab**: Install to `$ISAAC_LAB_PATH` (https://isaac-sim.github.io/IsaacLab/)
3. **LeRobot API**: Install from HuggingFace (https://huggingface.co/lerobot) for robot control interface
4. **Hardware**: CUDA-capable GPU with >16GB VRAM (tested on RTX 5090)
5. **Physical Robot** (for deployment): SO-101 arm with calibrated wrist camera

## Usage

The `scripts/run.sh` script handles the full pipeline:

```bash
# Train a policy
./scripts/run.sh train --task So101-LiftCube-v0 \
    --num-envs 32 --enable-cameras --headless

# Play/evaluate policy
./scripts/run.sh play --task So101-LiftCube-v0 \
    --checkpoint outputs/.../checkpoints/agent.pt \
    --enable-cameras --video --video-length 1000

# Export trained model
./scripts/run.sh export --task So101-LiftCube-v0 \
    --checkpoint outputs/.../checkpoints/agent.pt --enable-cameras

```

**Common Options:**
- `--task TASK`: Task name (default: `So101-JointVelGoUp-v0`)
- `--num-envs N`: Parallel environments (default: 1024)
- `--max-iterations N`: Training iterations
- `--checkpoint PATH`: Model checkpoint path
- `--headless`: Run without GUI
- `--enable-cameras`: Enable camera observations (required for vision tasks)
- `--video`: Record evaluation video
- `--video-length N`: Video length in frames (default: 1000)
- `--display N`: Set the X11 display; useful if executing training from SSH to remote machine with a connected display

## Architecture

**Vision Task:**
- **Actor**: RGB → ResNet18 → SpatialSoftmax (1024-D) + Joint Pos (6-D) → MLP[256,128,64] → Actions (6-D)
- **Critic**: Privileged observations (14-D: joints + cube state) → MLP[256,128,64] → Value
- **Training**: PPO with GAE, 2048 rollout steps, 16 epochs per update, KL-adaptive LR

**Reward Shaping:**
- Approach reward (exp decay with distance)
- Camera alignment (dot product of camera forward & cube direction)
- Lift reward (gated by contact, scaled by height)
- Touch reward (contact force threshold)
- Penalties for excessive velocity and action magnitude

## Sim-to-Real Transfer

1. **Camera Calibration**: Hand-eye calibration to match simulation camera pose
2. **Observation Preprocessing**: ImageNet normalization, joint angle scaling
3. **Deployment**: Export policy to Torch, load via LeRobot API
4. **Zero-Shot**: No fine-tuning on physical robot

## Physical Robot Deployment

After training and exporting policies, deploy them on the physical SO-101 arm. Pre-trained policies are available in the `trained_policies/` directory.

**Joint Position Policy (proprioception-only):**
```bash
python so101_controller/run_joint_policy_controller.py \
    --checkpoint trained_policies/so101_joint_pos_go_up_policy.pt \
    --robot-port /dev/ttyACM0 \
    --urdf-path so101_controller/assets/SO101/so101_new_calib.urdf
```

**Vision Policy (camera-based):**
```bash
# List available cameras
python so101_controller/run_vision_policy_controller.py --list-cameras

# Run vision policy with wrist camera
python so101_controller/run_vision_policy_controller.py \
    --checkpoint trained_policies/so101_lift_cube.pt \
    --camera 0 \
    --policy-type conv \
    --robot-port /dev/ttyACM0 \
    --urdf-path so101_controller/assets/SO101/so101_new_calib.urdf
```

**Policy Types:**
- `fc`: Uses 512-D features from ResNet18 avgpool layer
- `conv`: Uses 1024-D spatial features from Spatial Softmax (matches training)

## Results

See `report/final-report/final-report.pdf` for detailed experimental results, including training curves, ablation studies, and sim-to-real transfer analysis.

## Credits

- Framework: NVIDIA Isaac Sim + Isaac Lab
- RL Library: SKRL
- SO-101 URDF: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
