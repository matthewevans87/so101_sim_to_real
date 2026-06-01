# Vision-Based Sim-to-Real Manipulation on the SO-101

Learning wrist-camera-only RL policies in simulation, with a full pipeline for deploying to a physical 6-DOF arm. Zero-shot transfer is a work in progress.

![Sim policy approaching and grasping the cube](assets/demo.gif)
<!-- TODO: replace with actual gif once available -->

## Key Findings

Full details in the **[report](./report/report.md)**.

- **~88% episode success in simulation** with a wrist-camera-only PPO policy under heavy domain randomization (lighting, camera pose, cube appearance, distractors, image augmentation).
- **Task-specific CNN beats frozen ResNet18 by +4.0 pp** (87.1% vs. 83.1%): a small CNN pretrained from synthetic telemetry outperforms an off-the-shelf ImageNet backbone on this in-domain task. A second distillation generation adds a further +0.6 pp (87.7%).
- **A minimal 2-term reward matches a 13-term hand-tuned reward** within statistical noise (86.9% vs. 87.7%). Several individual terms were *actively counterproductive* — removing the dense absolute approach reward was the single best configuration tested at **89.7%**.
- **Fully reproducible pipeline** (`train → collect → curate → train-cnn`) with typed-YAML configs (no silent defaults), versioned per-run artifacts, and end-to-end provenance tracking.
- **End-to-end deployment pipeline**: OpenCV camera calibration, a self-contained export bundle, and on-hardware inference with no Isaac Lab dependency. The policy runs on the physical arm; zero-shot transfer quality is still being improved.

## System Requirements

- **NVIDIA Isaac Sim** (https://developer.nvidia.com/isaac-sim) and **Isaac Lab** (https://isaac-sim.github.io/IsaacLab/) — required for training and simulation
- CUDA GPU with ≥16 GB VRAM (24 GB recommended for sweeps)
- `deploy` has **no Isaac Lab dependency** and runs under any Python with `so101_real` installed

## Installation

```bash
pip install -e .             # so101 library: image processing, CNN training, visualizers
pip install -e ".[deploy]"   # adds real-robot deploy dependencies (no Isaac Lab needed)
```

Set `ISAAC_LAB_PATH` before running Isaac-dependent commands (`train`, `collect`, `eval`, `play`, `export`, `pipeline`, `sweep`).

## Repository Layout

```
configs/                YAML configs (env, CNN pretrain, pipeline, sweeps)
scripts/                Top-level CLI entry points (run.py, sweep.py, pipeline.py)
so101/                  Shared library (CNN training, curation, image utils, viz)
so101_rl/               Isaac Lab task package (So101-LiftCube-v0)
so101_real/             Real-robot deployment package (camera, controller, inference)
vision_backbone_demo/   Standalone visualizer for backbone activations / keypoints
artifacts/              Default output dir (timestamped per-run subdirs)
notes/experiment_journal/  Running log of experiments and findings
```

## Configuration

| Config file                             | Controls                                                                                                                                                                                                                                       |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/policy_train.yaml`             | Everything that affects the simulation: physics, joints, camera, image pipeline, distractors, domain randomization, declarative reward terms, termination conditions, `vision_encoder.type`. Single source of truth for the observation space. |
| `configs/cnn_pretrain.yaml`             | Supervised CNN pipeline: backbone architecture, curation settings, head shapes, image normalization. Backbone definition is embedded in `cnn_checkpoint.pt`.                                                                                   |
| `so101_rl/.../agents/skrl_ppo_cfg.yaml` | PPO hyperparameters, network architecture, training schedule, RNG seed. Anything about the network lives here; anything about the observation space lives in the env config.                                                                   |
| `configs/pipeline.yaml`                 | Drives the full `train → collect → curate → train-cnn` sequence; output of each step is auto-wired as input to the next.                                                                                                                       |
| `configs/sweep.yaml`                    | Cartesian-product ablation grid; each `config_set` is one dimension.                                                                                                                                                                           |
| `so101_real/configs/robot.yaml`         | Real-robot deploy: serial port, calibration file, camera device index and resolution.                                                                                                                                                          |

All configs are validated against typed dataclasses at startup. Missing or unknown keys raise immediately, before any expensive compute begins.

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

```bash
# ── Real-robot deployment (no Isaac Lab required) ────────────────────────────

# Run the latest exported bundle (uses the latest_bundle pin set by export):
./scripts/run.py deploy \
    --robot-config so101_real/configs/robot.yaml \
    --episodes 5 \
    --seed 42

# Point to a specific export bundle directory:
./scripts/run.py deploy \
    --bundle /mnt/nas_1/matthew-evans/so101_sim_to_real/exports \
    --robot-config so101_real/configs/robot.yaml \
    --episodes 5 \
    --seed 42

# Validate bundle + robot config without moving the robot:
./scripts/run.py deploy \
    --bundle /mnt/nas_1/matthew-evans/so101_sim_to_real/exports \
    --robot-config so101_real/configs/robot.yaml \
    --episodes 1 --dry-run

# With live OpenCV overlay and episode recording:
./scripts/run.py deploy \
    --bundle /mnt/nas_1/matthew-evans/so101_sim_to_real/exports \
    --robot-config so101_real/configs/robot.yaml \
    --episodes 5 --overlay --record
```

**`deploy` flags:**
| Flag                  | Description                                                                                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `--bundle PATH`       | Deploy bundle directory (contains `policy.pt`, `cnn_backbone.pt`, `manifest.json`, etc.). Defaults to the `latest_bundle` pin set by the most recent `export` run. |
| `--robot-config PATH` | Robot config YAML. Copy and edit `so101_real/configs/robot.yaml` (set serial port, calibration file, camera index).                                                |
| `--episodes N`        | Number of episodes to execute.                                                                                                                                     |
| `--seed N`            | RNG seed.                                                                                                                                                          |
| `--overlay`           | Show a live OpenCV window with camera feed, joint positions, and actions.                                                                                          |
| `--record`            | Save episode rollouts (frames + joint data) to `<bundle>/rollouts/`.                                                                                               |
| `--dry-run`           | Validate bundle and robot config, print summary, then exit without moving the robot.                                                                               |

**Before deploying**, edit `so101_real/configs/robot.yaml`:
- `robot.port` — serial port of the SO-101 follower arm (e.g. `/dev/ttyACM0`)
- `robot.calibration_file` — path to the LeRobot calibration JSON
- `camera.device_index` — V4L2 index of the wrist camera
- `camera.capture_width` / `capture_height` — physical resolution the camera supports

`deploy` has **no Isaac Lab dependency** and runs directly under the system or conda Python that has `so101_real` installed (`pip install -e ".[deploy]"`). Isaac Sim is not required.

`./scripts/run.py deploy` is a convenience wrapper around `python -m so101_real run` (it mainly adds bundle pin resolution and environment setup).
For full real-hardware runtime details and direct `so101_real` commands, see [so101_real/README.md](./so101_real/README.md).

```bash
# ── Digital twin (mirror real arm in Isaac Sim, no policy) ───────────────────
# Two processes, one per terminal. The `stream` side reads joint positions
# from the physical SO-101 and publishes them on a ROS2 JointState topic
# (default `/so101/joint_states`); `digital-twin` launches an Isaac Sim
# viewer that subscribes to the topic and mirrors the articulation live.
# Useful for visually verifying calibration, joint sign conventions, and
# camera mount alignment without running a policy.

# Terminal 1 — Isaac Sim viewer (needs ISAAC_LAB_PATH; uses ROS2 jazzy bridge)
./scripts/run.py digital-twin --display 2

# Terminal 2 — publish real joint states (lerobot env, torque off so you can
# pose the arm by hand)
conda activate lerobot
./scripts/run.py stream \
    --robot-config so101_real/configs/robot.yaml \
    --no-torque

# ── Camera extrinsics tuning (interactive overlay) ───────────────────────────
# Spawns Isaac Lab with the SO-101, mirrors live joint positions from the
# physical robot, and renders an ffplay overlay of the sim wrist camera
# against a previously captured real frame. Drag the CameraXframe prim in
# the viewport until the overlay converges, then press `s` to print the
# calibrated translation/quaternion. See the "Camera calibration" section
# above for the full workflow.

$ISAAC_LAB_PATH/isaaclab.sh -p so101_rl/scripts/tune_camera_pose.py \
    --robot-config so101_real/configs/robot.yaml
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

## Credits

- SO-101 hardware: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- This project traces its origins to a [class project](https://github.com/utd-fall-25-cs-6341-robotics/cs6341-robotics-project-direct) created with Kiran Hegde.
