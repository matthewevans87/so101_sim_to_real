# so101_real — Real-Robot Inference Package

Runs a trained SO-101 policy on physical hardware. No Isaac Lab dependency required.

## Frames

`so101_real` works in **one** internal frame: the canonical URDF frame where
every joint reads `0 rad` at the home pose, matching what trained policies and
the digital twin expect. Everything inside this package — joint bounds in
`robot.yaml::joint_limits`, observation/action tensors, recorded telemetry,
`run-static --joints "0,0,0,0,0,0"` — is in canonical radians.

The only other frame is LeRobot's native one (raw motor positions from the
follower bus), which is bridged by `robot.yaml::joint_calibration` using

    q_canonical_rad = scale * q_lerobot_rad + offset_rad

This conversion happens inside `So101Robot.read_joints` / `send_joints`; no
caller of `so101_real` ever sees raw LeRobot values.

## Quick start

```bash
conda activate lerobot
./scripts/run.py deploy \
  --robot-config so101_real/configs/robot.yaml \
  --episodes 10
```

Add `--overlay` to show a live OpenCV window on the workstation display, and `--record` to save per-episode MP4 video and NPZ telemetry.

### Live digital-twin overlay (Isaac Sim mirror)

To watch the Isaac Sim arm mirror the physical robot in real time while a policy runs, use two terminals:

```bash
# Terminal 1 — Isaac Sim viewer (subscriber)
./scripts/run.py digital-twin --display 2

# Terminal 2 — real-robot policy + ROS2 joint-state publisher
./scripts/run.py deploy \
  --robot-config so101_real/configs/robot.yaml \
  --episodes 10 \
  --overlay \
  --ros
```

`--ros` makes the deploy process publish measured joint states to `/so101/joint_states` at the control rate; the digital-twin viewer subscribes to that topic and writes positions into the SO-101 articulation. Useful for visually confirming the physical arm and the simulated arm land in the same pose (e.g. reset pose, end-of-episode pose) when validating real-robot inference against the training environment.

The same `--ros` plumbing is also exposed standalone as `./scripts/run.py stream` (publishes joint states without running a policy — convenient for hand-driving the arm with torque off and watching the twin follow).

### Display (X11) when running inside tmux, VS Code terminal, or SSH

Commands that open an OpenCV window (e.g. `calibrate-camera`, `--overlay`) require `DISPLAY` to be set. tmux, VS Code integrated terminals, and plain SSH sessions do not inherit or forward a display automatically.

#### Option A — Use the workstation's local display (simplest)

If the workstation has a physical monitor (or an active desktop session), find the display number:

```bash
who   # look for a line like:  matthew-evans  :2  ...
```

Export it for the session:

```bash
export DISPLAY=:2
```

To make this permanent for tmux sessions, add to `~/.bashrc`:

```bash
# Auto-set DISPLAY for tmux sessions (adjust :2 to match your workstation)
if [ -z "$DISPLAY" ] && [ -n "$TMUX" ]; then
    export DISPLAY=:2
fi
```

#### Option B — SSH X11 forwarding (window appears on your laptop)

Reconnect with `-Y` (trusted X11 forwarding):

```bash
ssh -Y matthew-evans@<workstation>
# DISPLAY is set automatically, e.g. localhost:10.0
```

`-Y` is preferred over `-X` for OpenCV/Qt applications, which often trip X11 security extension restrictions with `-X`.

---

## Control loop architecture

`InferenceLoop` runs a tight 60 Hz control loop. The naïve design — grab frame → encode → infer → send — is bounded by the camera frame period (~45 ms at 22 fps for MJPEG 1920×1080), which would cap the control rate at ~22 Hz.

Instead, `VisionJointObsBuilder` is automatically wrapped by `AsyncVisionJointObsBuilder` on construction: a daemon thread (`obs_async`) runs the camera + encoder pipeline continuously in the background while `_tick()` assembles each observation by combining the latest cached vision features with the freshly-read joint positions. The per-tick cost of `build()` is a single lock acquire + tensor cat (< 1 ms).

```
Background thread (obs_async):   get_frame → pipeline → encoder → cache features   ~20 Hz
Control loop (_tick, 60 Hz):      read_joints → build(q_meas) → policy → EMA → safety → send
                                                 ↑ tensor cat only
```

**Benefits:**
- Control rate is 60 Hz regardless of camera or encoder speed.
- Vision features refresh at the camera's natural rate (~20 Hz); the policy always sees the most recent available frame, never a frame held stale by serial I/O or encoder latency.
- `last_frame_rgb` (used by the recorder and overlay) is kept in sync by the background thread, so recorded video matches the features the policy acted on.

**Camera configuration matters:** at 1920×1080 the camera must be opened in MJPEG format (`fourcc: MJPG` in `robot.yaml`) or the V4L2 driver silently falls back to YUYV at 10 fps. The encoder runs on the device specified by `controller.device` (`cuda` is strongly recommended — 0.8 ms on an RTX-class GPU vs ~200 ms on CPU).

---

## CLI subcommand reference

Run `python -m so101_real <cmd> --help` for the full flag list of any command.
The `./scripts/run.py` shortcuts shown above (`deploy`, `stream`,
`digital-twin`) are thin wrappers over the same entry point.

| Subcommand         | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run`              | Run a trained policy from a deploy bundle on the physical robot. Supports `--record` (NPZ + per-episode MP4), `--overlay` (live OpenCV window), `--ros` (publish joint states for the digital twin), and `--dry-run` (validate bundle + config without opening hardware).                                                                                                                                                                     |
| `run-static`       | Hold the arm at a fixed joint target with no bundle / camera / encoder. Target is given by `--joints "a,b,c,d,e,f"` in the order defined by `robot.yaml::joint_limits`; `--unit {rad,deg,norm}` selects units (`norm` is `[-1, +1]` over each joint's physical range). Each episode runs `reset_pose` → hold for `--duration-s`; after the timed episodes the loop continues holding the target until Ctrl-C unless `--no-hold` is passed.    |
| `camera-test`      | Open the camera defined in `robot.yaml::camera` and show the live feed (sanity check the device index, resolution, exposure).                                                                                                                                                                                                                                                                                                                 |
| `configure-camera` | Write V4L2 controls (exposure, anti-flicker, brightness, etc.) to the camera from `camera_v4l2.yaml` using `v4l2-ctl`. Reads back and prints the applied values unless `--no-verify` is set. Settings reset on unplug — see the persistent udev rule in the Camera setup section.                                                                                                                                                             |
| `calibrate-camera` | Two-step camera-intrinsics workflow. **Step 1 (capture):** opens the camera and saves checkerboard frames to `--out-dir`. **Step 2 (solve):** pass `--solve` to run cv2 calibration over the captured frames and write `camera_intrinsics.yaml` (default location: `so101_real/configs/`). Board geometry is set with `--board-cols/--board-rows/--square-mm`.                                                                                |
| `compare-views`    | Composite a real camera frame (`--real`) and a sim render (`--sim`) into a single image with side-by-side, alpha blend, checkerboard interleave, and absolute-difference panels. Useful for visually verifying sim camera placement against `live_frame.png`.                                                                                                                                                                                 |
| `stream`           | Connect to the robot, read joint positions at `--hz` (default 30), and publish to `/so101/joint_states` (ROS2). Use with `--no-torque` to hand-move the arm and watch the digital-twin viewer mirror your motion. Does not run any policy.                                                                                                                                                                                                    |
| `robot-test`       | Print live joint positions in canonical `rad` / `deg` / `norm` (via `--unit`), using joint names from `robot.yaml::joint_limits`. Pair with `--no-torque` to verify calibration by hand-moving each joint and watching the readouts.                                                                                                                                                                                                          |
| `probe`            | Single-shot diagnostic for the canonical→LeRobot pipeline. Specify `--joint <name> --value <x> --unit {rad,deg,lrad,ldeg,norm}` and the command will linearly ramp from the current pose to the target over `--ramp-s` seconds at `--ramp-hz`, dumping every intermediate (calibration scale/offset, free-spinning branch projection, encoder wrap) before, during, and after the motion. Omit `--joint` to just dump current state and exit. |

Joint calibration is a separate module (own help text and modes):

```bash
python -m so101_real.joint_calibrate --help
```

See [Joint calibration (sim ↔ LeRobot)](#joint-calibration-sim--lerobot) below for the four modes (`single`, `sweep`, `discontinuity`, `wrap-sweep`).

---

## Camera setup

### Anti-flicker configuration (required for stable image)

Under fluorescent or LED lighting the camera will exhibit a **rolling horizontal scanline** if exposure is left on auto. This is a flicker beat artifact caused by the mains lighting frequency interacting with an unsynchronised shutter. Full-wave-rectified LEDs/fluorescents flicker at twice the mains frequency, so exposure must be locked to a multiple of that half-period.

Fix: disable dynamic framerate, set `power_line_frequency` to match your mains supply, and lock exposure to a safe multiple of the flicker period.

The settings live in [`so101_real/configs/camera_v4l2.yaml`](configs/camera_v4l2.yaml) and are applied with:

```bash
# Run once after plugging in the camera (resets on reconnect — see below)
python -m so101_real configure-camera \
  --camera-config so101_real/configs/camera_v4l2.yaml
```

Or using raw `v4l2-ctl` directly:

```bash
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1            # 1 = Manual
v4l2-ctl -d /dev/video0 --set-ctrl=power_line_frequency=2     # 2 = 60 Hz (US)
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=83  # 83 × 100 µs ≈ 8.3 ms
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=-50             # range: -64..64
```

If the scene is very dark, increase in safe multiples of ≈ 8.33 ms: `167` (≈ 16.7 ms), `250` (≈ 25 ms), etc.
If the scene is bright / blowing out, reduce gain rather than exposure — `83` is the minimum safe anti-flicker exposure for 60 Hz.

On a **50 Hz mains supply (UK/EU)**, the flicker frequency is 100 Hz (10 ms period). Use `power_line_frequency=1` and safe values `100` (10 ms), `200` (20 ms), `300` (30 ms), etc.

To verify the current settings:

```bash
v4l2-ctl -d /dev/video0 \
  --get-ctrl=auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,brightness,power_line_frequency
```

Expected output:

```
auto_exposure: 1
exposure_time_absolute: 83
exposure_dynamic_framerate: 0
brightness: -50
power_line_frequency: 2   # 2 = 60 Hz
```

### Capturing a single frame

```bash
ffmpeg -f v4l2 -video_size 1920x1080 -i /dev/video0 -frames:v 1 -update 1 \
  so101_real/calibration/captures/live_frame.png -y
```

### Tuning the sim camera position

If the sim wrist-camera view doesn't match the real camera (e.g. different number of visible gripper features), use the interactive tuning tool to find the correct `CameraXframe` mount transform.

**Step 1 — Capture a reference frame** at a convenient arm pose (or use an existing `live_frame.png`):

```bash
ffmpeg -f v4l2 -video_size 1920x1080 -i /dev/video0 -frames:v 1 -update 1 \
  so101_real/calibration/captures/live_frame.png -y
```

**Step 2 — Launch the tuning tool** (requires `env_isaaclab` and the Isaac Sim display):

```bash
# With real robot connected (sim arm mirrors live joint positions):
export DISPLAY=:2
$ISAAC_LAB_PATH/isaaclab.sh -p so101_rl/scripts/tune_camera_pose.py \
    --robot-config so101_real/configs/robot.yaml

# Without robot (fixed zero pose):
export DISPLAY=:2
$ISAAC_LAB_PATH/isaaclab.sh -p so101_rl/scripts/tune_camera_pose.py --no-robot
```

An OpenCV window opens alongside the Isaac Sim viewport showing the sim render blended with your real frame.

**Step 3 — Drag the `CameraXframe` prim** in the Isaac Sim viewport. Select the prim at:

```
/World/envs/env_0/Robot/gripper/mountscrew/camera_mount/CameraXframe
```

Use the Translate (`W`) and Rotate (`E`) gizmos until the overlay converges (gripper features align).

**Hotkeys** (in the OpenCV window):

| Key       | Action                                                                |
| --------- | --------------------------------------------------------------------- |
| `[` / `]` | Decrease / increase blend alpha                                       |
| `c`       | Cycle view: blend → side-by-side → checkerboard → abs-diff            |
| `r`       | Reload real image from disk (re-capture with `ffmpeg` then press `r`) |
| `s`       | Snapshot — print current transform as Python literals + YAML          |
| `q`       | Quit (also prints final transform)                                    |

**Step 4 — Apply the transform.** Copy the printed Python literals into `so101_rl/source/so101_rl/so101_rl/configurations/camera.py`:

```python
CAMERA_TRANSLATE_VEC = (x, y, z)          # from snapshot output
CAMERA_ROTATION_QUAT_WXYZ = (w, x, y, z)  # from snapshot output
```

Optionally write to a YAML file automatically with `--out-yaml camera_pose.yaml`.

### Making the settings persistent

The v4l2 controls reset every time the camera is unplugged or the system reboots. Add a udev rule to re-apply them automatically:

```bash
# /etc/udev/rules.d/99-so101-camera.rules
ACTION=="add", SUBSYSTEM=="video4linux", KERNEL=="video0", \
  RUN+="/usr/bin/v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0 --set-ctrl=auto_exposure=1 --set-ctrl=power_line_frequency=2 --set-ctrl=exposure_time_absolute=83 --set-ctrl=brightness=-50"
```

Then reload udev:

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Robot config

Edit `so101_real/configs/robot.yaml` before first use:

| Field                    | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| `robot.port`             | Serial port of the SO-101 arm (e.g. `/dev/ttyACM0`)        |
| `robot.calibration_file` | Path to the LeRobot calibration JSON                       |
| `robot.max_delta_rad`    | Maximum joint displacement per control step (safety clamp) |
| `camera.device_index`    | V4L2 device index (0 = `/dev/video0`)                      |
| `controller.ema_alpha`   | EMA smoothing on joint targets (1.0 = no smoothing)        |
| `controller.device`      | PyTorch device for inference (`cpu` or `cuda:0`)           |

### Edge case: gravity-stuck joint at startup (`max_relative_target` disabled)

`So101Robot.connect` intentionally passes `max_relative_target=None` to
LeRobot's `SO101FollowerConfig`. Per-tick motion limiting is enforced upstream
instead — by `SafetyLayer` in closed-loop control and by `reset_pose`'s linear
interpolation during startup recovery.

**Why this matters.** LeRobot's built-in clamp rewrites every outgoing goal to
`present_pos + clip(goal - present_pos, ±max_relative_target)`. The clamp is
defined in terms of *measured* position, so when a joint is far from its goal
the commanded error is shrunk to `±max_relative_target`. A Feetech STS3215's
torque output is roughly proportional to (Goal − Present), so a tight clamp
*starves the servo of torque*. We observed this concretely on `shoulder_lift`:

- Power had been released after a prior session; the arm sagged past the
  soft lower limit to `present ≈ -102°` (lower limit `-100°`), resting on
  the mechanical stop with gravity loading the joint.
- `run-static --joints "0,0,0,0,0,0"` commanded `goal ≈ 0°`.
- LeRobot clamped each tick's goal to `present + 5°`. With only 5° of PID
  error the servo could not overcome gravity + stop friction, so `present`
  stayed pinned and the same clamp fired every tick. The log spammed:

  ```
  WARNING:root:Relative goal position magnitude had to be clamped to be safe.
  {'shoulder_lift': {'original goal_pos': -0.40, 'safe goal_pos': -97.39}}
  ```

The fix removes the LeRobot-side clamp so `reset_pose` can interpolate the
goal in canonical radians over `reset_pose.duration_s`; the motor sees the
full position error and applies enough torque to lift the joint off the stop.
Our `SafetyLayer` still bounds closed-loop policy deltas to `max_delta_rad`,
but it is bypassed by `reset_pose` for exactly this reason — startup recovery
from a sagged pose needs unconstrained PID error to break free.

If a similar symptom reappears (one joint refuses to track its goal, log
floods with "Relative goal position magnitude had to be clamped"), check
first whether the joint has sagged past its soft limit while torque was off,
and confirm `max_relative_target=None` in `So101Robot.connect`.

---

## Joint calibration (sim ↔ LeRobot)

Each joint has a linear map `q_sim = scale * q_lerobot + offset` recorded under `joint_calibration:` in `so101_real/configs/robot.yaml`. Free-spinning joints (currently only `wrist_roll`) also need `wrap_period_rad` and `lero_branch_center_rad` so the read/send transforms cross the encoder discontinuity correctly.

Use `python -m so101_real.joint_calibrate` with one of four modes:

| Mode            | Purpose                                                              |
| --------------- | -------------------------------------------------------------------- |
| `single`        | Two-point manual fit at a chosen joint pose (legacy)                 |
| `sweep`         | Drive joint stop-to-stop, fit `scale`/`offset` from min/max readings |
| `discontinuity` | Measure the encoder wrap period of a free-spinning joint             |
| `wrap-sweep`    | Stop-to-stop fit that unwraps across the discontinuity               |

### Calibrating a free-spinning joint (e.g. `wrist_roll`)

1. **Measure the wrap period.** Slowly rotate the joint through several full turns when prompted; the script flags samples where consecutive readings jump by more than `--jump-threshold-rad` (default π/2) and reports the median jump magnitude (should be ≈ 2π = 6.2832 rad).

   ```bash
   python -m so101_real.joint_calibrate \
     --robot-config so101_real/configs/robot.yaml \
     --mode discontinuity \
     --joints wrist_roll
   ```

   Output is saved to `so101_real/calibration/<joint>_discontinuity_<timestamp>.yaml`.

2. **Fit scale, offset, and branch center.** Drive the joint to one physical hard stop (paired with the sim lower limit), press Enter, then *slowly* sweep to the other stop — crossing the wrap is fine, the script unwraps the stream cumulatively.

   ```bash
   python -m so101_real.joint_calibrate \
     --robot-config so101_real/configs/robot.yaml \
     --mode wrap-sweep \
     --joints wrist_roll \
     --wrap-period-rad 6.2832
   ```

   The script prints a pasteable YAML block; copy it into `joint_calibration.<joint>` in `robot.yaml`, replacing any prior entry. Both `wrap_period_rad` and `lero_branch_center_rad` must be present together (or both omitted).

3. **Verify.** With torque off, hand-rotate the joint across the wrap and confirm the sim-space readout is smooth and monotonic:

   ```bash
   python -m so101_real robot-test \
     --robot-config so101_real/configs/robot.yaml \
     --no-torque
   ```

   Then run `dev/test_joint_wrap_transform.py` for the unit-test round-trips, and do a short policy rollout while starting `wrist_roll` from several different physical branches to confirm reset takes the short path each time.

### Calibrating a non-wrapping joint

```bash
python -m so101_real.joint_calibrate \
  --robot-config so101_real/configs/robot.yaml \
  --mode sweep \
  --joints wrist_flex
```

Drive joint to each stop when prompted; paste the printed `scale` / `offset` into `robot.yaml`.

---

## Recorded output layout

With `--record`, a timestamped rollout directory is written inside the bundle:

```
deploy_bundle_<timestamp>/rollouts/rollout_<timestamp>/
  episode_000.mp4          # per-episode video at actual inference fps
  episode_001.mp4
  ...
  shard_00000.npz          # joint positions, targets, actions, frames
  rollout_manifest.json    # provenance: bundle hash, robot config, episode count
```

NPZ shard schema matches `collect_telemetry.py` so existing analysis tooling works on real-robot rollouts.
