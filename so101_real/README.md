# so101_real — Real-Robot Inference Package

Runs a trained SO-101 policy on physical hardware. No Isaac Lab dependency required.

## Quick start

```bash
conda activate lerobot
./scripts/run.py deploy \
  --robot-config so101_real/configs/robot.yaml \
  --episodes 10
```

Add `--overlay` to show a live OpenCV window on the workstation display, and `--record` to save per-episode MP4 video and NPZ telemetry.

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

## Camera setup

### Anti-flicker configuration (required for stable image)

Under fluorescent or LED lighting the camera will exhibit a **rolling horizontal scanline** if exposure is left on auto. This is a flicker beat artifact caused by the mains lighting frequency (50 Hz in the UK/EU) interacting with an unsynchronised shutter.

Fix: disable dynamic framerate and lock exposure to a multiple of the lighting cycle (10 ms for 50 Hz — full-wave-rectified LEDs/fluorescents flicker at 100 Hz, so any multiple of 10 ms is flicker-safe).

```bash
# Run once after plugging in the camera (resets on reconnect — see below)
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl -d /dev/video0 --set-ctrl=auto_exposure=1           # 1 = Manual
v4l2-ctl -d /dev/video0 --set-ctrl=exposure_time_absolute=100  # 100 × 100 µs = 10 ms
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=-50            # range: -64..64
```

If the scene is very dark, increase in multiples of 10 ms: `200` (20 ms), `300` (30 ms), etc.
If the scene is bright / blowing out, decrease: `50` is **not** safe (5 ms — not a multiple of 10 ms); the next safe step down is to reduce gain instead, or accept `100` as the minimum anti-flicker exposure.

If you are on a 60 Hz mains supply (US), the flicker frequency is 120 Hz (≈ 8.33 ms period). Safe values are multiples of ~8.33 ms: `83` (≈ 8.3 ms), `167` (≈ 16.7 ms), `250` (≈ 25 ms), etc.

To verify the current settings:

```bash
v4l2-ctl -d /dev/video0 \
  --get-ctrl=auto_exposure,exposure_time_absolute,exposure_dynamic_framerate,brightness,power_line_frequency
```

Expected output:

```
auto_exposure: 1
exposure_time_absolute: 100
exposure_dynamic_framerate: 0
brightness: -50
power_line_frequency: 1   # 1 = 50 Hz
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
  RUN+="/usr/bin/v4l2-ctl -d /dev/video0 --set-ctrl=exposure_dynamic_framerate=0 --set-ctrl=auto_exposure=1 --set-ctrl=exposure_time_absolute=100 --set-ctrl=brightness=-50"
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
