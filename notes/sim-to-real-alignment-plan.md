# Sim-to-Real Alignment Plan — SO-101 Lift Cube

Created: 2026-05-30  
Branch: `phase5`  
Baseline policy for comparison: `experiments/2026-05-29_19-33-02` (50k, seed 42, `frozen_resnet18`, post-d5f6054 camera fix)

---

## Motivation

After correcting the camera pose regression (commit d5f6054), the sim policy achieves ~89% success in
simulation but transfers poorly to hardware. The root causes are a cluster of sim-real gaps identified
by a systematic survey:

| Gap                              | Description                                                                                                                                                |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Action pipeline divergence**   | Sim and real applied EMA smoothing and delta-clamping through independent code paths with different step orders, creating a hidden distributional mismatch |
| **Lighting**                     | Sim used only a uniform dome light; real workspace has a directional desk lamp                                                                             |
| **Table colour**                 | Sim table was dark grey `(0.2, 0.2, 0.2)`; real table is white                                                                                             |
| **No visual DR**                 | Camera noise, brightness, contrast, blur, JPEG artefacts all absent in sim                                                                                 |
| **No proprio noise DR**          | Sim joint positions fed to the policy are perfect; real encoder readings have noise                                                                        |
| **No action latency DR**         | Real robot runs at 60 Hz with variable scheduling jitter; sim is deterministic                                                                             |
| **Camera pose uncertainty**      | Sim camera is placed at a fixed, manually tuned pose; real mount has ±mm play                                                                              |
| **No lens distortion**           | Real camera introduces barrel/radial distortion; sim camera is pinhole                                                                                     |
| **Cube/robot material mismatch** | Sim uses placeholder materials; real surfaces have different reflectance and texture                                                                       |

---

## Phase 1 — Action pipeline + lighting + table + visual DR

**Goal:** Eliminate the largest identifiable gaps before the first retrain.  
**Status: ✅ Complete (2026-05-30)**

All seven steps land together as one retrain — any one of them alone would confound the baseline.

### Step 1 — Shared `JointCommandSmoother` class

**File:** `so101/utils/control/joint_command_smoother.py` (new)  
**File:** `so101/utils/control/__init__.py` (new)

Implements the canonical 4-step action pipeline in a single, shared class used by both sim and real:

1. **Normalized → canonical** — `q = lower + 0.5 * (a + 1) * (upper - lower)`
2. **EMA smoothing** — `q_smooth = alpha * q + (1 - alpha) * q_prev`
3. **Per-step delta clamp vs. current position** — `q_safe = q_cur + clip(q_smooth - q_cur, -max_delta, +max_delta)`
4. **Joint-limit clamp** — `q_out = clip(q_safe, lower, upper)`

Handles any leading batch dims — `(n_joints,)` for real, `(num_envs, n_joints)` for sim.  
Pure PyTorch; no Isaac Lab dependency.

Key parameters (must match between sim and real):
```
ema_alpha    = 0.7      # EMA decay (from robot.yaml controller.ema_alpha)
max_delta_rad = 0.087   # ~5° per step at 60 Hz (from robot.yaml robot.max_delta_rad)
```

### Step 2 — Refactor real controller

**File:** `so101_real/controller.py`

Replaced the inline EMA + `SafetyLayer` + `JointUnitConverter` chain in `InferenceLoop._tick()`
with a single call to `self._smoother.step(action, q_meas)`.  
`reset_episode()` called at the start of each episode to clear EMA state.

### Step 3 — Wire smoother into sim env

**File:** `so101_rl/source/so101_rl/so101_rl/tasks/direct/so101_lift_cube/so101_lift_cube_env.py`

- `__init__`: instantiates `JointCommandSmoother` from URDF joint limits when `joint_command.enabled: true`.
- `_pre_physics_step`: routes actions through `self._smoother.step(actions_cmd, q_current)`. Binary gripper override is converted back to normalised space before the smoother so EMA and delta-clamp apply.
- `_reset_idx`: calls `self._smoother.reset(joint_pos[:, _dof_idx], env_ids=env_ids)` after the physics write.

Legacy path (`joint_command.enabled: false`) retained for ablation comparisons.

### Step 4 — `joint_command` config schema + YAML

**File:** `so101_rl/source/so101_rl/so101_rl/configurations/so101_env_params.py`  
**File:** `so101_rl/source/so101_rl/so101_rl/tasks/direct/so101_lift_cube/so101_lift_cube_env_cfg.py`  
**File:** `configs/policy_train.yaml`

Added `JointCommandCfg` dataclass and mirrored it into `So101LiftCubeCfg` alongside the other config groups.

```yaml
joint_command:
  enabled: true
  ema_alpha: 0.7          # must match robot.yaml controller.ema_alpha
  max_delta_rad: 0.087    # must match robot.yaml robot.max_delta_rad (~5°)
```

### Step 5 — Directional key light + lighting DR

**File:** `so101_rl/source/so101_rl/so101_rl/tasks/direct/so101_lift_cube/so101_lift_cube_env.py`  
**File:** `configs/policy_train.yaml`

Added a `DistantLightCfg` at `/World/KeyLight` (intensity 3000 lux, warm white 6500 K, ~35° tilt from vertical) to simulate a desk lamp. DR steps enabled:

```yaml
domain_randomization:
  world_lighting:
    enabled: true    # randomises dome light intensity/colour each reset
  env_lighting:
    enabled: true    # spawns a random point light per env (50% probability)
```

### Step 6 — White table

**File:** `so101_rl/source/so101_rl/so101_rl/configurations/table.py`

```python
# before
diffuse_color = (0.2, 0.2, 0.2)
# after
diffuse_color = (0.9, 0.9, 0.9)
```

### Step 7 — Visual DR

**File:** `configs/policy_train.yaml`

Enabled the following DR steps (all previously disabled):

| Step                  | Parameters                     |
| --------------------- | ------------------------------ |
| `gaussian_noise`      | `std_range: [0.005, 0.015]`    |
| `brightness`          | `range: [0.85, 1.15]`          |
| `contrast`            | `range: [0.85, 1.15]`          |
| `cheap_webcam_effect` | default                        |
| `motion_blur`         | `strength_range: [0.05, 0.15]` |
| `jpeg_compression`    | `quality_range: [60, 90]`      |

Still disabled: `gaussian_blur`, camera pose DR (Phase 3).

---

## Phase 2 — Proprio noise DR + action latency DR

**Goal:** Close the sensor-noise and timing gaps that remain after Phase 1.  
**Status: ⏳ Not started**

### Step 1 — Proprio noise DR

Add per-step Gaussian noise to joint-position observations in sim to match real encoder noise.

- Characterise real encoder noise: record a sequence of joint positions while the arm is stationary; measure per-joint standard deviation.
- Add a `proprio_noise` DR step in `dr_pipeline.py` that applies zero-mean Gaussian noise with `std_range` matching the measured value.
- Add `proprio_noise` to `So101EnvParams` / `policy_train.yaml`.

### Step 2 — Action latency DR

Add a random 1–3 step action delay in sim to match the scheduling jitter on the real 60 Hz loop.

- Implement a `LatencyBuffer` (circular buffer of the last N actions) in `so101_lift_cube_env.py`.
- At each step, sample `latency ~ U(latency_min, latency_max)` per env; use the buffered action from `latency` steps ago.
- Add `action_latency` config block (`min_steps`, `max_steps`) to `So101EnvParams` / `policy_train.yaml`.

### Step 3 — Pipeline YAML hash assertion

At bundle export time and at real-robot startup, compare the SHA-256 hash of the
`image_pipeline` section of the training config against the deployed config to guard
against silent mismatches.

---

## Phase 3 — Camera pose DR + material verification

**Goal:** Reduce sensitivity to camera mounting uncertainty; ensure material appearance is plausible.  
**Status: ⏳ Not started**

### Step 1 — Camera pose DR

Add small random perturbations to the camera pose at each episode reset:

- Translation noise: ±2 mm in x/y/z.
- Rotation noise: ±2° about x/y.
- Implement as a DR step in `dr_pipeline.py` using `XFormPrimView`.
- Add `camera_pose_dr` config block to `So101EnvParams` / `policy_train.yaml`.

### Step 2 — Cube and robot material verification

- Capture reference photos of the real black cube and SO-101 robot under the workspace light.
- Adjust sim material `diffuse_color`, `roughness`, and `metallic` values for the cube (`black_cube.py`) and robot links to reduce the appearance gap.
- Do not change geometry; material-only changes.

---

## Phase 4 — Lens distortion alignment

**Goal:** Remove the residual visual gap caused by real camera barrel distortion.  
**Status: ⏳ Not started — requires a stable Phase 1–3 baseline first**

### Approach

Undistort in real (do **not** change sim camera pose):

1. Run `camera_calibration.py` (or use OpenCV `calibrateCamera`) on a checkerboard sequence captured with the wrist camera to obtain intrinsics `K` and distortion coefficients `dist`.
2. Pre-compute `mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, K_new, (W, H), cv2.CV_32FC1)`.
3. Add an `UndistortStep` to `so101_real/image_pipeline.py` that applies `cv2.remap` before the resize step.
4. Store `K`, `dist`, and `K_new` in `so101_real/calibration/` alongside the capture sequence.

**Do not** restore principal-point offsets in `so101_real/camera.py` — the current manual visual tuning is load-bearing and should not be changed without a fresh calibration run.

---

## Validation protocol

After each phase:

| Check                                        | Pass criterion                                        |
| -------------------------------------------- | ----------------------------------------------------- |
| **Smoke train** (5k steps, 500 iters)        | Does not crash; `success_rate > 0`                    |
| **Full train** (50k steps)                   | `success_rate ≥ 0.85` in sim eval                     |
| **Sim play** (`--envs 8`, visual inspection) | Robot reaches and lifts cube; no erratic motion       |
| **Real robot** (20 episodes)                 | `success_rate ≥ 0.30` (Phase 1), improving each phase |

Expected sim performance degradation from Phase 1 visual DR is acceptable — harder DR ≠ worse policy; the policy must simply train longer or with more envs.

---

## Key implementation constraints

- The `JointCommandSmoother` parameters (`ema_alpha`, `max_delta_rad`) **must be identical** between `configs/policy_train.yaml` (sim) and `so101_real/configs/robot.yaml` (real). There is currently no runtime assertion enforcing this — it is enforced by convention.
- Do **not** switch to `FisheyeCameraCfg` — the current pinhole model + undistort-in-real approach (Phase 4) is the correct path.
- Do **not** change the camera pose in `so101_real/camera.py` — the principal-point tuning is manual and load-bearing until a proper calibration (Phase 4) replaces it.
- Lens distortion correction (Phase 4) must come **last**: it is a unidirectional change to the real pipeline and would confound earlier baselines if applied prematurely.
