# Sim-to-Real Journal — SO-101 Lift Cube

Ongoing notes on closing the gap between simulation training and real-world deployment.

---

## 2026-05-10 — Baseline deployment running; initial pose alignment

### Status

End-to-end pipeline completed today:

| Stage                                        | Status | Notes                             |
| -------------------------------------------- | ------ | --------------------------------- |
| Gen1 pipeline (collect → train CNN → export) | ✅      | `pipeline_20260509_135944`        |
| Gen2 pipeline (Gen1 CNN frozen, new policy)  | ✅      | `pipeline_20260509_161046`        |
| Gen3 15k training + export                   | ✅      | `experiments/2026-05-09_18-41-21` |
| Gen3 50k overnight training                  | 🔄      | Running in tmux, Gen2 CNN         |
| Real robot live with overlay                 | ✅      | Confirmed today                   |

### Observation: initial robot pose vs. sim reset pose

The sim resets each episode to a fixed nominal pose (from `optimal_policy_train_cnn.yaml`
`joints.starting_position`) with per-joint noise added. The real robot was starting from
wherever it ended the previous episode, which is a distribution mismatch at `t=0`.

**Sim nominal reset angles** (training config, `configs/optimal_policy_train_cnn.yaml`):

| Joint         | degrees | radians |
| ------------- | ------: | ------: |
| shoulder_pan  |     0.0 |   0.000 |
| shoulder_lift |     0.0 |   0.000 |
| elbow_flex    |   -25.0 |  -0.436 |
| wrist_flex    |    65.0 |   1.134 |
| wrist_roll    |   -90.0 |  -1.571 |
| gripper       |     0.0 |   0.000 |

**Sim noise ranges** (radians, per joint at reset):

| Joint         |   min |   max |
| ------------- | ----: | ----: |
| shoulder_pan  | -0.40 | +0.40 |
| shoulder_lift | -0.05 | +0.05 |
| elbow_flex    | -0.05 | +0.05 |
| wrist_flex    | -0.05 | +0.05 |
| wrist_roll    | -0.40 | +0.40 |
| gripper       | -0.40 | +0.40 |

### Fix applied

Added `reset_pose` to `so101_real/configs/robot.yaml` and implemented
`InferenceLoop._reset_to_start_pose()` in `so101_real/controller.py`.

At the start of each episode the controller:
1. Reads current joint positions from the robot.
2. Linearly interpolates from current → target over `reset_pose.duration_s = 3.0 s`
   at control Hz (one `send_joints` call per tick).
3. Logs "Start pose reached." then begins the inference loop.

The reset is skipped in `dry_run` mode and if `reset_pose.enabled: false`.

### Open questions / next steps

- [ ] Verify the real robot's kinematic pose at the sim reset angles looks correct on
      hardware (arm should be roughly vertical with wrist pointing forward).
- [ ] Measure whether the 3.0 s reset duration is comfortable — too slow is safe, too
      fast risks jerky motion.
- [ ] Once 50k Gen3 is trained: export and run a full eval episode on hardware.
- [ ] Camera field-of-view calibration: confirm the cube is visible in the same region
      of the frame as in sim (crop/resize pipeline must match).
- [ ] Consider adding a short pause (e.g. 0.5 s) after reset before episode starts to
      let any residual oscillation damp out.
- [ ] DR gap: sim uses FK-safe per-joint noise at reset; real deployment now starts from
      a fixed pose. Consider adding a small random offset at deploy time to match the
      distribution (requires FK safety check or conservative noise range).

---

## Template for future entries

```
## YYYY-MM-DD — <topic>

### Observation
<what was seen on the real robot>

### Hypothesis
<suspected root cause or gap>

### Change
<code/config modification>

### Result
<outcome after change>
```
