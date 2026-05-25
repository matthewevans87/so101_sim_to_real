#!/usr/bin/env python
"""joint_calibrate.py — Empirical joint convention diagnostic for sim-to-real.

Two calibration modes are supported; choose with --mode.

─── Mode: single (default) ──────────────────────────────────────────────────
Single-pose measurement.  You set one known sim pose, move the real arm to
match it, and the script prints an offset correction (assumes scale=1.0).

Workflow:
  1. Open Isaac Sim and load the robot USD.
  2. Move the robot to a convenient pose in the Articulation panel.
  3. Read the joint position values (degrees) and pass them via --sim-joints.
  4. The script unlocks motors, waits for you to match the pose, then reads.
  5. Outputs a joint_calibration YAML snippet ready to paste into robot.yaml.

Example::

    ./so101_real/joint_calibrate.py \\
        --robot-config so101_real/configs/robot.yaml \\
        --sim-joints="-10,5,0,0,0,0"

NOTE: if the first value is negative, use the = form to avoid argparse
misreading it as a flag: --sim-joints='-10,0,0,0,0,0'

─── Mode: sweep ──────────────────────────────────────────────────────────────
Two-stop per-joint sweep.  For each joint in turn, you move it to its LOWER
physical stop, press Enter, then to its UPPER physical stop, press Enter.
The script computes scale (including sign for reversed joints) and offset from
the two readings paired against the sim physics limits.

Sim limits are read from robot.yaml::sim_joint_limits, or from a bundle's
joint_config.yaml via --joint-config.

Example::

    ./so101_real/joint_calibrate.py \\
        --robot-config so101_real/configs/robot.yaml \\
        --mode sweep

    # Or supply limits directly from a bundle export:
    ./so101_real/joint_calibrate.py \\
        --robot-config so101_real/configs/robot.yaml \\
        --mode sweep \\
        --joint-config /path/to/bundle/joint_config.yaml

─── Background ───────────────────────────────────────────────────────────────
The sim (PhysX/USD) and LeRobot calibration use different zero-points and
scales for some joints.  The formula in robot.py is:

    q_sim = scale * q_lerobot_rad + offset_rad   (read_joints)
    q_lerobot = (q_sim - offset_rad) / scale     (send_joints)

A negative scale means the joint directions are physically reversed.

Joint order throughout: shoulder_pan, shoulder_lift, elbow_flex,
                        wrist_flex, wrist_roll, gripper.
"""

from __future__ import annotations

import argparse
import datetime
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import yaml

_JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


# ─── Live joint reading helper ───────────────────────────────────────────────


def _live_step(
    robot,
    joint_index: int,
    joint_name: str,
    direction_hint: str,
) -> float:
    """Show a live-updating joint reading and wait for the user to press Enter.

    Returns the joint position in **radians** at the moment Enter was pressed.

    Parameters
    ----------
    robot:
        Connected So101Robot with identity calibration.
    joint_index:
        0-based index into _JOINT_ORDER for the joint to display.
    joint_name:
        Name shown next to the live reading.
    direction_hint:
        Short phrase appended to each printed line, e.g.
        ``"↑ INCREASING  (aim for ~+68.9°)"``.
    """
    current_rad = [0.0]
    stop_flag = [False]

    def _reader() -> None:
        while not stop_flag[0]:
            try:
                q = robot.read_joints()
                current_rad[0] = float(q[joint_index])
                print(
                    f"\r  [{joint_name}] = {math.degrees(current_rad[0]):+8.3f}°"
                    f"  | {direction_hint}  ",
                    end="",
                    flush=True,
                )
            except Exception:
                pass
            time.sleep(0.1)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    sys.stdin.readline()  # blocks until Enter
    stop_flag[0] = True
    t.join(timeout=0.5)
    print()  # advance past the \r line
    return current_rad[0]


# ─── Shared robot connection helper ──────────────────────────────────────────


def _connect_raw(robot_config_path: Path):
    """Return a connected So101Robot with identity calibration (raw LeRobot values)."""
    try:
        from so101_real.robot import RobotConfig, So101Robot
    except ImportError:
        from robot import RobotConfig, So101Robot  # type: ignore[no-redef]

    cfg = RobotConfig.load(robot_config_path)
    cfg_raw = RobotConfig(
        port=cfg.port,
        calibration_file=cfg.calibration_file,
        max_delta_rad=cfg.max_delta_rad,
        reset_pose=None,
        joint_calibration={},
    )
    robot = So101Robot(cfg_raw, _JOINT_ORDER)
    robot.connect(dry_run=False)
    return robot


# ─── Mode: single ─────────────────────────────────────────────────────────────


def _read_raw_lerobot_deg(robot_config_path: Path) -> dict[str, float]:
    """Connect, unlock motors, wait for user to position arm, read raw degrees."""
    robot = _connect_raw(robot_config_path)
    robot._follower.bus.disable_torque()
    print("\n[Motors UNLOCKED] Manually move the arm to the target pose.")
    input("Press Enter when the arm is in position...")
    robot._follower.bus.enable_torque()
    print("Motors re-locked. Reading positions...")
    q_rad = robot.read_joints()
    robot.disconnect()
    return {name: math.degrees(float(q_rad[i])) for i, name in enumerate(_JOINT_ORDER)}


def _run_single(args: argparse.Namespace) -> None:
    print("\n=== SO-101 Joint Convention Diagnostic (single-pose) ===\n")

    # ── Get sim joint positions ───────────────────────────────────────────────
    if args.sim_joints is not None:
        try:
            sim_deg_list = [float(v.strip()) for v in args.sim_joints.split(",")]
        except ValueError:
            print("[ERROR] --sim-joints must be comma-separated numbers.")
            return
        if len(sim_deg_list) != len(_JOINT_ORDER):
            print(f"[ERROR] --sim-joints must have exactly {len(_JOINT_ORDER)} values.")
            return
        sim_deg = {name: sim_deg_list[i] for i, name in enumerate(_JOINT_ORDER)}
    else:
        print(
            "Enter the joint positions shown in Isaac Sim (degrees, sim convention).\n"
            "  Press Enter alone to accept 0.0.\n"
        )
        sim_deg = {}
        for name in _JOINT_ORDER:
            raw = input(f"  {name:15s} [deg]: ").strip()
            sim_deg[name] = float(raw) if raw else 0.0

    print("\nSim reference pose:")
    for name in _JOINT_ORDER:
        print(f"  {name:15s} = {sim_deg[name]:+.2f}°")

    print(
        "\nThe motors will now be UNLOCKED so you can move the arm freely.\n"
        "Position the real arm to match the sim pose above, then press Enter."
    )
    print("\nConnecting to robot...")
    try:
        positions_deg = _read_raw_lerobot_deg(args.robot_config)
    except Exception as exc:
        print(f"\n[ERROR] Could not read robot: {exc}")
        return

    # ── Load existing calibration for reference ───────────────────────────────
    with open(args.robot_config) as f:
        config_data = yaml.safe_load(f)
    existing_cal = config_data.get("joint_calibration") or {}

    # ── Print comparison table ─────────────────────────────────────────────────
    print()
    print(
        f"{'Joint':15s}  {'LeRobot (raw)':13s}  {'Sim expects':11s}  "
        f"{'Delta (L-S)':11s}  Status"
    )
    print("-" * 72)

    suggested_cal: dict[str, dict] = {}
    for name in _JOINT_ORDER:
        lero_deg = positions_deg[name]
        sim_d = sim_deg[name]
        delta = lero_deg - sim_d
        abs_delta = abs(delta)

        if abs_delta < 2.0:
            status = "✓ OK"
        elif abs_delta < 8.0:
            status = "⚠ warn"
        else:
            status = "✗ FIX"

        print(
            f"{name:15s}  {lero_deg:+11.2f}°  {sim_d:+9.2f}°  "
            f"{delta:+9.2f}°  {status}"
        )

        if abs_delta >= 2.0:
            existing = existing_cal.get(name, {})
            scale = existing.get("lero_to_sim_scale", 1.0)
            offset_deg = sim_d - scale * lero_deg
            suggested_cal[name] = {
                "lero_to_sim_scale": round(scale, 6),
                "lero_to_sim_offset_rad": round(math.radians(offset_deg), 6),
            }

    print()
    if suggested_cal:
        print("Suggested joint_calibration (paste into robot.yaml):\n")
        print("joint_calibration:")
        for name, entry in suggested_cal.items():
            print(f"  {name}:")
            print(f"    lero_to_sim_scale: {entry['lero_to_sim_scale']}")
            print(f"    lero_to_sim_offset_rad: {entry['lero_to_sim_offset_rad']}")
        print()
        print(
            "NOTE: Scale=1.0 is assumed (single measurement point).\n"
            "Use --mode sweep to derive scale empirically via two-stop calibration."
        )
    else:
        print("All joints within 2° tolerance — no additional corrections needed.")
    print()


# ─── Mode: sweep ──────────────────────────────────────────────────────────────


def _load_sim_limits(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Load sim joint limits from --joint-config bundle file or robot.yaml."""
    if args.joint_config is not None:
        jc_path = Path(args.joint_config)
        if not jc_path.exists():
            raise FileNotFoundError(f"--joint-config not found: {jc_path}")
        with open(jc_path) as f:
            jc = yaml.safe_load(f)
        joints = jc["active_joints"]
        lowers = jc["joint_lower_rad"]
        uppers = jc["joint_upper_rad"]
        if len(joints) != len(lowers) or len(joints) != len(uppers):
            raise ValueError(
                f"joint_config.yaml has mismatched lengths: joints={len(joints)}, lower={len(lowers)}, upper={len(uppers)}"
            )
        limits = {
            name: {"lower_rad": float(lo), "upper_rad": float(hi)}
            for name, lo, hi in zip(joints, lowers, uppers)
        }
        print(f"[sim limits] Loaded from {jc_path}")
        return limits

    # Fall back to robot.yaml::sim_joint_limits
    with open(args.robot_config) as f:
        config_data = yaml.safe_load(f)
    raw = config_data.get("sim_joint_limits")
    if not raw:
        raise ValueError(
            "No sim_joint_limits found in robot.yaml and --joint-config not given.\n"
            "Either add a sim_joint_limits section to robot.yaml (see template),\n"
            "or pass --joint-config /path/to/bundle/joint_config.yaml"
        )
    limits: dict[str, dict[str, float]] = {}
    for name, entry in raw.items():
        limits[name] = {
            "lower_rad": float(entry["lower_rad"]),
            "upper_rad": float(entry["upper_rad"]),
        }
    print(f"[sim limits] Loaded from {args.robot_config} :: sim_joint_limits")
    return limits


def _run_sweep(args: argparse.Namespace) -> None:
    print("\n=== SO-101 Joint Convention Sweep Calibration ===\n")

    # Resolve the joint filter (--joints), if any.
    if getattr(args, "joints", None) is not None:
        requested = {j.strip() for j in args.joints.split(",")}
        unknown = requested - set(_JOINT_ORDER)
        if unknown:
            print(f"[ERROR] Unknown joint name(s): {sorted(unknown)}")
            print(f"  Valid names: {_JOINT_ORDER}")
            return
        joint_filter = [j for j in _JOINT_ORDER if j in requested]
        print(f"[filter] Calibrating only: {joint_filter}\n")
    else:
        joint_filter = list(_JOINT_ORDER)

    try:
        sim_limits = _load_sim_limits(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return

    # Load existing calibration to detect sign-reversed joints so we can
    # give correct movement directions.
    with open(args.robot_config) as f:
        config_data = yaml.safe_load(f)
    existing_cal = config_data.get("joint_calibration") or {}

    # Show the limits we'll use and warn about reversed joints up front.
    print(f"{'Joint':15s}  {'Sim lower':>12s}  {'Sim upper':>12s}  Note")
    print("-" * 62)
    for name in joint_filter:
        lim = sim_limits.get(name)
        if not lim:
            print(f"{name:15s}  (no limits — will skip)")
            continue
        scale_sign = existing_cal.get(name, {}).get("lero_to_sim_scale", 1.0)
        note = "[REVERSED]" if scale_sign < 0 else ""
        print(
            f"{name:15s}  {math.degrees(lim['lower_rad']):+10.3f}°  "
            f"{math.degrees(lim['upper_rad']):+10.3f}°  {note}"
        )

    print(
        "\nFor each joint you will be asked to reach TWO reference positions.\n"
        "Only that joint's reading matters — other joints can be anywhere.\n"
        "[REVERSED] joints: sim-lower physically corresponds to the POSITIVE\n"
        "  LeRobot direction, and sim-upper to the NEGATIVE direction.\n"
        "  The prompts below will tell you which way to move.\n"
    )

    print("Connecting to robot...")
    try:
        robot = _connect_raw(args.robot_config)
    except Exception as exc:
        print(f"[ERROR] Could not connect: {exc}")
        return

    robot._follower.bus.disable_torque()
    print("[Motors UNLOCKED]\n")

    suggested_cal: dict[str, dict] = {}
    today = datetime.date.today().isoformat()

    try:
        for i, joint in enumerate(_JOINT_ORDER):
            if joint not in joint_filter:
                continue

            lim = sim_limits.get(joint)
            if lim is None:
                print(f"\n[SKIP] No sim limits for {joint}.")
                continue

            sim_lower_rad = lim["lower_rad"]
            sim_upper_rad = lim["upper_rad"]
            sim_lower_deg = math.degrees(sim_lower_rad)
            sim_upper_deg = math.degrees(sim_upper_rad)

            # Determine expected lerobot direction from existing calibration
            existing_scale = existing_cal.get(joint, {}).get("lero_to_sim_scale", 1.0)
            reversed_joint = existing_scale < 0

            if reversed_joint:
                # sim-lower corresponds to positive lerobot, sim-upper to negative
                step1_sim_desc = f"SIM-LOWER ({sim_lower_deg:+.3f}°)"
                step1_lero_dir = (
                    "POSITIVE lerobot direction (joint feels like it goes UP/forward)"
                )
                step2_sim_desc = f"SIM-UPPER ({sim_upper_deg:+.3f}°)"
                step2_lero_dir = (
                    "NEGATIVE lerobot direction (joint feels like it goes DOWN/back)"
                )
                # step1 lerobot reading will be larger, step2 smaller
                step1_sim_rad = sim_lower_rad
                step2_sim_rad = sim_upper_rad
            else:
                step1_sim_desc = f"SIM-LOWER ({sim_lower_deg:+.3f}°)"
                step1_lero_dir = (
                    "NEGATIVE lerobot direction (joint goes toward its lower limit)"
                )
                step2_sim_desc = f"SIM-UPPER ({sim_upper_deg:+.3f}°)"
                step2_lero_dir = (
                    "POSITIVE lerobot direction (joint goes toward its upper limit)"
                )
                step1_sim_rad = sim_lower_rad
                step2_sim_rad = sim_upper_rad

            # Build direction hints with predicted targets from existing calibration.
            existing_offset = existing_cal.get(joint, {}).get(
                "lero_to_sim_offset_rad", 0.0
            )
            try:
                pred1_deg = math.degrees(
                    (step1_sim_rad - existing_offset) / existing_scale
                )
                pred2_deg = math.degrees(
                    (step2_sim_rad - existing_offset) / existing_scale
                )
                step1_hint = (
                    f"↑ INCREASING  (aim for ~{pred1_deg:+.1f}°)"
                    if reversed_joint
                    else f"↓ DECREASING  (aim for ~{pred1_deg:+.1f}°)"
                )
                step2_hint = (
                    f"↓ DECREASING  (aim for ~{pred2_deg:+.1f}°)"
                    if reversed_joint
                    else f"↑ INCREASING  (aim for ~{pred2_deg:+.1f}°)"
                )
            except ZeroDivisionError:
                step1_hint = "↑ INCREASING" if reversed_joint else "↓ DECREASING"
                step2_hint = "↓ DECREASING" if reversed_joint else "↑ INCREASING"

            print(f"\n{'=' * 60}")
            print(f"  Joint {i + 1}/{len(_JOINT_ORDER)}: {joint}", end="")
            print(f"  [REVERSED]" if reversed_joint else "")
            print(f"  Sim range: {sim_lower_deg:+.3f}° → {sim_upper_deg:+.3f}°")
            print(f"{'=' * 60}")

            # ── Stop 1 ────────────────────────────────────────────────────────
            print(
                f"\n  Step 1/2 — {step1_sim_desc}\n"
                f"  Move [{joint}] in the {step1_lero_dir}.\n"
                f"  Push until you hit the physical stop (or cable limit).\n"
                f"  Watch the live reading — press Enter when at the stop:"
            )
            lero_stop1 = _live_step(robot, i, joint, step1_hint)
            print(
                f"  → Recorded: {math.degrees(lero_stop1):+.4f}°  ({lero_stop1:.6f} rad)"
            )

            # ── Stop 2 ────────────────────────────────────────────────────────
            print(
                f"\n  Step 2/2 — {step2_sim_desc}\n"
                f"  Now move [{joint}] in the {step2_lero_dir}.\n"
                f"  Push until you hit the physical stop (or cable limit).\n"
                f"  Watch the live reading — press Enter when at the stop:"
            )
            lero_stop2 = _live_step(robot, i, joint, step2_hint)
            print(
                f"  → Recorded: {math.degrees(lero_stop2):+.4f}°  ({lero_stop2:.6f} rad)"
            )

            # ── Compute calibration ───────────────────────────────────────────
            # For reversed joints: stop1 lero > stop2 lero (we moved to sim-lower
            # via positive lero, then sim-upper via negative lero).
            # Assign: lero_lower_val = stop2, lero_upper_val = stop1 for reversed.
            if reversed_joint:
                # step1 → sim_lower_rad, step2 → sim_upper_rad
                lero_at_sim_lower = lero_stop1
                lero_at_sim_upper = lero_stop2
            else:
                lero_at_sim_lower = lero_stop1
                lero_at_sim_upper = lero_stop2

            span_lero = lero_at_sim_upper - lero_at_sim_lower
            span_sim = sim_upper_rad - sim_lower_rad

            if abs(span_lero) < math.radians(5.0):
                print(
                    f"  [WARN] The two readings are only {abs(math.degrees(span_lero)):.2f}° apart "
                    f"in LeRobot space — skipping {joint}.\n"
                    f"  Possible causes: wrong direction, encoder wrap, joint not reaching stop."
                )
                continue

            scale = span_sim / span_lero
            offset_rad = sim_lower_rad - scale * lero_at_sim_lower

            # Sanity: residuals at both reference points (should be ~0)
            residual_lower_deg = abs(
                math.degrees(scale * lero_at_sim_lower + offset_rad - sim_lower_rad)
            )
            residual_upper_deg = abs(
                math.degrees(scale * lero_at_sim_upper + offset_rad - sim_upper_rad)
            )

            print(f"\n  Result:")
            print(f"    scale:      {scale:+.6f}")
            print(
                f"    offset_rad: {offset_rad:+.6f}  ({math.degrees(offset_rad):+.4f}°)"
            )
            print(
                f"    Residuals:  lower={residual_lower_deg:.4f}°  upper={residual_upper_deg:.4f}°"
            )

            if abs(scale) > 10.0:
                print(
                    f"  [WARN] Scale {scale:.4f} is suspiciously large.\n"
                    f"  This usually means the joint was moved in the WRONG DIRECTION\n"
                    f"  or the encoder wrapped mid-sweep. Discard this result."
                )
            elif scale < 0:
                print(
                    f"  [NOTE] Negative scale → LeRobot and sim rotate in OPPOSITE directions."
                )

            suggested_cal[joint] = {
                "lero_to_sim_scale": round(scale, 6),
                "lero_to_sim_offset_rad": round(offset_rad, 6),
                "_lero_lower_deg": round(math.degrees(lero_at_sim_lower), 4),
                "_lero_upper_deg": round(math.degrees(lero_at_sim_upper), 4),
                "_sim_lower_deg": round(sim_lower_deg, 4),
                "_sim_upper_deg": round(sim_upper_deg, 4),
                "_date": today,
            }

    finally:
        robot._follower.bus.enable_torque()
        print("\n[Motors LOCKED]")
        robot.disconnect()

    if not suggested_cal:
        print("\nNo calibration data collected.")
        return

    # ── Print final YAML ──────────────────────────────────────────────────────
    print("\n\n" + "=" * 60)
    print("SWEEP CALIBRATION RESULTS — paste into robot.yaml")
    print("=" * 60 + "\n")
    print("joint_calibration:")
    for joint, cal in suggested_cal.items():
        print(f"  {joint}:")
        print(
            f"    # Sweep calibration ({cal['_date']}):"
            f" lower stop lero={cal['_lero_lower_deg']:+.4f}° → sim={cal['_sim_lower_deg']:+.4f}°,"
            f" upper stop lero={cal['_lero_upper_deg']:+.4f}° → sim={cal['_sim_upper_deg']:+.4f}°"
        )
        print(f"    lero_to_sim_scale: {cal['lero_to_sim_scale']}")
        print(f"    lero_to_sim_offset_rad: {cal['lero_to_sim_offset_rad']}")
    print()


# ─── Mode: discontinuity (raw stream capture) ────────────────────────────────


def _stream_capture_one_joint(
    robot,
    joint_index: int,
    joint_name: str,
    poll_hz: float,
) -> list[tuple[float, float]]:
    """Capture (t_seconds, raw_lerobot_rad) samples for one joint until Enter is pressed."""
    samples: list[tuple[float, float]] = []
    stop_flag = [False]
    t0 = time.monotonic()
    period_s = 1.0 / poll_hz

    def _reader() -> None:
        last_print = 0.0
        while not stop_flag[0]:
            try:
                q = robot.read_joints()
                t_now = time.monotonic() - t0
                v = float(q[joint_index])
                samples.append((t_now, v))
                if t_now - last_print >= 0.1:
                    print(
                        f"\r  [{joint_name}] = {math.degrees(v):+9.3f}°"
                        f"  (n={len(samples)})  ",
                        end="",
                        flush=True,
                    )
                    last_print = t_now
            except Exception:
                pass
            time.sleep(period_s)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    sys.stdin.readline()  # blocks until Enter
    stop_flag[0] = True
    t.join(timeout=0.5)
    print()
    return samples


def _analyse_stream(
    samples: list[tuple[float, float]],
    jump_threshold_rad: float,
) -> dict:
    """Detect discontinuities and summarise continuous segments.

    A 'jump' is any consecutive pair with ``|Δq| > jump_threshold_rad``.
    """
    if len(samples) < 2:
        return {
            "segments": [],
            "jumps": [],
            "physical_span_rad": 0.0,
            "n_samples": len(samples),
        }

    jumps: list[dict] = []
    segments: list[dict] = []
    physical_span = 0.0

    seg_start = 0
    for i in range(1, len(samples)):
        prev = samples[i - 1][1]
        cur = samples[i][1]
        dq = cur - prev
        if abs(dq) > jump_threshold_rad:
            seg_vals = [s[1] for s in samples[seg_start:i]]
            segments.append(
                {
                    "start_idx": seg_start,
                    "end_idx": i - 1,
                    "lo_rad": min(seg_vals),
                    "hi_rad": max(seg_vals),
                    "span_rad": max(seg_vals) - min(seg_vals),
                }
            )
            jumps.append(
                {
                    "idx": i,
                    "prev_rad": prev,
                    "next_rad": cur,
                    "delta_rad": dq,
                }
            )
            seg_start = i
        else:
            physical_span += abs(dq)

    seg_vals = [s[1] for s in samples[seg_start:]]
    if seg_vals:
        segments.append(
            {
                "start_idx": seg_start,
                "end_idx": len(samples) - 1,
                "lo_rad": min(seg_vals),
                "hi_rad": max(seg_vals),
                "span_rad": max(seg_vals) - min(seg_vals),
            }
        )

    return {
        "segments": segments,
        "jumps": jumps,
        "physical_span_rad": physical_span,
        "n_samples": len(samples),
    }


def _run_discontinuity(args: argparse.Namespace) -> None:
    print("\n=== SO-101 Joint Discontinuity Diagnostic ===\n")

    if args.joints is None:
        print(
            "[ERROR] --joints NAME is required for discontinuity mode (one joint at a time)."
        )
        return
    requested = [j.strip() for j in args.joints.split(",")]
    if len(requested) != 1:
        print(f"[ERROR] discontinuity mode accepts exactly one joint, got: {requested}")
        return
    joint_name = requested[0]
    if joint_name not in _JOINT_ORDER:
        print(f"[ERROR] Unknown joint {joint_name!r}. Valid: {_JOINT_ORDER}")
        return
    joint_index = _JOINT_ORDER.index(joint_name)

    poll_hz = float(args.poll_hz)
    if poll_hz <= 0.0:
        print(f"[ERROR] --poll-hz must be positive, got {poll_hz}")
        return
    jump_threshold_rad = float(args.jump_threshold_rad)
    if jump_threshold_rad <= 0.0:
        print(
            f"[ERROR] --jump-threshold-rad must be positive, got {jump_threshold_rad}"
        )
        return

    print("Connecting to robot...")
    try:
        robot = _connect_raw(args.robot_config)
    except Exception as exc:
        print(f"[ERROR] Could not connect: {exc}")
        return

    robot._follower.bus.disable_torque()
    print("[Motors UNLOCKED]\n")
    print(
        f"Slowly sweep [{joint_name}] from one physical stop to the other.\n"
        f"  Poll rate: {poll_hz:.1f} Hz   Jump threshold: {math.degrees(jump_threshold_rad):.1f}°\n"
        f"  Press Enter when done.\n"
    )

    try:
        samples = _stream_capture_one_joint(robot, joint_index, joint_name, poll_hz)
    finally:
        robot._follower.bus.enable_torque()
        print("[Motors LOCKED]")
        robot.disconnect()

    if not samples:
        print("\n[ERROR] No samples captured.")
        return

    summary = _analyse_stream(samples, jump_threshold_rad)
    n = summary["n_samples"]
    segs = summary["segments"]
    jumps = summary["jumps"]
    physical_span = summary["physical_span_rad"]

    print(
        f"\nCaptured {n} samples in {samples[-1][0]:.2f} s "
        f"(actual rate ≈ {(n - 1) / max(samples[-1][0], 1e-6):.1f} Hz)"
    )
    print(f"\nContinuous segments ({len(segs)}):")
    for i, seg in enumerate(segs):
        print(
            f"  seg {i}: [{math.degrees(seg['lo_rad']):+9.3f}°,"
            f" {math.degrees(seg['hi_rad']):+9.3f}°]"
            f"  span={math.degrees(seg['span_rad']):8.3f}°"
            f"  ({seg['end_idx'] - seg['start_idx'] + 1} samples)"
        )

    print(f"\nDiscontinuities ({len(jumps)}):")
    two_pi = 2.0 * math.pi
    for j in jumps:
        d = j["delta_rad"]
        k_2pi = round(d / two_pi)
        residual_2pi_deg = math.degrees(d - k_2pi * two_pi)
        print(
            f"  Δ={math.degrees(d):+9.3f}°  ({d:+.4f} rad)"
            f"  ≈ {k_2pi:+d} × 2π  (residual {residual_2pi_deg:+.2f}°)"
        )

    print(
        f"\nTotal physical span (sum of continuous |Δq|): "
        f"{math.degrees(physical_span):.2f}°  ({physical_span:.4f} rad)"
    )

    if jumps:
        magnitudes = sorted(abs(j["delta_rad"]) for j in jumps)
        median = magnitudes[len(magnitudes) // 2]
        print(
            f"\nMedian jump magnitude: {math.degrees(median):.3f}°  "
            f"({median:.4f} rad)"
        )
        print(
            f"  vs 2π = {math.degrees(two_pi):.3f}°" f"   (ratio {median / two_pi:.3f})"
        )

    # ── Save to YAML for reproducibility ──────────────────────────────────────
    out_dir = Path(__file__).resolve().parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{joint_name}_discontinuity_{ts}.yaml"

    payload = {
        "joint": joint_name,
        "timestamp": ts,
        "poll_hz": poll_hz,
        "jump_threshold_rad": jump_threshold_rad,
        "n_samples": n,
        "duration_s": float(samples[-1][0]),
        "physical_span_rad": float(physical_span),
        "physical_span_deg": float(math.degrees(physical_span)),
        "segments": [
            {
                "start_idx": int(s["start_idx"]),
                "end_idx": int(s["end_idx"]),
                "lo_rad": float(s["lo_rad"]),
                "hi_rad": float(s["hi_rad"]),
                "span_rad": float(s["span_rad"]),
            }
            for s in segs
        ],
        "jumps": [
            {
                "idx": int(j["idx"]),
                "prev_rad": float(j["prev_rad"]),
                "next_rad": float(j["next_rad"]),
                "delta_rad": float(j["delta_rad"]),
            }
            for j in jumps
        ],
        "samples": [{"t_s": float(t_), "raw_lero_rad": float(v)} for t_, v in samples],
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"\n[saved] {out_path}")


# ─── Mode: wrap-sweep (continuous-capture calibration) ───────────────────────


def _unwrap_stream(values: list[float], period: float) -> list[float]:
    """Cumulative unwrap: each sample shifted to be within ±period/2 of the previous."""
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        k = round((out[-1] - v) / period)
        out.append(v + k * period)
    return out


def _run_wrap_sweep(args: argparse.Namespace) -> None:
    print("\n=== SO-101 Joint Wrap-Aware Sweep Calibration ===\n")

    if args.joints is None:
        print(
            "[ERROR] --joints NAME is required for wrap-sweep mode (one joint at a time)."
        )
        return
    requested = [j.strip() for j in args.joints.split(",")]
    if len(requested) != 1:
        print(f"[ERROR] wrap-sweep accepts exactly one joint, got: {requested}")
        return
    joint_name = requested[0]
    if joint_name not in _JOINT_ORDER:
        print(f"[ERROR] Unknown joint {joint_name!r}. Valid: {_JOINT_ORDER}")
        return
    joint_index = _JOINT_ORDER.index(joint_name)

    if args.wrap_period_rad is None:
        print("[ERROR] --wrap-period-rad is required for wrap-sweep mode.")
        return
    period = float(args.wrap_period_rad)
    if period <= 0.0:
        print(f"[ERROR] --wrap-period-rad must be positive, got {period}")
        return

    poll_hz = float(args.poll_hz)
    if poll_hz <= 0.0:
        print(f"[ERROR] --poll-hz must be positive, got {poll_hz}")
        return

    try:
        sim_limits = _load_sim_limits(args)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return
    if joint_name not in sim_limits:
        print(f"[ERROR] No sim_joint_limits entry for {joint_name!r}.")
        return
    sim_lower_rad = sim_limits[joint_name]["lower_rad"]
    sim_upper_rad = sim_limits[joint_name]["upper_rad"]

    print(
        f"Joint:        {joint_name}\n"
        f"Sim range:    [{math.degrees(sim_lower_rad):+.3f}°, "
        f"{math.degrees(sim_upper_rad):+.3f}°]\n"
        f"Wrap period:  {math.degrees(period):.3f}°  ({period:.4f} rad)\n"
        f"Poll rate:    {poll_hz:.1f} Hz\n"
    )

    print("Connecting to robot...")
    try:
        robot = _connect_raw(args.robot_config)
    except Exception as exc:
        print(f"[ERROR] Could not connect: {exc}")
        return

    robot._follower.bus.disable_torque()
    print("[Motors UNLOCKED]\n")

    try:
        # Stop A
        print(
            f"Step 1/2 — Move [{joint_name}] to one physical hard stop "
            f"(this will be paired with sim={math.degrees(sim_lower_rad):+.3f}°).\n"
            f"  Watch the live reading; press Enter when at the stop:"
        )
        lero_at_A = _live_step(robot, joint_index, joint_name, "STOP A")

        # Continuous sweep to stop B
        print(
            f"\nStep 2/2 — Now slowly sweep [{joint_name}] all the way to the "
            f"OTHER physical hard stop\n"
            f"  (this will be paired with sim={math.degrees(sim_upper_rad):+.3f}°).\n"
            f"  Sweep slowly enough that consecutive samples don't change by more "
            f"than ~{math.degrees(period / 4):.1f}°.\n"
            f"  Press Enter when at the second stop:"
        )
        samples = _stream_capture_one_joint(robot, joint_index, joint_name, poll_hz)
    finally:
        robot._follower.bus.enable_torque()
        print("[Motors LOCKED]")
        robot.disconnect()

    if len(samples) < 2:
        print("\n[ERROR] Not enough samples captured during sweep.")
        return

    raw = [v for _, v in samples]
    # Cumulatively unwrap the sweep stream onto a continuous coordinate, then
    # rigid-shift the whole chain so its first sample lands on the branch of
    # lero_at_A.  Stop B is then the last sample of the shifted chain.
    cumulative = _unwrap_stream(raw, period)
    k_align = round((lero_at_A - cumulative[0]) / period)
    cumulative = [v + k_align * period for v in cumulative]
    lero_at_B = cumulative[-1]

    span_lero = lero_at_B - lero_at_A
    span_sim = sim_upper_rad - sim_lower_rad

    if abs(span_lero) < math.radians(5.0):
        print(
            f"\n[ERROR] Unwrapped sweep span is only "
            f"{abs(math.degrees(span_lero)):.2f}° — sweep too short or unwrap failed."
        )
        return

    scale = span_sim / span_lero
    offset_rad = sim_lower_rad - scale * lero_at_A
    branch_center = (lero_at_A + lero_at_B) / 2.0

    residual_A_deg = abs(math.degrees(scale * lero_at_A + offset_rad - sim_lower_rad))
    residual_B_deg = abs(math.degrees(scale * lero_at_B + offset_rad - sim_upper_rad))

    print("\n" + "=" * 60)
    print("WRAP-SWEEP RESULT")
    print("=" * 60)
    print(
        f"  Stop A (raw):        {math.degrees(lero_at_A):+10.4f}°  ({lero_at_A:+.6f} rad)"
    )
    print(
        f"  Stop B (unwrapped):  {math.degrees(lero_at_B):+10.4f}°  ({lero_at_B:+.6f} rad)"
    )
    print(
        f"  Lero span:           {math.degrees(span_lero):+10.4f}°  ({span_lero:+.6f} rad)"
    )
    print(
        f"  Sim  span:           {math.degrees(span_sim):+10.4f}°  ({span_sim:+.6f} rad)"
    )
    print(f"  scale:               {scale:+.6f}")
    print(
        f"  offset_rad:          {offset_rad:+.6f}  ({math.degrees(offset_rad):+.4f}°)"
    )
    print(
        f"  branch_center_rad:   {branch_center:+.6f}  ({math.degrees(branch_center):+.4f}°)"
    )
    print(f"  Residuals:           A={residual_A_deg:.4f}°   B={residual_B_deg:.4f}°")

    today = datetime.date.today().isoformat()
    print(
        "\nPaste into robot.yaml::joint_calibration:\n\n"
        f"  {joint_name}:\n"
        f"    # Wrap-sweep calibration ({today}):\n"
        f"    #   stop A raw lero={math.degrees(lero_at_A):+.4f}° → sim={math.degrees(sim_lower_rad):+.4f}°\n"
        f"    #   stop B unwrapped lero={math.degrees(lero_at_B):+.4f}° → sim={math.degrees(sim_upper_rad):+.4f}°\n"
        f"    #   wrap period {math.degrees(period):.3f}° ({period:.4f} rad)\n"
        f"    lero_to_sim_scale: {round(scale, 6)}\n"
        f"    lero_to_sim_offset_rad: {round(offset_rad, 6)}\n"
        f"    wrap_period_rad: {round(period, 6)}\n"
        f"    lero_branch_center_rad: {round(branch_center, 6)}\n"
    )

    out_dir = Path(__file__).resolve().parent / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{joint_name}_wrapsweep_{ts}.yaml"
    payload = {
        "joint": joint_name,
        "timestamp": ts,
        "wrap_period_rad": float(period),
        "poll_hz": poll_hz,
        "sim_lower_rad": float(sim_lower_rad),
        "sim_upper_rad": float(sim_upper_rad),
        "lero_at_A_rad": float(lero_at_A),
        "lero_at_B_rad_unwrapped": float(lero_at_B),
        "scale": float(scale),
        "offset_rad": float(offset_rad),
        "branch_center_rad": float(branch_center),
        "residual_A_deg": float(residual_A_deg),
        "residual_B_deg": float(residual_B_deg),
        "samples_raw": [
            {"t_s": float(t_), "raw_lero_rad": float(v)} for t_, v in samples
        ],
        "samples_unwrapped_lero_rad": [float(v) for v in cumulative],
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[saved] {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--robot-config",
        required=True,
        type=Path,
        help="Path to robot.yaml (e.g. so101_real/configs/robot.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "sweep", "discontinuity", "wrap-sweep"],
        default="single",
        help=(
            "'single': one-pose offset measurement (default); "
            "'sweep': per-joint two-stop calibration that derives scale and offset; "
            "'discontinuity': stream-capture one joint while the user sweeps it, "
            "detect wrap jumps and report continuous segments; "
            "'wrap-sweep': continuous-capture calibration for one wrappable joint."
        ),
    )
    # single-mode args
    parser.add_argument(
        "--sim-joints",
        default=None,
        help=(
            "[single mode] Comma-separated sim joint positions (degrees), "
            "order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.  "
            "If omitted the script prompts interactively.  "
            "NOTE: if the first value is negative use = form: --sim-joints='-10,0,0,0,0,0'"
        ),
    )
    # sweep-mode args
    parser.add_argument(
        "--joint-config",
        default=None,
        metavar="PATH",
        help=(
            "[sweep mode] Path to a bundle's joint_config.yaml to read sim joint limits from. "
            "If omitted, limits are read from robot.yaml::sim_joint_limits."
        ),
    )
    parser.add_argument(
        "--joints",
        default=None,
        metavar="NAME[,NAME...]",
        help=(
            "[sweep / discontinuity / wrap-sweep] Comma-separated subset of joints. "
            "For sweep, omit to calibrate all. For discontinuity / wrap-sweep, "
            "exactly one joint is required (e.g. --joints wrist_roll)."
        ),
    )
    # discontinuity / wrap-sweep args
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=50.0,
        help="[discontinuity / wrap-sweep] Polling rate in Hz (default: 50).",
    )
    parser.add_argument(
        "--jump-threshold-rad",
        type=float,
        default=math.pi / 2,
        help=(
            "[discontinuity] Consecutive-sample delta above which a transition is "
            "flagged as a wrap jump (default: pi/2 rad = 90°)."
        ),
    )
    parser.add_argument(
        "--wrap-period-rad",
        type=float,
        default=None,
        help=(
            "[wrap-sweep] LeRobot-space wrap period in radians (e.g. 6.2832 for 2π). "
            "Required for wrap-sweep mode; obtain from a prior --mode discontinuity run."
        ),
    )
    args = parser.parse_args()

    if args.mode == "single":
        _run_single(args)
    elif args.mode == "sweep":
        _run_sweep(args)
    elif args.mode == "discontinuity":
        _run_discontinuity(args)
    elif args.mode == "wrap-sweep":
        _run_wrap_sweep(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
