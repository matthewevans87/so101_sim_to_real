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

Joint limits are read from robot.yaml::joint_limits, or from a bundle's
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
            scale = existing.get("scale", 1.0)
            offset_deg = sim_d - scale * lero_deg
            suggested_cal[name] = {
                "scale": round(scale, 6),
                "offset_rad": round(math.radians(offset_deg), 6),
            }

    print()
    if suggested_cal:
        print("Suggested joint_calibration (paste into robot.yaml):\n")
        print("joint_calibration:")
        for name, entry in suggested_cal.items():
            print(f"  {name}:")
            print(f"    scale: {entry['scale']}")
            print(f"    offset_rad: {entry['offset_rad']}")
        print()
        print(
            "NOTE: Scale=1.0 is assumed (single measurement point).\n"
            "Use --mode sweep to derive scale empirically via two-stop calibration."
        )
    else:
        print("All joints within 2° tolerance — no additional corrections needed.")
    print()


# ─── Mode: sweep ──────────────────────────────────────────────────────────────


def _load_joint_limits(args: argparse.Namespace) -> dict[str, dict[str, float]]:
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
        print(f"[joint limits] Loaded from {jc_path}")
        return limits

    # Fall back to robot.yaml::joint_limits
    with open(args.robot_config) as f:
        config_data = yaml.safe_load(f)
    raw = config_data.get("joint_limits")
    if not raw:
        raise ValueError(
            "No joint_limits found in robot.yaml and --joint-config not given.\n"
            "Either add a joint_limits section to robot.yaml (see template),\n"
            "or pass --joint-config /path/to/bundle/joint_config.yaml"
        )
    limits: dict[str, dict[str, float]] = {}
    for name, entry in raw.items():
        limits[name] = {
            "lower_rad": float(entry["lower_rad"]),
            "upper_rad": float(entry["upper_rad"]),
        }
    print(f"[joint limits] Loaded from {args.robot_config} :: joint_limits")
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
        joint_bounds = _load_joint_limits(args)
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
        lim = joint_bounds.get(name)
        if not lim:
            print(f"{name:15s}  (no limits — will skip)")
            continue
        scale_sign = existing_cal.get(name, {}).get("scale", 1.0)
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

            lim = joint_bounds.get(joint)
            if lim is None:
                print(f"\n[SKIP] No sim limits for {joint}.")
                continue

            sim_lower_rad = lim["lower_rad"]
            sim_upper_rad = lim["upper_rad"]
            sim_lower_deg = math.degrees(sim_lower_rad)
            sim_upper_deg = math.degrees(sim_upper_rad)

            # Determine expected lerobot direction from existing calibration
            existing_scale = existing_cal.get(joint, {}).get("scale", 1.0)
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
                "offset_rad", 0.0
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
                "scale": round(scale, 6),
                "offset_rad": round(offset_rad, 6),
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
        print(f"    scale: {cal['scale']}")
        print(f"    offset_rad: {cal['offset_rad']}")
    print()


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
        choices=["single", "sweep"],
        default="single",
        help=(
            "'single': one-pose offset measurement (default); "
            "'sweep': per-joint two-stop calibration that derives scale and offset."
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
            "If omitted, limits are read from robot.yaml::joint_limits."
        ),
    )
    parser.add_argument(
        "--joints",
        default=None,
        metavar="NAME[,NAME...]",
        help="[sweep] Comma-separated subset of joints. Omit to calibrate all.",
    )
    args = parser.parse_args()

    if args.mode == "single":
        _run_single(args)
    elif args.mode == "sweep":
        _run_sweep(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
