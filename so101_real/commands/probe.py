"""probe — single-shot diagnostic: send a joint command and trace the full conversion pipeline."""

from __future__ import annotations

import math
import time
from typing import Optional


def cmd_probe(args) -> None:
    import torch
    from ..robot import RobotConfig, So101Robot
    from ..units import from_robot_config, JointParser

    JOINTS = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    robot_config = RobotConfig.load(args.robot_config)
    joint_names = JOINTS
    if args.bundle:
        from ..bundle import load_bundle

        bundle = load_bundle(args.bundle)
        joint_names = bundle.active_joints

    jl_names, lower_rad, upper_rad = robot_config.joint_bounds()
    if jl_names != list(joint_names):
        raise ValueError(
            f"joint_limits joint order {jl_names} does not match "
            f"active joint list {list(joint_names)}.  Re-order one to match."
        )

    converter = from_robot_config(
        joint_names=joint_names,
        lower_rad=lower_rad,
        upper_rad=upper_rad,
        joint_calibration=robot_config.joint_calibration,
    )

    print(f"[probe] Connecting on {robot_config.port}...")
    robot = So101Robot(config=robot_config, joint_names=joint_names)
    robot.connect()

    try:
        print(
            "\n[probe] Calibration table (lero → canonical: q_rad = scale*q_lero_rad + offset_rad):"
        )
        print(f"  {'joint':<14} {'scale':>10} {'offset_rad':>12}")
        for name in joint_names:
            cal = robot_config.joint_calibration.get(name)
            if cal is None:
                print(f"  {name:<14} {'-':>10} {'-':>12}  (identity)")
            else:
                print(f"  {name:<14} {cal.scale:>10.6f} {cal.offset_rad:>12.6f}")

        show_norm = converter.has_joint_limits and args.unit == "norm"

        def _dump_state(label: str) -> torch.Tensor:
            q_rad = robot.read_joints().cpu()
            print(f"\n[probe] {label}")
            if show_norm:
                print(f"  {'joint':<14} {'q_rad':>12} {'q_deg':>12} {'q_norm':>12}")
            else:
                print(f"  {'joint':<14} {'q_rad':>12} {'q_deg':>12}")
            q_norm_vec: Optional[torch.Tensor] = (
                converter.canonical_to_normalized(q_rad) if show_norm else None
            )
            for i, name in enumerate(joint_names):
                q_i = float(q_rad[i])
                row = f"  {name:<14} {q_i:>12.5f} {math.degrees(q_i):>12.4f}"
                if q_norm_vec is not None:
                    row += f" {float(q_norm_vec[i]):>12.5f}"
                print(row)
            return q_rad

        q_now = _dump_state("Current state:")

        if args.joint is None:
            print("\n[probe] No --joint specified; state-only mode.  Exiting.")
            return

        if args.joint not in joint_names:
            raise ValueError(f"--joint {args.joint!r} not in joint list {joint_names}.")
        j = joint_names.index(args.joint)

        v = float(args.value)
        if converter.has_joint_limits:
            jp = JointParser(converter)
            target_j = jp.parse(v, args.unit, joint_index=j)
            lo_j = jp._lower[j]
            hi_j = jp._upper[j]
        else:
            target_j = float(converter.to_canonical_rad(v, args.unit, joint_index=j))
            lo_j = hi_j = None
        if args.unit == "norm" and lo_j is not None and hi_j is not None:
            norm_desc = f"normalized ∈ [-1,1] mapped via robot.yaml joint_limits [{lo_j:.4f}, {hi_j:.4f}] rad"
        else:
            norm_desc = "normalized ∈ [-1,1]"
        unit_desc = {
            "rad": "canonical radians",
            "deg": "canonical degrees",
            "lrad": "lero radians",
            "ldeg": "lero degrees",
            "norm": norm_desc,
        }[args.unit]

        target = q_now.clone()
        target[j] = target_j

        print(
            f"\n[probe] Requested move:\n"
            f"  joint        : {args.joint}\n"
            f"  input value  : {v} ({unit_desc})\n"
            f"  target sim   : {target_j:+.5f} rad ({math.degrees(target_j):+.4f}°)\n"
            f"  start sim    : {float(q_now[j]):+.5f} rad ({math.degrees(float(q_now[j])):+.4f}°)\n"
            f"  Δ sim        : {float(target[j] - q_now[j]):+.5f} rad "
            f"({math.degrees(float(target[j] - q_now[j])):+.4f}°)\n"
        )

        q_lerobot_nominal = converter.canonical_to_lero_rad(target)
        print("[probe] Predicted lerobot command (per-joint, for the *target* pose):")
        print(f"  {'joint':<14} {'lero_rad':>12} {'lero_deg':>12}")
        for i, name in enumerate(joint_names):
            print(
                f"  {name:<14} {float(q_lerobot_nominal[i]):>12.5f} "
                f"{math.degrees(float(q_lerobot_nominal[i])):>12.4f}"
            )

        ramp_s = float(args.ramp_s)
        ramp_hz = float(args.ramp_hz)
        n_steps = max(1, round(ramp_s * ramp_hz))
        dt = 1.0 / ramp_hz
        print(
            f"\n[probe] Sending interpolated trajectory over {ramp_s:.2f}s "
            f"({n_steps} steps @ {ramp_hz:.0f} Hz)..."
        )
        for step in range(n_steps):
            t = (step + 1) / n_steps
            robot.send_joints(q_now + t * (target - q_now))
            time.sleep(dt)
        print("[probe] Trajectory complete.")

        time.sleep(0.2)
        _dump_state("Final state (after settle):")

    finally:
        robot.disconnect()


def add_parser(sub) -> None:
    p = sub.add_parser(
        "probe",
        help="Send a single traceable joint command and dump every step of the sim → lerobot conversion.",
    )
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.add_argument(
        "--bundle",
        metavar="PATH",
        help="Optional bundle path (only used to override the joint-name list).",
    )
    p.add_argument(
        "--joint",
        metavar="NAME",
        help="Joint to command (omit to dump current state and exit).",
    )
    p.add_argument(
        "--value",
        type=float,
        help="Target value in --unit.  Required when --joint is set.",
    )
    p.add_argument(
        "--unit",
        choices=["rad", "deg", "lrad", "ldeg", "norm"],
        default="deg",
        help="Unit for --value (default: deg).",
    )
    p.add_argument(
        "--ramp-s",
        type=float,
        default=2.0,
        dest="ramp_s",
        help="Seconds to linearly ramp from current to target (default: 2.0).",
    )
    p.add_argument(
        "--ramp-hz",
        type=float,
        default=60.0,
        dest="ramp_hz",
        help="Tick rate for the ramp (default: 60).",
    )
    p.set_defaults(func=cmd_probe)
