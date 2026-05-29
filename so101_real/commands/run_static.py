"""run-static — hold the arm at a fixed joint target via a static policy."""

from __future__ import annotations

import time
from typing import Optional


def cmd_run_static(args) -> None:
    import math
    import torch

    from ..controller import ControllerConfig, InferenceLoop, NullObsBuilder
    from ..policies import StaticPositionPolicy
    from ..robot import RobotConfig, So101Robot
    from ..ros_publisher import RosPublisher
    from ..units import JointUnitConverter, JointParser

    raw_arg = args.joints.strip().strip("\"'")
    try:
        raw_values = [float(x) for x in raw_arg.split(",")]
    except ValueError as exc:
        raise ValueError(
            f"--joints must be a comma-separated list of floats; "
            f"got {args.joints!r}.  Tip: for negative values use "
            f"`--joints=-1,0,...` (with '=') to prevent argparse treating '-1,...' as a flag."
        ) from exc

    robot_config = RobotConfig.load(args.robot_config)
    joint_names, lower_list, upper_list = robot_config.joint_bounds()
    converter = JointUnitConverter(
        joint_names=joint_names, lower_rad=lower_list, upper_rad=upper_list
    )

    n = len(joint_names)
    if len(raw_values) != n:
        raise ValueError(
            f"--joints has {len(raw_values)} values but joint_limits "
            f"defines {n} joints ({joint_names})."
        )

    unit = args.unit
    jp = JointParser(converter)
    target_list: list[float] = [
        jp.parse(v, unit, joint_index=i) for i, v in enumerate(raw_values)
    ]

    ctrl_config = ControllerConfig.load(args.robot_config)
    duration_s = float(args.duration_s)
    if duration_s <= 0.0:
        raise ValueError(f"--duration-s must be > 0; got {duration_s}.")
    max_steps = max(1, round(duration_s * ctrl_config.control_hz))

    print(
        f"[so101_real] run-static target (input unit={unit}): "
        f"{dict(zip(joint_names, raw_values))}\n"
        f"             → canonical radians: "
        f"{ {n: round(q, 5) for n, q in zip(joint_names, target_list)} }\n"
        f"             Control: {ctrl_config.control_hz:.1f} Hz, "
        f"hold {duration_s:.2f}s/episode ({max_steps} ticks) × {args.episodes} ep"
    )

    if args.dry_run:
        print("[so101_real] DRY RUN — robot will NOT be opened.")
        return

    ros_publisher: Optional = None
    if getattr(args, "ros", False):
        ros_publisher = RosPublisher(joint_names=joint_names)

    robot = So101Robot(config=robot_config, joint_names=joint_names)
    try:
        print("[so101_real] Connecting to robot...")
        robot.connect(dry_run=args.dry_run)

        device = torch.device(ctrl_config.device)
        lower_t = torch.tensor(lower_list, dtype=torch.float32)
        upper_t = torch.tensor(upper_list, dtype=torch.float32)
        target_t = torch.tensor(target_list, dtype=torch.float32)
        policy = StaticPositionPolicy(
            target_q_rad=target_t, joint_lower_rad=lower_t, joint_upper_rad=upper_t
        )
        loop = InferenceLoop(
            robot=robot,
            robot_config=robot_config,
            ctrl_config=ctrl_config,
            policy=policy,
            obs_builder=NullObsBuilder(device=device),
            joint_lower_rad=lower_list,
            joint_upper_rad=upper_list,
            ros_publisher=ros_publisher,
            dry_run=args.dry_run,
        )
        loop.run(episodes=args.episodes, max_steps_per_episode=max_steps)

        if not args.no_hold and not args.dry_run:
            tick_period = 1.0 / float(ctrl_config.control_hz)
            print(
                f"[so101_real] Holding target pose at {ctrl_config.control_hz:.1f} Hz.  "
                f"Press Ctrl-C to release torque and exit."
            )
            try:
                while True:
                    t_start = time.monotonic()
                    robot.send_joints(target_t)
                    elapsed = time.monotonic() - t_start
                    if (remaining := tick_period - elapsed) > 0:
                        time.sleep(remaining)
            except KeyboardInterrupt:
                print("\n[so101_real] Ctrl-C received — resetting to start pose...")
                loop.reset_to_start_pose()
                print("[so101_real] Released.")
    finally:
        loop.destroy() if "loop" in dir() else None
        robot.disconnect()


def add_parser(sub) -> None:
    import argparse

    p = sub.add_parser(
        "run-static",
        help="Hold the arm at a fixed joint target (no bundle, no camera, no encoder).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Examples\n"
            "--------\n"
            "Drive to zero-at-home and hold for 8 s::\n\n"
            "    python -m so101_real run-static \\\n"
            "        --robot-config so101_real/configs/robot.yaml \\\n"
            "        --joints '0,0,0,0,0,0' --episodes 1 --duration-s 8\n"
        ),
    )
    p.add_argument(
        "--robot-config",
        required=True,
        dest="robot_config",
        metavar="PATH",
        help="Robot config YAML (must contain joint_limits and controller).",
    )
    p.add_argument(
        "--joints",
        required=True,
        dest="joints",
        metavar='"a,b,c,d,e,f"',
        help="Comma-separated joint targets in the order defined by joint_limits.",
    )
    p.add_argument(
        "--unit",
        choices=["rad", "deg", "norm"],
        default="rad",
        help="Units for --joints values (default: rad).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of episodes (default: 1).",
    )
    p.add_argument(
        "--duration-s",
        type=float,
        default=5.0,
        dest="duration_s",
        metavar="S",
        help="Seconds to hold per episode (default: 5.0).",
    )
    p.add_argument(
        "--ros",
        action="store_true",
        dest="ros",
        help="Publish measured joint states to /so101/joint_states (ROS2).",
    )
    p.add_argument(
        "--no-hold",
        action="store_true",
        dest="no_hold",
        help="Disconnect after timed episode(s) instead of holding until Ctrl-C.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Validate config without opening the robot.",
    )
    p.set_defaults(func=cmd_run_static)
