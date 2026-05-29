"""robot-test — print live joint positions until Ctrl-C."""

from __future__ import annotations

import time


def cmd_robot_test(args) -> None:
    import torch
    from ..robot import RobotConfig, So101Robot
    from ..units import JointUnitConverter

    config = RobotConfig.load(args.robot_config)
    joint_names, lower_list, upper_list = config.joint_bounds()
    converter = JointUnitConverter(
        joint_names=joint_names, lower_rad=lower_list, upper_rad=upper_list
    )

    unit = args.unit
    if unit not in ("rad", "deg", "norm"):
        raise ValueError(f"unknown --unit {unit!r}")

    print(f"[robot-test] Connecting on {config.port}...")
    robot = So101Robot(config=config, joint_names=joint_names)
    robot.connect()

    if args.no_torque:
        robot._follower.bus.disable_torque()
        print("[robot-test] Torque DISABLED — move the arm freely.")

    try:
        print(f"[robot-test] Press Ctrl-C to stop.  Units: {unit}.")
        while True:
            q = robot.read_joints()
            q_disp = converter.from_canonical_rad(q, unit)
            assert isinstance(q_disp, torch.Tensor)
            parts = [
                f"{name}={float(q_disp[i]):+.3f}" for i, name in enumerate(joint_names)
            ]
            print(f"\r  {', '.join(parts)}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        robot.disconnect()


def add_parser(sub) -> None:
    p = sub.add_parser(
        "robot-test",
        help="Print live joint positions (joint names from robot.yaml::joint_limits).",
    )
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.add_argument(
        "--unit",
        choices=["rad", "deg", "norm"],
        default="rad",
        help="Display unit: rad (default), deg, or norm [-1, 1].",
    )
    p.add_argument(
        "--no-torque",
        action="store_true",
        dest="no_torque",
        help="Disable servo torque so you can hand-move the arm during readout.",
    )
    p.set_defaults(func=cmd_robot_test)
