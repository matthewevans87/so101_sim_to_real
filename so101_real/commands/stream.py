"""stream — read joints at ~30 Hz and publish to /so101/joint_states (ROS2)."""

from __future__ import annotations

import time


def cmd_stream(args) -> None:
    from ..robot import RobotConfig, So101Robot
    from ..ros_publisher import RosPublisher

    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    config = RobotConfig.load(args.robot_config)
    print(f"[stream] Connecting on {config.port}...")
    robot = So101Robot(config=config, joint_names=joint_names)
    robot.connect()

    if args.no_torque:
        robot._follower.bus.disable_torque()
        print("[stream] Torque DISABLED — move the arm freely.")

    print("[stream] Starting ROS2 publisher on /so101/joint_states...")
    publisher = RosPublisher(joint_names=joint_names)

    dt = 1.0 / args.hz
    print(f"[stream] Publishing at {args.hz} Hz. Press Ctrl-C to stop.")
    try:
        while True:
            q = robot.read_joints()
            publisher.publish(q)
            parts = ", ".join(
                f"{name}={float(q[i]):.3f}" for i, name in enumerate(joint_names)
            )
            print(f"\r  {parts}", end="", flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        if args.no_torque:
            robot._follower.bus.enable_torque()
            print("[stream] Torque RE-ENABLED.")
        publisher.destroy()
        robot.disconnect()


def add_parser(sub) -> None:
    p = sub.add_parser(
        "stream",
        help="Read real robot joints and publish to ROS2 (for digital twin)",
    )
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.add_argument(
        "--hz", type=float, default=30.0, help="Publishing rate in Hz (default: 30)"
    )
    p.add_argument(
        "--no-torque",
        action="store_true",
        dest="no_torque",
        help="Disable motor torque so you can move the arm freely while streaming",
    )
    p.set_defaults(func=cmd_stream)
