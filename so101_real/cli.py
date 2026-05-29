"""cli.py — Command-line entry point for so101_real.

Subcommands (``python -m so101_real <cmd> --help`` for details):

    run                 Run a trained policy from a deploy bundle on the physical robot.
    run-static          Hold the arm at a fixed joint target (no bundle, no camera).
    camera-test         Display the live camera feed.
    calibrate-camera    Capture checkerboard frames / solve camera intrinsics.
    compare-views       Composite a real frame and a sim render for comparison.
    stream              Read joints and publish to /so101/joint_states (ROS2).
    robot-test          Print live joint positions.
    probe               Single-shot ramp with full canonical→lerobot conversion trace.

Joint-calibration utilities live in a separate module::

    python -m so101_real.joint_calibrate --help
"""

from __future__ import annotations

import argparse

from .commands import (
    calibrate_camera,
    camera_test,
    compare_views,
    configure_camera,
    probe,
    robot_test,
    run,
    run_static,
    stream,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m so101_real",
        description="SO-101 real-robot inference (no Isaac Lab required).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run.add_parser(sub)
    run_static.add_parser(sub)
    camera_test.add_parser(sub)
    calibrate_camera.add_parser(sub)
    configure_camera.add_parser(sub)
    compare_views.add_parser(sub)
    stream.add_parser(sub)
    robot_test.add_parser(sub)
    probe.add_parser(sub)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
