"""run — full inference loop: camera + robot + bundle-driven policy."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional


def cmd_run(args) -> None:
    import torch

    from ..bundle import load_bundle
    from ..camera import CameraConfig, CameraSource
    from ..controller import ControllerConfig, InferenceLoop, VisionJointObsBuilder
    from ..overlay import OverlayRenderer
    from ..policy import load_policy
    from ..recorder import EpisodeRecorder
    from ..robot import RobotConfig, So101Robot
    from ..ros_publisher import RosPublisher

    print("[so101_real] Loading bundle...")
    bundle = load_bundle(args.bundle)

    print("[so101_real] Loading robot config...")
    robot_config = RobotConfig.load(args.robot_config)
    ctrl_config = ControllerConfig.load(args.robot_config)
    camera_config = CameraConfig.load(args.robot_config)

    runtime_joints, runtime_lower, runtime_upper = robot_config.joint_bounds()
    _check_bundle_robot_consistency(
        bundle=bundle,
        ctrl_config=ctrl_config,
        runtime_joints=runtime_joints,
        runtime_lower=runtime_lower,
        runtime_upper=runtime_upper,
    )

    if args.episodes is None:
        raise ValueError(
            "--episodes is required. "
            "Set the number of episodes to run (e.g. --episodes 5)."
        )

    print(
        f"[so101_real] Bundle: {bundle.bundle_dir.name}\n"
        f"             Encoder: {bundle.encoder_type}\n"
        f"             Policy: obs={bundle.obs_dim} act={bundle.act_dim}\n"
        f"             Control: {ctrl_config.control_hz:.1f} Hz\n"
        f"             Joints: {bundle.active_joints}"
    )

    if args.dry_run:
        print(
            "[so101_real] DRY RUN — bundle and config validated.\n"
            "             Camera and robot will NOT be opened.\n"
            f"             Would run {args.episodes} episode(s) at "
            f"{ctrl_config.control_hz:.1f} Hz on device: {ctrl_config.device}"
        )
        return

    recorder: Optional[EpisodeRecorder] = None
    if args.record:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        record_dir = (
            Path(args.record_dir) / f"rollout_{timestamp}"
            if hasattr(args, "record_dir") and args.record_dir
            else bundle.bundle_dir / "rollouts" / f"rollout_{timestamp}"
        )
        bundle_hash: Optional[str] = None
        prov_path = bundle.bundle_dir / "bundle_provenance.json"
        if prov_path.is_file():
            with open(prov_path) as f:
                prov = json.load(f)
            bundle_hash = prov.get("checkpoint_sha256")
        recorder = EpisodeRecorder(
            output_dir=record_dir,
            bundle_hash=bundle_hash,
            fps=ctrl_config.control_hz,
        )

    overlay: Optional[OverlayRenderer] = None
    if args.overlay:
        overlay = OverlayRenderer(joint_names=bundle.active_joints)

    ros_publisher = None
    if getattr(args, "ros", False):
        ros_publisher = RosPublisher(joint_names=bundle.active_joints)

    camera = CameraSource(camera_config)
    robot = So101Robot(config=robot_config, joint_names=bundle.active_joints)

    try:
        print("[so101_real] Opening camera...")
        camera.open()
        print("[so101_real] Connecting to robot...")
        robot.connect(dry_run=args.dry_run)

        device = torch.device(ctrl_config.device)
        obs_builder = VisionJointObsBuilder(bundle=bundle, camera=camera, device=device)
        policy = load_policy(bundle, device=device)

        loop = InferenceLoop(
            robot=robot,
            robot_config=robot_config,
            ctrl_config=ctrl_config,
            policy=policy,
            obs_builder=obs_builder,
            joint_lower_rad=runtime_lower,
            joint_upper_rad=runtime_upper,
            recorder=recorder,
            overlay=overlay,
            ros_publisher=ros_publisher,
            dry_run=args.dry_run,
        )
        seed = args.seed if hasattr(args, "seed") else None
        try:
            loop.run(episodes=args.episodes, seed=seed)
        except KeyboardInterrupt:
            print("\n[so101_real] Ctrl-C received — resetting to start pose...")
            loop.reset_to_start_pose()
            print("[so101_real] Released.")

    finally:
        loop.destroy() if "loop" in dir() else None
        camera.close()
        robot.disconnect()
        if recorder is not None:
            recorder.close()
            recorder.write_manifest(
                bundle_dir=bundle.bundle_dir,
                robot_config_path=Path(args.robot_config).resolve(),
            )
            print(f"[so101_real] Episode data saved → {recorder._output_dir}")
        if overlay is not None:
            overlay.close()


def _check_bundle_robot_consistency(
    *,
    bundle,
    ctrl_config,
    runtime_joints: list[str],
    runtime_lower: list[float],
    runtime_upper: list[float],
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    mismatches: list[str] = []

    if list(bundle.active_joints) != list(runtime_joints):
        mismatches.append(
            f"  active_joints:\n"
            f"    bundle    = {list(bundle.active_joints)}\n"
            f"    robot.yaml= {list(runtime_joints)}"
        )

    def _close(a: list[float], b: list[float]) -> bool:
        if len(a) != len(b):
            return False
        return all(math.isclose(x, y, rel_tol=rtol, abs_tol=atol) for x, y in zip(a, b))

    if not _close(list(bundle.joint_lower_rad), runtime_lower):
        mismatches.append(
            f"  joint_lower_rad:\n"
            f"    bundle    = {list(bundle.joint_lower_rad)}\n"
            f"    robot.yaml= {runtime_lower}"
        )
    if not _close(list(bundle.joint_upper_rad), runtime_upper):
        mismatches.append(
            f"  joint_upper_rad:\n"
            f"    bundle    = {list(bundle.joint_upper_rad)}\n"
            f"    robot.yaml= {runtime_upper}"
        )
    if not math.isclose(
        float(bundle.control_hz),
        float(ctrl_config.control_hz),
        rel_tol=rtol,
        abs_tol=atol,
    ):
        mismatches.append(
            f"  control_hz:\n"
            f"    bundle    = {bundle.control_hz}\n"
            f"    robot.yaml= {ctrl_config.control_hz}"
        )

    if mismatches:
        raise ValueError(
            "Bundle / robot.yaml mismatch — refusing to run.  The policy was "
            "trained against different robot facts than what robot.yaml "
            "currently declares.  Disagreements:\n"
            + "\n".join(mismatches)
            + "\nUpdate robot.yaml to match the bundle, or retrain with the "
            "current robot.yaml."
        )


def add_parser(sub) -> None:
    import argparse

    p = sub.add_parser("run", help="Run policy inference on the physical robot")
    p.add_argument(
        "--bundle",
        required=True,
        metavar="PATH",
        help="Deploy bundle directory (from export step)",
    )
    p.add_argument(
        "--robot-config",
        required=True,
        dest="robot_config",
        metavar="PATH",
        help="Robot config YAML",
    )
    p.add_argument(
        "--episodes",
        type=int,
        required=True,
        metavar="N",
        help="Number of episodes to run",
    )
    p.add_argument("--seed", type=int, metavar="SEED")
    p.add_argument("--overlay", action="store_true", help="Show live OpenCV overlay")
    p.add_argument(
        "--record",
        action="store_true",
        help="Record episodes to NPZ files alongside the bundle",
    )
    p.add_argument(
        "--record-dir",
        metavar="PATH",
        dest="record_dir",
        help="Directory for recorded rollout data (default: <bundle>/rollouts/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Validate everything without moving the robot",
    )
    p.add_argument(
        "--ros",
        action="store_true",
        dest="ros",
        help="Publish measured joint states to /so101/joint_states (ROS2)",
    )
    p.set_defaults(func=cmd_run)
