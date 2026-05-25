"""cli.py — Command-line entry point for so101_real.

Usage:
    python -m so101_real run --bundle <path> --robot-config <path> [options]
    python -m so101_real camera-test --robot-config <path>
    python -m so101_real robot-test --robot-config <path>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


def cmd_run(args) -> None:
    """Full inference loop: camera + robot + policy."""
    from .bundle import load_bundle
    from .camera import CameraConfig, CameraSource
    from .controller import ControllerConfig, InferenceLoop
    from .overlay import OverlayRenderer
    from .recorder import EpisodeRecorder
    from .robot import RobotConfig, So101Robot
    from .ros_publisher import RosPublisher

    # ── Validate all inputs at startup ────────────────────────────────────────
    print("[so101_real] Loading bundle...")
    bundle = load_bundle(args.bundle)

    print("[so101_real] Loading robot config...")
    robot_config = RobotConfig.load(args.robot_config)
    ctrl_config = ControllerConfig.load(args.robot_config)
    camera_config = CameraConfig.load(args.robot_config)

    if args.episodes is None:
        raise ValueError(
            "--episodes is required. "
            "Set the number of episodes to run (e.g. --episodes 5)."
        )

    print(
        f"[so101_real] Bundle: {bundle.bundle_dir.name}\n"
        f"             Encoder: {bundle.encoder_type}\n"
        f"             Policy: obs={bundle.obs_dim} act={bundle.act_dim}\n"
        f"             Control: {bundle.control_hz:.1f} Hz\n"
        f"             Joints: {bundle.active_joints}"
    )

    if args.dry_run:
        print(
            "[so101_real] DRY RUN — bundle and config validated.\n"
            "             Camera and robot will NOT be opened.\n"
            f"             Would run {args.episodes} episode(s) at "
            f"{bundle.control_hz:.1f} Hz on device: {ctrl_config.device}"
        )
        return

    # ── Optional recorder ─────────────────────────────────────────────────────
    recorder: Optional[EpisodeRecorder] = None
    if args.record:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        record_dir = (
            Path(args.record_dir) / f"rollout_{timestamp}"
            if hasattr(args, "record_dir") and args.record_dir
            else bundle.bundle_dir / f"rollouts" / f"rollout_{timestamp}"
        )

        # Read bundle hash from provenance if available
        bundle_hash: Optional[str] = None
        prov_path = bundle.bundle_dir / "bundle_provenance.json"
        if prov_path.is_file():
            with open(prov_path) as f:
                prov = json.load(f)
            bundle_hash = prov.get("checkpoint_sha256")

        recorder = EpisodeRecorder(
            output_dir=record_dir,
            bundle_hash=bundle_hash,
            fps=bundle.control_hz,
        )

    # ── Optional overlay ──────────────────────────────────────────────────────
    overlay: Optional[OverlayRenderer] = None
    if args.overlay:
        overlay = OverlayRenderer(joint_names=bundle.active_joints)
    # ── Optional ROS2 digital-twin publisher ───────────────────────────────────
    ros_publisher: Optional[RosPublisher] = None
    if getattr(args, "ros", False):
        ros_publisher = RosPublisher(joint_names=bundle.active_joints)
    # ── Connect camera and robot ──────────────────────────────────────────────
    camera = CameraSource(camera_config)
    robot = So101Robot(config=robot_config, joint_names=bundle.active_joints)

    try:
        print("[so101_real] Opening camera...")
        camera.open()

        print("[so101_real] Connecting to robot...")
        robot.connect(dry_run=args.dry_run)

        loop = InferenceLoop(
            bundle=bundle,
            camera=camera,
            robot=robot,
            robot_config=robot_config,
            ctrl_config=ctrl_config,
            recorder=recorder,
            overlay=overlay,
            ros_publisher=ros_publisher,
            dry_run=args.dry_run,
        )

        seed = args.seed if hasattr(args, "seed") else None
        loop.run(episodes=args.episodes, seed=seed)

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


def cmd_camera_test(args) -> None:
    """Open the camera and display a live feed until Ctrl-C."""
    from .camera import CameraConfig, CameraSource

    config = CameraConfig.load(args.robot_config)
    print(
        f"[camera-test] Opening camera {config.device_index} "
        f"({config.capture_width}x{config.capture_height})..."
    )

    import cv2

    with CameraSource(config) as cam:
        print("[camera-test] Press Ctrl-C to stop.")
        try:
            while True:
                frame = cam.get_frame()
                cv2.imshow("camera-test", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()


def cmd_calibrate_camera(args) -> None:
    """Capture checkerboard frames or solve camera intrinsics."""
    from .camera import CameraConfig
    from .calibrate import run_capture, run_solve

    out_dir = Path(args.out_dir).expanduser().resolve()
    board_cols: int = args.board_cols
    board_rows: int = args.board_rows
    square_mm: float = args.square_mm

    _REPO_ROOT = Path(__file__).resolve().parents[1]
    intrinsics_path = (
        Path(
            args.intrinsics_out
            if args.intrinsics_out
            else _REPO_ROOT / "so101_real" / "configs" / "camera_intrinsics.yaml"
        )
        .expanduser()
        .resolve()
    )

    if args.solve:
        run_solve(
            out_dir=out_dir,
            board_cols=board_cols,
            board_rows=board_rows,
            square_mm=square_mm,
            intrinsics_path=intrinsics_path,
        )
    else:
        # Resolve camera settings: CLI flags take priority, then robot config, then defaults.
        if args.robot_config:
            config = CameraConfig.load(args.robot_config)
            device_index = config.device_index
            capture_width = config.capture_width
            capture_height = config.capture_height
        else:
            device_index = 0
            capture_width = 1920
            capture_height = 1080
        # Explicit CLI flags always win.
        if args.device_index is not None:
            device_index = args.device_index
        if args.capture_width is not None:
            capture_width = args.capture_width
        if args.capture_height is not None:
            capture_height = args.capture_height
        run_capture(
            device_index=device_index,
            capture_width=capture_width,
            capture_height=capture_height,
            out_dir=out_dir,
            board_cols=board_cols,
            board_rows=board_rows,
        )


def cmd_compare_views(args) -> None:
    """Side-by-side + blend overlay comparison of a real camera frame and a sim render."""
    import numpy as np

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python is required. Run: pip install opencv-python")
        sys.exit(1)

    real_path = Path(args.real)
    sim_path = Path(args.sim)
    if not real_path.exists():
        print(f"ERROR: real image not found: {real_path}")
        sys.exit(1)
    if not sim_path.exists():
        print(f"ERROR: sim image not found: {sim_path}")
        sys.exit(1)

    real = cv2.imread(str(real_path))
    sim = cv2.imread(str(sim_path))
    if real is None:
        print(f"ERROR: could not read real image: {real_path}")
        sys.exit(1)
    if sim is None:
        print(f"ERROR: could not read sim image: {sim_path}")
        sys.exit(1)

    # Resize sim to match real resolution (or vice-versa)
    if args.match_size == "real":
        sim = cv2.resize(
            sim, (real.shape[1], real.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    elif args.match_size == "sim":
        real = cv2.resize(
            real, (sim.shape[1], sim.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    else:
        # Match to explicit WxH
        w, h = map(int, args.match_size.split("x"))
        real = cv2.resize(real, (w, h), interpolation=cv2.INTER_LINEAR)
        sim = cv2.resize(sim, (w, h), interpolation=cv2.INTER_LINEAR)

    H, W = real.shape[:2]

    # --- Panel 1: real ---
    # --- Panel 2: sim ---
    # --- Panel 3: blend overlay ---
    alpha = max(0.0, min(1.0, args.alpha))
    blend = cv2.addWeighted(real, 1.0 - alpha, sim, alpha, 0.0)

    # --- Panel 4: checkerboard interleave (hard cut every N rows/cols) ---
    checker = real.copy()
    n = args.checker_size
    for row in range(0, H, 2 * n):
        for col in range(0, W, 2 * n):
            checker[row : row + n, col : col + n] = sim[row : row + n, col : col + n]
            r2 = min(row + 2 * n, H)
            c2 = min(col + 2 * n, W)
            checker[row + n : r2, col + n : c2] = sim[row + n : r2, col + n : c2]

    def _label(img, text):
        out = img.copy()
        cv2.putText(
            out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            out,
            text,
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    panels = [
        _label(real, "real"),
        _label(sim, "sim"),
        _label(blend, f"blend ({int((1-alpha)*100)}% real)"),
        _label(checker, f"checker ({n}px)"),
    ]

    top_row = np.concatenate(panels[:2], axis=1)
    bot_row = np.concatenate(panels[2:], axis=1)
    composite = np.concatenate([top_row, bot_row], axis=0)

    out_path = (
        Path(args.output)
        if args.output
        else real_path.with_stem(real_path.stem + "_compare").with_suffix(".jpg")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved: {out_path}  ({composite.shape[1]}x{composite.shape[0]})")

    if args.show:
        # Scale down to fit within the requested max display width
        max_w = args.display_width
        disp = composite
        ch, cw = composite.shape[:2]
        if cw > max_w:
            scale = max_w / cw
            disp = cv2.resize(
                composite, (max_w, int(ch * scale)), interpolation=cv2.INTER_AREA
            )
        cv2.imshow("compare-views", disp)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_robot_test(args) -> None:
    """Connect to the robot and print joint positions until Ctrl-C."""
    from .robot import RobotConfig, So101Robot

    config = RobotConfig.load(args.robot_config)
    # Joint names can be read from a bundle or defaulted to standard SO-101 order
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    if args.bundle:
        from .bundle import load_bundle

        bundle = load_bundle(args.bundle)
        joint_names = bundle.active_joints

    print(f"[robot-test] Connecting on {config.port}...")
    robot = So101Robot(config=config, joint_names=joint_names)
    robot.connect()

    try:
        print("[robot-test] Press Ctrl-C to stop.")
        while True:
            q = robot.read_joints()
            parts = ", ".join(
                f"{name}={float(q[i]):.3f}" for i, name in enumerate(joint_names)
            )
            print(f"\r  {parts}", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        robot.disconnect()


def cmd_stream(args) -> None:
    """Connect to the robot, read joints at ~30 Hz, and publish to /so101/joint_states.

    Use this alongside `./scripts/run.py digital-twin` to mirror the real arm
    in Isaac Sim without running a policy.
    """
    from .robot import RobotConfig, So101Robot
    from .ros_publisher import RosPublisher

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m so101_real",
        description="SO-101 real-robot inference (no Isaac Lab required).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ───────────────────────────────────────────────────────────────────
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
        help="Publish measured joint states to /so101/joint_states (ROS2) for the digital twin",
    )
    p.set_defaults(func=cmd_run)

    # ── camera-test ───────────────────────────────────────────────────────────
    p = sub.add_parser("camera-test", help="Display live camera feed")
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.set_defaults(func=cmd_camera_test)

    # ── calibrate-camera ──────────────────────────────────────────────────────
    p = sub.add_parser(
        "calibrate-camera",
        help="Capture checkerboard frames and/or solve camera intrinsics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Step 1 (capture):  python -m so101_real calibrate-camera "
            "--robot-config <cfg> --out-dir so101_real/calibration/captures\n"
            "Step 2 (solve):    python -m so101_real calibrate-camera --solve "
            "--out-dir so101_real/calibration/captures --square-mm 25"
        ),
    )
    p.add_argument(
        "--robot-config",
        dest="robot_config",
        metavar="PATH",
        help="Robot config YAML (optional; overridden by --device/--width/--height)",
    )
    p.add_argument(
        "--device",
        dest="device_index",
        type=int,
        default=None,
        metavar="N",
        help="V4L2 camera device index (default: 0, or value from --robot-config)",
    )
    p.add_argument(
        "--width",
        dest="capture_width",
        type=int,
        default=None,
        metavar="PX",
        help="Capture width in pixels (default: 1920, or value from --robot-config)",
    )
    p.add_argument(
        "--height",
        dest="capture_height",
        type=int,
        default=None,
        metavar="PX",
        help="Capture height in pixels (default: 1080, or value from --robot-config)",
    )
    p.add_argument(
        "--out-dir",
        dest="out_dir",
        metavar="PATH",
        default="so101_real/calibration/captures",
        help="Directory for captured frames (default: so101_real/calibration/captures)",
    )
    p.add_argument(
        "--board-cols",
        dest="board_cols",
        type=int,
        default=9,
        metavar="N",
        help="Number of inner corners along the horizontal axis (default: 9)",
    )
    p.add_argument(
        "--board-rows",
        dest="board_rows",
        type=int,
        default=6,
        metavar="N",
        help="Number of inner corners along the vertical axis (default: 6)",
    )
    p.add_argument(
        "--square-mm",
        dest="square_mm",
        type=float,
        default=25.0,
        metavar="MM",
        help="Side length of one checkerboard square in mm (default: 25.0)",
    )
    p.add_argument(
        "--solve",
        action="store_true",
        help="Solve intrinsics from previously captured frames",
    )
    p.add_argument(
        "--intrinsics-out",
        dest="intrinsics_out",
        metavar="PATH",
        default=None,
        help="Output path for camera_intrinsics.yaml "
        "(default: so101_real/configs/camera_intrinsics.yaml)",
    )
    p.set_defaults(func=cmd_calibrate_camera)

    # ── compare-views ─────────────────────────────────────────────────────────
    p = sub.add_parser(
        "compare-views",
        help="Side-by-side + blend overlay comparison of a real frame and a sim render",
    )
    p.add_argument("--real", required=True, metavar="PATH", help="Real camera image")
    p.add_argument("--sim", required=True, metavar="PATH", help="Sim render image")
    p.add_argument(
        "--output",
        "-o",
        default=None,
        metavar="PATH",
        help="Output composite image path (default: <real>_compare.jpg)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Blend weight for sim in overlay panel (0=all-real, 1=all-sim, default 0.5)",
    )
    p.add_argument(
        "--checker-size",
        type=int,
        default=64,
        dest="checker_size",
        metavar="N",
        help="Checkerboard tile size in pixels for the interleave panel (default 64)",
    )
    p.add_argument(
        "--match-size",
        default="real",
        dest="match_size",
        metavar="SPEC",
        help="Resize both images to match: 'real', 'sim', or 'WxH' (default: real)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the composite in a window (requires a display)",
    )
    p.add_argument(
        "--display-width",
        type=int,
        default=1920,
        dest="display_width",
        metavar="PX",
        help="Max pixel width of the displayed window (default: 1920). Saved file is always full resolution.",
    )
    p.set_defaults(func=cmd_compare_views)

    # ── stream ────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "stream",
        help="Read real robot joints and publish to ROS2 (for digital twin)",
    )
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.add_argument(
        "--hz",
        type=float,
        default=30.0,
        help="Publishing rate in Hz (default: 30)",
    )
    p.add_argument(
        "--no-torque",
        action="store_true",
        dest="no_torque",
        help="Disable motor torque so you can move the arm freely while streaming",
    )
    p.set_defaults(func=cmd_stream)

    # ── robot-test ────────────────────────────────────────────────────────────
    p = sub.add_parser("robot-test", help="Print live joint positions")
    p.add_argument("--robot-config", required=True, dest="robot_config", metavar="PATH")
    p.add_argument(
        "--bundle",
        metavar="PATH",
        help="Optional: read joint names from bundle manifest",
    )
    p.set_defaults(func=cmd_robot_test)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
