"""calibrate-camera — capture checkerboard frames or solve camera intrinsics."""

from __future__ import annotations

from pathlib import Path


def cmd_calibrate_camera(args) -> None:
    from ..camera import CameraConfig
    from ..calibrate import run_capture, run_solve

    out_dir = Path(args.out_dir).expanduser().resolve()
    board_cols: int = args.board_cols
    board_rows: int = args.board_rows
    square_mm: float = args.square_mm

    _REPO_ROOT = Path(__file__).resolve().parents[2]
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
        return

    if args.robot_config:
        config = CameraConfig.load(args.robot_config)
        device_index = config.device_index
        capture_width = config.capture_width
        capture_height = config.capture_height
    else:
        device_index = 0
        capture_width = 1920
        capture_height = 1080

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


def add_parser(sub) -> None:
    p = sub.add_parser(
        "calibrate-camera",
        help="Capture checkerboard frames and/or solve camera intrinsics",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
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
        help="V4L2 camera device index (default: 0)",
    )
    p.add_argument(
        "--width",
        dest="capture_width",
        type=int,
        default=None,
        metavar="PX",
        help="Capture width in pixels (default: 1920)",
    )
    p.add_argument(
        "--height",
        dest="capture_height",
        type=int,
        default=None,
        metavar="PX",
        help="Capture height in pixels (default: 1080)",
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
        help="Inner corners along horizontal axis (default: 9)",
    )
    p.add_argument(
        "--board-rows",
        dest="board_rows",
        type=int,
        default=6,
        metavar="N",
        help="Inner corners along vertical axis (default: 6)",
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
        help="Output path for camera_intrinsics.yaml",
    )
    p.set_defaults(func=cmd_calibrate_camera)
