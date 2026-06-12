"""capture-frame — capture a single still frame from the wrist camera to disk.

Replaces the ``ffmpeg -f v4l2 ...`` one-liner referenced throughout the docs:
opens the camera using the same ``CameraConfig`` as the live deploy loop,
discards warmup frames, then writes a single image. This keeps the
calibration capture path consistent with what the policy actually sees at
runtime (same device index, capture resolution, and FOURCC).
"""

from __future__ import annotations

from pathlib import Path


_DEFAULT_OUT = "so101_real/calibration/captures/live_frame.png"


def cmd_capture_frame(args) -> None:
    import cv2

    from ..camera import CameraConfig, CameraSource

    config = CameraConfig.load(args.robot_config)
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[capture-frame] Opening camera {config.device_index} "
        f"({config.capture_width}x{config.capture_height}, fourcc={config.fourcc})..."
    )
    with CameraSource(config) as cam:
        # Discard a few extra frames on top of warmup_frames to let
        # auto-exposure/AWB settle when the camera was just opened.
        for _ in range(max(0, int(args.discard_frames))):
            cam.get_frame()

        frame = cam.get_frame()
        if frame is None:
            raise RuntimeError("Camera returned no frame; check device index and FOURCC.")

        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            raise RuntimeError(
                f"cv2.imwrite failed for {out_path} — check directory permissions "
                "and that the file extension is supported by OpenCV."
            )

    h, w = frame.shape[:2]
    print(f"[capture-frame] Wrote {out_path}  ({w}x{h})")


def add_parser(sub) -> None:
    p = sub.add_parser(
        "capture-frame",
        help="Capture a single still frame from the wrist camera and write to disk.",
    )
    p.add_argument(
        "--robot-config",
        required=True,
        dest="robot_config",
        metavar="PATH",
        help="Path to robot YAML (uses its 'camera' section).",
    )
    p.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        dest="out",
        metavar="PATH",
        help=f"Output image path (default: {_DEFAULT_OUT}). "
        "Format inferred from extension (.png, .jpg, ...).",
    )
    p.add_argument(
        "--discard-frames",
        type=int,
        default=5,
        dest="discard_frames",
        metavar="N",
        help="Frames to discard after warmup before writing (default: 5; lets "
        "auto-exposure / AWB settle).",
    )
    p.set_defaults(func=cmd_capture_frame)
