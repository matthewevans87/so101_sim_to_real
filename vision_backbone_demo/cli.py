#!/usr/bin/env python3
"""CLI entry-point for the vision backbone demo tool.

All configuration values are explicit — no silent defaults.  Any required
argument that is absent raises ``ValueError`` immediately at startup.

Usage::

    python -m vision_backbone_demo \\
        --input PATH \\
        --output PATH \\
        --backbone {frozen_resnet18,frozen_cnn} \\
        --device {cuda,cpu} \\
        --max-channels INT \\
        --seed INT \\
        [--fps FLOAT]                   # required when --input is a video \\
        [--cnn-checkpoint PATH]         # required when --backbone frozen_cnn \\
        [--backbone-cfg PATH]           # required when --backbone frozen_cnn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a module from the project root without installing.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vision_backbone_demo",
        description="Run frames through a frozen vision backbone and write "
        "a composite visualisation (raw | pipelined | heatmap | keypoints | "
        "conv-layer grids) to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Path to an image (PNG/JPG/…) or video (MP4/AVI/…).",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path: .png for a single frame, .mp4 for video.",
    )
    p.add_argument(
        "--backbone",
        required=True,
        choices=["frozen_resnet18", "frozen_cnn"],
        help="Vision backbone to use.",
    )
    p.add_argument(
        "--device", required=True, choices=["cuda", "cpu"], help="Torch device."
    )
    p.add_argument(
        "--max-channels",
        required=True,
        type=int,
        metavar="INT",
        help="Maximum number of feature-map channels to show in conv grids.",
    )
    p.add_argument(
        "--seed",
        required=True,
        type=int,
        metavar="INT",
        help="RNG seed for reproducibility.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Output frames-per-second; required when --input is a video.",
    )
    p.add_argument(
        "--cnn-checkpoint",
        default=None,
        metavar="PATH",
        help="Path to MultiTaskCnn checkpoint; required when " "--backbone frozen_cnn.",
    )
    p.add_argument(
        "--backbone-cfg",
        default=None,
        metavar="PATH",
        help="YAML file with backbone architecture "
        "(channels, kernel_sizes, strides, mlp_hidden_dims, "
        "output_dim, image_height, image_width); required when "
        "--backbone frozen_cnn.",
    )
    return p


def _validate(args: argparse.Namespace) -> None:
    """Raise ValueError with a descriptive message for any missing combination."""
    inp = Path(args.input)
    if not inp.exists():
        raise ValueError(f"--input path does not exist: {inp}")

    is_video = inp.suffix.lower() in _VIDEO_EXTENSIONS
    is_image = inp.suffix.lower() in _IMAGE_EXTENSIONS
    if not is_video and not is_image:
        raise ValueError(
            f"--input extension {inp.suffix!r} is not recognised.  "
            f"Supported video: {sorted(_VIDEO_EXTENSIONS)}.  "
            f"Supported image: {sorted(_IMAGE_EXTENSIONS)}."
        )

    if is_video and args.fps is None:
        raise ValueError("--fps is required when --input is a video file.")

    if args.fps is not None and args.fps <= 0.0:
        raise ValueError(f"--fps must be positive; got {args.fps}.")

    if args.max_channels < 1:
        raise ValueError(f"--max-channels must be >= 1; got {args.max_channels}.")

    if args.backbone == "frozen_cnn":
        if args.cnn_checkpoint is None:
            raise ValueError("--cnn-checkpoint is required when --backbone frozen_cnn.")
        if args.backbone_cfg is None:
            raise ValueError("--backbone-cfg is required when --backbone frozen_cnn.")
        if not Path(args.cnn_checkpoint).exists():
            raise ValueError(
                f"--cnn-checkpoint path does not exist: {args.cnn_checkpoint}"
            )
        if not Path(args.backbone_cfg).exists():
            raise ValueError(f"--backbone-cfg path does not exist: {args.backbone_cfg}")

    out = Path(args.output)
    if is_image:
        if out.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise ValueError(
                f"--output for an image input should be an image path "
                f"(e.g. .png); got {out.suffix!r}."
            )
    if is_video:
        if out.suffix.lower() not in _VIDEO_EXTENSIONS:
            raise ValueError(
                f"--output for a video input should be a video path "
                f"(e.g. .mp4); got {out.suffix!r}."
            )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate(args)

    from .runner import run

    run(args)


if __name__ == "__main__":
    main()
