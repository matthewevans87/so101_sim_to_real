#!/usr/bin/env python3
"""Interactive visualizer for CNN pretraining data and model predictions.

Displays training examples from a curated (or raw) dataset alongside
predictions from a :class:`~cnn_trainer.model.model.PretrainCnn` model.

Layout
------
::

    ┌──────────────────────────┬──────────────────────────┐
    │                          │                          │
    │   Cube orientation (3D)  │   Camera image (2D)      │
    │   ● GT axes (solid)      │                          │
    │   ● Pred axes (dashed)   │                          │
    │                          │                          │
    └──────────────────────────┴──────────────────────────┘
          [sample info / prediction scores in figure title]

Keyboard controls
-----------------
- ``←`` / ``→``   — step to previous / next sample.
- ``0-9``          — type a sample index; shown live as ``[▶ NNN]``.
- ``Enter``        — jump to the typed index.
- ``Backspace``    — delete last typed digit.
- ``q`` / ``Esc``  — quit.

Usage::

    python -m cnn_trainer.visualizer \\
        --curated-dir /path/to/curated \\
        --split train \\
        --model /path/to/best_model.pt \\
        --config configs/cnn_pretrain.yaml \\
        [--device cpu]

If ``--model`` / ``--config`` are omitted, only ground-truth labels are shown.
A raw telemetry manifest can also be supplied directly via ``--manifest``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – required for 3D projection

# Allow running as a script from the project root without installing packages.
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from so101.curate.dataset import TelemetryDataset
from so101.model.model import MultiTaskCnn

# ── Colour palette ────────────────────────────────────────────────────────────

_GT_COLORS = ("#e74c3c", "#2ecc71", "#3498db")  # X=red, Y=green, Z=blue (GT)
_PRED_COLORS = ("#f1948a", "#82e0aa", "#85c1e9")  # lighter tints (predicted)

# ── Quaternion helpers ────────────────────────────────────────────────────────


def _quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a wxyz unit quaternion to a 3×3 rotation matrix.

    Args:
        quat_wxyz: Array of shape ``(4,)`` with components ``[w, x, y, z]``.

    Returns:
        ``(3, 3)`` rotation matrix whose columns are the rotated X/Y/Z axes.
    """
    try:
        from scipy.spatial.transform import Rotation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scipy is required for 3-D orientation visualisation: pip install scipy"
        ) from exc

    w, x, y, z = quat_wxyz.astype(float)
    # Normalise defensively (training data should already be unit, predictions may not be).
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    # scipy uses xyzw order.
    rot = Rotation.from_quat([x, y, z, w])
    return rot.as_matrix()


# ── Axis drawing helper ───────────────────────────────────────────────────────


def _draw_frame(
    ax: "Axes3D",
    R: np.ndarray,
    colors: tuple[str, str, str],
    alpha: float,
    linewidth: float,
    linestyle: str,
    scale: float = 0.85,
) -> None:
    """Draw three coordinate-axis arrows for a rotation matrix *R*.

    The three columns of *R* give the directions of the X, Y, Z axes of the
    rotated frame.  Each axis is drawn as a line from the origin to
    ``R[:, i] * scale``, with a filled dot at the tip.

    Args:
        ax:        3-D axes to draw on.
        R:         ``(3, 3)`` rotation matrix.
        colors:    Tuple of three colour strings for X, Y, Z respectively.
        alpha:     Opacity.
        linewidth: Width of the axis lines.
        linestyle: Matplotlib line style, e.g. ``'-'`` or ``'--'``.
        scale:     Length of each drawn axis (in axes units).
    """
    for i, color in enumerate(colors):
        d = R[:, i] * scale
        ax.plot(
            [0, d[0]],
            [0, d[1]],
            [0, d[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
        )
        # Tip marker
        ax.plot(
            [d[0]],
            [d[1]],
            [d[2]],
            color=color,
            alpha=alpha,
            marker="o",
            markersize=4 if linestyle == "-" else 3,
        )


# ── Main visualizer class ─────────────────────────────────────────────────────


class Visualizer:
    """Interactive matplotlib-based visualizer for CNN pretraining samples.

    Args:
        dataset:  A :class:`~cnn_trainer.data_pipeline.dataset.TelemetryDataset`.
        model:    Optional :class:`~cnn_trainer.model.model.PretrainCnn` in eval
                  mode.  If ``None``, predictions are not shown.
        device:   PyTorch device string used for inference (default: ``"cpu"``).
        start_idx: Sample index to display first (default: ``0``).
    """

    def __init__(
        self,
        dataset: TelemetryDataset,
        model: Optional[MultiTaskCnn] = None,
        device: str = "cpu",
        start_idx: int = 0,
    ) -> None:
        if len(dataset) == 0:
            raise ValueError("Dataset is empty — nothing to visualise.")

        self._dataset = dataset
        self._model = model
        self._device = torch.device(device)
        self._n = len(dataset)
        self._idx: int = max(0, min(start_idx, self._n - 1))
        self._typed: str = ""  # digits typed for direct-jump mode

        # ── Figure layout ────────────────────────────────────────────────────
        self._fig = plt.figure(figsize=(14, 7))
        self._fig.subplots_adjust(
            left=0.04, right=0.98, top=0.88, bottom=0.04, wspace=0.1
        )

        self._ax3d: Axes3D = self._fig.add_subplot(1, 2, 1, projection="3d")
        self._ax_img = self._fig.add_subplot(1, 2, 2)

        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update()
        plt.show()

    # ── Key-press event handler ───────────────────────────────────────────────

    def _on_key(self, event) -> None:  # type: ignore[type-arg]
        key = event.key
        if key == "right":
            self._idx = (self._idx + 1) % self._n
            self._typed = ""
        elif key == "left":
            self._idx = (self._idx - 1) % self._n
            self._typed = ""
        elif key in "0123456789":
            self._typed += key
            self._refresh_suptitle(pending=True)
            return
        elif key == "backspace":
            self._typed = self._typed[:-1]
            self._refresh_suptitle(pending=True)
            return
        elif key in ("enter", "return"):
            if self._typed:
                candidate = int(self._typed)
                if 0 <= candidate < self._n:
                    self._idx = candidate
                else:
                    # Out-of-range: show feedback and clear without navigating.
                    self._typed = ""
                    self._refresh_suptitle(
                        pending=False,
                        error=f"Index {candidate} out of range (0–{self._n - 1})",
                    )
                    return
                self._typed = ""
        elif key in ("q", "escape"):
            plt.close(self._fig)
            return
        else:
            return
        self._update()

    # ── Inference ────────────────────────────────────────────────────────────

    def _infer(self, image: torch.Tensor) -> Optional[dict]:
        """Run model inference; return ``None`` if no model is loaded."""
        if self._model is None:
            return None
        with torch.no_grad():
            x = image.unsqueeze(0).to(self._device)  # (1, C, H, W)
            raw = self._model(x)
        grip_prob = torch.sigmoid(raw["grip_position_logit"]).item()
        vis_prob = torch.sigmoid(raw["visibility_logit"]).item()
        quat_wxyz = (
            F.normalize(raw["orientation_pred"], p=2, dim=-1).cpu().numpy().squeeze(0)
        )
        return {
            "grip_prob": grip_prob,
            "visibility_prob": vis_prob,
            "quat_wxyz": quat_wxyz,
        }

    # ── Rendering ────────────────────────────────────────────────────────────

    def _update(self) -> None:
        image, targets = self._dataset[self._idx]
        pred = self._infer(image)

        self._ax3d.cla()
        self._ax_img.cla()

        self._draw_orientation(targets, pred)
        self._draw_image(image, targets, pred)
        self._refresh_suptitle(pending=bool(self._typed), targets=targets, pred=pred)

        self._fig.canvas.draw_idle()

    def _draw_orientation(self, targets: dict, pred: Optional[dict]) -> None:
        ax = self._ax3d
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel("X", labelpad=2)
        ax.set_ylabel("Y", labelpad=2)
        ax.set_zlabel("Z", labelpad=2)
        ax.set_title("Cube Orientation (grip-zone frame)", fontsize=10, pad=6)

        # Draw world-frame reference axes (thin grey) for orientation.
        I = np.eye(3)
        _draw_frame(
            ax,
            I,
            colors=("0.75", "0.75", "0.75"),
            alpha=0.4,
            linewidth=1.0,
            linestyle=":",
            scale=0.95,
        )

        # Ground-truth orientation.
        quat_gt = targets["cube_quat_gripzone_wxyz"].numpy()
        R_gt = _quat_wxyz_to_rotation_matrix(quat_gt)
        _draw_frame(
            ax,
            R_gt,
            colors=_GT_COLORS,
            alpha=1.0,
            linewidth=2.5,
            linestyle="-",
            scale=0.85,
        )

        # Predicted orientation (if model available).
        if pred is not None:
            R_pred = _quat_wxyz_to_rotation_matrix(pred["quat_wxyz"])
            _draw_frame(
                ax,
                R_pred,
                colors=_PRED_COLORS,
                alpha=0.9,
                linewidth=1.5,
                linestyle="--",
                scale=0.85,
            )

        # Legend.
        legend_handles = [
            Line2D(
                [0], [0], color="0.6", linewidth=1, linestyle=":", label="World frame"
            ),
            Line2D([0], [0], color=_GT_COLORS[0], linewidth=2.5, label="X GT"),
            Line2D([0], [0], color=_GT_COLORS[1], linewidth=2.5, label="Y GT"),
            Line2D([0], [0], color=_GT_COLORS[2], linewidth=2.5, label="Z GT"),
        ]
        if pred is not None:
            legend_handles += [
                Line2D(
                    [0],
                    [0],
                    color=_PRED_COLORS[0],
                    linewidth=1.5,
                    linestyle="--",
                    label="X Pred",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_PRED_COLORS[1],
                    linewidth=1.5,
                    linestyle="--",
                    label="Y Pred",
                ),
                Line2D(
                    [0],
                    [0],
                    color=_PRED_COLORS[2],
                    linewidth=1.5,
                    linestyle="--",
                    label="Z Pred",
                ),
            ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)

    def _draw_image(
        self,
        image: torch.Tensor,
        targets: dict,
        pred: Optional[dict],
    ) -> None:
        ax = self._ax_img
        # image is (C, H, W) float32 in [0, 1] — convert to HWC for imshow.
        img_hwc = image.permute(1, 2, 0).cpu().numpy()
        # Clip to [0, 1] in case normalisation shifted values outside this range.
        ax.imshow(np.clip(img_hwc, 0.0, 1.0))
        ax.axis("off")

        # Overlay caption with label values.
        grip_gt = bool(targets["is_cube_in_grip_position"].item() > 0.5)
        vis_gt = bool(targets["cube_in_camera_frame"].item() > 0.5)

        lines = [
            f"Grip (GT):    {'✓ YES' if grip_gt else '✗  NO'}",
            f"Visible (GT): {'✓ YES' if vis_gt else '✗  NO'}",
        ]
        if pred is not None:
            lines += [
                f"Grip (pred):    {pred['grip_prob']:.3f}",
                f"Visible (pred): {pred['visibility_prob']:.3f}",
            ]

        caption = "\n".join(lines)
        ax.text(
            0.02,
            0.02,
            caption,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.75),
            family="monospace",
        )
        ax.set_title("Camera Image", fontsize=10, pad=6)

    def _refresh_suptitle(
        self,
        pending: bool = False,
        targets: Optional[dict] = None,
        pred: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> None:
        if error:
            self._fig.suptitle(error, fontsize=11, color="red")
            self._fig.canvas.draw_idle()
            return

        parts = [f"Sample  {self._idx} / {self._n - 1}"]
        if pending and self._typed:
            parts.append(f"[▶ {self._typed}_]   ←/→ navigate   Enter jump   q quit")
        else:
            parts.append("←/→ navigate   type index + Enter to jump   q quit")
        self._fig.suptitle("     ".join(parts), fontsize=10)


# ── Config & model loading helpers ────────────────────────────────────────────


def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_model(model_path: str | Path, cfg: dict, device: str) -> MultiTaskCnn:
    """Instantiate ``PretrainCnn`` from *cfg* and load weights from *model_path*.

    Args:
        model_path: Path to a full ``PretrainCnn`` ``state_dict`` checkpoint
                    produced by ``cnn_trainer/train.py`` (e.g., ``best_model.pt``).
        cfg:        Config dict loaded from ``cnn_pretrain.yaml``.
        device:     Device string for inference.

    Returns:
        ``PretrainCnn`` in eval mode on *device*.
    """
    backbone_cfg = cfg.get("backbone")
    if backbone_cfg is None:
        raise ValueError("Config is missing the 'backbone' section.")
    heads_cfg = cfg.get("heads")
    if heads_cfg is None:
        raise ValueError("Config is missing the 'heads' section.")

    model = MultiTaskCnn(backbone_cfg=backbone_cfg, heads_cfg=heads_cfg)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _resolve_manifest(curated_dir: str | Path, split: str) -> Path:
    """Return the manifest path for *split* inside *curated_dir*."""
    curated_dir = Path(curated_dir)
    manifest = curated_dir / f"{split}_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest}\n"
            f"Available files: {sorted(p.name for p in curated_dir.iterdir())}"
        )
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualise CNN pretraining data and model predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Ground-truth only (no model):\n"
            "  python -m cnn_trainer.visualizer --curated-dir /path/to/curated --split train\n\n"
            "  # With model predictions:\n"
            "  python -m cnn_trainer.visualizer \\\n"
            "      --curated-dir /path/to/curated --split val \\\n"
            "      --model /path/to/best_model.pt \\\n"
            "      --config configs/cnn_pretrain.yaml\n"
        ),
    )

    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--curated-dir",
        metavar="PATH",
        help="Path to the curated dataset directory (contains *_manifest.json files).",
    )
    data_group.add_argument(
        "--manifest",
        metavar="PATH",
        help="Path to a single manifest JSON file (e.g. train_manifest.json).",
    )

    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Which split to load when --curated-dir is given (default: train).",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "Path to a full PretrainCnn checkpoint (best_model.pt or final_model.pt). "
            "Requires --config."
        ),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to cnn_pretrain.yaml.  Required when --model is given.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device for inference (default: cpu).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="IDX",
        help="Sample index to display first (default: 0).",
    )
    parser.add_argument(
        "--image-mean",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help=(
            "Per-channel image mean for un-normalising display "
            "(only needed if the dataset was normalised). "
            "Reads image_normalization.mean from --config when not given."
        ),
    )
    parser.add_argument(
        "--image-std",
        nargs=3,
        type=float,
        metavar=("R", "G", "B"),
        default=None,
        help="Per-channel image std (see --image-mean).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # ─ Validate argument combinations ────────────────────────────────────────
    if args.model and not args.config:
        parser.error("--config is required when --model is provided.")
    if args.config and not args.model:
        parser.error("--model is required when --config is provided.")

    # ─ Resolve manifest ───────────────────────────────────────────────────────
    if args.curated_dir:
        manifest_path = _resolve_manifest(args.curated_dir, args.split)
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            parser.error(f"Manifest not found: {manifest_path}")

    # ─ Image normalisation ────────────────────────────────────────────────────
    image_mean = args.image_mean
    image_std = args.image_std

    if (image_mean is None) != (image_std is None):
        parser.error(
            "--image-mean and --image-std must both be provided or both omitted."
        )

    # Attempt to read normalisation from config if not given on CLI.
    if image_mean is None and args.config:
        cfg = _load_config(args.config)
        norm = cfg.get("image_normalization")
        if isinstance(norm, dict):
            image_mean = norm.get("mean")
            image_std = norm.get("std")

    # ─ Build dataset ─────────────────────────────────────────────────────────
    print(f"Loading dataset from: {manifest_path}")
    dataset = TelemetryDataset(
        manifest_path=manifest_path,
        image_mean=image_mean,
        image_std=image_std,
    )
    print(f"  {len(dataset)} samples loaded.")

    # ─ Build model ───────────────────────────────────────────────────────────
    model: Optional[MultiTaskCnn] = None
    if args.model:
        cfg = _load_config(args.config)
        print(f"Loading model from: {args.model}")
        model = _load_model(args.model, cfg, args.device)
        print("  Model ready.")

    if len(dataset) == 0:
        print("ERROR: dataset is empty — nothing to visualise.", file=sys.stderr)
        sys.exit(1)

    # ─ Launch visualizer ─────────────────────────────────────────────────────
    print("Opening visualizer window …")
    print("  ← / → : navigate samples")
    print("  0-9 + Enter : jump to index")
    print("  q / Esc : quit")
    Visualizer(dataset=dataset, model=model, device=args.device, start_idx=args.start)


if __name__ == "__main__":
    main()
