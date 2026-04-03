#!/usr/bin/env python3
"""Interactive viewer for configuring the so101_utils image distortion pipeline.

Load any image file (or an NPZ telemetry shard) and toggle / tune each
pipeline step in real time.  Changes apply immediately so the pipeline can be
calibrated to match real-world camera appearance.

Layout
------
::

    ┌──────────────────────────────────────────────────────────────────────┐
    │ [Browse image…]   path/to/image.png          [Frame: ▲▼]  (NPZ only) │
    ├─────────────────────┬────────────────────────────────────────────────┤
    │  Pipeline Steps     │                                                │
    │  ─────────────────  │   Original              Processed              │
    │  ☐ Resize           │                                                │
    │  ☐ Cheap Webcam     │                                                │
    │  ☐ JPEG Compression │                                                │
    │  ☐ Motion Blur      │                                                │
    │     Strength ──●──  │                                                │
    │     Kernel   ──●──  │                                                │
    │  ☐ Gaussian Blur    │                                                │
    │  …                  │                                                │
    │  ☑ Clamp [0, 1]     │                                                │
    ├─────────────────────┴────────────────────────────────────────────────┤
    │ Status bar                                                           │
    └──────────────────────────────────────────────────────────────────────┘

Usage::

    python -m image_pipeline_viewer [--image PATH] [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from so101_utils.image_processing.image_pipeline import (
    CameraBrightnessPipelineStep,
    CameraContrastPipelineStep,
    ClampPipelineStep,
    GaussianBlurPipelineStep,
    GaussianNoisePipelineStep,
    ImagePipelineStep,
    JpegCompressionPipelineStep,
    ResizePipelineStep,
)

# ── Deterministic viewer-side wrappers ────────────────────────────────────────
#
# The production pipeline steps for motion blur and the cheap-webcam effect
# choose direction / scale randomly on each call.  For the viewer we need
# repeatable, user-controlled behaviour, so we supply thin replacements that
# take explicit scalar values instead of ranges.


class _MotionBlurFixed(ImagePipelineStep):
    """Motion blur with fixed strength and direction (no stochastic sampling)."""

    def __init__(self, strength: float, kernel_size: int, direction: int) -> None:
        self._s = float(strength)
        self._k = int(kernel_size)
        self._d = int(direction) % 4

    def process(self, images: torch.Tensor) -> torch.Tensor:
        if self._s < 1e-3:
            return images
        k, s, d = self._k, self._s, self._d
        ker = torch.zeros((k, k), dtype=torch.float32)
        if d == 0:  # Horizontal
            ker[k // 2, :] = 1.0
        elif d == 1:  # Vertical
            ker[:, k // 2] = 1.0
        elif d == 2:  # Diagonal (╲)
            for i in range(k):
                ker[i, i] = 1.0
        else:  # Diagonal (╱)
            for i in range(k):
                ker[i, k - 1 - i] = 1.0
        ker = ker / ker.sum() * s
        ker[k // 2, k // 2] += 1.0 - s  # blend toward identity at centre
        ker = ker.view(1, 1, k, k).expand(3, 1, k, k).contiguous()
        pad = k // 2
        return F.conv2d(
            F.pad(images, (pad, pad, pad, pad), mode="replicate"),
            ker,
            groups=3,
        )


class _CheapWebcamFixed(ImagePipelineStep):
    """Cheap webcam downsample/upsample with a fixed scale factor."""

    def __init__(self, scale: float) -> None:
        self._scale = float(max(0.05, min(1.0, scale)))

    def process(self, images: torch.Tensor) -> torch.Tensor:
        _N, _C, H, W = images.shape
        H2 = max(1, int(H * self._scale))
        W2 = max(1, int(W * self._scale))
        down = F.interpolate(images, size=(H2, W2), mode="area")
        return F.interpolate(down, size=(H, W), mode="bilinear", align_corners=False)


# ── Parameter and step descriptors ───────────────────────────────────────────


@dataclass
class ParamSpec:
    """Specification for one tunable slider in the UI."""

    key: str  # internal dict key
    label: str  # human-readable label shown in UI
    lo: float  # slider minimum
    hi: float  # slider maximum
    default: float  # initial value
    resolution: float  # snap granularity
    fmt: str = ".2f"  # Python format spec for the live value label


@dataclass
class StepDescriptor:
    """Describes a pipeline step and how to rebuild it from UI parameters."""

    name: str
    params: list[ParamSpec]
    build: Callable[[dict[str, float], str], ImagePipelineStep]
    enabled: bool = False


def _make_odd(v: float) -> int:
    """Round *v* to the nearest odd integer >= 1."""
    n = max(1, int(round(v)))
    return n if n % 2 == 1 else n + 1


# Catalogue is ordered: the same order is used when applying the pipeline.
STEP_CATALOGUE: list[StepDescriptor] = [
    StepDescriptor(
        name="Resize",
        enabled=False,
        params=[
            ParamSpec("height", "Height (px)", 32, 1080, 108, 1, ".0f"),
            ParamSpec("width", "Width  (px)", 32, 1920, 192, 1, ".0f"),
        ],
        build=lambda p, _dev: ResizePipelineStep(
            size=(int(p["height"]), int(p["width"])), mode="bilinear"
        ),
    ),
    StepDescriptor(
        name="Cheap Webcam",
        enabled=False,
        params=[
            ParamSpec("scale", "Downsample scale", 0.1, 1.0, 0.5, 0.01),
        ],
        build=lambda p, _dev: _CheapWebcamFixed(scale=p["scale"]),
    ),
    StepDescriptor(
        name="JPEG Compression",
        enabled=False,
        params=[
            ParamSpec(
                "quality", "Quality  (1 = worst, 95 = best)", 1, 95, 50, 1, ".0f"
            ),
        ],
        build=lambda p, dev: JpegCompressionPipelineStep(
            quality_range=(int(p["quality"]), int(p["quality"])), device=dev
        ),
    ),
    StepDescriptor(
        name="Motion Blur",
        enabled=False,
        params=[
            ParamSpec("strength", "Strength", 0.0, 1.0, 0.4, 0.01),
            ParamSpec("ksize", "Kernel size", 3, 21, 7, 2, ".0f"),
            ParamSpec("direction", "Direction  0=H  1=V  2=╲  3=╱", 0, 3, 0, 1, ".0f"),
        ],
        build=lambda p, _dev: _MotionBlurFixed(
            strength=p["strength"],
            kernel_size=_make_odd(p["ksize"]),
            direction=int(round(p["direction"])),
        ),
    ),
    StepDescriptor(
        name="Gaussian Blur",
        enabled=False,
        params=[
            ParamSpec("ksize", "Kernel size", 3, 31, 7, 2, ".0f"),
            ParamSpec("sigma", "Sigma", 0.1, 10, 2, 0.1),
        ],
        build=lambda p, _dev: GaussianBlurPipelineStep(
            kernel_size=_make_odd(p["ksize"]), sigma=p["sigma"]
        ),
    ),
    StepDescriptor(
        name="Gaussian Noise",
        enabled=False,
        params=[
            ParamSpec("std", "Std", 0.0, 0.2, 0.03, 0.001, ".4f"),
        ],
        build=lambda p, dev: GaussianNoisePipelineStep(
            noise_std_range=(p["std"], p["std"]), device=dev
        ),
    ),
    StepDescriptor(
        name="Brightness",
        enabled=False,
        params=[
            ParamSpec("factor", "Factor", 0.3, 2.0, 1.0, 0.01),
        ],
        build=lambda p, dev: CameraBrightnessPipelineStep(
            brightness_range=(p["factor"], p["factor"]), device=dev
        ),
    ),
    StepDescriptor(
        name="Contrast",
        enabled=False,
        params=[
            ParamSpec("factor", "Factor", 0.3, 2.0, 1.0, 0.01),
        ],
        build=lambda p, dev: CameraContrastPipelineStep(
            contrast_range=(p["factor"], p["factor"]), device=dev
        ),
    ),
    StepDescriptor(
        name="Clamp [0, 1]",
        enabled=True,
        params=[],
        build=lambda _p, _dev: ClampPipelineStep(0.0, 1.0),
    ),
]

# ── Image loading ─────────────────────────────────────────────────────────────


def load_image_file(path: Path, npz_index: int = 0) -> tuple[torch.Tensor, int]:
    """Load an image into a ``(1, 3, H, W)`` float32 CPU tensor in ``[0, 1]``.

    Supports standard image files (PNG, JPG, BMP, TIFF, …) and NPZ telemetry
    shards produced by ``collect_telemetry.py`` (uses the ``rgb`` array).

    Args:
        path: Path to the image or NPZ file.
        npz_index: Frame index when *path* is an NPZ shard.

    Returns:
        ``(tensor, n_frames)`` where *n_frames* is 1 for non-NPZ files and the
        number of frames in the shard for NPZ files.
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=False)
        if "rgb" not in data:
            raise ValueError(f"NPZ file has no 'rgb' key: {path}")
        rgb = data["rgb"]  # (N, H, W, C) uint8
        n_frames = int(rgb.shape[0])
        idx = max(0, min(npz_index, n_frames - 1))
        hwc = rgb[idx].astype(np.float32) / 255.0  # (H, W, C)
        tensor = torch.from_numpy(hwc).permute(2, 0, 1).unsqueeze(0)
        return tensor, n_frames
    else:
        try:
            from PIL import Image as _PIL
        except ImportError as exc:
            raise ImportError(
                "Pillow is required to load image files: pip install pillow"
            ) from exc
        img = _PIL.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor, 1


# ── Scrollable frame ──────────────────────────────────────────────────────────


class _ScrollableFrame(ttk.Frame):
    """A vertically scrollable container.  Add children to ``.inner``."""

    def __init__(self, parent: tk.Widget, width: int = 300, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, width=width)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        # Mouse wheel — Linux (Button-4/5) and Windows/macOS (MouseWheel)
        for evt in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            canvas.bind_all(evt, self._on_scroll)

    def _on_scroll(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        if event.num == 4:
            self.master.nametowidget(  # type: ignore[attr-defined]
                self.winfo_children()[1].winfo_name()
            )
        if hasattr(event, "delta") and event.delta:
            delta = -1 if event.delta > 0 else 1
        elif event.num == 4:
            delta = -1
        else:
            delta = 1
        # Scroll the canvas (first child)
        canvas: tk.Canvas = self.winfo_children()[1]  # type: ignore[assignment]
        canvas.yview_scroll(delta, "units")


# ── Per-step UI panel ─────────────────────────────────────────────────────────


class StepPanel(ttk.LabelFrame):
    """A labelled frame containing an enable checkbox and parameter sliders."""

    def __init__(
        self,
        parent: tk.Widget,
        descriptor: StepDescriptor,
        on_change: Callable[[], None],
    ) -> None:
        super().__init__(parent, text="", padding=(4, 2))
        self.name = descriptor.name
        self._desc = descriptor
        self._on_change = on_change

        # ── Enable toggle ─────────────────────────────────────────────────
        self._enabled_var = tk.BooleanVar(value=descriptor.enabled)
        ttk.Checkbutton(
            self,
            text=descriptor.name,
            variable=self._enabled_var,
            command=on_change,
            style="StepHeader.TCheckbutton",
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(2, 1))

        # ── Parameter sliders ─────────────────────────────────────────────
        self._param_vars: dict[str, tk.DoubleVar] = {}
        self._val_labels: dict[str, ttk.Label] = {}

        for row_idx, spec in enumerate(descriptor.params, start=1):
            var = tk.DoubleVar(value=spec.default)
            self._param_vars[spec.key] = var

            ttk.Label(self, text=spec.label, font=("TkDefaultFont", 8)).grid(
                row=row_idx, column=0, sticky="w", padx=(20, 4)
            )
            ttk.Scale(
                self,
                from_=spec.lo,
                to=spec.hi,
                orient="horizontal",
                variable=var,
                command=lambda _v, s=spec, v=var: self._on_slider(s, v),
                length=160,
            ).grid(row=row_idx, column=1, sticky="ew", padx=4)

            lbl = ttk.Label(
                self,
                text=_fmt(spec, var.get()),
                width=8,
                font=("TkFixedFont", 8),
                anchor="e",
            )
            lbl.grid(row=row_idx, column=2, sticky="e", padx=(0, 4))
            self._val_labels[spec.key] = lbl

        self.columnconfigure(1, weight=1)

    def _on_slider(self, spec: ParamSpec, var: tk.DoubleVar) -> None:
        raw = var.get()
        snapped = round(raw / spec.resolution) * spec.resolution
        if abs(snapped - raw) > 1e-9:
            var.set(snapped)
        self._val_labels[spec.key].config(text=_fmt(spec, snapped))
        self._on_change()

    @property
    def enabled(self) -> bool:
        return bool(self._enabled_var.get())

    def get_step(self, device: str) -> Optional[ImagePipelineStep]:
        """Return a freshly constructed step with current param values, or None."""
        if not self.enabled:
            return None
        params = {k: v.get() for k, v in self._param_vars.items()}
        return self._desc.build(params, device)


def _fmt(spec: ParamSpec, value: float) -> str:
    return format(value, spec.fmt)


# ── Main application ──────────────────────────────────────────────────────────


class PipelineViewer:
    """Main tkinter application containing the pipeline configuration viewer."""

    _SIDEBAR_W = 320

    def __init__(
        self,
        root: tk.Tk,
        initial_image: Optional[Path],
        device: str,
    ) -> None:
        self._root = root
        self._device = device
        self._image: Optional[torch.Tensor] = None  # (1, 3, H, W) float [0,1] cpu
        self._current_path: Optional[Path] = None
        self._npz_n_frames: int = 1

        root.title("Image Pipeline Viewer")
        root.geometry("1280x720")
        root.minsize(900, 500)

        self._build_toolbar()
        self._build_main_area()
        self._build_status_bar()

        if initial_image is not None:
            self._load_path(initial_image)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self._root, padding=(6, 4))
        bar.pack(side="top", fill="x")

        ttk.Button(bar, text="Browse image…", command=self._browse).pack(
            side="left", padx=(0, 8)
        )
        self._path_var = tk.StringVar(value="(no image loaded)")
        ttk.Label(bar, textvariable=self._path_var, font=("TkFixedFont", 9)).pack(
            side="left", fill="x", expand=True
        )

        # NPZ frame spinner — hidden until an NPZ file is loaded.
        self._npz_frame_widget = ttk.Frame(bar)
        ttk.Label(self._npz_frame_widget, text="Frame:").pack(side="left")
        self._npz_index_var = tk.IntVar(value=0)
        sp = ttk.Spinbox(
            self._npz_frame_widget,
            from_=0,
            to=9999,
            textvariable=self._npz_index_var,
            width=6,
            command=self._on_npz_index_changed,
        )
        sp.pack(side="left", padx=4)
        sp.bind("<Return>", lambda _e: self._on_npz_index_changed())
        self._npz_frame_widget.pack_forget()

    def _build_main_area(self) -> None:
        main = ttk.Frame(self._root)
        main.pack(side="top", fill="both", expand=True)

        # ── Left: scrollable step controls ───────────────────────────────
        sidebar = ttk.Frame(main, width=self._SIDEBAR_W)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ttk.Label(
            sidebar, text="Pipeline Steps", font=("TkDefaultFont", 9, "bold")
        ).pack(side="top", anchor="w", padx=8, pady=(6, 2))
        ttk.Separator(sidebar, orient="horizontal").pack(fill="x", padx=6, pady=(0, 4))

        scroll = _ScrollableFrame(sidebar, width=self._SIDEBAR_W - 20)
        scroll.pack(fill="both", expand=True)

        self._panels: list[StepPanel] = []
        for desc in STEP_CATALOGUE:
            panel = StepPanel(scroll.inner, desc, on_change=self._render)
            panel.pack(fill="x", padx=4, pady=3)
            self._panels.append(panel)

        # ── Right: matplotlib image display ──────────────────────────────
        self._fig = Figure(tight_layout=True)
        self._ax_orig = self._fig.add_subplot(1, 2, 1)
        self._ax_proc = self._fig.add_subplot(1, 2, 2)
        for ax in (self._ax_orig, self._ax_proc):
            ax.axis("off")

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=main)
        self._mpl_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _build_status_bar(self) -> None:
        self._status_var = tk.StringVar(value="Load an image to begin.")
        ttk.Label(
            self._root,
            textvariable=self._status_var,
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(side="bottom", anchor="w", padx=8, pady=2)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _browse(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Select image or NPZ shard",
            filetypes=[
                ("Images & NPZ", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.npz"),
                ("All files", "*.*"),
            ],
        )
        if path_str:
            self._load_path(Path(path_str))

    def _load_path(self, path: Path) -> None:
        if not path.exists():
            self._status_var.set(f"File not found: {path}")
            return
        try:
            tensor, n_frames = load_image_file(path, npz_index=0)
        except Exception as exc:
            self._status_var.set(f"Error loading file: {exc}")
            return

        self._image = tensor
        self._current_path = path
        self._npz_n_frames = n_frames
        self._npz_index_var.set(0)
        self._path_var.set(str(path))

        if path.suffix.lower() == ".npz" and n_frames > 1:
            self._npz_frame_widget.pack(side="right", padx=8)
        else:
            self._npz_frame_widget.pack_forget()

        C, H, W = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        self._status_var.set(
            f"Loaded: {path.name}  |  {W}×{H}  |  {C} channels"
            + (f"  |  {n_frames} frames" if n_frames > 1 else "")
        )
        self._render()

    def _on_npz_index_changed(self) -> None:
        if self._current_path is None:
            return
        idx = max(0, min(int(self._npz_index_var.get()), self._npz_n_frames - 1))
        self._npz_index_var.set(idx)
        try:
            tensor, _ = load_image_file(self._current_path, npz_index=idx)
        except Exception as exc:
            self._status_var.set(f"Error: {exc}")
            return
        self._image = tensor
        self._render()

    # ── Pipeline execution and display ────────────────────────────────────────

    def _render(self) -> None:
        if self._image is None:
            return

        original = self._image.clone()

        active_steps: list[tuple[str, ImagePipelineStep]] = []
        for panel in self._panels:
            step = panel.get_step(self._device)
            if step is not None:
                active_steps.append((panel.name, step))

        processed = original.clone()
        with torch.no_grad():
            for _name, step in active_steps:
                processed = step.process(processed)

        orig_np = _to_display(original)
        proc_np = _to_display(processed)

        self._ax_orig.cla()
        self._ax_proc.cla()

        self._ax_orig.imshow(orig_np)
        self._ax_orig.set_title("Original", fontsize=10)
        self._ax_orig.axis("off")

        step_names = [name for name, _ in active_steps]
        proc_title = "Processed" + (
            "\n" + " → ".join(step_names) if step_names else "\n(no steps active)"
        )
        self._ax_proc.imshow(proc_np)
        self._ax_proc.set_title(proc_title, fontsize=8)
        self._ax_proc.axis("off")

        orig_hw = f"{orig_np.shape[1]}×{orig_np.shape[0]}"
        proc_hw = f"{proc_np.shape[1]}×{proc_np.shape[0]}"
        self._status_var.set(
            f"{self._current_path.name if self._current_path else ''}  "
            f"Original: {orig_hw}  →  Processed: {proc_hw}  "
            f"| {len(active_steps)} step(s) active"
        )
        self._mpl_canvas.draw()


def _to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a ``(1, C, H, W)`` float tensor to an ``(H, W, 3)`` uint8-ready array.

    Values are clipped to ``[0, 1]``.  If normalisation has pushed values far
    outside that range, the array is rescaled so the display is always valid.
    """
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    lo, hi = arr.min(), arr.max()
    if lo < -0.1 or hi > 1.1:
        # Re-normalise for display (e.g. after ImageNet normalisation step)
        arr = (arr - lo) / (hi - lo + 1e-8)
    return np.clip(arr, 0.0, 1.0)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive image distortion pipeline viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m image_pipeline_viewer\n"
            "  python -m image_pipeline_viewer --image /path/to/frame.png\n"
            "  python -m image_pipeline_viewer --image /path/to/shard.npz\n"
        ),
    )
    parser.add_argument(
        "--image",
        metavar="PATH",
        default=None,
        help="Image file or NPZ shard to open on launch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device for pipeline steps (default: cpu).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass

    style = ttk.Style(root)
    try:
        style.configure("StepHeader.TCheckbutton", font=("TkDefaultFont", 9, "bold"))
    except tk.TclError:
        style.configure("StepHeader.TCheckbutton")

    PipelineViewer(
        root=root,
        initial_image=Path(args.image) if args.image else None,
        device=args.device,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
