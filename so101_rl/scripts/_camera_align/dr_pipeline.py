"""dr_pipeline.py — Hot-reloadable domain-randomisation augmentation pipeline.

Used by align_camera.py to insert DR augmentation steps into the sim render
pipeline for live tuning.  The steps are applied *after* Uint8ToFloatCHW and
*before* Resize, matching the training-time order in so101_lift_cube_env.py.

Supports a subset of the domain_randomization.camera.feed augmentations
from the policy training config.  The YAML schema mirrors the training config
so values can be copy-pasted directly.

Example dr_tuning.yaml
-----------------------
steps:
  - type: GaussianNoise
    enabled: true
    std_range: [0.005, 0.015]
  - type: Brightness
    enabled: true
    range: [0.85, 1.15]
  - type: Contrast
    enabled: true
    range: [0.85, 1.15]
  - type: CheapWebcamEffect
    enabled: true
  - type: MotionBlur
    enabled: true
    kernel_size: 5
    strength_range: [0.05, 0.15]
  - type: JpegCompression
    enabled: true
    quality_range: [60, 90]
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from so101.utils.image_processing import (
    CameraBrightnessPipelineStep,
    CameraContrastPipelineStep,
    CheapWebcamEffectPipelineStep,
    GaussianNoisePipelineStep,
    ImagePipelineStep,
    JpegCompressionPipelineStep,
    MotionBlurPipelineStep,
)


# ---------------------------------------------------------------------------
# Step factory
# ---------------------------------------------------------------------------

def _build_step(entry: dict[str, Any], device: str) -> ImagePipelineStep | None:
    """Build a single ImagePipelineStep from a YAML entry dict.

    Returns None if ``enabled: false``.
    Raises ValueError for unknown or mis-configured step types.
    """
    if not entry.get("enabled", True):
        return None

    step_type: str = entry.get("type", "")
    if not step_type:
        raise ValueError("DR step entry is missing required 'type' field.")

    if step_type == "GaussianNoise":
        std_range = entry.get("std_range", [0.005, 0.015])
        return GaussianNoisePipelineStep(
            noise_std_range=tuple(std_range),
            device=device,
        )

    if step_type == "Brightness":
        rng = entry.get("range", [0.85, 1.15])
        return CameraBrightnessPipelineStep(
            brightness_range=tuple(rng),
            device=device,
        )

    if step_type == "Contrast":
        rng = entry.get("range", [0.85, 1.15])
        return CameraContrastPipelineStep(
            contrast_range=tuple(rng),
            device=device,
        )

    if step_type == "CheapWebcamEffect":
        return CheapWebcamEffectPipelineStep(device=device)

    if step_type == "MotionBlur":
        strength_range = entry.get("strength_range", [0.05, 0.15])
        kernel_size = int(entry.get("kernel_size", 5))
        return MotionBlurPipelineStep(
            motion_blur_strength_range=tuple(strength_range),
            motion_blur_kernel_size=kernel_size,
            device=device,
        )

    if step_type == "JpegCompression":
        quality_range = entry.get("quality_range", [60, 90])
        return JpegCompressionPipelineStep(
            quality_range=tuple(quality_range),
            device=device,
        )

    raise ValueError(
        f"Unknown DR step type '{step_type}'.  "
        f"Supported: GaussianNoise, Brightness, Contrast, CheapWebcamEffect, "
        f"MotionBlur, JpegCompression."
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_dr_aug_pipeline(
    yaml_path: str | Path,
    device: torch.device | str,
) -> list[ImagePipelineStep]:
    """Load a DR augmentation YAML and return the list of active steps.

    Steps with ``enabled: false`` are skipped.  Raises ``ValueError`` at call
    time if the YAML is malformed or references an unknown step type.

    Parameters
    ----------
    yaml_path:
        Path to the YAML file (e.g. ``so101_real/configs/dr_tuning.yaml``).
    device:
        Torch device (or device string) to create tensors on inside each step.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"DR config not found: {yaml_path}")

    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict) or "steps" not in data:
        raise ValueError(
            f"DR config '{yaml_path}' must have a top-level 'steps:' list."
        )

    device_str: str = str(device)
    steps: list[ImagePipelineStep] = []
    for entry in data["steps"]:
        step = _build_step(entry, device_str)
        if step is not None:
            steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# File watcher
# ---------------------------------------------------------------------------

class DRConfigWatcher:
    """Background mtime-polling watcher for a DR config YAML.

    Polls the file's modification time every ``poll_interval`` seconds.
    Sets ``self.changed = True`` when a change is detected; the caller is
    responsible for resetting the flag after reloading.
    """

    def __init__(self, path: str | Path, poll_interval: float = 0.5) -> None:
        self.path = Path(path)
        self._poll_interval = poll_interval
        self.changed: bool = False
        self._last_mtime: float = self._get_mtime()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="DRConfigWatcher"
        )
        self._thread.start()

    def _get_mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            mtime = self._get_mtime()
            if mtime != self._last_mtime:
                self._last_mtime = mtime
                self.changed = True

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
