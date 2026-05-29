"""camera.py — Camera source for real-robot inference.

Uses OpenCV VideoCapture with V4L2 on Linux.  A background grab thread
continuously drains the capture buffer so ``get_frame()`` always returns
the freshest available frame without stale-buffer lag.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml


@dataclass
class CameraConfig:
    """Explicit camera configuration — all fields required, no silent defaults.

    Load from YAML via ``CameraConfig.load(path)``.
    """

    device_index: int
    """V4L2 device index (e.g. 0 for /dev/video0)."""

    capture_width: int
    """Camera capture width in pixels."""

    capture_height: int
    """Camera capture height in pixels."""

    warmup_frames: int
    """Number of frames to discard at startup before the first usable frame."""

    buffer_flush_count: int
    """Number of ``grab()`` calls before each ``retrieve()`` to drain stale frames."""

    fourcc: str
    """V4L2 pixel format (4-char code, e.g. ``'MJPG'`` or ``'YUYV'``).

    Use ``'MJPG'`` for USB webcams that support Motion-JPEG — it is the only
    format most cameras advertise at ≥30 fps for HD resolutions.  If this is
    set incorrectly the driver will silently fall back to a lower-resolution
    YUYV mode.
    """

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Camera config not found: {path}")
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cam = data.get("camera")
        if cam is None:
            raise ValueError(
                f"Camera config YAML must contain a top-level 'camera' key: {path}"
            )
        required = {"device_index", "capture_width", "capture_height"}
        missing = required - set(cam)
        if missing:
            raise ValueError(
                f"Camera config is missing required keys: {sorted(missing)}\n"
                f"Config path: {path}"
            )
        return cls(
            device_index=int(cam["device_index"]),
            capture_width=int(cam["capture_width"]),
            capture_height=int(cam["capture_height"]),
            warmup_frames=int(cam.get("warmup_frames", 10)),
            buffer_flush_count=int(cam.get("buffer_flush_count", 0)),
            fourcc=str(cam.get("fourcc", "MJPG")),
        )


class CameraSource:
    """OpenCV VideoCapture wrapper with background grab thread.

    The grab thread continuously calls ``cap.grab()`` so the decode buffer
    never accumulates stale frames.  ``get_frame()`` calls ``cap.retrieve()``
    on the most recently grabbed frame.

    Usage::

        cam = CameraSource(config)
        cam.open()
        frame = cam.get_frame()   # (H, W, 3) uint8 numpy array or torch tensor
        cam.close()
    """

    def __init__(self, config: CameraConfig) -> None:
        self._cfg = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None

    def open(self) -> None:
        """Open the camera and start the background grab thread."""
        cfg = self._cfg
        cap = cv2.VideoCapture(cfg.device_index, cv2.CAP_V4L2)
        # Request the pixel format first — MJPEG must be set before the
        # resolution or the driver defaults to YUYV, which on most webcams
        # only supports ≤10 fps at 1080p.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
        # Minimize internal buffer to 1 frame to reduce latency.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera at device index {cfg.device_index}.\n"
                "Check that the camera is connected and the device index is correct."
            )

        # Warm up — discard stale frames from device buffer
        for _ in range(cfg.warmup_frames):
            cap.grab()

        self._cap = cap
        self._stop_event.clear()
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True, name="camera_grab"
        )
        self._grab_thread.start()

    def _grab_loop(self) -> None:
        """Continuously grab frames without decoding to keep the buffer fresh."""
        while not self._stop_event.is_set():
            with self._lock:
                if self._cap is not None:
                    self._cap.grab()
            time.sleep(0.001)  # ~1 ms between grabs; decode is on-demand

    def get_frame(self) -> np.ndarray:
        """Retrieve the most recently grabbed frame.

        Returns
        -------
        np.ndarray
            Shape ``(H, W, 3)`` uint8, BGR colour order.
        """
        with self._lock:
            if self._cap is None:
                raise RuntimeError(
                    "Camera is not open. Call CameraSource.open() first."
                )
            # Flush buffer_flush_count additional grabs before decode
            for _ in range(self._cfg.buffer_flush_count):
                self._cap.grab()
            ret, frame = self._cap.retrieve()

        if not ret or frame is None:
            raise RuntimeError(
                "Camera retrieve() failed. Check that the device is still connected."
            )
        return frame

    def get_frame_rgb(self) -> np.ndarray:
        """Return the most recent frame as ``(H, W, 3)`` uint8 RGB."""
        return cv2.cvtColor(self.get_frame(), cv2.COLOR_BGR2RGB)

    def get_frame_tensor(self) -> torch.Tensor:
        """Return the most recent frame as ``(1, H, W, 3)`` uint8 RGB tensor."""
        rgb = self.get_frame_rgb()
        return torch.from_numpy(rgb).unsqueeze(0)

    def close(self) -> None:
        """Stop the grab thread and release the camera."""
        self._stop_event.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2.0)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
