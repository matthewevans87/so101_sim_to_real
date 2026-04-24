"""Frame I/O: read images/videos as NumPy HWC uint8 arrays; write composite frames."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def iter_frames(path: str | Path) -> Iterator[np.ndarray]:
    """Yield frames from an image or video file as ``(H, W, 3)`` uint8 arrays.

    Uses ``imageio.v3`` for both image and video; the caller does not need to
    know whether the input is a single image or a multi-frame video.

    Args:
        path: Path to an image or video file.

    Yields:
        ``(H, W, 3)`` uint8 NumPy arrays, one per frame.
    """
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for frame I/O: pip install 'imageio[ffmpeg]'"
        ) from exc

    path = Path(path)
    suffix = path.suffix.lower()
    _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    if suffix in _VIDEO_EXTS:
        for frame in iio.imiter(str(path)):
            # imiter yields (H, W, C) or (H, W) arrays
            frame = _ensure_rgb_uint8(np.asarray(frame))
            yield frame
    else:
        # Single image
        img = iio.imread(str(path))
        img = _ensure_rgb_uint8(np.asarray(img))
        yield img


def _ensure_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Convert any HWC array to RGB uint8."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        # Greyscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        # RGBA → RGB
        arr = arr[:, :, :3]
    return arr


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class VideoWriter:
    """Lazy video / image writer.

    Opens the output on the first :meth:`write` call so that the caller does
    not need to know the composite frame resolution ahead of time.

    Args:
        path: Output path (.png for image, .mp4 for video).
        fps: Frames-per-second; only used for video outputs.
    """

    def __init__(self, path: str | Path, fps: float | None) -> None:
        self._path = Path(path)
        self._fps = fps
        self._writer = None
        self._suffix = self._path.suffix.lower()
        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Write a single HWC uint8 composite frame."""
        try:
            import imageio as _imageio
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError(
                "imageio is required: pip install 'imageio[ffmpeg]'"
            ) from exc

        _VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        if self._suffix in _VIDEO_EXTS:
            if self._writer is None:
                if self._fps is None:
                    raise ValueError("fps must be provided for video output.")
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = _imageio.get_writer(str(self._path), fps=self._fps)
            self._writer.append_data(frame)
        else:
            # Single image — overwrite on each call (last frame wins for video
            # inputs written to an image path, but CLI validation forbids that).
            self._path.parent.mkdir(parents=True, exist_ok=True)
            iio.imwrite(str(self._path), frame)

        self._frame_count += 1

    def close(self) -> None:
        """Flush and close the writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def frame_count(self) -> int:
        return self._frame_count
