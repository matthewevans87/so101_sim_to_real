"""
Cross-platform camera source for vision-based policies.

Supports both Linux and macOS using OpenCV's VideoCapture.
"""

import torch
import numpy as np
import cv2
import platform
from typing import Optional, Tuple
import warnings


class CameraSource:
    """
    Cross-platform camera source that captures RGB frames from a USB webcam.

    Works on both Linux and macOS using OpenCV's VideoCapture.
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        auto_retry: bool = True,
        show_preview: bool = False,
    ):
        """
        Initialize camera source.

        Args:
            camera_id: Camera device index (0 for first camera, 1 for second, etc.)
                      On Linux: typically /dev/video0, /dev/video1, etc.
                      On macOS: typically FaceTime camera is 0, USB cameras are 1+
            width: Desired frame width (None = use camera default)
            height: Desired frame height (None = use camera default)
            auto_retry: If True, attempt to reconnect on read failures
            show_preview: If True, display camera feed in a window
        """
        self.camera_id = camera_id
        self.desired_width = width
        self.desired_height = height
        self.auto_retry = auto_retry
        self.show_preview = show_preview
        self._cap: Optional[cv2.VideoCapture] = None
        self._platform = platform.system()
        self._window_name = f"Camera {camera_id} Preview" if show_preview else None

        print(f"[CameraSource] Detected platform: {self._platform}")
        self._connect()

    def _connect(self):
        """Establish connection to the camera."""
        print(f"[CameraSource] Connecting to camera {self.camera_id}...")

        # On macOS, use AVFoundation backend; on Linux, use V4L2 or default
        if self._platform == "Darwin":  # macOS
            self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
        else:  # Linux and others
            self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            # Fall back to default backend if V4L2 fails
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.camera_id)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera {self.camera_id}. "
                f"Make sure the camera is connected and not in use by another application."
            )

        # Set resolution if specified
        if self.desired_width is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        if self.desired_height is not None:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)

        # Get actual resolution
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)

        print(f"[CameraSource] Camera {self.camera_id} opened successfully")
        print(
            f"[CameraSource] Resolution: {actual_width}x{actual_height} @ {fps:.1f} FPS"
        )

        # Warm up the camera (discard first few frames which might be dark/corrupted)
        for _ in range(5):
            self._cap.read()

    def get_frame(self) -> torch.Tensor:
        """
        Capture a single RGB frame from the camera.

        Returns:
            torch.Tensor: RGB image with shape (H, W, 3), dtype uint8, values in [0, 255]

        Raises:
            RuntimeError: If frame capture fails and auto_retry is False
        """
        if self._cap is None or not self._cap.isOpened():
            if self.auto_retry:
                warnings.warn("Camera disconnected, attempting to reconnect...")
                self._connect()
            else:
                raise RuntimeError("Camera is not connected")

        assert self._cap is not None  # for type checker

        # Flush the buffer by reading and discarding frames
        # This ensures we get the most recent frame, not a buffered old one
        # Read up to 5 frames quickly to clear the buffer
        for _ in range(5):
            self._cap.grab()

        # Now retrieve the most recent frame
        ret, frame = self._cap.retrieve()

        if not ret:
            if self.auto_retry:
                warnings.warn("Failed to read frame, attempting to reconnect...")
                self._reconnect()
                assert self._cap is not None  # for type checker
                ret, frame = self._cap.read()
                if not ret:
                    raise RuntimeError(
                        "Failed to read frame after reconnection attempt"
                    )
            else:
                raise RuntimeError("Failed to read frame from camera")

        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show preview if enabled
        if self.show_preview and self._window_name:
            cv2.imshow(self._window_name, frame)  # Show BGR frame
            cv2.waitKey(1)  # Process window events

        # Convert to torch tensor
        return torch.from_numpy(frame_rgb)

    def _reconnect(self):
        """Reconnect to the camera after a failure."""
        if self._cap is not None:
            self._cap.release()
        self._connect()

    def get_resolution(self) -> Tuple[int, int]:
        """
        Get current camera resolution.

        Returns:
            Tuple[int, int]: (width, height)
        """
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera is not connected")

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def __call__(self) -> torch.Tensor:
        """
        Callable interface for compatibility with PolicyController.

        Returns:
            torch.Tensor: RGB image with shape (H, W, 3)
        """
        return self.get_frame()

    def release(self):
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            print(f"[CameraSource] Camera {self.camera_id} released")
        if self.show_preview and self._window_name:
            cv2.destroyWindow(self._window_name)

    def __del__(self):
        """Ensure camera is released on object destruction."""
        self.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


def list_available_cameras(max_test: int = 10) -> list[int]:
    """
    Detect available camera devices by testing indices.

    Args:
        max_test: Maximum number of camera indices to test

    Returns:
        List of available camera indices
    """
    available = []
    print("[CameraSource] Scanning for available cameras...")

    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  Camera {i}: Available ({width}x{height})")
                available.append(i)
            cap.release()

    if not available:
        print("  No cameras found!")

    return available


def test_camera_source(camera_id: int = 0, num_frames: int = 10):
    """
    Test camera source by capturing a few frames.

    Args:
        camera_id: Camera device index
        num_frames: Number of test frames to capture
    """
    print(f"\n[Test] Testing camera {camera_id}...")

    with CameraSource(camera_id) as camera:
        width, height = camera.get_resolution()
        print(f"[Test] Resolution: {width}x{height}")

        import time

        print(f"[Test] Capturing {num_frames} frames...")
        start_time = time.time()

        for i in range(num_frames):
            frame = camera.get_frame()
            print(
                f"  Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}, "
                f"min={frame.min()}, max={frame.max()}"
            )

        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        print(f"[Test] Captured {num_frames} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        print("[Test] Camera test successful!")


if __name__ == "__main__":
    import sys

    # List available cameras
    available = list_available_cameras()

    if not available:
        print("\nNo cameras found. Please check your USB webcam connection.")
        sys.exit(1)

    # Test the first available camera
    camera_id = available[0]
    print(f"\nTesting camera {camera_id}...")
    test_camera_source(camera_id)
