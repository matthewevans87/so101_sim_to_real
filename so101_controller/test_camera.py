#!/usr/bin/env python3
"""
Test script for CameraSource - verify USB webcam connectivity.

This script helps you:
1. Detect available cameras
2. Test camera capture
3. Optionally save test frames for inspection
"""

import argparse
import sys
from pathlib import Path

from .CameraSource import (
    CameraSource,
    list_available_cameras,
    test_camera_source,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test USB webcam connectivity and capture"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera device ID to test (e.g., 0, 1, 2)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of test frames to capture (default: 10)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Directory to save test frames (optional)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Desired frame width (optional)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Desired frame height (optional)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Camera Source Test Utility")
    print("=" * 60)

    # List available cameras
    print("\nScanning for cameras...")
    available = list_available_cameras()

    if not available:
        print("\n❌ No cameras found!")
        print("\nTroubleshooting:")
        print("  1. Check USB webcam is connected")
        print("  2. On Linux, check permissions: ls -l /dev/video*")
        print("  3. On macOS, grant camera permissions in System Preferences")
        print("  4. Try: opencv-python installed? pip install opencv-python")
        sys.exit(1)

    # Select camera to test
    if args.camera is None:
        camera_id = available[0]
        print(f"\n✓ Using camera {camera_id} (first available)")
    else:
        camera_id = args.camera
        if camera_id not in available:
            print(f"\n⚠ Warning: Camera {camera_id} was not detected in scan")
            print(f"   Available cameras: {available}")
            response = input(f"   Try camera {camera_id} anyway? (y/n): ")
            if response.lower() != "y":
                sys.exit(1)

    # Test camera capture
    print(f"\n{'='*60}")
    print(f"Testing Camera {camera_id}")
    print(f"{'='*60}")

    try:
        with CameraSource(
            camera_id=camera_id, width=args.width, height=args.height
        ) as camera:
            width, height = camera.get_resolution()
            print(f"\n✓ Camera opened successfully")
            print(f"  Resolution: {width}x{height}")

            # Capture test frames
            print(f"\nCapturing {args.frames} test frames...")
            import time
            import torch

            frames = []
            start_time = time.time()

            for i in range(args.frames):
                frame = camera.get_frame()
                frames.append(frame)

                if (i + 1) % 5 == 0 or i == 0:
                    print(
                        f"  Frame {i+1}/{args.frames}: "
                        f"shape={frame.shape}, dtype={frame.dtype}, "
                        f"range=[{frame.min()}, {frame.max()}]"
                    )

            elapsed = time.time() - start_time
            fps = args.frames / elapsed

            print(f"\n✓ Successfully captured {args.frames} frames")
            print(f"  Elapsed time: {elapsed:.2f}s")
            print(f"  Average FPS: {fps:.1f}")

            # Save frames if requested
            if args.save:
                save_dir = Path(args.save)
                save_dir.mkdir(parents=True, exist_ok=True)

                print(f"\nSaving frames to {save_dir}...")

                try:
                    from PIL import Image
                    import numpy as np

                    for i, frame in enumerate(frames):
                        # Convert tensor to numpy array
                        frame_np = frame.cpu().numpy()
                        img = Image.fromarray(frame_np)
                        img_path = save_dir / f"frame_{i:03d}.png"
                        img.save(img_path)

                        if (i + 1) % 5 == 0 or i == 0:
                            print(f"  Saved: {img_path.name}")

                    print(f"\n✓ Saved {len(frames)} frames to {save_dir}")

                except ImportError:
                    print("\n⚠ PIL not installed, cannot save frames")
                    print("  Install with: pip install Pillow")

            print(f"\n{'='*60}")
            print("✓ Camera test PASSED")
            print(f"{'='*60}")
            print(f"\nCamera {camera_id} is working correctly!")
            print(f"You can use this camera with:")
            print(f"  python run_vision_policy_example.py --camera {camera_id}")

    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ Camera test FAILED")
        print(f"{'='*60}")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("  1. Check if another application is using the camera")
        print("  2. Try a different camera ID")
        print("  3. On Linux: check /dev/video* permissions")
        print("  4. On macOS: check System Preferences > Security & Privacy > Camera")
        sys.exit(1)


if __name__ == "__main__":
    main()
