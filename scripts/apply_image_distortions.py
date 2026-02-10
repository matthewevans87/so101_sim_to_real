#!/usr/bin/env python3
"""
Apply image distortions matching the So101LiftCube environment configuration.

Usage:
    python apply_image_distortions.py input.png [options]

Options:
    --all                   Apply all distortions
    --gaussian-blur         Apply Gaussian blur
    --webcam-effect         Apply cheap webcam effect (downscale/upscale)
    --brightness            Apply brightness variation
    --noise                 Apply Gaussian noise
    --contrast              Apply contrast variation
    --motion-blur           Apply motion blur
    --jpeg-compression      Apply JPEG compression artifacts
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import torch
from PIL import Image
import numpy as np

from so101_utils.image_processing import (
    ImagePipeline,
    GaussianBlurPipelineStep,
    CheapWebcamEffectPipelineStep,
    CameraBrightnessPipelineStep,
    GaussianNoisePipelineStep,
    CameraContrastPipelineStep,
    MotionBlurPipelineStep,
    JpegCompressionPipelineStep,
)


# ============================================================================
# Configuration parameters from so101_lift_cube_env_cfg.py
# ============================================================================

# Camera feed augmentation flags
ENABLE_GAUSSIAN_BLUR_RGB = True
ENABLE_CHEAP_WEBCAM_EFFECT = True
ENABLE_CAMERA_BRIGHTNESS = True
ENABLE_CAMERA_NOISE = True
ENABLE_CAMERA_CONTRAST = False

# Advanced camera augmentation
ENABLE_MOTION_BLUR = False
MOTION_BLUR_KERNEL_SIZE = 5
MOTION_BLUR_STRENGTH_RANGE = (0.1, 0.2)

ENABLE_JPEG_COMPRESSION = False
JPEG_QUALITY_RANGE = (60, 70)

# Gaussian noise parameters
CAMERA_GAUSSIAN_NOISE_STD = (0.01, 0.02)  # 1-3% noise
CAMERA_BRIGHTNESS_RANGE = (0.85, 1.15)  # ±15%
CAMERA_CONTRAST_RANGE = (0.8, 1.2)  # ±20%


# ============================================================================
# Main script
# ============================================================================


def load_image(image_path: Path) -> torch.Tensor:
    """Load image and convert to (1, 3, H, W) tensor in range [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]

    # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def save_image(img_tensor: torch.Tensor, output_path: Path):
    """Save (1, 3, H, W) tensor back to image file."""
    # (1, C, H, W) -> (C, H, W) -> (H, W, C)
    img_array = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_array = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array, mode="RGB")
    img.save(output_path)
    print(f"Saved distorted image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply image distortions matching So101LiftCube config"
    )
    parser.add_argument("input_image", type=str, help="Path to input PNG image")
    parser.add_argument("--all", action="store_true", help="Apply all distortions")
    parser.add_argument(
        "--gaussian-blur", action="store_true", help="Apply Gaussian blur"
    )
    parser.add_argument(
        "--webcam-effect", action="store_true", help="Apply cheap webcam effect"
    )
    parser.add_argument(
        "--brightness", action="store_true", help="Apply brightness variation"
    )
    parser.add_argument("--noise", action="store_true", help="Apply Gaussian noise")
    parser.add_argument(
        "--contrast", action="store_true", help="Apply contrast variation"
    )
    parser.add_argument("--motion-blur", action="store_true", help="Apply motion blur")
    parser.add_argument(
        "--jpeg-compression", action="store_true", help="Apply JPEG compression"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Check if input file exists
    input_path = Path(args.input_image)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine which distortions to apply
    apply_all = args.all
    distortions = {
        "gaussian_blur": args.gaussian_blur or apply_all,
        "webcam_effect": args.webcam_effect or apply_all,
        "brightness": args.brightness or apply_all,
        "noise": args.noise or apply_all,
        "contrast": args.contrast or apply_all,
        "motion_blur": args.motion_blur or apply_all,
        "jpeg_compression": args.jpeg_compression or apply_all,
    }

    # If no options specified, show help
    if not any(distortions.values()):
        parser.print_help()
        print("\nError: Please specify at least one distortion option or use --all")
        sys.exit(1)

    # Load image
    print(f"Loading image: {input_path}")
    img_tensor = load_image(input_path).to(args.device)
    print(f"Image shape: {img_tensor.shape}")

    # Build pipeline based on selected distortions
    print("\nBuilding image processing pipeline:")
    pipeline_steps = []

    if distortions["gaussian_blur"]:
        print("  - Gaussian blur")
        pipeline_steps.append(GaussianBlurPipelineStep(kernel_size=7, sigma=2.0))

    if distortions["webcam_effect"]:
        print("  - Cheap webcam effect")
        pipeline_steps.append(CheapWebcamEffectPipelineStep(device=args.device))

    if distortions["brightness"]:
        print(f"  - Brightness (range: {CAMERA_BRIGHTNESS_RANGE})")
        pipeline_steps.append(
            CameraBrightnessPipelineStep(
                brightness_range=CAMERA_BRIGHTNESS_RANGE, device=args.device
            )
        )

    if distortions["noise"]:
        print(f"  - Gaussian noise (std range: {CAMERA_GAUSSIAN_NOISE_STD})")
        pipeline_steps.append(
            GaussianNoisePipelineStep(
                noise_std_range=CAMERA_GAUSSIAN_NOISE_STD, device=args.device
            )
        )

    if distortions["contrast"]:
        print(f"  - Contrast (range: {CAMERA_CONTRAST_RANGE})")
        pipeline_steps.append(
            CameraContrastPipelineStep(
                contrast_range=CAMERA_CONTRAST_RANGE, device=args.device
            )
        )

    if distortions["motion_blur"]:
        print(
            f"  - Motion blur (kernel size: {MOTION_BLUR_KERNEL_SIZE}, strength: {MOTION_BLUR_STRENGTH_RANGE})"
        )
        pipeline_steps.append(
            MotionBlurPipelineStep(
                motion_blur_strength_range=MOTION_BLUR_STRENGTH_RANGE,
                motion_blur_kernel_size=MOTION_BLUR_KERNEL_SIZE,
                device=args.device,
            )
        )

    if distortions["jpeg_compression"]:
        print(f"  - JPEG compression (quality range: {JPEG_QUALITY_RANGE})")
        pipeline_steps.append(
            JpegCompressionPipelineStep(
                quality_range=JPEG_QUALITY_RANGE, device=args.device
            )
        )

    # Create and apply pipeline
    if pipeline_steps:
        print("\nApplying distortions...")
        pipeline = ImagePipeline(pipeline_steps)
        img_tensor = pipeline.process(img_tensor)
        # Clamp to valid range
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    else:
        print("\nNo distortions selected!")
        sys.exit(1)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = input_path.parent / f"{input_path.stem}_{timestamp}.png"

    # Save result
    save_image(img_tensor, output_path)
    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
