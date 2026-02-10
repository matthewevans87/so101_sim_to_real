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
import torch.nn.functional as F
from PIL import Image
import numpy as np


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
# Image distortion functions
# ============================================================================


def gaussian_blur_rgb(
    rgb: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Apply a channel-wise Gaussian blur to an (N, 3, H, W) tensor."""
    device = rgb.device
    # 1D Gaussian
    x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Outer product → 2D kernel
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]  # (K, K)
    kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size)  # (C, 1, K, K)

    # Depthwise convolution: one kernel per channel
    return F.conv2d(
        rgb,
        kernel_2d,
        padding=kernel_size // 2,
        groups=3,
    )


def cheap_webcam_effect(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: (N, 3, H, W) in [0,1]
    Mimic a low-res sensor + resize.
    """
    N, C, H, W = rgb.shape

    # pick a downscale factor (e.g., 0.4–0.7)
    scale = 0.4 + 0.3 * torch.rand(1, device=rgb.device).item()
    H_low = int(H * scale)
    W_low = int(W * scale)

    # Downsample (area or bilinear)
    low_res = F.interpolate(rgb, size=(H_low, W_low), mode="area")

    # Upsample back
    upsampled = F.interpolate(
        low_res, size=(H, W), mode="bilinear", align_corners=False
    )

    return upsampled


def apply_brightness(
    rgb: torch.Tensor, brightness_range: tuple[float, float]
) -> torch.Tensor:
    """Apply random brightness adjustment."""
    brightness_factor = (
        torch.rand(1, device=rgb.device) * (brightness_range[1] - brightness_range[0])
        + brightness_range[0]
    )
    return torch.clamp(rgb * brightness_factor, 0.0, 1.0)


def apply_noise(
    rgb: torch.Tensor, noise_std_range: tuple[float, float]
) -> torch.Tensor:
    """Apply Gaussian noise."""
    noise_std = (
        torch.rand(1, device=rgb.device) * (noise_std_range[1] - noise_std_range[0])
        + noise_std_range[0]
    )
    noise = torch.randn_like(rgb) * noise_std
    return torch.clamp(rgb + noise, 0.0, 1.0)


def apply_contrast(
    rgb: torch.Tensor, contrast_range: tuple[float, float]
) -> torch.Tensor:
    """Apply random contrast adjustment."""
    contrast_factor = (
        torch.rand(1, device=rgb.device) * (contrast_range[1] - contrast_range[0])
        + contrast_range[0]
    )
    # Contrast adjustment: (rgb - 0.5) * factor + 0.5
    return torch.clamp((rgb - 0.5) * contrast_factor + 0.5, 0.0, 1.0)


def apply_motion_blur(
    images: torch.Tensor,
    motion_blur_strength_range: tuple[float, float],
    motion_blur_kernel_size: int,
    device: str,
) -> torch.Tensor:
    """Apply directional motion blur to simulate camera motion."""
    # Random blur strength for this step
    blur_strength = (
        torch.rand(1, device=device)
        * (motion_blur_strength_range[1] - motion_blur_strength_range[0])
        + motion_blur_strength_range[0]
    )

    if blur_strength < 0.01:  # Skip if very weak
        return images

    kernel_size = motion_blur_kernel_size

    # Random blur direction: horizontal, vertical, or diagonal
    blur_type = torch.randint(0, 4, (1,), device=device).item()

    # Create blur kernel
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    if blur_type == 0:  # Horizontal
        kernel[kernel_size // 2, :] = 1.0
    elif blur_type == 1:  # Vertical
        kernel[:, kernel_size // 2] = 1.0
    elif blur_type == 2:  # Diagonal \
        for i in range(kernel_size):
            kernel[i, i] = 1.0
    else:  # Diagonal /
        for i in range(kernel_size):
            kernel[i, kernel_size - 1 - i] = 1.0

    kernel = kernel / kernel.sum()  # Normalize
    kernel = kernel * blur_strength  # Scale by strength

    # Add identity to preserve some sharpness
    identity = torch.zeros_like(kernel)
    identity[kernel_size // 2, kernel_size // 2] = 1.0 - blur_strength
    kernel = kernel + identity

    # Apply convolution to each channel
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    # Pad images
    padding = kernel_size // 2
    images_padded = torch.nn.functional.pad(
        images, (padding, padding, padding, padding), mode="replicate"
    )

    # Apply blur
    blurred = torch.nn.functional.conv2d(images_padded, kernel, groups=3)

    return blurred


def apply_jpeg_compression(
    images: torch.Tensor, jpeg_quality_range: tuple[int, int], device: str = "cpu"
) -> torch.Tensor:
    """Simulate JPEG compression artifacts."""
    # Random quality for this step
    quality = torch.randint(
        jpeg_quality_range[0],
        jpeg_quality_range[1] + 1,
        (1,),
        device=device,
    ).item()

    if quality >= 95:  # Skip if very high quality
        return images

    # Simplified JPEG simulation: add block artifacts
    # Real JPEG is complex (DCT, quantization), so we approximate
    block_size = 8

    # Quantization strength based on quality (inverse relationship)
    quant_strength = (100 - quality) / 100.0 * 0.1  # 0-0.1 range

    if quant_strength < 0.01:
        return images

    # Split into blocks and add noise to simulate quantization
    N, C, H, W = images.shape

    # Add blockiness by downsampling and upsampling
    scale_factor = max(1, int(4 * quant_strength))
    if scale_factor > 1:
        # Downsample
        small = torch.nn.functional.interpolate(
            images,
            scale_factor=1.0 / scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        # Upsample back
        images = torch.nn.functional.interpolate(
            small, size=(H, W), mode="bilinear", align_corners=False
        )

    # Add slight quantization noise in blocks
    block_noise = torch.randn_like(images) * quant_strength * 0.05
    images = images + block_noise

    return images


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

    # Apply distortions
    print("\nApplying distortions:")

    if distortions["gaussian_blur"]:
        print("  - Gaussian blur")
        img_tensor = gaussian_blur_rgb(img_tensor, kernel_size=7, sigma=2.0)

    if distortions["webcam_effect"]:
        print("  - Cheap webcam effect")
        img_tensor = cheap_webcam_effect(img_tensor)

    if distortions["brightness"]:
        print(f"  - Brightness (range: {CAMERA_BRIGHTNESS_RANGE})")
        img_tensor = apply_brightness(img_tensor, CAMERA_BRIGHTNESS_RANGE)

    if distortions["noise"]:
        print(f"  - Gaussian noise (std range: {CAMERA_GAUSSIAN_NOISE_STD})")
        img_tensor = apply_noise(img_tensor, CAMERA_GAUSSIAN_NOISE_STD)

    if distortions["contrast"]:
        print(f"  - Contrast (range: {CAMERA_CONTRAST_RANGE})")
        img_tensor = apply_contrast(img_tensor, CAMERA_CONTRAST_RANGE)

    if distortions["motion_blur"]:
        print(
            f"  - Motion blur (kernel size: {MOTION_BLUR_KERNEL_SIZE}, strength: {MOTION_BLUR_STRENGTH_RANGE})"
        )
        img_tensor = apply_motion_blur(
            img_tensor,
            MOTION_BLUR_STRENGTH_RANGE,
            MOTION_BLUR_KERNEL_SIZE,
            args.device,
        )

    if distortions["jpeg_compression"]:
        print(f"  - JPEG compression (quality range: {JPEG_QUALITY_RANGE})")
        img_tensor = apply_jpeg_compression(img_tensor, JPEG_QUALITY_RANGE, args.device)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = input_path.parent / f"{input_path.stem}_{timestamp}.png"

    # Save result
    save_image(img_tensor, output_path)
    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
