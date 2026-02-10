from abc import ABC, abstractmethod

from torch import device
import torch.nn.functional as F

import torch

class ImagePipelineStep(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def process(self, images: torch.Tensor) -> torch.Tensor:
        """
        Docstring for process
        
        :param images: A tensor of images to be processed with shape (batch_size, channels, height, width)
        :type images: torch.Tensor
        :return: The tensor of images after processing
        :rtype: torch.Tensor
        """
        pass

class JpegCompressionPipelineStep(ImagePipelineStep):
    def __init__(self, quality_range: tuple[int, int] = (30, 90), device: str = "cuda"):
        super().__init__()
        self.quality_range = quality_range
        self.device = device

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """Simulate JPEG compression artifacts."""
        # Random quality for this step
        quality = torch.randint(
            self.quality_range[0],
            self.quality_range[1] + 1,
            (1,),
            device=self.device,
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

class MotionBlurPipelineStep(ImagePipelineStep):
    def __init__(self, 
                 motion_blur_strength_range: tuple[float, float] = (0.1, 0.2),
                 motion_blur_kernel_size: int = 5,
                 device: str = "cuda"):
        super().__init__()
        self.motion_blur_strength_range = motion_blur_strength_range
        self.motion_blur_kernel_size = motion_blur_kernel_size
        self.device = device

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """Apply directional motion blur to simulate camera motion."""
        # Random blur strength for this step
        blur_strength = (
            torch.rand(1, device=self.device)
            * (self.motion_blur_strength_range[1] - self.motion_blur_strength_range[0])
            + self.motion_blur_strength_range[0]
        )

        if blur_strength < 0.01:  # Skip if very weak
            return images

        kernel_size = self.motion_blur_kernel_size

        # Random blur direction: horizontal, vertical, or diagonal
        blur_type = torch.randint(0, 4, (1,), device=self.device).item()

        # Create blur kernel
        kernel = torch.zeros((kernel_size, kernel_size), device=self.device)
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
    
class GaussianBlurPipelineStep(ImagePipelineStep):
    def __init__(self, 
    kernel_size: int = 7,
    sigma: float = 2.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """Apply a channel-wise Gaussian blur to an (N, 3, H, W) tensor."""
        device = images.device
        # 1D Gaussian
        x = torch.arange(self.kernel_size, device=device) - (self.kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-0.5 * (x / self.sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        # Outer product → 2D kernel
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]  # (K, K)
        kernel_2d = kernel_2d.expand(3, 1, self.kernel_size, self.kernel_size)  # (C, 1, K, K)

        # Depthwise convolution: one kernel per channel
        return F.conv2d(
            images,
            kernel_2d,
            padding=self.kernel_size // 2,
            groups=3,
        )

class GaussianNoisePipelineStep(ImagePipelineStep):
    def __init__(self, noise_std_range: tuple[float, float] = (0.01, 0.05), device: str = "cuda"):
        super().__init__()
        self.noise_std_range = noise_std_range
        self.device = device

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to simulate sensor noise."""
        # Random noise std for this step
        noise_std = (
            torch.rand(1, device=self.device)
            * (self.noise_std_range[1] - self.noise_std_range[0])
            + self.noise_std_range[0]
        )

        if noise_std < 0.005:  # Skip if very low
            return images

        noise = torch.randn_like(images) * noise_std
        return images + noise
    

class CheapWebcamEffectPipelineStep(ImagePipelineStep):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """
        rgb: (N, 3, H, W) in [0,1]
        Mimic a low-res sensor + resize.
        """
        N, C, H, W = images.shape

        # pick a downscale factor (e.g., 0.4–0.7)
        scale = 0.4 + 0.3 * torch.rand(1, device=images.device).item()
        H_low = int(H * scale)
        W_low = int(W * scale)

        # Downsample (area or bilinear)
        low_res = F.interpolate(images, size=(H_low, W_low), mode="area")

        # Upsample back
        upsampled = F.interpolate(
            low_res, size=(H, W), mode="bilinear", align_corners=False
        )

        return upsampled
    
class CameraBrightnessPipelineStep(ImagePipelineStep):
    def __init__(self, brightness_range: tuple[float, float] = (0.85, 1.15), device: str = "cuda"):
        super().__init__()
        self.brightness_range = brightness_range
        self.device = device

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """Randomly adjust brightness."""
        brightness_factor = (
            torch.rand(1, device=self.device)
            * (self.brightness_range[1] - self.brightness_range[0])
            + self.brightness_range[0]
        )
        return images * brightness_factor
    
class CameraContrastPipelineStep(ImagePipelineStep):
    def __init__(self, contrast_range: tuple[float, float] = (0.8, 1.2), device: str = "cuda"):
        super().__init__()
        self.contrast_range = contrast_range
        self.device = device

    def process(self, images: torch.Tensor) -> torch.Tensor:
        """Randomly adjust contrast."""
        contrast_factor = (
            torch.rand(1, device=self.device)
            * (self.contrast_range[1] - self.contrast_range[0])
            + self.contrast_range[0]
        )
        mean = images.mean(dim=[2, 3], keepdim=True)
        return (images - mean) * contrast_factor + mean
    

class ImagePipeline:
    def __init__(self, steps: list[ImagePipelineStep]):
        self.steps = steps

    def process(self, images: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            images = step.process(images)
        return images
