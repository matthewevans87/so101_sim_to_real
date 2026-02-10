# SO-101 Utilities

Shared utilities for SO-101 simulation and real-world deployment.

This package contains lightweight, reusable components that **do not depend on Isaac Lab**, making them suitable for use in:
- Simulation training (`so101_rl/`)
- Real robot control (`so101_controller/`)
- Offline processing scripts (`scripts/`)

## Packages

### `image_processing`

Image augmentation and distortion pipeline for domain randomization and sim-to-real transfer.

**Classes:**
- `ImagePipeline` - Main pipeline that chains multiple processing steps
- `ImagePipelineStep` - Abstract base class for pipeline steps
- `GaussianBlurPipelineStep` - Channel-wise Gaussian blur
- `CheapWebcamEffectPipelineStep` - Simulates low-quality webcam (downsample/upsample)
- `CameraBrightnessPipelineStep` - Random brightness adjustment
- `CameraContrastPipelineStep` - Random contrast adjustment
- `GaussianNoisePipelineStep` - Sensor noise simulation
- `MotionBlurPipelineStep` - Directional motion blur
- `JpegCompressionPipelineStep` - JPEG compression artifacts

**Example:**
```python
from so101_utils.image_processing import (
    ImagePipeline,
    GaussianBlurPipelineStep,
    GaussianNoisePipelineStep,
)

# Create pipeline
pipeline = ImagePipeline([
    GaussianBlurPipelineStep(kernel_size=7, sigma=2.0),
    GaussianNoisePipelineStep(noise_std_range=(0.01, 0.02)),
])

# Apply to images (N, 3, H, W) tensor
processed_images = pipeline.process(images)
```

## Requirements

- PyTorch
- No Isaac Lab dependencies

## Migration Note

The image processing code was previously located in `so101_rl/source/so101_rl/so101_rl/image_processing/`. The old location now re-exports from this package for backward compatibility, but new code should import directly from `so101_utils.image_processing`.
