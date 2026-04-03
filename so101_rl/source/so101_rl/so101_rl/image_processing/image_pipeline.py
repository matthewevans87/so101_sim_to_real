"""
DEPRECATED: This module has been moved to so101.utils.image_processing

This file is kept for backward compatibility and re-exports from the new location.
Please update your imports to use: from so101.utils.image_processing import ...

To install so101:
    cd /path/to/so101_sim_to_real
    pip install -e .
"""

# Re-export everything from the new location
from so101.utils.image_processing import (
    ImagePipeline,
    ImagePipelineStep,
    JpegCompressionPipelineStep,
    MotionBlurPipelineStep,
    GaussianBlurPipelineStep,
    GaussianNoisePipelineStep,
    CheapWebcamEffectPipelineStep,
    CameraBrightnessPipelineStep,
    CameraContrastPipelineStep,
)

__all__ = [
    "ImagePipeline",
    "ImagePipelineStep",
    "JpegCompressionPipelineStep",
    "MotionBlurPipelineStep",
    "GaussianBlurPipelineStep",
    "GaussianNoisePipelineStep",
    "CheapWebcamEffectPipelineStep",
    "CameraBrightnessPipelineStep",
    "CameraContrastPipelineStep",
]
