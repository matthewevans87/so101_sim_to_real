"""
Feature Extraction Utilities

Provides feature extraction functionality for processing and analyzing visual data.
"""

from .feature_extraction import (
    VisionFeatureExtractor,
    ResNet18SpatialSoftmaxFeatureExtractor,
)

__all__ = [
    "VisionFeatureExtractor",
    "ResNet18SpatialSoftmaxFeatureExtractor",
]
