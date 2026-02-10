"""Installation script for the 'so101_utils' python package."""

from setuptools import setup

setup(
    name="so101_utils",
    version="0.1.0",
    description="Shared utilities for SO-101 simulation and real-world deployment",
    author="Matthew Evans",
    packages=["so101_utils", "so101_utils.image_processing"],
    install_requires=["torch>=1.13.0"],
    python_requires=">=3.8",
)
