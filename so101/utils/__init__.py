"""
SO-101 Utilities

Shared utilities for SO-101 simulation and real-world deployment.
Provides feature extraction and image-processing pipeline components
that are used by both the RL task (Isaac Lab) and offline CNN training.

Modules
-------
so101.utils.units
    Joint unit conversions (``JointUnitConverter``, ``JointParser``,
    ``from_robot_config``).  Used by both :mod:`so101_real` (via a
    backward-compat shim) and :mod:`so101_rl` scripts.
"""

__version__ = "0.1.0"
