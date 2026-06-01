"""Action-processing pipeline utilities.

:class:`JointCommandSmoother` implements the sim/real-shared action pipeline:
normalized policy output → canonical radian targets with EMA smoothing,
per-step delta clamping, and joint-limit clamping.
"""

from .joint_command_smoother import JointCommandSmoother

__all__ = ["JointCommandSmoother"]
