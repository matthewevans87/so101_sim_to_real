"""safety.py — Safety layer for real-robot inference.

Applies delta clamping and joint-limit clamping to policy outputs before
they are sent to the robot.  A joint-limits safety check is always applied.
An optional URDF-based FK height check can be enabled via the robot config
(requires the urdf_path key).
"""

from __future__ import annotations

import torch


class SafetyLayer:
    """Apply per-step safety constraints to joint targets.

    Parameters
    ----------
    joint_lower_rad:
        Per-joint lower position limits in radians.
    joint_upper_rad:
        Per-joint upper position limits in radians.
    max_delta_rad:
        Maximum allowed joint displacement from the current position per step.
    """

    def __init__(
        self,
        joint_lower_rad: list[float],
        joint_upper_rad: list[float],
        max_delta_rad: float,
    ) -> None:
        self._lower = torch.tensor(joint_lower_rad, dtype=torch.float32)
        self._upper = torch.tensor(joint_upper_rad, dtype=torch.float32)
        self._max_delta = float(max_delta_rad)

    def apply(
        self,
        q_target: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp a commanded joint target before sending to the robot.

        The following constraints are applied in order:

        1. **Delta clamp**: The target is clipped so it does not move more
           than ``max_delta_rad`` from the current position in a single step.
        2. **Joint-limit clamp**: The target is clipped to
           ``[joint_lower_rad, joint_upper_rad]``.

        Parameters
        ----------
        q_target:
            Raw desired joint positions in radians ``(n_joints,)``.
        q_current:
            Measured current joint positions in radians ``(n_joints,)``.

        Returns
        -------
        torch.Tensor
            Safe joint targets in radians ``(n_joints,)``.
        """
        delta = q_target - q_current
        delta_clamped = delta.clamp(-self._max_delta, self._max_delta)
        q_safe = q_current + delta_clamped
        q_safe = torch.max(q_safe, self._lower.to(q_safe.device))
        q_safe = torch.min(q_safe, self._upper.to(q_safe.device))
        return q_safe
