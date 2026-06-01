"""policies.py — Swappable policy implementations for real-robot inference.

The :class:`Policy` protocol defines the minimal callable interface expected
by :class:`so101_real.controller.InferenceLoop`: given an observation tensor,
return an action tensor in the normalized space ``[-1, 1]`` matching the
training action contract.

Two implementations are provided here:

* :class:`StaticPositionPolicy` — ignores the observation and emits a fixed
  target joint position every tick.  Useful for hardware sanity-checks
  ("send sim zeros → arm goes home"), end-to-end calibration verification,
  and as a safe-state policy.

* The trained MLP policy lives in :mod:`so101_real.policy` (see
  :func:`so101_real.policy.load_policy`) and trivially satisfies this
  protocol as well.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .units import JointUnitConverter


class Policy(Protocol):
    """Callable returning a normalized action tensor in ``[-1, 1]``.

    The loop maps the returned action to canonical-radian joint targets via
    ``q = lower + 0.5 * (a + 1) * (upper - lower)`` before sending it to
    the robot.  Implementations must therefore emit values in ``[-1, 1]``.
    """

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        ...


class StaticPositionPolicy:
    """Emit a fixed joint-position target every tick, regardless of observation.

    The target is supplied in canonical radians.  It is encoded into normalized
    action space using the supplied joint bounds so that the inverse mapping
    performed inside the inference loop round-trips back to the requested
    canonical-radian target exactly.

    Parameters
    ----------
    target_q_rad:
        Desired joint positions in canonical radians, shape ``(n_joints,)``.
    joint_lower_rad:
        Per-joint lower bounds in canonical radians, shape ``(n_joints,)``.
    joint_upper_rad:
        Per-joint upper bounds in canonical radians, shape ``(n_joints,)``.

    Raises
    ------
    ValueError
        If any joint upper bound is not strictly greater than its lower bound,
        or if the three input tensors do not share the same shape.
    """

    def __init__(
        self,
        target_q_rad: torch.Tensor,
        joint_lower_rad: torch.Tensor,
        joint_upper_rad: torch.Tensor,
    ) -> None:
        if (
            target_q_rad.shape != joint_lower_rad.shape
            or target_q_rad.shape != joint_upper_rad.shape
        ):
            raise ValueError(
                "StaticPositionPolicy: target_q_rad, joint_lower_rad, and "
                "joint_upper_rad must share the same shape; got "
                f"{tuple(target_q_rad.shape)}, {tuple(joint_lower_rad.shape)}, "
                f"{tuple(joint_upper_rad.shape)}."
            )
        # JointUnitConverter validates upper > lower per joint and provides the
        # canonical-radian → normalized action mapping used by the inference
        # loop (so this round-trips exactly).
        converter = JointUnitConverter(
            joint_names=[f"j{i}" for i in range(target_q_rad.shape[-1])],
            lower_rad=joint_lower_rad.detach().cpu().tolist(),
            upper_rad=joint_upper_rad.detach().cpu().tolist(),
        )
        action = torch.clamp(
            converter.canonical_to_normalized(target_q_rad.detach().float()),
            -1.0,
            1.0,
        )
        # Cache the (1, n_act) action so __call__ is allocation-free per tick.
        self._action = action.unsqueeze(0)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        # Ignore obs; return a (1, n_act) action tensor on obs's device.
        return self._action.to(obs.device)
