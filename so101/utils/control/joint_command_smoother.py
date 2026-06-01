"""joint_command_smoother.py — Shared action-processing pipeline for SO-101.

Implements the exact ordering used by ``so101_real/controller.py::_tick`` to
map normalised policy actions to safe joint-radian targets:

1. **normalised → canonical**:  ``q = lower + 0.5*(a + 1)*(upper - lower)``
2. **EMA smoothing** (opt-in):  ``q_smooth = α·q + (1−α)·q_smooth_prev``
   Applied only to joints listed in ``ema_mask``.
3. **Per-step delta clamp** (opt-in): limit movement to ``max_delta_rad`` per
   step, measured relative to the *current* joint positions.
   Applied only to joints listed in ``clamp_mask``.
4. **Joint-limit clamp**:       hard-clip to ``[lower_rad, upper_rad]``.

Using a single shared implementation eliminates the sim/real action-pipeline
divergence so every policy trained in sim sees exactly the same action
processing as the real robot.

Usage (sim — batched)::

    smoother = JointCommandSmoother(
        lower_rad, upper_rad, ema_alpha=0.9, max_delta_rad=0.13,
        ema_mask=torch.tensor([True, True, True, True, True, False]),
        clamp_mask=torch.tensor([True, True, True, True, True, False]),
    )
    q_safe = smoother.step(actions, joint_pos[:, dof_idx])  # (N, n_joints)
    smoother.reset(joint_pos[:, dof_idx], env_ids=env_ids_t)

Usage (real — unbatched)::

    smoother = JointCommandSmoother(
        lower_rad, upper_rad, ema_alpha=0.9, max_delta_rad=0.13,
        ema_mask=ema_mask, clamp_mask=clamp_mask,
    )
    q_safe = smoother.step(action, q_meas)  # (n_joints,)
    smoother.reset_episode()  # between episodes
"""

from __future__ import annotations

from typing import Optional

import torch


class JointCommandSmoother:
    """Shared action-processing pipeline for SO-101 sim and real.

    All operations are element-wise, so the smoother handles any leading batch
    dimensions: ``(n_joints,)`` for the real robot and ``(num_envs, n_joints)``
    for the parallel sim.

    Parameters
    ----------
    lower_rad : torch.Tensor
        Per-joint lower position limits in canonical radians, shape
        ``(n_joints,)``.
    upper_rad : torch.Tensor
        Per-joint upper position limits in canonical radians, shape
        ``(n_joints,)``.
    ema_alpha : float
        EMA coefficient in ``(0, 1]``.  ``1.0`` = no smoothing (direct policy
        output).  Only used for joints in ``ema_mask``.
    max_delta_rad : float
        Maximum per-joint displacement from the *current* position per step
        (radians).  Must be ``> 0``.  Only used for joints in ``clamp_mask``.
    ema_mask : torch.Tensor or None
        Boolean mask of shape ``(n_joints,)``.  ``True`` = apply EMA smoothing
        to this joint (opt-in).  ``None`` = no EMA for any joint.
    clamp_mask : torch.Tensor or None
        Boolean mask of shape ``(n_joints,)``.  ``True`` = apply delta clamping
        to this joint (opt-in).  ``None`` = no delta clamping for any joint.
    """

    def __init__(
        self,
        lower_rad: torch.Tensor,
        upper_rad: torch.Tensor,
        ema_alpha: float,
        max_delta_rad: Optional[float],
        ema_mask: Optional[torch.Tensor] = None,
        clamp_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1]; got {ema_alpha}.")
        if clamp_mask is not None:
            if max_delta_rad is None:
                raise ValueError(
                    "max_delta_rad must be set when clamp_mask is provided."
                )
            if max_delta_rad <= 0.0:
                raise ValueError(f"max_delta_rad must be > 0; got {max_delta_rad}.")
        span = upper_rad - lower_rad
        if not torch.all(span > 0):
            raise ValueError(
                "Every joint must satisfy upper_rad > lower_rad; "
                f"got lower={lower_rad.tolist()}, upper={upper_rad.tolist()}."
            )
        n_joints = lower_rad.shape[0]
        for mask_name, mask in [("ema_mask", ema_mask), ("clamp_mask", clamp_mask)]:
            if mask is not None and mask.shape != (n_joints,):
                raise ValueError(
                    f"{mask_name} must have shape ({n_joints},); "
                    f"got {tuple(mask.shape)}."
                )

        # Stored on CPU; transferred lazily via `.to()` on every step() call.
        self._lower = lower_rad.float().cpu()
        self._upper = upper_rad.float().cpu()
        self._alpha = float(ema_alpha)
        self._max_delta: Optional[float] = (
            float(max_delta_rad) if max_delta_rad is not None else None
        )
        self._ema_mask: Optional[torch.Tensor] = (
            ema_mask.bool().cpu() if ema_mask is not None else None
        )
        self._clamp_mask: Optional[torch.Tensor] = (
            clamp_mask.bool().cpu() if clamp_mask is not None else None
        )

        # Lazily initialised on the first step() call (only when ema_mask is set).
        self._ema_state: Optional[torch.Tensor] = None

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def step(
        self,
        action_normalized: torch.Tensor,
        q_current_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Run one action through the full pipeline.

        Parameters
        ----------
        action_normalized : torch.Tensor
            Normalised policy output in ``[-1, 1]``, shape ``(..., n_joints)``.
            Values slightly outside ``[-1, 1]`` are tolerated; step 4 (joint-
            limit clamp) provides the hard bound.
        q_current_rad : torch.Tensor
            Current joint positions in canonical radians, shape
            ``(..., n_joints)``.  Used as the reference for the delta clamp
            (step 3), matching ``SafetyLayer.apply(q_smooth, q_meas)`` in
            ``so101_real/controller.py``.

        Returns
        -------
        torch.Tensor
            Safe joint targets in canonical radians, shape ``(..., n_joints)``.
        """
        dev = action_normalized.device
        dtype = action_normalized.dtype
        lower = self._lower.to(device=dev, dtype=dtype)
        upper = self._upper.to(device=dev, dtype=dtype)

        # 1. Normalised → canonical radian
        q_target = lower + 0.5 * (action_normalized + 1.0) * (upper - lower)

        # 2. EMA smoothing (opt-in: only joints where ema_mask is True)
        if self._ema_mask is not None:
            if self._ema_state is None:
                self._ema_state = q_target.clone()
            else:
                self._ema_state = (
                    self._alpha * q_target + (1.0 - self._alpha) * self._ema_state
                )
            ema_mask = self._ema_mask.to(device=dev)
            q_smooth = torch.where(ema_mask, self._ema_state, q_target)
        else:
            q_smooth = q_target

        # 3. Delta clamp (opt-in: only joints where clamp_mask is True)
        if self._clamp_mask is not None:
            delta = q_smooth - q_current_rad
            delta_clamped = delta.clamp(-self._max_delta, self._max_delta)
            q_after_clamp = q_current_rad + delta_clamped
            clamp_mask = self._clamp_mask.to(device=dev)
            q_out = torch.where(clamp_mask, q_after_clamp, q_smooth)
        else:
            q_out = q_smooth

        # 4. Joint-limit clamp (applied to all joints)
        q_safe = torch.max(q_out, lower)
        q_safe = torch.min(q_safe, upper)

        return q_safe

    # ── State management ──────────────────────────────────────────────────────

    def reset(
        self,
        q_init_rad: torch.Tensor,
        env_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Reset EMA state for specific environments (or all if env_ids is None).

        Call this from the sim's ``_reset_idx`` after writing the new joint
        positions so the smoother's EMA state reflects the actual post-reset
        joint positions.

        Parameters
        ----------
        q_init_rad : torch.Tensor
            Initial joint positions in canonical radians.

            * When *env_ids* is ``None``: shape ``(..., n_joints)`` — replaces
              the entire state.
            * When *env_ids* is given: shape ``(len(env_ids), n_joints)`` — only
              the specified rows are updated.
        env_ids : torch.Tensor or None
            1-D integer tensor of environment indices to reset.  If ``None``,
            the full EMA state is replaced by *q_init_rad*.
        """
        if env_ids is None:
            self._ema_state = q_init_rad.clone()
        elif self._ema_state is not None:
            self._ema_state[env_ids] = q_init_rad.clone()
        # else: state not yet initialised and we're resetting a subset —
        # leave _ema_state = None; lazy init on the first step() call handles it.

    def reset_episode(self) -> None:
        """Signal the start of a new episode (real-robot use).

        Clears the EMA state; it will be lazily re-initialised from the first
        policy target on the next ``step()`` call.  Use this in real-robot
        deployments after the arm reaches its start pose.
        """
        self._ema_state = None
