"""Per-episode statistics accumulator for training telemetry and evaluation.

``EpisodeStatsPipeline`` maintains per-environment episode buffers across
physics steps, flushes one summary record to rolling deques when each
episode ends (termination or timeout), and exposes aggregates for
TensorBoard logging via :meth:`get_log_dict`.

The same class acts as the single source of truth for per-episode stats
during evaluation: after each env step call :meth:`get_completed_episodes`
to drain the staging buffer and obtain raw per-episode dicts.

Design notes
------------
* All per-env state lives in tensors on ``device`` so GPU→CPU copies happen
  only once per episode end (not every step).
* Rolling deque windows are Python-side (CPU) since they are read
  infrequently (only for logging) and contain scalars, not full tensors.
* Global milestone flags are Python bools — set once, never cleared.
* ``required_metric_keys`` is a ``frozenset[str]`` class attribute so that
  callers can pass it directly to ``build_metric_pipeline(..., extra_keys=...)``.
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from so101_rl.metric_pipeline import StepContext


class EpisodeStatsPipeline:
    """Accumulates per-episode statistics across physics steps.

    Parameters
    ----------
    num_envs:
        Number of parallel environments.
    device:
        Torch device string (e.g. ``"cuda:0"``).
    window_size:
        Rolling window length for rate/mean computations.  Defaults to 100
        episodes.  All deques share this limit.
    """

    # Keys from ctx.metrics consumed by this pipeline.  Pass this set as
    # ``extra_keys`` to ``build_metric_pipeline`` so the required MetricSteps
    # are included even when the reward pipeline doesn't need them.
    required_metric_keys: frozenset[str] = frozenset(
        {
            "is_success_lift_fraction_terminal",
            "is_cube_gripped",
            "is_table_touched",
            "cube_linear_velocity",
            "approach_phase_terminal",
            "grasp_phase_terminal",
            "cube_height_w",
        }
    )

    def __init__(
        self,
        num_envs: int,
        device: str,
        lift_height_threshold: float,
        window_size: int = 100,
    ) -> None:
        if num_envs <= 0:
            raise ValueError(f"num_envs must be a positive integer; got {num_envs!r}.")
        if lift_height_threshold <= 0.0:
            raise ValueError(
                f"lift_height_threshold must be > 0; got {lift_height_threshold!r}."
            )
        if window_size <= 0:
            raise ValueError(
                f"window_size must be a positive integer; got {window_size!r}."
            )

        self._num_envs = num_envs
        self._device = device
        self._lift_height_threshold = lift_height_threshold
        self._window_size = window_size

        # ------------------------------------------------------------------
        # Per-env episode buffers (all on device, reset each episode)
        # ------------------------------------------------------------------
        self._ever_lifted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._ever_dropped = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Step index of first lift within the episode; -1 = not yet lifted.
        self._lift_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        # Accumulated sum of cube linear-velocity norm while cube is not gripped.
        self._cube_bump_accum = torch.zeros(
            num_envs, dtype=torch.float32, device=device
        )
        # Per-env step counter within the current episode.
        self._episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)

        # ------------------------------------------------------------------
        # Training-global milestone flags (set once, never cleared)
        # ------------------------------------------------------------------
        self._first_approach_ever: bool = False
        self._first_grasp_ever: bool = False
        self._first_lift_ever: bool = False

        # ------------------------------------------------------------------
        # Rolling deques  (one entry per completed episode)
        # ------------------------------------------------------------------
        self._lift_deque: deque[float] = deque(maxlen=window_size)
        self._drop_deque: deque[float] = deque(maxlen=window_size)
        self._success_deque: deque[float] = deque(maxlen=window_size)
        self._cube_bump_deque: deque[float] = deque(maxlen=window_size)
        # Only appended when the cube was successfully lifted that episode.
        self._time_to_lift_deque: deque[int] = deque(maxlen=window_size)

        # ------------------------------------------------------------------
        # Staging buffer drained by get_completed_episodes()
        # ------------------------------------------------------------------
        self._completed_episodes: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        ctx: StepContext,
        terminated: torch.Tensor,
        time_out: torch.Tensor,
    ) -> None:
        """Update per-env buffers for one physics step and flush finished episodes.

        Must be called after the metric pipeline has populated ``ctx.metrics``
        for this step (i.e. after ``_compute_step_metrics()`` in the env).

        Parameters
        ----------
        ctx:
            Current :class:`~so101_rl.metric_pipeline.StepContext`.
        terminated:
            Boolean tensor of shape ``(num_envs,)`` — ``True`` when an episode
            ended due to a terminal condition this step.
        time_out:
            Boolean tensor of shape ``(num_envs,)`` — ``True`` when an episode
            ended due to the step budget being exhausted.
        """
        metrics = ctx.metrics

        is_success: torch.Tensor = metrics["is_success_lift_fraction_terminal"].bool()
        is_gripped: torch.Tensor = metrics["is_cube_gripped"].bool()
        is_table: torch.Tensor = metrics["is_table_touched"].bool()
        # cube_linear_velocity has shape (num_envs, 3)
        cube_vel: torch.Tensor = metrics["cube_linear_velocity"].float()
        approach_terminal: torch.Tensor = metrics["approach_phase_terminal"].bool()
        grasp_terminal: torch.Tensor = metrics["grasp_phase_terminal"].bool()
        # cube_height_w is guaranteed to be in ctx.metrics because
        # is_success_lift_fraction_terminal depends on it transitively.
        # Gate on is_gripped so that spawn-drop (cube falling from DR spawn height)
        # does not register as a lift — only counts when the robot is responsible.
        is_lifted: torch.Tensor = (
            metrics["cube_height_w"] >= self._lift_height_threshold
        ) & is_gripped

        # -- Global milestone flags (set once) --
        if not self._first_approach_ever and approach_terminal.any().item():
            self._first_approach_ever = True
        if not self._first_grasp_ever and grasp_terminal.any().item():
            self._first_grasp_ever = True
        # first_lift milestone = first time the cube reached full target height
        if not self._first_lift_ever and is_success.any().item():
            self._first_lift_ever = True

        # -- Record first-lift step index --
        # Use is_lifted (any height off table) rather than full-height success so
        # that lift_rate, drop_rate, and time_to_lift are meaningful even in
        # episodes that never reach the success height threshold.
        newly_lifted = is_lifted & ~self._ever_lifted  # (num_envs,)
        self._lift_step = torch.where(
            newly_lifted,
            self._episode_step,
            self._lift_step,
        )
        self._ever_lifted = self._ever_lifted | is_lifted

        # -- Cube bump accumulation (velocity norm when not gripped) --
        vel_norm = cube_vel.norm(dim=-1)  # (num_envs,)
        bump_contribution = torch.where(
            is_gripped, torch.zeros_like(vel_norm), vel_norm
        )
        self._cube_bump_accum = self._cube_bump_accum + bump_contribution

        # -- Drop detection: ever_lifted AND cube returns to table AND not gripped --
        dropped_this_step = self._ever_lifted & is_table & ~is_gripped
        self._ever_dropped = self._ever_dropped | dropped_this_step

        # -- Advance per-env step counter --
        self._episode_step = self._episode_step + 1

        # -- Flush finished episodes --
        done = terminated | time_out  # (num_envs,)
        if done.any().item():
            done_indices = done.nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(done_indices, int):
                done_indices = [done_indices]
            for i in done_indices:
                self._flush_episode(
                    i, terminated[i].item(), time_out[i].item(), is_success[i].item()
                )

    def reset_envs(self, env_ids) -> None:
        """Reset per-episode buffers for the given environment indices.

        Called from ``_reset_idx`` after ``super()._reset_idx()``.

        Parameters
        ----------
        env_ids:
            Sequence of environment indices to reset (int or LongTensor).
        """
        idx = torch.as_tensor(list(env_ids), dtype=torch.long, device=self._device)
        if idx.numel() == 0:
            return
        self._ever_lifted[idx] = False
        self._ever_dropped[idx] = False
        self._lift_step[idx] = -1
        self._cube_bump_accum[idx] = 0.0
        self._episode_step[idx] = 0

    def get_log_dict(self) -> dict[str, float]:
        """Return rolling-window aggregate scalars for TensorBoard logging.

        All returned values are Python ``float``\\s so they can be inserted
        directly into ``extras["log"]``.

        Keys
        ----
        ``Episode_Stats/lift_rate``
            Fraction of the last *window_size* episodes in which the cube was
            lifted above the height threshold at least once.
        ``Episode_Stats/drop_rate``
            Fraction of episodes in which the cube was lifted then dropped
            back to the table.
        ``Episode_Stats/success_rate``
            Fraction of episodes that ended with a successful lift (terminated
            by the success condition before timeout).
        ``Episode_Stats/cube_bump``
            Mean per-episode accumulated cube linear-velocity norm (m/s · steps)
            while the cube was not gripped.
        ``Episode_Stats/time_to_lift``
            Mean step index of the first lift within each episode, averaged
            over episodes that achieved a lift.  Absent (key omitted) when no
            lift has occurred yet in the current window.
        ``TrainingMilestone/first_approach``
            ``1.0`` once the first successful approach phase has been observed,
            ``0.0`` before.  Remains ``1.0`` permanently thereafter.
        ``TrainingMilestone/first_grasp``
            ``1.0`` once the first successful grasp phase has been observed.
        ``TrainingMilestone/first_lift``
            ``1.0`` once the first successful lift terminal has been observed.
        """
        log: dict[str, float] = {}

        if self._lift_deque:
            log["Episode_Stats/lift_rate"] = sum(self._lift_deque) / len(
                self._lift_deque
            )
        else:
            log["Episode_Stats/lift_rate"] = 0.0

        if self._drop_deque:
            log["Episode_Stats/drop_rate"] = sum(self._drop_deque) / len(
                self._drop_deque
            )
        else:
            log["Episode_Stats/drop_rate"] = 0.0

        if self._success_deque:
            log["Episode_Stats/success_rate"] = sum(self._success_deque) / len(
                self._success_deque
            )
        else:
            log["Episode_Stats/success_rate"] = 0.0

        if self._cube_bump_deque:
            log["Episode_Stats/cube_bump"] = sum(self._cube_bump_deque) / len(
                self._cube_bump_deque
            )
        else:
            log["Episode_Stats/cube_bump"] = 0.0

        if self._time_to_lift_deque:
            log["Episode_Stats/time_to_lift"] = sum(self._time_to_lift_deque) / len(
                self._time_to_lift_deque
            )
        # time_to_lift is omitted when no lift has occurred — avoids a
        # misleading 0.0 flat line at the start of training.

        log["TrainingMilestone/first_approach"] = float(self._first_approach_ever)
        log["TrainingMilestone/first_grasp"] = float(self._first_grasp_ever)
        log["TrainingMilestone/first_lift"] = float(self._first_lift_ever)

        return log

    def get_completed_episodes(self) -> list[dict]:
        """Drain and return raw per-episode stat dicts since the last call.

        Each dict contains:

        * ``"lifted"``       — bool: cube reached lift threshold this episode.
        * ``"dropped"``      — bool: cube was lifted then dropped back to table.
        * ``"success"``      — bool: episode ended with a successful lift terminal.
        * ``"timed_out"``    — bool: episode ended by timeout (not a terminal condition).
        * ``"cube_bump"``    — float: accumulated cube velocity norm (not gripped).
        * ``"lift_step"``    — int or None: step of first lift; None if no lift occurred.
        * ``"episode_steps"``— int: total number of steps in the episode.

        The staging buffer is cleared after each call.  For training-time
        TensorBoard logging use :meth:`get_log_dict` instead (rolling window
        aggregates are cheaper than draining this list every step).

        Returns
        -------
        list[dict]
            One dict per completed episode across all environments since the
            last call.
        """
        result = self._completed_episodes
        self._completed_episodes = []
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flush_episode(
        self,
        env_idx: int,
        terminated: bool,
        timed_out: bool,
        is_success: bool,
    ) -> None:
        """Flush one completed episode into the rolling deques and staging buffer."""
        lifted = bool(self._ever_lifted[env_idx].item())
        dropped = bool(self._ever_dropped[env_idx].item())
        # An episode counts as a success only when it ended via the terminal
        # condition AND the success flag was True at that step.
        success = bool(terminated and is_success)
        cube_bump = float(self._cube_bump_accum[env_idx].item())
        lift_step_val = int(self._lift_step[env_idx].item())
        lift_step: int | None = lift_step_val if lifted else None
        episode_steps = int(self._episode_step[env_idx].item())

        self._lift_deque.append(float(lifted))
        self._drop_deque.append(float(dropped))
        self._success_deque.append(float(success))
        self._cube_bump_deque.append(cube_bump)
        if lifted and lift_step is not None:
            self._time_to_lift_deque.append(lift_step)

        self._completed_episodes.append(
            {
                "lifted": lifted,
                "dropped": dropped,
                "success": success,
                "timed_out": timed_out,
                "cube_bump": cube_bump,
                "lift_step": lift_step,
                "episode_steps": episode_steps,
            }
        )
