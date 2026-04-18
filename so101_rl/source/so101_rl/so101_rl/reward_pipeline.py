from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch

from so101_rl.configurations.cube import CUBE_WIDTH
from so101_rl.configurations.so101_env_params import (
    GateCfg,
    RewardCfg,
    DistanceRewardCfg,
    CloseGripperRewardCfg,
    EeLinearSpeedRewardCfg,
    AvoidBumpingCubeRewardCfg,
    WristRollPoseRewardCfg,
    ActionRewardCfg,
    _from_dict,
)
from so101_rl.metric_pipeline import StepContext

CfgT = TypeVar("CfgT", bound=RewardCfg)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class RewardStep(ABC, Generic[CfgT]):
    """Computes a scalar reward contribution for every environment."""

    name: str
    """Short identifier used for TensorBoard logging key. Must be set on each subclass."""

    requires_metrics: frozenset[str] = frozenset()
    """Metric keys from ``ctx.metrics`` that this step reads during ``compute``.
    Used by ``build_metric_pipeline`` to determine which metric steps to include."""

    requires_env_metrics: frozenset[str] = frozenset()
    """Keys from ``env.env_metrics`` (produced by :class:`EnvMetricStep`) that
    this step reads during :meth:`compute`."""

    def __init__(self, cfg: CfgT) -> None:
        self._cfg: CfgT = cfg
        self._gates: list[GateCfg] = cfg.gates
        self._fired: torch.Tensor | None = None
        """Per-env bool tensor tracking whether this fire_once step has already
        fired during the current episode.  Lazily initialised on first call."""
        self._prev_base: torch.Tensor | None = None
        """Per-env float tensor storing the previous step's unscaled base value
        for progressive reward modes.  ``None`` before the first call; set to
        NaN for reset environments so the first step of a new episode yields a
        zero delta rather than a spurious jump."""

    def _apply_fire_once(self, flag: torch.Tensor, env) -> torch.Tensor:
        """Return a mask that is True only for the first step ``flag`` is True
        for each environment this episode.  Updates internal state."""
        if self._fired is None:
            self._fired = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        first_time = flag & ~self._fired
        self._fired = self._fired | flag
        return first_time

    def reset_fired(self, env_ids) -> None:
        """Clear the fire_once state for *env_ids* (called at episode reset)."""
        if self._fired is not None:
            self._fired[env_ids] = False

    def _apply_mode_and_scale(self, base: torch.Tensor, env) -> torch.Tensor:
        """Apply the configured mode and scale to the unscaled *base* value.

        ``'absolute'``: returns ``base * scale``.
        ``'unsigned_progressive'``: returns ``max(0, \u0394base) * scale``; only
        improvements are rewarded, regressions yield zero.
        ``'signed_progressive'``: returns ``\u0394base * scale``; regressions yield
        negative reward, discouraging lift\u2192lower\u2192lift cycles.

        ``_prev_base`` is updated before gate masking is applied (in
        :meth:`RewardPipeline.compute`), so gated-off environments still
        maintain a valid baseline for the next step.
        """
        if self._cfg.mode == "absolute":
            return base * self._cfg.scale
        # Progressive modes: lazily initialise with NaN sentinels so the first
        # call (and any call immediately after a reset) yields a zero delta.
        if self._prev_base is None:
            self._prev_base = torch.full_like(base, float("nan"))
        has_prev = ~torch.isnan(self._prev_base)
        delta = torch.where(has_prev, base - self._prev_base, torch.zeros_like(base))
        self._prev_base = base.clone()
        if self._cfg.mode == "unsigned_progressive":
            return delta.clamp(min=0.0) * self._cfg.scale
        elif self._cfg.mode == "signed_progressive":
            return delta * self._cfg.scale
        raise ValueError(
            f"Unknown reward mode: {self._cfg.mode!r}. "
            "Must be 'absolute', 'unsigned_progressive', or 'signed_progressive'."
        )

    def reset_base_value(self, env_ids) -> None:
        """Reset the progressive baseline for *env_ids* (called at episode reset).

        Marks the stored baseline as NaN for the reset environments so that the
        first step of the new episode produces a zero delta rather than a
        potentially large jump from the previous episode's final state.
        """
        if self._prev_base is not None:
            self._prev_base[env_ids] = float("nan")

    def _prev(self, key: str, ctx: StepContext) -> torch.Tensor | None:
        """Return the previous-step value for *key* from ``ctx.prev_metrics``.

        Returns ``None`` when the key is absent (only on the very first call
        before :meth:`_reset_idx` has had a chance to populate ``prev_metrics``).
        """
        return ctx.prev_metrics.get(key)

    @abstractmethod
    def compute(self, ctx: StepContext) -> torch.Tensor:
        """Return the unscaled base value of shape ``(num_envs,)``.

        Scale and mode (``absolute`` / ``unsigned_progressive`` /
        ``signed_progressive``) are applied externally by
        :meth:`RewardPipeline.compute`.
        """
        ...


class TerminalRewardStep(RewardStep[CfgT]):
    """A :class:`RewardStep` that also signals episode termination.

    Subclasses must implement both :meth:`compute` (inherited) and
    :meth:`done`.  :class:`RewardPipeline` uses :meth:`done` to build
    the terminal mask returned by :meth:`RewardPipeline.get_dones`.
    """

    @abstractmethod
    def done(self, ctx: StepContext) -> torch.Tensor:
        """Return a bool tensor of shape ``(num_envs,)`` — True for environments
        that should terminate this step."""
        ...


class RewardPipeline:
    """Sums contributions from a sequence of :class:`RewardStep` objects.

    Steps are logged individually under ``Episode_Reward/<name>``.
    Only steps passed at construction are run — callers should filter by
    ``cfg.rewards.x.enabled`` once at startup.
    """

    def __init__(self, steps: list[RewardStep]) -> None:
        self.steps = steps

    @property
    def required_metric_keys(self) -> frozenset[str]:
        """Union of all ``requires_metrics`` declared by the steps in this pipeline,
        plus metric keys referenced by gate conditions."""
        step_keys = frozenset().union(*(s.requires_metrics for s in self.steps))
        gate_keys = frozenset(g.metric for s in self.steps for g in s._gates)
        return step_keys | gate_keys

    @property
    def terminal_steps(self) -> list[TerminalRewardStep]:
        """All :class:`TerminalRewardStep` instances in this pipeline."""
        return [s for s in self.steps if isinstance(s, TerminalRewardStep)]

    @staticmethod
    def _evaluate_gate_mask(gates: list[GateCfg], ctx: StepContext) -> torch.Tensor:
        """Return a bool mask of shape ``(num_envs,)`` — True where all gates pass.

        Each gate is resolved from ``ctx.metrics`` first, then ``ctx.env_metrics``.
        """
        env = ctx.env
        mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for gate in gates:
            if gate.metric in ctx.metrics:
                val = ctx.metrics[gate.metric].float()
            elif gate.metric in ctx.env_metrics:
                val = ctx.env_metrics[gate.metric].float()
            else:
                raise KeyError(
                    f"Gate metric '{gate.metric}' not found in ctx.metrics or "
                    f"ctx.env_metrics at runtime."
                )
            if val.dim() > 1:
                val = val.squeeze(-1)
            if gate.gt is not None:
                mask = mask & (val > gate.gt)
            elif gate.gte is not None:
                mask = mask & (val >= gate.gte)
            elif gate.lt is not None:
                mask = mask & (val < gate.lt)
            elif gate.lte is not None:
                mask = mask & (val <= gate.lte)
            elif gate.eq is not None:
                mask = mask & (val == gate.eq)
        return mask

    def get_done_reasons(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        """Return per-terminal-step done flags (float32, shape ``(num_envs,)``).

        Includes **all** terminal steps (even those with ``terminate=False``) so
        that milestone events (e.g. ``approach_phase_terminal``) are visible
        alongside hard-stop conditions.  Gate conditions are applied, matching
        the behaviour of :meth:`get_dones`.
        """
        reasons: dict[str, torch.Tensor] = {}
        for step in self.terminal_steps:
            done_flags = step.done(ctx)
            if step._gates:
                done_flags = done_flags & self._evaluate_gate_mask(step._gates, ctx)
            reasons[step.name] = done_flags.float()
        return reasons

    def get_dones(self, ctx: StepContext) -> torch.Tensor:
        """Return a bool tensor of shape ``(num_envs,)`` — True for any environment
        where at least one :class:`TerminalRewardStep` signals termination.

        Assumes metrics have already been computed for the current step.
        Gate conditions on terminal steps also suppress episode termination.
        Steps with ``cfg.terminate=False`` fire their reward but never end the episode.
        """
        env = ctx.env
        terminal = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for step in self.terminal_steps:
            if not step._cfg.terminate:
                continue
            done_flags = step.done(ctx)
            if step._gates:
                done_flags = done_flags & self._evaluate_gate_mask(step._gates, ctx)
            terminal = terminal | done_flags
        return terminal

    def reset_idx(self, env_ids) -> None:
        """Reset per-episode ``fire_once`` state for *env_ids*.

        Must be called from the environment's ``_reset_idx`` after every
        episode reset so that fire-once terminal steps can fire again in the
        next episode.
        """
        for step in self.steps:
            step.reset_base_value(env_ids)
            if step._cfg.fire_once:
                step.reset_fired(env_ids)

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if "log" not in env.extras:
            env.extras["log"] = {}
        if "per_env_log" not in env.extras:
            env.extras["per_env_log"] = {}

        total = torch.zeros(env.num_envs, device=env.device)
        totals_by_name: dict[str, torch.Tensor] = {}
        for step in self.steps:
            base = step.compute(ctx)
            rew = step._apply_mode_and_scale(base, env)
            if step._gates:
                rew = rew * self._evaluate_gate_mask(step._gates, ctx).float()
            if step._cfg.fire_once:
                rew = rew * step._apply_fire_once(rew != 0, env).float()
            totals_by_name.setdefault(
                step.name, torch.zeros(env.num_envs, device=env.device)
            )
            totals_by_name[step.name] = totals_by_name[step.name] + rew
            total += rew
        for name, t in totals_by_name.items():
            env.extras["log"][f"Episode_Reward/{name}"] = t.mean()
            env.extras["per_env_log"][f"Episode_Reward/{name}"] = t
        return total


# ---------------------------------------------------------------------------
# Reward steps — Primary
# ---------------------------------------------------------------------------


class StaticRewardStep(TerminalRewardStep[RewardCfg]):
    """Emits ``scale`` every step, unconditionally.

    Use ``gates`` to restrict to specific conditions, or ``fire_once: true``
    to award the bonus only on the first qualifying step per episode.
    Set ``terminate: true`` to end the episode when the (gated) condition fires.
    """

    name = "static"
    requires_metrics: frozenset[str] = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

    def done(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


class DistanceRewardStep(RewardStep[DistanceRewardCfg]):
    name = "distance"
    requires_metrics = frozenset({"normalized_grip_zone_cube_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return 1.0 - torch.exp(
            -self._cfg.distance_pressure
            * ctx.metrics["normalized_grip_zone_cube_distance"]
        )


class GripCubeRewardStep(RewardStep[RewardCfg]):
    name = "grip_cube"
    requires_metrics = frozenset(
        {"is_cube_in_grip_position", "gripper_cube_contact_force_magnitude"}
    )

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return (
            ctx.metrics["is_cube_in_grip_position"].float()
            * (ctx.metrics["gripper_cube_contact_force_magnitude"] > 0.0).float()
        )


class LiftCubeRewardStep(RewardStep[RewardCfg]):
    name = "lift_cube"
    requires_metrics = frozenset({"cube_lift_fraction"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["cube_lift_fraction"]


# ---------------------------------------------------------------------------
# Reward steps — Shaping
# ---------------------------------------------------------------------------


class GripperCubeAlignmentRewardStep(RewardStep[RewardCfg]):
    name = "gripper_cube_alignment"
    requires_metrics = frozenset({"is_cube_gripped", "gripper_cube_alignment"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return torch.maximum(
            ctx.metrics["is_cube_gripped"].float(),
            ctx.metrics["gripper_cube_alignment"],
        )


class CloseGripperRewardStep(RewardStep[CloseGripperRewardCfg]):
    name = "close_gripper"
    requires_metrics = frozenset({"is_cube_in_grip_position"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        gripper_pos = env.joint_pos[:, env._gripper_joint_idx]
        gripper_close_error = torch.abs(gripper_pos - self._cfg.close_target)
        fraction_to_target = 1.0 - (gripper_close_error / self._cfg.max_open).squeeze(
            -1
        )
        return ctx.metrics["is_cube_in_grip_position"].float() * fraction_to_target


# ---------------------------------------------------------------------------
# Reward steps — Smoothing
# ---------------------------------------------------------------------------


class ActionRewardStep(RewardStep[ActionRewardCfg]):
    name = "action"
    requires_metrics = frozenset()

    def __init__(self, cfg: ActionRewardCfg) -> None:
        super().__init__(cfg)
        self._joint_indices: list[int] | None = None

    def _resolve_indices(self, env) -> list[int]:
        if self._joint_indices is None:
            active = list(env.cfg.joints.active)
            if self._cfg.joints:
                missing = [j for j in self._cfg.joints if j not in active]
                if missing:
                    raise ValueError(
                        f"ActionRewardStep: joints {missing} not found in active joints {active}"
                    )
                self._joint_indices = [active.index(j) for j in self._cfg.joints]
            else:
                self._joint_indices = list(range(len(active)))
        return self._joint_indices

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.actions is None:
            return torch.zeros(env.num_envs, device=env.device)
        indices = self._resolve_indices(env)
        delta = env.actions[:, indices] - env.prev_actions[:, indices]
        return torch.sum(torch.abs(delta), dim=-1)


class EELinearSpeedRewardStep(RewardStep[EeLinearSpeedRewardCfg]):
    name = "ee_linear_speed"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        ee_lin_vel_w = env.robot.data.body_lin_vel_w[:, env._ee_body_idx[0], :]
        speed = torch.linalg.norm(ee_lin_vel_w, dim=-1)
        v_safe = self._cfg.safe_speed
        v_excess = torch.clamp(speed - v_safe, min=0.0)
        return v_excess + v_excess**2


class JointSpeedRewardStep(RewardStep[RewardCfg]):
    name = "joint_speed"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        joint_speed = torch.abs(env.joint_vel[:, env._dof_idx])
        return torch.sum(joint_speed**2, dim=-1)


# ---------------------------------------------------------------------------
# Reward steps — Terminal
# ---------------------------------------------------------------------------


class SuccessLiftFractionTerminalRewardStep(TerminalRewardStep[RewardCfg]):
    name = "success_lift_fraction_terminal"
    requires_metrics = frozenset({"is_success_lift_fraction_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_success_lift_fraction_terminal"].float()

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_success_lift_fraction_terminal"]


class SafetyTouchTableTerminalRewardStep(TerminalRewardStep[RewardCfg]):
    name = "safety_touch_table_terminal"
    requires_metrics = frozenset({"is_table_touched"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_table_touched"].float()

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_table_touched"]


class SafetyTouchTableRewardStep(RewardStep[RewardCfg]):
    name = "safety_touch_table"
    requires_metrics = frozenset({"is_table_touched"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_table_touched"].float()


class CubeOutOfRangeTerminalRewardStep(TerminalRewardStep[RewardCfg]):
    name = "cube_out_of_range_terminal"
    requires_metrics = frozenset({"is_cube_out_of_range"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_cube_out_of_range"].float()

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_cube_out_of_range"]


class ApproachPhaseTerminalRewardStep(TerminalRewardStep[RewardCfg]):
    name = "approach_phase_terminal"
    requires_metrics = frozenset({"approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["approach_phase_terminal"].float()

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["approach_phase_terminal"]


class GraspPhaseTerminalRewardStep(TerminalRewardStep[RewardCfg]):
    name = "grasp_phase_terminal"
    requires_metrics = frozenset({"grasp_phase_terminal", "approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return torch.logical_and(
            ctx.metrics["approach_phase_terminal"], ctx.metrics["grasp_phase_terminal"]
        ).float()

    def done(self, ctx: StepContext) -> torch.Tensor:
        flag = torch.logical_and(
            ctx.metrics["approach_phase_terminal"],
            ctx.metrics["grasp_phase_terminal"],
        )
        return flag


# ---------------------------------------------------------------------------
# Reward: Approach Phase
# ---------------------------------------------------------------------------


class ApproachDistanceRewardStep(RewardStep[RewardCfg]):
    name = "approach_distance"
    requires_metrics = frozenset({"approach_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["approach_distance"]


class ApproachAlignmentRewardStep(RewardStep[RewardCfg]):
    name = "approach_alignment"
    requires_metrics = frozenset({"approach_alignment"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["approach_alignment"]


class ApproachGripperPoseRewardStep(RewardStep[RewardCfg]):
    name = "approach_gripper_pose"
    requires_metrics = frozenset({"approach_gripper_pose"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["approach_gripper_pose"]


class ApproachPhaseRewardStep(RewardStep[RewardCfg]):
    name = "approach_phase"
    requires_metrics = frozenset({"approach_phase"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["approach_phase"]


class AvoidBumpingCubeRewardStep(RewardStep[AvoidBumpingCubeRewardCfg]):
    name = "avoid_bumping_cube"
    requires_metrics = frozenset(
        {"gripper_cube_contact_force_magnitude", "grip_zone_cube_distance"}
    )
    requires_env_metrics = frozenset({"dr_cube_scale"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        force_mag = ctx.metrics["gripper_cube_contact_force_magnitude"]
        # Use per-env DR cube scale (isotropic; X axis = side length scale factor)
        # so the proximity threshold tracks the actual cube size this episode.
        cube_width = ctx.env_metrics["dr_cube_scale"][:, 0] * CUBE_WIDTH
        cube_near_gz = ctx.metrics["grip_zone_cube_distance"] < (
            self._cfg.cube_widths * cube_width
        )
        return ((force_mag > 0.0) & ~cube_near_gz).float()


# ---------------------------------------------------------------------------
# Reward: Grasp Phase
# ---------------------------------------------------------------------------


class GraspPhaseRewardStep(RewardStep[RewardCfg]):
    name = "grasp_phase"
    requires_metrics = frozenset({"grasp_phase", "approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return (
            ctx.metrics["approach_phase_terminal"].float() * ctx.metrics["grasp_phase"]
        )


# ---------------------------------------------------------------------------
# Reward: Wrist roll pose
# ---------------------------------------------------------------------------


class WristRollPoseRewardStep(RewardStep[WristRollPoseRewardCfg]):
    """Rewards keeping the wrist roll joint at a target angle.

    Uses an exponential kernel: ``exp(-pressure * |q - target_rad|) * scale``.
    At the target the reward equals ``scale``; it decays sharply away from it.
    The ideal value for both top-down and side grasps is -90° (≈ -1.5708 rad).
    """

    name = "wrist_roll_pose"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        q = env.joint_pos[:, env._wrist_roll_joint_idx].squeeze(-1)
        error = torch.abs(q - self._cfg.target_rad)
        return torch.exp(-self._cfg.pressure * error)


class GoalZoneDistanceRewardStep(RewardStep[RewardCfg]):
    """Rewards moving the cube towards the goal zone.

    ``compute()`` returns ``goal_zone_distance = exp(-pressure * dist / cube_width)``
    from :class:`~so101_rl.metric_pipeline.GoalZoneCubeDistanceMetricStep`.
    This is in (0, 1] and is highest when the cube is at the goal zone.

    Combine with ``mode: unsigned_progressive`` and a positive scale to reward
    per-step improvements in proximity, or use ``mode: absolute`` with a
    positive scale for a dense potential-based reward.
    """

    name = "goal_zone_distance"
    requires_metrics = frozenset({"goal_zone_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["goal_zone_distance"]


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

REWARD_STEP_REGISTRY: dict[str, tuple[type[RewardStep[Any]], type[RewardCfg]]] = {
    "static": (StaticRewardStep, RewardCfg),
    "distance": (DistanceRewardStep, DistanceRewardCfg),
    "grip_cube": (GripCubeRewardStep, RewardCfg),
    "lift_cube": (LiftCubeRewardStep, RewardCfg),
    "gripper_cube_alignment": (GripperCubeAlignmentRewardStep, RewardCfg),
    "close_gripper": (CloseGripperRewardStep, CloseGripperRewardCfg),
    "action": (ActionRewardStep, ActionRewardCfg),
    "ee_linear_speed": (EELinearSpeedRewardStep, EeLinearSpeedRewardCfg),
    "joint_speed": (JointSpeedRewardStep, RewardCfg),
    "success_lift_fraction_terminal": (
        SuccessLiftFractionTerminalRewardStep,
        RewardCfg,
    ),
    "safety_touch_table_terminal": (SafetyTouchTableTerminalRewardStep, RewardCfg),
    "safety_touch_table": (SafetyTouchTableRewardStep, RewardCfg),
    "cube_out_of_range_terminal": (CubeOutOfRangeTerminalRewardStep, RewardCfg),
    "approach_phase_terminal": (ApproachPhaseTerminalRewardStep, RewardCfg),
    "grasp_phase_terminal": (GraspPhaseTerminalRewardStep, RewardCfg),
    "approach_distance": (ApproachDistanceRewardStep, RewardCfg),
    "approach_alignment": (ApproachAlignmentRewardStep, RewardCfg),
    "approach_gripper_pose": (ApproachGripperPoseRewardStep, RewardCfg),
    "approach_phase": (ApproachPhaseRewardStep, RewardCfg),
    "avoid_bumping_cube": (AvoidBumpingCubeRewardStep, AvoidBumpingCubeRewardCfg),
    "grasp_phase": (GraspPhaseRewardStep, RewardCfg),
    "wrist_roll_pose": (WristRollPoseRewardStep, WristRollPoseRewardCfg),
    "goal_zone_distance": (GoalZoneDistanceRewardStep, RewardCfg),
}


def build_reward_pipeline(cfg) -> RewardPipeline:
    """Construct the reward pipeline from ``cfg.rewards``.

    ``cfg.rewards`` is an ordered list of dicts, each with a ``type`` key plus
    any reward-specific parameters.  The same type may appear multiple times
    with different params or gates.  Disabled entries (``enabled: false``) are
    skipped entirely — their parameters are not validated.

    Args:
        cfg: The ``So101LiftCubeCfg`` instance (``env.cfg``).
    """
    steps: list[RewardStep] = []
    rewards = cfg.rewards
    # Migrate old named-map format (saved configs from before the list refactor).
    if isinstance(rewards, dict):
        rewards = [{"type": k, **v} for k, v in rewards.items()]
    for entry in rewards:
        type_name = entry.get("type")
        if type_name is None:
            raise ValueError(
                f"Reward list entry missing required 'type' key: {entry!r}"
            )
        params = {k: v for k, v in entry.items() if k != "type"}
        if not params.get("enabled", True):
            continue
        if type_name not in REWARD_STEP_REGISTRY:
            raise ValueError(
                f"Unknown reward type '{type_name}'. "
                f"Known types: {sorted(REWARD_STEP_REGISTRY)}"
            )
        step_cls, cfg_cls = REWARD_STEP_REGISTRY[type_name]
        instance_cfg = _from_dict(cfg_cls, params)
        steps.append(step_cls(cfg=instance_cfg))
    return RewardPipeline(steps)


def validate_gate_metrics(
    reward_pipeline: RewardPipeline,
    metric_pipeline,
    env_metric_pipeline,
) -> None:
    """Verify that every gate metric key is resolvable at runtime.

    Checks that each metric key referenced in a gate is either produced by
    ``metric_pipeline`` or provided by ``env_metric_pipeline``.  Raises
    ``ValueError`` at startup if any key is unknown, so failures are caught
    before the first training step.

    Args:
        reward_pipeline: The assembled :class:`RewardPipeline`.
        metric_pipeline: The assembled :class:`MetricPipeline`.
        env_metric_pipeline: The assembled :class:`EnvMetricPipeline`.
    """
    all_step_metric_keys: frozenset[str] = frozenset(
        key for step in metric_pipeline.steps for key in type(step).produces
    )
    all_env_metric_keys: frozenset[str] = env_metric_pipeline.provided_keys
    known_keys = all_step_metric_keys | all_env_metric_keys

    for step in reward_pipeline.steps:
        for gate in step._gates:
            if gate.metric not in known_keys:
                raise ValueError(
                    f"Gate on reward step '{step.name}' references metric key "
                    f"'{gate.metric}', but it is not produced by any MetricStep or "
                    f"EnvMetricStep. Known keys: {sorted(known_keys)}"
                )
