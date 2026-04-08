from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from so101_rl.configurations.cube import CUBE_WIDTH
from so101_rl.configurations.so101_env_params import GateCfg
from so101_rl.metric_pipeline import StepContext


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class RewardStep(ABC):
    """Computes a scalar reward contribution for every environment."""

    name: str
    """Short identifier used for TensorBoard logging key. Must be set on each subclass."""

    requires_metrics: frozenset[str] = frozenset()
    """Metric keys from ``ctx.metrics`` that this step reads during ``compute``.
    Used by ``build_metric_pipeline`` to determine which metric steps to include."""

    requires_env_metrics: frozenset[str] = frozenset()
    """Keys from ``env.env_metrics`` (produced by :class:`EnvMetricStep`) that
    this step reads during :meth:`compute`."""

    def __init__(self, gates: list[GateCfg] | None = None) -> None:
        self._gates: list[GateCfg] = gates if gates is not None else []

    def _prev(self, key: str, ctx: StepContext) -> torch.Tensor | None:
        """Return the previous-step value for *key* from ``ctx.prev_metrics``.

        Returns ``None`` when the key is absent (only on the very first call
        before :meth:`_reset_idx` has had a chance to populate ``prev_metrics``).
        """
        return ctx.prev_metrics.get(key)

    @abstractmethod
    def compute(self, ctx: StepContext) -> torch.Tensor:
        """Return reward tensor of shape ``(num_envs,)``."""
        ...


class TerminalRewardStep(RewardStep):
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

    def get_dones(self, ctx: StepContext) -> torch.Tensor:
        """Return a bool tensor of shape ``(num_envs,)`` — True for any environment
        where at least one :class:`TerminalRewardStep` signals termination.

        Assumes metrics have already been computed for the current step.
        Gate conditions on terminal steps also suppress episode termination.
        """
        env = ctx.env
        terminal = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for step in self.terminal_steps:
            done_flags = step.done(ctx)
            if step._gates:
                done_flags = done_flags & self._evaluate_gate_mask(step._gates, ctx)
            terminal = terminal | done_flags
        return terminal

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if "log" not in env.extras:
            env.extras["log"] = {}
        if "per_env_log" not in env.extras:
            env.extras["per_env_log"] = {}

        total = torch.zeros(env.num_envs, device=env.device)
        for step in self.steps:
            rew = step.compute(ctx)
            if step._gates:
                rew = rew * self._evaluate_gate_mask(step._gates, ctx).float()
            env.extras["log"][f"Episode_Reward/{step.name}"] = rew.mean()
            env.extras["per_env_log"][f"Episode_Reward/{step.name}"] = rew
            total += rew
        return total


# ---------------------------------------------------------------------------
# Reward steps — Primary
# ---------------------------------------------------------------------------


class DistanceRewardStep(RewardStep):
    name = "rew_distance"
    requires_metrics = frozenset({"grip_zone_cube_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.cfg.rewards.distance.mode == "progressive":
            prev = self._prev("grip_zone_cube_distance", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            # Improvement = reduction in distance; clamp so regression is not penalized.
            delta = (prev - ctx.metrics["grip_zone_cube_distance"]).clamp(min=0.0)
            return delta * env.cfg.rewards.distance.scale
        grip_zone_dist = 1 - (
            torch.exp(
                -env.cfg.rewards.distance.distance_pressure
                * ctx.metrics["grip_zone_cube_distance"]
            )
        )
        return grip_zone_dist * env.cfg.rewards.distance.scale


class GripCubeRewardStep(RewardStep):
    name = "rew_grip_cube"
    requires_metrics = frozenset(
        {"is_cube_in_grip_position", "gripper_cube_contact_force_magnitude"}
    )

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return (
            ctx.metrics["is_cube_in_grip_position"]
            * (ctx.metrics["gripper_cube_contact_force_magnitude"] > 0.0)
            * env.cfg.rewards.grip_cube.scale
        )


class LiftCubeRewardStep(RewardStep):
    name = "rew_lift_cube"
    requires_metrics = frozenset({"cube_lift_fraction"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.cfg.rewards.lift_cube.mode == "progressive":
            prev = self._prev("cube_lift_fraction", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            delta = (ctx.metrics["cube_lift_fraction"] - prev).clamp(min=0.0)
            return delta * env.cfg.rewards.lift_cube.scale
        return ctx.metrics["cube_lift_fraction"] * env.cfg.rewards.lift_cube.scale


# ---------------------------------------------------------------------------
# Reward steps — Shaping
# ---------------------------------------------------------------------------


class GripperCubeAlignmentRewardStep(RewardStep):
    name = "rew_gripper_cube_alignment"
    requires_metrics = frozenset({"is_cube_gripped", "gripper_cube_alignment"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return (
            torch.maximum(
                ctx.metrics["is_cube_gripped"],
                ctx.metrics["gripper_cube_alignment"],
            )
            * env.cfg.rewards.gripper_cube_alignment.scale
        )


class CloseGripperRewardStep(RewardStep):
    name = "rew_close_gripper"
    requires_metrics = frozenset({"is_cube_in_grip_position"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        gripper_pos = env.joint_pos[:, env._gripper_joint_idx]
        gripper_close_error = torch.abs(
            gripper_pos - env.cfg.rewards.close_gripper.close_target
        )
        fraction_to_target = 1.0 - (
            gripper_close_error / env.cfg.rewards.close_gripper.max_open
        ).squeeze(-1)
        return (
            ctx.metrics["is_cube_in_grip_position"]
            * fraction_to_target
            * env.cfg.rewards.close_gripper.scale
        )


# ---------------------------------------------------------------------------
# Reward steps — Smoothing
# ---------------------------------------------------------------------------


class ActionRewardStep(RewardStep):
    name = "rew_action"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.actions is None:
            return torch.zeros(env.num_envs, device=env.device)
        delta = env.actions - env.prev_actions
        return env.cfg.rewards.action.scale * torch.sum(torch.abs(delta), dim=-1)


class EELinearSpeedRewardStep(RewardStep):
    name = "rew_ee_linear_speed"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        ee_lin_vel_w = env.robot.data.body_lin_vel_w[:, env._ee_body_idx[0], :]
        speed = torch.linalg.norm(ee_lin_vel_w, dim=-1)
        v_safe = env.cfg.rewards.ee_linear_speed.safe_speed
        v_excess = torch.clamp(speed - v_safe, min=0.0)
        return env.cfg.rewards.ee_linear_speed.scale * (v_excess + v_excess**2)


class JointSpeedRewardStep(RewardStep):
    name = "rew_joint_speed"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        joint_speed = torch.abs(env.joint_vel[:, env._dof_idx])
        return env.cfg.rewards.joint_speed.scale * torch.sum(joint_speed**2, dim=-1)


# ---------------------------------------------------------------------------
# Reward steps — Terminal
# ---------------------------------------------------------------------------


class SuccessLiftFractionTerminalRewardStep(TerminalRewardStep):
    name = "rew_success_lift_fraction_terminal"
    requires_metrics = frozenset({"is_success_lift_fraction_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = ctx.metrics["is_success_lift_fraction_terminal"]
        return torch.where(
            flag >= 1.0,
            torch.full_like(
                flag.float(), env.cfg.rewards.success_lift_fraction_terminal.scale
            ),
            torch.zeros(env.num_envs, device=env.device),
        )

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_success_lift_fraction_terminal"]


class SafetyTouchTableTerminalRewardStep(TerminalRewardStep):
    name = "rew_safety_touch_table_terminal"
    requires_metrics = frozenset({"is_table_touched"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.where(
            ctx.metrics["is_table_touched"],
            torch.tensor(
                env.cfg.rewards.safety_touch_table_terminal.scale,
                device=env.device,
                dtype=torch.float32,
            ),
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
        )

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_table_touched"]


class SafetyTouchTableRewardStep(RewardStep):
    name = "rew_safety_touch_table"
    requires_metrics = frozenset({"is_table_touched"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.where(
            ctx.metrics["is_table_touched"],
            torch.tensor(
                env.cfg.rewards.safety_touch_table.scale,
                device=env.device,
                dtype=torch.float32,
            ),
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
        )


class CubeOutOfRangeTerminalRewardStep(TerminalRewardStep):
    name = "rew_cube_out_of_range_terminal"
    requires_metrics = frozenset({"is_cube_out_of_range"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return (
            ctx.metrics["is_cube_out_of_range"].float()
            * env.cfg.rewards.cube_out_of_range_terminal.scale
        )

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_cube_out_of_range"]


class ApproachPhaseTerminalRewardStep(TerminalRewardStep):
    name = "rew_approach_phase_terminal"
    requires_metrics = frozenset({"approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = ctx.metrics["approach_phase_terminal"]
        return flag.float() * env.cfg.rewards.approach_phase_terminal.scale

    def done(self, ctx: StepContext) -> torch.Tensor:
        flag = ctx.metrics["approach_phase_terminal"]
        return flag


class GraspPhaseTerminalRewardStep(TerminalRewardStep):
    name = "rew_grasp_phase_terminal"
    requires_metrics = frozenset({"grasp_phase_terminal", "approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = torch.logical_and(
            ctx.metrics["approach_phase_terminal"], ctx.metrics["grasp_phase_terminal"]
        )
        return flag.float() * env.cfg.rewards.grasp_phase_terminal.scale

    def done(self, ctx: StepContext) -> torch.Tensor:
        flag = torch.logical_and(
            ctx.metrics["approach_phase_terminal"],
            ctx.metrics["grasp_phase_terminal"],
        )
        return flag


# ---------------------------------------------------------------------------
# Reward: Approach Phase
# ---------------------------------------------------------------------------


class ApproachDistanceRewardStep(RewardStep):
    name = "rew_approach_distance"
    requires_metrics = frozenset({"approach_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.cfg.rewards.approach_distance.mode == "progressive":
            prev = self._prev("approach_distance", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            delta = (ctx.metrics["approach_distance"] - prev).clamp(min=0.0)
            return delta * env.cfg.rewards.approach_distance.scale
        return (
            ctx.metrics["approach_distance"] * env.cfg.rewards.approach_distance.scale
        )


class ApproachAlignmentRewardStep(RewardStep):
    name = "rew_approach_alignment"
    requires_metrics = frozenset({"approach_alignment"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.cfg.rewards.approach_alignment.mode == "progressive":
            prev = self._prev("approach_alignment", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            delta = (ctx.metrics["approach_alignment"] - prev).clamp(min=0.0)
            return delta * env.cfg.rewards.approach_alignment.scale
        return (
            ctx.metrics["approach_alignment"] * env.cfg.rewards.approach_alignment.scale
        )


class ApproachGripperPoseRewardStep(RewardStep):
    name = "rew_approach_gripper_pose"
    requires_metrics = frozenset({"approach_gripper_pose"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.cfg.rewards.approach_gripper_pose.mode == "progressive":
            prev = self._prev("approach_gripper_pose", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            delta = (ctx.metrics["approach_gripper_pose"] - prev).clamp(min=0.0)
            return delta * env.cfg.rewards.approach_gripper_pose.scale
        return (
            ctx.metrics["approach_gripper_pose"]
            * env.cfg.rewards.approach_gripper_pose.scale
        )


class ApproachPhaseRewardStep(RewardStep):
    name = "rew_approach_phase"
    requires_metrics = frozenset({"approach_phase"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if env.cfg.rewards.approach_phase.mode == "progressive":
            prev = self._prev("approach_phase", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            delta = (ctx.metrics["approach_phase"] - prev).clamp(min=0.0)
            return delta * env.cfg.rewards.approach_phase.scale
        return ctx.metrics["approach_phase"] * env.cfg.rewards.approach_phase.scale


class AvoidBumpingCubeRewardStep(RewardStep):
    name = "rew_avoid_bumping_cube"
    requires_metrics = frozenset(
        {"gripper_cube_contact_force_magnitude", "grip_zone_cube_distance"}
    )
    requires_env_metrics = frozenset({"dr_cube_scale"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        force_mag = ctx.metrics["gripper_cube_contact_force_magnitude"]
        # Use per-env DR cube scale (isotropic; X axis = side length scale factor)
        # so the proximity threshold tracks the actual cube size this episode.
        cube_width = ctx.env_metrics["dr_cube_scale"][:, 0] * CUBE_WIDTH
        cube_near_gz = ctx.metrics["grip_zone_cube_distance"] < (
            env.cfg.rewards.avoid_bumping_cube.cube_widths * cube_width
        )
        return torch.where(
            (force_mag > 0.0) & ~cube_near_gz,
            torch.tensor(
                env.cfg.rewards.avoid_bumping_cube.scale,
                device=env.device,
                dtype=torch.float32,
            ),
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Reward: Grasp Phase
# ---------------------------------------------------------------------------


class GraspPhaseRewardStep(RewardStep):
    name = "rew_grasp_phase"
    requires_metrics = frozenset({"grasp_phase", "approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        grasp_phase = ctx.metrics["grasp_phase"]
        if env.cfg.rewards.grasp_phase.mode == "progressive":
            prev = self._prev("grasp_phase", ctx)
            if prev is None:
                return torch.zeros(env.num_envs, device=env.device)
            grasp_phase = (grasp_phase - prev).clamp(min=0.0)
        return (
            ctx.metrics["approach_phase_terminal"].float()
            * grasp_phase
            * env.cfg.rewards.grasp_phase.scale
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_reward_pipeline(cfg) -> RewardPipeline:
    """Construct the reward pipeline, including only enabled steps.

    Args:
        cfg: The ``So101LiftCubeCfg`` instance (``env.cfg``).
    """
    r = cfg.rewards
    steps: list[RewardStep] = []

    # Primary
    if r.distance.enabled:
        steps.append(DistanceRewardStep(gates=r.distance.gates))
    if r.grip_cube.enabled:
        steps.append(GripCubeRewardStep(gates=r.grip_cube.gates))
    if r.lift_cube.enabled:
        steps.append(LiftCubeRewardStep(gates=r.lift_cube.gates))

    # Shaping
    if r.gripper_cube_alignment.enabled:
        steps.append(
            GripperCubeAlignmentRewardStep(gates=r.gripper_cube_alignment.gates)
        )
    if r.close_gripper.enabled:
        steps.append(CloseGripperRewardStep(gates=r.close_gripper.gates))

    # Smoothing
    if r.action.enabled:
        steps.append(ActionRewardStep(gates=r.action.gates))
    if r.ee_linear_speed.enabled:
        steps.append(EELinearSpeedRewardStep(gates=r.ee_linear_speed.gates))
    if r.joint_speed.enabled:
        steps.append(JointSpeedRewardStep(gates=r.joint_speed.gates))

    # Terminal
    if r.success_lift_fraction_terminal.enabled:
        steps.append(
            SuccessLiftFractionTerminalRewardStep(
                gates=r.success_lift_fraction_terminal.gates
            )
        )
    if r.safety_touch_table_terminal.enabled:
        steps.append(
            SafetyTouchTableTerminalRewardStep(
                gates=r.safety_touch_table_terminal.gates
            )
        )
    if r.safety_touch_table.enabled:
        steps.append(SafetyTouchTableRewardStep(gates=r.safety_touch_table.gates))
    if r.cube_out_of_range_terminal.enabled:
        steps.append(
            CubeOutOfRangeTerminalRewardStep(gates=r.cube_out_of_range_terminal.gates)
        )
    if r.approach_phase_terminal.enabled:
        steps.append(
            ApproachPhaseTerminalRewardStep(gates=r.approach_phase_terminal.gates)
        )
    if r.grasp_phase_terminal.enabled:
        steps.append(GraspPhaseTerminalRewardStep(gates=r.grasp_phase_terminal.gates))

    # Phase-specific
    if r.approach_distance.enabled:
        steps.append(ApproachDistanceRewardStep(gates=r.approach_distance.gates))
    if r.approach_alignment.enabled:
        steps.append(ApproachAlignmentRewardStep(gates=r.approach_alignment.gates))
    if r.approach_gripper_pose.enabled:
        steps.append(ApproachGripperPoseRewardStep(gates=r.approach_gripper_pose.gates))
    if r.approach_phase.enabled:
        steps.append(ApproachPhaseRewardStep(gates=r.approach_phase.gates))
    if r.avoid_bumping_cube.enabled:
        steps.append(AvoidBumpingCubeRewardStep(gates=r.avoid_bumping_cube.gates))
    if r.grasp_phase.enabled:
        steps.append(GraspPhaseRewardStep(gates=r.grasp_phase.gates))

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
