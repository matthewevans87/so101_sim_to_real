from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_unique,
)

from so101_rl.configurations.cube import CUBE_RESTING_HEIGHT
from so101_rl.helpers.utils import assert_tensor

if TYPE_CHECKING:
    from so101_rl.tasks.direct.so101_lift_cube.so101_lift_cube_env import So101LiftCube
    from so101_rl.env_metric_pipeline import EnvMetricPipeline
    from so101_rl.reward_pipeline import RewardPipeline


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class StepContext:
    """Shared context passed through both pipelines each step.

    ``env`` provides access to all Isaac Lab scene objects and cfg.
    ``metrics`` accumulates outputs from MetricSteps and is then read by RewardSteps.
    ``prev_metrics`` holds a clone of ``metrics`` from the previous step; used by
    progressive reward steps to compute improvement deltas.
    """

    env: So101LiftCube
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    prev_metrics: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def env_metrics(self) -> dict[str, torch.Tensor]:
        """Per-episode env values computed by :class:`EnvMetricPipeline` at reset."""
        return self.env.env_metrics


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class MetricStep(ABC):
    """Computes one or more step-level metrics, writing results into ``ctx.metrics``.

    Subclasses must declare which keys they write (``produces``) and which
    keys must already exist when they run (``depends_on``).  ``MetricPipeline``
    uses these declarations to topologically sort steps at construction time.
    """

    produces: frozenset[str] = frozenset()
    """Metric keys this step writes to ``ctx.metrics``."""

    depends_on: frozenset[str] = frozenset()
    """Metric keys that must be present in ``ctx.metrics`` before this step runs."""

    depends_on_env_metrics: frozenset[str] = frozenset()
    """Keys from ``env.env_metrics`` (produced by :class:`EnvMetricStep`) that
    this step reads during :meth:`compute`.  Validated at pipeline construction
    time against the set of keys provided by the active
    :class:`EnvMetricPipeline`."""

    obs_dim: int = 0
    """Number of columns this step contributes when included in an observation vector.
    0 means the step is not intended for direct use in observations.
    Scalars-per-env should set 1; vectors of length K should set K."""

    @abstractmethod
    def compute(self, ctx: StepContext) -> None: ...


class MetricPipeline:
    """Accepts a set of :class:`MetricStep` objects in any order, topologically sorts
    them by their ``produces`` / ``depends_on`` declarations, and runs them in dependency
    order each step.

    Raises:
        ValueError: If a declared dependency is not produced by any step, or if there
            is a dependency cycle among the steps.
    """

    def __init__(
        self, steps: list[MetricStep], known_env_keys: frozenset[str] = frozenset()
    ) -> None:
        self.steps = self._toposort(steps, known_env_keys)

    @staticmethod
    def _toposort(
        steps: list[MetricStep], known_env_keys: frozenset[str] = frozenset()
    ) -> list[MetricStep]:
        # Map each produced key to the step that produces it.
        key_to_step: dict[str, MetricStep] = {}
        for step in steps:
            for key in step.produces:
                if key in key_to_step:
                    raise ValueError(
                        f"Metric key '{key}' is produced by more than one step: "
                        f"{type(key_to_step[key]).__name__} and {type(step).__name__}"
                    )
                key_to_step[key] = step

        # Validate: every depends_on key must be produced by some step.
        # Keys in known_env_keys are satisfied by EnvMetricPipeline — skip them.
        for step in steps:
            for key in step.depends_on:
                if key not in key_to_step and key not in known_env_keys:
                    raise ValueError(
                        f"{type(step).__name__} depends on metric key '{key}', "
                        f"but no step produces it."
                    )
            # Validate depends_on_env_metrics keys are available from EnvMetricPipeline.
            for key in step.depends_on_env_metrics:
                if key not in known_env_keys:
                    raise ValueError(
                        f"{type(step).__name__} depends on env-metric key '{key}' "
                        f"via depends_on_env_metrics, but it is not provided by the "
                        f"EnvMetricPipeline (known_env_keys={known_env_keys!r})."
                    )

        # Build adjacency list: predecessor_step -> {dependent_steps}
        # and in-degree counts for Kahn's algorithm.
        dependents: dict[int, set[int]] = defaultdict(
            set
        )  # id(step) -> set of id(step)
        in_degree: dict[int, int] = {id(s): 0 for s in steps}
        step_by_id: dict[int, MetricStep] = {id(s): s for s in steps}

        for step in steps:
            for key in step.depends_on:
                if key in known_env_keys:
                    # satisfied by EnvMetricPipeline — no predecessor step in this graph
                    continue
                predecessor = key_to_step[key]
                if id(predecessor) != id(step):
                    dependents[id(predecessor)].add(id(step))
                    in_degree[id(step)] += 1

        # Kahn's algorithm
        queue: deque[int] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        sorted_ids: list[int] = []
        while queue:
            sid = queue.popleft()
            sorted_ids.append(sid)
            for dep_id in dependents[sid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

        if len(sorted_ids) != len(steps):
            raise ValueError(
                "Cycle detected among metric steps. Check the 'produces' and "
                "'depends_on' declarations for a circular dependency."
            )

        return [step_by_id[sid] for sid in sorted_ids]

    def compute(self, ctx: StepContext) -> None:
        ctx.metrics.clear()
        for step in self.steps:
            step.compute(ctx)


# ---------------------------------------------------------------------------
# Metric steps
# ---------------------------------------------------------------------------


class GripperContactForceMagnitudeMetricStep(MetricStep):
    produces = frozenset({"gripper_cube_contact_force_magnitude"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            torch.linalg.norm(
                env.gripper_contact_sensor.data.force_matrix_w[:, 0, 0, :],
                dim=-1,
                keepdim=True,
            )
            .squeeze(-1)
            .to(env.device)
        )
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["gripper_cube_contact_force_magnitude"] = val


class TableTouchedMetricStep(MetricStep):
    produces = frozenset({"is_table_touched"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        force_norms = torch.linalg.norm(
            env.table_contact_sensor.data.force_matrix_w, dim=-1
        )
        val = (force_norms > 0.0).any(dim=-1).any(dim=-1).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_table_touched"] = val


class CubePosEEMetricStep(MetricStep):
    produces = frozenset({"cube_pos_ee"})
    depends_on = frozenset()
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = env.gripper_tf.data.target_pos_source[:, 0, :]
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["cube_pos_ee"] = val


class GripperCubeAlignmentMetricStep(MetricStep):
    produces = frozenset({"gripper_cube_alignment"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        eps = 1e-6

        gripper_pos_w = env.robot.data.body_pos_w[:, env._ee_body_idx[0], :]
        cube_pos_w = env.cube.data.root_pos_w

        v = cube_pos_w - gripper_pos_w
        v = v / (v.norm(dim=-1, keepdim=True) + eps)

        gripper_quat_w = env.robot.data.body_quat_w[:, env._ee_body_idx[0], :]
        gripper_forward_w = quat_apply(
            gripper_quat_w,
            env.gripper_forward_ee.expand(env.num_envs, -1),
        )

        val = (v * gripper_forward_w).sum(dim=-1, keepdim=True).squeeze(-1)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["gripper_cube_alignment"] = val


class CubePosGZMetricStep(MetricStep):
    produces = frozenset({"cube_pos_gz"})
    depends_on = frozenset({"cube_pos_ee"})
    depends_on_env_metrics = frozenset({"grip_zone_offset"})
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = ctx.metrics["cube_pos_ee"] - ctx.env_metrics["grip_zone_offset"]
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["cube_pos_gz"] = val


class CubeRot6DGZMetricStep(MetricStep):
    produces = frozenset({"cube_rot6d_gz"})
    depends_on = frozenset()
    obs_dim = 6

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        q_gz = quat_unique(env.gripper_tf.data.target_quat_source[:, 0, :])
        R = matrix_from_quat(q_gz)
        val = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
        assert_tensor(val, (env.num_envs, 6), torch.float32)
        ctx.metrics["cube_rot6d_gz"] = val


class GripZoneCubeDistanceMetricStep(MetricStep):
    produces = frozenset({"grip_zone_cube_distance"})
    depends_on = frozenset({"cube_pos_gz"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = ctx.metrics["cube_pos_gz"].norm(dim=-1, keepdim=True).squeeze(-1)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["grip_zone_cube_distance"] = val


class CubeHeightWMetricStep(MetricStep):
    produces = frozenset({"cube_height_w"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (env.cube.data.root_pos_w[:, 2] - CUBE_RESTING_HEIGHT).clamp(min=0.0)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["cube_height_w"] = val


class CubeLiftFractionMetricStep(MetricStep):
    produces = frozenset({"cube_lift_fraction"})
    depends_on = frozenset({"cube_height_w"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["cube_height_w"]
            / env.cfg.get_reward_cfg("success_lift_fraction_terminal").height_threshold
        )
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["cube_lift_fraction"] = val


class IsSuccessLiftFractionTerminalMetricStep(MetricStep):
    produces = frozenset({"is_success_lift_fraction_terminal"})
    depends_on = frozenset({"cube_lift_fraction"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (ctx.metrics["cube_lift_fraction"] >= 1.0).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_success_lift_fraction_terminal"] = val


class IsCubeInGripPositionMetricStep(MetricStep):
    produces = frozenset({"is_cube_in_grip_position"})
    depends_on = frozenset({"grip_zone_cube_distance"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["grip_zone_cube_distance"]
            < env.cfg.get_reward_cfg("grip_cube").distance_threshold
        ).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_cube_in_grip_position"] = val


class IsCubeGrippedMetricStep(MetricStep):
    produces = frozenset({"is_cube_gripped"})
    depends_on = frozenset(
        {"is_cube_in_grip_position", "gripper_cube_contact_force_magnitude"}
    )
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["is_cube_in_grip_position"]
            & (
                ctx.metrics["gripper_cube_contact_force_magnitude"]
                > env.cfg.get_reward_cfg("grip_cube").touch_force_threshold
            )
        ).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_cube_gripped"] = val


class CubeDistanceFromBaseMetricStep(MetricStep):
    produces = frozenset({"cube_distance_from_base"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        cube_pos_w = env.cube.data.root_pos_w
        base_pos_w = env.robot.data.root_pos_w
        val = torch.linalg.norm(cube_pos_w - base_pos_w, dim=-1)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["cube_distance_from_base"] = val


class IsCubeOutOfRangeMetricStep(MetricStep):
    produces = frozenset({"is_cube_out_of_range"})
    depends_on = frozenset({"cube_distance_from_base"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["cube_distance_from_base"]
            > env.cfg.get_reward_cfg("cube_out_of_range_terminal").distance_threshold
        ).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_cube_out_of_range"] = val


class ApproachPhaseMetricStep(MetricStep):
    produces = frozenset(
        {
            "approach_distance",
            "approach_alignment",
            "approach_gripper_pose",
            "approach_phase",
        }
    )
    depends_on = frozenset({"grip_zone_cube_distance", "gripper_cube_alignment"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        d_cfg = env.cfg.metrics.approach_distance
        a_cfg = env.cfg.metrics.approach_alignment
        g_cfg = env.cfg.metrics.approach_gripper_pose
        p_cfg = env.cfg.metrics.approach_phase

        d = ctx.metrics["grip_zone_cube_distance"]
        dist_exp = torch.exp(-d_cfg.pressure * d)
        dist_linear = (1.0 - d / d_cfg.distance_max).clamp(min=0.0)
        approach_distance = dist_exp + d_cfg.linear_weight * dist_linear
        assert_tensor(approach_distance, (env.num_envs,), torch.float32)
        ctx.metrics["approach_distance"] = approach_distance

        # alignment in [-1, 1]; delta of 0 = perfect, delta of 2 = worst
        # linear term: (1 + alignment) / 2 maps [-1,1] → [0,1]
        a = ctx.metrics["gripper_cube_alignment"]
        align_exp = torch.exp(-a_cfg.pressure * (1.0 - a))
        align_linear = (1.0 + a) / 2.0
        approach_alignment = align_exp + a_cfg.linear_weight * align_linear
        assert_tensor(approach_alignment, (env.num_envs,), torch.float32)
        ctx.metrics["approach_alignment"] = approach_alignment

        gripper_pos = env.joint_pos[:, env._gripper_joint_idx]
        gripper_pos_delta = torch.abs(gripper_pos - g_cfg.gripper_pos_target).squeeze(
            -1
        )
        delta_norm = gripper_pos_delta / g_cfg.gripper_pos_target
        gripper_exp = torch.exp(-g_cfg.pressure * delta_norm)
        gripper_linear = (1.0 - delta_norm).clamp(min=0.0)
        approach_gripper_pose = gripper_exp + g_cfg.linear_weight * gripper_linear
        assert_tensor(approach_gripper_pose, (env.num_envs,), torch.float32)
        ctx.metrics["approach_gripper_pose"] = approach_gripper_pose

        # approach_phase uses its own pressure params so it can be tuned independently;
        # pure exp product so the terminal threshold (0.95) remains valid
        phase_dist_exp = torch.exp(-p_cfg.distance_pressure * d)
        phase_align_exp = torch.exp(-p_cfg.alignment_pressure * (1.0 - a))
        gripper_pos_delta_p = torch.abs(gripper_pos - p_cfg.gripper_pos_target).squeeze(
            -1
        )
        phase_gripper_exp = torch.exp(
            -p_cfg.gripper_pos_pressure
            * (gripper_pos_delta_p / p_cfg.gripper_pos_target)
        )
        approach_phase = phase_dist_exp * phase_align_exp * phase_gripper_exp
        assert_tensor(approach_phase, (env.num_envs,), torch.float32)
        ctx.metrics["approach_phase"] = approach_phase

        # Diagnostic sub-factors — logged automatically to Step_Metrics/ in TensorBoard
        ctx.metrics["_dbg_approach_dist_exp"] = phase_dist_exp
        ctx.metrics["_dbg_approach_align_exp"] = phase_align_exp
        ctx.metrics["_dbg_approach_gripper_exp"] = phase_gripper_exp


class GraspPhaseMetricStep(MetricStep):
    produces = frozenset({"grasp_phase"})
    depends_on = frozenset({"gripper_cube_contact_force_magnitude"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        _grasp_cfg = env.cfg.get_reward_cfg("grasp_phase")
        grasp_phase = torch.exp(
            -_grasp_cfg.grip_force_pressure
            * torch.abs(
                ctx.metrics["gripper_cube_contact_force_magnitude"]
                - _grasp_cfg.grip_force_target
            )
        )

        assert_tensor(grasp_phase, (env.num_envs,), torch.float32)
        ctx.metrics["grasp_phase"] = grasp_phase


class ApproachPhaseTerminalMetricStep(MetricStep):
    produces = frozenset({"approach_phase_terminal"})
    depends_on = frozenset({"approach_phase"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        approach_phase_terminal = (
            ctx.metrics["approach_phase"]
            > env.cfg.get_reward_cfg("approach_phase_terminal").threshold
        )

        assert_tensor(approach_phase_terminal, (env.num_envs,), torch.bool)
        ctx.metrics["approach_phase_terminal"] = approach_phase_terminal


class GraspPhaseTerminalMetricStep(MetricStep):
    produces = frozenset({"grasp_phase_terminal"})
    depends_on = frozenset({"grasp_phase", "approach_phase_terminal"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        grasp_phase_terminal = torch.logical_and(
            ctx.metrics["approach_phase_terminal"],
            ctx.metrics["grasp_phase"]
            > env.cfg.get_reward_cfg("grasp_phase_terminal").threshold,
        )

        assert_tensor(grasp_phase_terminal, (env.num_envs,), torch.bool)
        ctx.metrics["grasp_phase_terminal"] = grasp_phase_terminal


# ---------------------------------------------------------------------------
# Complete catalog of all available metric step classes (order irrelevant).
# MetricPipeline will topologically sort any subset passed to it.
# ---------------------------------------------------------------------------

ALL_METRIC_STEPS: list[type[MetricStep]] = [
    GripperContactForceMagnitudeMetricStep,
    TableTouchedMetricStep,
    CubePosEEMetricStep,
    GripperCubeAlignmentMetricStep,
    CubePosGZMetricStep,
    CubeRot6DGZMetricStep,
    GripZoneCubeDistanceMetricStep,
    CubeHeightWMetricStep,
    CubeLiftFractionMetricStep,
    IsSuccessLiftFractionTerminalMetricStep,
    IsCubeInGripPositionMetricStep,
    IsCubeGrippedMetricStep,
    CubeDistanceFromBaseMetricStep,
    IsCubeOutOfRangeMetricStep,
    ApproachPhaseMetricStep,
    GraspPhaseMetricStep,
    ApproachPhaseTerminalMetricStep,
    GraspPhaseTerminalMetricStep,
]

# Maps each observable metric key to the number of columns it contributes
# when flattened into an observation vector. Only keys with obs_dim > 0 appear.
KEY_OBS_DIMS: dict[str, int] = {
    key: cls.obs_dim
    for cls in ALL_METRIC_STEPS
    for key in cls.produces
    if cls.obs_dim > 0
}


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_metric_pipeline(
    reward_pipeline: RewardPipeline,
    extra_keys: frozenset[str] = frozenset(),
    env_metric_pipeline: EnvMetricPipeline | None = None,
) -> MetricPipeline:
    """Build a :class:`MetricPipeline` containing only the steps needed by
    ``reward_pipeline`` (via ``RewardStep.requires_metrics``) plus any
    additional keys requested via ``extra_keys``.

    Dependency chains are resolved automatically: if a required key is
    produced by a step that itself depends on another key, that upstream
    step is included too.

    The resulting pipeline is topologically sorted by ``MetricPipeline``.

    Args:
        reward_pipeline: The active :class:`RewardPipeline`; its
            ``required_metric_keys`` property seeds the selection.
        extra_keys: Additional metric keys to force-include (e.g. keys
            consumed by observations or ``_pre_physics_step`` rather than
            rewards).
        env_metric_pipeline: The active :class:`EnvMetricPipeline`, used to
            validate ``depends_on_env_metrics`` declarations and avoid chasing
            env-metric keys through the MetricStep catalog.
    """
    known_env_keys: frozenset[str] = (
        env_metric_pipeline.provided_keys
        if env_metric_pipeline is not None
        else frozenset()
    )

    # Build a key → step-class map from the full catalog.
    key_to_cls: dict[str, type[MetricStep]] = {}
    for cls in ALL_METRIC_STEPS:
        for key in cls.produces:
            key_to_cls[key] = cls

    # Compute the transitive closure of needed keys.
    needed_keys: set[str] = set(reward_pipeline.required_metric_keys) | set(extra_keys)
    frontier = set(needed_keys)
    while frontier:
        key = frontier.pop()
        if key not in key_to_cls:
            # Either an env-metric key (satisfied by EnvMetricPipeline) or will be
            # caught as an unsatisfied dependency by MetricPipeline._toposort.
            continue
        cls = key_to_cls[key]
        for dep_key in cls.depends_on:
            if dep_key not in needed_keys and dep_key not in known_env_keys:
                needed_keys.add(dep_key)
                frontier.add(dep_key)
        # depends_on_env_metrics keys are satisfied externally — don't chase them.

    # Collect the unique step classes required.
    needed_cls: set[type[MetricStep]] = {
        key_to_cls[k] for k in needed_keys if k in key_to_cls
    }

    return MetricPipeline([cls() for cls in needed_cls], known_env_keys=known_env_keys)
