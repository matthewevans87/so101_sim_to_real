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

from so101_rl.configurations.black_cube import CUBE_RESTING_HEIGHT
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


def _pinch_axis_w(env) -> torch.Tensor:
    """Unit vector pointing from the fixed jaw toward the moving jaw in world frame.

    Shape: ``(num_envs, 3)``.  Computed from the two jaw body positions so it
    adapts to any gripper configuration without requiring knowledge of the
    joint axis in local frame.  Forces perpendicular to this axis (e.g. drag)
    project to near-zero even when both jaws contact the cube.
    """
    fixed_pos = env.robot.data.body_pos_w[:, env._ee_body_idx[0], :]
    moving_pos = env.robot.data.body_pos_w[:, env._moving_jaw_body_idx[0], :]
    delta = moving_pos - fixed_pos
    return delta / (delta.norm(dim=-1, keepdim=True) + 1e-6)


class FixedJawContactForceMagnitudeMetricStep(MetricStep):
    """Pinch-axis-projected force on the fixed jaw (``Robot/gripper``) against the cube.

    Projects ``force_matrix_w[:, 0, 0, :]`` onto the dynamic pinch axis
    (direction from fixed jaw to moving jaw in world frame) and takes the
    absolute value.  Forces that are purely perpendicular to the pinch axis
    — such as drag-induced friction — project to near-zero even when both
    jaws contact the cube.

    Produces ``fixed_jaw_cube_contact_force_magnitude`` (N).
    """

    produces = frozenset({"fixed_jaw_cube_contact_force_magnitude"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        force_vec = env.gripper_contact_sensor.data.force_matrix_w[:, 0, 0, :].to(
            env.device
        )
        pinch_axis = _pinch_axis_w(env)
        val = torch.abs((force_vec * pinch_axis).sum(dim=-1))
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["fixed_jaw_cube_contact_force_magnitude"] = val


class MovingJawContactForceMagnitudeMetricStep(MetricStep):
    """Pinch-axis-projected force on the movable jaw (``Robot/moving_jaw_so101_v1``) against the cube.

    Same projection as :class:`FixedJawContactForceMagnitudeMetricStep` but
    reading from ``env.moving_jaw_contact_sensor``.

    Produces ``moving_jaw_cube_contact_force_magnitude`` (N).
    """

    produces = frozenset({"moving_jaw_cube_contact_force_magnitude"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        force_vec = env.moving_jaw_contact_sensor.data.force_matrix_w[:, 0, 0, :].to(
            env.device
        )
        pinch_axis = _pinch_axis_w(env)
        val = torch.abs((force_vec * pinch_axis).sum(dim=-1))
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["moving_jaw_cube_contact_force_magnitude"] = val


class GripperContactForceMagnitudeMetricStep(MetricStep):
    """Backward-compat alias: re-exposes the fixed-jaw force under the original key.

    Keeps ``gripper_cube_contact_force_magnitude`` available for critic
    observations and any other consumers without requiring config changes.
    """

    produces = frozenset({"gripper_cube_contact_force_magnitude"})
    depends_on = frozenset({"fixed_jaw_cube_contact_force_magnitude"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = ctx.metrics["fixed_jaw_cube_contact_force_magnitude"]
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


class NormalizedGripZoneCubeDistanceMetricStep(MetricStep):
    """Maps ``grip_zone_cube_distance`` onto [0, 1] using per-env cube half-width as
    the floor and ``cfg.behavior.max_cube_distance`` as the ceiling.

    * ``d_norm = 0.0`` — cube surface just touching the grip zone center.
    * ``d_norm = 1.0`` — cube centroid at or beyond ``max_cube_distance``.

    Produces ``normalized_grip_zone_cube_distance`` of shape ``(num_envs,)``.
    """

    produces = frozenset({"normalized_grip_zone_cube_distance"})
    depends_on = frozenset({"grip_zone_cube_distance"})
    depends_on_env_metrics = frozenset({"env_min_cube_distance"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        d = ctx.metrics["grip_zone_cube_distance"]  # (num_envs,)
        d_min = ctx.env_metrics[
            "env_min_cube_distance"
        ]  # (num_envs,) per-episode DR half-width
        d_max: float = env.cfg.behavior.max_cube_distance  # scalar ceiling
        # Clamp d into [d_min, d_max] per-env, then normalise to [0, 1].
        d_clamped = torch.min(d, torch.tensor(d_max, device=env.device)).clamp_min(
            d_min
        )
        val = (d_clamped - d_min) / (d_max - d_min)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["normalized_grip_zone_cube_distance"] = val


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
        val = ctx.metrics["cube_height_w"] / env.cfg.metrics.lift_phase.height_threshold
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["cube_lift_fraction"] = val


class IsSuccessLiftFractionTerminalMetricStep(MetricStep):
    produces = frozenset({"is_success_lift_fraction_terminal"})
    depends_on = frozenset({"cube_lift_fraction"})
    obs_dim = 1

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
            < env.cfg.metrics.grip_cube.distance_threshold
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
                > env.cfg.metrics.grip_cube.touch_force_threshold
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
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["cube_distance_from_base"]
            > env.cfg.metrics.cube_out_of_range.distance_threshold
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
    depends_on = frozenset(
        {"normalized_grip_zone_cube_distance", "gripper_cube_alignment"}
    )
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        d_cfg = env.cfg.metrics.approach_distance
        a_cfg = env.cfg.metrics.approach_alignment
        g_cfg = env.cfg.metrics.approach_gripper_pose
        p_cfg = env.cfg.metrics.approach_phase

        d = ctx.metrics["normalized_grip_zone_cube_distance"]
        dist_exp = torch.exp(-d_cfg.pressure * d)
        dist_linear = (1.0 - d).clamp(min=0.0)
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
    """Grasp quality metric based on the bilateral (geometric-mean) pinch force.

    ``bilateral_force = sqrt(f_fixed * f_moving)``

    Zero whenever either jaw has no contact with the cube (prevents single-jaw
    knock/drag patterns from earning reward).  Mapped onto [0, 1] via a
    saturation clamp:

        grasp_phase = clamp(bilateral_force / grip_force_sat_threshold, 0, 1)

    The metric rises linearly from 0 → 1 as force goes 0 → threshold, then
    saturates at 1.  This is compatible with a stiff PD gripper: genuine jaw
    closure immediately exceeds any reasonable threshold, earning full score,
    while passive drag contact produces forces well below the threshold.

    Debug metrics exposed for TensorBoard:
    - ``_dbg_fixed_jaw_force``   — pinch-axis force on fixed jaw
    - ``_dbg_moving_jaw_force``  — pinch-axis force on moving jaw
    - ``_dbg_gripper_joint_pos`` — gripper joint position (open=0.8, closed=0.15)
    """

    produces = frozenset({"grasp_phase"})
    depends_on = frozenset(
        {
            "fixed_jaw_cube_contact_force_magnitude",
            "moving_jaw_cube_contact_force_magnitude",
        }
    )
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        _grasp_cfg = env.cfg.metrics.grasp_phase

        f_fixed = ctx.metrics["fixed_jaw_cube_contact_force_magnitude"].clamp(min=0.0)
        f_moving = ctx.metrics["moving_jaw_cube_contact_force_magnitude"].clamp(min=0.0)

        # Geometric mean: zero if either jaw has no contact.
        bilateral_force = (f_fixed * f_moving).sqrt()

        # Saturation clamp: rises 0→1 as force goes 0→threshold, then holds at 1.
        # A stiff gripper immediately exceeds any reasonable threshold on genuine
        # closure; drag/graze forces are far below it.
        grasp_phase = (bilateral_force / _grasp_cfg.grip_force_sat_threshold).clamp(
            0.0, 1.0
        )

        assert_tensor(grasp_phase, (env.num_envs,), torch.float32)
        ctx.metrics["grasp_phase"] = grasp_phase

        # Diagnostics logged to Step_Metrics/ in TensorBoard.
        ctx.metrics["_dbg_fixed_jaw_force"] = f_fixed
        ctx.metrics["_dbg_moving_jaw_force"] = f_moving
        ctx.metrics["_dbg_gripper_joint_pos"] = env.joint_pos[
            :, env._gripper_joint_idx
        ].squeeze(-1)


class ApproachPhaseTerminalMetricStep(MetricStep):
    produces = frozenset({"approach_phase_terminal"})
    depends_on = frozenset({"approach_phase"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        approach_phase_terminal = (
            ctx.metrics["approach_phase"]
            > env.cfg.metrics.approach_phase_terminal.threshold
        )

        assert_tensor(approach_phase_terminal, (env.num_envs,), torch.bool)
        ctx.metrics["approach_phase_terminal"] = approach_phase_terminal


class GraspPhaseTerminalMetricStep(MetricStep):
    produces = frozenset({"grasp_phase_terminal"})
    depends_on = frozenset({"grasp_phase", "approach_phase_terminal"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        grasp_phase_terminal = torch.logical_and(
            ctx.metrics["approach_phase_terminal"],
            ctx.metrics["grasp_phase"] > env.cfg.metrics.grasp_phase_terminal.threshold,
        )

        assert_tensor(grasp_phase_terminal, (env.num_envs,), torch.bool)
        ctx.metrics["grasp_phase_terminal"] = grasp_phase_terminal


class GoalZonePosMetricStep(MetricStep):
    """Exposes the current episode's goal zone position in env-local frame.

    Produces ``goal_zone_pos_local = goal_zone_pos_w - env.scene.env_origins``
    of shape ``(num_envs, 3)``.  Using env-local coordinates ensures the actor
    observation is invariant across parallel environments.

    Requires :class:`~so101_rl.env_metric_pipeline.GoalZoneEnvMetricStep`
    (i.e. ``cfg.domain_randomization.goal_zone.enabled`` must be ``True``).
    """

    produces = frozenset({"goal_zone_pos_local"})
    depends_on = frozenset()
    depends_on_env_metrics = frozenset({"goal_zone_pos_w"})
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.env_metrics["goal_zone_pos_w"] - env.scene.env_origins
        )  # (num_envs, 3)
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["goal_zone_pos_local"] = val


class GoalZoneCubeDistanceMetricStep(MetricStep):
    """Computes distance-based metrics from the cube to the goal zone.

    Produced metrics
    ~~~~~~~~~~~~~~~~
    * ``goal_zone_cube_distance`` — Euclidean distance (m) from the cube
      centroid to the goal zone centre, shape ``(num_envs,)``.
    * ``goal_zone_distance`` — exponentially shaped score in (0, 1] that is
      highest (1.0) at zero distance and decays as
      ``exp(-pressure * dist / env_cube_width)``.
    * ``is_goal_zone_reached`` — float ``1.0`` when
      ``goal_zone_cube_distance <= cfg.distance_threshold``,
      else ``0.0``.
    """

    produces = frozenset(
        {"goal_zone_cube_distance", "goal_zone_distance", "is_goal_zone_reached"}
    )
    depends_on = frozenset()
    depends_on_env_metrics = frozenset({"goal_zone_pos_w", "env_cube_width"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        cube_pos_w = env.cube.data.root_pos_w  # (num_envs, 3)
        goal_pos_w = ctx.env_metrics["goal_zone_pos_w"]  # (num_envs, 3)
        dist = (cube_pos_w - goal_pos_w).norm(dim=-1)  # (num_envs,)
        cube_width = ctx.env_metrics["env_cube_width"]  # (num_envs,)

        cfg = env.cfg.metrics.goal_zone_distance
        gz_dist = torch.exp(
            -cfg.pressure * dist / cube_width.clamp(min=1e-6)
        )  # (num_envs,) in (0, 1]
        reached = (dist <= cfg.distance_threshold).float()  # (num_envs,)

        assert_tensor(dist, (env.num_envs,), torch.float32)
        assert_tensor(gz_dist, (env.num_envs,), torch.float32)
        assert_tensor(reached, (env.num_envs,), torch.float32)
        ctx.metrics["goal_zone_cube_distance"] = dist
        ctx.metrics["goal_zone_distance"] = gz_dist
        ctx.metrics["is_goal_zone_reached"] = reached


class CubeLinearVelocityMetricStep(MetricStep):
    """Linear velocity of the cube in world frame.

    Produces ``cube_linear_velocity`` of shape ``(num_envs, 3)`` (m/s).
    """

    produces = frozenset({"cube_linear_velocity"})
    depends_on = frozenset()
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = env.cube.data.root_lin_vel_w
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["cube_linear_velocity"] = val


class CubeAngularVelocityMetricStep(MetricStep):
    """Angular velocity of the cube in world frame.

    Produces ``cube_angular_velocity`` of shape ``(num_envs, 3)`` (rad/s).
    """

    produces = frozenset({"cube_angular_velocity"})
    depends_on = frozenset()
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = env.cube.data.root_ang_vel_w
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["cube_angular_velocity"] = val


class GripperOpenWidthMetricStep(MetricStep):
    """Distance between the two gripper tip xframes in world space.

    Computes the Euclidean distance between:

    * ``Robot/gripper/gripperframe``  (fixed-jaw tip)
    * ``Robot/moving_jaw_so101_v1/moving_gripperframe``  (moving-jaw tip)

    The local USD transform of each xframe relative to its parent body is read
    once from the USD stage on the first call and cached as class attributes,
    following the same pattern as ``GripZoneOffsetEnvMetricStep``.

    Produces ``gripper_open_width`` (m), shape ``(num_envs,)``.
    """

    produces = frozenset({"gripper_open_width"})
    depends_on = frozenset()
    obs_dim = 1

    # Class-level cache — populated once from USD, shared across all instances.
    _gripperframe_local_pos: torch.Tensor | None = None
    _moving_gripperframe_local_pos: torch.Tensor | None = None

    def _cache_tip_transforms(self, env) -> None:
        import omni.usd  # type: ignore
        from pxr import UsdGeom  # type: ignore

        stage = omni.usd.get_context().get_stage()
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

        def _read_local_pos(prim_path: str) -> torch.Tensor:
            prim = stage.GetPrimAtPath(prim_path)
            local_xform = UsdGeom.Xformable(prim).GetLocalTransformation()
            t = local_xform.ExtractTranslation()
            return torch.tensor(
                [
                    t[0] * meters_per_unit,
                    t[1] * meters_per_unit,
                    t[2] * meters_per_unit,
                ],
                device=env.device,
                dtype=torch.float32,
            )

        GripperOpenWidthMetricStep._gripperframe_local_pos = _read_local_pos(
            "/World/envs/env_0/Robot/gripper/gripperframe"
        )
        GripperOpenWidthMetricStep._moving_gripperframe_local_pos = _read_local_pos(
            "/World/envs/env_0/Robot/moving_jaw_so101_v1/moving_gripperframe"
        )

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env

        if GripperOpenWidthMetricStep._gripperframe_local_pos is None:
            self._cache_tip_transforms(env)

        # Fixed-jaw tip position in world frame
        gripper_pos_w = env.robot.data.body_pos_w[:, env._ee_body_idx[0], :]  # (N, 3)
        gripper_quat_w = env.robot.data.body_quat_w[:, env._ee_body_idx[0], :]  # (N, 4)
        tip_fixed_w = gripper_pos_w + quat_apply(
            gripper_quat_w,
            GripperOpenWidthMetricStep._gripperframe_local_pos.unsqueeze(0).expand(  # type: ignore[union-attr]
                env.num_envs, -1
            ),
        )

        # Moving-jaw tip position in world frame
        mj_pos_w = env.robot.data.body_pos_w[
            :, env._moving_jaw_body_idx[0], :
        ]  # (N, 3)
        mj_quat_w = env.robot.data.body_quat_w[
            :, env._moving_jaw_body_idx[0], :
        ]  # (N, 4)
        tip_moving_w = mj_pos_w + quat_apply(
            mj_quat_w,
            GripperOpenWidthMetricStep._moving_gripperframe_local_pos.unsqueeze(0).expand(  # type: ignore[union-attr]
                env.num_envs, -1
            ),
        )

        val = (tip_fixed_w - tip_moving_w).norm(dim=-1)  # (N,)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["gripper_open_width"] = val


# ---------------------------------------------------------------------------
# Complete catalog of all available metric step classes (order irrelevant).
# MetricPipeline will topologically sort any subset passed to it.
# ---------------------------------------------------------------------------

ALL_METRIC_STEPS: list[type[MetricStep]] = [
    FixedJawContactForceMagnitudeMetricStep,
    MovingJawContactForceMagnitudeMetricStep,
    GripperContactForceMagnitudeMetricStep,
    TableTouchedMetricStep,
    CubePosEEMetricStep,
    GripperCubeAlignmentMetricStep,
    CubePosGZMetricStep,
    CubeRot6DGZMetricStep,
    GripZoneCubeDistanceMetricStep,
    NormalizedGripZoneCubeDistanceMetricStep,
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
    GoalZonePosMetricStep,
    GoalZoneCubeDistanceMetricStep,
    CubeLinearVelocityMetricStep,
    CubeAngularVelocityMetricStep,
    GripperOpenWidthMetricStep,
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
