from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import omni.usd  # type: ignore
import torch
from pxr import Gf, UsdGeom, UsdLux, UsdShade  # type: ignore

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_unique,
    sample_uniform,
)
import isaaclab.utils.math as math_utils

from so101_rl.configurations.camera import (
    CAMERA_ROTATION_QUAT_WXYZ,
    CAMERA_TRANSLATE_VEC,
)
from so101_rl.configurations.cube import (
    CUBE_DEFAULT_DIMS,
    CUBE_RESTING_HEIGHT,
    CUBE_WIDTH,
)
from so101_rl.helpers.utils import assert_tensor

if TYPE_CHECKING:
    from so101_rl.tasks.direct.so101_lift_cube.so101_lift_cube_env import So101LiftCube


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class StepContext:
    """Shared context passed through both pipelines each step.

    ``env`` provides access to all Isaac Lab scene objects and cfg.
    ``metrics`` accumulates outputs from MetricSteps and is then read by RewardSteps.
    """

    env: So101LiftCube
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def env_metrics(self) -> dict[str, torch.Tensor]:
        """Per-episode env values computed by :class:`EnvMetricPipeline` at reset."""
        return self.env.env_metrics


@dataclass
class DRContext:
    """Context passed to each :class:`DRStep` during an episode reset.

    ``env`` provides access to the environment, its config, and scene objects.
    ``env_ids`` is the sequence of environment indices being reset this step.
    ``metrics`` is reserved for future DR steps that depend on computed metric
    values (e.g., current cube scale) rather than re-deriving them.
    """

    env: So101LiftCube
    env_ids: Sequence[int]
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)


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
        """Union of all ``requires_metrics`` declared by the steps in this pipeline."""
        return frozenset().union(*(s.requires_metrics for s in self.steps))

    @property
    def terminal_steps(self) -> list[TerminalRewardStep]:
        """All :class:`TerminalRewardStep` instances in this pipeline."""
        return [s for s in self.steps if isinstance(s, TerminalRewardStep)]

    def get_dones(self, ctx: StepContext) -> torch.Tensor:
        """Return a bool tensor of shape ``(num_envs,)`` — True for any environment
        where at least one :class:`TerminalRewardStep` signals termination.

        Assumes metrics have already been computed for the current step.
        """
        env = ctx.env
        terminal = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for step in self.terminal_steps:
            terminal = terminal | step.done(ctx)
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
            env.extras["log"][f"Episode_Reward/{step.name}"] = rew.mean()
            env.extras["per_env_log"][f"Episode_Reward/{step.name}"] = rew
            total += rew
        return total


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


class CameraCubeAlignmentMetricStep(MetricStep):
    produces = frozenset({"camera_cube_alignment"})
    depends_on = frozenset()
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        eps = 1e-6

        gripper_pos_w = env.robot.data.body_pos_w[:, env._ee_body_idx[0], :]
        gripper_quat_w = env.robot.data.body_quat_w[:, env._ee_body_idx[0], :]

        camera_pos_w = gripper_pos_w + quat_apply(
            gripper_quat_w,
            env._camera_offset_pos.expand(env.num_envs, -1),
        )
        camera_quat_w = math_utils.quat_mul(
            gripper_quat_w,
            env._camera_offset_quat.expand(env.num_envs, -1),
        )

        cube_pos_w = env.cube.data.root_pos_w
        v = cube_pos_w - camera_pos_w
        v = v / (v.norm(dim=-1, keepdim=True) + eps)

        camera_forward_w = quat_apply(
            camera_quat_w,
            env.camera_forward_local.expand(env.num_envs, -1),
        )

        val = (v * camera_forward_w).sum(dim=-1, keepdim=True).squeeze(-1)
        assert_tensor(val, (env.num_envs,), torch.float32)
        ctx.metrics["camera_cube_alignment"] = val


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
            / env.cfg.rewards.success_lift_fraction_terminal.height_threshold
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


class IsSuccessTouchTerminalMetricStep(MetricStep):
    produces = frozenset({"is_success_touch_terminal"})
    depends_on = frozenset({"gripper_cube_contact_force_magnitude"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["gripper_cube_contact_force_magnitude"]
            > env.cfg.rewards.success_touch_terminal.touch_force_threshold
        ).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_success_touch_terminal"] = val


class IsSuccessPointAtCubeTerminalMetricStep(MetricStep):
    produces = frozenset({"is_success_point_at_cube_terminal"})
    depends_on = frozenset({"gripper_cube_alignment"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (ctx.metrics["gripper_cube_alignment"] >= 1.0).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_success_point_at_cube_terminal"] = val


class IsCubeInGripPositionMetricStep(MetricStep):
    produces = frozenset({"is_cube_in_grip_position"})
    depends_on = frozenset({"grip_zone_cube_distance"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = (
            ctx.metrics["grip_zone_cube_distance"]
            < env.cfg.rewards.grip_cube.distance_threshold
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
                > env.cfg.rewards.success_touch_terminal.touch_force_threshold
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
            > env.cfg.rewards.cube_out_of_range_terminal.distance_threshold
        ).bool()
        assert_tensor(val, (env.num_envs,), torch.bool)
        ctx.metrics["is_cube_out_of_range"] = val


class ApproachPhaseMetricStep(MetricStep):
    produces = frozenset({"approach_phase"})
    depends_on = frozenset({"grip_zone_cube_distance", "gripper_cube_alignment"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env

        distance_z = torch.exp(
            env.cfg.rewards.approach_phase.distance_pressure
            * -ctx.metrics["grip_zone_cube_distance"]
        ) ** (env.cfg.rewards.approach_phase.distance_weight)

        gripper_pos = env.joint_pos[:, env._gripper_joint_idx]
        print("gripper_pos_target", env.cfg.rewards.approach_phase.gripper_pos_target)
        print("gripper_pos", gripper_pos[0, 0])
        gripper_pos_delta = torch.abs(
            gripper_pos - env.cfg.rewards.approach_phase.gripper_pos_target
        )
        print("gripper_delta", gripper_pos_delta[0, 0])
        pose_z = torch.exp(
            env.cfg.rewards.approach_phase.gripper_pos_pressure
            * -gripper_pos_delta
            / env.cfg.rewards.approach_phase.gripper_pos_target
        ).squeeze(-1) ** (env.cfg.rewards.approach_phase.gripper_pos_weight)
        print("gripper_close_error", pose_z[0])

        alignment_z = torch.exp(
            -1 - (ctx.metrics["gripper_cube_alignment"] + 1) / 2
        ) ** (env.cfg.rewards.approach_phase.alignment_weight)
        approach_phase = distance_z * alignment_z * pose_z

        assert_tensor(approach_phase, (env.num_envs,), torch.float32)
        ctx.metrics["approach_phase"] = approach_phase


class GraspPhaseMetricStep(MetricStep):
    produces = frozenset({"grasp_phase"})
    depends_on = frozenset({"gripper_cube_contact_force_magnitude", "approach_phase"})

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        grasp_phase = torch.exp(
            -env.cfg.rewards.grasp_phase.grip_force_pressure
            * torch.abs(
                ctx.metrics["gripper_cube_contact_force_magnitude"]
                - env.cfg.rewards.grasp_phase.grip_force_target
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
            > env.cfg.rewards.approach_phase_terminal.threshold
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
            ctx.metrics["grasp_phase"] > env.cfg.rewards.grasp_phase_terminal.threshold,
        )

        assert_tensor(grasp_phase_terminal, (env.num_envs,), torch.bool)
        ctx.metrics["grasp_phase_terminal"] = grasp_phase_terminal


# ---------------------------------------------------------------------------
# Reward steps — Primary
# ---------------------------------------------------------------------------


class DistanceRewardStep(RewardStep):
    name = "rew_distance"
    requires_metrics = frozenset({"grip_zone_cube_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
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


class GripperLookAtCubeRewardStep(RewardStep):
    name = "rew_gripper_look_at_cube"
    requires_metrics = frozenset({"is_cube_gripped"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        eps = 1e-6

        gripper_pos = env.robot.data.body_pos_w[:, env._ee_body_idx[0], :]
        gripper_quat = env.robot.data.body_quat_w[:, env._ee_body_idx[0], :]

        camera_offset = (
            torch.tensor(CAMERA_TRANSLATE_VEC, device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(env.num_envs, 3)
        )
        camera_pos_w = gripper_pos + quat_apply(gripper_quat, camera_offset)

        camera_rot_offset = (
            torch.tensor(
                CAMERA_ROTATION_QUAT_WXYZ, device=env.device, dtype=torch.float32
            )
            .unsqueeze(0)
            .expand(env.num_envs, 4)
        )
        camera_quat_w = math_utils.quat_mul(gripper_quat, camera_rot_offset)

        forward_local = (
            torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32)
            .view(1, 3)
            .expand(env.num_envs, 3)
        )
        cam_forward_w = quat_apply(camera_quat_w, forward_local)

        cube_pos_w = env.gripper_tf.data.target_pos_w[:, 0, :]
        vec_to_cube = cube_pos_w - camera_pos_w

        cam_forward_norm = cam_forward_w / (
            torch.linalg.norm(cam_forward_w, dim=-1, keepdim=True) + eps
        )
        vec_to_cube_norm = vec_to_cube / (
            torch.linalg.norm(vec_to_cube, dim=-1, keepdim=True) + eps
        )

        lookat_factor = torch.clamp(
            (cam_forward_norm * vec_to_cube_norm).sum(dim=-1), min=0.0, max=1.0
        )
        lookat_factor = torch.maximum(lookat_factor, ctx.metrics["is_cube_gripped"])

        return env.cfg.rewards.gripper_look_at_cube.scale * lookat_factor


class CameraCubeAlignmentRewardStep(RewardStep):
    name = "rew_camera_cube_alignment"
    requires_metrics = frozenset({"is_cube_gripped", "camera_cube_alignment"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return (
            torch.maximum(
                ctx.metrics["is_cube_gripped"],
                ctx.metrics["camera_cube_alignment"],
            )
            * env.cfg.rewards.camera_cube_alignment.scale
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


class GripperForceRewardStep(RewardStep):
    name = "rew_gripper_force"
    requires_metrics = frozenset(
        {"gripper_cube_contact_force_magnitude", "is_cube_in_grip_position"}
    )

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        target = env.cfg.rewards.gripper_force.force_target
        force_error = torch.abs(
            ctx.metrics["gripper_cube_contact_force_magnitude"] - target
        )
        rew = (
            torch.exp(-force_error / (target + 1e-6))
            * env.cfg.rewards.gripper_force.scale
        )
        return rew * ctx.metrics["is_cube_in_grip_position"].float()


class VantageRewardStep(RewardStep):
    name = "rew_vantage"
    requires_metrics = frozenset({"is_cube_gripped"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        cfg = env.cfg.rewards.vantage

        cube_gripper_dist = torch.linalg.norm(
            env.gripper_tf.data.source_pos_w
            - env.gripper_tf.data.target_pos_w[:, 0, :],
            dim=-1,
        )
        is_far = cube_gripper_dist > cfg.far_distance_threshold

        # --- Raw vantage value (mirrors _get_rew_vantage logic) ---
        d = cube_gripper_dist
        far_enough = d > cfg.min_distance_threshold

        dist_reward = torch.exp(
            -((d - cfg.ideal_distance) ** 2) / (2 * cfg.ideal_distance_sigma**2)
        )

        h_above_cube = (
            env.gripper_tf.data.source_pos_w[:, 2]
            - env.gripper_tf.data.target_pos_w[:, 0, 2]
        )
        height_reward = torch.where(
            h_above_cube >= 0,
            torch.exp(
                -((h_above_cube - cfg.ideal_height) ** 2)
                / (2 * cfg.ideal_height_sigma**2)
            ),
            torch.exp(-((h_above_cube) ** 2) / (2 * (cfg.ideal_height_sigma / 2) ** 2))
            * 0.3,
        )

        gripper_roll_error = torch.abs(
            env.robot.data.joint_pos[:, env._wrist_roll_idx] - math.radians(-90.0)
        ) / math.radians(-90.0)

        raw = torch.where(
            is_far & far_enough,
            cfg.scale * dist_reward * height_reward * gripper_roll_error,
            torch.zeros(env.num_envs, device=env.device),
        )

        return torch.where(
            ctx.metrics["is_cube_gripped"],
            torch.full((env.num_envs,), cfg.scale, device=env.device),
            raw,
        )


class KeepCameraUprightRewardStep(RewardStep):
    name = "rew_keep_camera_upright"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        error = torch.abs(
            env.robot.data.joint_pos[:, env._wrist_roll_idx] - math.radians(-90.0)
        )
        return error * env.cfg.rewards.keep_camera_upright.scale


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
        return env.cfg.rewards.action.scale * torch.sum(torch.abs(env.actions), dim=-1)


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


class EEHeightSafetyRewardStep(RewardStep):
    name = "rew_ee_height_safety"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        ee_height = env.robot.data.body_pos_w[:, env._ee_body_idx[0], 2]
        unsafe = ee_height < env.cfg.safety.min_ee_height
        return torch.where(
            unsafe,
            torch.full_like(ee_height, env.cfg.rewards.ee_height_safety.scale),
            torch.zeros_like(ee_height),
        )


# ---------------------------------------------------------------------------
# Reward steps — Terminal
# ---------------------------------------------------------------------------


class SuccessTouchTerminalRewardStep(TerminalRewardStep):
    name = "rew_success_touch_terminal"
    requires_metrics = frozenset({"is_success_touch_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = ctx.metrics["is_success_touch_terminal"]
        return torch.where(
            flag >= 1.0,
            torch.full_like(flag.float(), env.cfg.rewards.success_touch_terminal.scale),
            torch.zeros(env.num_envs, device=env.device),
        )

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_success_touch_terminal"]


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


class SuccessPointAtCubeTerminalRewardStep(TerminalRewardStep):
    name = "rew_success_point_at_cube_terminal"
    requires_metrics = frozenset({"is_success_point_at_cube_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = ctx.metrics["is_success_point_at_cube_terminal"]
        return torch.where(
            flag >= 1.0,
            torch.full_like(
                flag.float(), env.cfg.rewards.success_point_at_cube_terminal.scale
            ),
            torch.zeros(env.num_envs, device=env.device),
        )

    def done(self, ctx: StepContext) -> torch.Tensor:
        return ctx.metrics["is_success_point_at_cube_terminal"]


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
        env = ctx.env
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
        env = ctx.env
        flag = torch.logical_and(
            ctx.metrics["approach_phase_terminal"],
            ctx.metrics["grasp_phase_terminal"],
        )
        return flag


# ---------------------------------------------------------------------------
# Reward: Approach Phase
# ---------------------------------------------------------------------------


class ApproachPhaseRewardStep(RewardStep):
    name = "rew_approach_phase"
    requires_metrics = frozenset({"approach_phase"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return ctx.metrics["approach_phase"] * env.cfg.rewards.approach_phase.scale


class AvoidBumpingCubeRewardStep(RewardStep):
    name = "rew_avoid_bumping_cube"
    requires_metrics = frozenset(
        {"gripper_cube_contact_force_magnitude", "grip_zone_cube_distance"}
    )

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        force_mag = ctx.metrics["gripper_cube_contact_force_magnitude"]
        cube_near_gz = ctx.metrics["grip_zone_cube_distance"] < (
            env.cfg.rewards.avoid_bumping_cube.cube_widths * CUBE_WIDTH
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
        return (
            ctx.metrics["approach_phase_terminal"].float()
            * grasp_phase
            * env.cfg.rewards.grasp_phase.scale
        )


# ---------------------------------------------------------------------------
# Complete catalog of all available metric step classes (order irrelevant).
# MetricPipeline will topologically sort any subset passed to it.
# ---------------------------------------------------------------------------

ALL_METRIC_STEPS: list[type[MetricStep]] = [
    GripperContactForceMagnitudeMetricStep,
    TableTouchedMetricStep,
    CubePosEEMetricStep,
    GripperCubeAlignmentMetricStep,
    CameraCubeAlignmentMetricStep,
    CubePosGZMetricStep,
    CubeRot6DGZMetricStep,
    GripZoneCubeDistanceMetricStep,
    CubeHeightWMetricStep,
    CubeLiftFractionMetricStep,
    IsSuccessLiftFractionTerminalMetricStep,
    IsSuccessTouchTerminalMetricStep,
    IsSuccessPointAtCubeTerminalMetricStep,
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
# Factory helpers
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


def build_reward_pipeline(cfg) -> RewardPipeline:
    """Construct the reward pipeline, including only enabled steps.

    Args:
        cfg: The ``So101LiftCubeCfg`` instance (``env.cfg``).
    """
    r = cfg.rewards
    steps: list[RewardStep] = []

    # Primary
    if r.distance.enabled:
        steps.append(DistanceRewardStep())
    if r.grip_cube.enabled:
        steps.append(GripCubeRewardStep())
    if r.lift_cube.enabled:
        steps.append(LiftCubeRewardStep())

    # Shaping
    if r.gripper_cube_alignment.enabled:
        steps.append(GripperCubeAlignmentRewardStep())
    if r.gripper_look_at_cube.enabled:
        steps.append(GripperLookAtCubeRewardStep())
    if r.camera_cube_alignment.enabled:
        steps.append(CameraCubeAlignmentRewardStep())
    if r.close_gripper.enabled:
        steps.append(CloseGripperRewardStep())
    if r.gripper_force.enabled:
        steps.append(GripperForceRewardStep())
    if r.vantage.enabled:
        steps.append(VantageRewardStep())
    if r.keep_camera_upright.enabled:
        steps.append(KeepCameraUprightRewardStep())

    # Smoothing
    if r.action.enabled:
        steps.append(ActionRewardStep())
    if r.ee_linear_speed.enabled:
        steps.append(EELinearSpeedRewardStep())
    if r.joint_speed.enabled:
        steps.append(JointSpeedRewardStep())
    if r.ee_height_safety.enabled:
        steps.append(EEHeightSafetyRewardStep())

    # Terminal
    if r.success_touch_terminal.enabled:
        steps.append(SuccessTouchTerminalRewardStep())
    if r.success_lift_fraction_terminal.enabled:
        steps.append(SuccessLiftFractionTerminalRewardStep())
    if r.success_point_at_cube_terminal.enabled:
        steps.append(SuccessPointAtCubeTerminalRewardStep())
    if r.safety_touch_table_terminal.enabled:
        steps.append(SafetyTouchTableTerminalRewardStep())
    if r.safety_touch_table.enabled:
        steps.append(SafetyTouchTableRewardStep())
    if r.cube_out_of_range_terminal.enabled:
        steps.append(CubeOutOfRangeTerminalRewardStep())
    if r.approach_phase_terminal.enabled:
        steps.append(ApproachPhaseTerminalRewardStep())
    if r.grasp_phase_terminal.enabled:
        steps.append(GraspPhaseTerminalRewardStep())

    # Phase-specific
    if r.approach_phase.enabled:
        steps.append(ApproachPhaseRewardStep())
    if r.avoid_bumping_cube.enabled:
        steps.append(AvoidBumpingCubeRewardStep())
    if r.grasp_phase.enabled:
        steps.append(GraspPhaseRewardStep())

    return RewardPipeline(steps)


# ---------------------------------------------------------------------------
# DR base classes
# ---------------------------------------------------------------------------


class DRStep(ABC):
    """Applies one domain-randomisation operation to a subset of environments
    during an episode reset.

    Subclasses must implement :meth:`apply`.  The ``requires_metrics``
    declaration is empty for all current steps; it is reserved for the
    upcoming phase where DR steps will consume values from
    :class:`MetricStep` outputs (e.g., current cube scale) rather than
    re-deriving them.
    """

    requires_metrics: frozenset[str] = frozenset()
    """Metric keys from ``ctx.metrics`` that this step reads during :meth:`apply`."""

    requires_env_metrics: frozenset[str] = frozenset()
    """Keys from ``env.env_metrics`` (produced by :class:`EnvMetricPipeline`) that
    this step reads during :meth:`apply`."""

    @abstractmethod
    def apply(self, ctx: DRContext) -> None: ...


class DRPipeline:
    """Runs a sequence of :class:`DRStep` objects in order on every episode reset.

    Only the steps passed at construction are executed — callers should filter
    by the matching ``cfg`` enabled flag once at startup via
    :func:`build_dr_pipeline`.
    """

    def __init__(self, steps: list[DRStep]) -> None:
        self.steps = steps

    def apply(self, ctx: DRContext) -> None:
        for step in self.steps:
            step.apply(ctx)


# ---------------------------------------------------------------------------
# EnvMetricStep base classes
# ---------------------------------------------------------------------------


class EnvMetricStep(ABC):
    """Computes one or more per-episode, per-environment values and stores them in
    ``env.env_metrics`` during episode reset.

    Unlike :class:`MetricStep` (which is recomputed every physics step), an
    ``EnvMetricStep`` runs once per reset inside :class:`EnvMetricPipeline`.
    Results persist on ``env.env_metrics`` across the entire episode and are
    accessible to :class:`MetricStep`s via ``ctx.env_metrics``.

    Subclasses must declare ``produces`` and ``depends_on`` for toposorting,
    and implement :meth:`apply`.  Values must be written as full tensors of
    shape ``(num_envs, ...)``, initialised on first call (if absent) and
    updated in-place for ``ctx.env_ids`` only.
    """

    produces: frozenset[str] = frozenset()
    """Keys this step writes into ``env.env_metrics``."""

    depends_on: frozenset[str] = frozenset()
    """Other ``env.env_metrics`` keys that must be populated before this step runs."""

    @abstractmethod
    def apply(self, ctx: DRContext) -> None:
        """Write values for ``ctx.env_ids`` into ``ctx.env.env_metrics``."""
        ...


class EnvMetricPipeline:
    """Runs a topologically-sorted sequence of :class:`EnvMetricStep` objects
    at episode reset, populating ``env.env_metrics``.

    Raises:
        ValueError: On duplicate ``produces`` keys, unsatisfied dependencies,
            or dependency cycles.
    """

    def __init__(self, steps: list[EnvMetricStep]) -> None:
        self.steps = self._toposort(steps)

    @property
    def provided_keys(self) -> frozenset[str]:
        """Union of all ``produces`` sets across all steps in this pipeline."""
        return frozenset().union(*(s.produces for s in self.steps))

    @staticmethod
    def _toposort(steps: list[EnvMetricStep]) -> list[EnvMetricStep]:
        key_to_step: dict[str, EnvMetricStep] = {}
        for step in steps:
            for key in step.produces:
                if key in key_to_step:
                    raise ValueError(
                        f"Env-metric key '{key}' is produced by more than one step: "
                        f"{type(key_to_step[key]).__name__} and {type(step).__name__}"
                    )
                key_to_step[key] = step

        for step in steps:
            for key in step.depends_on:
                if key not in key_to_step:
                    raise ValueError(
                        f"{type(step).__name__} depends on env-metric key '{key}', "
                        f"but no step produces it."
                    )

        dependents: dict[int, set[int]] = defaultdict(set)
        in_degree: dict[int, int] = {id(s): 0 for s in steps}
        step_by_id: dict[int, EnvMetricStep] = {id(s): s for s in steps}

        for step in steps:
            for key in step.depends_on:
                predecessor = key_to_step[key]
                if id(predecessor) != id(step):
                    dependents[id(predecessor)].add(id(step))
                    in_degree[id(step)] += 1

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
                "Cycle detected among env-metric steps. Check the 'produces' and "
                "'depends_on' declarations for a circular dependency."
            )

        return [step_by_id[sid] for sid in sorted_ids]

    def apply(self, ctx: DRContext) -> None:
        for step in self.steps:
            step.apply(ctx)


# ---------------------------------------------------------------------------
# DR steps — Cube
# ---------------------------------------------------------------------------


class CubeColorDRStep(DRStep):
    """Randomise the cube diffuse colour for each resetting environment."""

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        stage = omni.usd.get_context().get_stage()
        for env_id in ctx.env_ids:
            color = torch.rand(3, device="cpu")
            rgb = (float(color[0]), float(color[1]), float(color[2]))

            mesh_prim_path = f"/World/envs/env_{env_id}/Object/geometry/mesh"
            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
            if not mesh_prim.IsValid():
                print(f"[CubeColorDRStep] Invalid mesh prim: {mesh_prim_path}")
                continue

            binding = UsdShade.MaterialBindingAPI(mesh_prim)
            material, _ = binding.ComputeBoundMaterial()
            if not material:
                print(f"[CubeColorDRStep] No material bound for env {env_id}")
                continue

            mat_prim = material.GetPrim()
            shader_prim = mat_prim.GetChild("Shader")
            shader = UsdShade.Shader(shader_prim)
            if not shader:
                print(f"[CubeColorDRStep] No Shader child under {mat_prim.GetPath()}")
                continue

            diffuse_input = shader.GetInput("diffuseColor")
            if not diffuse_input:
                print(
                    f"[CubeColorDRStep] Shader {shader_prim.GetPath()} has no "
                    "'diffuseColor' input"
                )
                continue

            diffuse_input.Set(Gf.Vec3f(*rgb))


class CubeSizeDRStep(DRStep):
    """Apply the per-env cube scale that was sampled by :class:`CubeDimsEnvMetricStep`.

    Reads ``env.env_metrics["dr_cube_scale"]`` (shape ``(num_envs, 3)``) and sets the
    ``XformOp.TypeScale`` on the cube prim for each resetting environment.
    """

    requires_env_metrics: frozenset[str] = frozenset({"dr_cube_scale"})

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        stage = omni.usd.get_context().get_stage()
        for env_id in ctx.env_ids:
            scale_xyz = env.env_metrics["dr_cube_scale"][env_id]  # (3,)
            sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])
            object_prim_path = f"/World/envs/env_{env_id}/Object"
            try:
                object_prim = stage.GetPrimAtPath(object_prim_path)
                if not object_prim.IsValid():
                    print(f"[CubeSizeDRStep] Invalid prim: {object_prim_path}")
                    continue
            except Exception as e:
                print(f"[CubeSizeDRStep] Error: {e}")
                continue
            try:
                xformable = UsdGeom.Xformable(object_prim)
                scale_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                        scale_op = op
                        break
                if scale_op is None:
                    scale_op = xformable.AddScaleOp()
                scale_op.Set(Gf.Vec3f(sx, sy, sz))
            except Exception as e:
                print(f"[CubeSizeDRStep] Error setting scale: {e}")


class CubePositionDRStep(DRStep):
    """Randomise the cube position (polar coordinates) for each resetting environment."""

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        pos_cfg = env.cfg.domain_randomization.cube.position_randomization
        env_ids = ctx.env_ids
        num_envs = len(env_ids)

        radius = sample_uniform(
            pos_cfg.radius_range[0],
            pos_cfg.radius_range[1],
            (num_envs, 1),
            device=env.device,
        )
        angle_rad = sample_uniform(
            math.radians(pos_cfg.angle_range[0]),
            math.radians(pos_cfg.angle_range[1]),
            (num_envs, 1),
            device=env.device,
        )
        obj_x = radius * torch.cos(angle_rad)
        obj_y = radius * torch.sin(angle_rad)
        obj_z = sample_uniform(
            pos_cfg.z_range[0], pos_cfg.z_range[1], (num_envs, 1), device=env.device
        )
        obj_pos = torch.cat([obj_x, obj_y, obj_z], dim=-1)
        obj_pos += env.scene.env_origins[env_ids]

        random_roll = sample_uniform(0, 2 * 3.14159, (num_envs,), device=env.device)
        random_pitch = sample_uniform(0, 2 * 3.14159, (num_envs,), device=env.device)
        random_yaw = sample_uniform(0, 2 * 3.14159, (num_envs,), device=env.device)
        obj_quat = math_utils.quat_from_euler_xyz(random_roll, random_pitch, random_yaw)

        root_state = env.cube.data.default_root_state[env_ids].clone()
        root_state[:, :3] = obj_pos
        root_state[:, 3:7] = obj_quat
        env.cube.write_root_pose_to_sim(root_state[:, :7], env_ids)
        env.cube.write_root_velocity_to_sim(root_state[:, 7:], env_ids)


# ---------------------------------------------------------------------------
# DR steps — Camera
# ---------------------------------------------------------------------------


class CameraPoseDRStep(DRStep):
    """Randomise the wrist camera mounting pose for each resetting environment."""

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        pose_cfg = env.cfg.domain_randomization.camera.pose
        stage = omni.usd.get_context().get_stage()

        for env_id in ctx.env_ids:
            pos_noise = sample_uniform(
                pose_cfg.position_noise_range[0],
                pose_cfg.position_noise_range[1],
                (3,),
                device="cpu",
            )
            rot_noise_deg = sample_uniform(
                pose_cfg.rotation_noise_deg_range[0],
                pose_cfg.rotation_noise_deg_range[1],
                (3,),
                device="cpu",
            )
            camera_prim_path = f"/World/envs/env_{env_id}/Robot/gripper/mountscrew/camera_mount/CameraXframe"
            try:
                camera_prim = stage.GetPrimAtPath(camera_prim_path)
                if not camera_prim.IsValid():
                    print(f"[CameraPoseDRStep] Invalid prim: {camera_prim_path}")
                    continue

                xformable = UsdGeom.Xformable(camera_prim)
                translate_op = None
                orient_op = None
                for op in xformable.GetOrderedXformOps():
                    op_type = op.GetOpType()
                    if (
                        op_type == UsdGeom.XformOp.TypeTranslate
                        and translate_op is None
                    ):
                        translate_op = op
                    elif op_type == UsdGeom.XformOp.TypeOrient and orient_op is None:
                        orient_op = op

                if translate_op is None:
                    print(
                        f"[CameraPoseDRStep] No translate op on camera at "
                        f"{camera_prim_path}; skipping position randomization."
                    )
                    raise ValueError("No translate op found")
                if orient_op is None:
                    print(
                        f"[CameraPoseDRStep] No orient op on camera at "
                        f"{camera_prim_path}; skipping rotation randomization."
                    )
                    raise ValueError("No orient op found")

                # Apply translation noise
                current_translate = translate_op.Get()
                if current_translate is None:
                    current_translate = Gf.Vec3d(0.0, 0.0, 0.0)
                translate_op.Set(
                    Gf.Vec3d(
                        current_translate[0] + pos_noise[0].item(),
                        current_translate[1] + pos_noise[1].item(),
                        current_translate[2] + pos_noise[2].item(),
                    )
                )

                # Apply rotation noise
                current_quat = orient_op.Get()
                if current_quat is None:
                    current_quat = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
                    print(
                        f"[CameraPoseDRStep] No existing orient on camera at "
                        f"{camera_prim_path}, assuming identity."
                    )
                    raise ValueError("No existing orient op value")

                current_rot = Gf.Rotation(current_quat)
                dx = rot_noise_deg[0].item()
                dy = rot_noise_deg[1].item()
                dz = rot_noise_deg[2].item()
                delta_rot = (
                    Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), dz)
                    * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), dy)
                    * Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), dx)
                )
                new_quat = (delta_rot * current_rot).GetQuat()
                if isinstance(current_quat, Gf.Quatf):
                    new_quat = Gf.Quatf(
                        float(new_quat.GetReal()),
                        Gf.Vec3f(*[float(c) for c in new_quat.GetImaginary()]),
                    )
                orient_op.Set(new_quat)

            except Exception as e:
                print(
                    f"[CameraPoseDRStep] Error modifying camera at "
                    f"{camera_prim_path}: {e}"
                )


# ---------------------------------------------------------------------------
# DR steps — Lighting
# ---------------------------------------------------------------------------


class WorldLightingDRStep(DRStep):
    """Randomise the global dome light once per reset batch (env 0 guard)."""

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        if 0 not in ctx.env_ids:
            return
        env = ctx.env
        wl_cfg = env.cfg.domain_randomization.world_lighting
        try:
            stage = omni.usd.get_context().get_stage()
            light_prim = stage.GetPrimAtPath("/World/Light")
            if not light_prim or not light_prim.IsValid():
                print("[WorldLightingDRStep] No valid light at /World/Light")
                return

            dome_light = UsdLux.DomeLight(light_prim)
            low, high = wl_cfg.intensity_range
            dome_light.GetIntensityAttr().Set(
                float(low + (high - low) * torch.rand(1).item())
            )

            base_color = torch.tensor([0.75, 0.75, 0.75], dtype=torch.float32)
            color = torch.clamp(
                base_color
                + (torch.rand(3, dtype=torch.float32) - 0.5)
                * 2
                * wl_cfg.color_variation,
                0.0,
                1.0,
            )
            dome_light.GetColorAttr().Set(
                Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            )
        except Exception as e:
            print(f"[WorldLightingDRStep] Error: {e}")


class EnvLightingDRStep(DRStep):
    """Randomise per-environment point lights for each resetting environment."""

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        el_cfg = env.cfg.domain_randomization.env_lighting
        stage = omni.usd.get_context().get_stage()
        p = 0.5

        for env_id in ctx.env_ids:
            light_prim_path = f"/World/envs/env_{env_id}/RandomPointLight"
            should_have_light = torch.rand(1, device="cuda").item() < p
            light_prim = stage.GetPrimAtPath(light_prim_path)

            if should_have_light:
                if not light_prim.IsValid():
                    light_prim = stage.DefinePrim(light_prim_path, "SphereLight")
                    point_light = UsdLux.SphereLight(light_prim)
                else:
                    point_light = UsdLux.SphereLight(light_prim)
                    light_prim.SetActive(True)

                x = (torch.rand(1).item() - 0.5) * 0.5
                y = (torch.rand(1).item() - 0.5) * 0.5
                z = (
                    torch.rand(1).item()
                    * (el_cfg.height_range[1] - el_cfg.height_range[0])
                    + el_cfg.height_range[0]
                )
                UsdGeom.XformCommonAPI(light_prim).SetTranslate(
                    Gf.Vec3d(float(x), float(y), float(z))
                )

                low, high = el_cfg.intensity_range
                point_light.GetIntensityAttr().Set(
                    float(low + (high - low) * torch.rand(1).item())
                )
                point_light.GetRadiusAttr().Set(random.uniform(0.1, 0.5))
                point_light.GetDiffuseAttr().Set(1.0)
                low_spec, high_spec = el_cfg.specular_range
                point_light.GetSpecularAttr().Set(
                    float(low_spec + (high_spec - low_spec) * torch.rand(1).item())
                )
                base_color = torch.tensor([0.75, 0.75, 0.75], dtype=torch.float32)
                color = torch.clamp(
                    base_color + (torch.rand(3) - 0.5) * 2 * el_cfg.color_variation,
                    0.0,
                    1.0,
                )
                point_light.GetColorAttr().Set(
                    Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
                )
            else:
                if light_prim.IsValid():
                    light_prim.SetActive(False)


# ---------------------------------------------------------------------------
# DR steps — Ground
# ---------------------------------------------------------------------------

_GROUND_MATERIAL_PATHS: list[str] = [
    "/World/Looks/GroundMat0",
    "/World/Looks/GroundMat1",
    "/World/Looks/GroundMat2",
]


class GroundMaterialDRStep(DRStep):
    """Swap the ground plane material once per reset batch (env 0 guard)."""

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        if 0 not in ctx.env_ids:
            return
        stage = omni.usd.get_context().get_stage()
        plane_path = "/World/ground/GroundPlane/CollisionPlane"
        plane_prim = stage.GetPrimAtPath(plane_path)
        if not plane_prim.IsValid():
            print(f"[GroundMaterialDRStep] Invalid ground plane prim: {plane_path}")
            return
        mat_path = random.choice(_GROUND_MATERIAL_PATHS)
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not mat_prim.IsValid():
            print(f"[GroundMaterialDRStep] Invalid material prim: {mat_path}")
            return
        UsdShade.MaterialBindingAPI(plane_prim).Bind(UsdShade.Material(mat_prim))


# ---------------------------------------------------------------------------
# DR steps — Distractors
# ---------------------------------------------------------------------------


class DistractorsDRStep(DRStep):
    """Reset and randomise every distractor object for each resetting environment.

    Handles default-state reset, colour randomisation, optional size
    randomisation, and position randomisation with an active/inactive mask
    (inactive distractors are hidden via USD visibility toggle).
    """

    requires_metrics: frozenset[str] = frozenset()

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        env_ids = ctx.env_ids
        stage = omni.usd.get_context().get_stage()

        for i, distractor in enumerate(env._distractors):
            distractor_name = f"distractor_{i}"

            # Reset to default state first
            default_root_state = distractor.data.default_root_state[env_ids].clone()
            default_root_state[:, :3] += env.scene.env_origins[env_ids]
            distractor.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            distractor.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

            # Randomize colour
            for env_id in env_ids:
                color = torch.rand(3, device="cpu")
                rgb = (float(color[0]), float(color[1]), float(color[2]))
                mesh_prim_path = (
                    f"/World/envs/env_{env_id}/{distractor_name}/geometry/mesh"
                )
                mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
                if not mesh_prim.IsValid():
                    print(f"[DistractorsDRStep] Invalid mesh prim: {mesh_prim_path}")
                    continue
                binding = UsdShade.MaterialBindingAPI(mesh_prim)
                material, _ = binding.ComputeBoundMaterial()
                if not material:
                    continue
                shader = UsdShade.Shader(material.GetPrim().GetChild("Shader"))
                if not shader:
                    continue
                diffuse_input = shader.GetInput("diffuseColor")
                if diffuse_input:
                    diffuse_input.Set(Gf.Vec3f(*rgb))

            # Randomize size
            if env.cfg.distractors.randomization.size_randomization_enabled:
                size_range = env.cfg.distractors.randomization.size_range
                size_factors = (
                    torch.rand(len(env_ids), device="cuda")
                    * (size_range[1] - size_range[0])
                    + size_range[0]
                )
                for idx, env_id in enumerate(env_ids):
                    size_factor = size_factors[idx].item()
                    prim_path = (
                        f"/World/envs/env_{env_id}/{distractor_name}/geometry/mesh"
                    )
                    try:
                        prim = stage.GetPrimAtPath(prim_path)
                        if not prim.IsValid():
                            print(f"[DistractorsDRStep] Invalid prim: {prim_path}")
                            continue
                        xformable = UsdGeom.Xformable(prim)
                        scale_op = None
                        for op in xformable.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                scale_op = op
                                break
                        if scale_op is None:
                            scale_op = xformable.AddScaleOp()
                        scale_op.Set(Gf.Vec3f(size_factor, size_factor, size_factor))
                    except Exception as e:
                        print(f"[DistractorsDRStep] Error setting scale: {e}")

            # Randomize position with active/inactive mask
            active_mask = (
                torch.rand(len(env_ids), device=env.device)
                < env.cfg.distractors.randomization.active_probability
            )
            env_ids_t = torch.as_tensor(env_ids, device=env.device)
            active_env_ids = env_ids_t[active_mask]
            inactive_env_ids = env_ids_t[~active_mask]

            if len(active_env_ids) > 0:
                num_active = len(active_env_ids)
                for env_id in active_env_ids.tolist():
                    prim_path = f"/World/envs/env_{env_id}/{distractor_name}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        UsdGeom.Imageable(prim).MakeVisible()
                pos_cfg = env.cfg.distractors.position
                obj_x = sample_uniform(
                    pos_cfg.x_range[0],
                    pos_cfg.x_range[1],
                    (num_active, 1),
                    device=env.device,
                )
                obj_y = sample_uniform(
                    pos_cfg.y_range[0],
                    pos_cfg.y_range[1],
                    (num_active, 1),
                    device=env.device,
                )
                obj_z = sample_uniform(
                    pos_cfg.z_range[0],
                    pos_cfg.z_range[1],
                    (num_active, 1),
                    device=env.device,
                )
                obj_pos = torch.cat([obj_x, obj_y, obj_z], dim=-1)
                obj_pos += env.scene.env_origins[active_env_ids]
                roll = sample_uniform(0, 2 * 3.14159, (num_active,), device=env.device)
                pitch = sample_uniform(0, 2 * 3.14159, (num_active,), device=env.device)
                yaw = sample_uniform(0, 2 * 3.14159, (num_active,), device=env.device)
                obj_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
                root_state = distractor.data.default_root_state[active_env_ids].clone()
                root_state[:, :3] = obj_pos
                root_state[:, 3:7] = obj_quat
                distractor.write_root_pose_to_sim(root_state[:, :7], active_env_ids)
                distractor.write_root_velocity_to_sim(root_state[:, 7:], active_env_ids)

            if len(inactive_env_ids) > 0:
                for env_id in inactive_env_ids.tolist():
                    prim_path = f"/World/envs/env_{env_id}/{distractor_name}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()


# ---------------------------------------------------------------------------
# EnvMetric steps
# ---------------------------------------------------------------------------


class CubeDimsEnvMetricStep(EnvMetricStep):
    """Samples a per-env, per-episode (X, Y, Z) scale for the cube.

    Produces ``env.env_metrics["dr_cube_scale"]`` of shape ``(num_envs, 3)``.
    Scales are drawn i.i.d. uniformly from
    ``cfg.domain_randomization.cube.size_range`` and stored as isotropic
    (X = Y = Z) vectors; independent per-axis non-isotropic scaling can be
    added later by extending this step.

    This step runs **before** :class:`CubeSizeDRStep` so that DR steps can
    consume the sampled values directly without re-sampling.
    """

    produces = frozenset({"dr_cube_scale"})
    depends_on = frozenset()

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        n = len(ctx.env_ids)
        env_ids_t = torch.as_tensor(list(ctx.env_ids), device=env.device)

        # Initialise full tensor on first call (all-ones = no scale change).
        if "dr_cube_scale" not in env.env_metrics:
            env.env_metrics["dr_cube_scale"] = torch.ones(
                env.num_envs, 3, device=env.device, dtype=torch.float32
            )

        size_range = env.cfg.domain_randomization.cube.size_range
        scalar_scales = (
            torch.rand(n, device=env.device) * (size_range[1] - size_range[0])
            + size_range[0]
        )  # (n,) — isotropic
        # Store as (n, 3) so downstream steps can handle per-axis scales uniformly.
        env.env_metrics["dr_cube_scale"][env_ids_t] = scalar_scales.unsqueeze(
            -1
        ).expand(-1, 3)


# Fixed clearance (metres) added above the tooth surface when placing the cube centroid.
_GZ_CLEARANCE: float = 0.001


class GripZoneOffsetEnvMetricStep(EnvMetricStep):
    """Computes the grip-zone offset in gripper EE frame using the ``gripperframe``
    XForm prim (located at the exact center of the tooth surface).

    ``gripperframe``'s local transform relative to the ``gripper`` link is read
    from the USD stage once on the first reset call and cached on the class.
    This avoids any hardcoded offsets.

    The grip zone is: gripperframe_origin_in_ee + tooth_normal_in_ee * offset_mag
    where offset_mag = DR_cube_half_height + _GZ_CLEARANCE.

    Produces ``env.env_metrics["grip_zone_offset"]`` of shape ``(num_envs, 3)``.
    """

    produces = frozenset({"grip_zone_offset"})
    depends_on = frozenset({"dr_cube_scale"})

    # Class-level cache — computed once from USD, shared across all instances.
    _gripperframe_pos_in_ee: torch.Tensor | None = None
    _tooth_normal_in_ee: torch.Tensor | None = None

    def _cache_gripperframe_transform(self, env) -> None:
        """Read gripperframe's local USD transform and cache in gripper-EE coordinates."""
        stage = omni.usd.get_context().get_stage()
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper/gripperframe")
        local_xform = UsdGeom.Xformable(prim).GetLocalTransformation()
        t = local_xform.ExtractTranslation()
        q = local_xform.ExtractRotationQuat()
        qi = q.GetImaginary()
        pos = torch.tensor(
            [t[0] * meters_per_unit, t[1] * meters_per_unit, t[2] * meters_per_unit],
            device=env.device,
            dtype=torch.float32,
        )
        # Isaac Lab uses wxyz quaternion convention; USD Quatd stores (imaginary, real).
        quat = torch.tensor(
            [float(q.GetReal()), float(qi[0]), float(qi[1]), float(qi[2])],
            device=env.device,
            dtype=torch.float32,
        )
        quat = quat / quat.norm()
        tooth_normal = quat_apply(
            quat.unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]], device=env.device, dtype=torch.float32),
        ).squeeze(0)

        GripZoneOffsetEnvMetricStep._gripperframe_pos_in_ee = pos
        GripZoneOffsetEnvMetricStep._tooth_normal_in_ee = tooth_normal

    def apply(self, ctx: DRContext) -> None:
        env = ctx.env
        env_ids_t = torch.as_tensor(list(ctx.env_ids), device=env.device)

        if GripZoneOffsetEnvMetricStep._gripperframe_pos_in_ee is None:
            self._cache_gripperframe_transform(env)

        # Initialise full tensor on first call.
        if "grip_zone_offset" not in env.env_metrics:
            env.env_metrics["grip_zone_offset"] = torch.zeros(
                env.num_envs, 3, device=env.device, dtype=torch.float32
            )

        # Per-episode DR height scale (isotropic DR, all dims equal; use Z axis).
        height_scale = env.env_metrics["dr_cube_scale"][env_ids_t, 2]  # (len,)
        offset_mag = (
            height_scale * (CUBE_DEFAULT_DIMS[2] / 2.0) + _GZ_CLEARANCE
        )  # (len,)

        val = GripZoneOffsetEnvMetricStep._gripperframe_pos_in_ee.unsqueeze(0).expand(
            len(env_ids_t), -1
        ) + GripZoneOffsetEnvMetricStep._tooth_normal_in_ee.unsqueeze(
            0
        ) * offset_mag.unsqueeze(
            -1
        )  # (len, 3)
        env.env_metrics["grip_zone_offset"][env_ids_t] = val


ALL_ENV_METRIC_STEPS: list[type[EnvMetricStep]] = [
    CubeDimsEnvMetricStep,
    GripZoneOffsetEnvMetricStep,
]


# ---------------------------------------------------------------------------
# DR pipeline factory
# ---------------------------------------------------------------------------


def build_dr_pipeline(cfg) -> DRPipeline:
    """Construct the domain-randomisation pipeline, including only enabled steps.

    Args:
        cfg: The ``So101LiftCubeCfg`` instance (``env.cfg``).
    """
    dr = cfg.domain_randomization
    steps: list[DRStep] = []

    # Cube
    if dr.cube.color_randomization_enabled:
        steps.append(CubeColorDRStep())
    if dr.cube.size_randomization_enabled:
        steps.append(CubeSizeDRStep())
    if dr.cube.position_randomization.enabled:
        steps.append(CubePositionDRStep())

    # Camera
    if dr.camera.pose.enabled:
        steps.append(CameraPoseDRStep())

    # Lighting
    if dr.world_lighting.enabled:
        steps.append(WorldLightingDRStep())
    if dr.env_lighting.enabled:
        steps.append(EnvLightingDRStep())

    # Ground
    if dr.ground.enabled:
        steps.append(GroundMaterialDRStep())

    # Distractors
    if cfg.distractors.randomization.enabled:
        steps.append(DistractorsDRStep())

    return DRPipeline(steps)


# ---------------------------------------------------------------------------
# EnvMetric pipeline factory
# ---------------------------------------------------------------------------


def build_env_metric_pipeline(cfg) -> EnvMetricPipeline:
    """Construct the :class:`EnvMetricPipeline` for the given config.

    Currently always includes :class:`CubeDimsEnvMetricStep` so that
    ``env.env_metrics["dr_cube_scale"]`` is always available (defaulting to
    ``1.0`` when size DR is disabled).

    Args:
        cfg: The ``So101LiftCubeCfg`` instance (``env.cfg``).
    """
    steps: list[EnvMetricStep] = [
        CubeDimsEnvMetricStep(),
        GripZoneOffsetEnvMetricStep(),
    ]
    return EnvMetricPipeline(steps)
