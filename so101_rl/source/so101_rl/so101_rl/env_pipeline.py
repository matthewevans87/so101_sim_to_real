from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_unique
import isaaclab.utils.math as math_utils

from so101_rl.configurations.camera import CAMERA_ROTATION_QUAT_WXYZ, CAMERA_TRANSLATE_VEC
from so101_rl.configurations.cube import CUBE_RESTING_HEIGHT
from so101_rl.helpers.utils import assert_tensor

if TYPE_CHECKING:
    from .so101_lift_cube_env import So101LiftCube


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

    obs_dim: int = 0
    """Number of columns this step contributes when included in an observation vector.
    0 means the step is not intended for direct use in observations.
    Scalars-per-env should set 1; vectors of length K should set K."""

    @abstractmethod
    def compute(self, ctx: StepContext) -> None:
        ...


class MetricPipeline:
    """Accepts a set of :class:`MetricStep` objects in any order, topologically sorts
    them by their ``produces`` / ``depends_on`` declarations, and runs them in dependency
    order each step.

    Raises:
        ValueError: If a declared dependency is not produced by any step, or if there
            is a dependency cycle among the steps.
    """

    def __init__(self, steps: list[MetricStep]) -> None:
        self.steps = self._toposort(steps)

    @staticmethod
    def _toposort(steps: list[MetricStep]) -> list[MetricStep]:
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
        for step in steps:
            for key in step.depends_on:
                if key not in key_to_step:
                    raise ValueError(
                        f"{type(step).__name__} depends on metric key '{key}', "
                        f"but no step produces it."
                    )

        # Build adjacency list: predecessor_step -> {dependent_steps}
        # and in-degree counts for Kahn's algorithm.
        dependents: dict[int, set[int]] = defaultdict(set)  # id(step) -> set of id(step)
        in_degree: dict[int, int] = {id(s): 0 for s in steps}
        step_by_id: dict[int, MetricStep] = {id(s): s for s in steps}

        for step in steps:
            for key in step.depends_on:
                predecessor = key_to_step[key]
                if id(predecessor) != id(step):
                    dependents[id(predecessor)].add(id(step))
                    in_degree[id(step)] += 1

        # Kahn's algorithm
        queue: deque[int] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )
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

    @abstractmethod
    def compute(self, ctx: StepContext) -> torch.Tensor:
        """Return reward tensor of shape ``(num_envs,)``."""
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

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        if "log" not in env.extras:
            env.extras["log"] = {}

        total = torch.zeros(env.num_envs, device=env.device)
        for step in self.steps:
            rew = step.compute(ctx)
            env.extras["log"][f"Episode_Reward/{step.name}"] = rew.mean()
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


class VGripZoneToCubeEEMetricStep(MetricStep):
    produces = frozenset({"v_grip_zone_to_cube_ee"})
    depends_on = frozenset({"cube_pos_ee"})
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = ctx.metrics["cube_pos_ee"] - env._grip_zone_offset
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["v_grip_zone_to_cube_ee"] = val


class CubePosGZMetricStep(MetricStep):
    produces = frozenset({"cube_pos_gz"})
    depends_on = frozenset()
    obs_dim = 3

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = env.grip_zone_tf.data.target_pos_source[:, 0, :]
        assert_tensor(val, (env.num_envs, 3), torch.float32)
        ctx.metrics["cube_pos_gz"] = val


class CubeRot6DGZMetricStep(MetricStep):
    produces = frozenset({"cube_rot6d_gz"})
    depends_on = frozenset()
    obs_dim = 6

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        q_gz = quat_unique(env.grip_zone_tf.data.target_quat_source[:, 0, :])
        R = matrix_from_quat(q_gz)
        val = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
        assert_tensor(val, (env.num_envs, 6), torch.float32)
        ctx.metrics["cube_rot6d_gz"] = val


class GripZoneCubeDistanceMetricStep(MetricStep):
    produces = frozenset({"grip_zone_cube_distance"})
    depends_on = frozenset({"v_grip_zone_to_cube_ee"})
    obs_dim = 1

    def compute(self, ctx: StepContext) -> None:
        env = ctx.env
        val = ctx.metrics["v_grip_zone_to_cube_ee"].norm(dim=-1, keepdim=True).squeeze(-1)
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
        val = ctx.metrics["cube_height_w"] / env.cfg.rewards.success_lift_fraction_terminal.height_threshold
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
    depends_on = frozenset({"is_cube_in_grip_position", "gripper_cube_contact_force_magnitude"})
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


# ---------------------------------------------------------------------------
# Reward steps — Primary
# ---------------------------------------------------------------------------

class DistanceRewardStep(RewardStep):
    name = "rew_distance"
    requires_metrics = frozenset({"grip_zone_cube_distance"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return ctx.metrics["grip_zone_cube_distance"] * env.cfg.rewards.distance.scale


class GripCubeRewardStep(RewardStep):
    name = "rew_grip_cube"
    requires_metrics = frozenset({"is_cube_in_grip_position", "gripper_cube_contact_force_magnitude"})

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

        gripper_pos = env.gripper_tf.data.source_pos_w
        gripper_quat = env.robot.data.body_quat_w[:, env._ee_body_idx[0], :]

        camera_offset = (
            torch.tensor(CAMERA_TRANSLATE_VEC, device=env.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(env.num_envs, 3)
        )
        camera_pos_w = gripper_pos + quat_apply(gripper_quat, camera_offset)

        camera_rot_offset = (
            torch.tensor(CAMERA_ROTATION_QUAT_WXYZ, device=env.device, dtype=torch.float32)
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

        cam_forward_norm = cam_forward_w / (torch.linalg.norm(cam_forward_w, dim=-1, keepdim=True) + eps)
        vec_to_cube_norm = vec_to_cube / (torch.linalg.norm(vec_to_cube, dim=-1, keepdim=True) + eps)

        lookat_factor = torch.clamp((cam_forward_norm * vec_to_cube_norm).sum(dim=-1), min=0.0, max=1.0)
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
        gripper_pos = env.joint_pos[:, env._ee_body_idx]
        gripper_close_error = torch.abs(gripper_pos - env.cfg.rewards.close_gripper.close_target)
        fraction_to_target = (
            1.0 - (gripper_close_error / env.cfg.rewards.close_gripper.max_open).squeeze(-1)
        )
        return (
            ctx.metrics["is_cube_in_grip_position"]
            * fraction_to_target
            * env.cfg.rewards.close_gripper.scale
        )


class GripperForceRewardStep(RewardStep):
    name = "rew_gripper_force"
    requires_metrics = frozenset({"gripper_cube_contact_force_magnitude", "is_cube_in_grip_position"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        target = env.cfg.rewards.gripper_force.force_target
        force_error = torch.abs(ctx.metrics["gripper_cube_contact_force_magnitude"] - target)
        rew = torch.exp(-force_error / (target + 1e-6)) * env.cfg.rewards.gripper_force.scale
        return rew * ctx.metrics["is_cube_in_grip_position"].float()


class VantageRewardStep(RewardStep):
    name = "rew_vantage"
    requires_metrics = frozenset({"is_cube_gripped"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        cfg = env.cfg.rewards.vantage

        cube_gripper_dist = torch.linalg.norm(
            env.gripper_tf.data.source_pos_w - env.gripper_tf.data.target_pos_w[:, 0, :],
            dim=-1,
        )
        is_far = cube_gripper_dist > cfg.far_distance_threshold

        # --- Raw vantage value (mirrors _get_rew_vantage logic) ---
        d = cube_gripper_dist
        far_enough = d > cfg.min_distance_threshold

        dist_reward = torch.exp(
            -((d - cfg.ideal_distance) ** 2) / (2 * cfg.ideal_distance_sigma ** 2)
        )

        h_above_cube = (
            env.gripper_tf.data.source_pos_w[:, 2]
            - env.gripper_tf.data.target_pos_w[:, 0, 2]
        )
        height_reward = torch.where(
            h_above_cube >= 0,
            torch.exp(-((h_above_cube - cfg.ideal_height) ** 2) / (2 * cfg.ideal_height_sigma ** 2)),
            torch.exp(-((h_above_cube) ** 2) / (2 * (cfg.ideal_height_sigma / 2) ** 2)) * 0.3,
        )

        gripper_roll_error = (
            torch.abs(env.robot.data.joint_pos[:, env._wrist_roll_idx] - math.radians(-90.0))
            / math.radians(-90.0)
        )

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
        return env.cfg.rewards.action.scale * torch.sum(env.actions ** 2, dim=-1)


class EELinearSpeedRewardStep(RewardStep):
    name = "rew_ee_linear_speed"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        ee_lin_vel_w = env.robot.data.body_lin_vel_w[:, env._ee_body_idx[0], :]
        speed = torch.linalg.norm(ee_lin_vel_w, dim=-1)
        v_safe = env.cfg.rewards.ee_linear_speed.safe_speed
        v_excess = torch.clamp(speed - v_safe, min=0.0)
        return env.cfg.rewards.ee_linear_speed.scale * (v_excess + v_excess ** 2)


class JointSpeedRewardStep(RewardStep):
    name = "rew_joint_speed"
    requires_metrics = frozenset()

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        joint_speed = torch.abs(env.joint_vel[:, env._dof_idx])
        return env.cfg.rewards.joint_speed.scale * torch.sum(joint_speed ** 2, dim=-1)


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

class SuccessTouchTerminalRewardStep(RewardStep):
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


class SuccessLiftFractionTerminalRewardStep(RewardStep):
    name = "rew_success_lift_fraction_terminal"
    requires_metrics = frozenset({"is_success_lift_fraction_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = ctx.metrics["is_success_lift_fraction_terminal"]
        return torch.where(
            flag >= 1.0,
            torch.full_like(flag.float(), env.cfg.rewards.success_lift_fraction_terminal.scale),
            torch.zeros(env.num_envs, device=env.device),
        )


class SuccessPointAtCubeTerminalRewardStep(RewardStep):
    name = "rew_success_point_at_cube_terminal"
    requires_metrics = frozenset({"is_success_point_at_cube_terminal"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        flag = ctx.metrics["is_success_point_at_cube_terminal"]
        return torch.where(
            flag >= 1.0,
            torch.full_like(flag.float(), env.cfg.rewards.success_point_at_cube_terminal.scale),
            torch.zeros(env.num_envs, device=env.device),
        )


class SafetyTouchTableTerminalRewardStep(RewardStep):
    name = "rew_safety_touch_table_terminal"
    requires_metrics = frozenset({"is_table_touched"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.where(
            ctx.metrics["is_table_touched"],
            torch.tensor(env.cfg.rewards.safety_touch_table_terminal.scale, device=env.device, dtype=torch.float32),
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
        )


class SafetyTouchTableRewardStep(RewardStep):
    name = "rew_safety_touch_table"
    requires_metrics = frozenset({"is_table_touched"})

    def compute(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.where(
            ctx.metrics["is_table_touched"],
            torch.tensor(env.cfg.rewards.safety_touch_table.scale, device=env.device, dtype=torch.float32),
            torch.tensor(0.0, device=env.device, dtype=torch.float32),
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
    VGripZoneToCubeEEMetricStep,
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
    """
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
            # Will be caught as an unsatisfied dependency by MetricPipeline._toposort.
            continue
        for dep_key in key_to_cls[key].depends_on:
            if dep_key not in needed_keys:
                needed_keys.add(dep_key)
                frontier.add(dep_key)

    # Collect the unique step classes required.
    needed_cls: set[type[MetricStep]] = {
        key_to_cls[k] for k in needed_keys if k in key_to_cls
    }

    return MetricPipeline([cls() for cls in needed_cls])


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

    return RewardPipeline(steps)
