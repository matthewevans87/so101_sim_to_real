from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque

import omni.usd  # type: ignore
import torch
from pxr import UsdGeom  # type: ignore

from isaaclab.utils.math import quat_apply

from so101_rl.configurations.cube import CUBE_DEFAULT_DIMS
from so101_rl.dr_pipeline import DRContext


# ---------------------------------------------------------------------------
# Base classes
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
# Factory helper
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
