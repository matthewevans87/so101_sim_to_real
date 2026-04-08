from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import omni.usd  # type: ignore
import torch
from pxr import Gf, UsdGeom, UsdLux, UsdShade  # type: ignore

from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from so101_rl.tasks.direct.so101_lift_cube.so101_lift_cube_env import So101LiftCube


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


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
# Factory helper
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
