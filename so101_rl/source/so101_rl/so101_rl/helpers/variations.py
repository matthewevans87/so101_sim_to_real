import torch
from collections.abc import Sequence
import omni.usd  # type: ignore
import math
from isaaclab.utils.math import sample_uniform
from pxr import UsdShade, Gf, UsdLux, UsdGeom, Usd  # type: ignore
import random
import isaaclab.sim as sim_utils
import torch.nn.functional as F
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene


def apply_motion_blur(
    images: torch.Tensor,
    motion_blur_strength_range: tuple[float, float],
    motion_blur_kernel_size: int,
    device: str,
) -> torch.Tensor:
    """Apply directional motion blur to simulate camera motion."""
    # Random blur strength for this step
    blur_strength = (
        torch.rand(1, device=device)
        * (motion_blur_strength_range[1] - motion_blur_strength_range[0])
        + motion_blur_strength_range[0]
    )

    if blur_strength < 0.01:  # Skip if very weak
        return images

    kernel_size = motion_blur_kernel_size

    # Random blur direction: horizontal, vertical, or diagonal
    blur_type = torch.randint(0, 4, (1,), device=device).item()

    # Create blur kernel
    kernel = torch.zeros((kernel_size, kernel_size), device=device)
    if blur_type == 0:  # Horizontal
        kernel[kernel_size // 2, :] = 1.0
    elif blur_type == 1:  # Vertical
        kernel[:, kernel_size // 2] = 1.0
    elif blur_type == 2:  # Diagonal \
        for i in range(kernel_size):
            kernel[i, i] = 1.0
    else:  # Diagonal /
        for i in range(kernel_size):
            kernel[i, kernel_size - 1 - i] = 1.0

    kernel = kernel / kernel.sum()  # Normalize
    kernel = kernel * blur_strength  # Scale by strength

    # Add identity to preserve some sharpness
    identity = torch.zeros_like(kernel)
    identity[kernel_size // 2, kernel_size // 2] = 1.0 - blur_strength
    kernel = kernel + identity

    # Apply convolution to each channel
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)

    # Pad images
    padding = kernel_size // 2
    images_padded = torch.nn.functional.pad(
        images, (padding, padding, padding, padding), mode="replicate"
    )

    # Apply blur
    blurred = torch.nn.functional.conv2d(images_padded, kernel, groups=3)

    return blurred


def apply_jpeg_compression(
    images: torch.Tensor, jpeg_quality_range: tuple[int, int], device: str = "cpu"
) -> torch.Tensor:
    """Simulate JPEG compression artifacts."""
    # Random quality for this step
    quality = torch.randint(
        jpeg_quality_range[0],
        jpeg_quality_range[1] + 1,
        (1,),
        device=device,
    ).item()

    if quality >= 95:  # Skip if very high quality
        return images

    # Simplified JPEG simulation: add block artifacts
    # Real JPEG is complex (DCT, quantization), so we approximate
    block_size = 8

    # Quantization strength based on quality (inverse relationship)
    quant_strength = (100 - quality) / 100.0 * 0.1  # 0-0.1 range

    if quant_strength < 0.01:
        return images

    # Split into blocks and add noise to simulate quantization
    N, C, H, W = images.shape

    # Add blockiness by downsampling and upsampling
    scale_factor = max(1, int(4 * quant_strength))
    if scale_factor > 1:
        # Downsample
        small = torch.nn.functional.interpolate(
            images,
            scale_factor=1.0 / scale_factor,
            mode="bilinear",
            align_corners=False,
        )
        # Upsample back
        images = torch.nn.functional.interpolate(
            small, size=(H, W), mode="bilinear", align_corners=False
        )

    # Add slight quantization noise in blocks
    block_noise = torch.randn_like(images) * quant_strength * 0.05
    images = images + block_noise

    return images


# def randomize_distractors(
#     env_ids: torch.Tensor,
#     distractors: list,
#     distractor_active_prob: float,
#     distractor_x_range: tuple[float, float],
#     distractor_y_range: tuple[float, float],
#     distractor_z_height: float,
#     distractor_size_range: torch.Tensor,
#     device: str,
# ):
#     """Randomize positions and colors of distractor objects in the scene."""
#     num_envs = len(env_ids)
#     env_origins = scene.env_origins[env_ids]

#     xy_low = torch.tensor(
#         [distractor_x_range[0], distractor_y_range[0]],
#         device=device,
#     )
#     xy_high = torch.tensor(
#         [distractor_x_range[1], distractor_y_range[1]],
#         device=device,
#     )
#     z_height = distractor_z_height

#     quat_identity = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
#     inactive_offset = torch.tensor([0.0, 0.0, -1.0], device=device)

#     size_values = None
#     stage = None

#     size_factors = sample_uniform(
#         distractor_size_range[0],
#         distractor_size_range[1],
#         (num_envs,),
#         device=device,
#     )
#     size_values = size_factors.detach().cpu().tolist()
#     stage = omni.usd.get_context().get_stage()

#     for obj_idx, obj in enumerate(distractors):
#         # Decide which envs keep this distractor active
#         active_mask = torch.rand(num_envs, device=device) < distractor_active_prob

#         xy = sample_uniform(xy_low, xy_high, active_mask.sum(), device=device)
#         pos = env_origins.clone()
#         pos[active_mask, 0:2] = env_origins[active_mask, 0:2] + xy
#         pos[active_mask, 2] = env_origins[active_mask, 2] + z_height

#         # Park inactive distractors underground in their own env frames
#         pos[~active_mask, :] = env_origins[~active_mask, :] + inactive_offset

#         rot = quat_identity.repeat(num_envs, 1)
#         obj.write_root_pose_to_sim(pos, rot, env_ids=env_ids)  # type: ignore

#         colors = torch.rand(num_envs, 3, device=device)
#         colors[~active_mask, :] = 0.5
#         try:
#             obj.write(colors, env_ids=env_ids)
#         except AttributeError:
#             pass

#         if size_values is not None and stage is not None:
#             for idx_env, env_id in enumerate(env_ids):
#                 prim_path = f"/World/envs/env_{int(env_id)}/distractor_{obj_idx}"
#                 prim = stage.GetPrimAtPath(prim_path)
#                 if not prim.IsValid():
#                     continue

#                 xformable = UsdGeom.Xformable(prim)
#                 scale_op = None
#                 for op in xformable.GetOrderedXformOps():
#                     if op.GetOpType() == UsdGeom.XformOp.TypeScale:
#                         scale_op = op
#                         break

#                 if scale_op is None:
#                     scale_op = xformable.AddScaleOp()

#                 scale = size_values[idx_env]
#                 scale_op.Set(Gf.Vec3f(scale, scale, scale))


def randomize_rigid_object_position(
    env_ids: Sequence[int],
    scene: InteractiveScene,
    rigid_object: RigidObject,
    object_name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    device: str,
):
    """Randomize position and orientation of any rigid object.

    Args:
        env_ids: Environment IDs to randomize
        scene: Interactive scene containing environment origins
        rigid_object: The rigid object to randomize
        object_name: Name of the object (for prim path construction)
        x_range: (min, max) range for x position
        y_range: (min, max) range for y position
        z_range: (min, max) range for z position
        device: Device for tensor operations
    """
    num_envs = len(env_ids)
    obj_x = sample_uniform(
        x_range[0],
        x_range[1],
        (num_envs, 1),
        device=device,
    )
    obj_y = sample_uniform(
        y_range[0],
        y_range[1],
        (num_envs, 1),
        device=device,
    )

    obj_z = sample_uniform(
        z_range[0],
        z_range[1],
        (num_envs, 1),
        device=device,
    )

    obj_pos = torch.cat([obj_x, obj_y, obj_z], dim=-1)  # (num, 3)

    # account for per-environment origins
    obj_pos += scene.env_origins[env_ids]

    # Randomize orientation
    random_roll = sample_uniform(0, 2 * 3.14159, (num_envs,), device=device)
    random_pitch = sample_uniform(0, 2 * 3.14159, (num_envs,), device=device)
    random_yaw = sample_uniform(0, 2 * 3.14159, (num_envs,), device=device)

    obj_quat = math_utils.quat_from_euler_xyz(random_roll, random_pitch, random_yaw)

    root_state = rigid_object.data.default_root_state[env_ids].clone()
    root_state[:, :3] = obj_pos
    root_state[:, 3:7] = obj_quat

    rigid_object.write_root_pose_to_sim(root_state[:, :7], env_ids)
    rigid_object.write_root_velocity_to_sim(root_state[:, 7:], env_ids)


def randomize_rigid_object_position_polar(
    env_ids: Sequence[int],
    scene: InteractiveScene,
    rigid_object: RigidObject,
    object_name: str,
    radius_range: tuple[float, float],
    angle_range: tuple[float, float],
    z_range: tuple[float, float],
    device: str,
):
    """Randomize position and orientation of a rigid object using polar coordinates.

    Args:
        env_ids: Environment IDs to randomize
        scene: Interactive scene containing environment origins
        rigid_object: The rigid object to randomize
        object_name: Name of the object (for prim path construction)
        radius_range: (min, max) radial distance from robot base in meters
        angle_range: (min, max) angle in degrees, where 0° is directly in front of gripper
        z_range: (min, max) range for z position (height)
        device: Device for tensor operations
    """
    num_envs = len(env_ids)

    # Sample radius and angle
    radius = sample_uniform(
        radius_range[0],
        radius_range[1],
        (num_envs, 1),
        device=device,
    )

    # Convert angle from degrees to radians
    angle_rad = sample_uniform(
        math.radians(angle_range[0]),
        math.radians(angle_range[1]),
        (num_envs, 1),
        device=device,
    )

    # Convert polar to Cartesian (x = r*cos(θ), y = r*sin(θ))
    obj_x = radius * torch.cos(angle_rad)
    obj_y = radius * torch.sin(angle_rad)

    obj_z = sample_uniform(
        z_range[0],
        z_range[1],
        (num_envs, 1),
        device=device,
    )

    obj_pos = torch.cat([obj_x, obj_y, obj_z], dim=-1)  # (num, 3)

    # account for per-environment origins
    obj_pos += scene.env_origins[env_ids]

    # Randomize orientation
    random_roll = sample_uniform(0, 2 * 3.14159, (num_envs,), device=device)
    random_pitch = sample_uniform(0, 2 * 3.14159, (num_envs,), device=device)
    random_yaw = sample_uniform(0, 2 * 3.14159, (num_envs,), device=device)

    obj_quat = math_utils.quat_from_euler_xyz(random_roll, random_pitch, random_yaw)

    root_state = rigid_object.data.default_root_state[env_ids].clone()
    root_state[:, :3] = obj_pos
    root_state[:, 3:7] = obj_quat

    rigid_object.write_root_pose_to_sim(root_state[:, :7], env_ids)
    rigid_object.write_root_velocity_to_sim(root_state[:, 7:], env_ids)


def randomize_rigid_object_size(
    env_ids: Sequence[int], object_name: str, size_range: tuple[float, float]
):
    """Randomize rigid object dimensions.

    Args:
        env_ids: Environment IDs to randomize
        object_name: Name of the object (for prim path construction, e.g., "Object")
        size_range: Direct size range (e.g., (0.5, 2.0) for scaling between 50% and 200%)

    Note: Provide either size_variation OR size_range, not both.
    """
    stage = omni.usd.get_context().get_stage()

    # Generate all random scales at once (NO .item() calls!)
    num_envs = len(env_ids)

    # Sample uniformly from the provided range
    size_factors = (
        torch.rand(num_envs, device="cuda") * (size_range[1] - size_range[0])
        + size_range[0]
    )

    # shape: (num_envs,) on GPU, e.g., [0.95, 1.03, 0.91, ...]

    for i, env_id in enumerate(env_ids):
        size_factor = size_factors[i].item()  # Single .item() at end (acceptable)

        object_prim_path = f"/World/envs/env_{env_id}/{object_name}"
        try:
            object_prim = stage.GetPrimAtPath(object_prim_path)
            if not object_prim.IsValid():
                print(f"[randomize_rigid_object_size] Invalid prim: {object_prim_path}")
                continue
        except Exception as e:
            print(f"[randomize_rigid_object_size] Error: {e}")
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
            scale_op.Set(Gf.Vec3f(size_factor, size_factor, size_factor))

        except Exception as e:
            print(f"[randomize_rigid_object_size] Error setting scale: {e}")


def randomize_camera_pose(
    env_ids: Sequence[int],
    camera_pos_noise_range: tuple[float, float],
    camera_rot_noise_deg_range: tuple[float, float],
    device: str = "cpu",
):
    """Randomize camera mounting position and orientation for given env IDs.

    - Position noise: applied in meters to the existing translate op.
    - Rotation noise: applied as a small delta quaternion on top of the existing orient op.
    """

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids:
        # Position noise (e.g., ±0.005 m in each axis)
        pos_noise = sample_uniform(
            camera_pos_noise_range[0],
            camera_pos_noise_range[1],
            (3,),
            device=device,
        )

        # Rotation noise (±X° in each axis, in DEGREES)
        rot_noise_deg = sample_uniform(
            camera_rot_noise_deg_range[0],
            camera_rot_noise_deg_range[1],
            (3,),
            device=device,
        )

        camera_prim_path = f"/World/envs/env_{env_id}/Robot/gripper/gripper_camera"

        try:
            camera_prim = stage.GetPrimAtPath(camera_prim_path)
            if not camera_prim.IsValid():
                print(f"[randomize_camera_pose] Invalid prim: {camera_prim_path}")
                continue

            xformable = UsdGeom.Xformable(camera_prim)

            # ------------------------------------------------------------------
            # Find existing translate and orient ops
            # ------------------------------------------------------------------
            translate_op = None
            orient_op = None

            for op in xformable.GetOrderedXformOps():
                op_type = op.GetOpType()
                if op_type == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                    translate_op = op
                elif op_type == UsdGeom.XformOp.TypeOrient and orient_op is None:
                    orient_op = op

            if translate_op is None:
                print(
                    f"[randomize_camera_pose] No translate op on camera at "
                    f"{camera_prim_path}; skipping position randomization."
                )
                raise ValueError("No translate op found")

            if orient_op is None:
                print(
                    f"[randomize_camera_pose] No orient op on camera at "
                    f"{camera_prim_path}; skipping rotation randomization."
                )
                raise ValueError("No orient op found")

            # ------------------------------------------------------------------
            # Apply translation noise
            # ------------------------------------------------------------------
            if translate_op is not None:
                current_translate = translate_op.Get()
                if current_translate is None:
                    current_translate = Gf.Vec3d(0.0, 0.0, 0.0)

                new_translate = Gf.Vec3d(
                    current_translate[0] + pos_noise[0].item(),
                    current_translate[1] + pos_noise[1].item(),
                    current_translate[2] + pos_noise[2].item(),
                )
                translate_op.Set(new_translate)

            # ------------------------------------------------------------------
            # Apply rotation noise via quaternion (on orient op)
            # ------------------------------------------------------------------
            if orient_op is not None:
                current_quat = orient_op.Get()
                if current_quat is None:
                    # Identity quaternion if nothing is set yet
                    current_quat = Gf.Quatf(1.0, 0.0, 0.0, 0.0)
                    print(
                        f"[randomize_camera_pose] No existing orient on camera at "
                        f"{camera_prim_path}, assuming identity."
                    )
                    raise ValueError("No existing orient op value")

                # Convert current quaternion to a Gf.Rotation
                current_rot = Gf.Rotation(current_quat)

                # Build a small delta rotation from the XYZ Euler noise (degrees)
                dx = rot_noise_deg[0].item()
                dy = rot_noise_deg[1].item()
                dz = rot_noise_deg[2].item()

                # Apply in Z * Y * X order (you can change this if you prefer)
                rot_x = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), dx)
                rot_y = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), dy)
                rot_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), dz)
                delta_rot = rot_z * rot_y * rot_x

                new_rot = delta_rot * current_rot
                new_quat = new_rot.GetQuat()

                # Match original quaternion precision type (Quatf vs Quatd)
                if isinstance(current_quat, Gf.Quatf):
                    new_quat = Gf.Quatf(
                        float(new_quat.GetReal()),
                        Gf.Vec3f(*[float(c) for c in new_quat.GetImaginary()]),
                    )

                orient_op.Set(new_quat)

                # print(
                #     f"[randomize_camera_pose] Applied noise to camera at {camera_prim_path}: "
                #     f"pos_noise={pos_noise}, rot_noise_deg={rot_noise_deg}"
                # )

        except Exception as e:
            print(
                f"[randomize_camera_pose] Error modifying camera at "
                f"{camera_prim_path}: {e}"
            )
            # Keep training from dying if one env/camera misbehaves
            pass


def gaussian_blur_rgb(
    rgb: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Apply a channel-wise Gaussian blur to an (N, 3, H, W) tensor."""
    device = rgb.device
    # 1D Gaussian
    x = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2.0
    gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gauss_1d = gauss_1d / gauss_1d.sum()

    # Outer product → 2D kernel
    kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]  # (K, K)
    kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size)  # (C, 1, K, K)

    # Depthwise convolution: one kernel per channel
    return F.conv2d(
        rgb,
        kernel_2d,
        padding=kernel_size // 2,
        groups=3,
    )


def cheap_webcam_effect(rgb: torch.Tensor) -> torch.Tensor:
    """
    rgb: (N, 3, H, W) in [0,1]
    Mimic a low-res sensor + resize.
    """
    N, C, H, W = rgb.shape

    # pick a downscale factor (e.g., 0.4–0.7)
    scale = 0.4 + 0.3 * torch.rand(1, device=rgb.device).item()
    H_low = int(H * scale)
    W_low = int(W * scale)

    # Downsample (area or bilinear)
    low_res = F.interpolate(rgb, size=(H_low, W_low), mode="area")

    # Upsample back
    upsampled = F.interpolate(
        low_res, size=(H, W), mode="bilinear", align_corners=False
    )

    return upsampled


def randomize_rigid_object_color(env_ids, object_name: str):
    """Randomize rigid object color by modifying UsdPreviewSurface.inputs:diffuseColor.

    Args:
        env_ids: Environment IDs to randomize
        object_name: Name of the object (for prim path construction, e.g., "Object")
    """
    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids:
        # random RGB in [0, 1] on CPU (good for USD)
        color = torch.rand(3, device="cpu")
        rgb = (float(color[0]), float(color[1]), float(color[2]))

        object_prim_path = f"/World/envs/env_{env_id}/{object_name}"
        mesh_prim_path = object_prim_path + "/geometry/mesh"
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)

        if not mesh_prim.IsValid():
            print(f"[randomize_rigid_object_color] Invalid mesh prim: {mesh_prim_path}")
            continue

        # Get bound material
        binding = UsdShade.MaterialBindingAPI(mesh_prim)
        material, _ = binding.ComputeBoundMaterial()
        if not material:
            print(f"[randomize_rigid_object_color] No material bound for env {env_id}")
            continue

        mat_prim = material.GetPrim()
        shader_prim = mat_prim.GetChild("Shader")
        shader = UsdShade.Shader(shader_prim)
        if not shader:
            print(
                f"[randomize_rigid_object_color] No Shader child under {mat_prim.GetPath()}"
            )
            continue

        # This is the exact input from your debug: inputs:diffuseColor
        diffuse_input = shader.GetInput("diffuseColor")
        if not diffuse_input:
            print(
                f"[randomize_rigid_object_color] Shader {shader_prim.GetPath()} "
                "has no 'diffuseColor' input"
            )
            continue

        diffuse_input.Set(Gf.Vec3f(*rgb))


def randomize_ground_appearance(
    env_ids: Sequence[int], ground_color_randomization, device: str = "cpu"
):
    """Randomize ground plane color."""

    if not ground_color_randomization:
        return

    # Ground is shared across all environments, so randomize once per reset
    if len(env_ids) == 0:
        return

    # Random ground color
    color = torch.rand(3, device=device)

    try:
        stage = omni.usd.get_context().get_stage()
        ground_prim_path = "/World/ground"
        ground_prim = stage.GetPrimAtPath(ground_prim_path)
        if ground_prim.IsValid():
            # Try to access ground plane mesh
            mesh_path = ground_prim_path + "/geom/mesh"
            mesh_prim = stage.GetPrimAtPath(mesh_path)
            if mesh_prim.IsValid():

                material = UsdShade.MaterialBindingAPI(
                    mesh_prim
                ).ComputeBoundMaterial()[0]
                if material:
                    shader = material.GetPrim().GetChild("Shader")
                    if shader:
                        shader.GetAttribute("inputs:diffuse_tint").Set(
                            Gf.Vec3f(color[0].item(), color[1].item(), color[2].item())
                        )
    except Exception as e:
        print(f"[randomize_ground_appearance] Error: {e}")
        pass


GROUND_MATERIAL_CFGS = [
    sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),  # dark gray
    sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.35, 0.3)),  # warm concrete
    sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.35, 0.25)),  # greenish
]

GROUND_MATERIAL_PATHS = [
    "/World/Looks/GroundMat0",
    "/World/Looks/GroundMat1",
    "/World/Looks/GroundMat2",
]


# # TO BE CALLED IN SETUP_SCENE
# def create_ground_materials():
#     """Create a small library of ground materials under /World/Looks."""
#     stage = omni.usd.get_context().get_stage()

#     for cfg, path in zip(GROUND_MATERIAL_CFGS, GROUND_MATERIAL_PATHS):
#         # If it already exists (e.g. reload), skip
#         prim = stage.GetPrimAtPath(path)
#         if prim.IsValid():
#             continue
#         # Create a PreviewSurface material at `path`
#         func(path, cfg)


def randomize_ground_material():
    stage = omni.usd.get_context().get_stage()

    plane_path = "/World/ground/GroundPlane/CollisionPlane"
    plane_prim = stage.GetPrimAtPath(plane_path)
    if not plane_prim.IsValid():
        print(f"[randomize_ground_material] Invalid ground plane prim: {plane_path}")
        return

    # Choose one of your predefined material prims
    mat_path = random.choice(GROUND_MATERIAL_PATHS)
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.IsValid():
        print(f"[randomize_ground_material] Invalid material prim: {mat_path}")
        return

    material = UsdShade.Material(mat_prim)
    UsdShade.MaterialBindingAPI(plane_prim).Bind(material)


def randomly_placed_lights(
    env_ids,
    height_range,
    light_intensity_range,
    light_color_variation,
    light_specular_range,
    p: float = 0.5,
):
    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids:
        light_prim_path = f"/World/envs/env_{env_id}/RandomPointLight"

        # Check if we want a light this episode
        should_have_light = torch.rand(1, device="cuda").item() < p

        light_prim = stage.GetPrimAtPath(light_prim_path)

        if should_have_light:
            # Create light if it doesn't exist
            if not light_prim.IsValid():
                light_prim = stage.DefinePrim(light_prim_path, "SphereLight")
                point_light = UsdLux.SphereLight(light_prim)
            else:
                # Reuse existing light
                point_light = UsdLux.SphereLight(light_prim)
                # Make it active (if it was hidden)
                light_prim.SetActive(True)

            # Randomize position
            x = (torch.rand(1).item() - 0.5) * 0.5
            y = (torch.rand(1).item() - 0.5) * 0.5
            z = (
                torch.rand(1).item() * (height_range[1] - height_range[0])
                + height_range[0]
            )

            xform_api = UsdGeom.XformCommonAPI(light_prim)
            xform_api.SetTranslate(Gf.Vec3d(float(x), float(y), float(z)))

            # Randomize intensity
            low, high = light_intensity_range
            intensity = low + (high - low) * torch.rand(1).item()
            point_light.GetIntensityAttr().Set(float(intensity))

            # Radius and specular
            point_light.GetRadiusAttr().Set(random.uniform(0.1, 0.5))
            point_light.GetDiffuseAttr().Set(1.0)
            low_spec, high_spec = light_specular_range
            specular = low_spec + (high_spec - low_spec) * torch.rand(1).item()
            point_light.GetSpecularAttr().Set(float(specular))

            # Color tint
            base_color = torch.tensor([0.75, 0.75, 0.75], dtype=torch.float32)
            color_offset = (torch.rand(3) - 0.5) * 2 * light_color_variation
            color = torch.clamp(base_color + color_offset, 0.0, 1.0)
            point_light.GetColorAttr().Set(
                Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
            )
        else:
            # Hide light instead of removing it
            if light_prim.IsValid():
                light_prim.SetActive(False)


def randomize_lighting(
    light_intensity_range: tuple[float, float],
    light_color_variation: float,
):
    """Randomize global dome light intensity and color."""

    try:
        stage = omni.usd.get_context().get_stage()
        light_prim_path = "/World/Light"
        light_prim = stage.GetPrimAtPath(light_prim_path)

        if not light_prim or not light_prim.IsValid():
            print(f"[randomize_lighting] No valid light at {light_prim_path}")
            return

        dome_light = UsdLux.DomeLight(light_prim)

        # Random intensity in [low, high]
        low, high = light_intensity_range
        intensity = low + (high - low) * torch.rand(1).item()
        dome_light.GetIntensityAttr().Set(float(intensity))

        # Random color tint around neutral gray
        base_color = torch.tensor([0.75, 0.75, 0.75], dtype=torch.float32)
        color_variation_vec = (
            (torch.rand(3, dtype=torch.float32) - 0.5) * 2 * light_color_variation
        )
        color = torch.clamp(base_color + color_variation_vec, 0.0, 1.0)

        dome_light.GetColorAttr().Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )

    except Exception as e:
        print(f"[randomize_lighting] Error: {e}")
