import torch
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils


def define_tip_markers() -> VisualizationMarkers:
    """A single small blue cube marker prototype for the gripper tip."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TipMarkers",
        markers={
            "tip": sim_utils.CuboidCfg(
                size=(0.01, 0.01, 0.01),  # 1 cm cube
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0)  # blue
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def define_camera_frame_markers() -> VisualizationMarkers:
    """Three cylinder markers for camera frame axes: X (red), Y (green), Z (blue)."""
    axis_length = 0.05  # 5cm long axes
    axis_radius = 0.002  # 2mm thick

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CameraFrameMarkers",
        markers={
            # X-axis: red cylinder along +X direction
            "x_axis": sim_utils.CylinderCfg(
                radius=axis_radius,
                height=axis_length,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)  # red
                ),
            ),
            # Y-axis: green cylinder along +Y direction
            "y_axis": sim_utils.CylinderCfg(
                radius=axis_radius,
                height=axis_length,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)  # green
                ),
            ),
            # Z-axis: blue cylinder along +Z direction
            "z_axis": sim_utils.CylinderCfg(
                radius=axis_radius,
                height=axis_length,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.0, 1.0)  # blue
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def define_gripper_arrow_markers() -> VisualizationMarkers:
    """Define a single arrow marker prototype, used for gripper->cube visualization."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/gripperMarkers",
        markers={
            "gripper_to_cube": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.05, 0.05, 0.10),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.0),  # orange-ish
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def visualize_tip_markers(
    markers: VisualizationMarkers,
    tip_pos: torch.Tensor,  # (N, 3)
    device: str,
) -> None:
    """Draw a small cube marker at each environment's gripper tip position."""
    N = tip_pos.shape[0]
    orientations = torch.zeros((N, 4), device=device)
    orientations[:, 0] = 1.0  # w=1 identity quaternion (wxyz)
    marker_indices = torch.zeros(N, dtype=torch.long, device=device)
    markers.visualize(
        translations=tip_pos, orientations=orientations, marker_indices=marker_indices
    )


def visualize_camera_frame_markers(
    markers: VisualizationMarkers,
    camera_pos_w: torch.Tensor,  # (N, 3)
    camera_quat_w: torch.Tensor,  # (N, 4) wxyz
    device: str,
    axis_length: float = 0.05,
) -> None:
    """Draw XYZ axis cylinders (red/green/blue) representing each environment's camera frame."""
    eps = 1e-6
    N = camera_pos_w.shape[0]
    half = axis_length / 2.0
    dtype = camera_pos_w.dtype

    x_local = torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(N, -1)
    y_local = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype).expand(N, -1)
    z_local = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=dtype).expand(N, -1)

    x_world = math_utils.quat_apply(camera_quat_w, x_local)
    y_world = math_utils.quat_apply(camera_quat_w, y_local)
    z_world = math_utils.quat_apply(camera_quat_w, z_local)

    x_pos = camera_pos_w + half * x_world
    y_pos = camera_pos_w + half * y_world
    z_pos = camera_pos_w + half * z_world

    # Cylinders default axis is +Y; build quats that rotate +Y to each world-frame axis.
    cyl_default = torch.zeros(N, 3, device=device, dtype=dtype)
    cyl_default[:, 1] = 1.0

    def _rot_y_to_dir(target: torch.Tensor) -> torch.Tensor:
        target = target / (target.norm(dim=-1, keepdim=True) + eps)
        dot = (cyl_default * target).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(dot)
        axis = torch.cross(cyl_default, target, dim=-1)
        axis_n = axis.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(axis)
        fallback[:, 0] = 1.0  # +X fallback for parallel/anti-parallel case
        axis = torch.where(axis_n > eps, axis / (axis_n + eps), fallback)
        return math_utils.quat_from_angle_axis(angle, axis)

    all_pos = torch.cat([x_pos, y_pos, z_pos], dim=0)  # (3N, 3)
    all_quat = torch.cat(
        [_rot_y_to_dir(x_world), _rot_y_to_dir(y_world), _rot_y_to_dir(z_world)], dim=0
    )
    marker_indices = torch.cat(
        [
            torch.zeros(N, dtype=torch.long, device=device),
            torch.ones(N, dtype=torch.long, device=device),
            torch.full((N,), 2, dtype=torch.long, device=device),
        ]
    )
    markers.visualize(all_pos, all_quat, marker_indices=marker_indices)


def define_grip_zone_markers() -> VisualizationMarkers:
    """A frame prim marker at the grip zone origin for each environment."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GripZoneMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def visualize_grip_zone_markers(
    markers: VisualizationMarkers,
    gz_pos_w: torch.Tensor,  # (N, 3) grip zone world positions
    gz_quat_w: torch.Tensor,  # (N, 4) wxyz — same orientation as gripper
    device: str,
) -> None:
    """Draw a frame prim marker at each environment's grip zone origin."""
    N = gz_pos_w.shape[0]
    marker_indices = torch.zeros(N, dtype=torch.long, device=device)
    markers.visualize(
        translations=gz_pos_w, orientations=gz_quat_w, marker_indices=marker_indices
    )


def visualize_gripper_arrow(
    markers: VisualizationMarkers,
    gripper_pos_w: torch.Tensor,  # (N, 3)
    gripper_quat_w: torch.Tensor,  # (N, 4) wxyz
    v_ee: torch.Tensor,  # (N, 3) direction vector in EE frame
    grip_zone_offset: tuple[
        float, float, float
    ],  # [x, y, z] offset for arrow placement
    device: str,
) -> None:
    """Draw an arrow at the gripper pointing along v_ee (expressed in EE frame)."""
    eps = 1e-6
    num_envs = gripper_pos_w.shape[0]

    v_ee = v_ee.reshape(num_envs, 3).to(device=device, dtype=gripper_pos_w.dtype)
    v_ee_unit = v_ee / (v_ee.norm(dim=-1, keepdim=True) + eps)

    v_world = math_utils.quat_apply(gripper_quat_w, v_ee_unit).reshape(num_envs, 3)
    v_world_norm = v_world / (v_world.norm(dim=-1, keepdim=True) + eps)

    # Build quaternion that rotates +X to v_world_norm
    base_dir = torch.zeros_like(v_world_norm)
    base_dir[:, 0] = 1.0

    dot = (base_dir * v_world_norm).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    axis = torch.cross(base_dir, v_world_norm, dim=-1)
    axis_norm = axis.norm(dim=-1, keepdim=True)
    default_axis = torch.zeros_like(axis)
    default_axis[:, 2] = 1.0  # world Z fallback
    axis = torch.where(axis_norm > eps, axis / (axis_norm + eps), default_axis)
    arrow_quat_w = math_utils.quat_from_angle_axis(angle, axis)

    offset = torch.tensor(list(grip_zone_offset), dtype=torch.float32, device=device)
    arrow_pos_w = gripper_pos_w + offset * v_world_norm

    marker_indices = torch.zeros(num_envs, dtype=torch.long, device=device)
    markers.visualize(arrow_pos_w, arrow_quat_w, marker_indices=marker_indices)


def define_goal_zone_markers(radius: float = 0.05) -> VisualizationMarkers:
    """A semi-transparent green sphere at the goal zone position for each environment.

    Args:
        radius: Sphere radius in metres — should match
            ``cfg.metrics.goal_zone_distance.distance_threshold``.
    """
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalZoneMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.8, 0.0),  # green
                    opacity=0.35,
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


def visualize_goal_zone_markers(
    markers: VisualizationMarkers,
    goal_pos_w: torch.Tensor,  # (N, 3) goal zone world positions
    device: str,
) -> None:
    """Draw a semi-transparent green sphere at each environment's goal zone."""
    N = goal_pos_w.shape[0]
    identity_quat = torch.zeros(N, 4, dtype=torch.float32, device=device)
    identity_quat[:, 0] = 1.0  # w=1 identity quaternion (wxyz)
    marker_indices = torch.zeros(N, dtype=torch.long, device=device)
    markers.visualize(
        translations=goal_pos_w,
        orientations=identity_quat,
        marker_indices=marker_indices,
    )
