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
