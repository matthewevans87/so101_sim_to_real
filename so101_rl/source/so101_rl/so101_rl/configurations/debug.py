from isaaclab.markers.config import (
    FRAME_MARKER_CFG,
    ISAAC_NUCLEUS_DIR,
    VisualizationMarkersCfg,
)
from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

VIS_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/GripZoneTransformer",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        ),
        "connecting_line": sim_utils.CylinderCfg(
            radius=0.002,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0), roughness=1.0
            ),
        ),
    },
)
