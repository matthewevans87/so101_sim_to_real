import os

from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

WORKSPACE_PATH = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")

# Camera calibration from actual hardware:
# - Mount angle: 146.31° (bend on mount)
# - Mount distance: 46.5mm from bottom of mount to center of camera lens
# - Translation: (y=64mm up, z=-35mm back) in gripper frame
# - Rotation: ~146.31° around Y-axis (quaternion verified from Isaac Sim calibration)
# Reference quaternion from matching project (w,x,y,z): (0.0, 0.0, 0.29237170472273677, 0.9563047559630354)
CAMERA_TRANSLATE_VEC = (0, 0.06400000303983688, -0.03500000014901161)
CAMERA_ROTATION_QUAT_WXYZ = (0.0, 0.0, 0.29237170472273677, 0.9563047559630354)

# Camera mount (SO-ARM101_camera_wrist_mount) USD asset and gripper-relative transform.
# The mount Xform prim is placed at the same gripper-relative position/rotation as the
# camera lens — the mount USD mesh origin coincides with the camera lens position.
# After visual inspection in Isaac Sim the values may be adjusted independently.
# Path resolves via ISAAC_LAB_WORKSPACE_PATH (same pattern as so101.py).
CAMERA_MOUNT_USD_PATH = os.path.join(WORKSPACE_PATH, "assets/robots/camera_mount.usd")
CAMERA_MOUNT_RELATIVE_POS = CAMERA_TRANSLATE_VEC
CAMERA_MOUNT_RELATIVE_QUAT_WXYZ = CAMERA_ROTATION_QUAT_WXYZ

CAMERA_CFG = CameraCfg(
    # Camera is parented under CameraXframe in the robot USD, which has the full
    # gripper-relative transform baked in.  The camera offset is identity here.
    prim_path="{ENV_REGEX_NS}/Robot/gripper/mountscrew/camera_mount/CameraXframe/gripper_camera",
    # resnet18 input size is 224x224, but the pytorch implementation can accept lower inputs; here we use 96x96 for efficiency
    # 30 Hz (rather than the simulations 120 Hz) is a realistic camera frame rate and saves on compute
    update_period=(1.0 / 30.0),
    height=128,
    width=128,
    # update_period=0,
    # height=224,
    # width=224,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=6.12,  # 6.12mm from calibration
        focus_distance=400.0,
        horizontal_aperture=6.3,  # 6.3mm sensor width
        clipping_range=(0.01, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(
            0.0,
            0.0,
            0.0,
        ),  # identity — mount Xform carries the gripper-relative transform
        rot=(1.0, 0.0, 0.0, 0.0),  # identity (w, x, y, z)
        convention="opengl",
    ),
)

# TiledCamera variant of CAMERA_CFG — same intrinsics, single GPU pass across all envs.
# data_types contains only "rgb" by default; collect_telemetry.py injects
# "instance_segmentation_fast" at startup before env creation.
# colorize_instance_segmentation=False → segmentation tensor is [B, H, W, 1] int32
# (integer IDs rather than RGBA colours, which simplifies cube-label lookup).
TILED_CAMERA_CFG = TiledCameraCfg(
    # Camera is parented under CameraXframe in the robot USD, which has the full
    # gripper-relative transform baked in.  The camera offset is identity here.
    prim_path="{ENV_REGEX_NS}/Robot/gripper/mountscrew/camera_mount/CameraXframe/gripper_camera",
    update_period=(1.0 / 30.0),
    height=128,
    width=128,
    data_types=["rgb"],
    colorize_instance_segmentation=False,
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=6.12,
        focus_distance=400.0,
        horizontal_aperture=6.3,
        clipping_range=(0.01, 10.0),
    ),
    offset=TiledCameraCfg.OffsetCfg(
        pos=(
            0.0,
            0.0,
            0.0,
        ),  # identity — mount Xform carries the gripper-relative transform
        rot=(1.0, 0.0, 0.0, 0.0),  # identity (w, x, y, z)
        convention="opengl",
    ),
)

# Overhead camera looking down on the workspace
OVERHEAD_CAMERA_CFG = CameraCfg(
    prim_path="{ENV_REGEX_NS}/overhead_camera",
    update_period=(1.0 / 30.0),
    height=768,
    width=1024,
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=6.12,
        focus_distance=600.0,
        horizontal_aperture=6.3,
        clipping_range=(0.01, 10.0),
    ),
    offset=CameraCfg.OffsetCfg(
        pos=(0.0, -0.5, 1.0),
        rot=(0.9763, 0.21644, 0.0, 0.0),
        convention="opengl",
    ),
)
