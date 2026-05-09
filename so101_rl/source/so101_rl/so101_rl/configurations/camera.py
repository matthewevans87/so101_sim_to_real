import os
from pathlib import Path

from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

from so101_rl.helpers.opencv_to_isaac_camera import (
    load_intrinsics,
    opencv_to_isaac_pinhole,
)

WORKSPACE_PATH = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")

# ---------------------------------------------------------------------------
# Camera intrinsics: loaded from OpenCV calibration output.
# Generate this file by running:
#   python -m so101_real calibrate-camera --solve
# ---------------------------------------------------------------------------
_INTRINSICS_PATH = (
    Path(WORKSPACE_PATH) / "so101_real" / "configs" / "camera_intrinsics.yaml"
)
if not _INTRINSICS_PATH.exists():
    raise FileNotFoundError(
        f"Camera intrinsics file not found: {_INTRINSICS_PATH}\n"
        "Run the calibration pipeline first:\n"
        "  python -m so101_real calibrate-camera --solve"
    )
_INTRINSICS = load_intrinsics(_INTRINSICS_PATH)
_ISAAC_CAM = opencv_to_isaac_pinhole(_INTRINSICS)
_PINHOLE_CFG_KWARGS = _ISAAC_CAM["pinhole_cfg"].copy()
# Principal-point offsets are zeroed out: the physical lens is close enough to
# centred that the calibrated offsets (< 0.1 mm) add no perceptible benefit
# while introducing a render-time principal-point shift that complicates
# visual inspection with tune_camera_pose.py.
_PINHOLE_CFG_KWARGS["horizontal_aperture_offset"] = 0.0
_PINHOLE_CFG_KWARGS["vertical_aperture_offset"] = 0.0

# Camera extrinsic mount transform — tuned with tune_camera_pose.py on 2026-05-09.
# Real robot joint positions were mirrored live into the sim arm and the
# CameraXframe prim was dragged until the overlay converged.
CAMERA_TRANSLATE_VEC = (0.00035243581412122693, 0.04831022672385376, 0.0264999898285746)
CAMERA_ROTATION_QUAT_WXYZ = (
    0.9803372249541024,
    -0.19707095311255154,
    -0.009634924733446605,
    -0.0030220909948313786,
)

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
        focus_distance=400.0,
        clipping_range=(0.01, 10.0),
        **_PINHOLE_CFG_KWARGS,
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
        focus_distance=400.0,
        clipping_range=(0.01, 10.0),
        **_PINHOLE_CFG_KWARGS,
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

# No post-spawn USD attributes needed for the pinhole camera model.
CAMERA_POST_SPAWN_USD_ATTRS: dict[str, float] = {}
