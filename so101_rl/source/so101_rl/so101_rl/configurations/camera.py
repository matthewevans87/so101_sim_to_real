import os
from typing import Any

from isaaclab.sensors import CameraCfg, TiledCameraCfg
import isaaclab.sim as sim_utils

from so101_rl.helpers.opencv_to_isaac_camera import load_intrinsics, opencv_to_isaac_pinhole

WORKSPACE_PATH = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")

# Prim-path template shared by all wrist-camera callers.
GRIPPER_CAMERA_PRIM_PATH = (
    "{ENV_REGEX_NS}/Robot/gripper/mountscrew/camera_mount/CameraXframe/gripper_camera"
)


def build_gripper_tiled_camera_cfg(
    intrinsics: dict[str, Any],
    height: int,
    width: int,
) -> TiledCameraCfg:
    """Build a ``TiledCameraCfg`` for the wrist camera from calibrated intrinsics.

    The PinholeCameraCfg FOV is derived from the calibrated ``fx``/``fy`` and
    sensor pixel pitch.  Principal-point offsets are zeroed — the
    ``OmniLensDistortionOpenCvPinholeAPI`` schema handles ``cx``/``cy`` when
    ``model == "opencv_pinhole"`` is applied post-spawn.

    Args:
        intrinsics: dict from :func:`load_intrinsics` (``camera_intrinsics.yaml``).
        height: Render height in pixels.
        width: Render width in pixels.
    """
    cam = opencv_to_isaac_pinhole(intrinsics)
    spawn_kwargs = cam["pinhole_cfg"].copy()
    spawn_kwargs["horizontal_aperture_offset"] = 0.0
    spawn_kwargs["vertical_aperture_offset"] = 0.0

    return TiledCameraCfg(
        prim_path=GRIPPER_CAMERA_PRIM_PATH,
        update_period=(1.0 / 30.0),
        height=height,
        width=width,
        data_types=["rgb"],
        colorize_instance_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(
            focus_distance=400.0,
            clipping_range=(0.01, 10.0),
            **spawn_kwargs,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


# Camera extrinsic mount transform — tuned with tune_camera_pose.py on 2026-05-09.
# Real robot joint positions were mirrored live into the sim arm and the
# CameraXframe prim was dragged until the overlay converged.
# CAMERA_TRANSLATE_VEC = (0.00035243581412122693, 0.04831022672385376, 0.0264999898285746)
# CAMERA_ROTATION_QUAT_WXYZ = (
#     0.9803372249541024,
#     -0.19707095311255154,
#     -0.009634924733446605,
#     -0.0030220909948313786,
# )

# These extrinsics were derived from the align_camera.py optimization tool on 2026-06-12:
CAMERA_TRANSLATE_VEC = (-0.0011563119432548294, 0.04953192393914805, 0.014280408597222396)
CAMERA_ROTATION_QUAT_WXYZ = (0.9725704591389929, -0.23243657486258057, -0.0057065324070226, -0.006883034520337212)

# Camera mount (SO-ARM101_camera_wrist_mount) USD asset and gripper-relative transform.
CAMERA_MOUNT_USD_PATH = os.path.join(WORKSPACE_PATH, "assets/robots/camera_mount.usd")
CAMERA_MOUNT_RELATIVE_POS = CAMERA_TRANSLATE_VEC
CAMERA_MOUNT_RELATIVE_QUAT_WXYZ = CAMERA_ROTATION_QUAT_WXYZ


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

# Kept for backwards-compat imports; always empty (distortion is applied post-spawn
# by callers that have their own intrinsics source: so101_lift_cube_env.py reads
# from SO101_ENV_PARAMS; align_camera.py reads from the deploy bundle).
CAMERA_POST_SPAWN_USD_ATTRS: dict[str, float] = {}
