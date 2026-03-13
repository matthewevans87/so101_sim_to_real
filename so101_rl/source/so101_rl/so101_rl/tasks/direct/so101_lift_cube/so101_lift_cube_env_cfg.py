# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from so101_rl.configurations.so101 import (
    SO101_CFG,
    GRIPPER_CONTACT_SENSOR_CFG,
)
from so101_rl.configurations.cube import DEX_CUBE_CFG
from so101_rl.configurations.camera import CAMERA_CFG, OVERHEAD_CAMERA_CFG
from so101_rl.configurations.table import TABLE_CFG, TABLE_CONTACT_SENSOR_CFG

from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from so101_rl.configurations.so101_env_params import So101EnvParams
from so101_rl.env_pipeline import KEY_OBS_DIMS

_Y: So101EnvParams = So101EnvParams.load(os.environ["SO101_ENV_CONFIG"])
_ENABLE_OVERHEAD_CAMERA = os.environ.get("SO101_ENABLE_OVERHEAD_CAMERA", "0").lower() in (
    "1",
    "true",
    "yes",
)


@configclass
class So101LiftCubeCfg(DirectRLEnvCfg):
    # ── Isaac Lab required direct members ──────────────────────────────────
    decimation = _Y.decimation
    episode_length_s = _Y.episode_length_s

    sim: SimulationCfg = SimulationCfg(dt=_Y.sim.dt, render_interval=_Y.decimation)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=_Y.scene.num_envs,
        env_spacing=_Y.scene.env_spacing,
        replicate_physics=_Y.scene.replicate_physics,
    )

    # ── Observation / action space (architecture-dependent, hardcoded) ──────
    NUM_ACTIVE_JOINTS = len(_Y.joints.active)
    action_space = NUM_ACTIVE_JOINTS

    SPATIAL_SOFTMAX_FEATURES = 1024
    observation_space = SPATIAL_SOFTMAX_FEATURES + NUM_ACTIVE_JOINTS
    state_space = (
        2 * NUM_ACTIVE_JOINTS
        + sum(KEY_OBS_DIMS[k] for k in _Y.observations.critic_obs_metrics)
    )

    # ── Asset configs ───────────────────────────────────────────────────────
    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    table_cfg: RigidObjectCfg = TABLE_CFG.replace(prim_path="/World/envs/env_.*/Table")  # type: ignore

    cube_cfg: RigidObjectCfg = DEX_CUBE_CFG.replace(prim_path="/World/envs/env_.*/Object")  # type: ignore

    camera_cfg: CameraCfg = CAMERA_CFG.replace(  # type: ignore
        prim_path="/World/envs/env_.*/Robot/gripper/gripper_camera",
        height=_Y.sensors.camera.height,
        width=_Y.sensors.camera.width,
    )

    overhead_camera_cfg: CameraCfg | None = (
        OVERHEAD_CAMERA_CFG.replace(  # type: ignore
            prim_path="/World/envs/env_.*/overhead_camera",
        )
        if _ENABLE_OVERHEAD_CAMERA
        else None
    )

    # ── Sensor configs ──────────────────────────────────────────────────────
    gripper_contact_sensor_cfg = GRIPPER_CONTACT_SENSOR_CFG.replace(  # type: ignore
        prim_path="/World/envs/env_.*/Robot/gripper",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
        debug_vis=_Y.sensors.gripper_contact.debug_vis,
    )

    table_contact_sensor_cfg = TABLE_CONTACT_SENSOR_CFG.replace(  # type: ignore
        prim_path="/World/envs/env_.*/Table",
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/upper_arm",
            "/World/envs/env_.*/Robot/lower_arm",
            "/World/envs/env_.*/Robot/wrist",
            "/World/envs/env_.*/Robot/gripper",
            "/World/envs/env_.*/Robot/moving_jaw_so101_v1",
        ],
        track_pose=_Y.sensors.table_contact.track_pose,
        debug_vis=_Y.sensors.table_contact.debug_vis,
    )

    gripper_transforms_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/gripper",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Object", name="cube")
        ],
        debug_vis=_Y.sensors.gripper_transform.debug_vis,
    )

    grip_zone_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/gripper",
        source_frame_offset=OffsetCfg(
            pos=_Y.gripper.grip_zone_offset,
            rot=_Y.gripper.grip_zone_rot,
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="/World/envs/env_.*/Object", name="cube")
        ],
        debug_vis=_Y.sensors.grip_zone_transform.debug_vis,
    )

    # ── Distractor rigid object configs ─────────────────────────────────────
    _spawn_pool = [
        sim_utils.CuboidCfg(
            size=_Y.distractors.geometry.cube_size,
            visual_material=sim_utils.PreviewSurfaceCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=None,
        ),
        sim_utils.ConeCfg(
            radius=_Y.distractors.geometry.cone_radius,
            height=_Y.distractors.geometry.cone_height,
            visual_material=sim_utils.PreviewSurfaceCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=None,
        ),
    ]
    distractor_cfgs: list[RigidObjectCfg] = []
    for _i in range(_Y.distractors.count):
        distractor_cfgs.append(
            RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/distractor_{_i}",
                spawn=_spawn_pool[_i % len(_spawn_pool)],
            )
        )

    # ── Typed config groups (self.cfg.<group>.<field>) ───────────────────────
    joints            = _Y.joints
    control           = _Y.control
    safety            = _Y.safety
    gripper           = _Y.gripper
    distractors       = _Y.distractors
    debug             = _Y.debug
    behavior          = _Y.behavior
    rewards           = _Y.rewards
    domain_randomization = _Y.domain_randomization
    observations      = _Y.observations

