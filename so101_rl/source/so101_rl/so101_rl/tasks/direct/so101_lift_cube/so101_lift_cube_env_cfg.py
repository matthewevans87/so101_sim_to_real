# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
from so101_rl.configurations.so101 import (
    SO101_CFG,
    SO101_NUM_JOINTS,
    GRIPPER_CONTACT_SENSOR_CFG,
)
from so101_rl.configurations.cube import DEX_CUBE_CFG
from so101_rl.configurations.camera import CAMERA_CFG
from so101_rl.configurations.table import TABLE_CFG, TABLE_CONTACT_SENSOR_CFG

from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors import FrameTransformerCfg
from sympy import false
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import materials as mat_utils
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg


@configclass
class So101LiftCubeCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0

    JOINTS = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    ACTIVE_JOINTS = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    wrist_roll_name = "wrist_roll"

    # action/obs dimensions
    NUM_ACTIVE_JOINTS = len(ACTIVE_JOINTS)  # all joints except gripper
    action_space = NUM_ACTIVE_JOINTS

    SPATIAL_SOFTMAX_FEATURES = 1024
    observation_space = SPATIAL_SOFTMAX_FEATURES + NUM_ACTIVE_JOINTS  # exclude gripper
    state_space = 18

    # control
    action_scale = 1.0  # rad/s per unit action in [-1, 1]

    # Safety limits
    safety_max_joint_velocity = 2.0  # rad/s
    safety_min_ee_height = 0.01  # m

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=2.0, replicate_physics=True
    )

    # Robot(s)
    robot_cfg: ArticulationCfg = SO101_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    table_cfg: RigidObjectCfg = TABLE_CFG.replace(prim_path="/World/envs/env_.*/Table")  # type: ignore

    # EE link (frame origin) and local offset to tip
    ee_link_name = "gripper"
    gripper_tip_offset = (0.015, 0.0, -0.1)  # in gripper frame; refine later
    grip_zone_offset = (0.01, 0.0, -0.09)  # Grasp center point
    gripper_open_target = 0.8  # 80% open when approaching
    gripper_closed_target = 0.15  # P1 FIX: Relaxed from 0.08 to match real robot

    # Cube
    cube_cfg: RigidObjectCfg = DEX_CUBE_CFG.replace(prim_path="/World/envs/env_.*/Object")  # type: ignore

    # Camera mounted on gripper
    camera_cfg: CameraCfg = CAMERA_CFG.replace(  # type: ignore
        prim_path="/World/envs/env_.*/Robot/gripper/gripper_camera",
        height=480 // 5 * 2,  # 192
        width=640 // 5 * 2,  # 256,
    )

    # Detect contacts between gripper and cube
    gripper_contact_sensor_cfg = GRIPPER_CONTACT_SENSOR_CFG.replace(  # type: ignore
        prim_path="/World/envs/env_.*/Robot/gripper",
        filter_prim_paths_expr=["/World/envs/env_.*/Object"],
        debug_vis=False,
    )

    table_contact_sensor_cfg = TABLE_CONTACT_SENSOR_CFG.replace(  # type: ignore
        prim_path="/World/envs/env_.*/Table",  # attach sensor to the table
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/upper_arm",
            "/World/envs/env_.*/Robot/lower_arm",
            "/World/envs/env_.*/Robot/wrist",
            "/World/envs/env_.*/Robot/gripper",
            "/World/envs/env_.*/Robot/moving_jaw_so101_v1",
        ],
        track_pose=False,
        debug_vis=True,
    )

    gripper_transforms_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/gripper",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object", name="cube"
            )
        ],
        debug_vis=False,
    )

    grip_zone_transformer_cfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/gripper",
        source_frame_offset=OffsetCfg(
            pos=grip_zone_offset,
            rot=(1.0, 0.0, 0.0, 0.0),  # identity quat (w,x,y,z)
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object", name="cube"
            )
        ],
        debug_vis=False,
    )

    NUM_DISTRACTORS = 10

    cube_spawn_cfg = sim_utils.CuboidCfg(
        size=(0.03, 0.03, 0.03),
        visual_material=sim_utils.PreviewSurfaceCfg(),  # color randomized later
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=None,  # purely visual if you want
    )
    sphere_spawn_cfg = sim_utils.SphereCfg(
        radius=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=None,
    )
    cone_spawn_cfg = sim_utils.ConeCfg(
        radius=0.02,
        height=0.04,
        visual_material=sim_utils.PreviewSurfaceCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=None,
    )

    spawn_cfgs = [
        cube_spawn_cfg,
        # sphere_spawn_cfg,
        cone_spawn_cfg,
    ]

    distractor_cfgs = []

    for i in range(NUM_DISTRACTORS):
        shape_cfg = spawn_cfgs[i % len(spawn_cfgs)]

        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/distractor_{i}", spawn=shape_cfg
        )
        distractor_cfgs.append(obj_cfg)

    enable_distractor_randomization: bool = True
    enable_distractor_size_randomization: bool = True
    distractor_active_prob: float = 0.8

    distractor_x_range = (0.5, 0.8)
    distractor_y_range = (-1.0, 1.0)
    distractor_z_range = (0.01, 0.25)
    distractor_size_range: tuple[float, float] = (1.0, 5.0)  # scale factors

    # debug flags
    save_images: bool = False
    save_image_interval: int = 10  # steps
    enable_gripper_arrow_markers: bool = False

    # rewards
    rew_distance_enabled: bool = True
    rew_distance_scale: float = -100.0  # Strong penalty to drive approach behavior

    rew_grip_cube_enabled: bool = True
    rew_grip_cube_scale: float = 100.0
    rew_grip_cube_distance_threshold: float = 0.02

    rew_lift_cube_enabled: bool = True
    rew_lift_cube_scale: float = 1000.0

    rew_gripper_cube_alignment_enabled: bool = False
    rew_gripper_cube_alignment_scale: float = 1.0

    rew_camera_cube_alignment_enabled: bool = True
    rew_camera_cube_alignment_scale: float = 1.0

    rew_close_gripper_enabled: bool = True
    rew_close_gripper_scale: float = 10.0
    gripper_close_target: float = math.radians(10)
    gripper_max_open: float = math.radians(100)

    rew_gripper_force_enabled: bool = True
    rew_gripper_force_scale: float = 10.0
    gripper_force_target: float = 1.0  # Newtons

    rew_gripper_look_at_cube_enabled: bool = False
    rew_gripper_look_at_cube_scale: float = 1.0

    rew_action_enabled: bool = True
    rew_action_scale: float = -0.001

    rew_ee_linear_speed_enabled: bool = True
    rew_rew_ee_linear_speed_scale: float = -0.01
    ee_safe_speed = 0.1  # m/s

    rew_joint_speed_enabled: bool = False
    rew_joint_speed_scale: float = -0.01

    rew_ee_height_safety_enabled: bool = False
    rew_ee_height_safety_scale: float = -100.0

    rew_safety_touch_table_enabled: bool = True
    rew_safety_touch_table_scale: float = -1.0

    rew_success_lift_fraction_terminal_enabled: bool = True
    rew_success_lift_fraction_terminal_scale: float = 2000.0
    cube_height_success_terminate_threshold: float = 0.10

    rew_success_touch_terminal_enabled: bool = False
    rew_success_touch_terminal_scale: float = 500.0
    touch_force_threshold: float = 0.001

    rew_success_point_at_cube_terminal_enabled: bool = False
    rew_success_point_at_cube_terminal_scale: float = 2000.0

    rew_safety_touch_table_terminal_enabled: bool = False
    rew_safety_touch_table_terminal_scale: float = -2000.0

    rew_vantage_enabled: bool = False
    rew_vantage_ideal_distance: float = 0.15  # m
    rew_vantage_ideal_distance_sigma: float = 0.10
    rew_vantage_ideal_height: float = 0.10  # m
    rew_vantage_ideal_height_sigma: float = 0.15
    rew_vantage_min_distance_threshold: float = 0.08
    rew_vantage_far_distance_threshold: float = (
        0.20  # Only active when > 20cm (Stage 1)
    )
    rew_scale_vantage: float = 1.0  # Increased for stage 1 (finding cube)

    rew_keep_camera_upright_enabled: bool = False
    rew_keep_camera_upright_scale: float = -1.0

    # ====================
    # Behavior Toggles
    # ====================
    enable_binary_gripper_action: bool = False

    # ====================
    # Domain Randomization
    # ====================

    # Camera feed augmentation
    enable_preshape_camera_image: bool = False
    # enable_gaussian_blur_rgb: bool = True
    # enable_cheap_webcam_effect: bool = True
    # enable_camera_brightness: bool = True
    # enable_camera_noise: bool = True
    enable_camera_contrast: bool = False
    camera_gaussian_noise_std: tuple[float, float] = (0.01, 0.02)  # 1-3% noise
    camera_brightness_range: tuple[float, float] = (0.85, 1.15)  # ±15%
    camera_contrast_range: tuple[float, float] = (0.8, 1.2)  # ±20%

    # Advanced camera augmentation
    enable_motion_blur: bool = False
    motion_blur_kernel_size: int = 5  # 3, 5, or 7
    motion_blur_strength_range: tuple[float, float] = (0.1, 0.2)

    enable_jpeg_compression: bool = False
    jpeg_quality_range: tuple[int, int] = (60, 70)

    # Camera pose randomization (mounting errors)
    enable_camera_pose_randomization: bool = True
    camera_pos_noise_range: tuple[float, float] = (-0.001, 0.001)  # ±1mm
    camera_rot_noise_deg_range: tuple[float, float] = (-0.5, 0.5)  # ±0.5°

    # Lighting randomization
    enable_lighting_randomization: bool = True
    light_intensity_range: tuple[float, float] = (500.0, 1500.0)
    light_color_variation: float = 0.05  # ±5% per RGB channel
    # Randomly Placed Lights
    rand_light_height_range: tuple[float, float] = (0.5, 2.0)  # m
    rand_light_intensity_range: tuple[float, float] = (10_000.0, 30_000.0)
    rand_light_specular_range: tuple[float, float] = (1.0, 5.0)

    # Cube appearance randomization
    enable_cube_color_randomization: bool = False

    # Cube size randomization
    enable_cube_size_randomization: bool = True
    cube_size_range: tuple[float, float] = (0.975, 1.025)  # ±2.5% from base size

    enable_randomize_cube_position: bool = True
    cube_radius_range = (0.35, 0.40)  # Distance from robot base (r_1, r_2)
    cube_angle_range = (
        -15.0,
        15.0,
    )  # Angle in degrees, 0° = directly in front (theta_1, theta_2)
    cube_z_range = (0.03, 0.04)

    # Ground surface randomization
    enable_ground_randomization: bool = False
