# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math
import os

from so101_rl.configurations.cube import (
    CUBE_RESTING_HEIGHT,
)
from so101_utils.feature_extraction.feature_extraction import ResNet18SpatialSoftmaxFeatureExtractor
from so101_utils.image_processing import (
    CameraBrightnessPipelineStep,
    CameraContrastPipelineStep,
    CheapWebcamEffectPipelineStep,
    ClampPipelineStep,
    GaussianBlurPipelineStep,
    GaussianNoisePipelineStep,
    ImageNetNormalizationPipelineStep,
    ImagePipeline,
    JpegCompressionPipelineStep,
    MotionBlurPipelineStep,
    ResizePipelineStep,
)
from torch import tensor, zeros_like

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from torchvision.utils import save_image
from so101_rl.configurations.camera import (
    CAMERA_ROTATION_QUAT_WXYZ,
    CAMERA_TRANSLATE_VEC,
)
from so101_rl.helpers.variations import (
    # create_ground_materials,
    randomize_camera_pose,
    randomize_rigid_object_color,
    randomize_rigid_object_position,
    randomize_rigid_object_position_polar,
    randomize_rigid_object_size,
    randomize_ground_material,
    randomize_lighting,
    randomly_placed_lights,
)
from .so101_lift_cube_env_cfg import So101LiftCubeCfg
import torch
from collections.abc import Sequence
import torch.nn.functional as F

from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.materials.physics_materials_cfg import PhysicsMaterialCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply

from isaaclab.sensors import Camera, ContactSensor, FrameTransformer
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat, quat_unique

# Sequence of hook calls
# pre_physics_step
#   |-- _pre_physics_step(action)
#   |-- _apply_action()

# post_physics_step
#   |-- _get_dones()
#   |-- _get_rewards()
#   |-- _reset_idx()
#   |-- _get_observations()


class So101LiftCube(DirectRLEnv):
    cfg: So101LiftCubeCfg

    def __init__(
        self, cfg: So101LiftCubeCfg, render_mode: str | None = None, **kwargs
    ):
        # Store render mode BEFORE super().__init__() because _setup_scene() needs it
        self._render_mode = render_mode

        super().__init__(cfg, render_mode, **kwargs)

        # Get handles to data views
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Isaac Lab style: find joint indices by name/regex
        self._wrist_roll_idx = self.robot.find_joints("wrist_roll")[0]
        print("Wrist roll joint index:", self._wrist_roll_idx)
        self._all_joint_ids = torch.arange(self.robot.num_joints, device=self.device)

        # Indices the policy controls (all except wrist_roll)
        self._actuated_idxs = self._all_joint_ids[
            self._all_joint_ids != self._wrist_roll_idx
        ]
        print("Actuated joint indices:", self._actuated_idxs)

        # Find indices of DOFs and EE link
        self._dof_idx, _ = self.robot.find_joints(self.cfg.joints.active)
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.gripper.ee_link_name)
        self._wrist_roll_idx = 4  # hardcoded for now
        self._grip_zone_offset = torch.tensor(
            self.cfg.gripper.grip_zone_offset,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        # tip offset as tensor
        self._tip_offset = torch.tensor(
            self.cfg.gripper.tip_offset,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        gripper_forward_ee = torch.tensor(
            [0.0, 0.0, -1.0],  #  -Z is "into the jaw"
            device=self.device,
            dtype=torch.float32,
        )
        gripper_forward_ee = (
            gripper_forward_ee / gripper_forward_ee.norm()
        )  # just to be safe
        self.gripper_forward_ee = gripper_forward_ee.unsqueeze(0)  # (1, 3)

        # Camera forward direction in camera's local frame (OpenGL convention: -Z)
        camera_forward_local = torch.tensor(
            [0.0, 0.0, -1.0],  # -Z is forward for OpenGL cameras
            device=self.device,
            dtype=torch.float32,
        )
        camera_forward_local = camera_forward_local / camera_forward_local.norm()
        self.camera_forward_local = camera_forward_local.unsqueeze(0)  # (1, 3)

        # Camera offset from gripper (from CAMERA_CFG)
        self._camera_offset_pos = torch.tensor(
            CAMERA_TRANSLATE_VEC,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._camera_offset_quat = torch.tensor(
            CAMERA_ROTATION_QUAT_WXYZ,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 4)

        # Take env 0's limits, then select our DOFs
        limits = self.robot.data.joint_pos_limits[
            0, self._dof_idx, :
        ]  # shape: (num_actions, 2)
        print("selected limits shape:", limits.shape)
        print("selected limits:", limits)

        # Split to lower/upper and add batch dim
        joint_lower_1d = limits[:, 0]  # (num_actions,)
        joint_upper_1d = limits[:, 1]  # (num_actions,)

        self._joint_lower = joint_lower_1d.unsqueeze(0).to(
            self.device
        )  # (1, num_actions)
        self._joint_upper = joint_upper_1d.unsqueeze(0).to(
            self.device
        )  # (1, num_actions)


        self.actions = torch.zeros(
            (self.num_envs, self.cfg.action_space),  # type: ignore
            device=self.device,
            dtype=torch.float32,
        )  # type: ignore

        self.step_metrics: dict[str, torch.Tensor] = None  # type: ignore

        self.vision_feature_extractor = ResNet18SpatialSoftmaxFeatureExtractor(device=self.device)

        _pipeline_steps = []
        if self.cfg.domain_randomization.camera.feed.preshape_image.enabled:
            _pipeline_steps.append(ResizePipelineStep((224, 224)))
        _pipeline_steps.extend([
            GaussianBlurPipelineStep(),
            JpegCompressionPipelineStep(),
            MotionBlurPipelineStep(),
            CheapWebcamEffectPipelineStep(),
            GaussianNoisePipelineStep(),
            CameraBrightnessPipelineStep(),
            CameraContrastPipelineStep(),
            ImageNetNormalizationPipelineStep(),
            ClampPipelineStep(),
        ])
        self.image_pipeline = ImagePipeline(_pipeline_steps)

    # Called by super class to setup the scene
    def _setup_scene(self):

        # 1) Create assets
        self.table = RigidObject(self.cfg.table_cfg)
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)
        self.camera = Camera(self.cfg.camera_cfg)
        self.gripper_contact_sensor = ContactSensor(self.cfg.gripper_contact_sensor_cfg)
        self.table_contact_sensor = ContactSensor(self.cfg.table_contact_sensor_cfg)
        self.gripper_tf = FrameTransformer(self.cfg.gripper_transforms_cfg)
        self.grip_zone_tf = FrameTransformer(self.cfg.grip_zone_transformer_cfg)

        self._distractors: list[RigidObject] = []
        for i in range(self.cfg.distractors.count):
            distractor = RigidObject(self.cfg.distractor_cfgs[i])
            self._distractors.append(distractor)
            self.scene.rigid_objects[f"distractor_{i}"] = distractor

        # 2) Register assets in the scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["cube"] = self.cube
        self.scene.sensors["gripper_camera"] = self.camera
        self.scene.sensors["gripper_contact_sensor"] = self.gripper_contact_sensor
        self.scene.sensors["table_contact_sensor"] = self.table_contact_sensor
        self.scene.sensors["gripper_tf"] = self.gripper_tf
        self.scene.sensors["grip_zone_tf"] = self.grip_zone_tf

        # # 3) Ground plane
        # spawn_ground_plane(
        #     prim_path="/World/ground",
        #     cfg=GroundPlaneCfg(color=(0.5, 0.5, 0.5)),
        # )

        set_material("/World/envs/env_0/Table")

        # 4) Clone envs
        self.scene.clone_environments(copy_from_source=False)

        # CPU collision filtering (not sure we need this)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Visualization markers for gripper tip
        # self.tip_markers = define_tip_markers()

        # # Visualization markers for camera frame (x, y, z axes)
        # self.camera_frame_markers = define_camera_frame_markers()

        if self.cfg.debug.enable_gripper_arrow_markers:
            self.gripper_arrow_markers = define_gripper_arrow_markers()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before stepping the physics; store and scale actions."""
        if actions is None:
            return

        actions = torch.clamp(actions, -1.0, 1.0)
        t = 0.5 * (actions + 1.0)  # (num_envs, num_actions)
        self.actions = actions.clone()

        if self.cfg.behavior.binary_gripper_action.enabled:
            # make the target_pos of the gripper binary: fully open or set to the "grip" position
            t[:, self._ee_body_idx] = torch.where(
                t[:, self._ee_body_idx] > 0.5,
                torch.tensor(self.cfg.gripper.open_target, device=self.device),
                torch.tensor(self.cfg.gripper.closed_target, device=self.device),
            )

        target_pos = self._joint_lower + t * (self._joint_upper - self._joint_lower)

        target_pos[:, self._wrist_roll_idx] = math.radians(
            -90.0
        )  # keep wrist_roll fixed
        self._target_pos = target_pos

        # Always update markers
        # self._visualize_tip_markers()
        # self._visualize_camera_frame_markers()

        v_ee = self.step_metrics["v_grip_zone_to_cube_ee"]
        if self.cfg.debug.enable_gripper_arrow_markers:
            if v_ee is not None:
                self._visualize_gripper_arrow(v_ee)

    def _apply_action(self) -> None:
        """Apply scaled actions as joint position targets."""
        self.robot.set_joint_position_target(self._target_pos, joint_ids=self._dof_idx)

    def _get_observations(self) -> dict:
        """Return visual features (ResNet18) + joint positions."""

        # Raw camera RGB: (num_envs, H, W, 3), uint8
        camera_data = self.camera.data.output["rgb"]

        # Transform to (N, 3, H, W) float in [0, 1]
        images = camera_data.permute(0, 3, 1, 2).float() / 255.0

        # Apply domain randomization
        images = self.image_pipeline.process(images)

        if (
            self.cfg.debug.save_images
            and self.common_step_counter % self.cfg.debug.save_image_interval == 0
        ):
            save_image(
                images[0],  # (3, H, W)
                os.path.join(f"aug_{self.common_step_counter:06d}.png"),
            )

        # Extract visual features
        visual_features = self.vision_feature_extractor.extract(images)  # (N, 1024)

        # Proprioception
        q = self.joint_pos[:, self._dof_idx]  # (N, num_joints)
        dq = self.joint_vel[:, self._dof_idx]  # (N, num_joints) - Joint velocities

        # Observations
        actor_obs = torch.cat([visual_features, q], dim=-1)  # (N, 1024 + num_joints)

        # Compute on first observation call (when _get_observations is called from elf._env.reset())
        if self.step_metrics is None:
            print("Computing initial step metrics...")
            self._compute_step_metrics()

        critic_obs = torch.cat(
            [
                q,  # joint positions, 3
                dq,  # joint velocities, 3
                self.step_metrics["cube_pos_gz"],  # 3
                self.step_metrics["cube_rot6d_gz"],  # 6
                self.step_metrics["cube_height_w"].unsqueeze(1),  # 1
                self.step_metrics["gripper_cube_contact_force_magnitude"].unsqueeze(
                    1
                ),  # 1
                self.step_metrics["is_table_touched"].unsqueeze(1),  # 1
            ],
            dim=-1,
        )  # (N, X)

        observations = {"policy": actor_obs, "critic": critic_obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:

        if "log" not in self.extras:
            self.extras["log"] = {}

        rew_total = torch.zeros((self.num_envs,), device=self.device)

        # ********************
        # Primary Rewards
        # ********************

        # Minimize Distance between Cube and EE Grip Zone (STAGE 2: Approach)
        if self.cfg.rewards.distance.enabled:
            distance = self.step_metrics["grip_zone_cube_distance"]

            rew_distance = (
                # (~self.step_metrics["is_cube_gripped"]).float() *
                distance
                * self.cfg.rewards.distance.scale
            )
            self.extras["log"]["Episode_Reward/rew_distance"] = rew_distance.mean()
            rew_total += rew_distance

        # Grip Cube
        if self.cfg.rewards.grip_cube.enabled:
            rew_grip_cube = (
                self.step_metrics["is_cube_in_grip_position"]
                * (
                    self.step_metrics["gripper_cube_contact_force_magnitude"] > 0.0
                )  # has contact
                * self.cfg.rewards.grip_cube.scale
            )
            self.extras["log"]["Episode_Reward/rew_grip_cube"] = rew_grip_cube.mean()
            rew_total += rew_grip_cube

        # Lift Cube
        if self.cfg.rewards.lift_cube.enabled:
            rew_lift_cube = (
                # (self.step_metrics["gripper_cube_contact_force_magnitude"] > 0.0) *
                self.step_metrics["cube_lift_fraction"]
                * self.cfg.rewards.lift_cube.scale
            )

            self.extras["log"]["Episode_Reward/rew_lift_cube"] = rew_lift_cube.mean()
            rew_total += rew_lift_cube

        # ********************
        # Shaping Rewards
        # ********************

        # Find Cube
        if self.cfg.rewards.gripper_cube_alignment.enabled:
            rew_gripper_cube_alignment = (
                torch.maximum(
                    self.step_metrics["is_cube_gripped"],
                    self.step_metrics["gripper_cube_alignment"],
                )
                * self.cfg.rewards.gripper_cube_alignment.scale
            )
            self.extras["log"][
                "Episode_Reward/rew_gripper_cube_alignment"
            ] = rew_gripper_cube_alignment.mean()
            rew_total += rew_gripper_cube_alignment

        if self.cfg.rewards.gripper_look_at_cube.enabled:
            rew_gripper_look_at_cube = self._get_rew_gripper_look_at_cube(
                self.gripper_tf.data.source_pos_w,
                self.gripper_tf.data.target_pos_w[:, 0, :],
            )
            self.extras["log"][
                "Episode_Reward/rew_gripper_look_at_cube"
            ] = rew_gripper_look_at_cube.mean()
            rew_total += rew_gripper_look_at_cube

        if self.cfg.rewards.camera_cube_alignment.enabled:
            rew_camera_cube_alignment = (
                torch.maximum(
                    self.step_metrics["is_cube_gripped"],
                    self.step_metrics["camera_cube_alignment"],
                )
                * self.cfg.rewards.camera_cube_alignment.scale
            )
            self.extras["log"][
                "Episode_Reward/rew_camera_cube_alignment"
            ] = rew_camera_cube_alignment.mean()
            rew_total += rew_camera_cube_alignment

        # Encourage Gripping
        if self.cfg.rewards.close_gripper.enabled:
            gripper_pos = self.joint_pos[:, self._ee_body_idx]
            gripper_close_error = torch.abs(gripper_pos - self.cfg.rewards.close_gripper.close_target)
            fraction_to_target = 1.0 - (
                gripper_close_error / self.cfg.rewards.close_gripper.max_open
            ).squeeze(
                -1
            )  # 0.0 to 1.0

            rew_close_gripper = (
                self.step_metrics["is_cube_in_grip_position"]
                * fraction_to_target
                * self.cfg.rewards.close_gripper.scale
            )
            self.extras["log"][
                "Episode_Reward/rew_close_gripper"
            ] = rew_close_gripper.mean()
            rew_total += rew_close_gripper

        if self.cfg.rewards.gripper_force.enabled:
            gripper_force_target = self.cfg.rewards.gripper_force.force_target
            gripper_force = self.step_metrics["gripper_cube_contact_force_magnitude"]
            is_in_grip_position = self.step_metrics["is_cube_in_grip_position"]

            # Only apply reward when cube is in grip position
            force_error = torch.abs(gripper_force - gripper_force_target)
            # Positive reward: max at target, decays as error increases
            rew_gripper_force = (
                torch.exp(-force_error / (gripper_force_target + 1e-6))
                * self.cfg.rewards.gripper_force.scale
            )

            # Mask reward to only be active when in grip position
            rew_gripper_force = rew_gripper_force * is_in_grip_position.float()

            self.extras["log"][
                "Episode_Reward/rew_gripper_force"
            ] = rew_gripper_force.mean()
            rew_total += rew_gripper_force

        # Vantage Reward: STAGE 1 only (finding cube from far away)
        # Only active when distance > threshold to avoid conflict with approach stage
        if self.cfg.rewards.vantage.enabled:
            cube_gripper_dist = torch.linalg.norm(
                self.gripper_tf.data.source_pos_w
                - self.gripper_tf.data.target_pos_w[:, 0, :],
                dim=-1,
            )

            # Gate: only apply vantage reward when far from cube (Stage 1)
            is_far = cube_gripper_dist > self.cfg.rewards.vantage.far_distance_threshold

            rew_vantage_raw = self._get_rew_vantage(
                cube_gripper_dist,
                self.gripper_tf.data.source_pos_w,
                self.gripper_tf.data.target_pos_w[:, 0, :],
            )

            # Only apply reward when far, fade out as approaching
            rew_vantage = torch.where(
                is_far,
                rew_vantage_raw,
                torch.zeros_like(rew_vantage_raw),
            )

            self.extras["log"]["Episode_Reward/rew_vantage"] = rew_vantage.mean()
            rew_total += rew_vantage

        if self.cfg.rewards.keep_camera_upright.enabled:
            gripper_roll_target_pos_rad = math.radians(-90.0)
            gripper_roll_error = torch.abs(
                self.robot.data.joint_pos[:, self._wrist_roll_idx]
                - gripper_roll_target_pos_rad
            )
            rew_keep_camera_upright = (
                gripper_roll_error * self.cfg.rewards.keep_camera_upright.scale
            )
            self.extras["log"][
                "Episode_Reward/rew_keep_camera_upright"
            ] = rew_keep_camera_upright.mean()
            rew_total += rew_keep_camera_upright

        # ********************
        # Smoothing Rewards
        # ********************

        # Penalize large actions
        if self.cfg.rewards.action.enabled:
            if self.actions is None:
                rew_action = torch.zeros((self.num_envs,), device=self.device)
            else:
                rew_action = self.cfg.rewards.action.scale * torch.sum(
                    self.actions**2, dim=-1
                )
            self.extras["log"]["Episode_Reward/rew_action"] = rew_action.mean()
            rew_total += rew_action

        # Penalize end-effector velocity
        if self.cfg.rewards.ee_linear_speed.enabled:
            ee_lin_vel_w = self.robot.data.body_lin_vel_w[
                :, self._ee_body_idx[0], :
            ]  # (num_envs, 3)
            ee_linear_speed = torch.linalg.norm(ee_lin_vel_w, dim=-1)  # (num_envs,)
            v_safe = self.cfg.rewards.ee_linear_speed.safe_speed  # e.g. 0.2  (m/s)
            v_excess = torch.clamp(ee_linear_speed - v_safe, min=0.0)
            rew_ee_linear_speed = self.cfg.rewards.ee_linear_speed.scale * (
                v_excess + v_excess**2
            )
            self.extras["log"][
                "Episode_Reward/rew_ee_linear_speed"
            ] = rew_ee_linear_speed.mean()
            rew_total += rew_ee_linear_speed

        # Penalize joint velocity safety violations
        if self.cfg.rewards.joint_speed.enabled:
            joint_speed = torch.abs(
                self.joint_vel[:, self._dof_idx]
            )  # (num_envs, num_joints)
            rew_joint_speed = self.cfg.rewards.joint_speed.scale * torch.sum(
                joint_speed**2, dim=-1
            )
            self.extras["log"][
                "Episode_Reward/rew_joint_speed"
            ] = rew_joint_speed.mean()
            rew_total += rew_joint_speed

        # Penalize gripper height safety violations
        if self.cfg.rewards.ee_height_safety.enabled:
            ee_pos_w = self.robot.data.body_pos_w[:, self._ee_body_idx[0], :]
            ee_height = ee_pos_w[:, 2]
            unsafe = ee_height < self.cfg.safety.min_ee_height
            rew_ee_height_safety = torch.where(
                unsafe,
                torch.full_like(ee_height, self.cfg.rewards.ee_height_safety.scale),
                torch.zeros_like(ee_height),
            )
            self.extras["log"][
                "Episode_Reward/rew_ee_height_safety"
            ] = rew_ee_height_safety.mean()
            rew_total += rew_ee_height_safety

        # ********************
        # Terminal Rewards
        # ********************

        if self.cfg.rewards.success_touch_terminal.enabled:
            rew_success_touch_terminal = torch.where(
                self.step_metrics["is_success_touch_terminal"] >= 1.0,
                torch.full_like(
                    self.step_metrics["is_success_touch_terminal"],
                    self.cfg.rewards.success_touch_terminal.scale,
                ),
                torch.zeros_like(self.step_metrics["is_success_touch_terminal"]),
            )
            self.extras["log"][
                "Episode_Reward/rew_success_touch_terminal"
            ] = rew_success_touch_terminal.mean()
            rew_total += rew_success_touch_terminal

        if self.cfg.rewards.success_lift_fraction_terminal.enabled:
            rew_success_lift_fraction_terminal = torch.where(
                self.step_metrics["is_success_lift_fraction_terminal"] >= 1.0,
                torch.full_like(
                    self.step_metrics["is_success_lift_fraction_terminal"],
                    self.cfg.rewards.success_lift_fraction_terminal.scale,
                ),
                torch.zeros_like(
                    self.step_metrics["is_success_lift_fraction_terminal"]
                ),
            )
            self.extras["log"][
                "Episode_Reward/rew_success_lift_fraction_terminal"
            ] = rew_success_lift_fraction_terminal.float().mean()
            rew_total += rew_success_lift_fraction_terminal

        if self.cfg.rewards.success_point_at_cube_terminal.enabled:
            rew_success_point_at_cube_terminal = torch.where(
                self.step_metrics["is_success_point_at_cube_terminal"] >= 1.0,
                torch.full_like(
                    self.step_metrics["is_success_point_at_cube_terminal"],
                    self.cfg.rewards.success_point_at_cube_terminal.scale,
                ),
                torch.zeros_like(
                    self.step_metrics["is_success_point_at_cube_terminal"]
                ),
            )
            self.extras["log"][
                "Episode_Reward/rew_success_point_at_cube_terminal"
            ] = rew_success_point_at_cube_terminal.float().mean()
            rew_total += rew_success_point_at_cube_terminal

        if self.cfg.rewards.safety_touch_table_terminal.enabled:
            rew_safety_touch_table_terminal = torch.where(
                self.step_metrics["is_table_touched"],
                torch.tensor(
                    self.cfg.rewards.safety_touch_table_terminal.scale,
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
            )
            self.extras["log"][
                "Episode_Reward/rew_safety_touch_table_terminal"
            ] = rew_safety_touch_table_terminal.float().mean()
            rew_total += rew_safety_touch_table_terminal

        if self.cfg.rewards.safety_touch_table.enabled:
            rew_safety_touch_table = torch.where(
                self.step_metrics["is_table_touched"],
                torch.tensor(
                    self.cfg.rewards.safety_touch_table.scale,
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
            )
            self.extras["log"][
                "Episode_Reward/rew_safety_touch_table"
            ] = rew_safety_touch_table.float().mean()
            rew_total += rew_safety_touch_table

        return rew_total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        if "log" not in self.extras:
            self.extras["log"] = {}

        self._compute_step_metrics()
        for key in self.step_metrics:
            if (
                self.step_metrics[key].size(-1) == 1
                or self.step_metrics[key].dim() == 1
            ):
                self.extras["log"][f"Step_Metrics/{key}"] = (
                    self.step_metrics[key].float().mean()
                )

        # if self.common_step_counter % 100 == 0:
        #     print(self.step_metrics)

        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminal = zeros_like(self.episode_length_buf, dtype=torch.bool)
        if self.cfg.rewards.success_lift_fraction_terminal.enabled:
            terminal = torch.logical_or(
                terminal, self.step_metrics["is_success_lift_fraction_terminal"]
            )

        if self.cfg.rewards.success_point_at_cube_terminal.enabled:
            terminal = torch.logical_or(
                terminal, self.step_metrics["is_success_point_at_cube_terminal"]
            )

        if self.cfg.rewards.success_touch_terminal.enabled:
            is_success_touch_terminal = self.step_metrics["is_success_touch_terminal"]
            terminal = torch.logical_or(terminal, is_success_touch_terminal)

        if self.cfg.rewards.safety_touch_table_terminal.enabled:
            is_table_touched_terminal = self.step_metrics["is_table_touched"]
            terminal = torch.logical_or(terminal, is_table_touched_terminal)

        return terminal, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset robot state for the given env Ids."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)

        # Reset robot to default joint state and root from asset
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()

        # Start at an easier pose
        joint_pos[:, 2] = math.radians(-25.0)  # elbow_flex
        joint_pos[:, 3] = math.radians(65.0)  # wrist_flex
        joint_pos[:, self._wrist_roll_idx] = math.radians(
            -90.0
        )  # keep wrist_roll fixed

        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Optional: Add some random noise to starting joint positions
        joint_pos[:, self._dof_idx] += sample_uniform(
            -0.05, 0.05, joint_pos[:, self._dof_idx].shape, self.device
        )

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write changes back to simulation
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Domain Randomization
        if self.cfg.domain_randomization.cube.color_randomization_enabled:
            randomize_rigid_object_color(env_ids, object_name="Object")

        if self.cfg.domain_randomization.cube.size_randomization_enabled:
            randomize_rigid_object_size(
                env_ids,
                object_name="Object",
                size_range=self.cfg.domain_randomization.cube.size_range,
            )

        if self.cfg.domain_randomization.cube.position_randomization.enabled:
            randomize_rigid_object_position_polar(
                env_ids=env_ids,
                scene=self.scene,
                rigid_object=self.cube,
                object_name="Object",
                radius_range=self.cfg.domain_randomization.cube.position_randomization.radius_range,
                angle_range=self.cfg.domain_randomization.cube.position_randomization.angle_range,
                z_range=self.cfg.domain_randomization.cube.position_randomization.z_range,
                device=self.device,
            )

        if self.cfg.domain_randomization.camera.pose.enabled:
            randomize_camera_pose(
                env_ids,
                self.cfg.domain_randomization.camera.pose.position_noise_range,
                self.cfg.domain_randomization.camera.pose.rotation_noise_deg_range,
            )

        if 0 in env_ids and self.cfg.domain_randomization.lighting.enabled:
            randomize_lighting(
                self.cfg.domain_randomization.lighting.intensity_range, self.cfg.domain_randomization.lighting.color_variation
            )

        randomly_placed_lights(
            env_ids,
            self.cfg.domain_randomization.lighting.random_lights.height_range,
            self.cfg.domain_randomization.lighting.random_lights.intensity_range,
            self.cfg.domain_randomization.lighting.color_variation,
            self.cfg.domain_randomization.lighting.random_lights.specular_range,
        )

        if 0 in env_ids and self.cfg.domain_randomization.ground.enabled:
            randomize_ground_material()

        if self.cfg.distractors.randomization.enabled:
            # Randomize each distractor object
            for i, distractor in enumerate(self._distractors):
                distractor_name = f"distractor_{i}"

                # Reset distractor to default state first
                distractor_default_root_state = distractor.data.default_root_state[
                    env_ids
                ].clone()
                distractor_default_root_state[:, :3] += self.scene.env_origins[env_ids]
                distractor.write_root_pose_to_sim(
                    distractor_default_root_state[:, :7], env_ids
                )
                distractor.write_root_velocity_to_sim(
                    distractor_default_root_state[:, 7:], env_ids
                )

                # Randomize color
                randomize_rigid_object_color(env_ids, object_name=distractor_name)

                # Randomize size
                if self.cfg.distractors.randomization.size_randomization_enabled:
                    randomize_rigid_object_size(
                        env_ids,
                        object_name=distractor_name + "/geometry/mesh",
                        size_range=self.cfg.distractors.randomization.size_range,
                    )

                # Randomize position
                randomize_rigid_object_position(
                    env_ids=env_ids,
                    scene=self.scene,
                    rigid_object=distractor,
                    object_name=distractor_name,
                    x_range=self.cfg.distractors.position.x_range,
                    y_range=self.cfg.distractors.position.y_range,
                    z_range=self.cfg.distractors.position.z_range,
                    device=self.device,
                )

    def _compute_step_metrics(self) -> None:
        """Compute custom metrics at each step for logging purposes."""
        step_metrics = {}

        # ********************
        # gripper_cube_contact_force_magnitude
        # ********************

        gripper_cube_contact_force_magnitude = (
            torch.linalg.norm(
                self.gripper_contact_sensor.data.force_matrix_w[:, 0, 0, :], dim=-1, keepdim=True  # type: ignore
            )
            .squeeze(-1)
            .to(self.device)
        )
        assert_tensor(
            gripper_cube_contact_force_magnitude, (self.num_envs,), torch.float32
        )

        step_metrics["gripper_cube_contact_force_magnitude"] = (
            gripper_cube_contact_force_magnitude
        )

        # ********************
        # is_table_touched
        # ********************

        force_norms = torch.linalg.norm(
            self.table_contact_sensor.data.force_matrix_w, dim=-1
        )

        # Detect any non-zero filtered contact in that env:
        is_table_touched = (force_norms > 0.0).any(dim=-1).any(dim=-1)  # [num_envs]
        is_table_touched = is_table_touched.bool()
        assert_tensor(is_table_touched, (self.num_envs,), torch.bool)
        step_metrics["is_table_touched"] = is_table_touched

        # ********************
        # cube_pos_ee
        # ********************

        eps = 1e-6
        cube_pos_ee = self.gripper_tf.data.target_pos_source[:, 0, :]
        assert_tensor(cube_pos_ee, (self.num_envs, 3), torch.float32)
        step_metrics["cube_pos_ee"] = cube_pos_ee

        # ********************
        # gripper_cube_alignment
        # ********************

        # Get positions in world frame
        gripper_pos_w = self.robot.data.body_pos_w[
            :, self._ee_body_idx[0], :
        ]  # (num_envs, 3)
        cube_pos_w = self.cube.data.root_pos_w  # (num_envs, 3)

        # Direction from gripper to cube in world frame
        v_gripper_to_cube_w = cube_pos_w - gripper_pos_w  # (num_envs, 3)
        v_gripper_to_cube_w = v_gripper_to_cube_w / (
            v_gripper_to_cube_w.norm(dim=-1, keepdim=True) + eps
        )

        # Transform gripper forward direction to world frame
        gripper_quat_w = self.robot.data.body_quat_w[
            :, self._ee_body_idx[0], :
        ]  # (num_envs, 4)
        gripper_forward_ee_batch = self.gripper_forward_ee.expand(
            self.num_envs, -1
        )  # (num_envs, 3)
        gripper_forward_w = quat_apply(
            gripper_quat_w, gripper_forward_ee_batch
        )  # (num_envs, 3)

        # Alignment: dot product
        gripper_cube_alignment = (
            (v_gripper_to_cube_w * gripper_forward_w)
            .sum(dim=-1, keepdim=True)
            .squeeze(-1)
        )
        assert_tensor(gripper_cube_alignment, (self.num_envs,), torch.float32)
        step_metrics["gripper_cube_alignment"] = gripper_cube_alignment

        # ********************
        # camera_cube_alignment
        # ********************

        # Get gripper pose in world frame
        gripper_pos_w = self.robot.data.body_pos_w[
            :, self._ee_body_idx[0], :
        ]  # (num_envs, 3)
        gripper_quat_w = self.robot.data.body_quat_w[
            :, self._ee_body_idx[0], :
        ]  # (num_envs, 4)

        # Compute camera position: gripper_pos + rotate(camera_offset_pos by gripper_quat)
        camera_offset_pos_batch = self._camera_offset_pos.expand(self.num_envs, -1)
        camera_pos_w = gripper_pos_w + quat_apply(
            gripper_quat_w, camera_offset_pos_batch
        )  # (num_envs, 3)

        # Compute camera orientation: gripper_quat * camera_offset_quat
        camera_offset_quat_batch = self._camera_offset_quat.expand(self.num_envs, -1)
        camera_quat_w = math_utils.quat_mul(
            gripper_quat_w, camera_offset_quat_batch
        )  # (num_envs, 4)

        # Get cube position
        cube_pos_w = self.cube.data.root_pos_w  # (num_envs, 3)

        # Direction from camera to cube in world frame
        v_camera_to_cube_w = cube_pos_w - camera_pos_w  # (num_envs, 3)
        v_camera_to_cube_w = v_camera_to_cube_w / (
            v_camera_to_cube_w.norm(dim=-1, keepdim=True) + eps
        )

        # Transform camera forward direction (-Z in local frame) to world frame
        camera_forward_local_batch = self.camera_forward_local.expand(
            self.num_envs, -1
        )  # (num_envs, 3)
        camera_forward_w = quat_apply(
            camera_quat_w, camera_forward_local_batch
        )  # (num_envs, 3)

        # Alignment: dot product (1.0 = perfectly aligned, -1.0 = opposite direction)
        camera_cube_alignment = (
            (v_camera_to_cube_w * camera_forward_w)
            .sum(dim=-1, keepdim=True)
            .squeeze(-1)
        )
        assert_tensor(camera_cube_alignment, (self.num_envs,), torch.float32)
        step_metrics["camera_cube_alignment"] = camera_cube_alignment

        # ********************
        # v_grip_zone_to_cube_ee
        # ********************

        # Vector from *gripping point* to cube, in EE frame
        v_grip_zone_to_cube_ee = cube_pos_ee - self._grip_zone_offset  # (num_envs, 3)
        assert_tensor(v_grip_zone_to_cube_ee, (self.num_envs, 3), torch.float32)
        step_metrics["v_grip_zone_to_cube_ee"] = v_grip_zone_to_cube_ee

        # ********************
        # cube_pos_gz
        # ********************
        cube_pos_gz = self.grip_zone_tf.data.target_pos_source[:, 0, :]
        cube_pos_gz = self.grip_zone_tf.data.target_pos_source[:, 0, :]

        assert_tensor(cube_pos_gz, (self.num_envs, 3), torch.float32)
        step_metrics["cube_pos_gz"] = cube_pos_gz

        # ********************
        # cube_rot6d_gz
        # ********************
        # q_gz: (N, 4) wxyz  (cube orientation in grip-zone/source frame)
        q_gz = self.grip_zone_tf.data.target_quat_source[:, 0, :]  # (N,4)

        # Optional but recommended: remove the q vs -q sign flip (w >= 0)
        q_gz = quat_unique(q_gz)  # :contentReference[oaicite:1]{index=1}

        # Convert to rotation matrix
        R = matrix_from_quat(q_gz)  # (N, 3, 3) :contentReference[oaicite:2]{index=2}

        # 6D rotation rep = first two columns concatenated (column-major)
        cube_rot6d_gz = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # (N, 6)
        assert_tensor(cube_rot6d_gz, (self.num_envs, 6), torch.float32)
        step_metrics["cube_rot6d_gz"] = cube_rot6d_gz

        # ********************
        # grip_zone_cube_distance
        # ********************

        grip_zone_cube_distance = v_grip_zone_to_cube_ee.norm(
            dim=-1, keepdim=True
        ).squeeze(-1)
        assert_tensor(grip_zone_cube_distance, (self.num_envs,), torch.float32)
        step_metrics["grip_zone_cube_distance"] = grip_zone_cube_distance

        # ********************
        # cube_height_w
        # ********************

        cube_pos_w = self.cube.data.root_pos_w
        cube_height_w = (cube_pos_w[:, 2] - CUBE_RESTING_HEIGHT).clamp(min=0.0)
        assert_tensor(cube_height_w, (self.num_envs,), torch.float32)
        step_metrics["cube_height_w"] = cube_height_w

        # ********************
        # cube_lift_fraction
        # ********************
        cube_lift_fraction = (
            cube_height_w
        ) / self.cfg.rewards.success_lift_fraction_terminal.height_threshold
        assert_tensor(cube_lift_fraction, (self.num_envs,), torch.float32)
        step_metrics["cube_lift_fraction"] = cube_lift_fraction

        # ********************
        # is_success_lift_fraction_terminal
        # ********************

        is_success_lift_fraction_terminal = step_metrics["cube_lift_fraction"] >= 1.0
        assert_tensor(is_success_lift_fraction_terminal, (self.num_envs,), torch.bool)
        step_metrics["is_success_lift_fraction_terminal"] = (
            is_success_lift_fraction_terminal
        )

        # ********************
        # is_success_touch_terminal
        # ********************

        is_success_touch_terminal = (
            step_metrics["gripper_cube_contact_force_magnitude"]
            > self.cfg.rewards.success_touch_terminal.touch_force_threshold
        )
        assert_tensor(is_success_touch_terminal, (self.num_envs,), torch.bool)
        step_metrics["is_success_touch_terminal"] = is_success_touch_terminal

        # ********************
        # is_success_point_at_cube_terminal
        # ********************

        is_success_point_at_cube_terminal = (
            step_metrics["gripper_cube_alignment"] >= 1.0
        )
        assert_tensor(is_success_point_at_cube_terminal, (self.num_envs,), torch.bool)
        step_metrics["is_success_point_at_cube_terminal"] = (
            is_success_point_at_cube_terminal
        )

        # ********************
        # is_cube_in_grip_position
        # ********************

        is_cube_in_grip_position = (
            step_metrics["grip_zone_cube_distance"]
            < self.cfg.rewards.grip_cube.distance_threshold
        )

        assert_tensor(is_cube_in_grip_position, (self.num_envs,), torch.bool)
        step_metrics["is_cube_in_grip_position"] = is_cube_in_grip_position

        # ********************
        # is_cube_gripped
        # ********************

        is_cube_gripped = is_cube_in_grip_position & (
            step_metrics["gripper_cube_contact_force_magnitude"]  # type: ignore
            > self.cfg.rewards.success_touch_terminal.touch_force_threshold
        )
        is_cube_gripped = is_cube_gripped.bool()
        assert_tensor(is_cube_gripped, (self.num_envs,), torch.bool)
        step_metrics["is_cube_gripped"] = is_cube_gripped

        step_metrics["is_cube_gripped"] = is_cube_gripped

        self.step_metrics = step_metrics

    def _visualize_gripper_arrow(self, v_ee: torch.Tensor) -> None:
        """Draw an arrow at the gripper, pointing along v_ee (given in EE frame)."""
        device = self.device
        eps = 1e-6

        # 1) Gripper pose in world frame
        src_pos = self.gripper_tf.data.source_pos_w  # (N, 3) or (N, 1, 3)
        src_quat = self.gripper_tf.data.source_quat_w  # (N, 4) or (N, 1, 4)

        if src_pos.ndim == 3:
            gripper_pos_w = src_pos[:, 0, :]  # (N, 3)
            gripper_quat_w = src_quat[:, 0, :]  # (N, 4)
        else:
            gripper_pos_w = src_pos  # (N, 3)
            gripper_quat_w = src_quat  # (N, 4)

        num_envs = gripper_pos_w.shape[0]

        # 2) Use provided v_ee (EE-frame vector)
        # Ensure shape (N, 3)
        v_ee = v_ee.reshape(num_envs, 3).to(device=device, dtype=gripper_pos_w.dtype)

        # Unit direction in EE frame
        v_ee_unit = v_ee / (v_ee.norm(dim=-1, keepdim=True) + eps)  # (N, 3)

        # 3) Rotate direction into world frame using gripper orientation (EE -> world)
        v_world = math_utils.quat_apply(gripper_quat_w, v_ee_unit)  # (..., 3)
        v_world = v_world.reshape(num_envs, 3)
        v_world_norm = v_world / (v_world.norm(dim=-1, keepdim=True) + eps)  # (N, 3)

        # 4) Build quaternion that rotates +X to v_world_norm
        base_dir = torch.zeros_like(v_world_norm)  # (N, 3)
        base_dir[:, 0] = 1.0  # +X

        dot = (base_dir * v_world_norm).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)
        angle = torch.acos(dot)  # (N,)

        axis = torch.cross(base_dir, v_world_norm, dim=-1)  # (N, 3)
        axis_norm = axis.norm(dim=-1, keepdim=True)  # (N, 1)

        default_axis = torch.zeros_like(axis)
        default_axis[:, 2] = 1.0  # world Z
        axis = torch.where(axis_norm > eps, axis / (axis_norm + eps), default_axis)

        arrow_quat_w = math_utils.quat_from_angle_axis(angle, axis)  # (N, 4)

        # Optional: push arrow out a bit from gripper
        offset = torch.tensor(
            self.cfg.gripper.grip_zone_offset, dtype=torch.float32, device=device
        )
        arrow_pos_w = gripper_pos_w + offset * v_world_norm  # (N, 3)

        # 5) Visualize
        marker_indices = torch.zeros(num_envs, dtype=torch.long, device=device)  # (N,)

        self.gripper_arrow_markers.visualize(
            arrow_pos_w,  # (N, 3)
            arrow_quat_w,  # (N, 4)
            marker_indices=marker_indices,
        )

    def _get_rew_vantage(
        self,
        ee_to_cube_dist: torch.Tensor,
        ee_tip_pos: torch.Tensor,
        cube_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reward for maintaining a good vantage point to view the cube.
        Encourages optimal viewing distance and height above the cube.
        Uses Gaussian distributions to reward being near ideal values.

        IMPORTANT: Only applies when distance > min_distance_threshold to avoid
        interfering with primary approach/grip/lift rewards.
        When cube is gripped, returns full reward value.
        """
        d = ee_to_cube_dist  # (num_envs,)

        # Only apply vantage reward when far enough from cube
        # This prevents interference with primary rewards (distance, grip, lift)
        far_enough = d > self.cfg.rewards.vantage.min_distance_threshold

        # 1) Optimal distance reward (Gaussian centered at ideal viewing distance)
        ideal_dist = (
            self.cfg.rewards.vantage.ideal_distance
        )  # meters - adjust based on your camera FOV
        dist_sigma = (
            self.cfg.rewards.vantage.ideal_distance_sigma
        )  # how strict the distance requirement is
        dist_reward = torch.exp(-((d - ideal_dist) ** 2) / (2 * dist_sigma**2))

        # 2) Height reward: prefer being slightly above cube
        h_above_cube = ee_tip_pos[:, 2] - cube_pos[:, 2]
        ideal_height = self.cfg.rewards.vantage.ideal_height  # meters above cube (20cm)
        height_sigma = self.cfg.rewards.vantage.ideal_height_sigma
        # Penalize being below cube more heavily
        height_reward = torch.where(
            h_above_cube >= 0,
            torch.exp(-((h_above_cube - ideal_height) ** 2) / (2 * height_sigma**2)),
            torch.exp(-((h_above_cube) ** 2) / (2 * (height_sigma / 2) ** 2))
            * 0.3,  # stronger penalty below
        )

        # 3) Gripper roll reward: prefer gripper to be roughly vertical (roll = -90 deg)
        gripper_roll_target_pos_rad = math.radians(-90.0)
        gripper_roll_error = (
            torch.abs(
                self.robot.data.joint_pos[:, self._wrist_roll_idx]
                - gripper_roll_target_pos_rad
            )
            / gripper_roll_target_pos_rad
        )

        # 4) When cube is gripped, give full reward; otherwise combine factors and gate by distance threshold
        is_gripped = self.step_metrics["is_cube_gripped"]
        rew_vantage = torch.where(
            is_gripped,
            self.cfg.rewards.vantage.scale * torch.ones_like(d),
            torch.where(
                far_enough,
                self.cfg.rewards.vantage.scale
                * dist_reward
                * height_reward
                * gripper_roll_error,
                torch.zeros_like(d),
            ),
        )

        return rew_vantage

    def _get_rew_gripper_look_at_cube(
        self, ee_pos_w: torch.Tensor, cube_pos_w: torch.Tensor
    ) -> torch.Tensor:
        # ------------------------------------------------------------------
        # 2) Look-at reward: camera pointing at cube
        # ------------------------------------------------------------------
        # We approximate camera pose from gripper pose + known offset
        gripper_pos = ee_pos_w  # (num_envs, 3)
        gripper_quat = self.robot.data.body_quat_w[
            :, self._ee_body_idx[0], :
        ]  # (num_envs, 4)

        # Camera position in world frame
        camera_offset = (
            torch.tensor(
                CAMERA_TRANSLATE_VEC,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(self.num_envs, 3)
        )
        camera_pos_w = gripper_pos + quat_apply(gripper_quat, camera_offset)

        # Camera orientation in world frame (gripper orientation * camera local rotation)
        camera_rot_offset = (
            torch.tensor(
                CAMERA_ROTATION_QUAT_WXYZ,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .expand(self.num_envs, 4)
        )
        camera_quat_w = math_utils.quat_mul(gripper_quat, camera_rot_offset)

        # Camera forward axis in its local frame.
        # For OpenGL convention, cameras look along -Z in local coordinates.
        forward_local = (
            torch.tensor(
                [0.0, 0.0, -1.0],
                device=self.device,
                dtype=torch.float32,
            )
            .view(1, 3)
            .expand(self.num_envs, 3)
        )

        # World-space forward vector
        cam_forward_w = quat_apply(camera_quat_w, forward_local)  # (num_envs, 3)

        # Vector from camera to cube
        vec_to_cube = cube_pos_w - camera_pos_w  # (num_envs, 3)

        # Normalize
        cam_forward_norm = cam_forward_w / (
            torch.linalg.norm(cam_forward_w, dim=-1, keepdim=True) + 1e-6
        )
        vec_to_cube_norm = vec_to_cube / (
            torch.linalg.norm(vec_to_cube, dim=-1, keepdim=True) + 1e-6
        )

        # Cosine of angle between forward and cube direction: 1 = perfectly aligned
        cos_angle = torch.sum(cam_forward_norm * vec_to_cube_norm, dim=-1)
        # Only reward alignment when cos_angle > 0 (front hemisphere)
        lookat_factor = torch.clamp(cos_angle, min=0.0, max=1.0)

        # If cube is gripped, consider it perfectly looked at
        lookat_factor = torch.maximum(
            lookat_factor, self.step_metrics["is_cube_gripped"]
        )

        # Scale: up to for perfectly looking at cube
        rew_lookat = self.cfg.rewards.gripper_look_at_cube.scale * lookat_factor

        return rew_lookat

    # def _visualize_tip_markers(self):
    #     # Skip if markers weren't created (video recording mode)
    #     tip_pos = self._compute_ee_pos_w(self._grip_zone_offset)  # (num_envs, 3)

    #     # orientations: identity quats (w x y z) = (1, 0, 0, 0) per environment
    #     orientations = torch.zeros((self.num_envs, 4), device=self.device)
    #     orientations[:, 0] = 1.0  # w component

    #     # all markers use prototype index 0 ("tip")
    #     marker_indices = torch.zeros(
    #         (self.num_envs,), dtype=torch.long, device=self.device
    #     )

    #     self.tip_markers.visualize(
    #         translations=tip_pos,
    #         orientations=orientations,
    #         marker_indices=marker_indices,
    #     )

    def _compute_ee_pos_w(self, offset: torch.Tensor | None = None) -> torch.Tensor:
        # body_pos_w: (num_envs, num_bodies, 3)
        # body_quat_w: (num_envs, num_bodies, 4)
        ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx[0], :]
        ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx[0], :]

        # rotate offset from link frame to world frame
        if offset is not None:
            tip_offset = offset.expand((self.num_envs, 3))  # (num_envs, 3)
            offset_w = quat_apply(ee_quat, tip_offset)  # (num_envs, 3)
            tip_pos = ee_pos + offset_w  # (num_envs, 3)
        else:
            tip_pos = ee_pos

        return tip_pos


def define_gripper_arrow_markers() -> VisualizationMarkers:
    """Define a single arrow marker prototype, used for gripper->cube visualization."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/gripperMarkers",
        markers={
            "gripper_to_cube": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                # tweak scale to taste
                scale=(0.05, 0.05, 0.10),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.2, 0.0),  # orange-ish
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


import omni.usd  # type: ignore
import omni.kit.commands  # type: ignore
from pxr import UsdShade, UsdGeom, Gf  # type: ignore

VINYL_MDL_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Materials/Base/Plastics/Vinyl.mdl"
)

VINYL_MATERIAL_PATH = "/World/Looks/VinylMaterial"


def set_material(prim_path):
    # Get the current stage
    stage = omni.usd.get_context().get_stage()

    # Make sure we have a /World/Looks scope to hold materials
    if not stage.GetPrimAtPath("/World/Looks"):
        omni.kit.commands.execute(
            "CreatePrim",
            prim_path="/World/Looks",
            prim_type="Scope",
            select_new_prim=False,
        )

    # Create the MDL material prim if it doesn't exist yet
    if not stage.GetPrimAtPath(VINYL_MATERIAL_PATH):
        omni.kit.commands.execute(
            "CreateMdlMaterialPrim",
            mtl_url=VINYL_MDL_URL,
            mtl_name="Vinyl",  # display name; not super critical
            mtl_path=VINYL_MATERIAL_PATH,
            select_new_prim=False,
        )

    # Get the material as a UsdShade.Material
    material_prim = stage.GetPrimAtPath(VINYL_MATERIAL_PATH)
    material = UsdShade.Material(material_prim)

    # Bind the material to your ground plane
    prim = stage.GetPrimAtPath(prim_path)
    UsdShade.MaterialBindingAPI(prim).Bind(
        material, UsdShade.Tokens.strongerThanDescendants
    )


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


def assert_tensor(tensor: torch.Tensor, shape: tuple, dtype):
    """Utility to assert tensor shape and dtype."""
    assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
    assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
