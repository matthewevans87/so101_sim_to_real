# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math
import os

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
from torch import zeros_like

from so101_rl.helpers.visual_markers import define_gripper_arrow_markers
from so101_rl.helpers.utils import set_material

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
    randomize_world_light,
    randomize_env_lights,
)
from .so101_lift_cube_env_cfg import So101LiftCubeCfg
from .env_pipeline import StepContext, MetricPipeline, RewardPipeline, build_metric_pipeline, build_reward_pipeline
import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform, quat_apply

from isaaclab.sensors import Camera, ContactSensor, FrameTransformer
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

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


        # Find indices of DOFs and EE link
        self._dof_idx, _ = self.robot.find_joints(self.cfg.joints.active)
        self._all_joint_idx, _ = self.robot.find_joints(self.cfg.joints.all)
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

        self._step_ctx = StepContext(env=self)
        self.metric_pipeline: MetricPipeline = build_metric_pipeline()
        self.reward_pipeline: RewardPipeline = build_reward_pipeline(self.cfg)

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
        return self.reward_pipeline.compute(self._step_ctx)

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
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Apply starting position from config (joints.all order, degrees or null)
        for i, deg in enumerate(self.cfg.joints.starting_position):
            if deg is not None:
                joint_pos[:, self._all_joint_idx[i]] = math.radians(deg)

        # Add some random noise to starting joint positions
        if self.cfg.joints.starting_position_noise.enabled:
            noise_range = self.cfg.joints.starting_position_noise.range
            joint_pos[:, self._dof_idx] += sample_uniform(
                noise_range[0], noise_range[1], joint_pos[:, self._dof_idx].shape, self.device
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

        if 0 in env_ids and self.cfg.domain_randomization.world_lighting.enabled:
            randomize_world_light(
                self.cfg.domain_randomization.world_lighting.intensity_range,
                self.cfg.domain_randomization.world_lighting.color_variation,
            )

        if self.cfg.domain_randomization.env_lighting.enabled:
            randomize_env_lights(
                env_ids,
                self.cfg.domain_randomization.env_lighting.height_range,
                self.cfg.domain_randomization.env_lighting.intensity_range,
                self.cfg.domain_randomization.env_lighting.color_variation,
                self.cfg.domain_randomization.env_lighting.specular_range,
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
        self.metric_pipeline.compute(self._step_ctx)
        self.step_metrics = self._step_ctx.metrics

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

