# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math
import os

from so101.utils.feature_extraction.feature_extraction import (
    CnnSpatialSoftmaxFeatureExtractor,
    ResNet18SpatialSoftmaxFeatureExtractor,
)
from so101.model.model import MultiTaskCnn, multitask_cnn_from_checkpoint
from so101.utils.image_processing import (
    CameraBrightnessPipelineStep,
    CameraContrastPipelineStep,
    CheapWebcamEffectPipelineStep,
    ClampPipelineStep,
    GaussianBlurPipelineStep,
    GaussianNoisePipelineStep,
    ImageNetNormalizationPipelineStep,
    ImagePipeline,
    ImagePipelineStep,
    JpegCompressionPipelineStep,
    MotionBlurPipelineStep,
    ResizePipelineStep,
    Uint8ToFloatCHWPipelineStep,
)
from so101_rl.helpers.visual_markers import (
    define_grip_zone_markers,
    define_gripper_arrow_markers,
    define_camera_frame_markers,
    visualize_grip_zone_markers,
    visualize_gripper_arrow,
    visualize_camera_frame_markers,
)
from so101_rl.helpers.utils import set_material

from torchvision.utils import save_image
from so101_rl.configurations.camera import (
    CAMERA_ROTATION_QUAT_WXYZ,
    CAMERA_TRANSLATE_VEC,
)
from .so101_lift_cube_env_cfg import So101LiftCubeCfg
from so101_rl.viz.vision_debug import VisionDebugLogger
from so101_rl.env_pipeline import (
    DRContext,
    DRPipeline,
    EnvMetricPipeline,
    StepContext,
    MetricPipeline,
    RewardPipeline,
    build_dr_pipeline,
    build_env_metric_pipeline,
    build_metric_pipeline,
    build_reward_pipeline,
    KEY_OBS_DIMS,
)
import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import sample_uniform, quat_apply

from isaaclab.sensors import Camera, TiledCamera, ContactSensor, FrameTransformer
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

    def __init__(self, cfg: So101LiftCubeCfg, render_mode: str | None = None, **kwargs):
        # Store render mode BEFORE super().__init__() because _setup_scene() needs it
        self._render_mode = render_mode

        super().__init__(cfg, render_mode, **kwargs)

        # Get handles to data views
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Find indices of DOFs and EE link
        self._dof_idx, _ = self.robot.find_joints(self.cfg.joints.active)
        self._all_joint_idx, _ = self.robot.find_joints(self.cfg.joints.all)
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.gripper.ee_link_name)
        self._gripper_joint_idx, _ = self.robot.find_joints(
            [self.cfg.gripper.ee_link_name]
        )

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
        self.prev_actions = torch.zeros_like(self.actions)

        self.step_metrics: dict[str, torch.Tensor] = None  # type: ignore
        self.env_metrics: dict[str, torch.Tensor] = {}

        self._step_ctx = StepContext(env=self)
        self.reward_pipeline: RewardPipeline = build_reward_pipeline(self.cfg)
        self.env_metric_pipeline: EnvMetricPipeline = build_env_metric_pipeline(
            self.cfg
        )
        self.metric_pipeline: MetricPipeline = build_metric_pipeline(
            self.reward_pipeline,
            extra_keys=frozenset(
                {
                    # consumed by _get_observations (critic features)
                    *self.cfg.observations.critic_obs_metrics,
                }
            ),
            env_metric_pipeline=self.env_metric_pipeline,
        )
        self.dr_pipeline: DRPipeline = build_dr_pipeline(self.cfg)
        _dr_feed = self.cfg.domain_randomization.camera.feed
        _image_pipeline_steps: list[ImagePipelineStep] = [Uint8ToFloatCHWPipelineStep()]
        if _dr_feed.gaussian_blur.enabled:
            _image_pipeline_steps.append(
                GaussianBlurPipelineStep(
                    kernel_size=_dr_feed.gaussian_blur.kernel_size,
                    sigma=_dr_feed.gaussian_blur.sigma,
                )
            )
        if _dr_feed.jpeg_compression.enabled:
            _image_pipeline_steps.append(
                JpegCompressionPipelineStep(
                    quality_range=(
                        _dr_feed.jpeg_compression.quality_range[0],
                        _dr_feed.jpeg_compression.quality_range[1],
                    ),
                    device=self.device,
                )
            )
        if _dr_feed.motion_blur.enabled:
            _image_pipeline_steps.append(
                MotionBlurPipelineStep(
                    motion_blur_strength_range=(
                        _dr_feed.motion_blur.strength_range[0],
                        _dr_feed.motion_blur.strength_range[1],
                    ),
                    motion_blur_kernel_size=_dr_feed.motion_blur.kernel_size,
                    device=self.device,
                )
            )
        if _dr_feed.cheap_webcam_effect.enabled:
            _image_pipeline_steps.append(
                CheapWebcamEffectPipelineStep(device=self.device)
            )
        if _dr_feed.gaussian_noise.enabled:
            _image_pipeline_steps.append(
                GaussianNoisePipelineStep(
                    noise_std_range=(
                        _dr_feed.gaussian_noise.std_range[0],
                        _dr_feed.gaussian_noise.std_range[1],
                    ),
                    device=self.device,
                )
            )
        if _dr_feed.brightness.enabled:
            _image_pipeline_steps.append(
                CameraBrightnessPipelineStep(
                    brightness_range=(
                        _dr_feed.brightness.range[0],
                        _dr_feed.brightness.range[1],
                    ),
                    device=self.device,
                )
            )
        if _dr_feed.contrast.enabled:
            _image_pipeline_steps.append(
                CameraContrastPipelineStep(
                    contrast_range=(
                        _dr_feed.contrast.range[0],
                        _dr_feed.contrast.range[1],
                    ),
                    device=self.device,
                )
            )
        _vision_type = self.cfg.vision_encoder.type
        if _vision_type == "frozen_resnet18":
            self.vision_feature_extractor = ResNet18SpatialSoftmaxFeatureExtractor(
                device=self.device
            )
            if self.cfg.domain_randomization.camera.feed.preshape_image.enabled:
                # 224x224 matches ImageNet pretraining resolution for best feature quality
                _image_pipeline_steps.insert(1, ResizePipelineStep((224, 224)))
            _image_pipeline_steps.append(ImageNetNormalizationPipelineStep())
            _image_pipeline_steps.append(ClampPipelineStep())
        elif _vision_type == "frozen_cnn":
            ve = self.cfg.vision_encoder
            if ve.backbone is None:
                raise ValueError(
                    "vision_encoder.backbone must be set when "
                    "vision_encoder.type == 'frozen_cnn'."
                )
            _image_pipeline_steps.insert(
                1,
                ResizePipelineStep((ve.image_height, ve.image_width)),
            )
            backbone_cfg = {
                "in_channels": 3,
                "channels": list(ve.backbone.channels),
                "kernel_sizes": list(ve.backbone.kernel_sizes),
                "strides": list(ve.backbone.strides),
                "mlp_hidden_dims": list(ve.backbone.mlp_hidden_dims),
                "output_dim": ve.backbone.output_dim,
            }
            if ve.cnn_checkpoint is not None:
                _model = multitask_cnn_from_checkpoint(
                    path=ve.cnn_checkpoint,
                    backbone_cfg=backbone_cfg,
                    device=self.device,
                )
            else:
                _model = MultiTaskCnn(backbone_cfg=backbone_cfg, heads_cfg=None)
            self.vision_feature_extractor = CnnSpatialSoftmaxFeatureExtractor(
                model=_model, device=self.device
            )
            _image_pipeline_steps.append(ClampPipelineStep())
        else:
            raise ValueError(
                f"Unknown vision_encoder.type: {_vision_type!r}. "
                "Must be 'frozen_resnet18' or 'frozen_cnn'."
            )
        self.image_pipeline = ImagePipeline(_image_pipeline_steps)

        _vd_cfg = self.cfg.debug.vision_debug
        if _vd_cfg.enabled:
            if self.cfg.log_dir is None:
                raise ValueError(
                    "cfg.log_dir must be set before the environment is created "
                    "when debug.vision_debug.enabled is True."
                )
        self._vision_debug_logger = VisionDebugLogger(
            extractor=self.vision_feature_extractor,
            log_dir=self.cfg.log_dir or "",
            cfg=_vd_cfg,
        )

    # Called by super class to setup the scene
    def _setup_scene(self):

        # 1) Create assets
        self.table = RigidObject(self.cfg.table_cfg)
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)

        self.camera = TiledCamera(self.cfg.camera_cfg)
        self.overhead_camera = (
            Camera(self.cfg.overhead_camera_cfg)
            if self.cfg.overhead_camera_cfg is not None
            else None
        )
        self.gripper_contact_sensor = ContactSensor(self.cfg.gripper_contact_sensor_cfg)
        self.table_contact_sensor = ContactSensor(self.cfg.table_contact_sensor_cfg)
        self.gripper_tf = FrameTransformer(self.cfg.gripper_transforms_cfg)

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
        if self.overhead_camera is not None:
            self.scene.sensors["overhead_camera"] = self.overhead_camera
        self.scene.sensors["gripper_contact_sensor"] = self.gripper_contact_sensor
        self.scene.sensors["table_contact_sensor"] = self.table_contact_sensor
        self.scene.sensors["gripper_tf"] = self.gripper_tf

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

        if self.cfg.debug.enable_camera_frame_markers:
            self.camera_frame_markers = define_camera_frame_markers()

        if self.cfg.debug.enable_gripper_arrow_markers:
            self.gripper_arrow_markers = define_gripper_arrow_markers()

        if self.cfg.debug.enable_grip_zone_markers:
            self.grip_zone_markers = define_grip_zone_markers()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before stepping the physics; store and scale actions."""
        if actions is None:
            return

        actions = torch.clamp(actions, -1.0, 1.0)
        t = 0.5 * (actions + 1.0)  # (num_envs, num_actions)
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        if self.cfg.behavior.binary_gripper_action.enabled:
            # make the target_pos of the gripper binary: fully open or set to the "grip" position
            t[:, self._ee_body_idx] = torch.where(
                t[:, self._ee_body_idx] > 0.5,
                torch.tensor(self.cfg.gripper.open_target, device=self.device),
                torch.tensor(self.cfg.gripper.closed_target, device=self.device),
            )

        target_pos = self._joint_lower + t * (self._joint_upper - self._joint_lower)
        self._target_pos = target_pos

        if self.cfg.debug.enable_camera_frame_markers:
            ee_pos = self.robot.data.body_pos_w[:, self._ee_body_idx[0], :]
            ee_quat = self.robot.data.body_quat_w[:, self._ee_body_idx[0], :]
            cam_pos_w = ee_pos + math_utils.quat_apply(
                ee_quat, self._camera_offset_pos.expand(self.num_envs, -1)
            )
            cam_quat_w = math_utils.quat_mul(
                ee_quat, self._camera_offset_quat.expand(self.num_envs, -1)
            )
            visualize_camera_frame_markers(
                self.camera_frame_markers, cam_pos_w, cam_quat_w, self.device
            )

        if (
            self.cfg.debug.enable_grip_zone_markers
            and self.env_metrics.get("grip_zone_offset") is not None
        ):
            src_pos = self.gripper_tf.data.source_pos_w
            src_quat = self.gripper_tf.data.source_quat_w
            gripper_pos_w = src_pos[:, 0, :] if src_pos.ndim == 3 else src_pos
            gripper_quat_w = src_quat[:, 0, :] if src_quat.ndim == 3 else src_quat
            gz_pos_w = gripper_pos_w + math_utils.quat_apply(
                gripper_quat_w, self.env_metrics["grip_zone_offset"]
            )
            visualize_grip_zone_markers(
                self.grip_zone_markers, gz_pos_w, gripper_quat_w, self.device
            )

    def _apply_action(self) -> None:
        """Apply scaled actions as joint position targets."""
        self.robot.set_joint_position_target(self._target_pos, joint_ids=self._dof_idx)

    def _get_observations(self) -> dict:
        """Return visual features + joint positions."""

        # Raw camera RGB: (num_envs, H, W, 3), uint8
        camera_data = self.camera.data.output["rgb"]

        _pre_cfg = self.cfg.debug.save_images.pre_processing
        if _pre_cfg.save and self.common_step_counter % _pre_cfg.interval == 0:
            if self.cfg.log_dir is None:
                raise RuntimeError(
                    "cfg.log_dir must be set before the environment is created."
                )
            pre_dir = os.path.join(self.cfg.log_dir, "debug_images", "pre_processing")
            os.makedirs(pre_dir, exist_ok=True)
            # camera_data is (N, H, W, 3) uint8 — convert to (3, H, W) float for save_image
            save_image(
                camera_data[0].permute(2, 0, 1).float() / 255.0,
                os.path.join(pre_dir, f"camera_{self.common_step_counter:06d}.png"),
            )

        # Apply image pipeline (uint8→float CHW conversion + any enabled DR steps)
        images = self.image_pipeline.process(camera_data)

        _post_cfg = self.cfg.debug.save_images.post_processing
        if _post_cfg.save and self.common_step_counter % _post_cfg.interval == 0:
            if self.cfg.log_dir is None:
                raise RuntimeError(
                    "cfg.log_dir must be set before the environment is created."
                )
            post_dir = os.path.join(self.cfg.log_dir, "debug_images", "post_processing")
            os.makedirs(post_dir, exist_ok=True)
            save_image(
                images[0],  # (3, H, W)
                os.path.join(post_dir, f"camera_{self.common_step_counter:06d}.png"),
            )

        # Proprioception
        q = self.joint_pos[:, self._dof_idx]  # (N, num_joints)
        dq = self.joint_vel[:, self._dof_idx]  # (N, num_joints) - Joint velocities

        # Actor observations: frozen feature extractor (RN18 or CNN) + joint positions
        visual_features = self.vision_feature_extractor.extract(
            images
        )  # (N, feature_dim)

        # Log vision debug visualisations to TensorBoard (no-op when disabled)
        self._vision_debug_logger.log(camera_data, images, self.common_step_counter)

        actor_obs = torch.cat([visual_features, q], dim=-1)

        # Compute on first observation call (when _get_observations is called from elf._env.reset())
        if self.step_metrics is None:
            print("Computing initial step metrics...")
            self._compute_step_metrics()

        critic_obs = torch.cat(
            [
                q,  # (N, num_joints)
                dq,  # (N, num_joints)
                *[
                    self.step_metrics[key].reshape(self.num_envs, KEY_OBS_DIMS[key])
                    for key in self.cfg.observations.critic_obs_metrics
                ],
            ],
            dim=-1,
        )  # (N, state_space)

        observations = {"policy": actor_obs, "critic": critic_obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.reward_pipeline.compute(self._step_ctx)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        if "log" not in self.extras:
            self.extras["log"] = {}
        if "per_env_log" not in self.extras:
            self.extras["per_env_log"] = {}

        self._compute_step_metrics()
        for key in self.step_metrics:
            if (
                self.step_metrics[key].size(-1) == 1
                or self.step_metrics[key].dim() == 1
            ):
                val = self.step_metrics[key].float()
                self.extras["log"][f"Step_Metrics/{key}"] = val.mean()
                self.extras["per_env_log"][f"Step_Metrics/{key}"] = val

        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminal = self.reward_pipeline.get_dones(self._step_ctx)

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
                noise_range[0],
                noise_range[1],
                joint_pos[:, self._dof_idx].shape,
                self.device,
            )

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write changes back to simulation
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.prev_actions[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Env metrics (per-episode values derived from reset state, e.g. DR cube scale)
        # Must run BEFORE DR so DR steps can consume env metrics (e.g. dr_cube_scale).
        self.env_metric_pipeline.apply(DRContext(env=self, env_ids=env_ids))

        # Domain Randomization
        self.dr_pipeline.apply(DRContext(env=self, env_ids=env_ids))

    def _compute_step_metrics(self) -> None:
        """Compute custom metrics at each step for logging purposes."""
        self.metric_pipeline.compute(self._step_ctx)
        self.step_metrics = self._step_ctx.metrics

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
