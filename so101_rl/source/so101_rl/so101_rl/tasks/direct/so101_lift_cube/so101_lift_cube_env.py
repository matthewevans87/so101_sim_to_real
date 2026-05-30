# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
import math
import os
import re

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
    define_goal_zone_markers,
    visualize_grip_zone_markers,
    visualize_gripper_arrow,
    visualize_camera_frame_markers,
    visualize_goal_zone_markers,
)
from so101_rl.helpers.utils import set_material

from torchvision.utils import save_image
from so101_rl.configurations.camera import (
    CAMERA_ROTATION_QUAT_WXYZ,
    CAMERA_TRANSLATE_VEC,
    CAMERA_POST_SPAWN_USD_ATTRS,
)
from so101_rl.helpers.opencv_to_isaac_camera import apply_post_spawn_attrs
from .so101_lift_cube_env_cfg import So101LiftCubeCfg
from so101_rl.viz.vision_debug import VisionDebugLogger
from so101_rl.env_pipeline import (
    DRContext,
    DRPipeline,
    EnvMetricPipeline,
    StepContext,
    MetricPipeline,
    RewardPipeline,
    TerminationPipeline,
    build_dr_pipeline,
    build_env_metric_pipeline,
    build_metric_pipeline,
    build_reward_pipeline,
    build_termination_pipeline,
    validate_gate_metrics,
    validate_termination_gate_metrics,
    KEY_OBS_DIMS,
    EpisodeStatsPipeline,
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

        # Apply post-spawn USD attributes for fisheyeRadTanThinPrism camera model.
        # These parameters (openCVFx/Fy, tangential p0/p1, thin-prism s0-s3) are not
        # exposed in FisheyeCameraCfg and must be set directly on each camera prim.
        # self.camera._view.prim_paths is populated by TiledCamera._initialize_impl(),
        # which runs via scene.update() inside super().__init__().
        for prim_path in self.camera._view.prim_paths:
            prim = self.camera.stage.GetPrimAtPath(prim_path)
            apply_post_spawn_attrs(prim, CAMERA_POST_SPAWN_USD_ATTRS)

        # Get handles to data views
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # Find indices of DOFs and EE link
        self._dof_idx, _ = self.robot.find_joints(self.cfg.joints.active)
        self._all_joint_idx, _ = self.robot.find_joints(self.cfg.joints.all)
        self._ee_body_idx, _ = self.robot.find_bodies(self.cfg.gripper.ee_link_name)
        self._moving_jaw_body_idx, _ = self.robot.find_bodies("moving_jaw_so101_v1")
        self._gripper_joint_idx, _ = self.robot.find_joints(
            [self.cfg.gripper.ee_link_name]
        )
        self._wrist_roll_joint_idx, _ = self.robot.find_joints(
            [self.cfg.joints.wrist_roll_name]
        )

        # ── Joint-noise FK-safety guard pre-resolution ─────────────────────
        # Resolve tracked-link body indices once at construction so the
        # per-reset rejection-sampling loop is index-only (no string matching
        # in the hot path).  ``find_bodies`` uses *regex* matching against
        # ``self.robot.body_names``, so we anchor each user-supplied name
        # with ^...$ to forbid accidental substring matches (e.g. plain
        # ``shoulder`` would otherwise match ``shoulder_pan_link``).  Only
        # resolved when the guard is enabled in the config; otherwise left
        # as None so an accidental access raises.
        _spn_cfg = self.cfg.joints.starting_position_noise
        if _spn_cfg.enabled and _spn_cfg.fk_safety.enabled:
            _patterns = [f"^{re.escape(n)}$" for n in _spn_cfg.fk_safety.tracked_links]
            _idxs, _names = self.robot.find_bodies(_patterns, preserve_order=True)
            if len(_idxs) != len(_spn_cfg.fk_safety.tracked_links):
                raise RuntimeError(
                    f"FK-safety: tracked_links="
                    f"{_spn_cfg.fk_safety.tracked_links!r} resolved to "
                    f"{_names!r} (indices={_idxs!r}); expected exact "
                    f"one-to-one match.  Available body names: "
                    f"{list(self.robot.body_names)!r}."
                )
            self._fk_safety_link_idxs: list[int] | None = _idxs
        else:
            self._fk_safety_link_idxs = None

        # Per-joint noise ranges as a (num_all_joints, 2) float tensor on the
        # device, ordered to match self._all_joint_idx (i.e. cfg.joints.all).
        if _spn_cfg.enabled:
            self._joint_noise_ranges = torch.tensor(
                [list(r) for r in _spn_cfg.ranges],
                device=self.device,
                dtype=torch.float32,
            )  # (num_all_joints, 2)
        else:
            self._joint_noise_ranges = None  # type: ignore[assignment]

        # Table top z in the world frame.  The canonical scene places the
        # table at env-local pos (0.45, 0.0, -0.5) with size (2, 2, 1) and
        # env_origins[:, 2] == 0, so the table top sits at world-z = 0.0
        # uniformly across envs.  Hard-code here with a comment so future
        # scene edits surface this assumption.
        self._table_top_z: float = 0.0

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

        # Joint-command smoother: shared sim/real action pipeline.
        if self.cfg.joint_command.enabled:
            from so101.utils.control import JointCommandSmoother

            _active = list(self.cfg.joints.active)
            _ema_names = set(self.cfg.joint_command.ema_joints or [])
            _clamp_names = set(self.cfg.joint_command.clamp_joints or [])
            _ema_mask = (
                torch.tensor([name in _ema_names for name in _active], dtype=torch.bool)
                if _ema_names
                else None
            )
            _clamp_mask = (
                torch.tensor(
                    [name in _clamp_names for name in _active], dtype=torch.bool
                )
                if _clamp_names
                else None
            )
            self._smoother = JointCommandSmoother(
                lower_rad=joint_lower_1d,  # (n_joints,) — 1-D, not unsqueezed
                upper_rad=joint_upper_1d,
                ema_alpha=self.cfg.joint_command.ema_alpha,
                max_delta_rad=self.cfg.joint_command.max_delta_rad,
                ema_mask=_ema_mask,
                clamp_mask=_clamp_mask,
            )
        else:
            self._smoother = None

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
        self.termination_pipeline: TerminationPipeline = build_termination_pipeline(
            self.cfg
        )
        self.env_metric_pipeline: EnvMetricPipeline = build_env_metric_pipeline(
            self.cfg
        )
        self.episode_stats_pipeline = EpisodeStatsPipeline(
            num_envs=self.num_envs,
            device=self.device,
            lift_height_threshold=self.cfg.episode_stats.lift_height_threshold,
        )
        self.metric_pipeline: MetricPipeline = build_metric_pipeline(
            self.reward_pipeline,
            extra_keys=frozenset(
                {
                    # consumed by _get_observations (actor features)
                    *(self.cfg.observations.actor_obs_metrics or []),
                    # consumed by _get_observations (critic features)
                    *(self.cfg.observations.critic_obs_metrics or []),
                    # always computed for telemetry collection (does not affect obs space)
                    *self.cfg.observations.telemetry_metrics,
                    # consumed by episode_stats_pipeline
                    *EpisodeStatsPipeline.required_metric_keys,
                    # consumed by termination_pipeline gates
                    *self.termination_pipeline.required_metric_keys,
                }
            ),
            env_metric_pipeline=self.env_metric_pipeline,
        )
        validate_gate_metrics(
            self.reward_pipeline, self.metric_pipeline, self.env_metric_pipeline
        )
        validate_termination_gate_metrics(
            self.termination_pipeline,
            self.metric_pipeline,
            self.env_metric_pipeline,
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
        self.moving_jaw_contact_sensor = ContactSensor(
            self.cfg.moving_jaw_contact_sensor_cfg
        )
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
        self.scene.sensors["moving_jaw_contact_sensor"] = self.moving_jaw_contact_sensor
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

        # Lights — dome provides soft ambient fill; key light adds a
        # directional component approximating the real workspace desk lamp.
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        key_light_cfg = sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(1.0, 0.95, 0.85),  # warm white matching typical desk lamp
            angle=0.53,  # degrees — narrow cone for defined shadows
        )
        # Orientation (w, x, y, z): tilt ~35° from vertical toward front-right
        # of the workspace so it approximates the real desk lamp direction.
        # Euler ZYX ≈ (0°, 35°, -45°) → quaternion computed offline.
        key_light_cfg.func(
            "/World/KeyLight",
            key_light_cfg,
            orientation=(0.8924, 0.2392, -0.3604, 0.0966),
        )

        if self.cfg.debug.enable_camera_frame_markers:
            self.camera_frame_markers = define_camera_frame_markers()

        if self.cfg.debug.enable_gripper_arrow_markers:
            self.gripper_arrow_markers = define_gripper_arrow_markers()

        if self.cfg.debug.enable_grip_zone_markers:
            self.grip_zone_markers = define_grip_zone_markers()

        if self.cfg.debug.enable_goal_zone_markers:
            self.goal_zone_markers = define_goal_zone_markers(
                radius=self.cfg.metrics.goal_zone_distance.distance_threshold
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before stepping the physics; store and scale actions."""
        if actions is None:
            return

        self.prev_actions = self.actions.clone()
        self.actions = actions.clone()

        if self._smoother is not None:
            # Shared action pipeline: normalized → canonical + EMA + delta clamp + limits.
            # Apply binary gripper override in normalized action space before the smoother
            # so it goes through EMA and delta clamping like any other joint.
            actions_cmd = actions
            if self.cfg.behavior.binary_gripper_action.enabled:
                open_norm = 2.0 * self.cfg.gripper.open_target - 1.0
                closed_norm = 2.0 * self.cfg.gripper.closed_target - 1.0
                actions_cmd = actions.clone()
                g = actions_cmd[:, self._ee_body_idx]
                actions_cmd[:, self._ee_body_idx] = torch.where(
                    g > 0.0,
                    torch.full_like(g, open_norm),
                    torch.full_like(g, closed_norm),
                )
            q_current = self.joint_pos[:, self._dof_idx]
            self._target_pos = self._smoother.step(actions_cmd, q_current)
        else:
            # Legacy path (joint_command.enabled: false — ablation only).
            actions_clamped = torch.clamp(actions, -1.0, 1.0)
            t = 0.5 * (actions_clamped + 1.0)
            if self.cfg.behavior.binary_gripper_action.enabled:
                t[:, self._ee_body_idx] = torch.where(
                    t[:, self._ee_body_idx] > 0.5,
                    torch.tensor(self.cfg.gripper.open_target, device=self.device),
                    torch.tensor(self.cfg.gripper.closed_target, device=self.device),
                )
            self._target_pos = self._joint_lower + t * (
                self._joint_upper - self._joint_lower
            )

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

        if (
            self.cfg.debug.enable_goal_zone_markers
            and self.env_metrics.get("goal_zone_pos_w") is not None
        ):
            visualize_goal_zone_markers(
                self.goal_zone_markers, self.env_metrics["goal_zone_pos_w"], self.device
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

        # Append any configured actor-only metric observations (e.g. goal_zone_pos_local).
        if self.cfg.observations.actor_obs_metrics:
            actor_obs = torch.cat(
                [
                    actor_obs,
                    *[
                        self.step_metrics[key].reshape(self.num_envs, KEY_OBS_DIMS[key])
                        for key in self.cfg.observations.actor_obs_metrics
                    ],
                ],
                dim=-1,
            )

        critic_obs_parts: list[torch.Tensor] = []
        if self.cfg.observations.critic_include_vision_features:
            critic_obs_parts.append(visual_features)  # (N, vision_feature_dim)
        critic_obs_parts.extend(
            [
                q,  # (N, num_joints)
                dq,  # (N, num_joints)
                *[
                    self.step_metrics[key].reshape(self.num_envs, KEY_OBS_DIMS[key])
                    for key in self.cfg.observations.critic_obs_metrics
                ],
            ]
        )
        critic_obs = torch.cat(critic_obs_parts, dim=-1)  # (N, state_space)

        # NOTE (skrl 1.4.3 — asymmetric actor-critic NOT supported by wrapper)
        # ----------------------------------------------------------------------
        # We return the standard {"policy", "critic"} dict that the IsaacLab
        # asymmetric AC contract specifies, BUT the installed skrl 1.4.3
        # IsaacLabWrapper.step() only reads observations["policy"] and feeds
        # those tensors to BOTH the policy and value networks (regardless of
        # models.separate).  See:
        #   ~/.conda/envs/env_isaaclab/lib/python3.11/site-packages/skrl/envs/
        #     wrappers/torch/isaaclab_envs.py
        #   IsaacLabWrapper.step:
        #       self._observations = flatten_tensorized_space(
        #           tensorize_space(self.observation_space,
        #                           observations["policy"]))   # critic ignored
        #
        # Empirical confirmation: see sweep ablation_shared_critic
        # (sweep_ablation_shared_critic_20260423_121336/summary.md) — holding
        # models.separate=true fixed and only toggling the critic obs payload
        # produced statistically indistinguishable training/eval results.
        #
        # The "critic" entry is kept here so this code is correct the moment
        # skrl gains real asymmetric-AC support; until then a startup guard
        # in train.py raises if a non-empty critic-only payload is configured.
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
            t = self.step_metrics[key].float()
            if t.dim() == 1 or t.size(-1) == 1:
                # Scalar metric — log directly.
                val = t.reshape(self.num_envs)
                self.extras["log"][f"Step_Metrics/{key}"] = val.mean()
                self.extras["per_env_log"][f"Step_Metrics/{key}"] = val
            else:
                # Vector metric — log the L2 norm so it's visible in TensorBoard.
                val = t.norm(dim=-1)  # (num_envs,)
                self.extras["log"][f"Step_Metrics/{key}_norm"] = val.mean()
                self.extras["per_env_log"][f"Step_Metrics/{key}_norm"] = val

        # Episode timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminal = self.termination_pipeline.get_dones(self._step_ctx)

        # Log per-reason termination signals for TensorBoard diagnostics.
        self.extras["log"]["Termination/time_out"] = time_out.float().mean()
        self.extras["per_env_log"]["Termination/time_out"] = time_out.float()
        for name, flags in self.termination_pipeline.get_done_reasons(
            self._step_ctx
        ).items():
            self.extras["log"][f"Termination/{name}"] = flags.mean()
            self.extras["per_env_log"][f"Termination/{name}"] = flags

        self.episode_stats_pipeline.step(
            self._step_ctx,
            terminal,
            time_out,
            common_step_counter=self.common_step_counter,
        )
        for key, val in self.episode_stats_pipeline.get_log_dict().items():
            self.extras["log"][key] = torch.tensor(val, device=self.device)

        return terminal, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset robot state for the given env Ids."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)
        self.episode_stats_pipeline.reset_envs(env_ids)

        # Reset robot to default joint state and root from asset
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # Apply starting position from config (joints.all order, degrees or null)
        for i, deg in enumerate(self.cfg.joints.starting_position):
            if deg is not None:
                joint_pos[:, self._all_joint_idx[i]] = math.radians(deg)

        # ── Per-joint starting-position noise + FK rejection-sampling ──────
        # The previous implementation added a single global ±0.4 rad of noise
        # to every active joint, which could dip the arm into the table at
        # t=0 and trigger ``safety_touch_table_terminal``.  The new path:
        #   1. samples per-joint ranges (cfg.joints.starting_position_noise.
        #      ranges, aligned with cfg.joints.all);
        #   2. when fk_safety.enabled, writes the candidate joints to sim,
        #      reads tracked-link world-positions via Isaac Lab's kinematic
        #      FK refresh (Articulation.data.body_pos_w), and resamples any
        #      env whose minimum tracked-link z is below the table top by
        #      less than fk_safety.min_link_z_above_table.
        # The loop is vectorised over envs and bounded by max_resamples.
        spn_cfg = self.cfg.joints.starting_position_noise
        if spn_cfg.enabled:
            n_envs = joint_pos.shape[0]
            ranges = self._joint_noise_ranges  # (num_all_joints, 2)
            assert ranges is not None  # set in __init__ when spn_cfg.enabled
            lows = ranges[:, 0]
            highs = ranges[:, 1]
            num_all = ranges.shape[0]

            # Snapshot the noise-free "base" positions for the noised joints
            # (i.e. the cfg.joints.all columns) so we can re-derive new
            # candidates on every resample iteration without losing the
            # nominal starting position.
            base_all = joint_pos[:, self._all_joint_idx].clone()  # (n_envs, num_all)

            if self._fk_safety_link_idxs is not None:
                env_ids_t = torch.as_tensor(
                    list(env_ids), device=self.device, dtype=torch.long
                )
                min_z_required = (
                    self._table_top_z + spn_cfg.fk_safety.min_link_z_above_table
                )
                accepted = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
                # Working candidate buffer; rows for accepted envs are frozen
                # from the iteration in which they were accepted.
                candidate = base_all.clone()

                for _attempt in range(spn_cfg.fk_safety.max_resamples + 1):
                    need = ~accepted
                    n_need = int(need.sum().item())
                    if n_need == 0:
                        break

                    # Sample new noise for the not-yet-accepted envs only.
                    rand01 = torch.empty(
                        n_need, num_all, device=self.device, dtype=torch.float32
                    ).uniform_(0.0, 1.0)
                    sample = lows + rand01 * (highs - lows)
                    candidate[need] = base_all[need] + sample

                    # Stage the FULL joint vector for the resetting envs and
                    # write to sim so the kinematic FK refresh sees the
                    # candidate pose.
                    staged = joint_pos.clone()
                    staged[:, self._all_joint_idx] = candidate
                    self.robot.write_joint_position_to_sim(staged, env_ids=env_ids_t)

                    # Reading body_pos_w invalidates the timestamp set by
                    # write_joint_position_to_sim and triggers
                    # _physics_sim_view.update_articulations_kinematic(),
                    # which is a pure FK call (no physics step).
                    body_z = self.robot.data.body_pos_w[
                        env_ids_t[:, None], self._fk_safety_link_idxs, 2
                    ]  # (n_envs, n_tracked_links)
                    min_link_z = body_z.min(dim=1).values  # (n_envs,)
                    newly_ok = (min_link_z >= min_z_required) & need
                    accepted = accepted | newly_ok

                if not bool(accepted.all().item()):
                    n_failed = int((~accepted).sum().item())
                    raise RuntimeError(
                        f"Joint-noise FK safety: {n_failed}/{n_envs} envs "
                        f"could not find a valid joint pose within "
                        f"{spn_cfg.fk_safety.max_resamples} resamples "
                        f"(min link z must be >= "
                        f"{min_z_required:.4f} m).  Tighten "
                        f"joints.starting_position_noise.ranges or relax "
                        f"fk_safety.min_link_z_above_table."
                    )

                joint_pos[:, self._all_joint_idx] = candidate
            else:
                # Single-shot per-joint noise without FK guard.
                rand01 = torch.empty(
                    n_envs, num_all, device=self.device, dtype=torch.float32
                ).uniform_(0.0, 1.0)
                sample = lows + rand01 * (highs - lows)
                joint_pos[:, self._all_joint_idx] = base_all + sample

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write changes back to simulation
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.prev_actions[env_ids] = 0.0

        # Reset smoother EMA state for the resetting environments so the first
        # commanded target after reset is delta-clamped from the post-reset pose.
        if self._smoother is not None:
            _env_ids_t = torch.as_tensor(
                list(env_ids), device=self.device, dtype=torch.long
            )
            self._smoother.reset(joint_pos[:, self._dof_idx], env_ids=_env_ids_t)

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Env metrics (per-episode values derived from reset state, e.g. DR cube scale)
        # Must run BEFORE DR so DR steps can consume env metrics (e.g. dr_cube_scale).
        self.env_metric_pipeline.apply(DRContext(env=self, env_ids=env_ids))

        # Domain Randomization
        self.dr_pipeline.apply(DRContext(env=self, env_ids=env_ids))

        # Compute post-reset metrics and initialize prev_metrics for the reset envs so
        # that progressive rewards see a delta of 0 on the first step of each episode.
        self.metric_pipeline.compute(self._step_ctx)
        self.step_metrics = self._step_ctx.metrics
        for key, val in self._step_ctx.metrics.items():
            if key in self._step_ctx.prev_metrics:
                self._step_ctx.prev_metrics[key][env_ids] = val[env_ids].clone()
            else:
                self._step_ctx.prev_metrics[key] = val.clone()

        # Clear fire_once state for reset envs so terminal steps can fire again.
        self.reward_pipeline.reset_idx(env_ids)
        self.termination_pipeline.reset_idx(env_ids)

    def _compute_step_metrics(self) -> None:
        """Compute custom metrics at each step for logging purposes."""
        # Snapshot the current metrics as prev_metrics before the pipeline clears and
        # repopulates ctx.metrics. This gives progressive reward steps their delta baseline.
        for key, val in self._step_ctx.metrics.items():
            self._step_ctx.prev_metrics[key] = val.clone()
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
