# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from skrl."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--cnn_checkpoint",
    type=str,
    default=None,
    help=(
        "Path to a pretrained MultiTaskCnn checkpoint produced by "
        "train_cnn.py (best_model.pt or final_model.pt). "
        "Loaded into vision_encoder.cnn_checkpoint before the environment is constructed."
    ),
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch
import numpy as np
from pathlib import Path

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import so101_rl.tasks  # noqa: F401
from so101_rl.configurations.camera import OVERHEAD_CAMERA_CFG

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    )
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Wire the CNN checkpoint path into vision_encoder so the env
    # constructor loads it via multitask_cnn_from_checkpoint at startup.
    if args_cli.cnn_checkpoint:
        if (
            getattr(env_cfg, "vision_encoder", None) is None
            or env_cfg.vision_encoder.type != "frozen_cnn"
        ):
            raise ValueError(
                "--cnn_checkpoint requires vision_encoder.type == "
                "'frozen_cnn'.  Switch to a frozen_cnn env config."
            )
        env_cfg.vision_encoder.cnn_checkpoint = args_cli.cnn_checkpoint

    # Enable overhead camera when recording video so camera_povs/ gets an overhead track
    if args_cli.video and getattr(env_cfg, "overhead_camera_cfg", None) is None:
        env_cfg.overhead_camera_cfg = OVERHEAD_CAMERA_CFG.replace(
            prim_path="/World/envs/env_.*/overhead_camera"
        )

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Setup camera POV recording (gripper + overhead)
        camera_video_dir = Path(log_dir) / "videos" / "play" / "camera_povs"
        camera_video_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Camera POV videos will be saved to: {camera_video_dir}")

        # Storage for camera frames (one list per environment) [legacy placeholder]
        camera_frames = {i: [] for i in range(env_cfg.scene.num_envs)}

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"][
        "write_interval"
    ] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"][
        "checkpoint_interval"
    ] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # Initialize video writers for camera POVs if recording
    camera_video_writers = None
    overhead_video_writers = None
    if args_cli.video:
        import cv2

        camera_video_writers = {}
        overhead_video_writers = {}

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in env.possible_agents
                }
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

            # Save camera frames if recording (gripper and overhead POVs)
            if args_cli.video and hasattr(env.unwrapped, "scene"):
                try:
                    # Access gripper camera (SO101 tasks) or generic 'camera'
                    gripper_cam = env.unwrapped.scene.sensors.get(
                        "gripper_camera"
                    ) or env.unwrapped.scene.sensors.get("camera")
                    if gripper_cam is not None and "rgb" in gripper_cam.data.output:
                        gripper_rgb = gripper_cam.data.output[
                            "rgb"
                        ]  # (num_envs, H, W, 3) uint8

                        # Save frames for each environment
                        for env_idx in range(gripper_rgb.shape[0]):
                            frame = gripper_rgb[env_idx].cpu().numpy()  # (H, W, 3)

                            # Initialize video writer on first frame
                            if env_idx not in camera_video_writers:
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                video_path = str(
                                    camera_video_dir / f"gripper_env_{env_idx:03d}.mp4"
                                )
                                fps = 30  # 30 FPS
                                camera_video_writers[env_idx] = cv2.VideoWriter(
                                    video_path, fourcc, fps, (w, h)
                                )

                            # Convert RGB to BGR for OpenCV and write frame
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            camera_video_writers[env_idx].write(frame_bgr)

                    # Access overhead camera if available
                    overhead_cam = env.unwrapped.scene.sensors.get("overhead_camera")
                    if overhead_cam is not None and "rgb" in overhead_cam.data.output:
                        overhead_rgb = overhead_cam.data.output[
                            "rgb"
                        ]  # (num_envs, H, W, 3) uint8

                        for env_idx in range(overhead_rgb.shape[0]):
                            frame = overhead_rgb[env_idx].cpu().numpy()

                            if env_idx not in overhead_video_writers:
                                h, w = frame.shape[:2]
                                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                                video_path = str(
                                    camera_video_dir / f"overhead_env_{env_idx:03d}.mp4"
                                )
                                fps = 30
                                overhead_video_writers[env_idx] = cv2.VideoWriter(
                                    video_path, fourcc, fps, (w, h)
                                )

                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            overhead_video_writers[env_idx].write(frame_bgr)
                except Exception as e:
                    print(f"[WARNING] Failed to save camera frame: {e}")

        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Release camera video writers
    if args_cli.video and (camera_video_writers or overhead_video_writers):
        print(f"[INFO] Finalizing camera POV videos...")
        if camera_video_writers:
            for env_idx, writer in camera_video_writers.items():
                writer.release()
        if overhead_video_writers:
            for env_idx, writer in overhead_video_writers.items():
                writer.release()
        total = (len(camera_video_writers) if camera_video_writers else 0) + (
            len(overhead_video_writers) if overhead_video_writers else 0
        )
        print(f"[INFO] Saved {total} camera POV videos to: {camera_video_dir}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
