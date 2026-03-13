# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate a trained RL agent from skrl and collect comprehensive metrics.

This script runs evaluation episodes with a single environment (defaults to 100), recording:
- Step metrics and rewards for all episodes
- Videos from perspective and wrist cameras for the first 5 episodes by default
- Results saved to JSON file
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Evaluate a trained RL agent from skrl with comprehensive metrics collection."
)
parser.add_argument(
    "--experiment-path",
    type=str,
    required=True,
    help="Path to experiment directory (e.g., artifacts/2026-03-12_09-52-10)",
)
parser.add_argument(
    "--task", type=str, default=None, help="Name of the task (e.g., So101-LiftCube-v0)."
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed used for the environment"
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
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=100,
    help="Number of evaluation episodes to run.",
)
parser.add_argument(
    "--num-videos",
    type=int,
    default=5,
    help="Number of episodes to record videos for.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for evaluation
args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

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

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import so101_rl.tasks  # noqa: F401

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


def find_checkpoint_and_task(experiment_path: Path) -> tuple[Path, str]:
    """Find the checkpoint file and task name from experiment directory."""
    # Look for checkpoint in skrl subdirectory
    skrl_dir = experiment_path / "skrl"
    if not skrl_dir.exists():
        raise FileNotFoundError(f"No skrl directory found in {experiment_path}")
    
    # Find task directory (should be only one subdirectory)
    task_dirs = [d for d in skrl_dir.iterdir() if d.is_dir()]
    if not task_dirs:
        raise FileNotFoundError(f"No task directory found in {skrl_dir}")
    if len(task_dirs) > 1:
        raise ValueError(f"Multiple task directories found: {task_dirs}")
    
    task_dir = task_dirs[0]
    task_name = task_dir.name
    
    # Look for best_agent.pt checkpoint
    checkpoint_dir = task_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "best_agent.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    return checkpoint_path, task_name


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Evaluate skrl agent."""
    
    # Parse experiment path
    experiment_path = Path(args_cli.experiment_path).resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")
    
    print(f"[INFO] Loading experiment from: {experiment_path}")
    
    # Find checkpoint and task name
    checkpoint_path, task_name_from_checkpoint = find_checkpoint_and_task(experiment_path)
    print(f"[INFO] Found checkpoint: {checkpoint_path}")
    print(f"[INFO] Task name from checkpoint: {task_name_from_checkpoint}")
    
    # Ensure task name matches if provided
    task_name = args_cli.task.split(":")[-1] if args_cli.task else task_name_from_checkpoint
    
    # Load env_config.yaml if it exists
    env_config_path = experiment_path / "env_config.yaml"
    if env_config_path.exists():
        print(f"[INFO] Found env_config.yaml: {env_config_path}")
        # Set environment variable for the environment to load
        os.environ["SO101_ENV_CONFIG"] = str(env_config_path)
    else:
        print(f"[WARNING] No env_config.yaml found at {env_config_path}")
    
    # Override configurations for evaluation
    env_cfg.scene.num_envs = 1  # Always use 1 environment for evaluation
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Configure the ML framework
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    
    # Set seed
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = experiment_cfg["seed"]
    
    # Create evaluation output directory
    eval_dir = experiment_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Evaluation results will be saved to: {eval_dir}")
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    
    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    
    # configure and instantiate the skrl runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)
    
    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner.agent.load(str(checkpoint_path))
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")
    
    # Evaluation parameters
    NUM_EPISODES = args_cli.num_episodes
    NUM_VIDEO_EPISODES = args_cli.num_videos
    
    # Storage for results
    all_episode_data = []
    
    # Video writers for first 5 episodes
    perspective_writer = None
    wrist_writer = None
    current_episode = 0
    episode_step = 0
    
    # Storage for current episode metrics
    current_episode_metrics = {
        "episode_num": 0,
        "steps": [],
        "total_reward": 0.0,
        "episode_length": 0,
    }
    
    # reset environment
    obs, info = env.reset()
    
    print(f"[INFO] Starting evaluation: {NUM_EPISODES} episodes")
    print(f"[INFO] Recording videos for first {NUM_VIDEO_EPISODES} episodes")
    
    # Run evaluation
    while current_episode < NUM_EPISODES:
        # Initialize video writers for first 5 episodes
        if current_episode < NUM_VIDEO_EPISODES and episode_step == 0:
            # Get first frame to determine video properties
            if hasattr(env.unwrapped, "scene"):
                try:
                    # Get gripper camera (wrist)
                    gripper_cam = env.unwrapped.scene.sensors.get("gripper_camera")
                    if gripper_cam is not None and "rgb" in gripper_cam.data.output:
                        gripper_rgb = gripper_cam.data.output["rgb"][0].cpu().numpy()
                        h, w = gripper_rgb.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        wrist_video_path = eval_dir / f"wrist_cam_episode_{current_episode:03d}.mp4"
                        wrist_writer = cv2.VideoWriter(str(wrist_video_path), fourcc, 30, (w, h))
                    
                    # Get overhead camera (perspective)
                    overhead_cam = env.unwrapped.scene.sensors.get("overhead_camera")
                    if overhead_cam is not None and "rgb" in overhead_cam.data.output:
                        overhead_rgb = overhead_cam.data.output["rgb"][0].cpu().numpy()
                        h, w = overhead_rgb.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        perspective_video_path = eval_dir / f"perspective_cam_episode_{current_episode:03d}.mp4"
                        perspective_writer = cv2.VideoWriter(str(perspective_video_path), fourcc, 30, (w, h))
                except Exception as e:
                    print(f"[WARNING] Failed to initialize video writers: {e}")
        
        # Run inference
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
            obs, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # Extract step-level data from environment
            step_data = {
                "step": episode_step,
                "reward": float(reward[0]),
            }
            
            # Get step metrics from environment if available
            if hasattr(env.unwrapped, "extras") and "log" in env.unwrapped.extras:
                extras_log = env.unwrapped.extras["log"]
                for key, value in extras_log.items():
                    if torch.is_tensor(value):
                        step_data[key] = float(value)
                    else:
                        step_data[key] = value
            
            current_episode_metrics["steps"].append(step_data)
            current_episode_metrics["total_reward"] += float(reward[0])
            
            # Record video frames if in recording episodes
            if current_episode < NUM_VIDEO_EPISODES and hasattr(env.unwrapped, "scene"):
                try:
                    # Wrist camera
                    if wrist_writer is not None:
                        gripper_cam = env.unwrapped.scene.sensors.get("gripper_camera")
                        if gripper_cam is not None and "rgb" in gripper_cam.data.output:
                            frame = gripper_cam.data.output["rgb"][0].cpu().numpy()
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            wrist_writer.write(frame_bgr)
                    
                    # Perspective camera
                    if perspective_writer is not None:
                        overhead_cam = env.unwrapped.scene.sensors.get("overhead_camera")
                        if overhead_cam is not None and "rgb" in overhead_cam.data.output:
                            frame = overhead_cam.data.output["rgb"][0].cpu().numpy()
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            perspective_writer.write(frame_bgr)
                except Exception as e:
                    print(f"[WARNING] Failed to record video frame: {e}")
            
            episode_step += 1
            
            # Check if episode is done
            if done[0]:
                current_episode_metrics["episode_length"] = episode_step
                all_episode_data.append(current_episode_metrics.copy())
                
                # Release video writers if recording
                if wrist_writer is not None:
                    wrist_writer.release()
                    wrist_writer = None
                    print(f"[INFO] Saved wrist camera video for episode {current_episode}")
                
                if perspective_writer is not None:
                    perspective_writer.release()
                    perspective_writer = None
                    print(f"[INFO] Saved perspective camera video for episode {current_episode}")
                
                # Move to next episode
                current_episode += 1
                episode_step = 0
                
                if current_episode < NUM_EPISODES:
                    # Reset environment
                    obs, info = env.reset()
                    
                    # Reset episode metrics
                    current_episode_metrics = {
                        "episode_num": current_episode,
                        "steps": [],
                        "total_reward": 0.0,
                        "episode_length": 0,
                    }
                    
                    # Print progress
                    if current_episode % 10 == 0:
                        print(f"[INFO] Completed {current_episode}/{NUM_EPISODES} episodes")
    
    print(f"[INFO] Evaluation complete: {NUM_EPISODES} episodes")
    
    # Compute summary statistics
    episode_rewards = [ep["total_reward"] for ep in all_episode_data]
    episode_lengths = [ep["episode_length"] for ep in all_episode_data]
    
    summary = {
        "num_episodes": NUM_EPISODES,
        "checkpoint_path": str(checkpoint_path),
        "task_name": task_name,
        "seed": args_cli.seed,
        "summary_statistics": {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "std_episode_length": float(np.std(episode_lengths)),
        },
        "episodes": all_episode_data,
    }
    
    # Save results to JSON
    results_path = eval_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"[INFO] Results saved to: {results_path}")
    print(f"[INFO] Summary statistics:")
    print(f"  Mean reward: {summary['summary_statistics']['mean_reward']:.3f} ± {summary['summary_statistics']['std_reward']:.3f}")
    print(f"  Mean episode length: {summary['summary_statistics']['mean_episode_length']:.1f} ± {summary['summary_statistics']['std_episode_length']:.1f}")
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
