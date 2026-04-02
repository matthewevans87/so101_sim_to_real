# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate a trained RL agent from skrl and collect comprehensive metrics.

This script runs evaluation episodes with one or more environments (defaults to 1), recording:
- Step metrics and rewards for all episodes
- Videos from overhead and wrist cameras for the first 5 episodes by default
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
from statistics import StatisticsError, mode
from tqdm import tqdm

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
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate.",
)
parser.add_argument(
    "--verbosity",
    type=str,
    default="basic",
    choices=["full", "basic"],
    help="Output verbosity for results.json (full includes step arrays).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras for evaluation
args_cli.enable_cameras = True

# Ensure overhead camera is enabled only for evaluation
os.environ["SO101_ENABLE_OVERHEAD_CAMERA"] = "1"

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
    """Find the checkpoint file from the experiment directory."""
    checkpoint_path = experiment_path / "skrl" / "checkpoints" / "best_agent.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return checkpoint_path, ""


def _first_step_index(values: list[float], predicate) -> int | None:
    for idx, val in enumerate(values):
        if predicate(val):
            return idx
    return None


def _summary_for_values(values: list[float]) -> dict:
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "mode": None,
            "stdev": None,
            "steps_till_min": None,
            "steps_till_max": None,
            "steps_till_value_gt_0": None,
            "steps_till_value_lt_0": None,
            "steps_till_value_eq_0": None,
        }

    min_val = float(np.min(values))
    max_val = float(np.max(values))
    mean_val = float(np.mean(values))
    stdev_val = float(np.std(values))

    rounded = [round(val, 6) for val in values]
    try:
        mode_val = float(mode(rounded))
    except StatisticsError:
        mode_val = None

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "mode": mode_val,
        "stdev": stdev_val,
        "steps_till_min": _first_step_index(values, lambda v: v == min_val),
        "steps_till_max": _first_step_index(values, lambda v: v == max_val),
        "steps_till_value_gt_0": _first_step_index(values, lambda v: v > 0),
        "steps_till_value_lt_0": _first_step_index(values, lambda v: v < 0),
        "steps_till_value_eq_0": _first_step_index(values, lambda v: v == 0),
    }


def _global_summary_for_key(
    flat_values: list[float], per_episode_summaries: list[dict]
) -> dict:
    """Compute global summary: statistics over all steps of all episodes, mean steps_till across episodes."""
    if not flat_values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "mode": None,
            "stdev": None,
            "mean_steps_till_min": None,
            "mean_steps_till_max": None,
            "mean_steps_till_value_gt_0": None,
            "mean_steps_till_value_lt_0": None,
            "mean_steps_till_value_eq_0": None,
        }

    min_val = float(np.min(flat_values))
    max_val = float(np.max(flat_values))
    mean_val = float(np.mean(flat_values))
    stdev_val = float(np.std(flat_values))

    rounded = [round(val, 6) for val in flat_values]
    try:
        mode_val = float(mode(rounded))
    except StatisticsError:
        mode_val = None

    def _mean_steps_till(key: str) -> float | None:
        vals = [s[key] for s in per_episode_summaries if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "mode": mode_val,
        "stdev": stdev_val,
        "mean_steps_till_min": _mean_steps_till("steps_till_min"),
        "mean_steps_till_max": _mean_steps_till("steps_till_max"),
        "mean_steps_till_value_gt_0": _mean_steps_till("steps_till_value_gt_0"),
        "mean_steps_till_value_lt_0": _mean_steps_till("steps_till_value_lt_0"),
        "mean_steps_till_value_eq_0": _mean_steps_till("steps_till_value_eq_0"),
    }


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
    checkpoint_path, task_name_from_checkpoint = find_checkpoint_and_task(
        experiment_path
    )
    print(f"[INFO] Found checkpoint: {checkpoint_path}")
    print(f"[INFO] Task name from checkpoint: {task_name_from_checkpoint}")

    # Ensure task name matches if provided
    task_name = (
        args_cli.task.split(":")[-1] if args_cli.task else task_name_from_checkpoint
    )

    # Load env_config.yaml if it exists
    env_config_path = experiment_path / "env_config.yaml"
    if env_config_path.exists():
        print(f"[INFO] Found env_config.yaml: {env_config_path}")
        # Set environment variable for the environment to load
        os.environ["SO101_ENV_CONFIG"] = str(env_config_path)
    else:
        print(f"[WARNING] No env_config.yaml found at {env_config_path}")

    # Override configurations for evaluation
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

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
    experiment_cfg["agent"]["experiment"][
        "write_interval"
    ] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"][
        "checkpoint_interval"
    ] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner.agent.load(str(checkpoint_path))
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # Evaluation parameters
    NUM_EPISODES = args_cli.num_episodes
    NUM_VIDEO_EPISODES = args_cli.num_videos
    num_envs = env_cfg.scene.num_envs

    # Storage for results
    all_episode_data = []
    completed_episodes = 0
    recorded_episodes = 0

    global_step_values: dict[str, list[float]] = {}
    global_reward_values: dict[str, list[float]] = {}
    global_step_summaries: dict[str, list[dict]] = {}
    global_reward_summaries: dict[str, list[dict]] = {}

    episode_counts = [0 for _ in range(num_envs)]
    episode_steps = [0 for _ in range(num_envs)]
    record_active = [False for _ in range(num_envs)]

    current_episode_metrics = [
        {
            "env_id": env_idx,
            "episode_num": 0,
            "steps": [],
            "total_reward": 0.0,
            "episode_length": 0,
        }
        for env_idx in range(num_envs)
    ]

    wrist_writers = {}
    overhead_writers = {}

    # reset environment
    obs, info = env.reset()

    print(f"[INFO] Starting evaluation: {NUM_EPISODES} episodes across {num_envs} envs")
    print(f"[INFO] Recording videos for first {NUM_VIDEO_EPISODES} episodes")

    # Run evaluation
    progress = tqdm(total=NUM_EPISODES, desc="Evaluating", unit="episode")
    while simulation_app.is_running() and completed_episodes < NUM_EPISODES:
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

        if torch.is_tensor(terminated):
            done = torch.logical_or(terminated, truncated)
        else:
            done = np.logical_or(terminated, truncated)

        extras_log = None
        extras_per_env_log = None
        if hasattr(env.unwrapped, "extras"):
            if "log" in env.unwrapped.extras:
                extras_log = env.unwrapped.extras["log"]
            if "per_env_log" in env.unwrapped.extras:
                extras_per_env_log = env.unwrapped.extras["per_env_log"]

        active_record_count = sum(record_active)

        for env_idx in range(num_envs):
            reward_val = reward[env_idx]
            if torch.is_tensor(reward_val):
                reward_val = reward_val.item()
            else:
                reward_val = float(reward_val)

            step_data = {
                "step": episode_steps[env_idx],
                "reward": reward_val,
            }

            if extras_per_env_log is not None:
                for key, value in extras_per_env_log.items():
                    if torch.is_tensor(value):
                        step_data[key] = float(value[env_idx])
                    else:
                        step_data[key] = value
            elif extras_log is not None:
                for key, value in extras_log.items():
                    if torch.is_tensor(value):
                        step_data[key] = float(value)
                    else:
                        step_data[key] = value

            current_episode_metrics[env_idx]["steps"].append(step_data)
            current_episode_metrics[env_idx]["total_reward"] += reward_val

            if (
                NUM_VIDEO_EPISODES > 0
                and episode_steps[env_idx] == 0
                and not record_active[env_idx]
            ):
                available_slots = (
                    NUM_VIDEO_EPISODES - recorded_episodes - active_record_count
                )
                if available_slots > 0:
                    if not hasattr(env.unwrapped, "scene"):
                        raise RuntimeError(
                            "Scene sensors are unavailable for video recording."
                        )

                    gripper_cam = env.unwrapped.scene.sensors.get("gripper_camera")
                    overhead_cam = env.unwrapped.scene.sensors.get("overhead_camera")
                    if (
                        gripper_cam is None
                        or "rgb" not in gripper_cam.data.output
                        or overhead_cam is None
                        or "rgb" not in overhead_cam.data.output
                    ):
                        raise RuntimeError(
                            "Required cameras are unavailable. Ensure SO101_ENABLE_OVERHEAD_CAMERA is set for evaluation."
                        )

                    gripper_rgb = gripper_cam.data.output["rgb"][env_idx].cpu().numpy()
                    overhead_rgb = (
                        overhead_cam.data.output["rgb"][env_idx].cpu().numpy()
                    )

                    wrist_h, wrist_w = gripper_rgb.shape[:2]
                    overhead_h, overhead_w = overhead_rgb.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    wrist_video_path = eval_dir / (
                        f"wrist_cam_env_{env_idx:03d}_ep_{episode_counts[env_idx]:03d}.mp4"
                    )
                    overhead_video_path = eval_dir / (
                        f"overhead_cam_env_{env_idx:03d}_ep_{episode_counts[env_idx]:03d}.mp4"
                    )

                    wrist_writers[env_idx] = cv2.VideoWriter(
                        str(wrist_video_path), fourcc, 30, (wrist_w, wrist_h)
                    )
                    overhead_writers[env_idx] = cv2.VideoWriter(
                        str(overhead_video_path), fourcc, 30, (overhead_w, overhead_h)
                    )

                    record_active[env_idx] = True
                    active_record_count += 1

            if record_active[env_idx] and hasattr(env.unwrapped, "scene"):
                gripper_cam = env.unwrapped.scene.sensors.get("gripper_camera")
                overhead_cam = env.unwrapped.scene.sensors.get("overhead_camera")
                if (
                    gripper_cam is not None
                    and overhead_cam is not None
                    and "rgb" in gripper_cam.data.output
                    and "rgb" in overhead_cam.data.output
                ):
                    gripper_frame = (
                        gripper_cam.data.output["rgb"][env_idx].cpu().numpy()
                    )
                    overhead_frame = (
                        overhead_cam.data.output["rgb"][env_idx].cpu().numpy()
                    )

                    wrist_writer = wrist_writers.get(env_idx)
                    overhead_writer = overhead_writers.get(env_idx)
                    if wrist_writer is not None:
                        wrist_writer.write(
                            cv2.cvtColor(gripper_frame, cv2.COLOR_RGB2BGR)
                        )
                    if overhead_writer is not None:
                        overhead_writer.write(
                            cv2.cvtColor(overhead_frame, cv2.COLOR_RGB2BGR)
                        )

            episode_steps[env_idx] += 1

            done_flag = done[env_idx]
            if torch.is_tensor(done_flag):
                done_flag = bool(done_flag.item())
            else:
                done_flag = bool(done_flag)

            if done_flag:
                current_episode_metrics[env_idx]["episode_length"] = episode_steps[
                    env_idx
                ]
                episode_entry = current_episode_metrics[env_idx].copy()

                step_values: dict[str, list[float]] = {}
                reward_values: dict[str, list[float]] = {}
                for step_item in episode_entry["steps"]:
                    for key, value in step_item.items():
                        if key in ("step", "reward"):
                            continue
                        if not isinstance(value, (int, float)):
                            continue
                        if key.startswith("Step_Metrics/"):
                            step_values.setdefault(key, []).append(float(value))
                        if key.startswith("Episode_Reward/"):
                            reward_values.setdefault(key, []).append(float(value))

                episode_entry["metrics"] = {
                    key: _summary_for_values(values)
                    for key, values in step_values.items()
                }
                episode_entry["rewards"] = {
                    key: _summary_for_values(values)
                    for key, values in reward_values.items()
                }

                for key, values in step_values.items():
                    global_step_values.setdefault(key, []).extend(values)
                for key, values in reward_values.items():
                    global_reward_values.setdefault(key, []).extend(values)

                for key, ep_summary in episode_entry["metrics"].items():
                    global_step_summaries.setdefault(key, []).append(ep_summary)
                for key, ep_summary in episode_entry["rewards"].items():
                    global_reward_summaries.setdefault(key, []).append(ep_summary)

                if args_cli.verbosity == "basic":
                    episode_entry.pop("steps", None)

                all_episode_data.append(episode_entry)
                completed_episodes += 1
                progress.update(1)

                if record_active[env_idx]:
                    wrist_writer = wrist_writers.pop(env_idx, None)
                    overhead_writer = overhead_writers.pop(env_idx, None)
                    if wrist_writer is not None:
                        wrist_writer.release()
                    if overhead_writer is not None:
                        overhead_writer.release()
                    record_active[env_idx] = False
                    recorded_episodes += 1
                    print(
                        f"[INFO] Saved camera videos for env {env_idx} episode {episode_counts[env_idx]}"
                    )

                episode_counts[env_idx] += 1
                episode_steps[env_idx] = 0
                current_episode_metrics[env_idx] = {
                    "env_id": env_idx,
                    "episode_num": episode_counts[env_idx],
                    "steps": [],
                    "total_reward": 0.0,
                    "episode_length": 0,
                }

                if completed_episodes % 10 == 0:
                    print(
                        f"[INFO] Completed {completed_episodes}/{NUM_EPISODES} episodes"
                    )

                if completed_episodes >= NUM_EPISODES:
                    break

    progress.close()
    print(f"[INFO] Evaluation complete: {completed_episodes} episodes")

    # Compute summary statistics
    episode_rewards = [ep["total_reward"] for ep in all_episode_data]
    episode_lengths = [ep["episode_length"] for ep in all_episode_data]

    summary = {
        "num_envs": num_envs,
        "num_episodes": len(all_episode_data),
        "requested_num_episodes": NUM_EPISODES,
        "verbosity": args_cli.verbosity,
        "checkpoint_path": str(checkpoint_path),
        "task_name": task_name,
        "seed": args_cli.seed,
        "metrics_summary": {
            key: _global_summary_for_key(values, global_step_summaries.get(key, []))
            for key, values in global_step_values.items()
        },
        "rewards_summary": {
            key: _global_summary_for_key(values, global_reward_summaries.get(key, []))
            for key, values in global_reward_values.items()
        },
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
    print(
        f"  Mean reward: {summary['summary_statistics']['mean_reward']:.3f} ± {summary['summary_statistics']['std_reward']:.3f}"
    )
    print(
        f"  Mean episode length: {summary['summary_statistics']['mean_episode_length']:.1f} ± {summary['summary_statistics']['std_episode_length']:.1f}"
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
