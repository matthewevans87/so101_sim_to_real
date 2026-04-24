# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to evaluate a trained RL agent from skrl and collect comprehensive metrics.

This script runs evaluation episodes with one or more environments (defaults to 1), recording:
- Step metrics and rewards for all episodes
- Videos from wrist and/or overhead cameras (opt-in via --record-wrist-cam / --record-overhead-cam)
- Results saved to JSON file

By default no videos are recorded and the overhead camera is not instantiated, keeping VRAM
usage comparable to training.

Opt-in recording flags:
  --record-wrist-cam      Per-env wrist camera (one .mp4 per episode per env)
  --record-overhead-cam   Per-env overhead camera (adds ~1x VRAM cost)
  --record-viewport-cam   Isaac Sim full viewport (all envs tiled, single .mp4)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import math
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
    help="Number of episodes to record videos for (only used when --record-wrist-cam or --record-overhead-cam is set).",
)
parser.add_argument(
    "--record-wrist-cam",
    action="store_true",
    default=False,
    help="Record wrist-camera video for the first --num-videos episodes.",
)
parser.add_argument(
    "--record-overhead-cam",
    action="store_true",
    default=False,
    help=(
        "Record overhead-camera video for the first --num-videos episodes. "
        "Enabling this instantiates an extra camera for every environment, "
        "roughly doubling eval VRAM compared to training."
    ),
)
parser.add_argument(
    "--record-viewport-cam",
    action="store_true",
    default=False,
    help=(
        "Record the Isaac Sim full viewport (all envs tiled) as a single .mp4. "
        "Activates render_mode='rgb_array', which adds rendering overhead. "
        "Saved to eval_dir/viewport_video/."
    ),
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
parser.add_argument(
    "--eval-subdir",
    type=str,
    default="evaluation",
    help=(
        "Subdirectory under the experiment directory where eval outputs "
        "(results.json, videos) are written.  Override (e.g. 'evaluation_v2') "
        "to re-evaluate without overwriting a prior eval."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# The wrist camera is always required for policy observations.
args_cli.enable_cameras = True
# The overhead camera is an extra viewport (~doubles VRAM vs training).
# Only instantiate it when overhead video recording is explicitly requested.
os.environ["SO101_ENABLE_OVERHEAD_CAMERA"] = (
    "1" if (args_cli.num_videos > 0 and args_cli.record_overhead_cam) else "0"
)

# ── Resolve env-config contract from the run manifest BEFORE any so101_rl import ─
# Reading the manifest here lets us set SO101_ENV_CONFIG (which is consumed at
# module-import time by so101_lift_cube_env_cfg) and verify on-disk integrity
# of every artifact in the experiment dir before any heavy initialisation.
_experiment_path = Path(args_cli.experiment_path).resolve()
if not _experiment_path.is_dir():
    raise FileNotFoundError(f"--experiment-path is not a directory: {_experiment_path}")

# Local import so we can use the manifest before AppLauncher fires.
# so101_rl is installed via `isaaclab.sh -p -m pip install -e ...` so this is
# available even though we have not yet imported the heavy Isaac Lab modules.
from so101_rl.run_manifest import RunManifest, MANIFEST_FILENAME  # noqa: E402

_manifest_path = _experiment_path / MANIFEST_FILENAME
if _manifest_path.is_file():
    _manifest = RunManifest.load(_experiment_path)
    _manifest.verify_against_disk(_experiment_path)
    os.environ["SO101_ENV_CONFIG"] = str(_manifest.env_config_abs(_experiment_path))
    print(f"[INFO] Loaded run manifest: {_manifest_path}")
    print(
        f"[INFO] Manifest verified against disk (env_config + cnn_checkpoint hashes match)."
    )
else:
    # Backward-compat fallback for experiments produced before the manifest
    # contract.  Auto-detects env_config.yaml in the experiment dir; raises
    # FileNotFoundError if absent.  This branch will be removed once the
    # active sweep dirs have been re-trained.
    _legacy_env_cfg = _experiment_path / "env_config.yaml"
    if not _legacy_env_cfg.is_file():
        raise FileNotFoundError(
            f"Neither {_manifest_path} nor legacy env_config.yaml was found in "
            f"{_experiment_path}.  This experiment dir is unusable for evaluation."
        )
    _manifest = None
    os.environ["SO101_ENV_CONFIG"] = str(_legacy_env_cfg)
    print(
        f"[WARNING] No run_manifest.json found at {_manifest_path}; falling back "
        f"to legacy env_config.yaml auto-detect at {_legacy_env_cfg}."
    )

# Eval seed: always equals the training seed from the manifest for reproducibility.
# Legacy experiments (no manifest) fall back to seed=42 with a warning.
_eval_seed: int = _manifest.seed if _manifest is not None else 42
if _manifest is None:
    print(
        "[WARNING] No manifest — using fallback seed=42. Eval reproducibility not guaranteed."
    )

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
    checkpoint_path = (
        experiment_path / "skrl" / "agent" / "checkpoints" / "best_agent.pt"
    )
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

    # The experiment path was already resolved at module load time and the
    # manifest was loaded into the module-level _manifest.  We re-bind here
    # for clarity.
    experiment_path = _experiment_path
    print(f"[INFO] Loading experiment from: {experiment_path}")

    # ── Per-experiment agent_config.yaml override ──────────────────────────
    # The hydra entry point loads the *base* skrl_ppo_cfg.yaml, but the
    # checkpoint was trained against the per-experiment frozen
    # agent_config.yaml in the experiment dir (with sweep agent_overrides
    # applied — e.g. models.separate=false produces a shared-trunk model
    # whose state_dict layout differs from the separate-trunk default).
    # We must replay those overrides here before the model is built so that
    # the loaded state_dict matches the trained checkpoint.
    _frozen_agent_cfg = experiment_path / "agent_config.yaml"
    if _frozen_agent_cfg.is_file():
        import copy as _copy
        import yaml as _yaml

        def _deep_merge(base: dict, overrides: dict) -> dict:
            result = _copy.deepcopy(base)
            for k, v in overrides.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = _copy.deepcopy(v)
            return result

        with open(_frozen_agent_cfg) as _f:
            _frozen = _yaml.safe_load(_f) or {}
        experiment_cfg = _deep_merge(experiment_cfg, _frozen)
        print(f"[INFO] Merged frozen agent_config.yaml: {_frozen_agent_cfg}")
    else:
        print(
            f"[INFO] No frozen agent_config.yaml at {_frozen_agent_cfg}; "
            f"using base agent cfg from hydra entry point."
        )

    # Seed all RNGs to match training conditions for deterministic DR sampling.
    random.seed(_eval_seed)
    np.random.seed(_eval_seed)
    torch.manual_seed(_eval_seed)
    torch.cuda.manual_seed_all(_eval_seed)
    print(f"[INFO] Seeded RNG (training seed={_eval_seed})")

    # ── Resolve checkpoint and CNN-checkpoint via the manifest ─────────────
    # When a manifest is present (the modern path), it is the single source of
    # truth.  When absent (legacy experiment dirs), we fall back to filename
    # auto-detect for one release cycle, and the existing helper functions
    # below preserve the old contract.
    if _manifest is not None:
        checkpoint_path = _manifest.final_checkpoint_abs(experiment_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Manifest names final checkpoint at {checkpoint_path} but the file "
                f"is not present.  Manifest may be stale."
            )
        print(
            f"[INFO] Using manifest-recorded final checkpoint: {checkpoint_path} "
            f"(step {_manifest.final_checkpoint_step})"
        )
    else:
        checkpoint_path, _ = find_checkpoint_and_task(experiment_path)
        print(f"[INFO] Found checkpoint: {checkpoint_path}")

    # Ensure task name matches if provided (kept for legacy callers).
    task_name = args_cli.task.split(":")[-1] if args_cli.task else ""

    # SO101_ENV_CONFIG was already set at module load time (manifest path or
    # legacy fallback).  Confirm for the user.
    print(f"[INFO] SO101_ENV_CONFIG = {os.environ.get('SO101_ENV_CONFIG')}")

    # ── Wire CNN checkpoint into vision_encoder via the manifest ──────────
    vision_encoder = getattr(env_cfg, "vision_encoder", None)
    _is_frozen_cnn = (
        vision_encoder is not None
        and getattr(vision_encoder, "type", None) == "frozen_cnn"
    )
    if _manifest is not None:
        cnn_ckpt = _manifest.cnn_checkpoint_abs(experiment_path)
        if cnn_ckpt is not None:
            if not _is_frozen_cnn:
                raise RuntimeError(
                    f"Manifest declares cnn_checkpoint at {cnn_ckpt} but env_cfg."
                    f"vision_encoder.type is {getattr(vision_encoder, 'type', None)!r}, "
                    f"not 'frozen_cnn'.  Cannot safely wire the checkpoint."
                )
            env_cfg.vision_encoder.cnn_checkpoint = str(cnn_ckpt)
            print(f"[INFO] Wired CNN checkpoint from manifest: {cnn_ckpt}")
        else:
            if _is_frozen_cnn:
                raise RuntimeError(
                    "env_cfg.vision_encoder.type == 'frozen_cnn' but the manifest "
                    "records no cnn_checkpoint.  Evaluation would run with random "
                    "CNN weights — aborting."
                )
            print(
                "[INFO] Manifest declares no CNN checkpoint (not a frozen_cnn experiment)."
            )
    else:
        # Legacy fallback: original auto-detect behaviour.
        embedded_cnn = experiment_path / "cnn_checkpoint.pt"
        if embedded_cnn.is_file():
            if not _is_frozen_cnn:
                raise RuntimeError(
                    f"Found cnn_checkpoint.pt at {embedded_cnn} but env_cfg.vision_encoder.type "
                    f"is not 'frozen_cnn' (got {getattr(vision_encoder, 'type', None)!r}). "
                    f"Cannot safely wire the checkpoint — aborting."
                )
            env_cfg.vision_encoder.cnn_checkpoint = str(embedded_cnn)
            print(f"[INFO] Auto-detected embedded CNN checkpoint: {embedded_cnn}")
        else:
            if _is_frozen_cnn:
                raise FileNotFoundError(
                    f"env_cfg.vision_encoder.type == 'frozen_cnn' but no cnn_checkpoint.pt found "
                    f"in experiment dir {experiment_path}. Evaluation would run with random CNN "
                    f"weights — aborting."
                )
            print(
                "[INFO] No embedded cnn_checkpoint.pt found in experiment dir (not a frozen_cnn experiment)."
            )

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

    # Set seed — always use the training seed from the manifest.
    experiment_cfg["seed"] = _eval_seed
    env_cfg.seed = _eval_seed

    # Create evaluation output directory.  The subdir is configurable so a
    # re-evaluation pass (e.g. with an updated eval pipeline) can be written
    # alongside the original results without clobbering them.
    _eval_subdir = args_cli.eval_subdir
    if not _eval_subdir or "/" in _eval_subdir or _eval_subdir.startswith("."):
        raise ValueError(
            f"--eval-subdir must be a single non-empty directory name (got {_eval_subdir!r})"
        )
    eval_dir = experiment_path / _eval_subdir
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Evaluation results will be saved to: {eval_dir}")

    # create isaac environment
    # render_mode="rgb_array" activates the Isaac Sim viewport renderer — only enable when
    # recording the viewport, as it adds measurable GPU overhead on every step.
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.record_viewport_cam else None,
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Wrap with RecordVideo to capture the full viewport (all envs tiled).
    # Must be applied before SkrlVecEnvWrapper so the gym interface is intact.
    if args_cli.record_viewport_cam:
        viewport_video_dir = eval_dir / "viewport_video"
        viewport_video_dir.mkdir(parents=True, exist_ok=True)
        # Estimate total steps to record (~num_videos episodes worth).
        try:
            _steps_per_ep = int(
                env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation)
            )
        except AttributeError:
            _steps_per_ep = 600  # conservative fallback
        _viewport_video_length = args_cli.num_videos * _steps_per_ep
        viewport_video_kwargs = {
            "video_folder": str(viewport_video_dir),
            "step_trigger": lambda step: step == 0,
            "video_length": _viewport_video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording viewport video to: {viewport_video_dir}")
        env = gym.wrappers.RecordVideo(env, **viewport_video_kwargs)

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

    # Grab the EpisodeStatsPipeline if available so we can drain per-episode
    # lift/drop/success/bump/time_to_lift stats during the eval loop.
    _episode_stats_pipeline = getattr(env.unwrapped, "episode_stats_pipeline", None)
    if _episode_stats_pipeline is not None:
        print("[INFO] EpisodeStatsPipeline found — will collect episode stats")
    else:
        raise RuntimeError(
            "EpisodeStatsPipeline not found on env.unwrapped — eval requires "
            "per-episode stats to enforce 1 episode = 1 data point invariants."
        )

    # Termination pipeline: required for primary-cause classification.
    # success_condition_log_name is the canonical id of the (single) is_success
    # termination condition; replaces the previous "success" string-match.
    _termination_pipeline = getattr(env.unwrapped, "termination_pipeline", None)
    if _termination_pipeline is None:
        raise RuntimeError(
            "termination_pipeline not found on env.unwrapped — eval requires "
            "the Phase A termination pipeline to classify primary termination "
            "causes without name matching."
        )
    _success_log_name: str = _termination_pipeline.success_condition_log_name
    if not _success_log_name:
        raise RuntimeError(
            "termination_pipeline.success_condition_log_name is empty; eval "
            "cannot classify primary termination cause.  Ensure exactly one "
            "termination condition declares is_success=true."
        )
    print(f"[INFO] Success termination condition: {_success_log_name!r}")

    # Evaluation parameters
    NUM_EPISODES = args_cli.num_episodes
    NUM_VIDEO_EPISODES = args_cli.num_videos
    num_envs = env_cfg.scene.num_envs
    episodes_per_env = math.ceil(NUM_EPISODES / num_envs)
    actual_episodes = episodes_per_env * num_envs
    # In "basic" mode per-step trace is not written to JSON.  We skip the per-step
    # dict allocation and the episode-end re-scan by maintaining per-key value lists
    # directly.  In "full" mode we additionally build the step trace as before.
    _collect_steps = args_cli.verbosity == "full"

    # Storage for results
    all_episode_data = []
    all_episode_stats: list[dict] = []  # drained from EpisodeStatsPipeline each step
    completed_episodes = 0
    recorded_episodes = 0
    env_episodes_done = [0] * num_envs
    # Raw per-flag termination counts (may sum > actual_episodes when multiple
    # reasons fire simultaneously, e.g. success on the final timeout step).
    termination_flag_counts: dict[str, int] = {}
    # Primary-cause counts: each episode contributes exactly 1 entry.
    # Precedence: success > other terminal > time_out.
    termination_primary_counts: dict[str, int] = {}

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

    # Per-env, per-key value lists accumulated directly during the loop.
    # Avoids per-step dict allocation + the O(steps×keys) re-scan at episode end.
    # step_lists[env_idx] = {"Step_Metrics/foo": [v0, v1, ...]}
    # reward_lists[env_idx] = {"Episode_Reward/foo": [v0, v1, ...]}
    step_lists: list[dict[str, list[float]]] = [{} for _ in range(num_envs)]
    reward_lists: list[dict[str, list[float]]] = [{} for _ in range(num_envs)]

    # reset environment
    obs, info = env.reset()

    print(
        f"[INFO] Starting evaluation: {actual_episodes} episodes across {num_envs} envs "
        f"({episodes_per_env} per env, requested={NUM_EPISODES})"
    )
    _active_cams = [
        c
        for c, flag in [
            ("wrist", args_cli.record_wrist_cam),
            ("overhead", args_cli.record_overhead_cam),
            ("viewport", args_cli.record_viewport_cam),
        ]
        if flag
    ]
    if _active_cams and NUM_VIDEO_EPISODES > 0:
        print(
            f"[INFO] Recording {'/'.join(_active_cams)} camera(s) for first {NUM_VIDEO_EPISODES} episodes"
        )
    else:
        print(
            "[INFO] Video recording disabled "
            "(pass --record-wrist-cam / --record-overhead-cam / --record-viewport-cam to enable)"
        )

    # Run evaluation
    progress = tqdm(total=actual_episodes, desc="Evaluating", unit="episode")
    while simulation_app.is_running() and not all(
        c >= episodes_per_env for c in env_episodes_done
    ):
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

        # ── Batch GPU→CPU ─────────────────────────────────────────────────────
        # A single .cpu().numpy() on the full (num_envs,) tensor is orders of
        # magnitude faster than num_envs individual .item() calls because it
        # issues one DMA transfer instead of serialising one sync per env.
        reward_np: np.ndarray = (
            reward.cpu().numpy()
            if torch.is_tensor(reward)
            else np.asarray(reward, dtype=np.float32)
        )
        done_np: np.ndarray = (
            done.cpu().numpy().astype(bool)
            if torch.is_tensor(done)
            else np.asarray(done, dtype=bool)
        )
        # Convert extras tensors to numpy once; per_env values are (num_envs,) arrays,
        # non-per-env (log) values become plain Python floats.
        extras_np: dict[str, Any] = {}
        if extras_per_env_log is not None:
            for _k, _v in extras_per_env_log.items():
                extras_np[_k] = _v.cpu().numpy() if torch.is_tensor(_v) else _v
        elif extras_log is not None:
            for _k, _v in extras_log.items():
                extras_np[_k] = float(_v.item()) if torch.is_tensor(_v) else _v
        # ─────────────────────────────────────────────────────────────────────

        _record_any = NUM_VIDEO_EPISODES > 0 and (
            args_cli.record_wrist_cam or args_cli.record_overhead_cam
        )
        active_record_count = sum(record_active) if _record_any else 0

        for env_idx in range(num_envs):
            reward_val = float(reward_np[env_idx])

            # Accumulate per-key metric/reward lists directly (no per-step dict).
            _is_per_env = extras_per_env_log is not None
            for key, arr in extras_np.items():
                val = float(arr[env_idx]) if _is_per_env else float(arr)
                if key.startswith("Step_Metrics/"):
                    step_lists[env_idx].setdefault(key, []).append(val)
                elif key.startswith("Episode_Reward/"):
                    reward_lists[env_idx].setdefault(key, []).append(val)

            # In "full" verbosity, additionally build the per-step JSON trace.
            if _collect_steps:
                step_data: dict = {"step": episode_steps[env_idx], "reward": reward_val}
                for key, arr in extras_np.items():
                    step_data[key] = float(arr[env_idx]) if _is_per_env else float(arr)
                current_episode_metrics[env_idx]["steps"].append(step_data)

            current_episode_metrics[env_idx]["total_reward"] += reward_val

            if (
                NUM_VIDEO_EPISODES > 0
                and (args_cli.record_wrist_cam or args_cli.record_overhead_cam)
                and episode_steps[env_idx] == 0
                and not record_active[env_idx]
                and env_episodes_done[env_idx] < episodes_per_env
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

                    if args_cli.record_wrist_cam:
                        if gripper_cam is None or "rgb" not in gripper_cam.data.output:
                            raise RuntimeError(
                                "Wrist (gripper) camera is unavailable for video recording."
                            )

                    if args_cli.record_overhead_cam:
                        if (
                            overhead_cam is None
                            or "rgb" not in overhead_cam.data.output
                        ):
                            raise RuntimeError(
                                "Overhead camera is unavailable for video recording. "
                                "Ensure SO101_ENABLE_OVERHEAD_CAMERA=1 is set before AppLauncher is created."
                            )

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                    if args_cli.record_wrist_cam:
                        gripper_rgb = (
                            gripper_cam.data.output["rgb"][env_idx].cpu().numpy()
                        )
                        wrist_h, wrist_w = gripper_rgb.shape[:2]
                        wrist_video_path = eval_dir / (
                            f"wrist_cam_env_{env_idx:03d}_ep_{episode_counts[env_idx]:03d}.mp4"
                        )
                        wrist_writers[env_idx] = cv2.VideoWriter(
                            str(wrist_video_path), fourcc, 30, (wrist_w, wrist_h)
                        )

                    if args_cli.record_overhead_cam:
                        overhead_rgb = (
                            overhead_cam.data.output["rgb"][env_idx].cpu().numpy()
                        )
                        overhead_h, overhead_w = overhead_rgb.shape[:2]
                        overhead_video_path = eval_dir / (
                            f"overhead_cam_env_{env_idx:03d}_ep_{episode_counts[env_idx]:03d}.mp4"
                        )
                        overhead_writers[env_idx] = cv2.VideoWriter(
                            str(overhead_video_path),
                            fourcc,
                            30,
                            (overhead_w, overhead_h),
                        )

                    record_active[env_idx] = True
                    active_record_count += 1

            if record_active[env_idx] and hasattr(env.unwrapped, "scene"):
                if args_cli.record_wrist_cam:
                    gripper_cam = env.unwrapped.scene.sensors.get("gripper_camera")
                    wrist_writer = wrist_writers.get(env_idx)
                    if (
                        wrist_writer is not None
                        and gripper_cam is not None
                        and "rgb" in gripper_cam.data.output
                    ):
                        gripper_frame = (
                            gripper_cam.data.output["rgb"][env_idx].cpu().numpy()
                        )
                        wrist_writer.write(
                            cv2.cvtColor(gripper_frame, cv2.COLOR_RGB2BGR)
                        )

                if args_cli.record_overhead_cam:
                    overhead_cam = env.unwrapped.scene.sensors.get("overhead_camera")
                    overhead_writer = overhead_writers.get(env_idx)
                    if (
                        overhead_writer is not None
                        and overhead_cam is not None
                        and "rgb" in overhead_cam.data.output
                    ):
                        overhead_frame = (
                            overhead_cam.data.output["rgb"][env_idx].cpu().numpy()
                        )
                        overhead_writer.write(
                            cv2.cvtColor(overhead_frame, cv2.COLOR_RGB2BGR)
                        )

            episode_steps[env_idx] += 1

            done_flag = bool(done_np[env_idx])

            if done_flag:
                _within_budget = env_episodes_done[env_idx] < episodes_per_env

                if _within_budget:
                    # Collect termination flags and determine primary cause.
                    _ep_term_flags: list[str] = []
                    for key, arr in extras_np.items():
                        if key.startswith("Termination/"):
                            val = float(arr[env_idx]) if _is_per_env else float(arr)
                            if val > 0.5:
                                cause = key[len("Termination/") :]
                                _ep_term_flags.append(cause)
                                termination_flag_counts[cause] = (
                                    termination_flag_counts.get(cause, 0) + 1
                                )
                    # Primary cause: declared-success > other terminal > time_out.
                    # The success cause is identified by its TerminationCfg.id
                    # (TerminationCondition.log_name), not by name matching.
                    if _ep_term_flags:
                        if _success_log_name in _ep_term_flags:
                            _primary = _success_log_name
                        else:
                            _non_timeout = [
                                c for c in _ep_term_flags if c != "time_out"
                            ]
                            _primary = _non_timeout[0] if _non_timeout else "time_out"
                        termination_primary_counts[_primary] = (
                            termination_primary_counts.get(_primary, 0) + 1
                        )

                    current_episode_metrics[env_idx]["episode_length"] = episode_steps[
                        env_idx
                    ]
                    episode_entry = current_episode_metrics[env_idx].copy()

                    # Use the directly-accumulated per-key lists — no re-scan of step dicts.
                    sv = step_lists[env_idx]
                    rv = reward_lists[env_idx]

                    episode_entry["metrics"] = {
                        key: _summary_for_values(values) for key, values in sv.items()
                    }
                    episode_entry["rewards"] = {
                        key: _summary_for_values(values) for key, values in rv.items()
                    }

                    for key, values in sv.items():
                        global_step_values.setdefault(key, []).extend(values)
                    for key, values in rv.items():
                        global_reward_values.setdefault(key, []).extend(values)

                    for key, ep_summary in episode_entry["metrics"].items():
                        global_step_summaries.setdefault(key, []).append(ep_summary)
                    for key, ep_summary in episode_entry["rewards"].items():
                        global_reward_summaries.setdefault(key, []).append(ep_summary)

                    if args_cli.verbosity == "basic":
                        episode_entry.pop("steps", None)

                    all_episode_data.append(episode_entry)
                    env_episodes_done[env_idx] += 1
                    completed_episodes = sum(env_episodes_done)
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

                    if completed_episodes % 10 == 0:
                        print(
                            f"[INFO] Completed {completed_episodes}/{actual_episodes} episodes"
                        )

                # Always reset per-env state so the env continues running cleanly.
                episode_counts[env_idx] += 1
                episode_steps[env_idx] = 0
                # Reset per-key accumulators for the next episode.
                step_lists[env_idx] = {}
                reward_lists[env_idx] = {}
                current_episode_metrics[env_idx] = {
                    "env_id": env_idx,
                    "episode_num": episode_counts[env_idx],
                    "steps": [],
                    "total_reward": 0.0,
                    "episode_length": 0,
                }

        # Drain EpisodeStatsPipeline after every env step (outside env loop).
        # get_completed_episodes() clears its buffer, so calling once per step
        # is both correct and cheap.
        if _episode_stats_pipeline is not None:
            all_episode_stats.extend(_episode_stats_pipeline.get_completed_episodes())

    progress.close()
    print(f"[INFO] Evaluation complete: {completed_episodes} episodes")

    # ── Single source of truth: per-episode unified records ────────────────
    # Both ``all_episode_data`` (filled at done from per-env reward
    # accumulators) and ``all_episode_stats`` (drained from
    # EpisodeStatsPipeline) are appended in global completion order.  With
    # the per-env episode budget, the pipeline drain may include a small
    # number of extra episodes from envs that continued running after
    # other envs hit their budget.  We bucket pipeline stats by ``env_idx``,
    # truncate each bucket to the budget, and join positionally with the
    # corresponding ``all_episode_data`` entries (which are budget-bounded
    # at append time).
    stats_by_env: dict[int, list[dict]] = {i: [] for i in range(num_envs)}
    for s in all_episode_stats:
        stats_by_env[int(s["env_idx"])].append(s)

    for i in range(num_envs):
        if len(stats_by_env[i]) < episodes_per_env:
            raise RuntimeError(
                f"EpisodeStatsPipeline drained only {len(stats_by_env[i])} "
                f"episodes for env {i}, but the budget required "
                f"{episodes_per_env}.  Pipeline drain order is broken or the "
                f"eval loop exited early."
            )
        # Drop overflow past the budget.
        stats_by_env[i] = stats_by_env[i][:episodes_per_env]

    # Hard invariants — 1 episode = 1 data point, no double-counting.
    assert len(all_episode_data) == sum(env_episodes_done) == actual_episodes, (
        f"Episode count mismatch: len(all_episode_data)={len(all_episode_data)}, "
        f"sum(env_episodes_done)={sum(env_episodes_done)}, "
        f"actual_episodes={actual_episodes}"
    )
    _stats_total = sum(len(v) for v in stats_by_env.values())
    assert _stats_total == actual_episodes, (
        f"EpisodeStatsPipeline post-truncation total ({_stats_total}) does "
        f"not match actual_episodes ({actual_episodes})."
    )
    _primary_total = sum(termination_primary_counts.values())
    assert _primary_total == actual_episodes, (
        f"Primary termination counts ({_primary_total}) do not sum to "
        f"actual_episodes ({actual_episodes}).  Some episodes were not "
        f"classified."
    )

    # Join per-episode reward record with per-episode pipeline stats.
    # ``ep["env_id"]`` and ``ep["episode_num"]`` come from
    # ``current_episode_metrics`` and increment monotonically per env, so
    # ``stats_by_env[env_id][episode_num]`` is the matching pipeline record.
    unified_episodes: list[dict] = []
    for ep in all_episode_data:
        env_id = int(ep["env_id"])
        ep_num = int(ep["episode_num"])
        stat = stats_by_env[env_id][ep_num]
        unified = dict(ep)
        # Pipeline-derived per-episode fields, namespaced to avoid colliding
        # with reward-loop field names.
        unified["lifted"] = bool(stat["lifted"])
        unified["dropped"] = bool(stat["dropped"])
        unified["success"] = bool(stat["success"])
        unified["timed_out"] = bool(stat["timed_out"])
        unified["cube_bump"] = float(stat["cube_bump"])
        unified["lift_step"] = stat["lift_step"]
        unified["pipeline_episode_steps"] = int(stat["episode_steps"])
        unified_episodes.append(unified)

    # Compute summary statistics from the unified records.
    episode_rewards = [ep["total_reward"] for ep in unified_episodes]
    episode_lengths = [ep["episode_length"] for ep in unified_episodes]

    # Aggregate episode stats over the unified records (single source).
    n_eps = len(unified_episodes)
    n_lifted = sum(1 for e in unified_episodes if e["lifted"])
    n_dropped = sum(1 for e in unified_episodes if e["dropped"])
    n_success = sum(1 for e in unified_episodes if e["success"])
    n_timed_out = sum(1 for e in unified_episodes if e["timed_out"])
    lift_steps = [
        e["lift_step"]
        for e in unified_episodes
        if e["lifted"] and e["lift_step"] is not None
    ]
    episode_stats_block: dict = {
        "n_episodes": n_eps,
        "n_lifted": n_lifted,
        "n_dropped": n_dropped,
        "n_success": n_success,
        "n_timed_out": n_timed_out,
        "lift_rate": n_lifted / n_eps,
        "drop_rate": n_dropped / n_eps,
        "success_rate": n_success / n_eps,
        "mean_cube_bump": float(np.mean([e["cube_bump"] for e in unified_episodes])),
        "mean_time_to_lift": float(np.mean(lift_steps)) if lift_steps else None,
    }

    summary = {
        "num_envs": num_envs,
        "requested_episodes": NUM_EPISODES,
        "episodes_per_env": episodes_per_env,
        "actual_episodes": actual_episodes,
        "num_episodes": len(unified_episodes),
        "verbosity": args_cli.verbosity,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": (
            _manifest.final_checkpoint_step if _manifest is not None else None
        ),
        "task_name": task_name,
        "training_seed": _eval_seed,
        "success_termination_id": _success_log_name,
        "termination_flag_counts": dict(sorted(termination_flag_counts.items())),
        "termination_primary_counts": dict(sorted(termination_primary_counts.items())),
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
        "episode_stats": episode_stats_block,
        "episodes": unified_episodes,
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
    pct = lambda r: f"{r * 100:.1f}%"
    print(f"[INFO] Episode stats ({episode_stats_block['n_episodes']} episodes):")
    print(f"  lift_rate:        {pct(episode_stats_block['lift_rate'])}")
    print(f"  drop_rate:        {pct(episode_stats_block['drop_rate'])}")
    print(f"  success_rate:     {pct(episode_stats_block['success_rate'])}")
    print(f"  mean_cube_bump:   {episode_stats_block['mean_cube_bump']:.4f}")
    ttl = episode_stats_block["mean_time_to_lift"]
    print(
        f"  mean_time_to_lift: {f'{ttl:.1f} steps' if ttl is not None else 'n/a (no lifts)'}"
    )
    if termination_flag_counts:
        print(
            f"[INFO] Termination causes ({completed_episodes} episodes, may sum > 100%):"
        )
        for cause, count in sorted(
            termination_flag_counts.items(), key=lambda kv: -kv[1]
        ):
            print(
                f"  {cause:<45s} {count:>6d}  ({count / completed_episodes * 100:.1f}%)"
            )
        if termination_primary_counts:
            print(
                f"[INFO] Primary termination causes ({completed_episodes} episodes, sums to 100%):"
            )
            for cause, count in sorted(
                termination_primary_counts.items(), key=lambda kv: -kv[1]
            ):
                print(
                    f"  {cause:<45s} {count:>6d}  ({count / completed_episodes * 100:.1f}%)"
                )
    else:
        print(
            "[WARNING] No termination causes recorded (Termination/* keys absent from extras)"
        )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
