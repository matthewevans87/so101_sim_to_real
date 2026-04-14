# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to roll out a trained skrl policy and collect telemetry samples.

This script runs a checkpoint in simulation and records telemetry every t steps:
- raw camera RGB
- active joint positions
- is_cube_in_grip_position step metric
- cube quaternion in grip-zone frame
- cube-in-camera-frame boolean from instance segmentation

Outputs are written as sharded NPZ files plus a metadata JSON manifest.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Roll out a trained skrl policy and collect telemetry samples."
)
parser.add_argument(
    "--experiment-path",
    type=str,
    required=True,
    help="Path to experiment directory (e.g., artifacts/2026-03-12_09-52-10)",
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Task name (e.g., So101-LiftCube-v0).",
)
parser.add_argument(
    "--sample-every-steps",
    type=int,
    required=True,
    help="Collect telemetry every t environment steps (must be > 0).",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    required=True,
    help="Stop once this many episodes are completed across all envs (must be > 0).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    required=True,
    help="Directory where telemetry shards and metadata are written.",
)
parser.add_argument(
    "--samples-per-shard",
    type=int,
    default=2048,
    help="Number of samples per NPZ shard (must be > 0).",
)
parser.add_argument(
    "--seed",
    type=int,
    required=True,
    help="Explicit RNG seed for reproducibility.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional explicit checkpoint path. If omitted, best_agent.pt is used from the experiment directory.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Override number of environments to simulate.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for the skrl agent.",
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
        "Name of the RL agent configuration entry point. Defaults to None, in which case "
        "--algorithm determines the default entry point."
    ),
)
parser.add_argument(
    "--cnn_checkpoint",
    type=str,
    default=None,
    help=(
        "Path to a pretrained CNN backbone checkpoint (.pt). "
        "If omitted, the script auto-detects cnn_checkpoint.pt inside the experiment directory. "
        "Required when vision_encoder.type == 'frozen_cnn' and no embedded checkpoint exists."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Cameras are required for telemetry collection.
args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import skrl
from packaging import version

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.math import quat_unique

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import so101_rl.tasks  # noqa: F401


SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner
else:
    raise ValueError(
        f"Unsupported ml framework: {args_cli.ml_framework}. "
        "Expected one of: torch, jax, jax-numpy"
    )


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


def _ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0. Received: {value}")


def _find_checkpoint_and_task(experiment_path: Path) -> tuple[Path, str]:
    checkpoint_path = (
        experiment_path / "skrl" / "agent" / "checkpoints" / "best_agent.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return checkpoint_path, ""


def _to_numpy(value: torch.Tensor | np.ndarray | float | bool | int, dtype=None):
    if torch.is_tensor(value):
        out = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        out = value
    else:
        out = np.asarray(value)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def _to_int(value) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _get_simulation_step_value(env_unwrapped: Any) -> int:
    """Return the simulator-level step counter as an int.

    This function is strict by design: telemetry collection fails fast if the
    environment does not expose ``common_step_counter``.
    """
    if not hasattr(env_unwrapped, "common_step_counter"):
        raise RuntimeError(
            "Environment is missing required 'common_step_counter'; cannot record simulation_step telemetry."
        )

    step_val = getattr(env_unwrapped, "common_step_counter")
    if torch.is_tensor(step_val):
        if step_val.numel() == 1:
            return int(step_val.item())
        return int(step_val.reshape(-1)[0].item())
    if isinstance(step_val, np.ndarray):
        if step_val.size == 1:
            return int(step_val.item())
        return int(step_val.reshape(-1)[0])

    parsed = _to_int(step_val)
    if parsed is None:
        raise RuntimeError(
            "common_step_counter has unsupported type/value "
            f"({type(step_val).__name__}: {step_val!r}); expected int-like value."
        )
    return parsed


def _get_tiled_segmentation_info(camera_info):
    """Extract the instance_segmentation_fast info dict from a TiledCamera info object.

    TiledCamera produces a single flat dict (not a per-env list) containing:
      {
        "instance_segmentation_fast": {
          "idToLabels":   {<int_id>: "<prim_path>", ...},
          "idToSemantics": {<int_id>: {"class": "<label>"}, ...}
        }
      }
    Integer IDs are shared across the full tiled batch; prim paths encode the env
    index (e.g. "/World/envs/env_0/Object") allowing per-env cube-ID filtering.
    """
    if not isinstance(camera_info, dict):
        raise RuntimeError(
            "TiledCamera info must be a dict. "
            f"Received type: {type(camera_info).__name__}. "
            "If Camera (not TiledCamera) is still in use the info will be a list — "
            "ensure the environment is using TiledCamera."
        )
    if "instance_segmentation_fast" not in camera_info:
        raise RuntimeError(
            "Camera info dict is missing required key 'instance_segmentation_fast'. "
            f"Keys present: {list(camera_info.keys())}."
        )
    inner = camera_info["instance_segmentation_fast"]
    if not isinstance(inner, dict):
        raise RuntimeError(
            "camera_info['instance_segmentation_fast'] must be a dict. "
            f"Received type: {type(inner).__name__}."
        )
    return inner


def _to_rgba_tuple(value) -> tuple[int, int, int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) == 4:
        rgba = []
        for item in value:
            parsed = _to_int(item)
            if parsed is None:
                return None
            rgba.append(parsed)
        return (rgba[0], rgba[1], rgba[2], rgba[3])
    return None


def _extract_id_label_pairs(node) -> list[tuple[int, str]]:
    pairs: list[tuple[int, str]] = []

    if isinstance(node, dict):
        for key, value in node.items():
            key_int = _to_int(key)
            if key_int is not None:
                if isinstance(value, str):
                    pairs.append((key_int, value))
                elif isinstance(value, dict):
                    label_bits = []
                    for candidate_key in (
                        "class",
                        "className",
                        "name",
                        "path",
                        "primPath",
                        "semanticLabel",
                        "semantic_label",
                        "label",
                    ):
                        if candidate_key in value and isinstance(
                            value[candidate_key], str
                        ):
                            label_bits.append(value[candidate_key])
                    if label_bits:
                        pairs.append((key_int, " ".join(label_bits)))
            pairs.extend(_extract_id_label_pairs(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            pairs.extend(_extract_id_label_pairs(item))

    return pairs


def _extract_rgba_label_pairs(node) -> list[tuple[tuple[int, int, int, int], str]]:
    pairs: list[tuple[tuple[int, int, int, int], str]] = []

    if isinstance(node, dict):
        for key, value in node.items():
            key_rgba = _to_rgba_tuple(key)
            val_rgba = _to_rgba_tuple(value)

            if key_rgba is not None and isinstance(value, str):
                pairs.append((key_rgba, value))
            if isinstance(key, str) and val_rgba is not None:
                pairs.append((val_rgba, key))

            pairs.extend(_extract_rgba_label_pairs(value))
    elif isinstance(node, (list, tuple)):
        for item in node:
            pairs.extend(_extract_rgba_label_pairs(item))

    return pairs


def _extract_cube_targets(seg_info, env_idx: int) -> tuple[str, set[Any]]:
    """Extract the set of segmentation IDs (or RGBA tuples) that correspond to the
    cube in the given env.

    For TiledCamera, *seg_info* is the ``idToLabels`` / ``idToSemantics`` dict shared
    across all envs.  Prim paths in that dict encode the env index
    (e.g. ``/World/envs/env_0/Object``), so we filter to IDs whose label contains the
    env-specific token ``env_{env_idx}/`` when extracting cube IDs.  This prevents
    cube IDs from neighbouring envs from contaminating the frame-occupancy check for
    this env.

    Falls back to the original tokenised search if the env-specific filter yields
    nothing (e.g. when running with a single env without an env-index path prefix).
    """
    cube_tokens = (
        "cube",
        "/object",
        " object",
        "object/",
        "object_",
    )
    env_token = f"env_{env_idx}/"

    id_pairs = _extract_id_label_pairs(seg_info)

    # First try: env-specific filter (TiledCamera prim-path schema).
    cube_ids_env = {
        inst_id
        for inst_id, label in id_pairs
        if env_token in label.lower()
        and any(token in label.lower() for token in cube_tokens)
    }
    if cube_ids_env:
        return "id", cube_ids_env

    # Second try: unfiltered ID search (single-env or non-TiledCamera layouts).
    cube_ids = {
        inst_id
        for inst_id, label in id_pairs
        if any(token in label.lower() for token in cube_tokens)
    }
    if cube_ids:
        return "id", cube_ids

    rgba_pairs = _extract_rgba_label_pairs(seg_info)
    cube_rgba = {
        rgba
        for rgba, label in rgba_pairs
        if any(token in label.lower() for token in cube_tokens)
    }
    if cube_rgba:
        return "rgba", cube_rgba

    discovered_labels = sorted(
        {label for _, label in id_pairs}.union({label for _, label in rgba_pairs})
    )
    raise RuntimeError(
        "Segmentation info did not contain any cube/object labels for "
        f"env {env_idx}. Discovered labels: {discovered_labels}."
    )


def _extract_segmentation_ids(segmentation_env) -> set[int]:
    seg = _to_numpy(segmentation_env)

    if seg.ndim == 2:
        pass
    elif seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    else:
        raise RuntimeError(
            "ID-based segmentation must be [H, W] or [H, W, 1]. "
            f"Received shape {seg.shape}."
        )

    if not np.issubdtype(seg.dtype, np.integer):
        raise RuntimeError(
            f"ID-based segmentation dtype must be integer; received {seg.dtype}."
        )

    ids = set(np.unique(seg).tolist())
    ids.discard(0)
    return {int(x) for x in ids}


def _extract_segmentation_rgba(segmentation_env) -> set[tuple[int, int, int, int]]:
    seg = _to_numpy(segmentation_env)
    if seg.ndim != 3 or seg.shape[-1] != 4:
        raise RuntimeError(
            "RGBA segmentation must be [H, W, 4]. " f"Received shape {seg.shape}."
        )

    if not np.issubdtype(seg.dtype, np.integer):
        raise RuntimeError(
            f"RGBA segmentation dtype must be integer; received {seg.dtype}."
        )

    flat = seg.reshape(-1, 4)
    unique_rows = np.unique(flat, axis=0)
    colors: set[tuple[int, int, int, int]] = set()
    for row in unique_rows:
        colors.add((int(row[0]), int(row[1]), int(row[2]), int(row[3])))
    colors.discard((0, 0, 0, 0))
    return colors


def _cube_in_frame(segmentation_env, cube_targets: tuple[str, set[Any]]) -> bool:
    target_mode, target_values = cube_targets
    if target_mode == "id":
        ids_in_frame = _extract_segmentation_ids(segmentation_env)
        return len(ids_in_frame.intersection(target_values)) > 0
    if target_mode == "rgba":
        rgba_in_frame = _extract_segmentation_rgba(segmentation_env)
        return len(rgba_in_frame.intersection(target_values)) > 0

    raise RuntimeError(f"Unsupported cube target mode: {target_mode!r}.")


def _flush_shard(
    shard_index: int,
    shards_dir: Path,
    buffer: dict[str, list],
    shard_manifest: list[dict],
) -> int:
    num_samples = len(buffer["env_id"])
    if num_samples == 0:
        return shard_index

    shard_path = shards_dir / f"telemetry_{shard_index:05d}.npz"
    np.savez_compressed(
        shard_path,
        rgb=np.stack(buffer["rgb"], axis=0).astype(np.uint8, copy=False),
        joint_pos=np.stack(buffer["joint_pos"], axis=0).astype(np.float32, copy=False),
        cube_pos_gz=np.stack(buffer["cube_pos_gz"], axis=0).astype(
            np.float32, copy=False
        ),
        gripper_cube_alignment=np.asarray(
            buffer["gripper_cube_alignment"], dtype=np.float32
        ),
        cube_rot6d_gz=np.stack(buffer["cube_rot6d_gz"], axis=0).astype(
            np.float32, copy=False
        ),
        cube_height_w=np.asarray(buffer["cube_height_w"], dtype=np.float32),
        cube_in_camera_frame=np.asarray(buffer["cube_in_camera_frame"], dtype=np.bool_),
        env_id=np.asarray(buffer["env_id"], dtype=np.int32),
        episode_id=np.asarray(buffer["episode_id"], dtype=np.int32),
        episode_step=np.asarray(buffer["episode_step"], dtype=np.int32),
        global_step=np.asarray(buffer["global_step"], dtype=np.int64),
        simulation_step=np.asarray(buffer["simulation_step"], dtype=np.int64),
        sim_time_s=np.asarray(buffer["sim_time_s"], dtype=np.float64),
        done=np.asarray(buffer["done"], dtype=np.bool_),
    )

    shard_manifest.append(
        {
            "shard": shard_path.name,
            "num_samples": num_samples,
            "global_step_min": int(np.min(buffer["global_step"])),
            "global_step_max": int(np.max(buffer["global_step"])),
        }
    )

    for key in buffer:
        buffer[key].clear()

    return shard_index + 1


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Collect telemetry while rolling out a trained skrl policy."""

    _ensure_positive("sample-every-steps", args_cli.sample_every_steps)
    _ensure_positive("num-episodes", args_cli.num_episodes)
    _ensure_positive("samples-per-shard", args_cli.samples_per_shard)

    experiment_path = Path(args_cli.experiment_path).resolve()
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path not found: {experiment_path}")

    if args_cli.checkpoint is not None:
        checkpoint_path = Path(args_cli.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        inferred_task_name = args_cli.task.split(":")[-1]
    else:
        checkpoint_path, inferred_task_name = _find_checkpoint_and_task(experiment_path)

    task_name = args_cli.task.split(":")[-1]
    if inferred_task_name and task_name != inferred_task_name:
        print(
            "[WARNING] Task name from checkpoint directory does not match --task. "
            f"Using --task={task_name}, checkpoint task directory was {inferred_task_name}."
        )

    env_config_path = experiment_path / "env_config.yaml"
    if env_config_path.exists():
        os.environ["SO101_ENV_CONFIG"] = str(env_config_path)

    # Resolve CNN checkpoint for frozen_cnn experiments.
    # Priority: explicit --cnn_checkpoint flag > embedded cnn_checkpoint.pt in experiment dir.
    cnn_checkpoint_path: Path | None = None
    if args_cli.cnn_checkpoint:
        cnn_checkpoint_path = Path(args_cli.cnn_checkpoint).resolve()
        if not cnn_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"--cnn_checkpoint not found: {cnn_checkpoint_path}"
            )
    else:
        embedded = experiment_path / "cnn_checkpoint.pt"
        if embedded.is_file():
            cnn_checkpoint_path = embedded
            print(
                f"[INFO] Auto-detected embedded CNN checkpoint: {cnn_checkpoint_path}"
            )

    if cnn_checkpoint_path is not None:
        vision_encoder = getattr(env_cfg, "vision_encoder", None)
        if vision_encoder is not None and vision_encoder.type == "frozen_cnn":
            env_cfg.vision_encoder.cnn_checkpoint = str(cnn_checkpoint_path)
            print(f"[INFO] CNN checkpoint wired into env_cfg: {cnn_checkpoint_path}")
        else:
            print(
                "[WARNING] --cnn_checkpoint supplied but vision_encoder.type != 'frozen_cnn'; "
                "checkpoint will be ignored."
            )

    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Ensure instance segmentation stream is explicitly enabled.
    camera_cfg = getattr(env_cfg, "camera_cfg", None)
    if camera_cfg is None:
        raise RuntimeError(
            "Environment config does not expose camera_cfg; this script requires a camera-backed task."
        )

    camera_data_types = list(getattr(camera_cfg, "data_types", []))
    if "instance_segmentation_fast" not in camera_data_types:
        camera_data_types.append("instance_segmentation_fast")
    camera_cfg.data_types = camera_data_types

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    output_dir = Path(args_cli.output_dir).resolve()

    collect_log_dir = output_dir
    collect_log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(collect_log_dir)

    shards_dir = collect_log_dir / "telemetry_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Experiment path: {experiment_path}")
    print(f"[INFO] Checkpoint path: {checkpoint_path}")
    print(f"[INFO] Telemetry output directory: {collect_log_dir}")
    print(f"[INFO] Sampling every {args_cli.sample_every_steps} steps")

    env: Any = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
    runner.agent.load(str(checkpoint_path))
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()

    if not hasattr(env.unwrapped, "scene"):
        raise RuntimeError("Scene sensors are unavailable. Cannot collect telemetry.")

    gripper_cam = env.unwrapped.scene.sensors.get("gripper_camera")
    if gripper_cam is None:
        raise RuntimeError("gripper_camera sensor not found in env scene sensors.")

    num_envs = env.unwrapped.num_envs
    episode_counts = [0 for _ in range(num_envs)]
    episode_steps = [0 for _ in range(num_envs)]

    completed_episodes = 0
    global_step = 0
    samples_collected = 0

    shard_manifest: list[dict] = []
    shard_index = 0

    buffer: dict[str, list] = {
        "rgb": [],
        "joint_pos": [],
        "cube_pos_gz": [],
        "gripper_cube_alignment": [],
        "cube_rot6d_gz": [],
        "cube_height_w": [],
        "cube_in_camera_frame": [],
        "env_id": [],
        "episode_id": [],
        "episode_step": [],
        "global_step": [],
        "simulation_step": [],
        "sim_time_s": [],
        "done": [],
    }

    cube_targets_by_env: dict[int, tuple[str, set[Any]]] = {}
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    started_at = datetime.now(timezone.utc)

    # Validation checks are invariant across loop iterations; run them once on the
    # first step so that sensor tensors are populated before we inspect them.
    _startup_validated = False

    while simulation_app.is_running() and completed_episodes < args_cli.num_episodes:
        with torch.inference_mode():
            outputs: Any = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {
                    agent: outputs[-1][agent].get("mean_actions", outputs[0][agent])
                    for agent in env.possible_agents
                }
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            obs, _, terminated, truncated, _ = env.step(actions)

        if torch.is_tensor(terminated):
            done = torch.logical_or(terminated, truncated)
        else:
            done = np.logical_or(terminated, truncated)

        global_step += 1
        simulation_step = _get_simulation_step_value(env.unwrapped)

        # One-time startup validation — runs after the first env.step() so that
        # camera tensors and step_metrics are guaranteed to be populated.
        if not _startup_validated:
            if "rgb" not in gripper_cam.data.output:
                raise RuntimeError("Camera output is missing 'rgb'.")
            if "instance_segmentation_fast" not in gripper_cam.data.output:
                raise RuntimeError(
                    "Camera output is missing 'instance_segmentation_fast'. Ensure "
                    "camera_cfg.data_types includes this channel."
                )
            if not hasattr(gripper_cam.data, "info"):
                raise RuntimeError(
                    "Camera output info is unavailable; cannot decode instance segmentation IDs."
                )
            # TiledCamera produces a flat dict, not a per-env list.
            if not isinstance(gripper_cam.data.info, dict):
                raise RuntimeError(
                    "Camera info must be a dict (TiledCamera). "
                    f"Received type: {type(gripper_cam.data.info).__name__}. "
                    "Ensure the environment is using TiledCamera, not Camera."
                )
            if not hasattr(env.unwrapped, "step_metrics"):
                raise RuntimeError("Environment does not expose step_metrics.")
            for _required_key in (
                "cube_pos_gz",
                "gripper_cube_alignment",
                "cube_rot6d_gz",
                "cube_height_w",
            ):
                if _required_key not in env.unwrapped.step_metrics:
                    raise RuntimeError(
                        f"step_metrics is missing required key {_required_key!r}. "
                        "Ensure the env config includes this key in critic_obs_metrics "
                        "or telemetry_metrics."
                    )
            _startup_validated = True

        # Only pay the cost of camera tensor reads on steps where at least one env
        # is scheduled for sampling.
        sampling_envs = [
            i
            for i in range(num_envs)
            if (episode_steps[i] % args_cli.sample_every_steps) == 0
        ]

        if sampling_envs:
            rgb_batch = gripper_cam.data.output["rgb"]
            seg_batch = gripper_cam.data.output["instance_segmentation_fast"]
            # TiledCamera info dict is shared across all envs; fetch it once.
            tiled_seg_info = _get_tiled_segmentation_info(gripper_cam.data.info)

            joint_pos_batch = env.unwrapped.joint_pos[:, env.unwrapped._dof_idx]
            cube_pos_gz_batch = env.unwrapped.step_metrics["cube_pos_gz"]
            gripper_cube_alignment_batch = env.unwrapped.step_metrics[
                "gripper_cube_alignment"
            ]
            cube_rot6d_gz_batch = env.unwrapped.step_metrics["cube_rot6d_gz"]
            cube_height_w_batch = env.unwrapped.step_metrics["cube_height_w"]

            for env_idx in sampling_envs:
                if env_idx not in cube_targets_by_env:
                    cube_targets_by_env[env_idx] = _extract_cube_targets(
                        tiled_seg_info, env_idx
                    )

                cube_in_frame = _cube_in_frame(
                    seg_batch[env_idx], cube_targets_by_env[env_idx]
                )

                done_flag = done[env_idx]
                if torch.is_tensor(done_flag):
                    done_flag = bool(done_flag.item())
                else:
                    done_flag = bool(done_flag)

                buffer["rgb"].append(_to_numpy(rgb_batch[env_idx], dtype=np.uint8))
                buffer["joint_pos"].append(
                    _to_numpy(joint_pos_batch[env_idx], dtype=np.float32)
                )
                buffer["cube_pos_gz"].append(
                    _to_numpy(cube_pos_gz_batch[env_idx], dtype=np.float32)
                )
                buffer["gripper_cube_alignment"].append(
                    float(gripper_cube_alignment_batch[env_idx].item())
                )
                buffer["cube_rot6d_gz"].append(
                    _to_numpy(cube_rot6d_gz_batch[env_idx], dtype=np.float32)
                )
                buffer["cube_height_w"].append(
                    float(cube_height_w_batch[env_idx].item())
                )
                buffer["cube_in_camera_frame"].append(bool(cube_in_frame))
                buffer["env_id"].append(env_idx)
                buffer["episode_id"].append(episode_counts[env_idx])
                buffer["episode_step"].append(episode_steps[env_idx])
                buffer["global_step"].append(global_step)
                buffer["simulation_step"].append(simulation_step)
                buffer["sim_time_s"].append(float(global_step * dt))
                buffer["done"].append(done_flag)
                samples_collected += 1

        if len(buffer["env_id"]) >= args_cli.samples_per_shard:
            shard_index = _flush_shard(
                shard_index=shard_index,
                shards_dir=shards_dir,
                buffer=buffer,
                shard_manifest=shard_manifest,
            )

        for env_idx in range(num_envs):
            episode_steps[env_idx] += 1

            done_flag = done[env_idx]
            if torch.is_tensor(done_flag):
                done_flag = bool(done_flag.item())
            else:
                done_flag = bool(done_flag)

            if done_flag:
                episode_counts[env_idx] += 1
                completed_episodes += 1
                episode_steps[env_idx] = 0

                print(
                    f"[INFO] Completed {completed_episodes}/{args_cli.num_episodes} episodes",
                    flush=True,
                )

                if completed_episodes >= args_cli.num_episodes:
                    break

    shard_index = _flush_shard(
        shard_index=shard_index,
        shards_dir=shards_dir,
        buffer=buffer,
        shard_manifest=shard_manifest,
    )

    finished_at = datetime.now(timezone.utc)
    metadata = {
        "run": {
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "task": args_cli.task,
            "checkpoint_path": str(checkpoint_path),
            "experiment_path": str(experiment_path),
            "env_config_path": (
                str(env_config_path) if env_config_path.exists() else None
            ),
            "seed": args_cli.seed,
            "num_envs": num_envs,
            "sample_every_steps": args_cli.sample_every_steps,
            "step_dt_seconds": float(dt),
            "num_episodes_target": args_cli.num_episodes,
            "num_episodes_completed": completed_episodes,
            "samples_per_shard": args_cli.samples_per_shard,
            "samples_collected": samples_collected,
            "ml_framework": args_cli.ml_framework,
            "algorithm": args_cli.algorithm,
        },
        "schema": {
            "rgb": "uint8 [S, H, W, 3]",
            "joint_pos": "float32 [S, num_active_joints]",
            "cube_pos_gz": "float32 [S, 3]",
            "gripper_cube_alignment": "float32 [S]",
            "cube_rot6d_gz": "float32 [S, 6]",
            "cube_height_w": "float32 [S]",
            "cube_in_camera_frame": "bool [S]",
            "env_id": "int32 [S]",
            "episode_id": "int32 [S]",
            "episode_step": "int32 [S]",
            "global_step": "int64 [S]",
            "simulation_step": "int64 [S]",
            "sim_time_s": "float64 [S]",
            "done": "bool [S]",
        },
        "camera": {
            "enabled_data_types": list(camera_data_types),
        },
        "shards": shard_manifest,
        "argv": vars(args_cli),
    }

    metadata_path = collect_log_dir / "telemetry_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Telemetry collection complete. Episodes: {completed_episodes}")
    print(f"[INFO] Samples collected: {samples_collected}")
    print(f"[INFO] Shards written: {len(shard_manifest)}")
    print(f"[INFO] Metadata written: {metadata_path}")

    env.close()


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
    simulation_app.close()
