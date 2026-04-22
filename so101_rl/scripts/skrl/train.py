# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
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
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
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
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint to resume training.",
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
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
    "--artifacts_dir",
    type=str,
    default=None,
    help="Root artifacts directory for this run (e.g. /path/to/artifacts/2026-03-11_10-20-38).",
)
parser.add_argument(
    "--cnn_checkpoint",
    type=str,
    default=None,
    help=(
        "Path to a pretrained MultiTaskCnn checkpoint produced by "
        "train_cnn.py (best_model.pt or final_model.pt). "
        "Only valid when vision_encoder.type == 'frozen_cnn'. "
        "Weights are loaded into the frozen CNN feature extractor before training starts."
    ),
)
parser.add_argument(
    "--agent_config",
    type=str,
    default=None,
    help=(
        "Path to a YAML file whose contents are deep-merged into the agent config "
        "loaded from the task registry.  Used by the sweep orchestrator to apply "
        "per-experiment agent overrides (e.g. learning_rate, rollouts).  "
        "Override values take precedence over the registered defaults."
    ),
)
parser.add_argument(
    "--env_config",
    type=str,
    default=None,
    help=(
        "Path to the env config YAML that drives the So101LiftCubeCfg module.  "
        "When provided, this script sets SO101_ENV_CONFIG explicitly before any "
        "task module is imported.  If omitted, the SO101_ENV_CONFIG environment "
        "variable must already be set; otherwise startup fails immediately with a "
        "descriptive error."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# ── Resolve env-config contract BEFORE any so101_rl import ────────────────────
# The env config is read at module-import time inside
# so101_rl.tasks.direct.so101_lift_cube.so101_lift_cube_env_cfg via
# os.environ["SO101_ENV_CONFIG"].  This block makes the contract explicit:
# either the user passes --env_config (preferred, sweep orchestrator does this),
# or the env var is already set externally.  Anything else fails loud here
# rather than as an opaque KeyError deep inside Hydra.
if args_cli.env_config is not None:
    _env_cfg_path = Path(args_cli.env_config).resolve()
    if not _env_cfg_path.is_file():
        raise FileNotFoundError(
            f"--env_config path does not exist: {_env_cfg_path}"
        )
    os.environ["SO101_ENV_CONFIG"] = str(_env_cfg_path)
elif "SO101_ENV_CONFIG" not in os.environ:
    raise EnvironmentError(
        "SO101_ENV_CONFIG is not set and --env_config was not provided. "
        "Pass --env_config <path/to/env.yaml> or export SO101_ENV_CONFIG before "
        "launching train.py."
    )

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import random
import torch

import omni
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
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import so101_rl.tasks  # noqa: F401


def _flatten_for_hparams(obj, prefix: str = "") -> dict:
    """Recursively flatten a nested dict/object into a flat dict for TensorBoard add_hparams.

    Keys are joined with '/' (e.g. ``"env/rewards/distance/scale"``).  Values are
    coerced to types accepted by PyTorch's SummaryWriter.add_hparams:
    int, float, str, bool; everything else is stringified.
    """
    result: dict[str, int | float | str | bool] = {}
    if isinstance(obj, dict):
        items = obj.items()
    elif hasattr(obj, "__dict__"):
        items = vars(obj).items()
    else:
        # Scalar leaf — shouldn't normally be called directly with a scalar
        key = prefix or "value"
        if isinstance(obj, (int, float, bool)):
            return {key: obj}
        return {key: str(obj) if obj is not None else "null"}

    for k, v in items:
        full_key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict) or hasattr(v, "__dict__"):
            result.update(_flatten_for_hparams(v, full_key))
        elif isinstance(v, (int, float, bool)):
            result[full_key] = v
        elif v is None:
            result[full_key] = "null"
        else:
            # lists, tuples, and any other type → human-readable string
            result[full_key] = str(v)
    return result


def _log_configs_to_tensorboard(
    log_dir: str, env_yaml_path: str, agent_yaml_path: str
) -> None:
    """Write env and agent configs to TensorBoard as both TEXT and HPARAMS entries.

    TEXT entries (``config/env``, ``config/agent``) show the raw YAML in the TEXT tab
    for quick human inspection.  The HPARAMS entry writes a single row to the HPARAMS
    tab so that multiple runs can be compared side-by-side.

    Both config YAML files are expected to already exist on disk (written by
    ``dump_yaml`` earlier in ``main()``).
    """
    import yaml
    from torch.utils.tensorboard import SummaryWriter

    # Isaac Lab's dump_yaml serialises Python tuples as !!python/tuple tags.
    # yaml.safe_load refuses these, so we add a single targeted constructor that
    # coerces tuples to lists — safe because these files are machine-generated
    # by our own code and contain no arbitrary Python objects.
    class _Loader(yaml.SafeLoader):
        pass

    _Loader.add_constructor(
        "tag:yaml.org,2002:python/tuple",
        lambda loader, node: list(loader.construct_sequence(node)),
    )

    with open(env_yaml_path, "r") as f:
        env_yaml_text = f.read()
    with open(agent_yaml_path, "r") as f:
        agent_yaml_text = f.read()

    env_cfg_dict = (
        yaml.load(env_yaml_text, Loader=_Loader) or {}
    )  # noqa: S506 (controlled input)
    agent_cfg_dict = yaml.load(agent_yaml_text, Loader=_Loader) or {}  # noqa: S506

    # Write to the run's log_dir directly; SKRL writes its own events into
    # log_dir/runs/{uuid}/ — TensorBoard discovers all event files recursively.
    writer = SummaryWriter(log_dir=log_dir)
    try:
        writer.add_text("config/env", f"```yaml\n{env_yaml_text}\n```", global_step=0)
        writer.add_text(
            "config/agent", f"```yaml\n{agent_yaml_text}\n```", global_step=0
        )

        flat: dict[str, int | float | str | bool] = {
            **_flatten_for_hparams(env_cfg_dict, "env"),
            **_flatten_for_hparams(agent_cfg_dict, "agent"),
        }
        # add_hparams requires at least one metric; use a dummy placeholder.
        writer.add_hparams(flat, {"_placeholder": 0.0})
    finally:
        writer.close()


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


def _deep_merge_dicts(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*, returning a new dict."""
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge_dicts(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with skrl agent."""
    # Apply per-experiment agent config overrides supplied by the sweep orchestrator.
    # This must happen before any other use of agent_cfg so that overrides such as
    # learning_rate or rollouts take effect for all downstream calculations.
    if args_cli.agent_config is not None:
        import yaml as _yaml

        _override_path = Path(args_cli.agent_config)
        if not _override_path.is_file():
            raise FileNotFoundError(
                f"--agent_config path not found: {_override_path.resolve()}"
            )
        with open(_override_path) as _f:
            _agent_overrides = _yaml.safe_load(_f) or {}
        agent_cfg = _deep_merge_dicts(agent_cfg, _agent_overrides)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # check for invalid combination of CPU device with distributed training
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device
    ):
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = (
            args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        )
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    )
    env_cfg.seed = agent_cfg["seed"]

    # Seed all global RNGs for reproducibility
    _seed = agent_cfg["seed"]
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)

    # specify directory for logging experiments
    log_root_path = os.path.abspath(os.path.join(args_cli.artifacts_dir, "skrl"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Fix experiment_name to "agent" so skrl always creates skrl/agent/ (no timestamp suffix)
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = "agent"
    agent_cfg["agent"]["experiment"]["write_interval"] = 100

    log_dir = log_root_path

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # log configs to TensorBoard (TEXT tab + HPARAMS comparison tab)
    _log_configs_to_tensorboard(
        log_dir=log_dir,
        env_yaml_path=os.path.join(log_dir, "params", "env.yaml"),
        agent_yaml_path=os.path.join(log_dir, "params", "agent.yaml"),
    )

    # get checkpoint path (to resume training)
    resume_path = (
        retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    )

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

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
        # Embed a copy of the CNN checkpoint into the experiment directory so
        # that collect_telemetry.py (and future tools) can locate it without
        # any extra flags, and the experiment remains self-contained.
        cnn_src = os.path.abspath(args_cli.cnn_checkpoint)
        artifacts_path = Path(args_cli.artifacts_dir).resolve()
        embedded_cnn = artifacts_path / "cnn_checkpoint.pt"
        shutil.copy2(cnn_src, embedded_cnn)
        provenance = {
            "source_path": cnn_src,
            "copied_at": datetime.now(timezone.utc).isoformat(),
        }
        (artifacts_path / "cnn_checkpoint_provenance.json").write_text(
            json.dumps(provenance, indent=2)
        )
        print(f"[INFO] CNN checkpoint embedded: {embedded_cnn}")
        env_cfg.vision_encoder.cnn_checkpoint = str(embedded_cnn)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # Inject milestone log into the episode stats pipeline so that exact
    # env_transitions counts are written to milestones.json at each milestone.
    from so101_rl.milestone_log import MilestoneLog

    _milestone_log = MilestoneLog(
        output_path=os.path.join(args_cli.artifacts_dir, "milestones.json"),
        num_envs=env_cfg.scene.num_envs,
    )
    _underlying_env = env.unwrapped
    if hasattr(_underlying_env, "episode_stats_pipeline"):
        _underlying_env.episode_stats_pipeline.set_milestone_log(_milestone_log)
    else:
        omni.log.warn(
            "[MilestoneLog] episode_stats_pipeline not found on unwrapped env; "
            "milestones.json will not be written."
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )  # same as: `wrap_env(env, wrapper="auto")`

    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # ── Write run manifest ──────────────────────────────────────────────────
    # The manifest is the contract eval reads to know exactly which configs and
    # checkpoints were produced by this run.  Anything that affects the run's
    # reproducibility belongs here.  Failures during manifest write are
    # FATAL — without it, eval cannot trust the experiment dir.
    from so101_rl.run_manifest import RunManifest

    artifacts_path = Path(args_cli.artifacts_dir).resolve()

    # Ensure env_config.yaml lives at the canonical experiment-dir location.
    # Sweep / run.py already copy it there; we double-check and copy if not.
    canonical_env_cfg = artifacts_path / "env_config.yaml"
    src_env_cfg = Path(os.environ["SO101_ENV_CONFIG"]).resolve()
    if not canonical_env_cfg.is_file():
        shutil.copy2(src_env_cfg, canonical_env_cfg)
    elif canonical_env_cfg.resolve() != src_env_cfg:
        # Both exist but at different paths — the upstream caller staged it for us.
        # Verify byte-equality so the hash recorded in the manifest matches what
        # the env config module actually loaded.
        if canonical_env_cfg.read_bytes() != src_env_cfg.read_bytes():
            raise RuntimeError(
                f"env_config drift detected: SO101_ENV_CONFIG points to {src_env_cfg} "
                f"but the experiment dir already contains a different env_config.yaml at "
                f"{canonical_env_cfg}.  Refusing to write an inconsistent manifest."
            )

    # Locate the highest-step checkpoint produced by skrl.
    # Convention is agent_<step>.pt; we treat the maximum step as the final.
    ckpt_dir = artifacts_path / "skrl" / "agent" / "checkpoints"
    if not ckpt_dir.is_dir():
        raise RuntimeError(
            f"Expected skrl checkpoint directory at {ckpt_dir} but it does not exist. "
            f"Training appears not to have produced any checkpoints."
        )
    step_ckpts: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("agent_*.pt"):
        stem = p.stem  # e.g. "agent_15000"
        try:
            step_ckpts.append((int(stem.split("_", 1)[1]), p))
        except (ValueError, IndexError):
            continue
    if not step_ckpts:
        raise RuntimeError(
            f"No agent_<step>.pt checkpoints found in {ckpt_dir}. "
            f"Cannot record a final checkpoint in the manifest."
        )
    final_step, final_ckpt = max(step_ckpts, key=lambda t: t[0])

    # Find the agent_config.yaml produced by sweep (preferred) or fall back to
    # the dumped agent.yaml in the skrl params/ dir.
    agent_cfg_canonical = artifacts_path / "agent_config.yaml"
    if not agent_cfg_canonical.is_file():
        # Use train.py's own dumped copy as the canonical file and mirror it.
        dumped_agent = artifacts_path / "skrl" / "params" / "agent.yaml"
        if not dumped_agent.is_file():
            raise RuntimeError(
                f"No agent_config.yaml at {agent_cfg_canonical} and no fallback at "
                f"{dumped_agent}.  Cannot write manifest."
            )
        shutil.copy2(dumped_agent, agent_cfg_canonical)

    cnn_ckpt_path: Path | None = None
    cnn_ckpt_source: Path | None = None
    if args_cli.cnn_checkpoint:
        cnn_ckpt_path = artifacts_path / "cnn_checkpoint.pt"
        cnn_ckpt_source = Path(args_cli.cnn_checkpoint).resolve()

    repo_root = Path(__file__).resolve().parents[3]
    manifest = RunManifest.build(
        experiment_dir=artifacts_path,
        repo_root=repo_root,
        task=args_cli.task,
        seed=int(agent_cfg["seed"]),
        trainer_timesteps=int(agent_cfg["trainer"]["timesteps"]),
        training_command=list(sys.argv),
        env_config_path=canonical_env_cfg,
        agent_config_path=agent_cfg_canonical,
        final_checkpoint_path=final_ckpt,
        final_checkpoint_step=final_step,
        cnn_checkpoint_path=cnn_ckpt_path,
        cnn_checkpoint_source=cnn_ckpt_source,
    )
    manifest_path = manifest.write(artifacts_path)
    print(f"[INFO] RunManifest written: {manifest_path}")
    print(f"[INFO] Final checkpoint: {final_ckpt} (step {final_step})")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
