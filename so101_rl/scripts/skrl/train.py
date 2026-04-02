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
import sys

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
import numpy as np
import os
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with skrl agent."""
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
    # Write directly into log_root_path — no experiment_name subdir
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = ""
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
        env_cfg.vision_encoder.cnn_checkpoint = args_cli.cnn_checkpoint

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
