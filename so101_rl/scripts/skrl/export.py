# so101_rl/scripts/skrl/export.py

import argparse
import sys
import os
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Export SKRL PPO policy as plain PyTorch MLP."
)
parser.add_argument(
    "--task", type=str, required=True, help="Task name (e.g. So101-JointPosGoUp-v0)."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to SKRL agent checkpoint (e.g. best_agent.pt).",
)
# AppLauncher args (device, headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn
import skrl
from packaging import version

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import so101_rl.tasks  # noqa: F401

# decide agent config entry point (same as train.py)
algorithm = "ppo"
agent_cfg_entry_point = "skrl_cfg_entry_point"


class MinimalPolicy(nn.Module):
    """Simple MLP: obs_dim -> hidden1 -> hidden2 -> hidden3 -> act_dim with ELU activations."""

    def __init__(
        self, obs_dim: int, hidden1: int, hidden2: int, hidden3: int, act_dim: int
    ):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, act_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.elu(self.fc1(x))
        x = torch.elu(self.fc2(x))
        x = torch.elu(self.fc3(x))
        return self.fc4(x)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    from skrl.utils.runner.torch import Runner

    # minimal logging location - not used for export itself
    log_root_path = os.path.abspath(
        os.path.join("logs", "export", agent_cfg["agent"]["experiment"]["directory"])
    )
    os.makedirs(log_root_path, exist_ok=True)
    log_dir = os.path.join(
        log_root_path,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_export",
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Export log dir: {log_dir}")
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = os.path.basename(log_dir)

    # create env
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # build runner (agent + SharedModel)
    runner = Runner(env, agent_cfg)

    # load checkpoint
    ckpt_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")
    runner.agent.load(ckpt_path)

    # extract shared policy model
    shared_model = runner.agent.models["policy"]  # SharedModel
    shared_model.eval()

    # initialize LazyLinear layers by running one compute() call
    obs_space = env.observation_space
    act_space = env.action_space
    assert hasattr(obs_space, "shape") and hasattr(act_space, "shape")
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]

    device = next(shared_model.parameters()).device
    dummy_states = torch.zeros(1, obs_dim, device=device)
    # this will materialize LazyLinear weights
    with torch.no_grad():
        shared_model.compute({"states": dummy_states}, role="policy")

    # Extract hidden layer sizes from the loaded model
    # net_container structure: [Linear, ELU, Linear, ELU, Linear, ELU, Linear]
    # For 3 hidden layers: [fc1, elu, fc2, elu, fc3, elu, fc4_out]
    hidden1_size = shared_model.net_container[0].out_features
    hidden2_size = shared_model.net_container[2].out_features
    hidden3_size = shared_model.net_container[4].out_features
    print(
        f"[INFO] Detected architecture: {obs_dim} -> {hidden1_size} -> {hidden2_size} -> {hidden3_size} -> {act_dim}"
    )

    # build minimal MLP with detected sizes
    minimal = MinimalPolicy(
        obs_dim, hidden1_size, hidden2_size, hidden3_size, act_dim
    ).to(device)

    # copy weights from SharedModel -> MinimalPolicy
    with torch.no_grad():
        # net_container: [LazyLinear(->hidden1), ELU, LazyLinear(->hidden2), ELU, LazyLinear(->hidden3), ELU, LazyLinear(->act_dim)]
        minimal.fc1.weight.copy_(shared_model.net_container[0].weight)
        minimal.fc1.bias.copy_(shared_model.net_container[0].bias)
        minimal.fc2.weight.copy_(shared_model.net_container[2].weight)
        minimal.fc2.bias.copy_(shared_model.net_container[2].bias)
        minimal.fc3.weight.copy_(shared_model.net_container[4].weight)
        minimal.fc3.bias.copy_(shared_model.net_container[4].bias)
        # output layer is the last layer in net_container
        minimal.fc4.weight.copy_(shared_model.net_container[6].weight)
        minimal.fc4.bias.copy_(shared_model.net_container[6].bias)

    # move to CPU and save
    minimal_cpu = minimal.to("cpu")

    # Generate filename based on task name
    task_name = args_cli.task.replace("-", "_").lower()
    export_path = os.path.join(log_dir, f"{task_name}_policy.pt")

    # Try to get joint/DOF names from config
    dof_names = None
    if hasattr(env.unwrapped.cfg, "dof_names"):
        dof_names = env.unwrapped.cfg.dof_names
    elif hasattr(env.unwrapped.cfg, "ACTIVE_JOINTS"):
        dof_names = env.unwrapped.cfg.ACTIVE_JOINTS
    elif hasattr(env.unwrapped.cfg, "JOINTS"):
        dof_names = env.unwrapped.cfg.JOINTS

    torch.save(
        {
            "state_dict": minimal_cpu.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden1": hidden1_size,
            "hidden2": hidden2_size,
            "hidden3": hidden3_size,
            "dof_names": dof_names,
        },
        export_path,
    )
    print(f"[INFO] Exported minimal policy to: {export_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
