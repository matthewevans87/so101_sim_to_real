"""export_bundle.py — Export a trained policy + CNN backbone to a self-contained
deploy bundle.

The bundle is designed to be loaded offline, without any Isaac Lab dependency.
It contains everything required to reproduce exact inference on a physical robot:

    deploy_bundle_<timestamp>/
        manifest.json               ← complete deploy contract
        policy.pt                   ← PolicyMLP weights + arch metadata
        cnn_backbone.pt             ← CNN backbone weights (frozen_cnn only)
        deploy_image_pipeline.yaml  ← ordered image preprocessing steps
        joint_config.yaml           ← active joints, lower/upper (rad), control rate
        bundle_provenance.json      ← git sha, source artifact hashes, timestamp
        README.md                   ← auto-generated quickstart

Input contract — the experiment directory is the SOLE source of inputs:

    <experiment-path>/
        env_config.yaml             ← required
        skrl/.../best_agent.pt      ← required
        cnn_checkpoint.pt           ← required iff vision_encoder.type == frozen_cnn

There are no env-var, CLI, or YAML-field overrides for these inputs. If you
need to swap an artifact, replace the file in the experiment directory (or
symlink it) and re-run.

Usage (called via run.py export or pipeline step 05_export):
    isaaclab.sh -p so101_rl/scripts/skrl/export_bundle.py
        --task So101-LiftCube-v0
        --experiment-path artifacts/<timestamp>
        --output artifacts/<timestamp>/deploy_bundle_<timestamp>
        [--torchscript]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Export a trained SKRL policy to a self-contained deploy bundle."
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Task name (e.g. So101-LiftCube-v0).",
)
parser.add_argument(
    "--experiment-path",
    type=str,
    required=True,
    help="Path to the training experiment directory (contains skrl/ and env_config.yaml).",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="Output directory for the deploy bundle.",
)
parser.add_argument(
    "--torchscript",
    action="store_true",
    help="Also trace and save a TorchScript combined model (combined.ts.pt).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import torch.nn as nn
import yaml

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import so101_rl.tasks  # noqa: F401

algorithm = "ppo"
agent_cfg_entry_point = "skrl_cfg_entry_point"

# ── Supported actor_obs_metrics at deploy time ────────────────────────────────
# Extend this set when the runtime controller supports additional metrics.
_DEPLOY_SUPPORTED_METRICS: frozenset[str] = frozenset()


# ── PolicyMLP ─────────────────────────────────────────────────────────────────


class PolicyMLP(nn.Module):
    """Plain ELU-activated MLP with tanh output.

    Shared definition between the exporter and so101_real.policy.  Architecture
    is auto-detected from the SKRL SharedModel's net_container, so any number
    of hidden layers is handled correctly.
    """

    def __init__(self, obs_dim: int, hidden_dims: list[int], act_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ELU()])
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _resolve_policy_checkpoint(experiment_path: Path) -> Path:
    """Return the policy checkpoint to export.

    Selection order (highest-step = last model, not SKRL's internal 'best'):

    1. ``run_manifest.json`` ``final_checkpoint_relpath`` — the manifest
       records the highest-step ``agent_<N>.pt`` at training completion and is
       the single source of truth for modern experiments.
    2. Highest-numbered ``agent_<N>.pt`` in ``skrl/`` — used when no manifest
       is present (legacy experiment dirs).

    ``best_agent.pt`` is intentionally NOT used. SKRL selects it by an entropy /
    KL metric that does not correlate with task success rate.
    """
    from so101_rl.run_manifest import RunManifest, MANIFEST_FILENAME

    skrl_dir = experiment_path / "skrl"
    if not skrl_dir.is_dir():
        raise FileNotFoundError(
            f"Experiment is missing required 'skrl/' subdirectory: {skrl_dir}\n"
            "Run training to completion before exporting."
        )

    manifest_path = experiment_path / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = RunManifest.load(experiment_path)
        ckpt = manifest.final_checkpoint_abs(experiment_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"run_manifest.json names final checkpoint at {ckpt} "
                "but the file is not present on disk."
            )
        return ckpt

    # Legacy path: no manifest — pick the highest-numbered agent_<N>.pt.
    ckpt_dir = skrl_dir / "agent" / "checkpoints"
    step_ckpts: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("agent_*.pt"):
        try:
            step_ckpts.append((int(p.stem.split("_", 1)[1]), p))
        except (ValueError, IndexError):
            continue
    if not step_ckpts:
        raise FileNotFoundError(
            f"No 'agent_<N>.pt' checkpoints found in {ckpt_dir}.\n"
            "Run training to completion before exporting."
        )
    _, ckpt = max(step_ckpts, key=lambda t: t[0])
    return ckpt


@dataclass(frozen=True)
class ExperimentInputs:
    """Resolved input artifacts for an export run.

    All fields are absolute paths to files that exist on disk. ``cnn_ckpt`` is
    ``None`` when the experiment does not use a frozen CNN backbone; otherwise
    it is required.
    """

    experiment_path: Path
    env_config: Path
    policy_ckpt: Path
    cnn_ckpt: Path | None


def _resolve_experiment_inputs(
    experiment_path: Path, vision_type: str
) -> ExperimentInputs:
    """Resolve all required input artifacts from the experiment directory.

    The experiment directory is the SOLE source of inputs to the exporter.
    Missing artifacts produce a hard error — there are no fallbacks.
    """
    if not experiment_path.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")

    env_config = experiment_path / "env_config.yaml"
    if not env_config.is_file():
        raise FileNotFoundError(
            f"Experiment is missing required env_config.yaml: {env_config}"
        )

    policy_ckpt = _resolve_policy_checkpoint(experiment_path)

    cnn_ckpt: Path | None = None
    if vision_type == "frozen_cnn":
        cnn_ckpt = experiment_path / "cnn_checkpoint.pt"
        if not cnn_ckpt.is_file():
            raise FileNotFoundError(
                f"vision_encoder.type == 'frozen_cnn' but the experiment is "
                f"missing the required CNN checkpoint: {cnn_ckpt}\n"
                "The training pipeline copies cnn_checkpoint.pt into the "
                "experiment directory; re-run training or place the CNN "
                "checkpoint at this exact path."
            )

    return ExperimentInputs(
        experiment_path=experiment_path,
        env_config=env_config,
        policy_ckpt=policy_ckpt,
        cnn_ckpt=cnn_ckpt,
    )


def _build_deploy_image_pipeline(so101_params, vision_type: str) -> list[dict]:
    """Return ordered image pipeline steps for deployment (no DR augmentations)."""
    return [{"type": "Uint8ToFloatCHW"}]


def _extract_backbone_cfg(so101_params) -> dict | None:
    """Serialize the backbone architecture from so101_params for frozen_cnn."""
    if so101_params.vision_encoder.type != "frozen_cnn":
        return None
    ve = so101_params.vision_encoder
    if ve.backbone is None:
        raise ValueError(
            "vision_encoder.backbone must be set when vision_encoder.type == 'frozen_cnn'."
        )
    return {
        "in_channels": 3,
        "channels": list(ve.backbone.channels),
        "kernel_sizes": list(ve.backbone.kernel_sizes),
        "strides": list(ve.backbone.strides),
        "mlp_hidden_dims": list(ve.backbone.mlp_hidden_dims),
        "output_dim": ve.backbone.output_dim,
    }


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: dict,
) -> None:
    from skrl.utils.runner.torch import Runner

    experiment_path = Path(args_cli.experiment_path).resolve()
    output_dir = Path(args_cli.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load So101EnvParams from the experiment dir (sole input contract) ────
    # env.unwrapped.cfg is So101LiftCubeCfg (IsaacLab dataclass) — it does NOT carry
    # So101-specific fields like vision_encoder, joints, observations, etc.
    # Those live in So101EnvParams, which is loaded from env_config.yaml.
    from so101_rl.configurations.so101_env_params import So101EnvParams

    env_config_path = experiment_path / "env_config.yaml"
    if not env_config_path.is_file():
        raise FileNotFoundError(
            f"Experiment is missing required env_config.yaml: {env_config_path}"
        )
    so101_params = So101EnvParams.load(str(env_config_path))
    vision_type: str = so101_params.vision_encoder.type

    # ── Resolve all input artifacts up front (hard errors on missing inputs) ──
    inputs = _resolve_experiment_inputs(experiment_path, vision_type)
    print(f"[export_bundle] env_config:    {inputs.env_config}")
    print(f"[export_bundle] policy_ckpt:   {inputs.policy_ckpt}")
    print(f"[export_bundle] cnn_ckpt:      {inputs.cnn_ckpt}")

    log_dir = str(inputs.policy_ckpt.parent.parent)
    agent_cfg["agent"]["experiment"]["directory"] = log_dir
    agent_cfg["agent"]["experiment"]["experiment_name"] = "agent"

    # ── Create env (needed to materialize LazyLinear and read obs/act dims) ──
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)
    runner.agent.load(retrieve_file_path(str(inputs.policy_ckpt)))

    shared_model = runner.agent.models["policy"]
    shared_model.eval()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device = next(shared_model.parameters()).device

    # Materialize LazyLinear
    with torch.no_grad():
        shared_model.compute(
            {"states": torch.zeros(1, obs_dim, device=device)}, role="policy"
        )

    # ── Detect hidden layer sizes from net_container ──────────────────────────
    # net_container: [Linear, ELU, Linear, ELU, ..., Linear(act_dim)]
    net = shared_model.net_container
    hidden_dims: list[int] = []
    for layer in net:
        if isinstance(layer, nn.Linear) and layer.out_features != act_dim:
            hidden_dims.append(layer.out_features)
    print(f"[export_bundle] Architecture: {obs_dim} → {hidden_dims} → {act_dim}")

    # ── Build and populate PolicyMLP ──────────────────────────────────────────
    policy = PolicyMLP(obs_dim=obs_dim, hidden_dims=hidden_dims, act_dim=act_dim)

    with torch.no_grad():
        linear_layers = [l for l in net if isinstance(l, nn.Linear)]
        policy_linears = [l for l in policy.net if isinstance(l, nn.Linear)]
        if len(linear_layers) != len(policy_linears):
            raise RuntimeError(
                f"Linear layer count mismatch: shared_model has {len(linear_layers)}, "
                f"PolicyMLP has {len(policy_linears)}."
            )
        for src, dst in zip(linear_layers, policy_linears):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)

    policy_cpu = policy.to("cpu").eval()

    # ── Validate actor_obs_metrics ────────────────────────────────────────────
    actor_obs_metrics: list[str] = list(
        so101_params.observations.actor_obs_metrics or []
    )
    unsupported = set(actor_obs_metrics) - _DEPLOY_SUPPORTED_METRICS
    if unsupported:
        raise ValueError(
            f"Policy was trained with actor_obs_metrics that are not yet supported "
            f"at deploy time: {sorted(unsupported)}.\n"
            f"Supported metrics: {sorted(_DEPLOY_SUPPORTED_METRICS)}.\n"
            f"Add support for these metrics in so101_real/controller.py, then extend "
            f"_DEPLOY_SUPPORTED_METRICS in this script."
        )

    active_joints: list[str] = list(so101_params.joints.active)
    # sim.dt and decimation come from the IsaacLab cfg (the resolved Hydra object)
    isaac_cfg = env.unwrapped.cfg
    sim_dt: float = float(isaac_cfg.sim.dt)
    decimation: int = int(isaac_cfg.decimation)
    control_hz: float = 1.0 / (sim_dt * decimation)

    # Action-pipeline parameters — baked into training; must travel with the bundle
    # so the real robot uses the same values without requiring manual robot.yaml edits.
    ema_alpha: float = float(so101_params.joint_command.ema_alpha)
    max_delta_rad: float = float(so101_params.joint_command.max_delta_rad)
    ema_joints: list[str] = list(so101_params.joint_command.ema_joints or [])
    clamp_joints: list[str] = list(so101_params.joint_command.clamp_joints or [])

    # Joint limits from env tensors (populated by env._setup_scene). The
    # attributes are required — if absent, the env contract changed and we
    # must not silently emit empty joint limits to the bundle.
    env_unwrapped = env.unwrapped
    if not hasattr(env_unwrapped, "_joint_lower") or not hasattr(
        env_unwrapped, "_joint_upper"
    ):
        raise RuntimeError(
            "Environment does not expose '_joint_lower'/'_joint_upper' tensors. "
            "These are required for the deploy bundle's joint limits. "
            "Has the So101LiftCubeEnv contract changed?"
        )
    joint_lower_rad: list[float] = env_unwrapped._joint_lower[0].tolist()
    joint_upper_rad: list[float] = env_unwrapped._joint_upper[0].tolist()
    if len(joint_lower_rad) != len(active_joints) or len(joint_upper_rad) != len(
        active_joints
    ):
        raise RuntimeError(
            f"Joint-limit tensor length ({len(joint_lower_rad)}) does not match "
            f"number of active joints ({len(active_joints)})."
        )

    camera_height: int = int(so101_params.vision_encoder.image_height)
    camera_width: int = int(so101_params.vision_encoder.image_width)

    # ── Save policy.pt ────────────────────────────────────────────────────────
    policy_path = output_dir / "policy.pt"
    torch.save(
        {
            "state_dict": policy_cpu.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dims": hidden_dims,
        },
        policy_path,
    )
    print(f"[export_bundle] Saved policy.pt → {policy_path}")

    # ── Copy CNN backbone (already validated to exist by the resolver) ───────
    cnn_checkpoint_path: Path | None = None
    if vision_type == "frozen_cnn":
        assert inputs.cnn_ckpt is not None  # guaranteed by resolver
        cnn_dest = output_dir / "cnn_backbone.pt"
        shutil.copy2(inputs.cnn_ckpt, cnn_dest)
        cnn_checkpoint_path = cnn_dest
        print(f"[export_bundle] Saved cnn_backbone.pt → {cnn_dest}")

    # ── Copy camera intrinsics (if model == "opencv_pinhole") ────────────────
    camera_intrinsics_file: str | None = None
    camera_cfg = so101_params.sensors.camera
    if camera_cfg.model == "opencv_pinhole":
        workspace_path = os.environ.get("ISAAC_LAB_WORKSPACE_PATH", "/workspace")
        intrinsics_src = Path(workspace_path) / camera_cfg.intrinsics_path
        if not intrinsics_src.is_file():
            raise FileNotFoundError(
                f"Camera intrinsics file not found: {intrinsics_src}\n"
                "Run the calibration pipeline first:\n"
                "  python -m so101_real calibrate-camera --solve\n"
                "Then set sensors.camera.intrinsics_path in the env config YAML."
            )
        intrinsics_dest = output_dir / "camera_intrinsics.yaml"
        shutil.copy2(intrinsics_src, intrinsics_dest)
        camera_intrinsics_file = "camera_intrinsics.yaml"
        print(f"[export_bundle] Saved camera_intrinsics.yaml → {intrinsics_dest}")

    # ── Save deploy_image_pipeline.yaml ──────────────────────────────────────
    pipeline_steps = _build_deploy_image_pipeline(so101_params, vision_type)
    pipeline_path = output_dir / "deploy_image_pipeline.yaml"
    with open(pipeline_path, "w") as f:
        yaml.dump(
            {"steps": pipeline_steps}, f, default_flow_style=False, sort_keys=False
        )
    print(f"[export_bundle] Saved deploy_image_pipeline.yaml → {pipeline_path}")

    # ── Save joint_config.yaml ────────────────────────────────────────────────
    joint_cfg = {
        "active_joints": active_joints,
        "joint_lower_rad": joint_lower_rad,
        "joint_upper_rad": joint_upper_rad,
        "control_hz": control_hz,
        "sim_dt": sim_dt,
        "decimation": decimation,
    }
    joint_path = output_dir / "joint_config.yaml"
    with open(joint_path, "w") as f:
        yaml.dump(joint_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[export_bundle] Saved joint_config.yaml → {joint_path}")

    # ── Build and save manifest.json ──────────────────────────────────────────
    backbone_cfg = _extract_backbone_cfg(so101_params)
    manifest = {
        "schema_version": "1",
        "task": args_cli.task,
        "vision_encoder": {
            "type": vision_type,
            "image_height": camera_height,
            "image_width": camera_width,
            "backbone": backbone_cfg,
        },
        "policy": {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dims": hidden_dims,
            "file": "policy.pt",
        },
        "cnn_backbone_file": "cnn_backbone.pt" if vision_type == "frozen_cnn" else None,
        "deploy_image_pipeline_file": "deploy_image_pipeline.yaml",
        "joint_config_file": "joint_config.yaml",
        "camera_intrinsics_file": camera_intrinsics_file,
        "active_joints": active_joints,
        "joint_lower_rad": joint_lower_rad,
        "joint_upper_rad": joint_upper_rad,
        "control_hz": control_hz,
        "ema_alpha": ema_alpha,
        "max_delta_rad": max_delta_rad,
        "ema_joints": ema_joints,
        "clamp_joints": clamp_joints,
        "actor_obs_metrics": actor_obs_metrics,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export_bundle] Saved manifest.json → {manifest_path}")

    # ── Save bundle_provenance.json ───────────────────────────────────────────
    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "task": args_cli.task,
        "experiment_path": str(experiment_path),
        "checkpoint_path": str(inputs.policy_ckpt),
        "checkpoint_sha256": _sha256(inputs.policy_ckpt),
        "env_config_sha256": _sha256(inputs.env_config),
        "cnn_checkpoint_sha256": (
            _sha256(cnn_checkpoint_path) if cnn_checkpoint_path else None
        ),
    }
    provenance_path = output_dir / "bundle_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"[export_bundle] Saved bundle_provenance.json → {provenance_path}")

    # ── Optional TorchScript trace ────────────────────────────────────────────
    if args_cli.torchscript:
        try:
            scripted = torch.jit.script(policy_cpu)
            ts_path = output_dir / "policy.ts.pt"
            scripted.save(str(ts_path))
            print(f"[export_bundle] Saved TorchScript policy → {ts_path}")
        except Exception as exc:
            print(
                f"[export_bundle] WARNING: TorchScript trace failed: {exc}\n"
                "  Bundle is still usable in eager mode."
            )

    # ── Save README.md ────────────────────────────────────────────────────────
    readme = f"""\
# Deploy Bundle: {args_cli.task}

Generated: {provenance["timestamp"]}
Experiment: {experiment_path.name}

## Contents

| File | Description |
|------|-------------|
| manifest.json | Complete deploy contract (obs spec, image pipeline, joint config) |
| policy.pt | PolicyMLP weights (obs_dim={obs_dim}, act_dim={act_dim}, hidden={hidden_dims}) |
{"| cnn_backbone.pt | Frozen CNN backbone weights |" if vision_type == "frozen_cnn" else ""}
| deploy_image_pipeline.yaml | Ordered image preprocessing steps |
| joint_config.yaml | Active joints, joint limits (rad), control rate ({control_hz:.1f} Hz) |
| bundle_provenance.json | Git SHA, checkpoint hashes, creation timestamp |

## Quick-start

```bash
# Install the so101_real runtime package first (see so101_real/README.md)

python -m so101_real run \\
    --bundle path/to/this/dir \\
    --robot-config so101_real/configs/robot.yaml \\
    --overlay
```

## Vision encoder: {vision_type}

{"Obs: [CNN spatial-softmax features (2×channels[-1]) | joint positions (rad)]" if vision_type == "frozen_cnn" else "Obs: [ResNet18 spatial-softmax features (1024) | joint positions (rad)]"}

## Action space

{act_dim} joints (same order as `active_joints` in manifest.json).
Action ∈ [-1, 1] is mapped to joint targets via:
  q_target = joint_lower + 0.5 * (action + 1.0) * (joint_upper - joint_lower)
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)

    env.close()
    print(f"\n[export_bundle] Bundle complete → {output_dir}")
    print(f"[export_bundle] Files: {sorted(p.name for p in output_dir.iterdir())}")


if __name__ == "__main__":
    main()
    simulation_app.close()
