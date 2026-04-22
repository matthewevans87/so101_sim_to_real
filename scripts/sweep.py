#!/usr/bin/env python3
"""
sweep.py — SweepOrchestrator for sequentially running named Train + Eval experiments.

Each experiment runs:  train → eval

All artifacts are written under a single sweep directory so results are easy to
compare with TensorBoard or the generated Markdown/JSON summary.

Directory layout:
  sweeps/sweep_{name}_{timestamp}/
    sweep.yaml              ← original sweep config, copied at start
    sweep_state.json        ← per-experiment status (pending/running/done/failed),
                              written atomically after every step
    summary.json            ← comparison table (written when sweep finishes)
    summary.md              ← Markdown report   (written when sweep finishes)
    experiments/
      01_baseline/
        env_config.yaml     ← materialized env config for this experiment
        agent_config.yaml   ← materialized agent config for this experiment
        train.log
        eval.log
        skrl/               ← train.py output (checkpoints/, runs/)
        evaluation/         ← evaluate.py output (results.json, videos/)
      02_approach_scale_2x/
        ...
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── constants ──────────────────────────────────────────────────────────────────
STATE_FILE = "sweep_state.json"
CONFIG_FILE = "sweep.yaml"

# ── terminal colours (same palette as pipeline.py / run.py) ──────────────────
_RED = "\033[0;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_BLUE = "\033[0;34m"
_NC = "\033[0m"


def _info(msg: str) -> None:
    print(f"{_YELLOW}ℹ  {msg}{_NC}")


def _success(msg: str) -> None:
    print(f"{_GREEN}✓  {msg}{_NC}")


def _error(msg: str) -> None:
    print(f"{_RED}✗  {msg}{_NC}", file=sys.stderr)


def _header(msg: str) -> None:
    bar = "=" * 60
    print(f"{_BLUE}{bar}{_NC}")
    print(f"{_BLUE}{msg}{_NC}")
    print(f"{_BLUE}{bar}{_NC}")


# ── subprocess tee helper (mirrors pipeline.py) ───────────────────────────────
def _run_step_subprocess(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> int:
    """
    Run a command, tee-ing stdout+stderr to the terminal and an optional log file.
    Returns the process exit code.
    """
    resolved_env = env if env is not None else os.environ.copy()
    log_fh = open(log_path, "w") if log_path else None
    rc = 1
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=resolved_env,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        proc.wait()
        rc = proc.returncode
    finally:
        if log_fh:
            log_fh.close()
    return rc


# ══════════════════════════════════════════════════════════════════════════════
# Config merge helpers
# ══════════════════════════════════════════════════════════════════════════════


def _deep_merge(base: dict, overrides: dict) -> dict:
    """
    Recursively merge *overrides* into *base*.  Returns a new dict.
    - Nested dicts are merged recursively.
    - All other values (scalars, lists, None) are replaced by the override value.
    The original dicts are not mutated.
    """
    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _merge_rewards_list(base_rewards: list, reward_overrides: list) -> list:
    """
    Merge reward override entries into *base_rewards*.

    Matching rules (checked in order for each base entry):
    1. **Exact match** — override entry has both ``type`` and ``id`` fields that
       match a base entry's ``type`` and ``id``.  Patches only that specific
       instance, leaving any sibling entries of the same ``type`` untouched.
    2. **Type-only match** — override entry has a ``type`` field but no ``id``.
       Patches the **first** base entry whose ``type`` matches (legacy behaviour,
       backward-compatible with sweep configs written before ``id`` was added).
    3. **No match** — the override entry is appended as a new reward step.

    Only the fields explicitly listed in the override entry are patched; all
    other fields on the matched base entry are preserved unchanged.

    The original lists are not mutated.
    """
    if not reward_overrides:
        return copy.deepcopy(base_rewards)
    result = copy.deepcopy(base_rewards)

    # Build two override indexes:
    #   by_type_and_id — keyed by (type, id); used for exact-match targeting
    #   by_type_only   — keyed by type; used for broad/legacy matching
    by_type_and_id: Dict[Tuple[str, str], dict] = {}
    by_type_only: Dict[str, dict] = {}
    for r in reward_overrides:
        t = r.get("type")
        if t is None:
            continue
        rid = r.get("id")
        if rid is not None:
            by_type_and_id[(t, rid)] = r
        else:
            by_type_only[t] = r

    matched_type_id: set = set()
    matched_type: set = set()

    for entry in result:
        t = entry.get("type")
        if not t:
            continue
        rid = entry.get("id")

        # Exact-match check first
        if rid is not None and (t, rid) in by_type_and_id:
            override = by_type_and_id[(t, rid)]
            for field, value in override.items():
                if field not in ("type", "id"):
                    entry[field] = copy.deepcopy(value)
            matched_type_id.add((t, rid))

        # Type-only fallback (first matching base entry wins)
        elif t in by_type_only and t not in matched_type:
            override = by_type_only[t]
            for field, value in override.items():
                if field != "type":
                    entry[field] = copy.deepcopy(value)
            matched_type.add(t)

    # Append unmatched override entries as new reward steps
    for (t, rid), entry in by_type_and_id.items():
        if (t, rid) not in matched_type_id:
            result.append(copy.deepcopy(entry))
    for t, entry in by_type_only.items():
        if t not in matched_type:
            result.append(copy.deepcopy(entry))

    return result


def _apply_env_overrides(base_env_cfg: dict, env_overrides: dict) -> dict:
    """
    Apply env config overrides to *base_env_cfg*.

    The ``rewards`` key receives special list-merge treatment (matched by ``type``).
    All other keys are deep-merged normally.
    """
    if not env_overrides:
        return copy.deepcopy(base_env_cfg)
    result = copy.deepcopy(base_env_cfg)
    for k, v in env_overrides.items():
        if k == "rewards" and isinstance(v, list):
            base_rewards = result.get("rewards") or []
            result["rewards"] = _merge_rewards_list(base_rewards, v)
        elif k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Experiment definition expansion
# ══════════════════════════════════════════════════════════════════════════════


def expand_experiment_definition(config: dict) -> dict:
    """
    Expand an ``experiment_definition`` Cartesian product into a flat
    ``experiments`` list and return the modified config dict.

    Schema::

        experiment_definition:
          - config_set:
              - config:
                  env_overrides: ...
              - config:
                  env_overrides: ...
          - config_set:
              - config:
                  agent_overrides: ...
              - config:
                  agent_overrides: ...

    Each ``config_set`` defines one dimension of the grid.  The Cartesian
    product of all config_sets produces the full experiment list.  Experiments
    are named ``exp_001``, ``exp_002``, … in product order (leftmost
    config_set varies slowest).

    If *config* has an ``experiments`` key but no ``experiment_definition``
    key, it is returned unchanged (backward-compatible pass-through for legacy
    sweep configs and pre-expanded resume configs).

    Raises ``ValueError`` if both keys are present simultaneously.
    """
    has_defn = "experiment_definition" in config
    has_exps = "experiments" in config

    if has_defn and has_exps:
        raise ValueError(
            "Sweep config must not contain both 'experiment_definition' and "
            "'experiments'. Use 'experiment_definition' only."
        )

    if not has_defn:
        # Legacy 'experiments' list or pre-expanded resume config — pass through.
        return config

    defn = config["experiment_definition"]
    if not isinstance(defn, list) or len(defn) == 0:
        raise ValueError(
            "experiment_definition must be a non-empty list of config_set entries."
        )

    config_sets: List[List[dict]] = []
    for i, entry in enumerate(defn):
        if not isinstance(entry, dict) or "config_set" not in entry:
            raise ValueError(
                f"experiment_definition[{i}] must be a mapping with a "
                f"'config_set' key; got {type(entry).__name__}."
            )
        cs = entry["config_set"]
        if not isinstance(cs, list) or len(cs) == 0:
            raise ValueError(
                f"experiment_definition[{i}].config_set must be a "
                f"non-empty list of config entries."
            )
        configs_in_set: List[dict] = []
        for j, item in enumerate(cs):
            if not isinstance(item, dict) or "config" not in item:
                raise ValueError(
                    f"experiment_definition[{i}].config_set[{j}] must be a "
                    f"mapping with a 'config' key."
                )
            cfg = item["config"]
            if not isinstance(cfg, dict):
                raise ValueError(
                    f"experiment_definition[{i}].config_set[{j}].config "
                    f"must be a mapping (got {type(cfg).__name__})."
                )
            configs_in_set.append(cfg)
        config_sets.append(configs_in_set)

    experiments: List[dict] = []
    for idx, combination in enumerate(itertools.product(*config_sets)):
        # Merge override dicts from all configs in this combination.
        # Start from empty dicts — the actual base env/agent configs are
        # applied later by _materialize().
        merged_env: dict = {}
        merged_agent: dict = {}
        cnn_checkpoints: List[str] = []
        for cfg in combination:
            env_ov = cfg.get("env_overrides") or {}
            agent_ov = cfg.get("agent_overrides") or {}
            if env_ov:
                merged_env = _apply_env_overrides(merged_env, env_ov)
            if agent_ov:
                merged_agent = _deep_merge(merged_agent, agent_ov)
            if cfg.get("cnn_checkpoint") is not None:
                cnn_checkpoints.append(cfg["cnn_checkpoint"])

        if len(cnn_checkpoints) > 1:
            raise ValueError(
                f"experiment_definition: combination {idx + 1} has "
                f"{len(cnn_checkpoints)} configs that each set cnn_checkpoint "
                f"— only one config per combination may set cnn_checkpoint. "
                f"Found: {cnn_checkpoints}"
            )

        exp: dict = {"name": f"exp_{idx + 1:03d}"}
        if merged_env:
            exp["env_overrides"] = merged_env
        if merged_agent:
            exp["agent_overrides"] = merged_agent
        if cnn_checkpoints:
            exp["cnn_checkpoint"] = cnn_checkpoints[0]
        experiments.append(exp)

    result = copy.deepcopy(config)
    del result["experiment_definition"]
    result["experiments"] = experiments
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SweepOrchestrator
# ══════════════════════════════════════════════════════════════════════════════


class SweepOrchestrator:
    """
    Orchestrates a sequence of named Train + Eval experiments.

    Typical usage (new sweep):
        orch = SweepOrchestrator(config, config_path, sweep_dir, ...)
        orch.run()

    Resume existing sweep:
        orch = SweepOrchestrator.from_existing(sweep_dir, ...)
        orch.run(from_experiment="my_experiment_name")
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_path: Path,
        sweep_dir: Path,
        isaac_lab_path: str,
        project_root: Path,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.sweep_dir = Path(sweep_dir).resolve()
        self.isaac_lab_path = isaac_lab_path
        self.project_root = Path(project_root).resolve()
        self.task_root = self.project_root / "so101_rl"
        self._state: Optional[Dict[str, Any]] = None

    # ── factory: resume from existing sweep dir ───────────────────────────────
    @classmethod
    def from_existing(
        cls,
        sweep_dir: Path,
        isaac_lab_path: str,
        project_root: Path,
    ) -> "SweepOrchestrator":
        sweep_dir = Path(sweep_dir).resolve()
        config_path = sweep_dir / CONFIG_FILE
        state_path = sweep_dir / STATE_FILE

        if not config_path.is_file():
            _error(f"No {CONFIG_FILE} found in {sweep_dir}")
            sys.exit(1)
        if not state_path.is_file():
            _error(f"No {STATE_FILE} found in {sweep_dir}")
            sys.exit(1)

        with open(config_path) as f:
            config = yaml.safe_load(f)
        config = expand_experiment_definition(config)

        orch = cls(
            config=config,
            config_path=config_path,
            sweep_dir=sweep_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=project_root,
        )
        with open(state_path) as f:
            orch._state = json.load(f)

        _info(f"Loaded existing sweep: {sweep_dir}")
        _info("Experiment status:")
        for exp_name, info in orch._state["experiments"].items():
            status = info["status"]
            colour = (
                _GREEN
                if status == "done"
                else (_RED if status == "failed" else _YELLOW)
            )
            print(f"    {colour}{exp_name}: {status}{_NC}")
        return orch

    # ── validation ────────────────────────────────────────────────────────────
    def validate(self) -> None:
        """
        Collect every configuration problem and report them all at once.
        Exits with code 1 if any errors are found.
        """
        errors: List[str] = []
        cfg = self.config
        base = cfg.get("base") or {}

        # ── top-level required fields ─────────────────────────────────────────
        if not cfg.get("name"):
            errors.append("config.name is required (string)")

        # ── base section ──────────────────────────────────────────────────────
        if not base.get("task"):
            errors.append("config.base.task is required")

        if not base.get("seed") and base.get("seed") != 0:
            errors.append("config.base.seed is required (integer); no default allowed")
        elif base.get("seed") is not None and not isinstance(base["seed"], int):
            errors.append(
                f"config.base.seed must be an integer, got {type(base['seed']).__name__}"
            )

        if base.get("iters") is not None and (
            not isinstance(base["iters"], int) or base["iters"] <= 0
        ):
            errors.append("config.base.iters must be a positive integer when set")
        # iters is optional: when absent, no --max_iterations is passed and
        # trainer.timesteps from the agent config controls training length.

        # env_config
        env_config = base.get("env_config")
        if not env_config:
            errors.append("config.base.env_config is required")
        else:
            p = Path(env_config)
            if not p.is_absolute():
                p = self.project_root / p
            if not p.is_file():
                errors.append(f"config.base.env_config not found: {p.resolve()}")

        # agent_config
        agent_config = base.get("agent_config")
        if not agent_config:
            errors.append("config.base.agent_config is required")
        else:
            p = Path(agent_config)
            if not p.is_absolute():
                p = self.project_root / p
            if not p.is_file():
                errors.append(f"config.base.agent_config not found: {p.resolve()}")

        # ── ISAAC_LAB_PATH ────────────────────────────────────────────────────
        # A placeholder value (e.g. "<ISAAC_LAB_PATH>") is accepted for
        # dry-runs where no live installation is required.
        if not self.isaac_lab_path:
            errors.append("ISAAC_LAB_PATH is not set")
        elif (
            not self.isaac_lab_path.startswith("<")
            and not Path(self.isaac_lab_path).is_dir()
        ):
            errors.append(f"ISAAC_LAB_PATH does not exist: {self.isaac_lab_path}")

        # ── experiment_definition must be expanded before reaching validate() ─
        if "experiment_definition" in cfg:
            errors.append(
                "'experiment_definition' key found during validation. "
                "Call expand_experiment_definition(config) before constructing "
                "SweepOrchestrator — this is a bug in the caller."
            )

        # ── experiments list ──────────────────────────────────────────────────
        experiments = cfg.get("experiments")
        if not experiments:
            errors.append("config.experiments is required and must be a non-empty list")
        elif not isinstance(experiments, list):
            errors.append("config.experiments must be a list")
        else:
            seen_names: set = set()
            for i, exp in enumerate(experiments):
                if not isinstance(exp, dict):
                    errors.append(f"config.experiments[{i}] must be a mapping")
                    continue
                name = exp.get("name")
                if not name:
                    errors.append(f"config.experiments[{i}].name is required")
                elif name in seen_names:
                    errors.append(
                        f"config.experiments[{i}].name '{name}' is duplicated"
                    )
                else:
                    seen_names.add(name)

                # validate reward overrides have 'type' field
                env_ov = exp.get("env_overrides") or {}
                for j, r in enumerate(env_ov.get("rewards") or []):
                    if "type" not in r:
                        errors.append(
                            f"config.experiments[{i}].env_overrides.rewards[{j}] "
                            f"is missing required 'type' field"
                        )

                # validate per-experiment cnn_checkpoint path exists
                exp_ckpt = exp.get("cnn_checkpoint")
                if exp_ckpt is not None:
                    ckpt_path = Path(exp_ckpt)
                    if not ckpt_path.is_absolute():
                        ckpt_path = self.project_root / ckpt_path
                    if not ckpt_path.is_file():
                        errors.append(
                            f"config.experiments[{i}].cnn_checkpoint not found: "
                            f"{ckpt_path.resolve()}"
                        )

        self._report_errors(errors)

    @staticmethod
    def _report_errors(errors: List[str]) -> None:
        if not errors:
            return
        _error(f"Sweep configuration has {len(errors)} error(s):")
        for i, msg in enumerate(errors, 1):
            print(f"  {_RED}[{i}] {msg}{_NC}", file=sys.stderr)
        sys.exit(1)

    # ── experiment naming ─────────────────────────────────────────────────────
    def _exp_dir(self, idx: int, name: str) -> Path:
        """Return the experiment output directory for experiment at index idx."""
        safe_name = name.replace(" ", "_").replace("/", "_")
        return self.sweep_dir / "experiments" / f"{idx + 1:02d}_{safe_name}"

    # ── config materialisation ────────────────────────────────────────────────
    def _load_base_configs(self) -> Tuple[dict, dict]:
        """Load and return (base_env_cfg, base_agent_cfg) as plain dicts."""
        base = self.config["base"]

        env_path = Path(base["env_config"])
        if not env_path.is_absolute():
            env_path = self.project_root / env_path
        with open(env_path) as f:
            base_env_cfg: dict = yaml.safe_load(f) or {}

        agent_path = Path(base["agent_config"])
        if not agent_path.is_absolute():
            agent_path = self.project_root / agent_path
        with open(agent_path) as f:
            base_agent_cfg: dict = yaml.safe_load(f) or {}

        return base_env_cfg, base_agent_cfg

    def _materialize(
        self,
        exp: dict,
        base_env_cfg: dict,
        base_agent_cfg: dict,
    ) -> Tuple[dict, dict, dict]:
        """
        Build (env_cfg, agent_cfg, settings) for a single experiment.

        *settings* is a dict with the effective values for:
          seed, iters, envs, headless, cameras, cnn_checkpoint, task
        """
        base = self.config["base"]

        # effective settings (per-experiment values take priority over base)
        iters = base.get("iters")
        # Compute effective_timesteps for provenance: when iters is set it
        # overrides trainer.timesteps (= iters × rollouts); otherwise
        # trainer.timesteps from the agent config is the authoritative value.
        rollouts = (base_agent_cfg.get("agent") or {}).get("rollouts")
        if iters is not None and rollouts is not None:
            effective_timesteps = iters * rollouts
        else:
            effective_timesteps = (base_agent_cfg.get("trainer") or {}).get("timesteps")
        settings: dict = {
            "task": base["task"],
            "seed": exp.get("seed", base["seed"]),
            "iters": iters,
            "effective_timesteps": effective_timesteps,
            "envs": base.get("envs"),
            "headless": base.get("headless", True),
            "cameras": base.get("cameras", True),
            # Per-experiment cnn_checkpoint takes priority; falls back to base.
            "cnn_checkpoint": exp.get("cnn_checkpoint", base.get("cnn_checkpoint")),
        }

        # apply env overrides
        env_overrides = exp.get("env_overrides") or {}
        env_cfg = _apply_env_overrides(base_env_cfg, env_overrides)

        # apply agent overrides
        agent_overrides = exp.get("agent_overrides") or {}
        agent_cfg = _deep_merge(base_agent_cfg, agent_overrides)

        return env_cfg, agent_cfg, settings

    def _save_materialized_configs(
        self, exp_dir: Path, env_cfg: dict, agent_cfg: dict, settings: dict
    ) -> None:
        """
        Write materialised configs to exp_dir:
          - env_config.yaml      — merged env config (base + overrides)
          - agent_config.yaml    — merged agent config (base + overrides)
          - experiment_settings.yaml — effective CLI-level settings (seed, iters,
                                       envs, cnn_checkpoint, …) for this experiment.
                                       Written before training starts so the dir is
                                       self-documenting even after --create-configs-only.
        """
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "env_config.yaml", "w") as f:
            yaml.safe_dump(env_cfg, f, default_flow_style=False, sort_keys=False)
        with open(exp_dir / "agent_config.yaml", "w") as f:
            yaml.safe_dump(agent_cfg, f, default_flow_style=False, sort_keys=False)

        # Atomically write experiment_settings.yaml so the experiment dir is
        # self-documenting before any training runs.  cnn_checkpoint_provenance.json
        # (written by train.py) supersedes this for post-training provenance.
        settings_path = exp_dir / "experiment_settings.yaml"
        tmp = settings_path.with_suffix(".yaml.tmp")
        with open(tmp, "w") as f:
            yaml.safe_dump(settings, f, default_flow_style=False, sort_keys=False)
        tmp.rename(settings_path)

    # ── command builders ──────────────────────────────────────────────────────
    def _build_train_cmd(self, exp_dir: Path, settings: dict) -> List[str]:
        cmd = [
            f"{self.isaac_lab_path}/isaaclab.sh",
            "-p",
            str(self.task_root / "scripts" / "skrl" / "train.py"),
            "--task",
            settings["task"],
            "--artifacts_dir",
            str(exp_dir),
            "--agent_config",
            str(exp_dir / "agent_config.yaml"),
            "--env_config",
            str(exp_dir / "env_config.yaml"),
            f"hydra.run.dir={exp_dir}/hydra",
            "--seed",
            str(settings["seed"]),
        ]
        if settings.get("iters") is not None:
            cmd += ["--max_iterations", str(settings["iters"])]
        if settings.get("headless"):
            cmd += ["--headless"]
        if settings.get("cameras"):
            cmd += ["--enable_cameras"]
        if settings.get("envs"):
            cmd += ["--num_envs", str(settings["envs"])]
        if settings.get("cnn_checkpoint"):
            cmd += ["--cnn_checkpoint", str(settings["cnn_checkpoint"])]
        return cmd

    def _build_eval_cmd(self, exp_dir: Path) -> List[str]:
        base = self.config["base"]
        eval_cfg = self.config.get("eval") or {}
        cmd = [
            f"{self.isaac_lab_path}/isaaclab.sh",
            "-p",
            str(self.task_root / "scripts" / "skrl" / "evaluate.py"),
            "--experiment-path",
            str(exp_dir),
            "--task",
            base["task"],
        ]
        if eval_cfg.get("episodes") is not None:
            cmd += ["--num-episodes", str(eval_cfg["episodes"])]
        if eval_cfg.get("num_videos") is not None:
            cmd += ["--num-videos", str(eval_cfg["num_videos"])]
        record_videos = eval_cfg.get("record_videos") or {}
        if record_videos.get("wrist_cam"):
            cmd += ["--record-wrist-cam"]
        if record_videos.get("overhead_cam"):
            cmd += ["--record-overhead-cam"]
        if record_videos.get("viewport_cam"):
            cmd += ["--record-viewport-cam"]
        if eval_cfg.get("headless", True):
            cmd += ["--headless"]
        if eval_cfg.get("num_envs") is not None:
            cmd += ["--num_envs", str(eval_cfg["num_envs"])]
        if eval_cfg.get("verbosity"):
            cmd += ["--verbosity", eval_cfg["verbosity"]]
        return cmd

    # ── GUI / subprocess environment ──────────────────────────────────────────
    def _get_gui_env(self, exp_dir: Path) -> Dict[str, str]:
        """Return subprocess env with Isaac Lab workspace and env config configured."""
        base = self.config["base"]
        env = os.environ.copy()
        env["ISAAC_LAB_WORKSPACE_PATH"] = str(
            Path(self.isaac_lab_path) / "workspace" / base["task"]
        )
        env["SO101_ENV_CONFIG"] = str(exp_dir / "env_config.yaml")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    # ── one-time Isaac Lab setup ──────────────────────────────────────────────
    def _stage_assets(self) -> None:
        task = self.config["base"]["task"]
        dest = Path(self.isaac_lab_path) / "workspace" / task / "assets"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.project_root / "assets", dest, dirs_exist_ok=True)
        _success(f"Assets staged to {dest}")

    def _install_task(self) -> None:
        _run_step_subprocess(
            [
                f"{self.isaac_lab_path}/isaaclab.sh",
                "-p",
                "-m",
                "pip",
                "install",
                "-e",
                str(self.task_root / "source" / "so101_rl"),
            ]
        )
        _success("Task package installed.")

    def _check_gpu(self) -> None:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _error("Cannot access NVIDIA GPU. Aborting sweep.")
            sys.exit(1)
        _success(f"GPU: {result.stdout.strip()}")

    # ── state management ──────────────────────────────────────────────────────
    def _init_state(self, experiments: List[dict]) -> Dict[str, Any]:
        exp_state = {}
        for idx, exp in enumerate(experiments):
            exp_dir = self._exp_dir(idx, exp["name"])
            exp_state[exp["name"]] = {
                "index": idx,
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "exp_dir": str(exp_dir),
                "train_log": str(exp_dir / "train.log"),
                "eval_log": str(exp_dir / "eval.log"),
                "train_return_code": None,
                "eval_return_code": None,
            }
        return {
            "sweep_id": self.sweep_dir.name,
            "sweep_dir": str(self.sweep_dir),
            "config_path": str(self.sweep_dir / CONFIG_FILE),
            "experiments": exp_state,
        }

    def _write_state(self) -> None:
        """Write sweep_state.json atomically via a temp file + rename."""
        state_path = self.sweep_dir / STATE_FILE
        tmp = state_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        tmp.rename(state_path)

    # ── artifact verification ─────────────────────────────────────────────────
    def _verify_train_output(self, exp_dir: Path) -> bool:
        """Return True if train.py produced a usable checkpoint.

        Checks both that the skrl/ directory was created and that at least one
        checkpoint file exists within it.  SKRL writes checkpoints only when
        training completes a full rollout cycle, so a missing checkpoint
        indicates a crash or early interrupt rather than a clean run.
        """
        skrl_dir = exp_dir / "skrl"
        if not skrl_dir.is_dir():
            return False
        checkpoints = list(skrl_dir.glob("agent/checkpoints/*.pt"))
        return len(checkpoints) > 0

    def _verify_eval_output(self, exp_dir: Path) -> bool:
        """Return True if evaluation/results.json was created by evaluate.py."""
        return (exp_dir / "evaluation" / "results.json").is_file()

    # ── dry run ───────────────────────────────────────────────────────────────
    def _dry_run(
        self,
        experiments: List[dict],
        base_env_cfg: dict,
        base_agent_cfg: dict,
        from_experiment: Optional[str],
    ) -> None:
        _header(f"DRY RUN — {len(experiments)} experiment(s)")
        print(f"  Sweep dir (preview): {self.sweep_dir}")
        print()

        skip_until = from_experiment
        for idx, exp in enumerate(experiments):
            exp_name = exp["name"]

            # Resume skip logic (same as live run)
            if skip_until is not None:
                if exp_name == skip_until:
                    skip_until = None
                else:
                    already = (
                        (self._state or {}).get("experiments", {}).get(exp_name, {})
                    )
                    if already.get("status") == "done":
                        print(
                            f"{_GREEN}[{idx+1:02d}] {exp_name}: SKIP (already done){_NC}"
                        )
                        continue

            exp_dir = self._exp_dir(idx, exp_name)
            _, _, settings = self._materialize(exp, base_env_cfg, base_agent_cfg)
            train_cmd = self._build_train_cmd(exp_dir, settings)
            eval_cmd = self._build_eval_cmd(exp_dir)

            print(f"{_BLUE}[{idx+1:02d}] {exp_name}{_NC}")
            if exp.get("seed"):
                print(f"  seed           : {settings['seed']} (experiment override)")
            else:
                print(f"  seed           : {settings['seed']}")
            ckpt = settings.get("cnn_checkpoint")
            if ckpt:
                src = "experiment override" if exp.get("cnn_checkpoint") else "base"
                print(f"  cnn_checkpoint : {ckpt} ({src})")
            else:
                print(f"  cnn_checkpoint : None")
            print(f"  exp_dir        : {exp_dir}")
            print(f"  train          : {' '.join(str(c) for c in train_cmd)}")
            print(f"  eval           : {' '.join(str(c) for c in eval_cmd)}")
            print()

    # ── summary generation ────────────────────────────────────────────────────
    def generate_summary(self) -> None:
        """
        Read evaluation/results.json for each done experiment and write:
          - sweep_dir/summary.json  (machine-readable)
          - sweep_dir/summary.md    (human-readable Markdown table)
        """
        rows: List[Dict[str, Any]] = []
        for exp_name, info in self._state["experiments"].items():
            exp_dir = Path(info["exp_dir"])
            results_path = exp_dir / "evaluation" / "results.json"
            status = info["status"]

            row: Dict[str, Any] = {
                "name": exp_name,
                "index": info["index"] + 1,
                "status": status,
                "exp_dir": str(exp_dir),
                # reward stats (from results.json summary_statistics)
                "mean_reward": None,
                "std_reward": None,
                "min_reward": None,
                "max_reward": None,
                "mean_episode_length": None,
                # eval episode stats (from results.json episode_stats)
                "lift_rate": None,
                "drop_rate": None,
                "success_rate": None,
                "mean_cube_bump": None,
                "mean_time_to_lift": None,
                # termination causes (from results.json — Phase B)
                "success_termination_id": None,
                "termination_primary_counts": None,
                "termination_flag_counts": None,
                # training milestones (from milestones.json; env_transitions at first event)
                "milestone_first_approach": None,
                "milestone_first_grasp": None,
                "milestone_first_lift": None,
                "milestone_first_success": None,
            }

            if status == "done" and results_path.is_file():
                try:
                    with open(results_path) as f:
                        results = json.load(f)
                    ss = results.get("summary_statistics") or {}
                    row["mean_reward"] = ss.get("mean_reward")
                    row["std_reward"] = ss.get("std_reward")
                    row["min_reward"] = ss.get("min_reward")
                    row["max_reward"] = ss.get("max_reward")
                    row["mean_episode_length"] = ss.get("mean_episode_length")
                    es = results.get("episode_stats") or {}
                    row["lift_rate"] = es.get("lift_rate")
                    row["drop_rate"] = es.get("drop_rate")
                    row["success_rate"] = es.get("success_rate")
                    row["mean_cube_bump"] = es.get("mean_cube_bump")
                    row["mean_time_to_lift"] = es.get("mean_time_to_lift")
                    row["success_termination_id"] = results.get(
                        "success_termination_id"
                    )
                    row["termination_primary_counts"] = results.get(
                        "termination_primary_counts"
                    )
                    row["termination_flag_counts"] = results.get(
                        "termination_flag_counts"
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    _error(f"Could not read results for '{exp_name}': {exc}")

            # Training milestones — written by MilestoneLog during training.
            milestones_path = exp_dir / "milestones.json"
            if milestones_path.is_file():
                try:
                    with open(milestones_path) as f:
                        milestones = json.load(f)
                    for key in (
                        "first_approach",
                        "first_grasp",
                        "first_lift",
                        "first_success",
                    ):
                        entry = milestones.get(key)
                        if entry is not None:
                            row[f"milestone_{key}"] = entry.get("env_transitions")
                except (json.JSONDecodeError, OSError) as exc:
                    _error(f"Could not read milestones for '{exp_name}': {exc}")

            rows.append(row)

        rows.sort(key=lambda r: r["index"])

        # ── summary.json ──────────────────────────────────────────────────────
        summary = {
            "sweep_id": self._state["sweep_id"],
            "sweep_dir": self._state["sweep_dir"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "experiments": rows,
        }
        summary_json_path = self.sweep_dir / "summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        _success(f"Summary JSON: {summary_json_path}")

        # ── summary.md ────────────────────────────────────────────────────────
        sweep_name = self.config.get("name", self._state["sweep_id"])
        generated_at = summary["generated_at"]
        total = len(rows)
        done = sum(1 for r in rows if r["status"] == "done")
        failed = sum(1 for r in rows if r["status"] == "failed")
        pending = sum(1 for r in rows if r["status"] not in ("done", "failed"))

        def _fmt(v: Optional[float], decimals: int = 3) -> str:
            return f"{v:.{decimals}f}" if v is not None else "—"

        def _pct(v: Optional[float]) -> str:
            return f"{v * 100:.1f}%" if v is not None else "—"

        lines: List[str] = [
            f"# Sweep: {sweep_name}",
            "",
            f"Generated: {generated_at}  ",
            f"Sweep dir: `{self._state['sweep_dir']}`  ",
            f"Experiments: {total} total — {done} done, {failed} failed, {pending} pending",
            "",
            "## Results",
            "",
            "| # | Experiment | Status | Mean Reward | Std Reward | Lift Rate | Drop Rate | Success Rate | Cube Bump | Time to Lift | Approach | Grasp | Lift | Success |",
            "|---|-----------|--------|-------------|------------|-----------|-----------|--------------|-----------|--------------|----------|-------|------|---------|",
        ]
        for r in rows:
            status_fmt = r["status"]
            lines.append(
                f"| {r['index']} "
                f"| {r['name']} "
                f"| {status_fmt} "
                f"| {_fmt(r['mean_reward'])} "
                f"| {_fmt(r['std_reward'])} "
                f"| {_pct(r['lift_rate'])} "
                f"| {_pct(r['drop_rate'])} "
                f"| {_pct(r['success_rate'])} "
                f"| {_fmt(r['mean_cube_bump'], 4)} "
                f"| {_fmt(r['mean_time_to_lift'], 1)} "
                f"| {_fmt(r['milestone_first_approach'], 0)} "
                f"| {_fmt(r['milestone_first_grasp'], 0)} "
                f"| {_fmt(r['milestone_first_lift'], 0)} "
                f"| {_fmt(r['milestone_first_success'], 0)} |"
            )

        lines += [
            "",
            "## Paths",
            "",
            f"- **Sweep dir:** `{self._state['sweep_dir']}`",
            f"- **TensorBoard:** `tensorboard --logdir {self._state['sweep_dir']}/experiments`",
            "",
            "## Experiment Details",
            "",
        ]
        for r in rows:
            lines.append(f"### {r['index']:02d}. {r['name']} ({r['status']})")
            lines.append(f"- Dir: `{r['exp_dir']}`")
            if r["mean_reward"] is not None:
                lines.append(
                    f"- Mean reward: {_fmt(r['mean_reward'])} ± {_fmt(r['std_reward'])}"
                )
                lines.append(
                    f"- Reward range: [{_fmt(r['min_reward'])}, {_fmt(r['max_reward'])}]"
                )
                lines.append(
                    f"- Mean episode length: {_fmt(r['mean_episode_length'], 1)}"
                )
            if r["lift_rate"] is not None:
                lines.append(
                    f"- Eval stats: lift {_pct(r['lift_rate'])} | drop {_pct(r['drop_rate'])} "
                    f"| success {_pct(r['success_rate'])} | bump {_fmt(r['mean_cube_bump'], 4)} "
                    f"| time_to_lift {_fmt(r['mean_time_to_lift'], 1)}"
                )
            if r["termination_primary_counts"]:
                _total = sum(r["termination_primary_counts"].values()) or 1
                _parts = ", ".join(
                    f"{cause} {count} ({count / _total * 100:.1f}%)"
                    for cause, count in sorted(
                        r["termination_primary_counts"].items(),
                        key=lambda kv: -kv[1],
                    )
                )
                _success_marker = (
                    f" (success={r['success_termination_id']})"
                    if r.get("success_termination_id")
                    else ""
                )
                lines.append(
                    f"- Primary termination causes{_success_marker}: {_parts}"
                )
            if r["milestone_first_approach"] is not None:
                lines.append(
                    f"- Milestones (env_transitions): approach={r['milestone_first_approach']:,} "
                    f"grasp={r['milestone_first_grasp']:,} "
                    f"lift={r['milestone_first_lift']:,} "
                    f"success={r['milestone_first_success']:,}"
                )
            lines.append("")

        summary_md_path = self.sweep_dir / "summary.md"
        with open(summary_md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        _success(f"Summary Markdown: {summary_md_path}")

    # ── materialize only (no training) ────────────────────────────────────────
    def materialize_all(self) -> None:
        """
        Create the sweep directory and write materialised ``env_config.yaml`` and
        ``agent_config.yaml`` for every experiment, then exit without launching
        any training or evaluation.

        Used by ``--create-configs-only`` to inspect the full set of configs that
        a sweep would use before committing to a long run.
        """
        self.validate()
        experiments = self.config["experiments"]
        base_env_cfg, base_agent_cfg = self._load_base_configs()

        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        (self.sweep_dir / "experiments").mkdir(exist_ok=True)

        config_dest = self.sweep_dir / CONFIG_FILE
        if not config_dest.exists():
            with open(config_dest, "w") as f:
                yaml.safe_dump(
                    self.config, f, default_flow_style=False, sort_keys=False
                )
            _success(f"Sweep config saved: {config_dest}")

        _header(
            f"Creating configs — {len(experiments)} experiment(s) → {self.sweep_dir}"
        )
        for idx, exp in enumerate(experiments):
            exp_dir = self._exp_dir(idx, exp["name"])
            env_cfg, agent_cfg, settings = self._materialize(
                exp, base_env_cfg, base_agent_cfg
            )
            self._save_materialized_configs(exp_dir, env_cfg, agent_cfg, settings)
            ckpt = settings.get("cnn_checkpoint") or "none"
            _success(f"  [{idx + 1:02d}] {exp['name']}: {exp_dir}")
            _info(f"       cnn_checkpoint: {ckpt}")

        print()
        _success(
            f"All {len(experiments)} config(s) materialised — no training launched."
        )
        _success(f"Sweep dir: {self.sweep_dir}")

    # ── main run loop ──────────────────────────────────────────────────────────
    def run(
        self,
        from_experiment: Optional[str] = None,
        dry_run: bool = False,
        retry_eval: bool = False,
    ) -> None:
        """
        Run the sweep.

        Args:
            from_experiment: If set, skip all experiments whose name comes
                before this one alphabetically in order (by index) AND that
                are already marked "done".  Use to restart from a specific
                experiment.
            dry_run: Print commands without executing anything.
            retry_eval: When resuming, skip the training subprocess for any
                experiment whose stored train_return_code is already 0 and
                the checkpoint is present on disk.  Useful when all experiments
                trained successfully but eval failed.
        """
        self.validate()

        experiments = self.config["experiments"]
        base_env_cfg, base_agent_cfg = self._load_base_configs()

        if dry_run:
            self._dry_run(experiments, base_env_cfg, base_agent_cfg, from_experiment)
            return

        # ── set up sweep directory ────────────────────────────────────────────
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        (self.sweep_dir / "experiments").mkdir(exist_ok=True)

        # Save expanded sweep config for reproducibility (only on first run).
        # Write self.config (already expanded) rather than copying the original
        # so that resume always loads the ready-to-use 'experiments' list.
        config_dest = self.sweep_dir / CONFIG_FILE
        if not config_dest.exists():
            with open(config_dest, "w") as f:
                yaml.safe_dump(
                    self.config, f, default_flow_style=False, sort_keys=False
                )
            _success(f"Sweep config saved: {config_dest}")

        # Initialise state (preserve existing state on resume)
        if self._state is None:
            self._state = self._init_state(experiments)
            self._write_state()

        # ── one-time Isaac Lab setup ──────────────────────────────────────────
        self._check_gpu()
        self._stage_assets()
        self._install_task()

        total = len(experiments)
        _header(
            f"Sweep: {self.config.get('name', self.sweep_dir.name)} "
            f"— {total} experiment(s)"
        )

        # Determine where to start: if from_experiment is set, find its index
        # and reset skip-tracking accordingly.
        from_idx: int = 0
        if from_experiment is not None:
            for i, exp in enumerate(experiments):
                if exp["name"] == from_experiment:
                    from_idx = i
                    break
            else:
                _error(
                    f"--from-experiment '{from_experiment}' not found in sweep config. "
                    f"Available: {[e['name'] for e in experiments]}"
                )
                sys.exit(1)

        # ── experiment loop ───────────────────────────────────────────────────
        completed_count = 0
        for idx, exp in enumerate(experiments):
            exp_name = exp["name"]
            exp_state = self._state["experiments"].get(exp_name, {})

            # Skip experiments that are already done and come before from_idx
            if idx < from_idx and exp_state.get("status") == "done":
                _info(f"[{idx+1:02d}/{total}] '{exp_name}': skipping (already done)")
                completed_count += 1
                continue

            _header(f"[{idx+1:02d}/{total}] Experiment: {exp_name}")

            exp_dir = self._exp_dir(idx, exp_name)
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Materialise configs
            env_cfg, agent_cfg, settings = self._materialize(
                exp, base_env_cfg, base_agent_cfg
            )
            self._save_materialized_configs(exp_dir, env_cfg, agent_cfg, settings)
            _success(f"Configs materialised → {exp_dir}")

            # Update state: running
            now = datetime.now(timezone.utc).isoformat()
            self._state["experiments"][exp_name].update(
                {
                    "status": "running",
                    "started_at": now,
                    "exp_dir": str(exp_dir),
                    "train_log": str(exp_dir / "train.log"),
                    "eval_log": str(exp_dir / "eval.log"),
                }
            )
            self._write_state()

            gui_env = self._get_gui_env(exp_dir)

            # ── train ─────────────────────────────────────────────────────────
            _header(f"  [{idx+1:02d}] Training: {exp_name}")
            train_cmd = self._build_train_cmd(exp_dir, settings)
            train_log = exp_dir / "train.log"

            # --retry-eval: skip training when a previous run already succeeded
            # and the checkpoint is still present on disk.
            _skipped_training = False
            stored_train_rc = exp_state.get("train_return_code")
            if retry_eval and stored_train_rc == 0:
                if self._verify_train_output(exp_dir):
                    _info(
                        f"  [{idx+1:02d}] Skipping training for '{exp_name}' "
                        f"(train_return_code=0 and checkpoint present)."
                    )
                    _skipped_training = True
                    train_rc = 0
                else:
                    _error(
                        f"  [{idx+1:02d}] --retry-eval: stored train_return_code=0 for '{exp_name}' "
                        f"but checkpoint is missing — re-running training."
                    )

            if not _skipped_training:
                _info(f"Command: {' '.join(str(c) for c in train_cmd)}")
                train_rc = _run_step_subprocess(
                    train_cmd, env=gui_env, log_path=train_log
                )

                # Isaac Sim can exit 0 despite a crash; verify the skrl/ dir was created
                if train_rc == 0 and not self._verify_train_output(exp_dir):
                    _error(
                        f"train.py reported exit 0 but skrl/ output is missing in {exp_dir}. "
                        f"Isaac Sim likely crashed silently. Check: {train_log}"
                    )
                    train_rc = 1

            self._state["experiments"][exp_name]["train_return_code"] = train_rc

            if train_rc != 0:
                now = datetime.now(timezone.utc).isoformat()
                self._state["experiments"][exp_name].update(
                    {"status": "failed", "completed_at": now}
                )
                self._write_state()
                _error(f"Train step failed for '{exp_name}' (exit code {train_rc})")
                _error(f"Log: {train_log}")
                _info("Continuing to next experiment...")
                continue

            _success(f"Train complete: {exp_name}")

            # ── eval ──────────────────────────────────────────────────────────
            _header(f"  [{idx+1:02d}] Evaluating: {exp_name}")
            eval_cmd = self._build_eval_cmd(exp_dir)
            _info(f"Command: {' '.join(str(c) for c in eval_cmd)}")
            eval_log = exp_dir / "eval.log"
            eval_rc = _run_step_subprocess(eval_cmd, env=gui_env, log_path=eval_log)

            if eval_rc == 0 and not self._verify_eval_output(exp_dir):
                _error(
                    f"evaluate.py reported exit 0 but evaluation/results.json is missing. "
                    f"Check: {eval_log}"
                )
                eval_rc = 1

            self._state["experiments"][exp_name]["eval_return_code"] = eval_rc

            now = datetime.now(timezone.utc).isoformat()
            if eval_rc == 0:
                self._state["experiments"][exp_name].update(
                    {"status": "done", "completed_at": now}
                )
                self._write_state()
                _success(f"Experiment '{exp_name}' complete → {exp_dir}")
                completed_count += 1
            else:
                self._state["experiments"][exp_name].update(
                    {"status": "failed", "completed_at": now}
                )
                self._write_state()
                _error(f"Eval step failed for '{exp_name}' (exit code {eval_rc})")
                _error(f"Log: {eval_log}")
                _info("Continuing to next experiment...")

        # ── generate summary ──────────────────────────────────────────────────
        _header("Sweep complete — generating summary")
        self.generate_summary()

        done_count = sum(
            1
            for info in self._state["experiments"].values()
            if info["status"] == "done"
        )
        failed_count = sum(
            1
            for info in self._state["experiments"].values()
            if info["status"] == "failed"
        )
        _success(
            f"Sweep finished: {done_count}/{total} done, {failed_count}/{total} failed"
        )
        _success(f"Sweep dir: {self.sweep_dir}")
        _success(f"Summary:   {self.sweep_dir / 'summary.md'}")
