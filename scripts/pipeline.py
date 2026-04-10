#!/usr/bin/env python3
"""
pipeline.py — PipelineOrchestrator for the so101_sim_to_real training pipeline.

Steps (in order):
  1. train      — RL policy training       (requires Isaac Lab)
  2. collect    — Telemetry collection     (requires Isaac Lab)
  3. curate     — Dataset curation         (offline)
  4. train-cnn  — CNN backbone training    (offline)

Directory layout produced:
  artifacts/pipeline_YYYYMMDD_HHMMSS/
    pipeline.yaml            ← resolved config, copied before any step runs
    pipeline_state.json      ← live step status, written atomically after each step
    logs/
      01_train.log
      02_collect.log
      03_curate.log
      04_train_cnn.log
    01_train/                ← env_config.yaml copy + hydra/ + skrl/
    02_collect/              ← NPZ shards + telemetry_metadata.json
    03_curate/               ← manifests + curation_report.json
    04_train_cnn/            ← checkpoints/ + report.json + tensorboard/
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# ── step constants ─────────────────────────────────────────────────────────────
STEP_ORDER: List[str] = ["train", "collect", "curate", "train-cnn"]

STEP_DIR: Dict[str, str] = {
    "train": "01_train",
    "collect": "02_collect",
    "curate": "03_curate",
    "train-cnn": "04_train_cnn",
}

LOG_NAME: Dict[str, str] = {
    "train": "01_train.log",
    "collect": "02_collect.log",
    "curate": "03_curate.log",
    "train-cnn": "04_train_cnn.log",
}

STATE_FILE = "pipeline_state.json"
CONFIG_FILE = "pipeline.yaml"

# ── terminal output ────────────────────────────────────────────────────────────
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
    bar = "=" * 52
    print(f"{_BLUE}{bar}{_NC}")
    print(f"{_BLUE}{msg}{_NC}")
    print(f"{_BLUE}{bar}{_NC}")


# ── episode progress bar (used by the tee loop for the collect step) ──────────
_COLLECT_PROGRESS_RE = re.compile(r"\[INFO\] Completed (\d+)/(\d+) episodes")
_PROGRESS_BAR_WIDTH = 40


def _render_collect_progress(line: str) -> bool:
    """
    If *line* is a collect-progress line, print a compact progress bar to stdout
    and return True (suppressing the raw line from terminal output).
    Returns False for all other lines.
    """
    m = _COLLECT_PROGRESS_RE.search(line)
    if not m:
        return False
    done, total = int(m.group(1)), int(m.group(2))
    pct = done / total if total > 0 else 0.0
    filled = int(_PROGRESS_BAR_WIDTH * pct)
    bar = "\u2588" * filled + "\u2591" * (_PROGRESS_BAR_WIDTH - filled)
    sys.stdout.write(f"{_GREEN}  [collect] [{bar}] {done}/{total} ({pct:.0%}){_NC}\n")
    sys.stdout.flush()
    return True


# ── subprocess tee helper ──────────────────────────────────────────────────────
def _run_step_subprocess(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    line_interceptor=None,
) -> int:
    """
    Run a command, tee-ing stdout+stderr to terminal and log file.

    If *line_interceptor* is provided it is called with each output line.
    When it returns True the raw line is suppressed from terminal output
    (but still written to the log file).

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
            if line_interceptor and line_interceptor(line):
                pass  # interceptor handled terminal output
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
        rc = proc.returncode
    finally:
        if log_fh:
            log_fh.close()
    return rc


# ════════════════════════════════════════════════════════════════════════════════
# PipelineOrchestrator
# ════════════════════════════════════════════════════════════════════════════════


class PipelineOrchestrator:
    """
    Orchestrates the train → collect → curate → train-cnn pipeline.

    Typical usage (new run):
        orch = PipelineOrchestrator(config, config_path, pipeline_dir, ...)
        orch.run(from_step="train", to_step="train-cnn")

    Resume existing run:
        orch = PipelineOrchestrator.from_existing(pipeline_dir, ...)
        orch.run(from_step="train-cnn")
    """

    def __init__(
        self,
        config: Dict[str, Any],
        config_path: Path,
        pipeline_dir: Path,
        isaac_lab_path: str,
        project_root: Path,
        ad_hoc_experiment: Optional[str] = None,
        ad_hoc_input: Optional[str] = None,
        force_display: bool = False,
    ) -> None:
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.pipeline_dir = Path(pipeline_dir).resolve()
        self.isaac_lab_path = isaac_lab_path
        self.project_root = Path(project_root).resolve()
        self.task_root = self.project_root / "so101_rl"
        self.ad_hoc_experiment = (
            Path(ad_hoc_experiment).resolve() if ad_hoc_experiment else None
        )
        self.ad_hoc_input = Path(ad_hoc_input).resolve() if ad_hoc_input else None
        # When True, --headless is suppressed for Isaac steps (--display was supplied)
        self.force_display = force_display
        self._state: Optional[Dict[str, Any]] = None

    # ── factory: resume from existing pipeline dir ────────────────────────────
    @classmethod
    def from_existing(
        cls,
        pipeline_dir: Path,
        isaac_lab_path: str,
        project_root: Path,
    ) -> "PipelineOrchestrator":
        pipeline_dir = Path(pipeline_dir).resolve()
        config_path = pipeline_dir / CONFIG_FILE
        state_path = pipeline_dir / STATE_FILE

        if not config_path.is_file():
            _error(f"No {CONFIG_FILE} found in {pipeline_dir}")
            sys.exit(1)
        if not state_path.is_file():
            _error(f"No {STATE_FILE} found in {pipeline_dir}")
            sys.exit(1)

        with open(config_path) as f:
            config = yaml.safe_load(f)

        orch = cls(
            config=config,
            config_path=config_path,
            pipeline_dir=pipeline_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=project_root,
        )
        with open(state_path) as f:
            orch._state = json.load(f)
        _info(f"Loaded existing pipeline: {pipeline_dir}")
        return orch

    # ── upfront multi-error validation ────────────────────────────────────────
    def validate(self, from_step: str, to_step: str) -> None:
        """
        Collect every configuration problem and report them all at once.
        Exits with code 1 if any errors are found.
        """
        errors: List[str] = []

        # Step name / order validity
        if from_step not in STEP_ORDER:
            errors.append(
                f"--from '{from_step}' is not a valid step. Valid: {STEP_ORDER}"
            )
        if to_step not in STEP_ORDER:
            errors.append(f"--to '{to_step}' is not a valid step. Valid: {STEP_ORDER}")
        if from_step in STEP_ORDER and to_step in STEP_ORDER:
            if STEP_ORDER.index(from_step) > STEP_ORDER.index(to_step):
                errors.append(
                    f"--from '{from_step}' comes after --to '{to_step}' in step order"
                )

        # ISAAC_LAB_PATH required for Isaac-dependent steps
        if not self.isaac_lab_path:
            errors.append("ISAAC_LAB_PATH is not set")
        elif not Path(self.isaac_lab_path).is_dir():
            errors.append(f"ISAAC_LAB_PATH does not exist: {self.isaac_lab_path}")

        # ── conda env must have isaacsim installed (isaaclab.sh uses CONDA_PREFIX) ──
        isaac_steps_needed = any(
            s
            in [from_step, to_step]
            + STEP_ORDER[
                STEP_ORDER.index(from_step) if from_step in STEP_ORDER else 0 : (
                    STEP_ORDER.index(to_step) + 1
                    if to_step in STEP_ORDER
                    else len(STEP_ORDER)
                )
            ]
            for s in ("train", "collect")
        )
        if isaac_steps_needed:
            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                python_exe = Path(conda_prefix) / "bin" / "python"
                if python_exe.is_file():
                    check = subprocess.run(
                        [str(python_exe), "-c", "import isaacsim"],
                        capture_output=True,
                    )
                    if check.returncode != 0:
                        env_name = Path(conda_prefix).name
                        errors.append(
                            f"Active conda env '{env_name}' does not have 'isaacsim' installed. "
                            f"isaaclab.sh will use {python_exe} (from CONDA_PREFIX). "
                            f"Activate the Isaac Lab conda env first, e.g.: conda activate env_isaaclab"
                        )

        cfg = self.config
        if from_step not in STEP_ORDER or to_step not in STEP_ORDER:
            # Can't do further range-based checks with invalid step names
            self._report_errors(errors)
            return

        steps_in_range = STEP_ORDER[
            STEP_ORDER.index(from_step) : STEP_ORDER.index(to_step) + 1
        ]

        # ── seed (required if collect, curate, or train-cnn is in range) ──────
        seed_needed = any(
            s in steps_in_range for s in ("collect", "curate", "train-cnn")
        )
        if seed_needed:
            if cfg.get("seed") is None:
                errors.append(
                    "config.seed is required (integer) for collect / curate / train-cnn steps"
                )
            elif not isinstance(cfg["seed"], int):
                errors.append(
                    f"config.seed must be an integer, got {type(cfg['seed']).__name__}"
                )

        # ── task (required for train + collect) ──────────────────────────────
        if any(s in steps_in_range for s in ("train", "collect")):
            if not cfg.get("task"):
                errors.append("config.task is required")

        # ── train-step checks ─────────────────────────────────────────────────
        if "train" in steps_in_range:
            env_config = cfg.get("env_config")
            if not env_config:
                errors.append("config.env_config is required for the train step")
            else:
                p = Path(env_config)
                if not p.is_absolute():
                    p = self.project_root / p
                if not p.is_file():
                    errors.append(f"config.env_config not found: {p.resolve()}")

        # ── collect-step checks ───────────────────────────────────────────────
        if "collect" in steps_in_range:
            collect_cfg = cfg.get("collect") or {}
            if not collect_cfg.get("sample_interval"):
                errors.append("config.collect.sample_interval is required")
            if not collect_cfg.get("episodes"):
                errors.append("config.collect.episodes is required")
            # Ad-hoc start: experiment dir must be supplied
            if from_step == "collect" and self._state is None:
                if not self.ad_hoc_experiment:
                    errors.append(
                        "--experiment is required when --from collect without --pipeline-dir"
                    )
                elif not self.ad_hoc_experiment.is_dir():
                    errors.append(
                        f"--experiment directory not found: {self.ad_hoc_experiment}"
                    )

        # ── curate-step checks ────────────────────────────────────────────────
        if "curate" in steps_in_range:
            self._validate_cnn_config(errors)
            if from_step == "curate" and self._state is None:
                if not self.ad_hoc_input:
                    errors.append(
                        "--input is required when --from curate without --pipeline-dir"
                    )
                elif not self.ad_hoc_input.is_dir():
                    errors.append(f"--input directory not found: {self.ad_hoc_input}")

        # ── train-cnn-step checks ─────────────────────────────────────────────
        if "train-cnn" in steps_in_range:
            self._validate_cnn_config(errors)
            if from_step == "train-cnn" and self._state is None:
                if not self.ad_hoc_input:
                    errors.append(
                        "--input is required when --from train-cnn without --pipeline-dir"
                    )
                elif not self.ad_hoc_input.is_dir():
                    errors.append(f"--input directory not found: {self.ad_hoc_input}")

        self._report_errors(errors)

    def _validate_cnn_config(self, errors: List[str]) -> None:
        """Append an error if config.cnn.config is missing or the file is not found."""
        cnn_cfg = self.config.get("cnn") or {}
        cnn_config = cnn_cfg.get("config")
        _msg_required = "config.cnn.config is required for curate / train-cnn steps"
        if not cnn_config:
            if _msg_required not in errors:
                errors.append(_msg_required)
        else:
            p = Path(cnn_config)
            if not p.is_absolute():
                p = self.project_root / p
            if not p.is_file():
                msg = f"config.cnn.config not found: {p.resolve()}"
                if msg not in errors:
                    errors.append(msg)

    @staticmethod
    def _report_errors(errors: List[str]) -> None:
        if not errors:
            return
        _error(f"Pipeline configuration has {len(errors)} error(s):")
        for i, msg in enumerate(errors, 1):
            print(f"  {_RED}[{i}] {msg}{_NC}", file=sys.stderr)
        sys.exit(1)

    # ── I/O resolution ────────────────────────────────────────────────────────
    def _step_output_dir(self, step: str) -> Path:
        return self.pipeline_dir / STEP_DIR[step]

    def _resolve_step_input(self, step: str) -> Optional[Path]:
        """Resolve the input directory for a step from prior state or ad-hoc flags."""
        step_idx = STEP_ORDER.index(step)
        if step_idx == 0:
            return None  # train has no input dir

        upstream = STEP_ORDER[step_idx - 1]

        # Prefer completed upstream output from state
        if self._state:
            upstream_info = self._state["steps"].get(upstream, {})
            if (
                upstream_info.get("output_dir")
                and upstream_info.get("status") == "completed"
            ):
                return Path(upstream_info["output_dir"])

        # Ad-hoc overrides for the starting step
        if step == "collect" and self.ad_hoc_experiment:
            return self.ad_hoc_experiment
        if step in ("curate", "train-cnn") and self.ad_hoc_input:
            return self.ad_hoc_input

        return None

    # ── offline Python resolver ───────────────────────────────────────────────
    def _resolve_python_cmd(self) -> List[str]:
        cnn = self.config.get("cnn") or {}
        python_exe = cnn.get("python")
        conda_env = cnn.get("conda_env")
        if python_exe:
            p = Path(python_exe)
            if not p.is_absolute():
                p = self.project_root / p
            return [str(p)]
        if conda_env:
            return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
        return ["python"]

    # ── X11 / GUI environment ─────────────────────────────────────────────────
    def _get_gui_env(
        self,
        workspace_path: Path,
        staged_config: Optional[Path] = None,
    ) -> Dict[str, str]:
        env = os.environ.copy()
        env["ISAAC_LAB_WORKSPACE_PATH"] = str(workspace_path)
        if staged_config:
            env["SO101_ENV_CONFIG"] = str(staged_config)
        # Force Python to flush stdout even when writing to a pipe, so that
        # progress prints from collect_telemetry.py reach the terminal promptly.
        env["PYTHONUNBUFFERED"] = "1"
        return env

    # ── staging helpers ───────────────────────────────────────────────────────
    def _stage_assets(self) -> None:
        task = self.config["task"]
        dest = Path(self.isaac_lab_path) / "workspace" / task / "assets"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.project_root / "assets", dest, dirs_exist_ok=True)
        _success(f"Assets staged to {dest}")

    def _stage_env_config(self) -> Optional[Path]:
        env_config = self.config.get("env_config")
        if not env_config:
            return None
        task = self.config["task"]
        src = Path(env_config)
        if not src.is_absolute():
            src = self.project_root / src
        src = src.resolve()
        dest_dir = Path(self.isaac_lab_path) / "workspace" / task / "configs"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.copy2(src, dest)
        _success(f"Env config staged: {dest}")
        return dest

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
            _error("Cannot access NVIDIA GPU.")
            sys.exit(1)
        _success(f"GPU: {result.stdout.strip()}")

    # ── command builders ──────────────────────────────────────────────────────
    def _build_train_cmd(
        self,
        output_dir: Path,
        staged_cfg: Optional[Path],
    ) -> List[str]:
        cfg = self.config
        cmd = [
            f"{self.isaac_lab_path}/isaaclab.sh",
            "-p",
            str(self.task_root / "scripts" / "skrl" / "train.py"),
            "--task",
            cfg["task"],
            "--artifacts_dir",
            str(output_dir),
            f"hydra.run.dir={output_dir}/hydra",
        ]
        if cfg.get("headless") and not self.force_display:
            cmd += ["--headless"]
        if cfg.get("cameras"):
            cmd += ["--enable_cameras"]
        if cfg.get("envs"):
            cmd += ["--num_envs", str(cfg["envs"])]
        if cfg.get("seed") is not None:
            cmd += ["--seed", str(cfg["seed"])]
        return cmd

    def _build_collect_cmd(
        self,
        experiment_dir: Path,
        output_dir: Path,
    ) -> List[str]:
        cfg = self.config
        collect = cfg.get("collect") or {}
        cmd = [
            f"{self.isaac_lab_path}/isaaclab.sh",
            "-p",
            str(self.task_root / "scripts" / "skrl" / "collect_telemetry.py"),
            "--experiment-path",
            str(experiment_dir),
            "--task",
            cfg["task"],
            "--sample-every-steps",
            str(collect["sample_interval"]),
            "--num-episodes",
            str(collect["episodes"]),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(cfg["seed"]),
        ]
        if collect.get("shard_size"):
            cmd += ["--samples-per-shard", str(collect["shard_size"])]
        if cfg.get("headless") and not self.force_display:
            cmd += ["--headless"]
        envs = collect.get("envs") or cfg.get("envs")
        if envs:
            cmd += ["--num_envs", str(envs)]
        return cmd

    def _build_curate_cmd(self, input_dir: Path, output_dir: Path) -> List[str]:
        cfg = self.config
        cnn = cfg.get("cnn") or {}
        cnn_config = Path(cnn["config"])
        if not cnn_config.is_absolute():
            cnn_config = self.project_root / cnn_config
        cmd = self._resolve_python_cmd() + [
            str(self.project_root / "so101" / "curate" / "curate.py"),
            "--telemetry-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--config",
            str(cnn_config),
        ]
        if cfg.get("seed") is not None:
            cmd += ["--seed", str(cfg["seed"])]
        return cmd

    def _build_train_cnn_cmd(self, input_dir: Path, output_dir: Path) -> List[str]:
        cfg = self.config
        cnn = cfg.get("cnn") or {}
        cnn_config = Path(cnn["config"])
        if not cnn_config.is_absolute():
            cnn_config = self.project_root / cnn_config
        cmd = self._resolve_python_cmd() + [
            str(self.project_root / "so101" / "train_cnn.py"),
            "--curated-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--config",
            str(cnn_config),
        ]
        if cnn.get("device"):
            cmd += ["--device", cnn["device"]]
        if cfg.get("seed") is not None:
            cmd += ["--seed", str(cfg["seed"])]
        return cmd

    # ── state management ──────────────────────────────────────────────────────
    def _init_state(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_dir.name,
            "pipeline_dir": str(self.pipeline_dir),
            "config_path": str(self.pipeline_dir / CONFIG_FILE),
            "steps": {
                s: {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "output_dir": None,
                    "return_code": None,
                    "log": str(self.pipeline_dir / "logs" / LOG_NAME[s]),
                }
                for s in STEP_ORDER
            },
        }

    def _write_state(self) -> None:
        """Write pipeline_state.json atomically via a temp file + rename."""
        state_path = self.pipeline_dir / STATE_FILE
        tmp = state_path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._state, f, indent=2)
        tmp.rename(state_path)

    # ── artifact verification (guards against Isaac Sim's spurious exit-0) ───
    # Expected sentinel files: their absence means the step silently crashed.
    _STEP_SENTINELS: Dict[str, str] = {
        "train": "skrl",  # directory created by the skrl trainer
        "collect": "telemetry_metadata.json",
    }

    def _verify_step_output(self, step: str, output_dir: Path) -> Tuple[int, Path]:
        """Return (0, output_dir) if expected sentinel exists, (1, output_dir) otherwise."""
        sentinel = self._STEP_SENTINELS.get(step)
        if sentinel is None:
            return 0, output_dir
        if (output_dir / sentinel).exists():
            return 0, output_dir
        _error(
            f"Step '{step}' reported exit code 0 but expected output is missing: "
            f"{output_dir / sentinel}"
        )
        _error("Isaac Sim likely crashed silently. Check the log for details.")
        return 1, output_dir

    # ── dry run ───────────────────────────────────────────────────────────────
    def _dry_run(self, steps_to_run: List[str]) -> None:
        preview_dir = self.pipeline_dir
        _header(f"DRY RUN — {len(steps_to_run)} step(s)")
        print(f"    Pipeline dir (preview): {preview_dir}")
        print()

        sim_output: Dict[str, Path] = {
            s: preview_dir / STEP_DIR[s] for s in steps_to_run
        }

        for i, step in enumerate(steps_to_run, 1):
            output_dir = sim_output[step]

            if step == "train":
                cmd = self._build_train_cmd(output_dir, staged_cfg=None)
                input_display = "—"
            elif step == "collect":
                if self.ad_hoc_experiment:
                    exp_dir = self.ad_hoc_experiment
                else:
                    exp_dir = sim_output.get("train", preview_dir / STEP_DIR["train"])
                cmd = self._build_collect_cmd(exp_dir, output_dir)
                input_display = str(exp_dir)
            elif step == "curate":
                if self.ad_hoc_input and "collect" not in steps_to_run:
                    in_dir = self.ad_hoc_input
                else:
                    in_dir = sim_output.get(
                        "collect", preview_dir / STEP_DIR["collect"]
                    )
                cmd = self._build_curate_cmd(in_dir, output_dir)
                input_display = str(in_dir)
            elif step == "train-cnn":
                if self.ad_hoc_input and "curate" not in steps_to_run:
                    in_dir = self.ad_hoc_input
                else:
                    in_dir = sim_output.get("curate", preview_dir / STEP_DIR["curate"])
                cmd = self._build_train_cnn_cmd(in_dir, output_dir)
                input_display = str(in_dir)

            cmd_str = " ".join(str(c) for c in cmd)
            print(f"{_BLUE}[{i}] {step}{_NC}")
            print(f"    cmd    : {cmd_str}")
            print(f"    input  : {input_display}")
            print(f"    output : {output_dir}")
            print()

    # ── main run loop ──────────────────────────────────────────────────────────
    def run(
        self,
        from_step: Optional[str] = None,
        to_step: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        from_step = from_step or STEP_ORDER[0]
        to_step = to_step or STEP_ORDER[-1]

        # Validate all inputs upfront — exits on any error
        self.validate(from_step, to_step)

        from_idx = STEP_ORDER.index(from_step)
        to_idx = STEP_ORDER.index(to_step)
        steps_to_run = STEP_ORDER[from_idx : to_idx + 1]

        if dry_run:
            self._dry_run(steps_to_run)
            return

        # ── Set up pipeline directory ─────────────────────────────────────────
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        (self.pipeline_dir / "logs").mkdir(exist_ok=True)

        # Save config for reproducibility (only on first run)
        config_dest = self.pipeline_dir / CONFIG_FILE
        if not config_dest.exists():
            shutil.copy2(self.config_path, config_dest)
            _success(f"Config saved: {config_dest}")

        # Initialise state
        if self._state is None:
            self._state = self._init_state()
            self._write_state()

        _success(f"Pipeline directory: {self.pipeline_dir}")
        _header(f"Pipeline: {' → '.join(steps_to_run)}")

        # ── One-time Isaac setup for Isaac-dependent steps ────────────────────
        isaac_steps = [s for s in steps_to_run if s in ("train", "collect")]
        if isaac_steps:
            self._check_gpu()
            self._stage_assets()
            staged_cfg = self._stage_env_config()
            self._install_task()
        else:
            staged_cfg = None

        # ── Execute each step ─────────────────────────────────────────────────
        for step in steps_to_run:
            output_dir = self._step_output_dir(step)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.pipeline_dir / "logs" / LOG_NAME[step]

            _header(f"Step: {step}")

            # Mark as running
            now = datetime.now(timezone.utc).isoformat()
            self._state["steps"][step].update(
                {
                    "status": "running",
                    "started_at": now,
                    "output_dir": str(output_dir),
                }
            )
            self._write_state()

            # Build step command
            if step == "train":
                env_config = self.config.get("env_config")
                if env_config:
                    ep = Path(env_config)
                    if not ep.is_absolute():
                        ep = self.project_root / ep
                    shutil.copy2(ep.resolve(), output_dir / "env_config.yaml")
                cmd = self._build_train_cmd(output_dir, staged_cfg)
                env = self._get_gui_env(
                    Path(self.isaac_lab_path) / "workspace" / self.config["task"],
                    staged_cfg,
                )
            elif step == "collect":
                experiment_dir = self._resolve_step_input(step)
                cmd = self._build_collect_cmd(experiment_dir, output_dir)
                env = self._get_gui_env(
                    Path(self.isaac_lab_path) / "workspace" / self.config["task"],
                    staged_cfg,
                )
            elif step == "curate":
                input_dir = self._resolve_step_input(step)
                cmd = self._build_curate_cmd(input_dir, output_dir)
                env = None
            elif step == "train-cnn":
                input_dir = self._resolve_step_input(step)
                cmd = self._build_train_cnn_cmd(input_dir, output_dir)
                env = None

            _info(f"Log: {log_path}")

            interceptor = _render_collect_progress if step == "collect" else None
            rc = _run_step_subprocess(
                cmd, env=env, log_path=log_path, line_interceptor=interceptor
            )

            # Isaac Sim sometimes exits 0 despite a Python crash; verify expected
            # output artifacts are present before treating the step as complete.
            if rc == 0:
                rc, actual_output_dir = self._verify_step_output(step, output_dir)
            else:
                actual_output_dir = output_dir

            now = datetime.now(timezone.utc).isoformat()
            if rc == 0:
                self._state["steps"][step].update(
                    {
                        "status": "completed",
                        "completed_at": now,
                        "return_code": 0,
                        "output_dir": str(actual_output_dir),
                    }
                )
                self._write_state()
                _success(f"Step '{step}' completed → {actual_output_dir}")
            else:
                self._state["steps"][step].update(
                    {
                        "status": "failed",
                        "completed_at": now,
                        "return_code": rc,
                    }
                )
                self._write_state()
                _error(f"Step '{step}' failed (exit code {rc})")
                _error(f"Log:            {log_path}")
                _error(f"Pipeline state: {self.pipeline_dir / STATE_FILE}")
                sys.exit(rc)

        _success(f"Pipeline complete — {len(steps_to_run)} step(s) finished.")
        _success(f"Pipeline dir: {self.pipeline_dir}")
