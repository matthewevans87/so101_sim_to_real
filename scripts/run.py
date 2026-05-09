#!/usr/bin/env python3
"""
run.py — so101_sim_to_real pipeline CLI.

Usage:
    ./scripts/run.py COMMAND [OPTIONS]

Commands:
    train          Train an RL policy
    collect        Collect telemetry from a trained policy
    curate         Curate telemetry into a CNN training dataset
    train-cnn      Train a CNN backbone on curated data
    eval           Evaluate a trained agent
    play           Play back a trained agent
    export         Export a trained agent
    install        Install the task package into Isaac Lab
    pin            Create named symlinks for frequently-used paths
    doctor         Diagnose X11 / display configuration
    viz-cnn        Visualize CNN training data and model predictions
    pipeline       Run the full train → collect → curate → train-cnn pipeline
    sweep          Run a grid of Train+Eval experiments and compare results

Sweep quick-start:
    New sweep:
        ./scripts/run.py sweep --sweep configs/sweep_example.yaml
    Dry run (print commands only):
        ./scripts/run.py sweep --sweep configs/sweep_example.yaml --dry-run
    Resume a killed sweep:
        ./scripts/run.py sweep --resume sweeps/sweep_<name>_<timestamp>/
    Full help:
        ./scripts/run.py sweep --help

Environment Variables:
    ISAAC_LAB_PATH   Path to Isaac Lab installation (required for Isaac-dependent commands)
    XAUTHORITY       Path to X authority file for GUI forwarding (auto-discovered if unset)
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

# Local helper for constructing the export_bundle.py command line.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _export_cmd import build_export_command  # noqa: E402

# ── constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_ROOT = PROJECT_ROOT / "so101_rl"
PINS_DIR = PROJECT_ROOT / "scripts" / "pins"

# Maps pin key → symlink filename inside PINS_DIR.
# The filename suffix is chosen to match the expected file type so tab-completion
# and human inspection both work naturally.
_PINS: dict = {
    "bundle": "latest_bundle",
    "cnn_checkpoint": "cnn_checkpoint.pt",
    "checkpoint": "checkpoint.pt",
    "experiment": "experiment",
    "input": "input",
    "output": "output",
}

# Auto-managed symlink names (not exposed as pin --<flag> targets).
_PIN_LATEST_EXPERIMENT = "latest_experiment"
_PIN_LATEST_PIPELINE = "latest_pipeline"
_PIN_LATEST_BUNDLE = "latest_bundle"


def _update_auto_pin(link_name: str, target: Path) -> None:
    """Atomically update an auto-managed symlink inside PINS_DIR."""
    PINS_DIR.mkdir(parents=True, exist_ok=True)
    link = PINS_DIR / link_name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())
    success(f"Auto-pin updated: {link.name}  →  {target.resolve()}")


# ── terminal output ────────────────────────────────────────────────────────────
_RED = "\033[0;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_BLUE = "\033[0;34m"
_NC = "\033[0m"


def info(msg: str) -> None:
    print(f"{_YELLOW}ℹ  {msg}{_NC}")


def success(msg: str) -> None:
    print(f"{_GREEN}✓  {msg}{_NC}")


def error(msg: str) -> None:
    print(f"{_RED}✗  {msg}{_NC}", file=sys.stderr)


def warn(msg: str) -> None:
    print(f"{_YELLOW}!  {msg}{_NC}")


def header(msg: str) -> None:
    bar = "=" * 52
    print(f"{_BLUE}{bar}{_NC}")
    print(f"{_BLUE}{msg}{_NC}")
    print(f"{_BLUE}{bar}{_NC}")


# ── subprocess helper ──────────────────────────────────────────────────────────
def run_subprocess(
    cmd: list,
    env: dict = None,
    log_path: Path = None,
    check: bool = True,
) -> int:
    """Run a command, tee-ing stdout+stderr to the terminal and an optional log file."""
    info(f"Executing: {' '.join(str(c) for c in cmd)}")
    log_fh = open(log_path, "w") if log_path else None
    rc = 1
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env if env is not None else os.environ.copy(),
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
        proc.wait()
        rc = proc.returncode
    finally:
        if log_fh:
            log_fh.close()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


# ── X11 helpers ────────────────────────────────────────────────────────────────
_X11_PROCS = re.compile(r"gnome-shell|plasmashell|xfce4-session|Xorg|Xwayland")


def _discover_x11() -> tuple:
    """Scan running desktop processes for DISPLAY and XAUTHORITY env vars."""
    discovered_display = None
    discovered_xauth = None
    for env_file in sorted(glob.glob("/proc/*/environ")):
        try:
            pid = env_file.split("/")[2]
            try:
                comm = Path(f"/proc/{pid}/comm").read_text().strip()
            except OSError:
                comm = ""
            try:
                cmdline = (
                    Path(f"/proc/{pid}/cmdline")
                    .read_bytes()
                    .replace(b"\x00", b" ")
                    .decode("utf-8", errors="replace")
                )
            except OSError:
                cmdline = ""
            if not _X11_PROCS.search(comm + " " + cmdline):
                continue
            env_data = Path(env_file).read_bytes()
            env_vars = {}
            for entry in env_data.split(b"\x00"):
                if b"=" in entry:
                    k, _, v = entry.partition(b"=")
                    env_vars[k.decode("utf-8", errors="replace")] = v.decode(
                        "utf-8", errors="replace"
                    )
            d = env_vars.get("DISPLAY", "")
            xa = env_vars.get("XAUTHORITY", "")
            if d or xa:
                info(f"Candidate GUI process: {comm} (pid {pid})")
                if d:
                    print(f"    DISPLAY={d}")
                if xa:
                    print(f"    XAUTHORITY={xa}")
                if d and not discovered_display:
                    discovered_display = d
                if xa and not discovered_xauth:
                    discovered_xauth = xa
                break
        except (OSError, ValueError, IndexError):
            continue
    return discovered_display, discovered_xauth


def resolve_x11(display_sock: int = None) -> None:
    """Resolve and set DISPLAY/XAUTHORITY in os.environ."""
    if display_sock is not None:
        os.environ["DISPLAY"] = f":{display_sock}"
    disp = os.environ.get("DISPLAY", "")
    xauth = os.environ.get("XAUTHORITY", "")
    if not disp or not xauth:
        disc_disp, disc_xauth = _discover_x11()
        if disc_disp and not disp:
            os.environ["DISPLAY"] = disc_disp
        if disc_xauth and not xauth:
            os.environ["XAUTHORITY"] = disc_xauth
    # Last-resort XAUTHORITY fallback
    if "XAUTHORITY" not in os.environ:
        user = os.environ.get("USER", "")
        for candidate in [
            Path.home() / ".Xauthority",
            Path(f"/home/{user}") / ".Xauthority",
        ]:
            if candidate.is_file():
                os.environ["XAUTHORITY"] = str(candidate)
                break


def get_gui_env(workspace_path: Path, staged_config: Path = None) -> dict:
    """Return an env dict with ISAAC_LAB_WORKSPACE_PATH and X11 vars set."""
    env = os.environ.copy()
    env["ISAAC_LAB_WORKSPACE_PATH"] = str(workspace_path)
    if staged_config:
        env["SO101_ENV_CONFIG"] = str(staged_config)
    return env


def _log_x11_status() -> None:
    disp = os.environ.get("DISPLAY", "")
    xauth = os.environ.get("XAUTHORITY", "")
    if disp:
        info(f"Using DISPLAY={disp}")
    else:
        warn("DISPLAY is not set. GUI windows may fail to open.")
    if xauth:
        info(f"Using XAUTHORITY={xauth}")
    else:
        warn("XAUTHORITY is not set. X11 auth may fail over SSH.")


# ── GPU check ──────────────────────────────────────────────────────────────────
def check_gpu() -> None:
    if not shutil.which("nvidia-smi"):
        error("nvidia-smi not found. Please install NVIDIA drivers.")
        sys.exit(1)
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
        error("Cannot access NVIDIA GPU. Please check your drivers.")
        sys.exit(1)
    success(f"GPU detected: {result.stdout.strip()}")


# ── asset / config staging ─────────────────────────────────────────────────────
def stage_assets(isaac_lab_path: str, task: str) -> None:
    dest = Path(isaac_lab_path) / "workspace" / task / "assets"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT / "assets", dest, dirs_exist_ok=True)
    success(f"Assets staged to {dest}")

    # Stage so101_real/configs (contains camera_intrinsics.yaml) so that
    # camera.py can resolve it via ISAAC_LAB_WORKSPACE_PATH at runtime.
    real_cfg_src = PROJECT_ROOT / "so101_real" / "configs"
    if real_cfg_src.is_dir():
        real_cfg_dest = (
            Path(isaac_lab_path) / "workspace" / task / "so101_real" / "configs"
        )
        real_cfg_dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(real_cfg_src, real_cfg_dest, dirs_exist_ok=True)
        success(f"so101_real/configs staged to {real_cfg_dest}")


def stage_env_config(env_config_path: str, isaac_lab_path: str, task: str) -> Path:
    """Copy env config into Isaac Lab workspace. Returns the staged path."""
    src = Path(env_config_path)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    src = src.resolve()
    if not src.is_file():
        error(f"Environment config not found: {src}")
        sys.exit(1)
    dest_dir = Path(isaac_lab_path) / "workspace" / task / "configs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    success(f"Environment config staged: {dest}")
    return dest


# ── offline Python resolver ────────────────────────────────────────────────────
def resolve_python_cmd(conda_env: str = None, python_exe: str = None) -> list:
    """Return the Python command prefix for offline (non-Isaac) steps."""
    if python_exe:
        p = Path(python_exe)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.is_file() or not os.access(p, os.X_OK):
            error(f"Python executable not found or not executable: {p}")
            sys.exit(1)
        return [str(p)]
    if conda_env:
        if not shutil.which("conda"):
            error("conda is required when using --conda-env")
            sys.exit(1)
        return ["conda", "run", "--no-capture-output", "-n", conda_env, "python"]
    return ["python"]


# ── ISAAC_LAB_PATH resolver ────────────────────────────────────────────────────
def require_isaac_lab() -> str:
    path = os.environ.get("ISAAC_LAB_PATH", "").strip()
    if not path:
        error("ISAAC_LAB_PATH environment variable is not set.")
        error("Set it before calling this script, e.g.:")
        error("  export ISAAC_LAB_PATH=/path/to/IsaacLab")
        sys.exit(1)
    p = Path(path)
    if not p.is_dir():
        error(f"ISAAC_LAB_PATH does not exist or is not a directory: {path}")
        sys.exit(1)
    return str(p)


def install_task(isaac_lab_path: str) -> None:
    run_subprocess(
        [
            f"{isaac_lab_path}/isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            "-e",
            str(TASK_ROOT / "source" / "so101_rl"),
        ]
    )
    # Also install the so101 library so the task code can import so101.utils.*
    run_subprocess(
        [
            f"{isaac_lab_path}/isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            "-e",
            str(PROJECT_ROOT),
        ]
    )
    success("Task package installed.")


# ════════════════════════════════════════════════════════════════════════════════
# Subcommand implementations
# ════════════════════════════════════════════════════════════════════════════════


def cmd_train(args) -> None:
    isaac_lab_path = require_isaac_lab()

    env_config = Path(args.config)
    if not env_config.is_absolute():
        env_config = PROJECT_ROOT / env_config
    env_config = env_config.resolve()
    if not env_config.is_file():
        error(f"Env config not found: {env_config}")
        sys.exit(1)

    check_gpu()
    stage_assets(isaac_lab_path, args.task)
    staged_cfg = stage_env_config(str(env_config), isaac_lab_path, args.task)
    install_task(isaac_lab_path)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = (
        Path(args.output).resolve()
        if getattr(args, "output", None)
        else PROJECT_ROOT / "artifacts"
    )
    artifacts_dir = base / timestamp
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(env_config, artifacts_dir / "env_config.yaml")
    success(f"Env config saved to {artifacts_dir}/env_config.yaml")

    cmd = [
        f"{isaac_lab_path}/isaaclab.sh",
        "-p",
        str(TASK_ROOT / "scripts" / "skrl" / "train.py"),
        "--task",
        args.task,
        "--artifacts_dir",
        str(artifacts_dir),
        "--env_config",
        str(artifacts_dir / "env_config.yaml"),
        f"hydra.run.dir={artifacts_dir}/hydra",
    ]
    if args.headless:
        cmd += ["--headless"]
    if args.cameras:
        cmd += ["--enable_cameras"]
    if args.envs:
        cmd += ["--num_envs", str(args.envs)]
    if args.iters:
        cmd += ["--max_iterations", str(args.iters)]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if args.cnn_checkpoint:
        cmd += ["--cnn_checkpoint", str(args.cnn_checkpoint)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    resolve_x11(getattr(args, "display", None))
    env = get_gui_env(Path(isaac_lab_path) / "workspace" / args.task, staged_cfg)
    env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ})
    _log_x11_status()
    run_subprocess(cmd, env=env)
    _update_auto_pin(_PIN_LATEST_EXPERIMENT, artifacts_dir)


def cmd_collect(args) -> None:
    isaac_lab_path = require_isaac_lab()

    experiment = Path(args.experiment).resolve()
    if not experiment.is_dir():
        error(f"Experiment directory not found: {experiment}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = Path(args.output).resolve() / timestamp
    output.mkdir(parents=True, exist_ok=True)
    info(f"Collect output: {output}")

    staged_cfg = None
    env_cfg = experiment / "env_config.yaml"
    if env_cfg.is_file():
        info(f"Found env_config.yaml: {env_cfg}")
        staged_cfg = stage_env_config(str(env_cfg), isaac_lab_path, args.task)
    else:
        warn(f"No env_config.yaml found in {experiment}")

    check_gpu()
    stage_assets(isaac_lab_path, args.task)
    install_task(isaac_lab_path)

    cmd = [
        f"{isaac_lab_path}/isaaclab.sh",
        "-p",
        str(TASK_ROOT / "scripts" / "skrl" / "collect_telemetry.py"),
        "--experiment-path",
        str(experiment),
        "--task",
        args.task,
        "--sample-every-steps",
        str(args.sample_interval),
        "--num-episodes",
        str(args.episodes),
        "--output-dir",
        str(output),
        "--seed",
        str(args.seed),
    ]
    if args.shard_size:
        cmd += ["--samples-per-shard", str(args.shard_size)]
    if args.envs:
        cmd += ["--num_envs", str(args.envs)]
    if args.headless:
        cmd += ["--headless"]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if getattr(args, "cnn_checkpoint", None):
        cmd += ["--cnn_checkpoint", str(args.cnn_checkpoint)]

    resolve_x11(getattr(args, "display", None))
    env = get_gui_env(Path(isaac_lab_path) / "workspace" / args.task, staged_cfg)
    env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ})
    _log_x11_status()
    run_subprocess(cmd, env=env)


def cmd_curate(args) -> None:
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    info(f"Curate output: {output_dir}")

    config = Path(args.config)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    config = config.resolve()
    if not config.is_file():
        error(f"CNN config not found: {config}")
        sys.exit(1)

    py_cmd = resolve_python_cmd(
        conda_env=getattr(args, "conda_env", None),
        python_exe=getattr(args, "python", None),
    )
    cmd = py_cmd + [
        str(PROJECT_ROOT / "so101" / "curate" / "curate.py"),
        "--telemetry-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
    ]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    run_subprocess(cmd)


def cmd_train_cnn(args) -> None:
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output).resolve() / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    info(f"Train-CNN output: {output_dir}")

    config = Path(args.config)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    config = config.resolve()
    if not config.is_file():
        error(f"CNN config not found: {config}")
        sys.exit(1)

    py_cmd = resolve_python_cmd(
        conda_env=getattr(args, "conda_env", None),
        python_exe=getattr(args, "python", None),
    )
    cmd = py_cmd + [
        str(PROJECT_ROOT / "so101" / "train_cnn.py"),
        "--curated-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config),
    ]
    if args.device:
        cmd += ["--device", args.device]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    run_subprocess(cmd)


def cmd_eval(args) -> None:
    isaac_lab_path = require_isaac_lab()

    experiment = Path(args.experiment).resolve()
    if not experiment.is_dir():
        error(f"Experiment directory not found: {experiment}")
        sys.exit(1)

    task = args.task
    staged_cfg = None
    env_cfg = experiment / "env_config.yaml"
    if env_cfg.is_file():
        staged_cfg = stage_env_config(str(env_cfg), isaac_lab_path, task)

    check_gpu()
    stage_assets(isaac_lab_path, task)
    install_task(isaac_lab_path)

    cmd = [
        f"{isaac_lab_path}/isaaclab.sh",
        "-p",
        str(TASK_ROOT / "scripts" / "skrl" / "evaluate.py"),
        "--experiment-path",
        str(experiment),
        "--task",
        task,
    ]
    if args.episodes:
        cmd += ["--num-episodes", str(args.episodes)]
    if args.videos:
        cmd += ["--num-videos", str(args.videos)]
    if getattr(args, "record_wrist_cam", False):
        cmd += ["--record-wrist-cam"]
    if getattr(args, "record_overhead_cam", False):
        cmd += ["--record-overhead-cam"]
    if getattr(args, "record_viewport_cam", False):
        cmd += ["--record-viewport-cam"]
    if args.envs:
        cmd += ["--num_envs", str(args.envs)]
    if args.verbosity:
        cmd += ["--verbosity", args.verbosity]
    if args.headless:
        cmd += ["--headless"]
    if getattr(args, "cameras", False):
        cmd += ["--enable_cameras"]

    resolve_x11(getattr(args, "display", None))
    env = get_gui_env(Path(isaac_lab_path) / "workspace" / task, staged_cfg)
    env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ})
    _log_x11_status()
    run_subprocess(cmd, env=env)


def cmd_play(args) -> None:
    isaac_lab_path = require_isaac_lab()

    if not args.experiment and not args.checkpoint:
        error("Either --experiment or --checkpoint must be provided.")
        sys.exit(1)

    staged_cfg = None

    if args.experiment:
        experiment_dir = Path(args.experiment).resolve()
        if not experiment_dir.is_dir():
            error(f"Experiment directory not found: {experiment_dir}")
            sys.exit(1)
        env_cfg_path = experiment_dir / "env_config.yaml"
        if env_cfg_path.is_file():
            staged_cfg = stage_env_config(str(env_cfg_path), isaac_lab_path, args.task)
        else:
            warn(f"No env_config.yaml found in {experiment_dir}")
        if args.checkpoint:
            checkpoint = Path(args.checkpoint).resolve()
        else:
            checkpoint = (
                experiment_dir / "skrl" / "agent" / "checkpoints" / "best_agent.pt"
            )
            info(f"Derived checkpoint: {checkpoint}")
        if not args.cnn_checkpoint:
            candidate = experiment_dir / "cnn_checkpoint.pt"
            if candidate.is_file():
                args.cnn_checkpoint = candidate
                info(f"Derived cnn_checkpoint: {candidate}")
            else:
                warn(f"No cnn_checkpoint.pt found in experiment dir: {experiment_dir}")
        ckpt_root = experiment_dir
    else:
        # --checkpoint only: --config is required
        if not args.config:
            error("--config is required when --experiment is not provided.")
            sys.exit(1)
        checkpoint = Path(args.checkpoint).resolve()
        ckpt_root = (
            checkpoint.parent.parent
        )  # <task_dir>/checkpoints/<file> → <task_dir>
        staged_cfg = stage_env_config(args.config, isaac_lab_path, args.task)

    if not checkpoint.is_file():
        error(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)

    check_gpu()
    stage_assets(isaac_lab_path, args.task)
    install_task(isaac_lab_path)

    cmd = [
        f"{isaac_lab_path}/isaaclab.sh",
        "-p",
        str(TASK_ROOT / "scripts" / "skrl" / "play.py"),
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        f"hydra.run.dir={ckpt_root}/hydra_play",
    ]
    if args.headless:
        cmd += ["--headless"]
    if args.cameras:
        cmd += ["--enable_cameras"]
    if args.envs:
        cmd += ["--num_envs", str(args.envs)]
    if args.video:
        cmd += ["--video"]
    if args.video_len:
        cmd += ["--video_length", str(args.video_len)]
    if args.cnn_checkpoint:
        cmd += ["--cnn_checkpoint", str(args.cnn_checkpoint)]

    resolve_x11(getattr(args, "display", None))
    env = get_gui_env(Path(isaac_lab_path) / "workspace" / args.task, staged_cfg)
    env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ})
    _log_x11_status()
    run_subprocess(cmd, env=env)


def cmd_export(args) -> None:
    """Export a trained policy + CNN backbone to a self-contained deploy bundle."""
    isaac_lab_path = require_isaac_lab()

    # ── Resolve experiment directory ──────────────────────────────────────────
    if args.experiment:
        experiment_dir = Path(args.experiment).resolve()
    else:
        pin_experiment = PINS_DIR / _PIN_LATEST_EXPERIMENT
        if pin_experiment.is_symlink():
            experiment_dir = pin_experiment.resolve()
            info(f"Using latest_experiment pin: {experiment_dir}")
        else:
            error(
                "--experiment is required (or run a training first to set the "
                "latest_experiment pin)."
            )
            sys.exit(1)

    if not experiment_dir.is_dir():
        error(f"Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # ── Resolve output directory ──────────────────────────────────────────────
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = experiment_dir / f"deploy_bundle_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage env config ──────────────────────────────────────────────────────
    env_cfg_path = experiment_dir / "env_config.yaml"
    staged_cfg = None
    if env_cfg_path.is_file():
        staged_cfg = stage_env_config(str(env_cfg_path), isaac_lab_path, args.task)

    check_gpu()
    stage_assets(isaac_lab_path, args.task)
    install_task(isaac_lab_path)

    cmd = build_export_command(
        isaac_lab_path=isaac_lab_path,
        task_root=TASK_ROOT,
        task=args.task,
        experiment_path=experiment_dir,
        output_dir=output_dir,
        torchscript=getattr(args, "torchscript", False),
    )

    resolve_x11(getattr(args, "display", None))
    env = get_gui_env(Path(isaac_lab_path) / "workspace" / args.task, staged_cfg)
    env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ})
    _log_x11_status()
    rc = run_subprocess(cmd, env=env)
    if rc == 0:
        manifest_check = output_dir / "manifest.json"
        if manifest_check.is_file():
            _update_auto_pin(_PIN_LATEST_BUNDLE, output_dir)
        else:
            error(
                "export_bundle.py exited 0 but manifest.json was not created. "
                "Check the log for errors."
            )
            sys.exit(1)


def cmd_deploy(args) -> None:
    """Run real-robot inference from a deploy bundle (no Isaac Lab required)."""
    # Resolve bundle path
    if args.bundle:
        bundle_path = Path(args.bundle).resolve()
    else:
        pin_bundle = PINS_DIR / _PIN_LATEST_BUNDLE
        if pin_bundle.is_symlink():
            bundle_path = pin_bundle.resolve()
            info(f"Using latest_bundle pin: {bundle_path}")
        else:
            error(
                "--bundle is required (or run an export first to set the "
                "latest_bundle pin)."
            )
            sys.exit(1)

    if not bundle_path.is_dir():
        error(f"Bundle directory not found: {bundle_path}")
        sys.exit(1)

    if not args.robot_config:
        error("--robot-config is required.")
        sys.exit(1)

    robot_config = Path(args.robot_config).resolve()
    if not robot_config.is_file():
        error(f"Robot config not found: {robot_config}")
        sys.exit(1)

    cmd = [
        "python",
        "-m",
        "so101_real",
        "run",
        "--bundle",
        str(bundle_path),
        "--robot-config",
        str(robot_config),
    ]
    if getattr(args, "episodes", None):
        cmd += ["--episodes", str(args.episodes)]
    if getattr(args, "seed", None) is not None:
        cmd += ["--seed", str(args.seed)]
    if getattr(args, "overlay", False):
        cmd.append("--overlay")
    if getattr(args, "record", False):
        cmd.append("--record")
    if getattr(args, "dry_run", False):
        cmd.append("--dry-run")

    resolve_x11(getattr(args, "display", None))
    env = os.environ.copy()
    env.update({k: os.environ[k] for k in ("DISPLAY", "XAUTHORITY") if k in os.environ})
    run_subprocess(cmd, env=env)


def cmd_install(args) -> None:
    isaac_lab_path = require_isaac_lab()
    stage_env_config(args.config, isaac_lab_path, args.task)
    install_task(isaac_lab_path)


def cmd_pin(args) -> None:
    """Create or list named symlinks in scripts/pins/ for frequently-used paths."""

    if args.list:
        PINS_DIR.mkdir(parents=True, exist_ok=True)
        header("Pinned paths")
        for key, filename in _PINS.items():
            link = PINS_DIR / filename
            flag = f"--{key.replace('_', '-')}"
            if link.is_symlink():
                target = os.readlink(str(link))
                exists_marker = "" if link.exists() else "  [TARGET MISSING]"
                success(f"{flag:<22s}  {link}  →  {target}{exists_marker}")
            else:
                info(f"{flag:<22s}  <not pinned>")
        # Auto-managed pins
        for name, label in (
            (_PIN_LATEST_EXPERIMENT, "latest_experiment (auto)"),
            (_PIN_LATEST_PIPELINE, "latest_pipeline   (auto)"),
        ):
            link = PINS_DIR / name
            label_col = f"  {label}"
            if link.is_symlink():
                target = os.readlink(str(link))
                exists_marker = "" if link.exists() else "  [TARGET MISSING]"
                success(f"{label_col:<32s}  {link}  →  {target}{exists_marker}")
            else:
                info(f"{label_col:<32s}  <not yet set>")
        return

    def _set_pin(key: str, raw_path: str) -> None:
        src = Path(raw_path)
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        else:
            src = src.resolve()
        if not src.exists():
            warn(f"Target does not exist (pinning anyway): {src}")
        PINS_DIR.mkdir(parents=True, exist_ok=True)
        link = PINS_DIR / _PINS[key]
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src)
        success(f"Pinned --{key.replace('_', '-')}: {link}  →  {src}")

    any_set = False
    for key in _PINS:
        val = getattr(args, key, None)
        if val is not None:
            _set_pin(key, val)
            any_set = True

    if not any_set:
        error(
            "No pin target specified. Pass at least one of: "
            + ", ".join(f"--{k.replace('_', '-')}" for k in _PINS)
            + "  (or --list to view current pins)"
        )
        sys.exit(1)


def cmd_doctor(args) -> None:
    header("X11 / Display Doctor")
    info("Current shell values")
    print(f"    USER={os.environ.get('USER', '<unset>')}")
    print(f"    DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    print(f"    XAUTHORITY={os.environ.get('XAUTHORITY', '<unset>')}")

    xauth = os.environ.get("XAUTHORITY", "")
    if xauth:
        if Path(xauth).is_file():
            success(f"XAUTHORITY file exists: {xauth}")
        else:
            warn(f"XAUTHORITY is set but file does not exist: {xauth}")

    info("Attempting discovery from active desktop processes")
    disc_disp, disc_xauth = _discover_x11()
    if not disc_disp and not disc_xauth:
        warn("No GUI process with DISPLAY/XAUTHORITY env vars found")

    resolve_x11(getattr(args, "display", None))

    info("Resolved values")
    print(f"    DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    print(f"    XAUTHORITY={os.environ.get('XAUTHORITY', '<unset>')}")

    disp = os.environ.get("DISPLAY", "")
    if disp:
        sock = disp.lstrip(":").split(".")[0]
        info("Suggested command flag")
        print(
            f"    ./scripts/run.py train --config configs/baseline.yaml ... --display {sock}"
        )
        if shutil.which("xauth"):
            xauth_file = os.environ.get("XAUTHORITY", str(Path.home() / ".Xauthority"))
            result = subprocess.run(
                ["xauth", "-f", xauth_file, "list", disp],
                capture_output=True,
            )
            if result.returncode == 0:
                success(f"xauth can read cookie for {disp}")
            else:
                warn(f"xauth could not confirm cookie for {disp}")
        else:
            warn("xauth not found; skipping cookie check")


def cmd_viz_cnn(args) -> None:
    if not args.input and not args.manifest:
        error("--input or --manifest is required for viz-cnn")
        sys.exit(1)

    config = Path(args.config)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    config = config.resolve()

    py_cmd = resolve_python_cmd(
        conda_env=getattr(args, "conda_env", None),
        python_exe=getattr(args, "python", None),
    )
    cmd = py_cmd + ["-m", "so101.viz_cnn"]

    if args.input:
        cmd += ["--curated-dir", str(Path(args.input).resolve()), "--split", args.split]
    else:
        cmd += ["--manifest", str(Path(args.manifest).resolve())]

    if args.model:
        if not config.is_file():
            error(f"CNN config not found: {config}  (required with --model)")
            sys.exit(1)
        cmd += ["--model", str(Path(args.model).resolve()), "--config", str(config)]

    cmd += ["--start", str(args.start)]
    if args.device:
        cmd += ["--device", args.device]

    run_subprocess(cmd)


def cmd_sweep(args) -> None:
    resolve_x11(getattr(args, "display", None))
    sys.path.insert(0, str(Path(__file__).parent))
    from sweep import SweepOrchestrator, expand_experiment_definition

    # For dry-runs, a missing/invalid ISAAC_LAB_PATH is non-fatal: the path
    # is shown in the printed commands so the user can verify them without a
    # live Isaac Lab installation.  For real runs require_isaac_lab() enforces
    # the check strictly and exits on failure.
    if args.dry_run:
        isaac_lab_path = (
            os.environ.get("ISAAC_LAB_PATH", "").strip() or "<ISAAC_LAB_PATH>"
        )
    else:
        isaac_lab_path = require_isaac_lab()

    if args.resume:
        sweep_dir = Path(args.resume).resolve()
        orchestrator = SweepOrchestrator.from_existing(
            sweep_dir=sweep_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=PROJECT_ROOT,
        )
    else:
        if not args.sweep:
            error("--sweep is required when not resuming with --resume")
            sys.exit(1)
        sweep_path = Path(args.sweep)
        if not sweep_path.is_absolute():
            sweep_path = PROJECT_ROOT / sweep_path
        sweep_path = sweep_path.resolve()
        if not sweep_path.is_file():
            error(f"Sweep config not found: {sweep_path}")
            sys.exit(1)
        with open(sweep_path) as f:
            import yaml as _yaml

            config = _yaml.safe_load(f)

        try:
            config = expand_experiment_definition(config)
        except ValueError as exc:
            error(f"Invalid sweep config: {exc}")
            sys.exit(1)

        sweep_name = config.get("name", "sweep")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (
            Path(args.output).resolve()
            if getattr(args, "output", None)
            else PROJECT_ROOT / "sweeps"
        )
        sweep_dir = base / f"sweep_{sweep_name}_{timestamp}"
        orchestrator = SweepOrchestrator(
            config=config,
            config_path=sweep_path,
            sweep_dir=sweep_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=PROJECT_ROOT,
        )

    if getattr(args, "create_configs_only", False):
        orchestrator.materialize_all()
        return

    orchestrator.run(
        from_experiment=getattr(args, "from_experiment", None),
        dry_run=args.dry_run,
        retry_eval=getattr(args, "retry_eval", False),
    )


def cmd_pipeline(args) -> None:
    resolve_x11(getattr(args, "display", None))
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import PipelineOrchestrator

    isaac_lab_path = require_isaac_lab()

    if args.pipeline_dir:
        pipeline_dir = Path(args.pipeline_dir).resolve()
        orchestrator = PipelineOrchestrator.from_existing(
            pipeline_dir=pipeline_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=PROJECT_ROOT,
        )
    else:
        if not args.config:
            error("--config is required when not resuming with --pipeline-dir")
            sys.exit(1)
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        if not config_path.is_file():
            error(f"Pipeline config not found: {config_path}")
            sys.exit(1)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = (
            Path(args.output).resolve()
            if getattr(args, "output", None)
            else PROJECT_ROOT / "artifacts"
        )
        pipeline_dir = base / f"pipeline_{timestamp}"
        orchestrator = PipelineOrchestrator(
            config=config,
            config_path=config_path,
            pipeline_dir=pipeline_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=PROJECT_ROOT,
            ad_hoc_experiment=getattr(args, "experiment", None),
            ad_hoc_input=getattr(args, "input", None),
            ad_hoc_cnn_checkpoint=getattr(args, "cnn_checkpoint", None),
            force_display=getattr(args, "display", None) is not None,
        )

    orchestrator.run(
        from_step=args.from_step,
        to_step=args.to_step,
        dry_run=args.dry_run,
    )


# ════════════════════════════════════════════════════════════════════════════════
# Argument parser
# ════════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="so101_sim_to_real pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Train an RL policy")
    p.add_argument(
        "--task",
        required=True,
        metavar="TASK",
        help="Task name (e.g. So101-LiftCube-v0)",
    )
    p.add_argument("--config", required=True, metavar="PATH", help="Env config YAML")
    p.add_argument("--headless", action="store_true", help="Run without GUI")
    p.add_argument("--cameras", action="store_true", help="Enable cameras")
    p.add_argument("--envs", type=int, metavar="N", help="Override num_envs")
    p.add_argument(
        "--iters", type=int, metavar="N", help="Override max training iterations"
    )
    p.add_argument("--checkpoint", metavar="PATH", help="Resume from checkpoint")
    p.add_argument(
        "--cnn-checkpoint",
        metavar="PATH",
        help="Pretrained MultiTaskCnn checkpoint (.pt)",
    )
    p.add_argument("--seed", type=int, metavar="N", help="RNG seed")
    p.add_argument("--display", type=int, metavar="N", help="X display socket number")
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Base output dir; artifacts saved to <output>/<timestamp>/ (default: artifacts/)",
    )
    p.set_defaults(func=cmd_train)

    # ── collect ───────────────────────────────────────────────────────────────
    p = sub.add_parser("collect", help="Collect telemetry from a trained policy")
    p.add_argument("--task", required=True, metavar="TASK", help="Task name")
    p.add_argument(
        "--experiment", required=True, metavar="PATH", help="RL experiment directory"
    )
    p.add_argument(
        "--sample-interval",
        required=True,
        type=int,
        metavar="N",
        dest="sample_interval",
        help="Collect a sample every N environment steps",
    )
    p.add_argument(
        "--episodes",
        required=True,
        type=int,
        metavar="N",
        help="Stop after N episodes complete",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Base output dir; telemetry saved to <output>/<timestamp>/",
    )
    p.add_argument("--seed", required=True, type=int, metavar="N", help="RNG seed")
    p.add_argument(
        "--shard-size",
        type=int,
        metavar="N",
        dest="shard_size",
        help="Samples per NPZ shard",
    )
    p.add_argument("--envs", type=int, metavar="N", help="Override num_envs")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--checkpoint", metavar="PATH", help="Override checkpoint path")
    p.add_argument(
        "--cnn-checkpoint",
        metavar="PATH",
        dest="cnn_checkpoint",
        help="CNN backbone checkpoint (.pt); auto-detected from experiment dir if omitted",
    )
    p.add_argument("--display", type=int, metavar="N")
    p.set_defaults(func=cmd_collect)

    # ── curate ────────────────────────────────────────────────────────────────
    p = sub.add_parser("curate", help="Curate telemetry into a CNN training dataset")
    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Telemetry directory (= collect --output)",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Base output dir; curated data saved to <output>/<timestamp>/",
    )
    p.add_argument(
        "--config",
        default="configs/cnn_pretrain.yaml",
        metavar="PATH",
        help="CNN pretrain config YAML (default: configs/cnn_pretrain.yaml)",
    )
    p.add_argument("--seed", type=int, metavar="N")
    p.add_argument("--conda-env", metavar="NAME", dest="conda_env")
    p.add_argument("--python", metavar="PATH", help="Explicit Python executable")
    p.set_defaults(func=cmd_curate)

    # ── train-cnn ─────────────────────────────────────────────────────────────
    p = sub.add_parser("train-cnn", help="Train a CNN backbone on curated data")
    p.add_argument(
        "--input",
        required=True,
        metavar="PATH",
        help="Curated dataset directory (= curate --output)",
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Base output dir; checkpoints saved to <output>/<timestamp>/",
    )
    p.add_argument(
        "--config",
        default="configs/cnn_pretrain.yaml",
        metavar="PATH",
        help="CNN pretrain config YAML (default: configs/cnn_pretrain.yaml)",
    )
    p.add_argument("--device", metavar="DEVICE", help="PyTorch device (e.g. cuda:0)")
    p.add_argument("--seed", type=int, metavar="N")
    p.add_argument("--conda-env", metavar="NAME", dest="conda_env")
    p.add_argument("--python", metavar="PATH", help="Explicit Python executable")
    p.set_defaults(func=cmd_train_cnn)

    # ── eval ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("eval", help="Evaluate a trained agent")
    p.add_argument(
        "--experiment", required=True, metavar="PATH", help="Experiment directory"
    )
    p.add_argument(
        "--task",
        required=True,
        metavar="TASK",
        help="Task name",
    )
    p.add_argument("--episodes", type=int, metavar="N")
    p.add_argument("--videos", type=int, metavar="N")
    p.add_argument(
        "--record-wrist-cam",
        action="store_true",
        default=False,
        help="Record wrist-camera video for --videos episodes.",
    )
    p.add_argument(
        "--record-overhead-cam",
        action="store_true",
        default=False,
        help="Record overhead-camera video (adds significant VRAM cost).",
    )
    p.add_argument(
        "--record-viewport-cam",
        action="store_true",
        default=False,
        help="Record Isaac Sim full viewport (all envs tiled) as a single .mp4.",
    )
    p.add_argument("--envs", type=int, metavar="N")
    p.add_argument(
        "--cameras",
        action="store_true",
        help="Enable cameras (required for vision-based policies)",
    )
    p.add_argument("--verbosity", choices=["full", "basic"], default="basic")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--display", type=int, metavar="N")
    p.set_defaults(func=cmd_eval)

    # ── play ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("play", help="Play back a trained agent")
    p.add_argument("--task", required=True, metavar="TASK")
    p.add_argument(
        "--experiment",
        metavar="PATH",
        help="RL experiment directory; supplies env_config.yaml and default checkpoint",
    )
    p.add_argument(
        "--checkpoint",
        metavar="PATH",
        help="Override checkpoint path (default: <experiment>/skrl/<task>/checkpoints/best_agent.pt)",
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        help="Env config YAML (required when --experiment is not provided)",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--cameras", action="store_true")
    p.add_argument("--envs", type=int, metavar="N")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video-len", type=int, metavar="N", dest="video_len")
    p.add_argument(
        "--cnn-checkpoint",
        metavar="PATH",
        help="Pretrained MultiTaskCnn checkpoint (.pt)",
    )
    p.add_argument("--display", type=int, metavar="N")
    p.set_defaults(func=cmd_play)

    # ── export ────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "export",
        help="Export a trained agent to a self-contained deploy bundle",
    )
    p.add_argument("--task", required=True, metavar="TASK")
    p.add_argument(
        "--experiment",
        metavar="PATH",
        help="Training experiment directory (default: latest_experiment pin)",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Output directory for the deploy bundle (default: <experiment>/deploy_bundle_<ts>)",
    )
    p.add_argument(
        "--torchscript",
        action="store_true",
        help="Also trace and save a TorchScript combined model",
    )
    p.add_argument("--display", type=int, metavar="N")
    p.set_defaults(func=cmd_export)

    # ── deploy ────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "deploy",
        help="Run real-robot inference from a deploy bundle (no Isaac Lab required)",
    )
    p.add_argument(
        "--bundle",
        metavar="PATH",
        help="Deploy bundle directory (default: latest_bundle pin)",
    )
    p.add_argument(
        "--robot-config",
        metavar="PATH",
        required=True,
        dest="robot_config",
        help="Robot config YAML (so101_real/configs/robot.yaml template)",
    )
    p.add_argument(
        "--episodes", type=int, metavar="N", help="Number of episodes to run"
    )
    p.add_argument("--seed", type=int, metavar="SEED")
    p.add_argument("--overlay", action="store_true", help="Show live OpenCV overlay")
    p.add_argument("--record", action="store_true", help="Record episodes to NPZ files")
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Validate bundle and robot config without moving the robot",
    )
    p.add_argument(
        "--display",
        type=int,
        metavar="N",
        default=None,
        help="X11 display number (e.g. 0 → DISPLAY=:0). Auto-discovered if omitted.",
    )
    p.set_defaults(func=cmd_deploy)

    # ── install ───────────────────────────────────────────────────────────────
    p = sub.add_parser("install", help="Install the task package into Isaac Lab")
    p.add_argument("--task", required=True, metavar="TASK")
    p.add_argument("--config", required=True, metavar="PATH", help="Env config YAML")
    p.set_defaults(func=cmd_install)

    # ── doctor ────────────────────────────────────────────────────────────────
    p = sub.add_parser("doctor", help="Diagnose X11 / display configuration")
    p.add_argument("--display", type=int, metavar="N")
    p.set_defaults(func=cmd_doctor)

    # ── viz-cnn ───────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "viz-cnn", help="Visualize CNN training data and model predictions"
    )
    p.add_argument(
        "--input",
        metavar="PATH",
        help="Curated dataset directory (mutually exclusive with --manifest)",
    )
    p.add_argument("--manifest", metavar="PATH", help="Raw manifest JSON")
    p.add_argument("--model", metavar="PATH", help="Full PretrainCnn checkpoint")
    p.add_argument(
        "--config",
        default="configs/cnn_pretrain.yaml",
        metavar="PATH",
        help="CNN pretrain config YAML (required with --model)",
    )
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--start", type=int, default=0, metavar="IDX")
    p.add_argument("--device", metavar="DEVICE")
    p.add_argument("--conda-env", metavar="NAME", dest="conda_env")
    p.add_argument("--python", metavar="PATH")
    p.set_defaults(func=cmd_viz_cnn)

    # ── pipeline ──────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "pipeline",
        help="Run the full train → collect → curate → train-cnn pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Runs the training pipeline. Steps: train → collect → curate → train-cnn.\n"
            "\n"
            "New run:     --config pipeline.yaml [--from STEP] [--to STEP]\n"
            "Resume run:  --pipeline-dir artifacts/pipeline_TS/ [--from STEP]\n"
            "Ad-hoc mid:  --config ... --from collect --experiment path/to/experiment\n"
            "             --config ... --from curate   --input path/to/telemetry\n"
            "             --config ... --from train-cnn --input path/to/curated\n"
        ),
    )
    p.add_argument(
        "--config", metavar="PATH", help="YAML pipeline config (required for new runs)"
    )
    p.add_argument(
        "--pipeline-dir",
        metavar="PATH",
        dest="pipeline_dir",
        help="Resume an existing pipeline run from its directory",
    )
    p.add_argument(
        "--from",
        metavar="STEP",
        dest="from_step",
        choices=["train", "collect", "curate", "train-cnn"],
        help="Start at this step (default: train)",
    )
    p.add_argument(
        "--to",
        metavar="STEP",
        dest="to_step",
        choices=["train", "collect", "curate", "train-cnn"],
        help="Stop after this step, inclusive (default: train-cnn)",
    )
    p.add_argument(
        "--experiment",
        metavar="PATH",
        help="RL experiment dir (required for ad-hoc --from collect)",
    )
    p.add_argument(
        "--input",
        metavar="PATH",
        help="Input dir (required for ad-hoc --from curate or --from train-cnn)",
    )
    p.add_argument(
        "--cnn-checkpoint",
        metavar="PATH",
        dest="cnn_checkpoint",
        help="Pretrained CNN backbone checkpoint passed to the train step",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print resolved commands without executing anything",
    )
    p.add_argument(
        "--display",
        type=int,
        metavar="N",
        help="X display socket number (e.g. 2 for DISPLAY=:2)",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Base output dir; pipeline dir created as <output>/pipeline_<timestamp>/ (default: artifacts/)",
    )
    p.set_defaults(func=cmd_pipeline)

    # ── pin ───────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "pin",
        help="Create named symlinks for frequently-used paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Creates persistent symlinks under scripts/pins/ so long absolute paths\n"
            "can be referenced by short stable names in subsequent commands.\n"
            "\n"
            "Pin a CNN checkpoint:\n"
            "  ./scripts/run.py pin --cnn-checkpoint /mnt/nas/runs/2026-04-09/cnn/best.pt\n"
            "  ./scripts/run.py train ... --cnn-checkpoint scripts/pins/cnn_checkpoint.pt\n"
            "\n"
            "Pin an experiment directory:\n"
            "  ./scripts/run.py pin --experiment /mnt/nas/experiments/2026-04-09_12-31-41\n"
            "  ./scripts/run.py collect --experiment scripts/pins/experiment ...\n"
            "\n"
            "List all current pins:\n"
            "  ./scripts/run.py pin --list\n"
        ),
    )
    p.add_argument(
        "--bundle",
        metavar="PATH",
        dest="bundle",
        help="Pin a deploy bundle directory  →  scripts/pins/latest_bundle",
    )
    p.add_argument(
        "--cnn-checkpoint",
        metavar="PATH",
        dest="cnn_checkpoint",
        help="Pin a CNN backbone checkpoint (.pt)  →  scripts/pins/cnn_checkpoint.pt",
    )
    p.add_argument(
        "--checkpoint",
        metavar="PATH",
        dest="checkpoint",
        help="Pin an RL policy checkpoint (.pt)  →  scripts/pins/checkpoint.pt",
    )
    p.add_argument(
        "--experiment",
        metavar="PATH",
        dest="experiment",
        help="Pin an RL experiment directory  →  scripts/pins/experiment",
    )
    p.add_argument(
        "--input",
        metavar="PATH",
        dest="input",
        help="Pin a telemetry or curated-data directory  →  scripts/pins/input",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        dest="output",
        help="Pin a base output directory  →  scripts/pins/output",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all currently pinned paths and their targets",
    )
    p.set_defaults(func=cmd_pin)

    # ── sweep ─────────────────────────────────────────────────────────────────
    p = sub.add_parser(
        "sweep",
        help="Run a grid of Train+Eval experiments defined by experiment_definition and summarise results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Run a grid of experiments sequentially (train → eval) and compare results.\n"
            "\n"
            "Sweep configs use experiment_definition to define a Cartesian product of\n"
            "parameter variations.  Each config_set is one grid dimension; the product\n"
            "of all config_sets produces the full experiment list automatically.\n"
            "\n"
            "See configs/sweep_example.yaml for an annotated example.\n"
            "\n"
            "Typical usage\n"
            "─────────────\n"
            "  New sweep:\n"
            "    ./scripts/run.py sweep --sweep configs/my_sweep.yaml\n"
            "\n"
            "  Dry run (print commands, create nothing):\n"
            "    ./scripts/run.py sweep --sweep configs/my_sweep.yaml --dry-run\n"
            "\n"
            "  Inspect materialised configs without training:\n"
            "    ./scripts/run.py sweep --sweep configs/my_sweep.yaml --create-configs-only\n"
            "\n"
            "  Resume a killed sweep from where it left off:\n"
            "    ./scripts/run.py sweep --resume sweeps/sweep_<name>_<timestamp>/\n"
            "\n"
            "  Resume, skip training for experiments that already trained (e.g. eval-only retry):\n"
            "    ./scripts/run.py sweep --resume sweeps/sweep_<name>_<ts>/ --retry-eval\n"
            "\n"
            "  Resume from a specific experiment:\n"
            "    ./scripts/run.py sweep --resume sweeps/sweep_<name>_<ts>/ --from-experiment exp_003\n"
            "\n"
            "Output\n"
            "──────\n"
            "  sweeps/sweep_<name>_<timestamp>/\n"
            "    sweep.yaml               expanded experiment list (for reproducibility)\n"
            "    sweep_state.json         per-experiment status, updated after each step\n"
            "    summary.json / .md       comparison table written when sweep finishes\n"
            "    experiments/\n"
            "      01_exp_001/\n"
            "        env_config.yaml      materialised env config (base + overrides)\n"
            "        agent_config.yaml    materialised agent config\n"
            "        milestones.json      env_transitions at first_approach/grasp/lift/success\n"
            "        skrl/                train.py output (checkpoints, TensorBoard)\n"
            "        evaluation/          evaluate.py output (results.json)\n"
        ),
    )
    p.add_argument(
        "--sweep",
        metavar="PATH",
        help="Sweep definition YAML (required for new sweeps)",
    )
    p.add_argument(
        "--resume",
        metavar="PATH",
        help="Resume an existing sweep from its directory",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Base output dir; sweep dir created as <output>/sweep_<name>_<ts>/ (default: sweeps/)",
    )
    p.add_argument(
        "--from-experiment",
        metavar="NAME",
        dest="from_experiment",
        help="Start (or restart) from this named experiment; earlier done experiments are skipped",
    )
    p.add_argument(
        "--retry-eval",
        action="store_true",
        dest="retry_eval",
        help=(
            "When resuming, skip the training subprocess for any experiment whose "
            "stored train_return_code is 0 and whose checkpoint is present on disk. "
            "Useful when all experiments trained successfully but eval failed."
        ),
    )
    p.add_argument(
        "--create-configs-only",
        action="store_true",
        dest="create_configs_only",
        help=(
            "Materialise env_config.yaml and agent_config.yaml for every experiment "
            "then exit without launching any training or evaluation"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print resolved commands for each experiment without executing anything",
    )
    p.add_argument(
        "--display",
        type=int,
        metavar="N",
        help="X display socket number (e.g. 2 for DISPLAY=:2)",
    )
    p.set_defaults(func=cmd_sweep)

    return parser


def main() -> None:
    header("so101_sim_to_real Pipeline CLI")
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
