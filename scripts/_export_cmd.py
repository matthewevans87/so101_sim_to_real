"""Shared builder for the export_bundle.py command line.

Single source of truth for the `isaaclab.sh -p export_bundle.py ...` invocation
used by both `scripts/run.py:cmd_export` and `scripts/pipeline.py:_build_export_cmd`.

The export contract is intentionally minimal: the experiment directory carries
all required inputs (env_config.yaml, skrl/best_agent.pt, cnn_checkpoint.pt).
The export step takes no overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def build_export_command(
    *,
    isaac_lab_path: str,
    task_root: Path,
    task: str,
    experiment_path: Path,
    output_dir: Path,
    torchscript: bool = False,
) -> List[str]:
    """Construct the export_bundle.py invocation.

    All parameters are required and explicit. The exporter resolves its inputs
    solely from ``experiment_path``; no CNN checkpoint, env config, or other
    overrides are accepted at the command line.
    """
    cmd: List[str] = [
        f"{isaac_lab_path}/isaaclab.sh",
        "-p",
        str(task_root / "scripts" / "skrl" / "export_bundle.py"),
        "--task",
        task,
        "--experiment-path",
        str(experiment_path),
        "--output",
        str(output_dir),
        "--headless",
        "--enable_cameras",
    ]
    if torchscript:
        cmd.append("--torchscript")
    return cmd
