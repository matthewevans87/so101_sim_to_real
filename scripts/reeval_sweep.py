#!/usr/bin/env python3
"""
Re-evaluate every "done" experiment in an existing sweep using the current
``evaluate.py`` (Phase B audited eval) without overwriting the original
``evaluation/`` directory.

For each experiment the script:

  1. Builds the same eval command the sweep originally used (from the sweep's
     ``sweep.yaml::eval`` section).
  2. Appends ``--eval-subdir <eval_subdir>`` so eval outputs land under
     ``<exp_dir>/<eval_subdir>/`` (default ``evaluation_v2``).
  3. Skips experiments whose v2 results.json already exists, unless ``--force``.

After all experiments finish, ``SweepOrchestrator.generate_summary`` is invoked
twice:

  * once with the original ``evaluation`` subdir (regenerates ``summary.json`` /
    ``summary.md`` with the latest schema, drawing on the original eval data),
  * once with the new ``evaluation_v2`` subdir (writes ``summary_v2.json`` /
    ``summary_v2.md``).

The two summaries can then be diffed for a side-by-side comparison.

Usage
-----

    ISAAC_LAB_PATH=/path/to/IsaacLab \
    conda run -n env_isaaclab --no-capture-output \
        scripts/reeval_sweep.py \
        --sweep-dir /mnt/.../sweeps/sweep_reward_scale_grid_20260422_110823

Optional flags
--------------

    --eval-subdir NAME    Output subdir per experiment (default: evaluation_v2)
    --out-suffix STR      Filename suffix for summaries (default: _v2)
    --force               Re-run eval even if <eval_subdir>/results.json exists
    --only NAME [NAME...] Limit to specific experiment names
    --dry-run             Print commands without executing them
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow `import sweep` from this script's directory.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Reuse the orchestrator so re-eval honours the exact same conventions as the
# original sweep (env injection, isaaclab.sh wrapper, eval cmd template).
from sweep import (  # type: ignore[import-not-found]
    SweepOrchestrator,
    _error,
    _header,
    _info,
    _run_step_subprocess,
    _success,
)


def _require_isaac_lab() -> str:
    path = os.environ.get("ISAAC_LAB_PATH", "").strip()
    if not path:
        _error("ISAAC_LAB_PATH environment variable is not set.")
        _error("  export ISAAC_LAB_PATH=/path/to/IsaacLab")
        sys.exit(1)
    if not Path(path).is_dir():
        _error(f"ISAAC_LAB_PATH does not exist or is not a directory: {path}")
        sys.exit(1)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a finished sweep into a sibling output directory "
            "without overwriting the original eval results."
        )
    )
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Path to a finished sweep directory (must contain sweep.yaml + sweep_state.json).",
    )
    parser.add_argument(
        "--eval-subdir",
        type=str,
        default="evaluation_v2",
        help="Per-experiment subdir for v2 eval outputs (default: evaluation_v2).",
    )
    parser.add_argument(
        "--out-suffix",
        type=str,
        default="_v2",
        help="Filename suffix for the v2 summary files (default: _v2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run eval even if <eval_subdir>/results.json already exists.",
    )
    parser.add_argument(
        "--only",
        type=str,
        nargs="+",
        default=None,
        help="Limit re-eval to the given experiment names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.is_dir():
        _error(f"--sweep-dir is not a directory: {sweep_dir}")
        return 1

    if "/" in args.eval_subdir or args.eval_subdir.startswith("."):
        _error(
            f"--eval-subdir must be a single non-empty dir name (got {args.eval_subdir!r})"
        )
        return 1
    if args.eval_subdir == "evaluation":
        _error(
            "--eval-subdir='evaluation' would overwrite the original results. "
            "Choose a different name (default: evaluation_v2)."
        )
        return 1

    isaac_lab_path = _require_isaac_lab()
    project_root = Path(__file__).resolve().parent.parent

    orch = SweepOrchestrator.from_existing(
        sweep_dir=sweep_dir,
        isaac_lab_path=isaac_lab_path,
        project_root=project_root,
    )
    assert orch._state is not None  # populated by from_existing

    # ── select experiments ────────────────────────────────────────────────────
    selected: list[tuple[str, dict]] = []
    for name, info in orch._state["experiments"].items():
        if args.only is not None and name not in args.only:
            continue
        if info["status"] != "done":
            _info(f"  skip '{name}': status={info['status']!r} (not 'done')")
            continue
        selected.append((name, info))

    if not selected:
        _error("No 'done' experiments selected for re-evaluation.")
        return 1

    _header(
        f"Re-evaluating {len(selected)} experiment(s) into '{args.eval_subdir}/' "
        f"(force={args.force}, dry_run={args.dry_run})"
    )

    # ── per-experiment eval ───────────────────────────────────────────────────
    failures: list[str] = []
    for idx, (name, info) in enumerate(selected, start=1):
        exp_dir = Path(info["exp_dir"])
        v2_results = exp_dir / args.eval_subdir / "results.json"
        v2_log = exp_dir / f"eval{args.out_suffix}.log"

        _header(f"  [{idx:02d}/{len(selected):02d}] {name}")
        _info(f"  exp_dir: {exp_dir}")

        if v2_results.is_file() and not args.force:
            _info(
                f"  skip: {v2_results} already exists (use --force to re-run)."
            )
            continue

        eval_cmd = orch._build_eval_cmd(exp_dir) + [
            "--eval-subdir",
            args.eval_subdir,
        ]
        gui_env = orch._get_gui_env(exp_dir)

        _info(f"  command: {' '.join(str(c) for c in eval_cmd)}")
        _info(f"  log:     {v2_log}")

        if args.dry_run:
            continue

        rc = _run_step_subprocess(eval_cmd, env=gui_env, log_path=v2_log)
        if rc != 0 or not v2_results.is_file():
            _error(
                f"  re-eval FAILED for '{name}' (rc={rc}, results_present={v2_results.is_file()}). "
                f"See {v2_log}."
            )
            failures.append(name)
            continue

        _success(f"  re-eval done: {v2_results}")

    # ── summaries ────────────────────────────────────────────────────────────
    if not args.dry_run:
        _header("Regenerating summaries")
        # Original summary refresh (uses current schema; reads original
        # evaluation/ data unchanged).
        orch.generate_summary(eval_subdir="evaluation", out_suffix="")
        # v2 summary from the freshly written evaluation_v2/ data.
        orch.generate_summary(
            eval_subdir=args.eval_subdir, out_suffix=args.out_suffix
        )

    # ── report ────────────────────────────────────────────────────────────────
    _header("Re-eval complete")
    _info(f"  generated_at: {datetime.now(timezone.utc).isoformat()}")
    _info(f"  sweep_dir:    {sweep_dir}")
    _info(
        f"  v2 summary:   {sweep_dir / ('summary' + args.out_suffix + '.json')}"
    )
    _info(
        f"  v2 summary:   {sweep_dir / ('summary' + args.out_suffix + '.md')}"
    )
    if failures:
        _error(f"  failures:     {len(failures)} experiments — {failures}")
        return 1
    _success("  no failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
