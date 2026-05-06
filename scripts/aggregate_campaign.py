#!/usr/bin/env python3
"""
Aggregate a multi-sub-sweep ablation campaign into LaTeX-ready CSVs.

Walks a campaign directory of the form::

    <campaign-dir>/
        sweep_<name1>_<ts>/
            sweep.yaml
            sweep_state.json
            experiments/...
            summary.json                    (optional, refreshed by this script)
            summary_aggregated.json         (optional, refreshed by this script)
        sweep_<name2>_<ts>/
            ...

For each sub-sweep:
    1. (Re)generates ``summary.json`` and ``summary_aggregated.json`` via
       :class:`SweepOrchestrator` so partial / in-progress sub-sweeps are
       brought up to date.
    2. Aggregates per-condition headline metrics into a single wide CSV
       (``campaign-summary.csv``).
    3. Aggregates per-condition termination causes into a long-form tidy CSV
       (``termination-causes.csv``) using per-experiment ``summary.json`` data.

Headers and string fields are sanitised by replacing ``_`` with ``-`` so the
output imports cleanly into LaTeX (``pgfplotstable`` / ``csvsimple``) without
escaping issues.

Usage
-----

    scripts/aggregate_campaign.py \\
        --campaign-dir /mnt/nas_1/.../sweeps/ablations/2026-04-24
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow `import sweep` from this script's directory (mirrors reeval_sweep.py).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from sweep import SweepOrchestrator  # type: ignore[import-not-found]

DEFAULT_CAMPAIGN_DIR = Path(
    "/mnt/nas_1/matthew-evans/so101_sim_to_real/sweeps/ablations/2026-04-24"
)

# Canonical metric ordering for the wide CSV. Source-of-truth keys are the
# underscore form used inside ``summary_aggregated.json``; the CSV header
# substitutes ``-`` for ``_``.
CANONICAL_METRICS: Tuple[str, ...] = (
    "success_rate",
    "lift_rate",
    "drop_rate",
    "mean_reward",
    "mean_episode_length",
    "mean_cube_bump",
    "mean_time_to_lift",
    "milestone_first_approach",
    "milestone_first_grasp",
    "milestone_first_lift",
    "milestone_first_success",
)

STAT_FIELDS: Tuple[str, ...] = ("mean", "std", "stderr", "min", "max", "n")


# ─────────────────────────────────────────────────────────────────────────────
# Sanitisation
# ─────────────────────────────────────────────────────────────────────────────


def _sanitise(s: str) -> str:
    """Replace LaTeX-hostile underscores in a header or string field."""
    return s.replace("_", "-")


def _fmt_num(v: Any) -> str:
    """Format a numeric cell as repr(float). Empty string for None / NaN."""
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if math.isnan(f):
        return ""
    return repr(f)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-sweep refresh
# ─────────────────────────────────────────────────────────────────────────────


def _discover_sub_sweeps(campaign_dir: Path) -> List[Path]:
    candidates = sorted(campaign_dir.glob("sweep_*"))
    out: List[Path] = []
    for p in candidates:
        if not p.is_dir():
            continue
        if not (p / "sweep_state.json").is_file():
            print(f"[skip] {p.name}: no sweep_state.json", file=sys.stderr)
            continue
        if not (p / "sweep.yaml").is_file():
            print(f"[skip] {p.name}: no sweep.yaml", file=sys.stderr)
            continue
        out.append(p)
    return out


def _refresh_sub_sweep(
    sub_sweep_dir: Path,
    eval_subdir: str,
    isaac_lab_path: str,
    project_root: Path,
) -> bool:
    """Regenerate ``summary.json`` and ``summary_aggregated.json``.

    Returns True on success; False if the orchestrator could not be loaded.
    The summary methods do not require a working Isaac install — they only
    read ``results.json`` / ``milestones.json`` from each experiment dir.
    """
    try:
        orch = SweepOrchestrator.from_existing(
            sweep_dir=sub_sweep_dir,
            isaac_lab_path=isaac_lab_path,
            project_root=project_root,
        )
    except SystemExit:
        # from_existing calls sys.exit on missing files; we already filtered
        # those out, but be defensive.
        print(f"[WARN] {sub_sweep_dir.name}: from_existing failed", file=sys.stderr)
        return False
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] {sub_sweep_dir.name}: orchestrator load failed: {exc}",
            file=sys.stderr,
        )
        return False

    try:
        orch.generate_summary(eval_subdir=eval_subdir, out_suffix="")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] {sub_sweep_dir.name}: generate_summary failed: {exc}",
            file=sys.stderr,
        )
    try:
        orch.generate_aggregated_summary(eval_subdir=eval_subdir, out_suffix="")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[WARN] {sub_sweep_dir.name}: generate_aggregated_summary failed: "
            f"{exc}",
            file=sys.stderr,
        )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Wide CSV (per-condition aggregated metrics)
# ─────────────────────────────────────────────────────────────────────────────


def _format_rewards_effected(env_overrides: Optional[Dict[str, Any]]) -> str:
    """Format the list of reward overrides for this condition as e.g.
    ``grasp-phase[absolute];approach-phase[progressive]``.

    Reads the ``rewards`` list inside the experiment's ``env_overrides`` and
    emits one ``type[id]`` token per override entry (``[id]`` is omitted when
    the entry has no ``id`` field). Underscores are sanitised. Returns the
    empty string when no reward overrides are present.
    """
    if not env_overrides:
        return ""
    rewards = env_overrides.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        return ""
    tokens: List[str] = []
    for entry in rewards:
        if not isinstance(entry, dict):
            continue
        rtype = entry.get("type")
        if rtype is None:
            continue
        rid = entry.get("id")
        token = _sanitise(str(rtype))
        if rid is not None:
            token = f"{token}[{_sanitise(str(rid))}]"
        tokens.append(token)
    return ";".join(tokens)


def _wide_header() -> List[str]:
    identity = [
        "sub-sweep",
        "condition",
        "n-total",
        "n-done",
        "n-failed",
        "seeds",
        "cnn-checkpoint",
        "rewards-effected",
    ]
    metric_cols: List[str] = []
    for m in CANONICAL_METRICS:
        m_safe = _sanitise(m)
        for f in STAT_FIELDS:
            metric_cols.append(f"{m_safe}-{f}")
    return identity + metric_cols


def _wide_rows_for_sub_sweep(
    sub_sweep_dir: Path, agg: Dict[str, Any]
) -> List[List[str]]:
    rows: List[List[str]] = []
    sub_name = _sanitise(sub_sweep_dir.name)
    for grp in agg.get("groups") or []:
        condition = _sanitise(str(grp.get("label", "")))
        seeds = grp.get("seeds") or []
        seeds_str = ";".join(str(s) for s in seeds if s is not None)
        ckpt = grp.get("cnn_checkpoint")
        ckpt_str = _sanitise(os.path.basename(str(ckpt))) if ckpt else ""
        rewards_str = _format_rewards_effected(grp.get("env_overrides"))

        identity = [
            sub_name,
            condition,
            str(grp.get("n_total", "") or ""),
            str(grp.get("n_done", "") or ""),
            str(grp.get("n_failed", "") or ""),
            seeds_str,
            ckpt_str,
            rewards_str,
        ]

        metric_cells: List[str] = []
        metrics = grp.get("metrics") or {}
        for m in CANONICAL_METRICS:
            stats = metrics.get(m) or {}
            for f in STAT_FIELDS:
                v = stats.get(f)
                if f == "n":
                    metric_cells.append(str(int(v)) if v is not None else "")
                else:
                    metric_cells.append(_fmt_num(v))

        rows.append(identity + metric_cells)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Tidy termination CSV (per condition × cause)
# ─────────────────────────────────────────────────────────────────────────────


TERM_HEADER = [
    "sub-sweep",
    "condition",
    "kind",
    "cause",
    "is-success",
    "n-seeds-with-data",
    "total-episodes",
    "fraction-mean",
    "fraction-std",
    "fraction-stderr",
    "fraction-min",
    "fraction-max",
]


def _aggregate_fractions(values: List[float]) -> Dict[str, Optional[float]]:
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "stderr": None, "min": None, "max": None}
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = var**0.5
        stderr = std / (n**0.5)
    else:
        std = 0.0
        stderr = 0.0
    return {
        "mean": mean,
        "std": std,
        "stderr": stderr,
        "min": min(values),
        "max": max(values),
    }


def _tidy_rows_for_sub_sweep(
    sub_sweep_dir: Path,
    agg: Dict[str, Any],
    summary: Dict[str, Any],
) -> List[List[str]]:
    """
    For each (condition, kind, cause) emit one row whose statistics are
    computed across the per-seed fractions ``count_i / sum(counts_i)``.
    """
    sub_name = _sanitise(sub_sweep_dir.name)

    # Index per-experiment data by name for O(1) lookup.
    exp_by_name: Dict[str, Dict[str, Any]] = {
        e["name"]: e for e in (summary.get("experiments") or [])
    }

    rows: List[List[str]] = []
    for grp in agg.get("groups") or []:
        condition = _sanitise(str(grp.get("label", "")))
        members: List[str] = list(grp.get("members") or [])

        # Collect per-seed (cause -> fraction) and (cause -> raw count) maps for
        # both 'primary' and 'flag' kinds.
        for kind, key in (
            ("primary", "termination_primary_counts"),
            ("flag", "termination_flag_counts"),
        ):
            # cause -> list[float] (per-seed fractions)
            per_cause_fractions: Dict[str, List[float]] = {}
            # cause -> int (cumulative raw count)
            per_cause_totals: Dict[str, int] = {}
            success_ids: List[Optional[str]] = []

            for name in members:
                exp = exp_by_name.get(name)
                if exp is None:
                    continue
                counts = exp.get(key) or {}
                if not counts:
                    continue
                total = sum(int(v) for v in counts.values())
                if total <= 0:
                    continue
                success_ids.append(exp.get("success_termination_id"))
                for cause, count in counts.items():
                    cnt = int(count)
                    per_cause_totals[cause] = per_cause_totals.get(cause, 0) + cnt
                    per_cause_fractions.setdefault(cause, []).append(cnt / total)

            # Ensure every cause that appeared in any seed gets a row, with
            # zeros for seeds where it was absent (so means reflect the full
            # member set, not just seeds that triggered the cause).
            n_members_with_data = sum(
                1 for name in members if (exp_by_name.get(name) or {}).get(key)
            )
            if n_members_with_data == 0:
                continue
            for cause, fracs in per_cause_fractions.items():
                # Pad with zeros for seeds that produced eval data but never
                # observed this cause.
                pad = n_members_with_data - len(fracs)
                if pad > 0:
                    fracs = fracs + [0.0] * pad
                stats = _aggregate_fractions(fracs)

                # Determine is-success: only meaningful for primary kind.
                # Treat as True iff this cause matches the success id reported
                # by every contributing seed (defensive — they should all
                # agree, but skip if they disagree).
                is_success = False
                if kind == "primary" and success_ids:
                    distinct = {sid for sid in success_ids if sid is not None}
                    if len(distinct) == 1 and cause in distinct:
                        is_success = True

                rows.append(
                    [
                        sub_name,
                        condition,
                        kind,
                        _sanitise(str(cause)),
                        "true" if is_success else "false",
                        str(len(fracs)),
                        str(per_cause_totals[cause]),
                        _fmt_num(stats["mean"]),
                        _fmt_num(stats["std"]),
                        _fmt_num(stats["stderr"]),
                        _fmt_num(stats["min"]),
                        _fmt_num(stats["max"]),
                    ]
                )

    # Stable sort: sub-sweep, condition, kind (primary before flag), -mean.
    kind_order = {"primary": 0, "flag": 1}

    def _sort_key(r: List[str]) -> Tuple[str, str, int, float]:
        try:
            mean = -float(r[7])
        except ValueError:
            mean = 0.0
        return (r[0], r[1], kind_order.get(r[2], 99), mean)

    rows.sort(key=_sort_key)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV writers
# ─────────────────────────────────────────────────────────────────────────────


def _write_csv(path: Path, header: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description="Aggregate an ablation campaign into LaTeX-ready CSVs.",
    )
    p.add_argument(
        "--campaign-dir",
        type=Path,
        default=DEFAULT_CAMPAIGN_DIR,
        help=f"Campaign root containing sweep_* sub-dirs "
        f"(default: {DEFAULT_CAMPAIGN_DIR}).",
    )
    p.add_argument(
        "--out-name",
        type=str,
        default="campaign-summary.csv",
        help="Filename for the wide per-condition CSV "
        "(written into --campaign-dir).",
    )
    p.add_argument(
        "--terminations-name",
        type=str,
        default="termination-causes.csv",
        help="Filename for the tidy termination-cause CSV "
        "(written into --campaign-dir).",
    )
    p.add_argument(
        "--meta-name",
        type=str,
        default="campaign-summary.meta.json",
        help="Filename for the provenance JSON.",
    )
    p.add_argument(
        "--eval-subdir",
        type=str,
        default="evaluation",
        help="Per-experiment eval output subdir (default: evaluation).",
    )
    p.add_argument(
        "--refresh",
        dest="refresh",
        action="store_true",
        help="Regenerate per-sub-sweep summary files before aggregation " "(default).",
    )
    p.add_argument(
        "--no-refresh",
        dest="refresh",
        action="store_false",
        help="Read existing summary files only; do not regenerate.",
    )
    p.set_defaults(refresh=True)
    args = p.parse_args()

    campaign_dir: Path = args.campaign_dir.resolve()
    if not campaign_dir.is_dir():
        print(f"[ERROR] campaign-dir does not exist: {campaign_dir}", file=sys.stderr)
        return 2

    # SweepOrchestrator constructor requires these; summary generation does
    # not actually use them. Provide pass-through defaults so the script does
    # not require a working Isaac install just to read JSON files.
    isaac_lab_path = os.environ.get("ISAAC_LAB_PATH", "")
    project_root = Path(__file__).resolve().parent.parent

    sub_sweeps = _discover_sub_sweeps(campaign_dir)
    if not sub_sweeps:
        print(
            f"[ERROR] no sweep_* sub-dirs found under {campaign_dir}", file=sys.stderr
        )
        return 2

    print(f"[campaign] {campaign_dir}")
    print(f"[campaign] {len(sub_sweeps)} sub-sweep(s) discovered")
    if args.refresh:
        print("[campaign] refreshing per-sub-sweep summary files…")
        for ss in sub_sweeps:
            print(f"  - refresh: {ss.name}")
            _refresh_sub_sweep(ss, args.eval_subdir, isaac_lab_path, project_root)

    wide_rows: List[List[str]] = []
    tidy_rows: List[List[str]] = []
    meta_sub: List[Dict[str, Any]] = []

    print()
    print(f"{'sub-sweep':60s}  done/total  failed  conditions")
    print(f"{'-' * 60}  ----------  ------  ----------")

    for ss in sub_sweeps:
        agg_path = ss / "summary_aggregated.json"
        sum_path = ss / "summary.json"
        if not agg_path.is_file():
            print(f"[WARN] {ss.name}: no summary_aggregated.json — skipping")
            continue
        try:
            with open(agg_path) as f:
                agg = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] {ss.name}: cannot read {agg_path.name}: {exc}")
            continue

        summary: Dict[str, Any] = {"experiments": []}
        if sum_path.is_file():
            try:
                with open(sum_path) as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[WARN] {ss.name}: cannot read summary.json: {exc}")

        ss_wide = _wide_rows_for_sub_sweep(ss, agg)
        ss_tidy = _tidy_rows_for_sub_sweep(ss, agg, summary)
        wide_rows.extend(ss_wide)
        tidy_rows.extend(ss_tidy)

        # Stats block for stdout / meta.
        n_total = sum(int(g.get("n_total") or 0) for g in agg.get("groups") or [])
        n_done = sum(int(g.get("n_done") or 0) for g in agg.get("groups") or [])
        n_failed = sum(int(g.get("n_failed") or 0) for g in agg.get("groups") or [])
        n_groups = len(agg.get("groups") or [])
        print(
            f"{ss.name[:60]:60s}  {n_done:4d}/{n_total:<5d} {n_failed:6d}  {n_groups:10d}"
        )
        meta_sub.append(
            {
                "name": ss.name,
                "n_total": n_total,
                "n_done": n_done,
                "n_failed": n_failed,
                "n_conditions": n_groups,
            }
        )

    out_csv = campaign_dir / args.out_name
    out_term = campaign_dir / args.terminations_name
    out_meta = campaign_dir / args.meta_name

    _write_csv(out_csv, _wide_header(), wide_rows)
    _write_csv(out_term, TERM_HEADER, tidy_rows)
    with open(out_meta, "w") as f:
        json.dump(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "campaign_dir": str(campaign_dir),
                "eval_subdir": args.eval_subdir,
                "refresh": bool(args.refresh),
                "wide_csv": out_csv.name,
                "termination_csv": out_term.name,
                "n_sub_sweeps": len(meta_sub),
                "sub_sweeps": meta_sub,
            },
            f,
            indent=2,
        )

    print()
    print(f"[wrote] {out_csv}  ({len(wide_rows)} rows)")
    print(f"[wrote] {out_term}  ({len(tidy_rows)} rows)")
    print(f"[wrote] {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
