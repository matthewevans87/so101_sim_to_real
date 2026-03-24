#!/usr/bin/env python3
"""Offline telemetry dataset curation.

Given a raw telemetry directory produced by collect_telemetry.py, this script:
  1. Reads all shards (loading only metadata columns, not full RGB).
  2. Splits episodes into train / val / test at the episode level to prevent
     data leakage across splits.
  3. Applies post-collection step-bin rebalancing within each split to
     approximate a uniform distribution of episode_step indices, reducing bias
     from repeated start-state configurations and variable episode lengths.
  4. Writes curated dataset manifests (JSON) and a machine-readable curation
     report.

All stochastic choices (episode shuffle, per-bin downsampling) are seeded from
cfg.curation.split_seed and cfg.seed respectively, ensuring reproducibility.

Output layout (--output-dir)::

    train_manifest.json
    val_manifest.json
    test_manifest.json
    curation_report.json
    curation_config.yaml   ← snapshot of the config used

Manifest format::

    {
        "version": "1",
        "split": "train",
        "telemetry_dir": "/abs/path/to/telemetry",
        "total_samples": N,
        "shards": [
            {"path": "relative/path/shard_00000.npz", "rows": [0, 5, ...]},
            ...
        ]
    }

Usage::

    python cnn_trainer/data_pipeline/curate.py \\
        --telemetry-dir artifacts/telemetry/2026-03-24 \\
        --output-dir    artifacts/curated/2026-03-24    \\
        --config        configs/cnn_pretrain.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _require(d: dict, *keys: str, section: str = "") -> None:
    """Raise ValueError if any key is absent or None in *d*."""
    prefix = f"{section}." if section else ""
    for k in keys:
        if k not in d or d[k] is None:
            raise ValueError(
                f"cfg.{prefix}{k} must be set explicitly; no default is allowed."
            )


def _validate_config(cfg: dict) -> None:
    _require(cfg, "seed")
    cur = cfg.get("curation")
    if cur is None:
        raise ValueError("cfg.curation section is required")
    _require(
        cur,
        "step_bin_width",
        "bin_cap_factor",
        "val_fraction",
        "test_fraction",
        "split_seed",
        section="curation",
    )
    if not (0 < cur["val_fraction"] < 1):
        raise ValueError("curation.val_fraction must be in (0, 1)")
    if not (0 < cur["test_fraction"] < 1):
        raise ValueError("curation.test_fraction must be in (0, 1)")
    if cur["val_fraction"] + cur["test_fraction"] >= 1.0:
        raise ValueError("curation.val_fraction + test_fraction must be < 1")
    if int(cur["step_bin_width"]) < 1:
        raise ValueError("curation.step_bin_width must be >= 1")
    if float(cur["bin_cap_factor"]) <= 0:
        raise ValueError("curation.bin_cap_factor must be > 0")


# ── Shard index loading ───────────────────────────────────────────────────────


def _load_shard_index(metadata: dict, telemetry_dir: Path) -> list[dict]:
    """Load minimal metadata columns from every shard.

    Returns a list of records::

        {"shard_path": str, "row": int, "episode_step": int,
         "episode_id": int, "env_id": int}

    Only the three integer arrays are loaded; full RGB is not read here.
    """
    records: list[dict] = []
    _SHARD_SUBDIR = "telemetry_shards"
    for shard_info in metadata["shards"]:
        # Metadata stores just the filename under key "shard"; shards live in
        # a "telemetry_shards/" subdirectory of the telemetry root.
        shard_filename = shard_info["shard"]
        shard_path = telemetry_dir / _SHARD_SUBDIR / shard_filename
        if not shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        data = np.load(shard_path, allow_pickle=False)
        episode_steps = data["episode_step"]  # (S,) int32
        episode_ids = data["episode_id"]  # (S,) int32
        env_ids = data["env_id"]  # (S,) int32
        shard_path_str = str(shard_path)
        for row in range(len(episode_steps)):
            records.append(
                {
                    "shard_path": shard_path_str,
                    "row": int(row),
                    "episode_step": int(episode_steps[row]),
                    "episode_id": int(episode_ids[row]),
                    "env_id": int(env_ids[row]),
                }
            )
    return records


# ── Episode-level splitting ───────────────────────────────────────────────────


def _episode_key(rec: dict) -> tuple[int, int]:
    return (rec["env_id"], rec["episode_id"])


def _split_episodes(
    records: list[dict],
    val_fraction: float,
    test_fraction: float,
    split_seed: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return (train_episodes, val_episodes, test_episodes).

    Each set contains (env_id, episode_id) tuples.  Sorting before shuffling
    ensures the split is deterministic regardless of insertion order.
    """
    all_episodes: list[tuple[int, int]] = sorted(set(_episode_key(r) for r in records))
    rng = random.Random(split_seed)
    rng.shuffle(all_episodes)
    n = len(all_episodes)
    n_val = max(1, math.floor(n * val_fraction))
    n_test = max(1, math.floor(n * test_fraction))
    val_eps = set(all_episodes[:n_val])
    test_eps = set(all_episodes[n_val : n_val + n_test])
    train_eps = set(all_episodes[n_val + n_test :])
    if not train_eps:
        raise RuntimeError(
            "No training episodes remain after val/test split. "
            "Provide more episodes or reduce val_fraction/test_fraction."
        )
    return train_eps, val_eps, test_eps


# ── Post-collection step-bin rebalancing ──────────────────────────────────────


def _rebalance_by_step_bin(
    records: list[dict],
    step_bin_width: int,
    bin_cap_factor: float,
    rng: random.Random,
    max_samples_per_episode: int | None,
) -> list[dict]:
    """Approximate a uniform step-index distribution.

    Two-stage approach:

    1. **Per-episode cap** (optional): for each (env_id, episode_id) pair,
       if the episode contributes more than *max_samples_per_episode* samples,
       randomly downsample to that limit.  This reduces within-episode
       correlation and limits the over-representation of the starting pose
       across many parallel environments.

    2. **Step-bin cap**: group samples into bins of width *step_bin_width*
       along the episode_step axis.  Compute a per-bin cap as
       ``floor(median_of_nonempty_bin_counts * bin_cap_factor)``.
       Bins above the cap are randomly downsampled; under-represented bins
       are kept intact.
    """
    # ── Stage 1: per-episode cap ──
    if max_samples_per_episode is not None and max_samples_per_episode > 0:
        by_episode: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for rec in records:
            by_episode[_episode_key(rec)].append(rec)
        capped: list[dict] = []
        for ep_recs in by_episode.values():
            if len(ep_recs) > max_samples_per_episode:
                ep_recs = rng.sample(ep_recs, max_samples_per_episode)
            capped.extend(ep_recs)
        records = capped

    # ── Stage 2: step-bin cap ──
    by_bin: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        bin_idx = rec["episode_step"] // step_bin_width
        by_bin[bin_idx].append(rec)

    bin_counts = [len(v) for v in by_bin.values()]
    median_count = float(np.median(bin_counts))
    cap = max(1, math.floor(median_count * bin_cap_factor))

    balanced: list[dict] = []
    for bin_recs in by_bin.values():
        if len(bin_recs) > cap:
            balanced.extend(rng.sample(bin_recs, cap))
        else:
            balanced.extend(bin_recs)
    return balanced


# ── Manifest construction ─────────────────────────────────────────────────────


def _build_manifest(records: list[dict], split: str, telemetry_dir: Path) -> dict:
    """Aggregate records into a shard-centric manifest.

    Shard paths are stored relative to *telemetry_dir* for portability.
    """
    by_shard: dict[str, list[int]] = defaultdict(list)
    for rec in records:
        by_shard[rec["shard_path"]].append(rec["row"])

    shards = []
    for shard_abs, rows in sorted(by_shard.items()):
        rows.sort()
        try:
            rel_path = str(Path(shard_abs).relative_to(telemetry_dir))
        except ValueError:
            rel_path = shard_abs
        shards.append({"path": rel_path, "rows": rows})

    return {
        "version": "1",
        "split": split,
        "telemetry_dir": str(telemetry_dir),
        "total_samples": sum(len(s["rows"]) for s in shards),
        "shards": shards,
    }


# ── Histogram helper ─────────────────────────────────────────────────────────


def _step_histogram(records: list[dict], step_bin_width: int) -> dict[str, int]:
    hist: dict[int, int] = defaultdict(int)
    for rec in records:
        bin_idx = rec["episode_step"] // step_bin_width
        hist[bin_idx] += 1
    return {str(k): v for k, v in sorted(hist.items())}


# ── Entrypoint ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curate a telemetry dataset for CNN backbone pretraining."
    )
    parser.add_argument(
        "--telemetry-dir",
        required=True,
        help="Raw telemetry directory (must contain telemetry_metadata.json).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination for manifests, report, and config snapshot.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="CNN pretrain config YAML (configs/cnn_pretrain.yaml).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override cfg.seed for the curation RNG.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    _validate_config(cfg)

    if args.seed is not None:
        cfg["seed"] = args.seed
        print(f"[curate] Overriding config seed with CLI value: {args.seed}")

    telemetry_dir = Path(args.telemetry_dir).resolve()
    if not telemetry_dir.exists():
        raise FileNotFoundError(f"Telemetry directory not found: {telemetry_dir}")

    metadata_path = telemetry_dir / "telemetry_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"telemetry_metadata.json not found in: {telemetry_dir}"
        )
    with open(metadata_path) as f:
        telemetry_metadata = json.load(f)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cur = cfg["curation"]
    step_bin_width = int(cur["step_bin_width"])
    bin_cap_factor = float(cur["bin_cap_factor"])
    val_fraction = float(cur["val_fraction"])
    test_fraction = float(cur["test_fraction"])
    split_seed = int(cur["split_seed"])
    max_samples_per_episode = cur.get("max_samples_per_episode")
    if max_samples_per_episode is not None:
        max_samples_per_episode = int(max_samples_per_episode)

    print(f"[curate] Loading shard index from {telemetry_dir} …")
    all_records = _load_shard_index(telemetry_metadata, telemetry_dir)
    total_raw = len(all_records)
    print(
        f"[curate] {total_raw} raw samples across "
        f"{len(telemetry_metadata['shards'])} shards."
    )

    pre_hist_all = _step_histogram(all_records, step_bin_width)

    print(
        f"[curate] Splitting episodes "
        f"(val={val_fraction}, test={test_fraction}, split_seed={split_seed}) …"
    )
    train_eps, val_eps, test_eps = _split_episodes(
        all_records, val_fraction, test_fraction, split_seed
    )
    print(
        f"[curate] Episodes — train: {len(train_eps)}, "
        f"val: {len(val_eps)}, test: {len(test_eps)}"
    )

    train_recs = [r for r in all_records if _episode_key(r) in train_eps]
    val_recs = [r for r in all_records if _episode_key(r) in val_eps]
    test_recs = [r for r in all_records if _episode_key(r) in test_eps]

    print(
        f"[curate] Pre-rebalance — train: {len(train_recs)}, "
        f"val: {len(val_recs)}, test: {len(test_recs)}"
    )

    rng = random.Random(cfg["seed"])
    train_recs = _rebalance_by_step_bin(
        train_recs, step_bin_width, bin_cap_factor, rng, max_samples_per_episode
    )
    val_recs = _rebalance_by_step_bin(
        val_recs, step_bin_width, bin_cap_factor, rng, max_samples_per_episode
    )
    test_recs = _rebalance_by_step_bin(
        test_recs, step_bin_width, bin_cap_factor, rng, max_samples_per_episode
    )

    print(
        f"[curate] Post-rebalance — train: {len(train_recs)}, "
        f"val: {len(val_recs)}, test: {len(test_recs)}"
    )

    # ── Write manifests ──
    for split, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        manifest = _build_manifest(recs, split, telemetry_dir)
        manifest_path = output_dir / f"{split}_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(
            f"[curate] {split:5s}  {manifest['total_samples']:6d} samples → {manifest_path}"
        )

    # ── Save config snapshot ──
    shutil.copy2(args.config, output_dir / "curation_config.yaml")

    # ── Write curation report ──
    report = {
        "telemetry_dir": str(telemetry_dir),
        "output_dir": str(output_dir),
        "config": args.config,
        "seed": cfg["seed"],
        "curation": {k: v for k, v in cur.items()},
        "raw_total_samples": total_raw,
        "step_histogram_pre_all": pre_hist_all,
        "splits": {
            "train": {
                "episodes": len(train_eps),
                "samples_pre_rebalance": sum(
                    1 for r in all_records if _episode_key(r) in train_eps
                ),
                "samples_post_rebalance": len(train_recs),
                "step_histogram_post": _step_histogram(train_recs, step_bin_width),
            },
            "val": {
                "episodes": len(val_eps),
                "samples_pre_rebalance": sum(
                    1 for r in all_records if _episode_key(r) in val_eps
                ),
                "samples_post_rebalance": len(val_recs),
                "step_histogram_post": _step_histogram(val_recs, step_bin_width),
            },
            "test": {
                "episodes": len(test_eps),
                "samples_pre_rebalance": sum(
                    1 for r in all_records if _episode_key(r) in test_eps
                ),
                "samples_post_rebalance": len(test_recs),
                "step_histogram_post": _step_histogram(test_recs, step_bin_width),
            },
        },
    }
    report_path = output_dir / "curation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[curate] Report written to {report_path}")


if __name__ == "__main__":
    import json  # already imported above; placed here for clarity in bare runs

    main()
