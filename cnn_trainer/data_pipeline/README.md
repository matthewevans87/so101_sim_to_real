# Telemetry Curation Pipeline

This directory contains the offline data curation pipeline for preparing raw telemetry collections for CNN backbone pretraining.

## Overview

The **curation script** (`curate.py`) processes raw policy rollout telemetry and produces balanced, train/val/test splits suitable for supervised CNN training. It addresses two key biases in collected data:

1. **Episode-level imbalance**: Raw collections may have unequal episode lengths and episode-step distributions.
2. **Repeated start states**: Early steps of episodes are over-represented (robots resetting to start configurations in parallel environments).

## Pipeline Stages

### 1. Shard Index Loading
- Reads metadata from `telemetry_metadata.json` to locate all NPZ shards
- Loads epoch-level metadata (episode_step, episode_id, env_id) — **RGB is not loaded**
- Returns a flat list of records, each mapping a (shard_path, row) pair to its metadata

### 2. Episode-Level Splitting
- Identifies unique (env_id, episode_id) pairs
- Deterministically splits episodes into train/val/test using a seeded RNG
- **No data leakage**: entire episodes stay in one split
- Fractions are configurable; defaults: train 80%, val 10%, test 10%

### 3. Step-Bin Rebalancing
Two-stage process to approximate a uniform episode_step distribution:

**Stage 1: Per-episode cap** (optional)
- If an episode contributes many samples (e.g., long episode), downsample to a limit
- Reduces within-episode correlation and limits start-state bias

**Stage 2: Step-bin cap**
- Group samples into bins of width `step_bin_width` along the episode_step axis
- Compute a per-bin cap: `floor(median_bin_count * bin_cap_factor)`
- Randomly downsample bins above the cap
- Keep under-represented bins intact

### 4. Manifest Construction & Reporting
- Aggregates curated samples into shard-centric manifests (JSON)
- Saves a snapshot of the config used
- Writes a detailed curation report with step histograms

## Input Format

### Telemetry Directory Structure
```
artifacts/telemetry/2026-03-24/
├── telemetry_metadata.json          # Required metadata index
└── telemetry_shards/
    ├── telemetry_00000.npz
    ├── telemetry_00001.npz
    └── ...
```

### Telemetry Metadata (`telemetry_metadata.json`)
```json
{
  "shards": [
    {"shard": "telemetry_00000.npz", "num_samples": 481, ...},
    ...
  ],
  ...
}
```

### NPZ Shard Format
Each `.npz` file contains:
- `rgb[S, H, W, 3]` — uint8 images (S = num_samples, H = height, W = width)
- `episode_step[S]` — int32, step within episode (0 = reset state)
- `episode_id[S]` — int32, episode identifier
- `env_id[S]` — int32, parallel environment index
- `is_cube_in_grip_position[S]` — bool, task label
- `cube_quat_gripzone_wxyz[S, 4]` — float32, orientation target
- `cube_in_camera_frame[S]` — bool, visibility label

## Output Format

### Directory Structure
```
artifacts/curated/2026-03-24/
├── train_manifest.json
├── val_manifest.json
├── test_manifest.json
├── curation_report.json
└── curation_config.yaml          # Snapshot of input config
```

### Manifest Files (JSON)
```json
{
  "version": "1",
  "split": "train",
  "telemetry_dir": "/abs/path/to/telemetry",
  "total_samples": 12345,
  "shards": [
    {
      "path": "telemetry_shards/telemetry_00000.npz",
      "rows": [0, 5, 8, 12, ...]
    },
    ...
  ]
}
```

Paths in shards are **relative to telemetry_dir** for portability.

### Curation Report (`curation_report.json`)
Machine-readable summary with:
- Raw total samples
- Per-split episode and sample counts (pre- and post-rebalance)
- Step histograms before/after rebalancing
- Seed and configuration used

Example excerpt:
```json
{
  "raw_total_samples": 481,
  "splits": {
    "train": {
      "episodes": 7,
      "samples_pre_rebalance": 393,
      "samples_post_rebalance": 210,
      "step_histogram_post": {"0": 30, "1": 28, "2": 26, ...}
    },
    ...
  }
}
```

## Configuration

Curation is controlled via a YAML config file passed to `curate.py`. Key section: `curation`:

```yaml
seed: 42

curation:
  step_bin_width: 50              # Bin size for step-index histogram
  bin_cap_factor: 1.5             # cap = floor(median * factor)
  val_fraction: 0.1               # Fraction of episodes to val
  test_fraction: 0.1              # Fraction of episodes to test
  split_seed: 0                   # RNG seed for episode shuffle
  max_samples_per_episode: 30     # (Optional) per-episode downsample limit
```

### Configuration Semantics

| Parameter                 | Meaning                                               | Typical Range |
| ------------------------- | ----------------------------------------------------- | ------------- |
| `step_bin_width`          | Granularity of step-index histogram                   | 20–100        |
| `bin_cap_factor`          | Aggressiveness of rebalancing (>1 keeps more samples) | 1.0–2.0       |
| `val_fraction`            | Fraction of episodes reserved for validation          | 0.05–0.2      |
| `test_fraction`           | Fraction of episodes reserved for testing             | 0.05–0.2      |
| `split_seed`              | Seed for deterministic episode split                  | Any int       |
| `max_samples_per_episode` | Limit samples per episode (None = no limit)           | 10–50 or None |

## Usage

### Basic Usage
```bash
conda activate env_so101_direct
python cnn_trainer/data_pipeline/curate.py \
    --telemetry-dir artifacts/telemetry/2026-03-24 \
    --output-dir    artifacts/curated/2026-03-24 \
    --config        configs/cnn_pretrain.yaml
```

### Via `run.sh`
```bash
./scripts/run.sh curate \
    --telemetry-dir artifacts/telemetry/2026-03-24 \
    --curated-dir   artifacts/curated/2026-03-24
```

### With Explicit Seed Override
```bash
python cnn_trainer/data_pipeline/curate.py \
    --telemetry-dir artifacts/telemetry/2026-03-24 \
    --output-dir    artifacts/curated/2026-03-24 \
    --config        configs/cnn_pretrain.yaml \
    --seed          100
```

## Understanding the Report

### Step Histograms
`step_histogram_post` in the report shows the distribution of `episode_step` indices after rebalancing:

```json
"step_histogram_post": {
  "0": 30,   # 30 samples at step 0–49
  "1": 28,   # 28 samples at step 50–99
  "2": 26,   # 26 samples at step 100–149
  ...
}
```

A nearly-flat histogram indicates successful rebalancing. Spikes suggest:
- Episodes are clustered in length (e.g., many episodes end at step 4)
- `bin_cap_factor` or `step_bin_width` may need adjustment

### Sample Count Changes
Comparing `samples_pre_rebalance` to `samples_post_rebalance`:

```
pre_rebalance:   393 samples
├─ max() per bin: 180
post_rebalance:  210 samples
├─ max() per bin: ~30 (balanced)
```

The script randomly downsampled from bins that were over-represented.

## Reproducibility

All stochastic choices are deterministic and seeded:
- **Episode split**: controlled by `cfg.curation.split_seed`
- **Per-bin downsampling**: controlled by `cfg.seed`

To reproduce a curation run, save and rerun with the same config and seed.

## Next Steps

After curation:
1. **Train CNN**: pass the curated directory to `cnn_trainer/train.py`
2. **RL training with frozen backbone**: use the trained CNN checkpoint in RL policy training

See `cnn_trainer/train.py` README for supervised CNN training details.
