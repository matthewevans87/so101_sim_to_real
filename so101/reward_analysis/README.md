# Reward Dominance Analyser

A standalone tool for analytically computing best-case reward maxima and detecting reward pathologies from a `So101EnvParams`-compatible YAML config. **No Isaac Lab, Isaac Sim, or GPU required.**

## Usage

```bash
python -m so101.reward_analysis --config configs/frozen_cnn.yaml
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | *(required)* | Path to env config YAML |
| `--k FLOAT` | `2.0` | Dominance threshold: flag when `Σ(shaping max_episode) > k × terminal_scale` |
| `--json PATH` | — | Write full analysis as JSON to PATH |
| `--no-rich` | — | Plain-text output even if `rich` is installed |

Exit code is **1** when pathologies are detected (useful in pre-run checks or CI).

## What it computes

For each **enabled** reward, it analytically computes:

| Column | Meaning |
|---|---|
| `max/step` | Best possible reward at a single step (ideal physics) |
| `max/episode` | Best possible cumulative reward over the full episode |
| `terminal` | Whether this is a terminal reward type |
| `termNow` | Whether it actually terminates the episode (`terminate: false` → `no`) |
| `reachable` | Whether the terminal condition can be triggered analytically |

**`max_episode` by mode:**
- **absolute**: `max_step × num_steps`
- **progressive**: `metric_range × scale` (one-shot maximum delta — signal dies at optimum)
- **terminal, `terminate: true`**: `scale` (fires at most once)
- **terminal, `terminate: false`**: `scale × num_steps` (fires every step the condition is met)

Negative-scale penalty rewards show `max_episode = 0` — their best case is simply not firing.

## Pathologies detected

| Flag | Description |
|---|---|
| `NO_POSITIVE_TERMINAL` | No enabled terminal has a positive scale. Shaping rewards fill the episode unopposed. |
| `SHAPING_DOMINATES_TERMINAL` | `Σ(shaping max_episode) > k × terminal_scale`. Terminal bonus is too small. |
| `NONTERMINATING_MILESTONE_DOMINATES` | A `terminate=false` reward's episode max eclipses a true terminal by `> k×`. |
| `UNREACHABLE_TERMINAL` | Terminal condition can never be met (threshold exceeds metric maximum). |
| `PROGRESSIVE_NO_FALLBACK` | Progressive-only reward — gradient vanishes once the agent reaches the optimum. |
| `MODE_IGNORED` | `mode: progressive` set for a reward type that always behaves as absolute. |

## Example output (frozen_cnn.yaml)

```
Config : configs/frozen_cnn.yaml
Episode: 7.0s  |  Steps: 420

type                              mode          scale     max/step     max/episode     terminal    termNow    reachable
--------------------------------  ------------  --------  -----------  --------------  ----------  ---------  ---------
approach_phase_terminal           absolute      500       500          210,000         yes         no         yes
    NOTE: terminate=false — fires every step condition is met; max_episode shown as 500 × 420 steps
wrist_roll_pose  [gated]          progressive   1.00      1.00         420
    NOTE: mode=progressive set but not implemented — behaves as absolute
approach_distance                 progressive   5.00      30.00        30.00
...

PATHOLOGIES (7):
  [!] NONTERMINATING_MILESTONE_DOMINATES [approach_phase_terminal]: ...
  [!] SHAPING_DOMINATES_TERMINAL [success_lift_fraction_terminal]: ...
```

## Metric maxima reference

The best-case value for each metric at ideal physical state (d=0, a=1, gripper at target):

| Metric / reward type | Formula at optimum | Max value |
|---|---|---|
| `approach_phase` | `exp(0) × exp(0) × exp(0)` | `1.0` |
| `approach_distance` | `exp(0) + linear_weight × 1` | `1 + linear_weight` |
| `approach_alignment` | `exp(0) + linear_weight × 1` | `1 + linear_weight` |
| `approach_gripper_pose` | `exp(0) + linear_weight × 1` | `1 + linear_weight` |
| `wrist_roll_pose` | `exp(-pressure × 0)` | `1.0` |
| `grasp_phase` | `exp(-pressure × 0)` | `1.0` |
| `lift_cube` / `grip_cube` | fraction or binary at best | `1.0` |
