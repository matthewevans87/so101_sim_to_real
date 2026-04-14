"""Reward dominance analyser — no Isaac Lab or so101_rl import required.

Loads a So101EnvParams-compatible YAML directly and, for each enabled reward,
analytically computes:

  * ``max_step``    — best possible reward at any single step (best-case physics).
  * ``max_episode`` — best possible cumulative reward over the full episode.
      * absolute mode: ``max_step × num_steps``
      * progressive mode: ``metric_range × scale`` (bounded, one-shot)
      * terminal: ``scale`` (fires at most once)

Negative-scale (penalty) rewards report ``max_step = 0`` and
``max_episode = 0`` — their best case is simply not firing.

Pathologies detected
--------------------
NO_POSITIVE_TERMINAL
    No enabled terminal reward has a positive scale.  The agent has no large
    one-shot signal to aim for; shaped rewards may fill the episode unopposed.

SHAPING_DOMINATES_TERMINAL
    ``sum(max_episode for shaping rewards) > k × terminal_scale`` for some
    positive-scale terminal.  The terminal bonus is too small relative to what
    the agent can accumulate through continuous shaping.

UNREACHABLE_TERMINAL
    A terminal reward's condition cannot be met analytically (e.g. the required
    metric can never exceed its threshold given its formula).

PROGRESSIVE_NO_FALLBACK
    A reward type uses progressive mode exclusively (no absolute counterpart
    for the same metric).  Once the agent reaches the optimum, the gradient
    from this reward disappears entirely.

MODE_IGNORED
    ``mode: progressive`` is set for a reward whose implementation ignores the
    mode field (always behaves as absolute).  The max_episode shown is
    therefore ``max_step × num_steps``, not the one-shot bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Known reward type sets
# ---------------------------------------------------------------------------

TERMINAL_TYPES: frozenset[str] = frozenset(
    {
        "success_lift_fraction_terminal",
        "safety_touch_table_terminal",
        "cube_out_of_range_terminal",
        "approach_phase_terminal",
        "grasp_phase_terminal",
    }
)

# Reward types whose RewardStep.compute() actually switches on self._cfg.mode.
# Any type *not* in this set will flag MODE_IGNORED if mode=progressive is set.
PROGRESSIVE_AWARE_TYPES: frozenset[str] = frozenset(
    {
        "approach_distance",
        "approach_alignment",
        "approach_gripper_pose",
        "approach_phase",
        "lift_cube",
        "distance",
        "grasp_phase",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RewardRow:
    type_name: str
    mode: str
    scale: float
    max_step: float
    max_episode: float
    is_terminal: bool
    terminates: bool
    terminal_reachable: bool | None
    gates: list[str]
    notes: list[str]

    @property
    def gate_str(self) -> str:
        return " & ".join(self.gates) if self.gates else ""


@dataclass
class AnalysisResult:
    config_path: str
    num_steps: int
    episode_length_s: float
    rows: list[RewardRow]
    pathologies: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_gates(raw_gates: list[dict]) -> list[str]:
    strs = []
    for g in raw_gates:
        metric = g.get("metric", "?")
        for op, sym in [
            ("gt", ">"),
            ("gte", "≥"),
            ("lt", "<"),
            ("lte", "≤"),
            ("eq", "="),
        ]:
            if g.get(op) is not None:
                strs.append(f"{metric} {sym} {g[op]}")
                break
    return strs


def _metric_max(type_name: str, metrics_cfg: dict) -> float:
    """Best-case value of the underlying metric consumed by *type_name*.

    Derived analytically from the corresponding MetricStep formula at ideal
    physical state (d=0, a=1, gripper at target, force at target, etc.).
    """
    if type_name == "approach_phase":
        # exp(-dp×0) × exp(-ap×0) × exp(-gp×0) = 1.0
        return 1.0

    if type_name == "approach_distance":
        lw = metrics_cfg.get("approach_distance", {}).get("linear_weight", 0.0)
        # exp(-pressure×0) + lw × (1 - 0/d_max) = 1 + lw
        return 1.0 + lw

    if type_name == "approach_alignment":
        lw = metrics_cfg.get("approach_alignment", {}).get("linear_weight", 0.0)
        # exp(-pressure×(1-1)) + lw × (1+1)/2 = 1 + lw
        return 1.0 + lw

    if type_name == "approach_gripper_pose":
        lw = metrics_cfg.get("approach_gripper_pose", {}).get("linear_weight", 0.0)
        # exp(-pressure×0) + lw × 1 = 1 + lw
        return 1.0 + lw

    if type_name in (
        "grip_cube",
        "gripper_cube_alignment",
        "lift_cube",
        "close_gripper",
        "grasp_phase",
        "wrist_roll_pose",
    ):
        return 1.0  # binary / fraction / exp kernel at its optimum

    # distance reward: (1 - exp(-p×d)) × scale; at d=0 → 0 (with negative scale,
    # best case = least negative = 0)
    if type_name == "distance":
        return 0.0

    # Penalties and zero-scale terminals: best case is 0.
    return 0.0


def _terminal_reachable(type_name: str, entry: dict, cfg: dict) -> bool:
    """Can the terminal condition be triggered under ideal (unrestricted) physics?"""
    if type_name == "approach_phase_terminal":
        threshold = float(entry.get("threshold", 0.9))
        return 1.0 > threshold  # approach_phase max = 1.0

    if type_name == "grasp_phase_terminal":
        grasp_threshold = float(entry.get("threshold", 0.95))
        # Also needs approach_phase_terminal to be reachable.
        for e in cfg.get("rewards", []):
            if e.get("type") == "approach_phase_terminal" and e.get("enabled", True):
                apt_threshold = float(e.get("threshold", 0.9))
                if 1.0 <= apt_threshold:
                    return False
        return 1.0 > grasp_threshold  # grasp_phase max = 1.0

    if type_name == "success_lift_fraction_terminal":
        return True  # robot can always lift the cube in ideal conditions

    if type_name in ("safety_touch_table_terminal", "cube_out_of_range_terminal"):
        # Adverse but physically possible.
        return True

    return True


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyse(config_path: str | Path, k: float = 2.0) -> AnalysisResult:
    """Load *config_path* and return a full :class:`AnalysisResult`.

    Args:
        config_path: Path to a So101EnvParams-compatible YAML file.
        k: Dominance threshold multiplier. A pathology is flagged when
           ``sum(shaping max_episode) > k × terminal_scale``.

    Returns:
        :class:`AnalysisResult` with per-reward rows sorted by
        ``max_episode`` descending (terminals last).
    """
    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg: dict = yaml.safe_load(fh) or {}

    # Episode timing
    dt: float = cfg.get("sim", {}).get("dt", 1.0 / 120.0)
    decimation: int = int(cfg.get("decimation", 2))
    episode_length_s: float = float(cfg.get("episode_length_s", 7.0))
    step_dt = dt * decimation
    num_steps = int(round(episode_length_s / step_dt))

    metrics_cfg: dict = cfg.get("metrics", {})
    rewards_list: list[dict] = cfg.get("rewards", [])
    # Support old named-map format.
    if isinstance(rewards_list, dict):
        rewards_list = [{"type": k_name, **v} for k_name, v in rewards_list.items()]

    rows: list[RewardRow] = []
    for entry in rewards_list:
        if not entry.get("enabled", True):
            continue

        type_name: str = entry.get("type", "unknown")
        scale: float = float(entry.get("scale", 1.0))
        mode: str = entry.get("mode", "absolute")
        gates = _parse_gates(entry.get("gates", []))
        is_term = type_name in TERMINAL_TYPES
        notes: list[str] = []

        terminates: bool = bool(entry.get("terminate", True))
        fire_once: bool = bool(entry.get("fire_once", False))

        if is_term:
            term_reachable = _terminal_reachable(type_name, entry, cfg)
            if terminates:
                # True terminal: fires once, episode ends.
                max_step = max(scale, 0.0)
                max_episode = max(scale, 0.0)
            elif fire_once:
                # Milestone bonus with fire_once: fires at most once per episode.
                max_step = max(scale, 0.0)
                max_episode = max(scale, 0.0)
                notes.append(
                    "terminate=false, fire_once=true — fires at most once per episode "
                    "(first step condition is met)"
                )
            else:
                # Milestone bonus: no episode reset, condition may stay True every step.
                max_step = max(scale, 0.0)
                max_episode = max(scale, 0.0) * num_steps
                notes.append(
                    "terminate=false — fires every step condition is met; "
                    f"max_episode shown as {max(scale,0.0):,.0f} × {num_steps} steps"
                )

        elif scale < 0:
            # Penalty: best case is 0 (never fires / agent avoids it).
            max_step = 0.0
            max_episode = 0.0
            term_reachable = None

        else:
            m_max = _metric_max(type_name, metrics_cfg)
            effective_mode = mode

            # Detect mode=progressive set for types that don't implement it.
            if mode == "progressive" and type_name not in PROGRESSIVE_AWARE_TYPES:
                notes.append(
                    "mode=progressive set but not implemented — behaves as absolute"
                )
                effective_mode = "absolute"

            if effective_mode == "progressive":
                # One-shot: total gain bounded by full metric range (0 → m_max).
                max_episode = (m_max - 0.0) * scale
                max_step = max_episode  # could all happen in one step
            else:
                max_step = m_max * scale
                max_episode = max_step * num_steps

            term_reachable = None

        rows.append(
            RewardRow(
                type_name=type_name,
                mode=mode,
                scale=scale,
                max_step=max_step,
                max_episode=max_episode,
                is_terminal=is_term,
                terminates=terminates,
                terminal_reachable=term_reachable,
                gates=gates,
                notes=notes,
            )
        )

    # Sort: shaping + nonterminating-terminal rows by max_episode desc,
    # then true-terminal rows.
    shaping_rows = sorted(
        [r for r in rows if not r.is_terminal or not r.terminates],
        key=lambda r: -r.max_episode,
    )
    terminal_rows = sorted(
        [r for r in rows if r.is_terminal and r.terminates],
        key=lambda r: -r.max_episode,
    )
    sorted_rows = shaping_rows + terminal_rows

    pathologies = _detect_pathologies(sorted_rows, num_steps, k)

    return AnalysisResult(
        config_path=str(path),
        num_steps=num_steps,
        episode_length_s=episode_length_s,
        rows=sorted_rows,
        pathologies=pathologies,
    )


def _detect_pathologies(rows: list[RewardRow], num_steps: int, k: float) -> list[str]:
    pathologies: list[str] = []

    # True terminals: episode-ending with positive scale.
    true_terminals = [r for r in rows if r.is_terminal and r.terminates]
    positive_terminals = [r for r in true_terminals if r.max_episode > 0]
    reachable_pos_terminals = [r for r in positive_terminals if r.terminal_reachable]

    # Non-terminating milestones are shaping signals for dominance purposes.
    shaping = [
        r for r in rows if (not r.is_terminal or not r.terminates) and r.scale > 0
    ]
    total_shaping = sum(r.max_episode for r in shaping)

    # -----------------------------------------------------------------------
    # NO_POSITIVE_TERMINAL
    # -----------------------------------------------------------------------
    if not positive_terminals:
        pathologies.append(
            "NO_POSITIVE_TERMINAL: No enabled terminal reward has a positive scale. "
            "The agent has no large one-shot goal signal; shaping rewards fill the "
            "episode unopposed."
        )

    # -----------------------------------------------------------------------
    # NONTERMINATING_MILESTONE_DOMINATES
    # -----------------------------------------------------------------------
    for r in shaping:
        if r.is_terminal and not r.terminates and r.scale > 0:
            for t in reachable_pos_terminals:
                if r.max_episode > t.max_episode * k:
                    pathologies.append(
                        f"NONTERMINATING_MILESTONE_DOMINATES [{r.type_name}]: "
                        f"terminate=false reward max_episode = {r.max_episode:,.0f} "
                        f"> {k}× true terminal [{t.type_name}] = {t.max_episode:,.0f}. "
                        f"Consider gating or reducing scale."
                    )

    # -----------------------------------------------------------------------
    # UNREACHABLE_TERMINAL
    # -----------------------------------------------------------------------
    for t in true_terminals:
        if t.terminal_reachable is False:
            pathologies.append(
                f"UNREACHABLE_TERMINAL [{t.type_name}]: The terminal condition "
                f"cannot be met analytically — threshold exceeds the metric's maximum."
            )

    # -----------------------------------------------------------------------
    # SHAPING_DOMINATES_TERMINAL
    # -----------------------------------------------------------------------
    for t in reachable_pos_terminals:
        if total_shaping > t.max_episode * k:
            ratio = total_shaping / t.max_episode
            pathologies.append(
                f"SHAPING_DOMINATES_TERMINAL [{t.type_name}]: "
                f"sum(shaping max_episode) = {total_shaping:,.0f}  "
                f"> {k}× terminal = {t.max_episode:,.0f}  "
                f"(ratio {ratio:.1f}×)"
            )

    # -----------------------------------------------------------------------
    # PROGRESSIVE_NO_FALLBACK
    # -----------------------------------------------------------------------
    # A type is "progressive-only" if it appears in progressive mode but
    # never in absolute mode among the enabled rewards.
    progressive_types = {
        r.type_name
        for r in rows
        if not r.is_terminal
        and r.mode == "progressive"
        and r.scale > 0
        and r.type_name in PROGRESSIVE_AWARE_TYPES
    }
    absolute_types = {
        r.type_name
        for r in rows
        if not r.is_terminal and r.mode == "absolute" and r.scale > 0
    }
    for pt in sorted(progressive_types - absolute_types):
        pathologies.append(
            f"PROGRESSIVE_NO_FALLBACK [{pt}]: Progressive-only — gradient "
            f"vanishes once the agent reaches the optimum."
        )

    # -----------------------------------------------------------------------
    # MODE_IGNORED
    # -----------------------------------------------------------------------
    for r in rows:
        if "mode=progressive set but not implemented" in " ".join(r.notes):
            pathologies.append(
                f"MODE_IGNORED [{r.type_name}]: mode=progressive is set "
                f"but this reward type always behaves as absolute. "
                f"max_episode shown as {r.max_episode:,.0f} (absolute × num_steps)."
            )

    return pathologies
