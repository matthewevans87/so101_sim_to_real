"""First-class termination pipeline.

Termination is a pipeline of :class:`TerminationCondition` objects, each
evaluated independently per env per step.  A condition fires (and the
episode ends) when its :meth:`done` returns True AND every gate in
``cfg.gates`` evaluates True for that env on that step.

Termination is fully decoupled from rewards.  Reward shaping that should
accompany an end-of-episode event is configured separately in ``rewards``
(typically via a :class:`~so101_rl.reward_pipeline.RewardStep` with
``fire_once: true`` and matching ``gates``).

The pipeline exposes:

* :meth:`TerminationPipeline.get_dones` — bool tensor, OR over enabled
  conditions; consumed by :meth:`So101LiftCubeEnv._get_dones`.
* :meth:`TerminationPipeline.get_done_reasons` — per-condition flags for
  ``Termination/<id>`` TensorBoard logging.
* :meth:`TerminationPipeline.reset_idx` — clears any per-episode state.
* :attr:`TerminationPipeline.success_condition_log_name` — the log name
  of the (singular) ``is_success: true`` condition; used by the eval
  pipeline to classify success episodes without string matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch

from so101_rl.configurations.so101_env_params import (
    GateCfg,
    TerminationCfg,
    _from_dict,
)
from so101_rl.metric_pipeline import StepContext

CfgT = TypeVar("CfgT", bound=TerminationCfg)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class TerminationCondition(ABC, Generic[CfgT]):
    """A single per-env boolean termination signal.

    Subclasses implement :meth:`done` returning a pre-gate boolean tensor of
    shape ``(num_envs,)``.  The pipeline AND-combines that with the gate
    mask before OR-ing across all enabled conditions.
    """

    name: str
    """Registered type name (matches the ``type`` key in YAML)."""

    requires_metrics: frozenset[str] = frozenset()
    """Metric keys read by :meth:`done`.  Used by ``build_metric_pipeline``
    to determine which metric steps to include."""

    requires_env_metrics: frozenset[str] = frozenset()
    """Per-episode env-metric keys read by :meth:`done`."""

    def __init__(self, cfg: CfgT) -> None:
        self._cfg: CfgT = cfg
        self._gates: list[GateCfg] = cfg.gates

    @property
    def log_name(self) -> str:
        """TensorBoard logging key suffix.

        When :attr:`TerminationCfg.id` is set, returns the id verbatim
        (terminations are uniquely identified by id, not by type).  When
        no id is set, falls back to the registered :attr:`name`.
        """
        if self._cfg.id is not None:
            return self._cfg.id
        return self.name

    @abstractmethod
    def done(self, ctx: StepContext) -> torch.Tensor:
        """Return a bool tensor of shape ``(num_envs,)`` — True for envs that
        should terminate this step (pre-gate; gates are AND-applied externally).
        """
        ...


# ---------------------------------------------------------------------------
# TerminationPipeline
# ---------------------------------------------------------------------------


class TerminationPipeline:
    """Combines per-env termination flags from a sequence of
    :class:`TerminationCondition` objects.

    Conditions are logged individually under ``Termination/<log_name>``.
    Only enabled conditions passed at construction are evaluated — callers
    should filter by ``cfg.enabled`` once at startup.
    """

    def __init__(self, conditions: list[TerminationCondition]) -> None:
        self.conditions = conditions

    @property
    def required_metric_keys(self) -> frozenset[str]:
        """Union of all ``requires_metrics`` declared by the conditions in
        this pipeline, plus metric keys referenced by gate conditions."""
        cond_keys = frozenset().union(
            *(c.requires_metrics for c in self.conditions)
        )
        gate_keys = frozenset(g.metric for c in self.conditions for g in c._gates)
        return cond_keys | gate_keys

    @property
    def success_condition_log_name(self) -> str | None:
        """The :attr:`log_name` of the (singular) ``is_success: true``
        condition.  ``None`` when the pipeline has no success terminal
        (e.g. evaluation-only edge cases).  Cross-field validation in
        :class:`So101EnvParams` enforces exactly one such condition exists
        in production configs.
        """
        for c in self.conditions:
            if c._cfg.is_success:
                return c.log_name
        return None

    @staticmethod
    def _evaluate_gate_mask(gates: list[GateCfg], ctx: StepContext) -> torch.Tensor:
        """Return a bool mask of shape ``(num_envs,)`` — True where all gates
        pass.  Each gate is resolved from ``ctx.metrics`` first, then
        ``ctx.env_metrics``.
        """
        env = ctx.env
        mask = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for gate in gates:
            if gate.metric in ctx.metrics:
                val = ctx.metrics[gate.metric].float()
            elif gate.metric in ctx.env_metrics:
                val = ctx.env_metrics[gate.metric].float()
            else:
                raise KeyError(
                    f"Termination gate metric '{gate.metric}' not found in "
                    f"ctx.metrics or ctx.env_metrics at runtime."
                )
            if val.dim() > 1:
                val = val.squeeze(-1)
            if gate.gt is not None:
                mask = mask & (val > gate.gt)
            elif gate.gte is not None:
                mask = mask & (val >= gate.gte)
            elif gate.lt is not None:
                mask = mask & (val < gate.lt)
            elif gate.lte is not None:
                mask = mask & (val <= gate.lte)
            elif gate.eq is not None:
                mask = mask & (val == gate.eq)
        return mask

    def _evaluate_condition(
        self, cond: TerminationCondition, ctx: StepContext
    ) -> torch.Tensor:
        """Compute the post-gate done flag for *cond* at the current step."""
        flag = cond.done(ctx).bool()
        if cond._gates:
            flag = flag & self._evaluate_gate_mask(cond._gates, ctx)
        return flag

    def get_dones(self, ctx: StepContext) -> torch.Tensor:
        """Return a bool tensor of shape ``(num_envs,)`` — True for envs where
        any enabled :class:`TerminationCondition` fires this step."""
        env = ctx.env
        terminal = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        for cond in self.conditions:
            terminal = terminal | self._evaluate_condition(cond, ctx)
        return terminal

    def get_done_reasons(self, ctx: StepContext) -> dict[str, torch.Tensor]:
        """Return per-condition post-gate flags (float32, shape ``(num_envs,)``)
        keyed by :attr:`TerminationCondition.log_name`.  Values are 0.0/1.0
        and represent only **real** ending events (post-Phase-A there are no
        ``terminate: false`` terminals — every condition in the pipeline
        actually ends the episode when it fires).
        """
        return {
            cond.log_name: self._evaluate_condition(cond, ctx).float()
            for cond in self.conditions
        }

    def reset_idx(self, env_ids) -> None:
        """Hook called from the environment's ``_reset_idx`` after every
        episode reset.  Currently no per-episode state is held by the
        gate-driven base condition — present for symmetry with
        :meth:`RewardPipeline.reset_idx` and to give future stateful
        conditions an integration point.
        """
        # No state to reset for the current condition catalogue.
        del env_ids


# ---------------------------------------------------------------------------
# Concrete conditions
# ---------------------------------------------------------------------------


class MetricThresholdTerminationCondition(TerminationCondition[TerminationCfg]):
    """Always-True base condition; gates do all of the real work.

    The condition fires for every env whose ``cfg.gates`` collectively
    evaluate True at the current step.  This single class covers every
    metric-threshold termination in the current task catalogue (success
    via ``cube_lift_fraction >= 1.0``, ``is_cube_out_of_range``,
    ``is_table_touched``, etc.) without subclassing.
    """

    name = "metric_threshold"
    # No metrics required directly; required keys come from gates and are
    # collected by :attr:`TerminationPipeline.required_metric_keys`.
    requires_metrics: frozenset[str] = frozenset()

    def done(self, ctx: StepContext) -> torch.Tensor:
        env = ctx.env
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

TERMINATION_REGISTRY: dict[
    str, tuple[type[TerminationCondition[Any]], type[TerminationCfg]]
] = {
    "metric_threshold": (MetricThresholdTerminationCondition, TerminationCfg),
}

DEFAULT_TERMINATION_TYPE = "metric_threshold"


def build_termination_pipeline(cfg) -> TerminationPipeline:
    """Construct the termination pipeline from ``cfg.terminations``.

    ``cfg.terminations`` is an ordered list of dicts.  Each entry MAY
    include a ``type`` key naming a registered :class:`TerminationCondition`;
    when absent, ``"metric_threshold"`` is used (the only registered type
    today).  Disabled entries (``enabled: false``) are skipped entirely.

    Args:
        cfg: The ``So101LiftCubeCfg`` instance (``env.cfg``).
    """
    conditions: list[TerminationCondition] = []
    terminations = cfg.terminations
    if not isinstance(terminations, list):
        raise TypeError(
            f"cfg.terminations must be a list; got {type(terminations).__name__}."
        )
    for entry in terminations:
        if not isinstance(entry, dict):
            raise TypeError(
                f"Each termination entry must be a mapping; got "
                f"{type(entry).__name__}: {entry!r}"
            )
        type_name = entry.get("type", DEFAULT_TERMINATION_TYPE)
        params = {k: v for k, v in entry.items() if k != "type"}
        if not params.get("enabled", True):
            continue
        if type_name not in TERMINATION_REGISTRY:
            raise ValueError(
                f"Unknown termination type '{type_name}'. "
                f"Known types: {sorted(TERMINATION_REGISTRY)}"
            )
        cond_cls, cfg_cls = TERMINATION_REGISTRY[type_name]
        instance_cfg = _from_dict(cfg_cls, params)
        conditions.append(cond_cls(cfg=instance_cfg))
    return TerminationPipeline(conditions)


def validate_termination_gate_metrics(
    termination_pipeline: TerminationPipeline,
    metric_pipeline,
    env_metric_pipeline,
) -> None:
    """Verify that every termination gate metric key is resolvable at runtime.

    Mirrors :func:`so101_rl.reward_pipeline.validate_gate_metrics` for the
    termination pipeline.  Raises ``ValueError`` at startup if any gate
    references an unknown metric so failures are caught before the first
    training step.
    """
    all_step_metric_keys: frozenset[str] = frozenset(
        key for step in metric_pipeline.steps for key in type(step).produces
    )
    all_env_metric_keys: frozenset[str] = env_metric_pipeline.provided_keys
    known_keys = all_step_metric_keys | all_env_metric_keys

    for cond in termination_pipeline.conditions:
        for gate in cond._gates:
            if gate.metric not in known_keys:
                raise ValueError(
                    f"Gate on termination condition '{cond.log_name}' references "
                    f"metric key '{gate.metric}', but it is not produced by any "
                    f"MetricStep or EnvMetricStep. "
                    f"Known keys: {sorted(known_keys)}"
                )
