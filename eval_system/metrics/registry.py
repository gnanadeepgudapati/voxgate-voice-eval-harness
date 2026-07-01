"""Plug-in registry. Drop a new metric file in metrics/semantic/ or
metrics/acoustic/, decorate its class with @register — zero edits here."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from eval_system.metrics.base import BaseMetric, MetricKind, MetricScore, Status

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

REGISTRY: list[BaseMetric] = []


def register(cls: type[BaseMetric]) -> type[BaseMetric]:
    REGISTRY.append(cls())
    return cls


class Sampler(Protocol):
    def should_run(self, metric: BaseMetric, ctx: "MetricContext", scores: list[MetricScore]) -> bool: ...


def _safe(metric: BaseMetric, ctx: "MetricContext") -> MetricScore:
    """Isolate one evaluator's crash from the rest of the suite: ERROR, not a
    dropped run and not a false FAIL."""
    try:
        return metric.compute(ctx)
    except Exception as e:
        return MetricScore(
            call_id=ctx.call_id,
            metric=metric.name,
            kind=metric.kind,
            status=Status.ERROR,
            gating=metric.default_gating,
            score=None,
            details={"exc": repr(e)},
            evaluator_version=metric.version,
            judge_prompt_version=None,
        )


def run(
    ctx: "MetricContext",
    sampler: Sampler | None = None,
    metrics_filter: set[str] | None = None,
) -> list[MetricScore]:
    """`metrics_filter`, if given, restricts execution to metrics whose name
    is in the set (e.g. CLI `--metrics faithfulness,barge_in` for fast
    iteration) -- additive; `None` runs every registered metric, unchanged
    from before this param existed."""
    candidates = REGISTRY if metrics_filter is None else [m for m in REGISTRY if m.name in metrics_filter]
    scores: list[MetricScore] = []

    # Phase 1: deterministic + signal metrics run on every call, unconditionally.
    for m in candidates:
        if m.kind is MetricKind.JUDGE:
            continue
        scores.append(_safe(m, ctx))

    # Phase 2: judge metrics run per the sampling policy (defaults to 100% coverage
    # on the fixture set when no sampler is supplied).
    for m in candidates:
        if m.kind is not MetricKind.JUDGE:
            continue
        if sampler is not None and not sampler.should_run(m, ctx, scores):
            continue
        scores.append(_safe(m, ctx))

    return scores
