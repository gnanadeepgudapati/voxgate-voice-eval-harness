"""Stratified judge-coverage policy (←C1(5)). Deterministic/signal metrics
already run on every call unconditionally (registry.run() phase 1); this only
governs how much of the judge phase (phase 2) runs. Defaults to FULL coverage
-- CLAUDE.md is explicit that sampling defaults to 100% on the fixture/eval
set. Coverage is something you deliberately dial down for cost/latency on
production monitoring traffic, never a silent gap in an eval report."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eval_system.metrics.base import BaseMetric, Gating, MetricScore, Status

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


@dataclass
class StratifiedJudgeSampler:
    sample_rate: float = 1.0
    always_sample_calls: frozenset[str] = field(default_factory=frozenset)
    seed: int = 0

    def should_run(self, metric: BaseMetric, ctx: "MetricContext", scores: list[MetricScore]) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if ctx.call_id in self.always_sample_calls:
            return True
        if any(s.gating is Gating.GATE and s.status is Status.FAIL for s in scores):
            # A hard-gate metric already flagged this call -- always get the
            # judge's read on it too, regardless of the sampling rate.
            return True

        rng = random.Random(f"{self.seed}:{metric.name}:{ctx.call_id}")
        return rng.random() < self.sample_rate
