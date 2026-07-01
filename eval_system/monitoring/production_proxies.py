"""Ground-truth-free subset for live production traffic: production calls
have no expected.json/scenario_db, so only metrics with
`requires_ground_truth = False` (barge_in, turn_taking_latency,
pitch_prosody, instruction_adherence_judge, emotional_appropriateness, ...)
can run on them. task_success/tool_call_ordering/faithfulness/
entity_intelligibility need authored ground truth and only ever run on the
fixture/eval path."""
from __future__ import annotations

from typing import TYPE_CHECKING

from eval_system.metrics.base import BaseMetric, MetricScore
from eval_system.metrics.registry import _safe

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


def ground_truth_free_metrics(registered_metrics: list[BaseMetric]) -> list[BaseMetric]:
    return [m for m in registered_metrics if not m.requires_ground_truth]


def run_production_monitoring(ctx: "MetricContext") -> list[MetricScore]:
    from eval_system.metrics.registry import REGISTRY

    return [_safe(m, ctx) for m in ground_truth_free_metrics(REGISTRY)]
