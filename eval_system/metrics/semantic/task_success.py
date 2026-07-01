"""Did the call reach the caller's goal? Deterministic: compares the last
tool call actually made against `expected.success_criteria` (final_tool +
required result fields). No fixture defining success_criteria.final_tool
(e.g. reschedule_trap, which exists for tool_call_ordering, not this metric)
yields SKIPPED rather than a false PASS/FAIL."""
from __future__ import annotations

from typing import TYPE_CHECKING

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


@register
class TaskSuccessMetric(BaseMetric):
    name = "task_success"
    version = "1"
    kind = MetricKind.DETERMINISTIC
    default_gating = Gating.GATE
    requires_ground_truth = True

    def compute(self, ctx: "MetricContext") -> MetricScore:
        criteria = ctx.expected.get("success_criteria", {})
        final_tool = criteria.get("final_tool")
        result_contains = criteria.get("result_contains", {})

        if not final_tool:
            return self._score(
                ctx, Status.SKIPPED, None,
                {"reason": "no success_criteria.final_tool for this fixture"},
            )

        actual_final = ctx.tool_events[-1].name if ctx.tool_events else None
        if actual_final != final_tool:
            return self._score(
                ctx, Status.FAIL, 0.0,
                {"expected_final_tool": final_tool, "actual_final_tool": actual_final},
            )

        final_result = ctx.tool_events[-1].result or {}
        mismatched = {
            k: v for k, v in result_contains.items() if final_result.get(k) != v
        }
        if mismatched:
            return self._score(
                ctx, Status.FAIL, 0.0,
                {"mismatched_result_fields": mismatched, "final_result": final_result},
            )

        return self._score(
            ctx, Status.PASS, 1.0,
            {"final_tool": final_tool, "final_result": final_result},
        )

    def _score(self, ctx, status, score, details) -> MetricScore:
        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=status,
            gating=self.default_gating,
            score=score,
            details=details,
            evaluator_version=self.version,
        )
