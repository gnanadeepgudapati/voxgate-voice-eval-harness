"""Deterministic first-token/inter-turn silence check against a fixed
threshold. Unlike `turn_taking_latency` (VAD-derived distribution, advisory
by nature), this reads the authored transcript timeline directly -- plain
timestamp arithmetic, no signal processing -- and is a Category-1 addition
(←C1(7)): advisory by default until calibration promotes it to a gate, so a
single slow turn never blocks a deploy on its own yet."""
from __future__ import annotations

from typing import TYPE_CHECKING

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext, Turn

FTL_THRESHOLD_SEC = 3.0


def caller_to_agent_transcript_gaps(transcript: list["Turn"]) -> list[dict]:
    gaps = []
    for turn, next_turn in zip(transcript, transcript[1:]):
        if turn.speaker == "caller" and next_turn.speaker == "agent":
            gaps.append({
                "t_caller_end": turn.t_end,
                "t_agent_start": next_turn.t_start,
                "gap": next_turn.t_start - turn.t_end,
            })
    return gaps


@register
class LatencyThresholdsMetric(BaseMetric):
    name = "latency_thresholds"
    version = "1"
    kind = MetricKind.DETERMINISTIC
    default_gating = Gating.ADVISORY
    requires_ground_truth = False
    ftl_threshold_sec = FTL_THRESHOLD_SEC

    def compute(self, ctx: "MetricContext") -> MetricScore:
        gaps = caller_to_agent_transcript_gaps(ctx.transcript)
        if not gaps:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "no caller-to-agent transitions in transcript"},
                evaluator_version=self.version,
            )

        violations = [g for g in gaps if g["gap"] > self.ftl_threshold_sec]
        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.FAIL if violations else Status.PASS,
            gating=self.default_gating,
            score=0.0 if violations else 1.0,
            details={
                "gaps": gaps,
                "violations": violations,
                "threshold_sec": self.ftl_threshold_sec,
                "first_token_latency_sec": gaps[0]["gap"],
            },
            evaluator_version=self.version,
        )
