"""Caller-end -> agent-onset gap distribution (p50/p90/p99, not just a mean --
a mean hides a long tail of slow responses a caller actually notices).
Overlapping (barge-in) transitions are excluded: a negative gap isn't a wait,
it's a different phenomenon `barge_in` already scores. Advisory: describes the
call's responsiveness, doesn't gate on its own until `latency_thresholds`
(deterministic) is promoted."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eval_system.metrics.acoustic.vad import SpeechSegment, VadFn, silero_vad_segments
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


def caller_to_agent_gaps(
    caller_segments: list[SpeechSegment], agent_segments: list[SpeechSegment]
) -> list[float]:
    labeled = sorted(
        [("caller", s) for s in caller_segments] + [("agent", s) for s in agent_segments],
        key=lambda item: item[1].t_start,
    )
    gaps = []
    for (speaker, seg), (next_speaker, next_seg) in zip(labeled, labeled[1:]):
        if speaker == "caller" and next_speaker == "agent":
            gap = next_seg.t_start - seg.t_end
            if gap >= 0:
                gaps.append(gap)
    return gaps


@register
class TurnTakingLatencyMetric(BaseMetric):
    name = "turn_taking_latency"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.ADVISORY
    requires_ground_truth = False

    def __init__(self, vad_fn: VadFn | None = None):
        self.vad_fn = vad_fn or silero_vad_segments

    def compute(self, ctx: "MetricContext") -> MetricScore:
        caller_segments = self.vad_fn(ctx.audio_caller, ctx.sr)
        agent_segments = self.vad_fn(ctx.audio_agent, ctx.sr)
        gaps = caller_to_agent_gaps(caller_segments, agent_segments)

        if not gaps:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "no caller-to-agent turn transitions detected"},
                evaluator_version=self.version,
            )

        p50, p90, p99 = np.percentile(gaps, [50, 90, 99])
        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS,
            gating=self.default_gating,
            score=float(p50),
            details={"gaps": gaps, "n": len(gaps), "p50": float(p50), "p90": float(p90), "p99": float(p99)},
            evaluator_version=self.version,
        )
