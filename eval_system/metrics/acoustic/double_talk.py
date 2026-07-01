"""Overlapping-speech ("double-talk") duration: both channels active at
once, independent of who started it or who yielded (that's `barge_in`'s
job). This is the DoD's live-follow-up drop-in -- proof the registry accepts
a brand-new acoustic metric with zero runner edits, reusing the same VAD seam
as barge_in/turn_taking_latency. Signal/advisory: overlap alone isn't
necessarily bad (short backchannels like "mm-hm" overlap constantly in
natural conversation), so this reports duration/ratio rather than a pass/fail
judgment on its own."""
from __future__ import annotations

from typing import TYPE_CHECKING

from eval_system.metrics.acoustic.vad import SpeechSegment, VadFn, silero_vad_segments
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


def total_overlap_seconds(caller_segments: list[SpeechSegment], agent_segments: list[SpeechSegment]) -> float:
    overlap = 0.0
    for c in caller_segments:
        for a in agent_segments:
            start = max(c.t_start, a.t_start)
            end = min(c.t_end, a.t_end)
            if end > start:
                overlap += end - start
    return overlap


@register
class DoubleTalkMetric(BaseMetric):
    name = "double_talk"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.ADVISORY
    requires_ground_truth = False

    def __init__(self, vad_fn: VadFn | None = None):
        self.vad_fn = vad_fn or silero_vad_segments

    def compute(self, ctx: "MetricContext") -> MetricScore:
        call_duration = len(ctx.audio_caller) / ctx.sr if ctx.sr else 0.0
        if call_duration <= 0:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "zero-duration audio"},
                evaluator_version=self.version,
            )

        overlap_seconds = total_overlap_seconds(
            self.vad_fn(ctx.audio_caller, ctx.sr), self.vad_fn(ctx.audio_agent, ctx.sr)
        )
        overlap_ratio = overlap_seconds / call_duration

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS,
            gating=self.default_gating,
            score=overlap_ratio,
            details={
                "overlap_seconds": overlap_seconds,
                "call_duration_sec": call_duration,
            },
            evaluator_version=self.version,
        )
