"""Headline acoustic metric: detect caller-over-agent overlap, measure
time-to-yield, and flag two distinct failure modes -- fail-to-yield (agent
kept talking too long after a genuine barge-in) and false-yield (agent
stopped for a noise burst too short to be real speech, e.g. a cough).

Runs 2-channel VAD independently per channel (no cross-channel leakage
assumed) and reasons purely over the resulting speech segments in seconds on
the canonical clock -- this metric never re-derives timing from raw audio
beyond what `MetricContext` already carries. VAD is a swappable seam
(`vad_fn`) for the same reason judges are: unit tests exercise the overlap/
threshold logic against synthetic segments; only one smoke test runs the real
model against real fixture audio."""
from __future__ import annotations

from typing import TYPE_CHECKING

from eval_system.metrics.acoustic.vad import SpeechSegment, VadFn, silero_vad_segments
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

FAIL_TO_YIELD_THRESHOLD_SEC = 1.0
MIN_GENUINE_SPEECH_DURATION_SEC = 0.3
# VAD has real onset/offset latency (padding, silence-duration thresholds), so
# a genuine overlap can land just outside a segment's boundary -- allow a
# small grace window rather than requiring an exact/boundary match.
OVERLAP_TOLERANCE_SEC = 0.2


def find_barge_ins(
    caller_segments: list[SpeechSegment],
    agent_segments: list[SpeechSegment],
    *,
    fail_to_yield_threshold_sec: float = FAIL_TO_YIELD_THRESHOLD_SEC,
    min_genuine_speech_duration_sec: float = MIN_GENUINE_SPEECH_DURATION_SEC,
    overlap_tolerance_sec: float = OVERLAP_TOLERANCE_SEC,
) -> list[dict]:
    """For each caller speech segment that starts while the agent is still
    speaking, report the onset, time-to-yield, and whether it's a
    fail-to-yield (agent kept going past the threshold) or a false-yield
    (the "caller speech" was too short to be genuine -- e.g. a cough --
    so any agent stop here wasn't a real barge-in response)."""
    barge_ins = []
    for caller_seg in caller_segments:
        overlapping_agent = next(
            (
                a
                for a in agent_segments
                if a.t_start <= caller_seg.t_start <= a.t_end + overlap_tolerance_sec
            ),
            None,
        )
        if overlapping_agent is None:
            continue

        is_genuine = (caller_seg.t_end - caller_seg.t_start) >= min_genuine_speech_duration_sec
        time_to_yield = max(0.0, overlapping_agent.t_end - caller_seg.t_start)
        barge_ins.append({
            "t_onset": caller_seg.t_start,
            "time_to_yield": time_to_yield,
            "false_yield": not is_genuine,
            "fail_to_yield": is_genuine and time_to_yield > fail_to_yield_threshold_sec,
        })
    return barge_ins


@register
class BargeInMetric(BaseMetric):
    name = "barge_in"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.GATE
    requires_ground_truth = False

    def __init__(self, vad_fn: VadFn | None = None):
        self.vad_fn = vad_fn or silero_vad_segments

    def compute(self, ctx: "MetricContext") -> MetricScore:
        caller_segments = self.vad_fn(ctx.audio_caller, ctx.sr)
        agent_segments = self.vad_fn(ctx.audio_agent, ctx.sr)
        barge_ins = find_barge_ins(caller_segments, agent_segments)

        issues = [b for b in barge_ins if b["fail_to_yield"] or b["false_yield"]]
        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.FAIL if issues else Status.PASS,
            gating=self.default_gating,
            score=0.0 if issues else 1.0,
            details={"barge_ins": barge_ins, "issue_count": len(issues)},
            evaluator_version=self.version,
        )
