"""Was the agent's tone appropriate for the moment (e.g. apologetic after a
mistake, not cheerful during a complaint)? Always advisory -- CLAUDE.md is
explicit that emotion never earns gate authority, unlike other judges that
can be promoted once calibration trusts them (see faithfulness). Honest
design note: this is a TEXT judge reading a prosody *summary* (F0 std/range/
monotone-flag, from the same extractor `pitch_prosody` uses), not a true
multimodal audio judge -- no multimodal-audio-capable JudgeClient is wired up
here, so the acoustic signal is described in words rather than heard
directly. That's a real proxy-validity gap, which is exactly why this metric
can only ever advise, never gate."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from pydantic import BaseModel

from eval_system.judges.client import JudgeClient
from eval_system.metrics.acoustic.pitch_prosody import parselmouth_f0, prosody_stats
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

ProsodyFn = Callable[[np.ndarray, int], dict]


def _default_prosody_fn(audio: np.ndarray, sr: int) -> dict:
    return prosody_stats(parselmouth_f0(audio, sr))


class EmotionalAppropriatenessJudgment(BaseModel):
    appropriate: bool
    score: float
    notes: str = ""


@register
class EmotionalAppropriatenessMetric(BaseMetric):
    name = "emotional_appropriateness"
    version = "1"
    kind = MetricKind.JUDGE
    default_gating = Gating.ADVISORY
    requires_ground_truth = False
    prompt_version = "v1"

    def __init__(self, client: JudgeClient | None = None, prosody_fn: ProsodyFn | None = None):
        self.client = client
        self.prosody_fn = prosody_fn or _default_prosody_fn

    def _get_client(self) -> JudgeClient:
        if self.client is None:
            from eval_system.judges.anthropic_client import AnthropicJudgeClient

            self.client = AnthropicJudgeClient()
        return self.client

    def compute(self, ctx: "MetricContext") -> MetricScore:
        judgment = self._get_client().structured_complete(self._build_prompt(ctx), EmotionalAppropriatenessJudgment)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS if judgment.appropriate else Status.FAIL,
            gating=self.default_gating,  # always advisory -- never promoted, by design
            score=judgment.score,
            details={"notes": judgment.notes},
            evaluator_version=self.version,
            judge_prompt_version=self.prompt_version,
        )

    def _build_prompt(self, ctx: "MetricContext") -> str:
        lines = "\n".join(f"[{turn.t_start:.1f}s] {turn.speaker}: {turn.text}" for turn in ctx.transcript)
        prosody = self.prosody_fn(ctx.audio_agent, ctx.sr)

        return (
            "You are grading whether a clinic-scheduling voice agent's emotional tone "
            "was appropriate for the conversational moment (e.g. apologetic after a "
            "mistake, not cheerful during a complaint).\n\n"
            f"Transcript:\n{lines}\n\n"
            "Agent vocal-prosody summary (no direct audio available -- described, not "
            f"heard): voiced_frames={prosody['voiced_frames']}, "
            f"pitch_std_hz={prosody['pitch_std_hz']}, pitch_range_hz={prosody['pitch_range_hz']}, "
            f"monotone={prosody['monotone']}\n"
        )
