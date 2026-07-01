"""F0 (pitch) contour and speech rate over the AGENT channel -- caller
prosody isn't the system under test. Both are perceptual proxies (a flat F0
contour reads as "monotone"/robotic; a words-per-second rate outside typical
human bounds reads as rushed or sluggish) rather than ground truth, so this
metric is signal/advisory: informative, never gating on its own. Pitch
extraction is a swappable seam (`f0_fn`, same pattern as VAD/judges) so unit
tests exercise the threshold logic without needing Praat; one smoke test
runs the real extractor against real fixture audio."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext, Turn

MONOTONE_STD_THRESHOLD_HZ = 10.0
MIN_SPEECH_RATE_WPS = 1.5
MAX_SPEECH_RATE_WPS = 4.0

F0Fn = Callable[[np.ndarray, int], np.ndarray]


def parselmouth_f0(audio: np.ndarray, sr: int) -> np.ndarray:
    import parselmouth

    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch()
    values = pitch.selected_array["frequency"]
    return values[values > 0]  # unvoiced frames report 0.0; keep voiced only


def prosody_stats(f0_values: np.ndarray, monotone_std_threshold_hz: float = MONOTONE_STD_THRESHOLD_HZ) -> dict:
    if len(f0_values) == 0:
        return {"voiced_frames": 0, "pitch_std_hz": None, "pitch_range_hz": None, "monotone": None}

    std = float(np.std(f0_values))
    return {
        "voiced_frames": len(f0_values),
        "pitch_std_hz": std,
        "pitch_range_hz": float(np.max(f0_values) - np.min(f0_values)),
        "monotone": std < monotone_std_threshold_hz,
    }


def speech_rate_wps(transcript: list["Turn"], speaker: str = "agent") -> float | None:
    turns = [t for t in transcript if t.speaker == speaker]
    if not turns:
        return None
    total_words = sum(len(t.text.split()) for t in turns)
    total_duration = sum(t.t_end - t.t_start for t in turns)
    if total_duration <= 0:
        return None
    return total_words / total_duration


@register
class PitchProsodyMetric(BaseMetric):
    name = "pitch_prosody"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.ADVISORY
    requires_ground_truth = False
    monotone_std_threshold_hz = MONOTONE_STD_THRESHOLD_HZ
    min_speech_rate_wps = MIN_SPEECH_RATE_WPS
    max_speech_rate_wps = MAX_SPEECH_RATE_WPS

    def __init__(self, f0_fn: F0Fn | None = None):
        self.f0_fn = f0_fn or parselmouth_f0

    def compute(self, ctx: "MetricContext") -> MetricScore:
        prosody = prosody_stats(self.f0_fn(ctx.audio_agent, ctx.sr), self.monotone_std_threshold_hz)
        rate = speech_rate_wps(ctx.transcript, "agent")

        if prosody["voiced_frames"] == 0 and rate is None:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "no agent speech to evaluate"},
                evaluator_version=self.version,
            )

        issues = []
        if prosody["monotone"]:
            issues.append("monotone_pitch")
        if rate is not None and not (self.min_speech_rate_wps <= rate <= self.max_speech_rate_wps):
            issues.append("speech_rate_out_of_range")

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.FAIL if issues else Status.PASS,
            gating=self.default_gating,
            score=0.0 if issues else 1.0,
            details={**prosody, "speech_rate_wps": rate, "issues": issues},
            evaluator_version=self.version,
        )
