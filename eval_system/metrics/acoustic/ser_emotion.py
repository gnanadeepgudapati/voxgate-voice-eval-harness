"""Objective speech-emotion-recognition (SER) from the raw agent waveform --
one of two complementary, PERMANENTLY-advisory attacks on "emotional
appropriateness" (assessment.md line 57; see docs/design_writeup.md §3 for
the full argument). This one answers "what emotion does the audio carry?":
objective, reproducible, fully offline, and calibratable against labeled
corpora (RAVDESS/CREMA-D) since it's a real classifier, not a judge.

Hardcoded non-promotable: this is a SIGNAL-kind metric, and the registry's
promotion path (`gating.gate.is_gate_eligible`) only ever promotes JUDGE-kind
metrics via `trusted_judge_metrics` -- so there is structurally no path for
this metric to ever hard-gate, regardless of any future calibration result.
That's deliberate: acted-emotion SER is a genuinely noisy proxy for real
appropriateness -- even human labelers on the IEMOCAP corpus these 4 classes
come from only agree with each other at Fleiss' kappa ~0.27-0.48. A classifier
that's noisier than human-to-human agreement has no business gating a ship
decision, however calibrated it gets.

Segments the AGENT channel by the same authored transcript turn boundaries
the rest of the acoustic suite treats as ground truth (`pitch_prosody` uses
the same turns for speech rate) -- not VAD-derived segments, so the turn
boundaries here are exactly the ones `emotion_appropriateness_mm.py` and the
cross-metric disagreement check in `report/combine.py` also use.

Classification is a swappable seam (`classify_fn`, same pattern as VAD/STT/
judges elsewhere): unit tests inject a fake; one smoke test runs the real
model. Model-load failure or missing audio must surface as Status.ERROR, not
FAIL (CLAUDE.md: "ERROR = evaluator broke, != FAIL") -- achieved here by NOT
catching exceptions ourselves and letting them propagate to the registry's
`_safe()` wrapper, exactly like every other metric in this codebase."""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable

import numpy as np

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

SER_MODEL_ID = "superb/wav2vec2-base-superb-er"
SER_SAMPLE_RATE = 16000

ClassifyFn = Callable[[np.ndarray, int], dict]

_PIPELINE_CACHE = None


def ser_label_to_valence(label: str) -> str:
    """Coarse valence bucket for the IEMOCAP 4-class label set this model
    outputs ("hap"/"neu"/"sad"/"ang"). Also reused by report/combine.py to
    bucket the multimodal judge's free-text detected_tone with the same
    keyword heuristic, so both proxies are compared on the same axis."""
    normalized = label.lower()
    if "hap" in normalized or "excit" in normalized:
        return "positive"
    if "sad" in normalized or "ang" in normalized or "fear" in normalized or "disgust" in normalized:
        return "negative"
    return "neutral"


def _wav2vec2_classify(audio: np.ndarray, sr: int) -> dict:
    import librosa

    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        from transformers import pipeline

        _PIPELINE_CACHE = pipeline("audio-classification", model=SER_MODEL_ID)

    resampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=SER_SAMPLE_RATE)
    predictions = _PIPELINE_CACHE(resampled, sampling_rate=SER_SAMPLE_RATE)
    top = max(predictions, key=lambda p: p["score"])
    return {"label": top["label"], "confidence": float(top["score"])}


def _slice_turn_audio(audio_agent: np.ndarray, sr: int, t_start: float, t_end: float) -> np.ndarray:
    start_sample = max(0, round(t_start * sr))
    end_sample = min(len(audio_agent), round(t_end * sr))
    return audio_agent[start_sample:end_sample]


@register
class SerEmotionMetric(BaseMetric):
    name = "ser_emotion"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.ADVISORY
    requires_ground_truth = False

    def __init__(self, classify_fn: ClassifyFn | None = None):
        self.classify_fn = classify_fn or _wav2vec2_classify

    def compute(self, ctx: "MetricContext") -> MetricScore:
        if ctx.audio_agent is None or len(ctx.audio_agent) == 0:
            raise ValueError("ser_emotion: no agent audio available")

        agent_turns = [t for t in ctx.transcript if t.speaker == "agent"]
        if not agent_turns:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "no agent turns in transcript"},
                evaluator_version=self.version,
            )

        per_turn = []
        for turn in agent_turns:
            segment = _slice_turn_audio(ctx.audio_agent, ctx.sr, turn.t_start, turn.t_end)
            prediction = self.classify_fn(segment, ctx.sr)
            per_turn.append({
                "start": turn.t_start,
                "end": turn.t_end,
                "label": prediction["label"],
                "confidence": prediction["confidence"],
            })

        dominant_label = Counter(pt["label"] for pt in per_turn).most_common(1)[0][0]
        mean_confidence = sum(pt["confidence"] for pt in per_turn) / len(per_turn)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS,
            gating=self.default_gating,
            score=mean_confidence,
            details={"per_turn": per_turn, "dominant_label": dominant_label, "mean_confidence": mean_confidence},
            evaluator_version=self.version,
        )
