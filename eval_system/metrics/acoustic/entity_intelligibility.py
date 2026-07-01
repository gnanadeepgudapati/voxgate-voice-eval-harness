"""Round-trip STT: does the agent's rendered AUDIO still convey the critical
entities (confirmation numbers, provider names, ...) clearly enough for an
independent speech recognizer to pick them up? This is a stronger test of
acoustic clarity than `instruction_adherence_rule`'s text-only check on the
authored transcript -- it can catch audio that's technically correct text but
poorly synthesized/garbled. Gate: if a critical entity doesn't survive, the
caller likely couldn't understand it either. STT is a swappable seam
(`stt_fn`) like VAD/judges; unit tests inject a fake, one smoke test runs
faster-whisper for real."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

import numpy as np

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

STT_SAMPLE_RATE = 16000

SttFn = Callable[[np.ndarray, int], str]

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def _faster_whisper_transcribe(audio: np.ndarray, sr: int) -> str:
    import librosa
    from faster_whisper import WhisperModel

    resampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=STT_SAMPLE_RATE)
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(resampled, language="en")
    return " ".join(segment.text for segment in segments)


def _normalize_spoken_numbers(text: str) -> str:
    """Real STT tends to render spoken digits/small numbers as numerals
    ("4-8213" for "four eight two one three", "10 AM" for "ten AM") -- a naive
    substring check against the authored (word-form) entity misses this
    entirely, producing a false FAIL on a gate metric. Normalize both sides to
    the same word->digit form before comparing."""
    text = re.sub(r"[.,\-]", " ", text.lower())
    words = text.split()
    return " ".join(_NUMBER_WORDS.get(w, w) for w in words)


def _is_numeric_entity(entity: str) -> bool:
    words = entity.lower().split()
    return bool(words) and all(w in _NUMBER_WORDS or w.isdigit() for w in words)


def missing_critical_entities(critical_entities: list[str], stt_text: str) -> list[str]:
    normalized_text = _normalize_spoken_numbers(stt_text)
    text_digits = re.sub(r"\D", "", normalized_text)

    missing = []
    for entity in critical_entities:
        if _is_numeric_entity(entity):
            entity_digits = re.sub(r"\D", "", _normalize_spoken_numbers(entity))
            if entity_digits not in text_digits:
                missing.append(entity)
        elif _normalize_spoken_numbers(entity) not in normalized_text:
            missing.append(entity)
    return missing


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    if not reference.strip():
        return None
    import jiwer

    return jiwer.wer(reference, hypothesis)


@register
class EntityIntelligibilityMetric(BaseMetric):
    name = "entity_intelligibility"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.GATE
    requires_ground_truth = True

    def __init__(self, stt_fn: SttFn | None = None):
        self.stt_fn = stt_fn or _faster_whisper_transcribe

    def compute(self, ctx: "MetricContext") -> MetricScore:
        critical_entities = ctx.expected.get("critical_entities", [])
        if not critical_entities:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "no critical_entities defined for this fixture"},
                evaluator_version=self.version,
            )

        stt_text = self.stt_fn(ctx.audio_agent, ctx.sr)
        missing = missing_critical_entities(critical_entities, stt_text)
        reference = " ".join(turn.text for turn in ctx.transcript if turn.speaker == "agent")

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.FAIL if missing else Status.PASS,
            gating=self.default_gating,
            score=0.0 if missing else 1.0,
            details={
                "missing_entities": missing,
                "stt_text": stt_text,
                "wer": word_error_rate(reference, stt_text),
            },
            evaluator_version=self.version,
        )
