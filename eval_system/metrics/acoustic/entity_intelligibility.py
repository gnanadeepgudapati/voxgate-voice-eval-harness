"""Round-trip STT: does the agent's rendered AUDIO still convey the critical
entities (confirmation numbers, provider names, ...) clearly enough for an
independent speech recognizer to pick them up? This is a stronger test of
acoustic clarity than `instruction_adherence_rule`'s text-only check on the
authored transcript -- it can catch audio that's technically correct text but
poorly synthesized/garbled. Gate: if a critical entity doesn't survive, the
caller likely couldn't understand it either -- and gating is based SOLELY on
critical-entity survival, never on overall WER (WER is informational only,
banded for a report, see `wer_band()`). STT is a swappable seam (`stt_fn`)
like VAD/judges; unit tests inject a fake, one smoke test runs faster-whisper
for real.

ASR backend note: faster-whisper's own `word_timestamps=True` (native to the
already-installed "tiny" model, no new dependency) is used to locate WHERE
each critical entity landed, not just whether it survived -- verified against
`WhisperX` as an alternative first; that path was rejected because installing
it forces a downgrade of transformers/torch/torchaudio that broke the
already-working `ser_emotion.py` SER metric (confirmed empirically, then
reverted). `details["asr_engine"]` records which engine actually ran, so a
future swap stays visible in the report."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

import numpy as np

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

STT_SAMPLE_RATE = 16000
ASR_ENGINE = "faster-whisper"

SttFn = Callable[[np.ndarray, int], dict]

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12",
}


def _faster_whisper_transcribe(audio: np.ndarray, sr: int) -> dict:
    """Returns {"text": str, "words": [{"word","start","end","probability"}, ...]}."""
    import librosa
    from faster_whisper import WhisperModel

    resampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=STT_SAMPLE_RATE)
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(resampled, language="en", word_timestamps=True)

    text_parts = []
    words = []
    for segment in segments:
        text_parts.append(segment.text)
        for w in segment.words or []:
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end, "probability": w.probability})
    return {"text": " ".join(text_parts), "words": words}


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


def wer_band(wer: float | None) -> str | None:
    """Informational only -- never affects gating (that's critical-entity
    survival alone). 10-20% has no named band in the spec this follows;
    labeled "fair" here rather than left unclassified."""
    if wer is None:
        return None
    if wer < 0.05:
        return "excellent"
    if wer <= 0.10:
        return "good"
    if wer > 0.20:
        return "poor"
    return "fair"


def _word_span_for_digits(entity_digits: str, words: list[dict]) -> dict | None:
    digit_chars: list[tuple[str, int]] = []
    for i, w in enumerate(words):
        normalized = _normalize_spoken_numbers(w["word"])
        for ch in re.sub(r"\D", "", normalized):
            digit_chars.append((ch, i))

    full_digits = "".join(c for c, _ in digit_chars)
    idx = full_digits.find(entity_digits)
    if idx == -1 or not entity_digits:
        return None

    matched_word_indices = sorted({digit_chars[i][1] for i in range(idx, idx + len(entity_digits))})
    matched = [words[i] for i in matched_word_indices]
    return {
        "start": matched[0]["start"],
        "end": matched[-1]["end"],
        "confidence": sum(w["probability"] for w in matched) / len(matched),
    }


def _word_span_for_text(entity: str, words: list[dict]) -> dict | None:
    entity_normalized = _normalize_spoken_numbers(entity)
    entity_tokens = entity_normalized.split()
    n = len(entity_tokens)
    if n == 0:
        return None
    word_tokens = [_normalize_spoken_numbers(w["word"]) for w in words]

    for i in range(len(words) - n + 1):
        if " ".join(word_tokens[i:i + n]) == entity_normalized:
            matched = words[i:i + n]
            return {
                "start": matched[0]["start"],
                "end": matched[-1]["end"],
                "confidence": sum(w["probability"] for w in matched) / len(matched),
            }
    return None


def locate_critical_entities(critical_entities: list[str], words: list[dict]) -> list[dict]:
    """For each critical entity, locates the word-level timestamp span where
    it was actually recognized -- not just whether it survived (that's
    `missing_critical_entities`, unchanged), but WHERE, so a mangled entity
    can be pinpointed in the audio rather than only flagged."""
    results = []
    for entity in critical_entities:
        if _is_numeric_entity(entity):
            span = _word_span_for_digits(re.sub(r"\D", "", _normalize_spoken_numbers(entity)), words)
        else:
            span = _word_span_for_text(entity, words)

        if span is None:
            results.append({"entity": entity, "found": False})
        else:
            results.append({"entity": entity, "found": True, **span})
    return results


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

        stt_result = self.stt_fn(ctx.audio_agent, ctx.sr)
        stt_text = stt_result["text"]
        words = stt_result["words"]

        missing = missing_critical_entities(critical_entities, stt_text)
        reference = " ".join(turn.text for turn in ctx.transcript if turn.speaker == "agent")
        wer = word_error_rate(reference, stt_text)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            # Gating is unchanged: based solely on critical-entity survival,
            # never on overall WER (informational only, see wer_band()).
            status=Status.FAIL if missing else Status.PASS,
            gating=self.default_gating,
            score=0.0 if missing else 1.0,
            details={
                "missing_entities": missing,
                "stt_text": stt_text,
                "wer": wer,
                "wer_band": wer_band(wer),
                "asr_engine": ASR_ENGINE,
                "critical_entity_locations": locate_critical_entities(critical_entities, words),
            },
            evaluator_version=self.version,
        )
