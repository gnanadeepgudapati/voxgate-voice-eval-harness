from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.acoustic.entity_intelligibility import (
    EntityIntelligibilityMetric,
    locate_critical_entities,
    missing_critical_entities,
    wer_band,
    word_error_rate,
)
from eval_system.metrics.base import Gating, MetricKind, Status

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(transcript, critical_entities):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(10), audio_caller=np.zeros(10),
        transcript=transcript, tool_events=[], events=[], expected={"critical_entities": critical_entities},
        scenario_db={},
    )


# --- pure logic ---

def test_missing_entities_detected_case_insensitively():
    missing = missing_critical_entities(["Lee", "four eight two one three"], "booked with dr lee, confirmed")

    assert missing == ["four eight two one three"]


def test_no_missing_entities_when_all_present():
    missing = missing_critical_entities(["Lee"], "booked with Dr. Lee")

    assert missing == []


def test_spelled_out_digits_match_numeral_form_in_stt_output():
    # Real STT tends to render spoken digits as numerals ("4-8213" for "four
    # eight two one three") -- a naive substring check misses this entirely.
    missing = missing_critical_entities(
        ["four eight two one three"], "your confirmation number is 4-8213."
    )

    assert missing == []


def test_spelled_out_hour_matches_numeral_form():
    missing = missing_critical_entities(["ten AM"], "an opening at 10 AM works.")

    assert missing == []


def test_partial_word_match_is_not_a_false_positive():
    # "Lee" must not be considered present just because it's a substring of
    # "Leon" -- that's the model mis-transcribing "Lee" + "on", a genuine miss.
    missing = missing_critical_entities(["Lee"], "an opening with dr leon tuesday")

    assert missing == ["Lee"]


def test_word_error_rate_zero_for_identical_text():
    assert word_error_rate("hello there", "hello there") == pytest.approx(0.0)


def test_word_error_rate_none_for_empty_reference():
    assert word_error_rate("", "anything") is None


# --- WER banding (assessment.md-adjacent reporting bands, doesn't affect gating) ---

def test_wer_band_excellent_below_5_percent():
    assert wer_band(0.03) == "excellent"


def test_wer_band_good_between_5_and_10_percent():
    assert wer_band(0.07) == "good"


def test_wer_band_poor_above_20_percent():
    assert wer_band(0.25) == "poor"


def test_wer_band_none_for_none_wer():
    assert wer_band(None) is None


# --- entity location (word-level timestamps via faster-whisper's native
# word_timestamps, so a mangled critical entity can be pinpointed, not just
# flagged) ---

def test_locate_finds_text_entity_span():
    words = [
        {"word": "with", "start": 1.0, "end": 1.2, "probability": 0.99},
        {"word": "Lee", "start": 1.2, "end": 1.5, "probability": 0.95},
        {"word": "Tuesday", "start": 1.5, "end": 2.0, "probability": 0.9},
    ]

    locations = locate_critical_entities(["Lee"], words)

    assert locations == [{"entity": "Lee", "found": True, "start": 1.2, "end": 1.5, "confidence": pytest.approx(0.95)}]


def test_locate_reports_not_found_when_entity_mangled():
    # the real "Lee" -> "Leon" mangling: no word matches "Lee" at all.
    words = [{"word": "Leon", "start": 1.2, "end": 1.6, "probability": 0.62}]

    locations = locate_critical_entities(["Lee"], words)

    assert locations == [{"entity": "Lee", "found": False}]


def test_locate_finds_numeric_entity_span_across_multiple_words():
    words = [
        {"word": "is", "start": 5.0, "end": 5.1, "probability": 0.99},
        {"word": "4", "start": 5.1, "end": 5.3, "probability": 0.9},
        {"word": "-8213.", "start": 5.3, "end": 5.8, "probability": 0.85},
    ]

    locations = locate_critical_entities(["four eight two one three"], words)

    assert locations == [{
        "entity": "four eight two one three", "found": True,
        "start": 5.1, "end": 5.8, "confidence": pytest.approx((0.9 + 0.85) / 2),
    }]


# --- metric-level, injected STT (stt_fn now returns {"text":, "words":}) ---

def test_metric_passes_when_all_entities_survive_stt():
    def fake_stt(audio, sr):
        return {"text": "booked with doctor lee, confirmation four eight two one three", "words": []}

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with Lee, confirmation four eight two one three")]
    ctx = _make_ctx(transcript, ["Lee", "four eight two one three"])

    score = EntityIntelligibilityMetric(stt_fn=fake_stt).compute(ctx)

    assert score.status is Status.PASS
    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.GATE
    assert score.details["missing_entities"] == []
    assert score.details["asr_engine"] == "faster-whisper"


def test_metric_fails_when_an_entity_does_not_survive_stt():
    def fake_stt(audio, sr):
        return {"text": "booked with doctor lee", "words": []}  # confirmation number garbled/missing

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with Lee, confirmation four eight two one three")]
    ctx = _make_ctx(transcript, ["Lee", "four eight two one three"])

    score = EntityIntelligibilityMetric(stt_fn=fake_stt).compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["missing_entities"] == ["four eight two one three"]


def test_metric_fails_on_missing_entity_regardless_of_wer_band():
    # Gating is unchanged: overall WER is informational only. A "poor" WER
    # with no missing critical entities must still PASS the gate.
    def fake_stt(audio, sr):
        # very different wording (high WER) but the critical entity survives
        return {"text": "uh yeah so anyway Lee is the doctor I guess", "words": []}

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with Lee")]
    ctx = _make_ctx(transcript, ["Lee"])

    score = EntityIntelligibilityMetric(stt_fn=fake_stt).compute(ctx)

    assert score.status is Status.PASS  # critical entity present -> gate passes
    assert score.details["wer_band"] in ("poor", "fair", "good", "excellent")  # informational only


def test_metric_includes_critical_entity_locations_with_timestamps():
    def fake_stt(audio, sr):
        return {
            "text": "booked with lee",
            "words": [
                {"word": "booked", "start": 0.0, "end": 0.3, "probability": 0.99},
                {"word": "with", "start": 0.3, "end": 0.5, "probability": 0.99},
                {"word": "lee", "start": 0.5, "end": 0.8, "probability": 0.9},
            ],
        }

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with Lee")]
    ctx = _make_ctx(transcript, ["Lee"])

    score = EntityIntelligibilityMetric(stt_fn=fake_stt).compute(ctx)

    assert score.details["critical_entity_locations"] == [
        {"entity": "Lee", "found": True, "start": 0.5, "end": 0.8, "confidence": pytest.approx(0.9)}
    ]


def test_metric_skipped_when_no_critical_entities_defined():
    ctx = _make_ctx([], [])

    score = EntityIntelligibilityMetric(stt_fn=lambda a, sr: {"text": "", "words": []}).compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_real_fixture_runs_end_to_end_with_real_stt():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = EntityIntelligibilityMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.GATE
    assert score.status in (Status.PASS, Status.FAIL)
    assert score.details["asr_engine"] == "faster-whisper"
    assert len(score.details["critical_entity_locations"]) == len(fixture.expected["critical_entities"])
