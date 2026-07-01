from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.acoustic.entity_intelligibility import (
    EntityIntelligibilityMetric,
    missing_critical_entities,
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


# --- metric-level, injected STT ---

def test_metric_passes_when_all_entities_survive_stt():
    def fake_stt(audio, sr):
        return "booked with doctor lee, confirmation four eight two one three"

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with Lee, confirmation four eight two one three")]
    ctx = _make_ctx(transcript, ["Lee", "four eight two one three"])

    score = EntityIntelligibilityMetric(stt_fn=fake_stt).compute(ctx)

    assert score.status is Status.PASS
    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.GATE
    assert score.details["missing_entities"] == []


def test_metric_fails_when_an_entity_does_not_survive_stt():
    def fake_stt(audio, sr):
        return "booked with doctor lee"  # confirmation number garbled/missing

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with Lee, confirmation four eight two one three")]
    ctx = _make_ctx(transcript, ["Lee", "four eight two one three"])

    score = EntityIntelligibilityMetric(stt_fn=fake_stt).compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["missing_entities"] == ["four eight two one three"]


def test_metric_skipped_when_no_critical_entities_defined():
    ctx = _make_ctx([], [])

    score = EntityIntelligibilityMetric(stt_fn=lambda a, sr: "").compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_real_fixture_runs_end_to_end_with_real_stt():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = EntityIntelligibilityMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.GATE
    assert score.status in (Status.PASS, Status.FAIL)
