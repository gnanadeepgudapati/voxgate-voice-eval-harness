from pathlib import Path

import numpy as np

from eval_system.context.fixture_loader import RawFixture, load_fixture
from eval_system.validators.preflight import (
    ProcessingResult,
    preflight_check,
    process_fixture_with_retry,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SR = 8000


def _fixture(audio_caller, audio_agent, transcript=None):
    return RawFixture(
        call_id="synthetic", sr=SR, audio_caller=audio_caller, audio_agent=audio_agent,
        transcript=transcript or [], tool_events=[], events=[], scenario_db={},
        expected={"tool_sequence": [], "invariants": [], "critical_entities": []},
    )


def test_clean_real_fixture_has_no_issues():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")

    assert preflight_check(fixture) == []


def test_flags_channel_length_mismatch():
    fixture = _fixture(np.zeros(100), np.zeros(50))

    issues = preflight_check(fixture)

    assert any(i.check == "channels" for i in issues)


def test_flags_clipping():
    audio = np.zeros(100)
    audio[10] = 1.0  # full-scale sample
    fixture = _fixture(audio, np.zeros(100))

    issues = preflight_check(fixture)

    assert any(i.check == "clipping" for i in issues)


def test_flags_turn_extending_past_call_duration():
    from eval_system.context.metric_context import Turn

    fixture = _fixture(np.zeros(100), np.zeros(100), transcript=[
        Turn(speaker="agent", t_start=0.0, t_end=1000.0, text="way past the end"),
    ])

    issues = preflight_check(fixture)

    assert any(i.check == "timeline" for i in issues)


def test_process_with_retry_returns_ok_immediately_for_clean_fixture():
    def load_fn(_):
        return load_fixture(FIXTURES_DIR / "happy_path_book")

    result = process_fixture_with_retry("happy_path_book", load_fn)

    assert result.status == "ok"
    assert result.attempts == 1
    assert result.fixture is not None


def test_process_with_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def load_fn(_):
        calls["n"] += 1
        if calls["n"] < 2:
            raise IOError("transient")
        return load_fixture(FIXTURES_DIR / "happy_path_book")

    result = process_fixture_with_retry("happy_path_book", load_fn, max_retries=3)

    assert result.status == "ok"
    assert result.attempts == 2


def test_process_with_retry_quarantines_after_exhausting_attempts():
    def load_fn(_):
        raise IOError("permanent failure")

    result = process_fixture_with_retry("broken", load_fn, max_retries=2)

    assert result.status == "quarantined"
    assert result.attempts == 2
    assert result.fixture is None
    assert len(result.issues) == 1
