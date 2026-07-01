from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SCENARIOS = ["happy_path_book", "reschedule_trap", "barge_in_basic"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_fixture_loads_and_validates(scenario):
    fixture = load_fixture(FIXTURES_DIR / scenario)

    assert fixture.call_id == scenario
    assert fixture.sr > 0
    assert isinstance(fixture.audio_caller, np.ndarray)
    assert isinstance(fixture.audio_agent, np.ndarray)
    assert fixture.audio_caller.shape == fixture.audio_agent.shape
    assert fixture.audio_caller.ndim == 1  # channels already split
    assert len(fixture.audio_caller) > 0

    assert isinstance(fixture.scenario_db, dict)
    assert isinstance(fixture.expected, dict)
    assert "tool_sequence" in fixture.expected
    assert "invariants" in fixture.expected
    assert "critical_entities" in fixture.expected

    for turn in fixture.transcript:
        assert turn.speaker in ("agent", "caller")
        assert turn.t_end >= turn.t_start
        assert turn.t_end <= len(fixture.audio_caller) / fixture.sr + 1e-6

    for evt in fixture.tool_events:
        assert evt.t >= 0

    for evt in fixture.events:
        assert evt.t >= 0


def test_happy_path_has_book_appointment_call():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    names = [t.name for t in fixture.tool_events]
    assert names == ["check_availability", "book_appointment"]


def test_reschedule_trap_has_zero_appointment_window_markers():
    fixture = load_fixture(FIXTURES_DIR / "reschedule_trap")
    names = [t.name for t in fixture.tool_events]
    assert names == ["cancel_appointment", "book_appointment", "book_appointment"]
    event_names = [e.name for e in fixture.events]
    assert "zero_appointments_start" in event_names
    assert "zero_appointments_end" in event_names
    assert fixture.expected["invariants"] == ["never_zero_appointments"]


def test_barge_in_basic_has_yield_events_with_no_tool_calls():
    fixture = load_fixture(FIXTURES_DIR / "barge_in_basic")
    assert fixture.tool_events == []
    event_names = [e.name for e in fixture.events]
    assert event_names.count("agent_yield") == 2
    assert "cough" in event_names
    assert "barge_in_start" in event_names
