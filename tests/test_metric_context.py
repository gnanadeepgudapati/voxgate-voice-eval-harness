import numpy as np
import pytest

from eval_system.context.fixture_loader import RawFixture, load_fixture
from eval_system.context.metric_context import (
    Event,
    MetricContext,
    Turn,
    build_metric_context,
)

FIXTURES_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "fixtures"
SR = 8000


def _blank_fixture(audio_caller: np.ndarray, audio_agent: np.ndarray) -> RawFixture:
    return RawFixture(
        call_id="synthetic",
        sr=SR,
        audio_caller=audio_caller,
        audio_agent=audio_agent,
        transcript=[],
        tool_events=[],
        events=[],
        scenario_db={},
        expected={"tool_sequence": [], "invariants": [], "critical_entities": []},
    )


def test_zero_offset_passes_audio_through_unchanged():
    caller = np.zeros(100, dtype=np.float32)
    agent = np.zeros(100, dtype=np.float32)
    agent[40] = 1.0
    fixture = _blank_fixture(caller, agent)

    ctx = build_metric_context(fixture)

    assert isinstance(ctx, MetricContext)
    np.testing.assert_array_equal(ctx.audio_agent, agent)
    np.testing.assert_array_equal(ctx.audio_caller, caller)


def test_known_channel_offset_is_corrected_onto_canonical_clock():
    n = 200
    true_marker_sample = 120  # where the marker sits on the canonical clock
    caller = np.zeros(n, dtype=np.float32)
    caller[true_marker_sample] = 1.0

    offset_samples = 15  # agent channel's raw capture started 15 samples late
    raw_agent = np.zeros(n, dtype=np.float32)
    raw_agent[true_marker_sample - offset_samples] = 1.0
    fixture = _blank_fixture(caller, raw_agent)

    ctx = build_metric_context(fixture, agent_channel_offset_sec=offset_samples / SR)

    corrected_marker_sample = int(np.argmax(ctx.audio_agent))
    assert corrected_marker_sample == true_marker_sample


def test_negative_channel_offset_is_corrected_onto_canonical_clock():
    n = 200
    true_marker_sample = 80
    caller = np.zeros(n, dtype=np.float32)
    caller[true_marker_sample] = 1.0

    offset_samples = -10  # agent channel's raw capture started 10 samples early
    raw_agent = np.zeros(n, dtype=np.float32)
    raw_agent[true_marker_sample - offset_samples] = 1.0
    fixture = _blank_fixture(caller, raw_agent)

    ctx = build_metric_context(fixture, agent_channel_offset_sec=offset_samples / SR)

    corrected_marker_sample = int(np.argmax(ctx.audio_agent))
    assert corrected_marker_sample == true_marker_sample


def test_transcript_tool_events_and_markers_pass_through_unchanged():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")

    ctx = build_metric_context(fixture)

    assert ctx.transcript == fixture.transcript
    assert ctx.tool_events == fixture.tool_events
    assert ctx.events == fixture.events
    assert ctx.expected == fixture.expected
    assert ctx.scenario_db == fixture.scenario_db
    assert ctx.call_id == fixture.call_id
    assert ctx.sr == fixture.sr

    call_duration = len(ctx.audio_caller) / ctx.sr
    for turn in ctx.transcript:
        assert turn.t_end <= call_duration + 1e-6
    for tool_event in ctx.tool_events:
        assert tool_event.t <= call_duration + 1e-6


def test_missing_asr_confidence_is_unchanged_by_join():
    caller = np.zeros(10, dtype=np.float32)
    agent = np.zeros(10, dtype=np.float32)
    fixture = _blank_fixture(caller, agent)
    fixture.transcript = [Turn(speaker="caller", t_start=0.0, t_end=0.1, text="hi")]

    ctx = build_metric_context(fixture)

    assert ctx.transcript[0].asr_confidence is None
