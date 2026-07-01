from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.acoustic.pitch_prosody import PitchProsodyMetric, prosody_stats, speech_rate_wps
from eval_system.metrics.base import Gating, MetricKind, Status

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(transcript):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(10), audio_caller=np.zeros(10),
        transcript=transcript, tool_events=[], events=[], expected={}, scenario_db={},
    )


# --- pure logic ---

def test_flat_pitch_is_monotone():
    stats = prosody_stats(np.full(50, 150.0), monotone_std_threshold_hz=10.0)

    assert stats["monotone"] is True
    assert stats["pitch_std_hz"] == 0.0


def test_varied_pitch_is_not_monotone():
    f0 = np.array([120.0, 180.0, 140.0, 200.0, 110.0] * 10)

    stats = prosody_stats(f0, monotone_std_threshold_hz=10.0)

    assert stats["monotone"] is False


def test_no_voiced_frames_reports_none():
    stats = prosody_stats(np.array([]), monotone_std_threshold_hz=10.0)

    assert stats["voiced_frames"] == 0
    assert stats["monotone"] is None
    assert stats["pitch_mean_hz"] is None


def test_pitch_mean_hz_reported_alongside_std_and_range():
    f0 = np.array([100.0, 200.0])

    stats = prosody_stats(f0, monotone_std_threshold_hz=10.0)

    assert stats["pitch_mean_hz"] == pytest.approx(150.0)


def test_speech_rate_computed_from_agent_turns_only():
    transcript = [
        Turn(speaker="caller", t_start=0.0, t_end=10.0, text="one two three four five six seven eight"),
        Turn(speaker="agent", t_start=10.0, t_end=12.0, text="one two three four"),  # 4 words / 2s
    ]

    rate = speech_rate_wps(transcript, speaker="agent")

    assert rate == pytest.approx(2.0)


def test_speech_rate_none_when_no_matching_turns():
    rate = speech_rate_wps([Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hi")], speaker="agent")

    assert rate is None


# --- metric-level, injected F0 extractor ---

def test_metric_flags_monotone_pitch():
    def fake_f0(audio, sr):
        return np.full(50, 150.0)

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=2.0, text="one two three four")]
    score = PitchProsodyMetric(f0_fn=fake_f0).compute(_make_ctx(transcript))

    assert score.status is Status.FAIL
    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert "monotone_pitch" in score.details["issues"]


def test_metric_flags_speech_rate_out_of_range():
    def fake_f0(audio, sr):
        return np.array([120.0, 180.0, 140.0, 200.0, 110.0] * 10)

    transcript = [
        Turn(speaker="agent", t_start=0.0, t_end=1.0, text="one two three four five six seven eight nine ten")
    ]
    score = PitchProsodyMetric(f0_fn=fake_f0).compute(_make_ctx(transcript))

    assert score.status is Status.FAIL
    assert "speech_rate_out_of_range" in score.details["issues"]


def test_metric_passes_with_healthy_prosody_and_rate():
    def fake_f0(audio, sr):
        return np.array([120.0, 180.0, 140.0, 200.0, 110.0] * 10)

    transcript = [Turn(speaker="agent", t_start=0.0, t_end=2.0, text="one two three four five")]
    score = PitchProsodyMetric(f0_fn=fake_f0).compute(_make_ctx(transcript))

    assert score.status is Status.PASS
    assert score.details["issues"] == []


def test_metric_skipped_when_no_agent_data():
    def fake_f0(audio, sr):
        return np.array([])

    score = PitchProsodyMetric(f0_fn=fake_f0).compute(_make_ctx([]))

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_real_fixture_runs_end_to_end():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = PitchProsodyMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.status in (Status.PASS, Status.FAIL, Status.SKIPPED)
