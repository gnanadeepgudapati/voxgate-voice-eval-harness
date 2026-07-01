from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, build_metric_context
from eval_system.metrics.acoustic.double_talk import DoubleTalkMetric, total_overlap_seconds
from eval_system.metrics.acoustic.vad import SpeechSegment
from eval_system.metrics.base import Gating, MetricKind, Status

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_computes_total_overlap_across_multiple_segments():
    caller = [SpeechSegment(0.0, 2.0), SpeechSegment(5.0, 6.0)]
    agent = [SpeechSegment(1.5, 3.0), SpeechSegment(5.5, 5.8)]

    overlap = total_overlap_seconds(caller, agent)

    assert overlap == pytest.approx(0.5 + 0.3)  # 1.5-2.0 and 5.5-5.8


def test_zero_overlap_when_no_intervals_intersect():
    caller = [SpeechSegment(0.0, 1.0)]
    agent = [SpeechSegment(2.0, 3.0)]

    assert total_overlap_seconds(caller, agent) == 0.0


def test_metric_reports_overlap_seconds_and_ratio():
    def fake_vad(audio, sr):
        return {"caller": [SpeechSegment(0.0, 2.0)], "agent": [SpeechSegment(1.0, 3.0)]}[
            "caller" if id(audio) == caller_id else "agent"
        ]

    ctx = MetricContext(
        call_id="call-1", sr=10, audio_agent=np.zeros(40), audio_caller=np.zeros(40),
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )
    caller_id = id(ctx.audio_caller)

    score = DoubleTalkMetric(vad_fn=fake_vad).compute(ctx)

    assert score.status is Status.PASS
    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.details["overlap_seconds"] == pytest.approx(1.0)
    assert score.details["call_duration_sec"] == pytest.approx(4.0)
    assert score.score == pytest.approx(0.25)


def test_skipped_for_zero_duration_audio():
    def fake_vad(audio, sr):
        return []

    ctx = MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(0), audio_caller=np.zeros(0),
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )

    score = DoubleTalkMetric(vad_fn=fake_vad).compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_real_fixture_runs_end_to_end():
    fixture = load_fixture(FIXTURES_DIR / "barge_in_basic")
    ctx = build_metric_context(fixture)

    score = DoubleTalkMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.status in (Status.PASS, Status.SKIPPED)
