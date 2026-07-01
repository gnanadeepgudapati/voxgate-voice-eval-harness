from pathlib import Path

import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import build_metric_context
from eval_system.metrics.acoustic.turn_taking_latency import (
    TurnTakingLatencyMetric,
    caller_to_agent_gaps,
)
from eval_system.metrics.acoustic.vad import SpeechSegment
from eval_system.metrics.base import Gating, MetricKind, Status

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def test_computes_caller_to_agent_gaps_only():
    caller = [SpeechSegment(0.0, 1.0), SpeechSegment(3.0, 3.5)]
    agent = [SpeechSegment(1.2, 2.0), SpeechSegment(3.9, 4.5)]

    gaps = caller_to_agent_gaps(caller, agent)

    assert gaps == pytest.approx([0.2, 0.4])


def test_agent_to_caller_transitions_are_not_counted():
    # Only caller-end -> agent-onset gaps matter (measures agent responsiveness).
    caller = [SpeechSegment(1.5, 2.0)]
    agent = [SpeechSegment(0.0, 1.0)]

    gaps = caller_to_agent_gaps(caller, agent)

    assert gaps == []


def test_overlapping_barge_in_transitions_are_excluded():
    caller = [SpeechSegment(0.0, 2.0)]
    agent = [SpeechSegment(1.5, 3.0)]  # agent started before caller finished

    gaps = caller_to_agent_gaps(caller, agent)

    assert gaps == []


def test_metric_reports_percentile_distribution_not_just_mean():
    def fake_vad(audio, sr):
        return {"caller": [SpeechSegment(0.0, 1.0)], "agent": [SpeechSegment(1.1, 1.5)]}[
            "caller" if id(audio) == caller_id else "agent"
        ]

    ctx = _make_ctx()
    caller_id = id(ctx.audio_caller)

    score = TurnTakingLatencyMetric(vad_fn=fake_vad).compute(ctx)

    assert score.status is Status.PASS
    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.score == pytest.approx(0.1)
    assert set(score.details.keys()) >= {"p50", "p90", "p99", "n", "gaps"}


def test_skipped_when_no_caller_to_agent_transitions():
    def fake_vad(audio, sr):
        return []

    score = TurnTakingLatencyMetric(vad_fn=fake_vad).compute(_make_ctx())

    assert score.status is Status.SKIPPED
    assert score.score is None


def _make_ctx():
    import numpy as np
    from eval_system.context.metric_context import MetricContext

    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(10), audio_caller=np.zeros(10),
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )


def test_real_fixture_runs_end_to_end_with_real_vad():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = TurnTakingLatencyMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.status in (Status.PASS, Status.SKIPPED)
