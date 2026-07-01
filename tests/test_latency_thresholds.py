from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.acoustic.latency_thresholds import LatencyThresholdsMetric
from eval_system.metrics.base import Gating, MetricKind, Status

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(transcript):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=None, audio_caller=None,
        transcript=transcript, tool_events=[], events=[], expected={}, scenario_db={},
    )


def test_fails_when_a_gap_exceeds_the_threshold():
    transcript = [
        Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hi"),
        Turn(speaker="agent", t_start=5.0, t_end=6.0, text="sorry for the wait"),
    ]
    ctx = _make_ctx(transcript)

    score = LatencyThresholdsMetric().compute(ctx)

    assert score.status is Status.FAIL
    assert score.gating is Gating.ADVISORY  # C1(7): advisory, doesn't gate on its own
    assert score.kind is MetricKind.DETERMINISTIC
    assert len(score.details["violations"]) == 1


def test_passes_when_all_gaps_within_threshold():
    transcript = [
        Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hi"),
        Turn(speaker="agent", t_start=1.5, t_end=2.5, text="hello"),
    ]
    ctx = _make_ctx(transcript)

    score = LatencyThresholdsMetric().compute(ctx)

    assert score.status is Status.PASS
    assert score.details["violations"] == []


def test_skipped_when_no_caller_to_agent_transitions():
    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hello?")])

    score = LatencyThresholdsMetric().compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_real_fixture_runs_end_to_end():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = LatencyThresholdsMetric().compute(ctx)

    assert score.kind is MetricKind.DETERMINISTIC
    assert score.gating is Gating.ADVISORY
    assert score.status in (Status.PASS, Status.FAIL, Status.SKIPPED)
