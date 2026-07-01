from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.acoustic.naturalness_mos import NaturalnessMosMetric, mos_band
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.registry import _safe

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(transcript, audio_agent=None):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(16000) if audio_agent is None else audio_agent,
        audio_caller=np.zeros(16000), transcript=transcript, tool_events=[], events=[],
        expected={}, scenario_db={},
    )


def _fake_mos_sequence(values):
    it = iter(values)

    def mos_fn(audio, sr):
        return next(it)

    return mos_fn


def test_mos_band_good_at_or_above_4():
    assert mos_band(4.0) == "good"
    assert mos_band(4.5) == "good"


def test_mos_band_acceptable_between_3_5_and_4():
    assert mos_band(3.7) == "acceptable"


def test_mos_band_dissatisfaction_below_3_5():
    assert mos_band(3.2) == "dissatisfaction"


def test_mos_band_none_for_none():
    assert mos_band(None) is None


def test_metric_runs_and_is_always_advisory():
    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="Sure, one moment.")]
    ctx = _make_ctx(transcript)

    score = NaturalnessMosMetric(mos_fn=_fake_mos_sequence([4.2])).compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.status is Status.PASS
    assert score.score == pytest.approx(4.2)
    assert score.details["mean_mos"] == pytest.approx(4.2)
    assert score.details["mos_engine"] == "dnsmos"
    assert score.details["per_turn"] == [{"start": 0.0, "end": 1.0, "mos": 4.2}]


def test_metric_flags_dissatisfaction_but_stays_advisory():
    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hi")]
    ctx = _make_ctx(transcript)

    score = NaturalnessMosMetric(mos_fn=_fake_mos_sequence([2.5])).compute(ctx)

    assert score.status is Status.FAIL  # flagged
    assert score.gating is Gating.ADVISORY  # never blocks ship regardless
    assert score.details["band"] == "dissatisfaction"


def test_mean_mos_averages_across_turns():
    transcript = [
        Turn(speaker="agent", t_start=0.0, t_end=1.0, text="a"),
        Turn(speaker="agent", t_start=1.0, t_end=2.0, text="b"),
    ]
    ctx = _make_ctx(transcript)

    score = NaturalnessMosMetric(mos_fn=_fake_mos_sequence([4.0, 3.0])).compute(ctx)

    assert score.details["mean_mos"] == pytest.approx(3.5)


def test_skipped_when_no_agent_turns():
    ctx = _make_ctx([Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hi")])

    score = NaturalnessMosMetric(mos_fn=_fake_mos_sequence([])).compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_missing_audio_is_error_via_registry_safe():
    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hi")], audio_agent=np.zeros(0))
    metric = NaturalnessMosMetric(mos_fn=_fake_mos_sequence([]))

    score = _safe(metric, ctx)

    assert score.status is Status.ERROR


def test_model_failure_is_error_via_registry_safe():
    def broken_mos_fn(audio, sr):
        raise RuntimeError("model load failed")

    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hi")])
    metric = NaturalnessMosMetric(mos_fn=broken_mos_fn)

    score = _safe(metric, ctx)

    assert score.status is Status.ERROR
    assert score.status is not Status.FAIL


def test_real_fixture_runs_end_to_end_with_real_model():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = NaturalnessMosMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert 1.0 <= score.details["mean_mos"] <= 5.0
    assert score.details["mos_engine"] == "dnsmos"
