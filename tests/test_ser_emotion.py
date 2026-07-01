from pathlib import Path

import numpy as np
import pytest

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.acoustic.ser_emotion import SerEmotionMetric, ser_label_to_valence
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.registry import _safe

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(transcript, audio_agent=None):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(16000) if audio_agent is None else audio_agent,
        audio_caller=np.zeros(16000), transcript=transcript, tool_events=[], events=[],
        expected={}, scenario_db={},
    )


def _fake_classify_sequence(results):
    it = iter(results)

    def classify(audio, sr):
        return next(it)

    return classify


def test_valence_mapping():
    assert ser_label_to_valence("hap") == "positive"
    assert ser_label_to_valence("sad") == "negative"
    assert ser_label_to_valence("ang") == "negative"
    assert ser_label_to_valence("neu") == "neutral"
    assert ser_label_to_valence("unknown_label") == "neutral"


def test_ser_runs_and_is_advisory():
    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="Sure, one moment.")]
    ctx = _make_ctx(transcript)
    classify = _fake_classify_sequence([{"label": "neu", "confidence": 0.8}])

    score = SerEmotionMetric(classify_fn=classify).compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.status is Status.PASS
    assert score.score == pytest.approx(0.8)
    assert score.details["per_turn"] == [{"start": 0.0, "end": 1.0, "label": "neu", "confidence": 0.8}]
    assert score.details["dominant_label"] == "neu"
    assert score.details["mean_confidence"] == pytest.approx(0.8)


def test_dominant_label_is_most_frequent_across_turns():
    transcript = [
        Turn(speaker="agent", t_start=0.0, t_end=1.0, text="a"),
        Turn(speaker="agent", t_start=1.0, t_end=2.0, text="b"),
        Turn(speaker="agent", t_start=2.0, t_end=3.0, text="c"),
    ]
    ctx = _make_ctx(transcript)
    classify = _fake_classify_sequence([
        {"label": "hap", "confidence": 0.9},
        {"label": "hap", "confidence": 0.6},
        {"label": "neu", "confidence": 0.7},
    ])

    score = SerEmotionMetric(classify_fn=classify).compute(ctx)

    assert score.details["dominant_label"] == "hap"
    assert score.details["mean_confidence"] == pytest.approx((0.9 + 0.6 + 0.7) / 3)


def test_only_agent_turns_are_classified():
    transcript = [
        Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hi"),
        Turn(speaker="agent", t_start=1.0, t_end=2.0, text="hello"),
    ]
    ctx = _make_ctx(transcript)
    classify = _fake_classify_sequence([{"label": "neu", "confidence": 0.5}])

    score = SerEmotionMetric(classify_fn=classify).compute(ctx)

    assert len(score.details["per_turn"]) == 1
    assert score.details["per_turn"][0]["start"] == 1.0


def test_skipped_when_no_agent_turns():
    ctx = _make_ctx([Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hi")])
    classify = _fake_classify_sequence([])

    score = SerEmotionMetric(classify_fn=classify).compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_missing_audio_is_error_via_registry_safe():
    ctx = _make_ctx(
        [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hi")],
        audio_agent=np.zeros(0),
    )
    metric = SerEmotionMetric(classify_fn=_fake_classify_sequence([]))

    score = _safe(metric, ctx)

    assert score.status is Status.ERROR


def test_classifier_exception_is_error_via_registry_safe():
    def classify(audio, sr):
        raise RuntimeError("model load failed")

    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hi")])
    metric = SerEmotionMetric(classify_fn=classify)

    score = _safe(metric, ctx)

    assert score.status is Status.ERROR
    assert score.status is not Status.FAIL


def test_real_fixture_runs_end_to_end_with_real_model():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = SerEmotionMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.ADVISORY
    assert score.status is Status.PASS
    assert len(score.details["per_turn"]) > 0
