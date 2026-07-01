import numpy as np
import pytest

from eval_system.context.metric_context import MetricContext, Turn
from eval_system.metrics.acoustic.emotion_appropriateness_mm import (
    JUDGE_PROMPT_VERSION,
    MultimodalAppropriatenessJudgment,
    MultimodalEmotionAppropriatenessMetric,
)
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.registry import _safe


def _make_ctx(transcript, audio_agent=None):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(32000) if audio_agent is None else audio_agent,
        audio_caller=np.zeros(32000), transcript=transcript, tool_events=[], events=[],
        expected={}, scenario_db={},
    )


class FakeGeminiClient:
    model = "fake-gemini-model"

    def __init__(self, judgments):
        self.judgments = iter(judgments)
        self.calls = 0
        self.last_context = None

    def judge_audio(self, audio_bytes, context_text, response_model):
        self.calls += 1
        self.last_context = context_text
        assert response_model is MultimodalAppropriatenessJudgment
        assert isinstance(audio_bytes, bytes) and len(audio_bytes) > 0
        return next(self.judgments)


def test_missing_api_key_is_error_not_fail(monkeypatch, tmp_path):
    from eval_system.judges import factory

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(factory, "REPO_ROOT_DOTENV", tmp_path / "does_not_exist.env")

    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="Sorry about the wait.")])
    metric = MultimodalEmotionAppropriatenessMetric(cache_path=tmp_path / "cache.json")

    score = _safe(metric, ctx)

    assert score.status is Status.ERROR
    assert score.status is not Status.FAIL


def test_cache_prevents_second_call(tmp_path):
    fake = FakeGeminiClient([
        MultimodalAppropriatenessJudgment(appropriate=True, score=5, detected_tone="calm", rationale="fine"),
    ])
    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="Sorry about the wait.")])
    cache_path = tmp_path / "cache.json"

    MultimodalEmotionAppropriatenessMetric(client=fake, cache_path=cache_path).compute(ctx)
    assert fake.calls == 1

    # A second metric instance (fresh in-memory state) sharing the same cache
    # file must NOT call the SDK again -- proves reproducibility across runs.
    MultimodalEmotionAppropriatenessMetric(client=fake, cache_path=cache_path).compute(ctx)
    assert fake.calls == 1


def test_per_turn_results_and_overall_status_fails_if_any_turn_inappropriate(tmp_path):
    fake = FakeGeminiClient([
        MultimodalAppropriatenessJudgment(appropriate=True, score=5, detected_tone="calm", rationale="fine"),
        MultimodalAppropriatenessJudgment(appropriate=False, score=2, detected_tone="cheerful", rationale="wrong tone"),
    ])
    transcript = [
        Turn(speaker="agent", t_start=0.0, t_end=1.0, text="One moment please."),
        Turn(speaker="caller", t_start=1.0, t_end=2.0, text="My appointment got cancelled?"),
        Turn(speaker="agent", t_start=2.0, t_end=3.0, text="Yep all set, have a great day!"),
    ]
    ctx = _make_ctx(transcript)

    score = MultimodalEmotionAppropriatenessMetric(client=fake, cache_path=tmp_path / "cache.json").compute(ctx)

    assert score.status is Status.FAIL
    assert score.kind is MetricKind.JUDGE
    assert score.gating is Gating.ADVISORY
    assert score.judge_prompt_version == JUDGE_PROMPT_VERSION
    assert len(score.details["per_turn"]) == 2
    assert score.details["per_turn"][0]["appropriate"] is True
    assert score.details["per_turn"][1]["appropriate"] is False
    assert score.details["per_turn"][1]["detected_tone"] == "cheerful"
    assert score.score == pytest.approx((5 + 2) / 2)


def test_context_includes_prior_turn_text(tmp_path):
    fake = FakeGeminiClient([
        MultimodalAppropriatenessJudgment(appropriate=True, score=4, detected_tone="calm", rationale="ok"),
    ])
    transcript = [
        Turn(speaker="caller", t_start=0.0, t_end=1.0, text="My appointment got cancelled?"),
        Turn(speaker="agent", t_start=1.0, t_end=2.0, text="Yes, I'm sorry about that."),
    ]
    ctx = _make_ctx(transcript)

    MultimodalEmotionAppropriatenessMetric(client=fake, cache_path=tmp_path / "cache.json").compute(ctx)

    assert "My appointment got cancelled?" in fake.last_context


def test_skipped_when_no_agent_turns(tmp_path):
    fake = FakeGeminiClient([])
    ctx = _make_ctx([Turn(speaker="caller", t_start=0.0, t_end=1.0, text="hello?")])

    score = MultimodalEmotionAppropriatenessMetric(client=fake, cache_path=tmp_path / "cache.json").compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None


def test_missing_audio_is_error_via_registry_safe(tmp_path):
    fake = FakeGeminiClient([])
    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="hi")], audio_agent=np.zeros(0))
    metric = MultimodalEmotionAppropriatenessMetric(client=fake, cache_path=tmp_path / "cache.json")

    score = _safe(metric, ctx)

    assert score.status is Status.ERROR
