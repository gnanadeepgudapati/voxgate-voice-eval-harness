import numpy as np

from eval_system.context.metric_context import MetricContext, Turn
from eval_system.metrics.acoustic.emotional_appropriateness import (
    EmotionalAppropriatenessJudgment,
    EmotionalAppropriatenessMetric,
)
from eval_system.metrics.base import Gating, MetricKind, Status


class FakeJudgeClient:
    def __init__(self, judgment: EmotionalAppropriatenessJudgment):
        self.judgment = judgment
        self.last_prompt: str | None = None

    def structured_complete(self, prompt, response_model):
        self.last_prompt = prompt
        assert response_model is EmotionalAppropriatenessJudgment
        return self.judgment


def _fake_prosody(audio, sr):
    return {"voiced_frames": 40, "pitch_std_hz": 25.0, "pitch_range_hz": 90.0, "monotone": False}


def _make_ctx(transcript):
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=np.zeros(10), audio_caller=np.zeros(10),
        transcript=transcript, tool_events=[], events=[], expected={}, scenario_db={},
    )


def test_passes_when_judge_finds_tone_appropriate():
    fake = FakeJudgeClient(EmotionalAppropriatenessJudgment(appropriate=True, score=0.9, notes="calm, apologetic"))
    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="I'm sorry for the confusion.")])

    score = EmotionalAppropriatenessMetric(client=fake, prosody_fn=_fake_prosody).compute(ctx)

    assert score.status is Status.PASS
    assert score.kind is MetricKind.JUDGE
    assert score.gating is Gating.ADVISORY
    assert score.judge_prompt_version == EmotionalAppropriatenessMetric.prompt_version


def test_fails_but_stays_advisory_when_judge_finds_tone_inappropriate():
    fake = FakeJudgeClient(
        EmotionalAppropriatenessJudgment(appropriate=False, score=0.2, notes="cheerful tone during a complaint")
    )
    ctx = _make_ctx([])

    score = EmotionalAppropriatenessMetric(client=fake, prosody_fn=_fake_prosody).compute(ctx)

    # Emotion is advisory ALWAYS -- CLAUDE.md invariant -- even on a FAIL verdict.
    assert score.status is Status.FAIL
    assert score.gating is Gating.ADVISORY
    assert score.details["notes"] == "cheerful tone during a complaint"


def test_prompt_includes_transcript_and_prosody_summary():
    fake = FakeJudgeClient(EmotionalAppropriatenessJudgment(appropriate=True, score=1.0))
    transcript = [Turn(speaker="caller", t_start=0.0, t_end=1.0, text="I am very frustrated right now.")]

    EmotionalAppropriatenessMetric(client=fake, prosody_fn=_fake_prosody).compute(_make_ctx(transcript))

    assert "I am very frustrated right now." in fake.last_prompt
    assert "monotone" in fake.last_prompt.lower()
