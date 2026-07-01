from eval_system.context.metric_context import MetricContext, Turn
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.semantic.instruction_adherence import (
    InstructionAdherenceJudgeMetric,
    InstructionAdherenceJudgment,
)


class FakeJudgeClient:
    def __init__(self, judgment: InstructionAdherenceJudgment):
        self.judgment = judgment
        self.last_prompt: str | None = None

    def structured_complete(self, prompt, response_model):
        self.last_prompt = prompt
        assert response_model is InstructionAdherenceJudgment
        return self.judgment


def _make_ctx(transcript):
    return MetricContext(
        call_id="call-1",
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=transcript,
        tool_events=[],
        events=[],
        expected={},
        scenario_db={},
    )


def test_passes_when_judge_finds_rules_followed():
    fake = FakeJudgeClient(InstructionAdherenceJudgment(followed_rules=True, score=0.9, notes="fine"))
    ctx = _make_ctx([Turn(speaker="agent", t_start=0.0, t_end=1.0, text="Sure, one moment.")])

    score = InstructionAdherenceJudgeMetric(client=fake).compute(ctx)

    assert score.status is Status.PASS
    assert score.score == 0.9
    assert score.kind is MetricKind.JUDGE
    assert score.gating is Gating.ADVISORY
    assert score.judge_prompt_version == InstructionAdherenceJudgeMetric.prompt_version
    assert "Sure, one moment." in fake.last_prompt


def test_fails_when_judge_finds_rules_broken():
    fake = FakeJudgeClient(InstructionAdherenceJudgment(followed_rules=False, score=0.1, notes="made a promise it can't keep"))
    ctx = _make_ctx([])

    score = InstructionAdherenceJudgeMetric(client=fake).compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["notes"] == "made a promise it can't keep"
