from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, ToolEvent, Turn, build_metric_context
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.semantic.faithfulness import FaithfulnessJudgment, FaithfulnessMetric

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


class FakeJudgeClient:
    def __init__(self, judgment: FaithfulnessJudgment):
        self.judgment = judgment
        self.last_prompt: str | None = None

    def structured_complete(self, prompt, response_model):
        self.last_prompt = prompt
        assert response_model is FaithfulnessJudgment
        return self.judgment


def _make_ctx(transcript, tool_events):
    return MetricContext(
        call_id="call-1",
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=transcript,
        tool_events=tool_events,
        events=[],
        expected={},
        scenario_db={},
    )


def test_passes_when_judge_finds_statements_grounded():
    fake = FakeJudgeClient(FaithfulnessJudgment(grounded=True, score=0.95, rationale="ok"))
    ctx = _make_ctx(transcript=[], tool_events=[])

    score = FaithfulnessMetric(client=fake).compute(ctx)

    assert score.status is Status.PASS
    assert score.score == 0.95
    assert score.kind is MetricKind.JUDGE
    assert score.gating is Gating.ADVISORY
    assert score.judge_prompt_version == FaithfulnessMetric.prompt_version


def test_fails_when_judge_finds_ungrounded_claims():
    fake = FakeJudgeClient(
        FaithfulnessJudgment(
            grounded=False, score=0.2, ungrounded_claims=["invented a discount"], rationale="not in tool result"
        )
    )
    ctx = _make_ctx(transcript=[], tool_events=[])

    score = FaithfulnessMetric(client=fake).compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["ungrounded_claims"] == ["invented a discount"]


def test_prompt_grounds_in_real_tool_results_and_agent_transcript():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)
    fake = FakeJudgeClient(FaithfulnessJudgment(grounded=True, score=1.0, rationale="ok"))

    FaithfulnessMetric(client=fake).compute(ctx)

    assert "48213" in fake.last_prompt  # real confirmation number from the tool result
    assert "book_appointment" in fake.last_prompt
    assert "confirmation number is four eight two one three" in fake.last_prompt


def test_low_asr_confidence_turn_is_flagged_in_prompt():
    transcript = [Turn(speaker="caller", t_start=0.0, t_end=1.0, text="garbled", asr_confidence=0.3)]
    ctx = _make_ctx(transcript, tool_events=[])
    fake = FakeJudgeClient(FaithfulnessJudgment(grounded=True, score=1.0, rationale="ok"))

    FaithfulnessMetric(client=fake).compute(ctx)

    assert "low-confidence" in fake.last_prompt


def test_missing_asr_confidence_does_not_add_low_confidence_note():
    transcript = [Turn(speaker="caller", t_start=0.0, t_end=1.0, text="clear speech")]
    ctx = _make_ctx(transcript, tool_events=[])
    fake = FakeJudgeClient(FaithfulnessJudgment(grounded=True, score=1.0, rationale="ok"))

    FaithfulnessMetric(client=fake).compute(ctx)

    assert "low-confidence" not in fake.last_prompt


def test_score_outside_0_to_1_is_rejected():
    with pytest.raises(ValidationError):
        FaithfulnessJudgment(grounded=True, score=1.5, rationale="oops")
