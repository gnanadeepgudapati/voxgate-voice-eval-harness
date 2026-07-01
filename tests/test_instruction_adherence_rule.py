from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, Turn, build_metric_context
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.semantic.instruction_adherence import InstructionAdherenceRuleMetric

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(transcript, critical_entities):
    return MetricContext(
        call_id="call-1",
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=transcript,
        tool_events=[],
        events=[],
        expected={"critical_entities": critical_entities},
        scenario_db={},
    )


def test_passes_on_happy_path_fixture_where_agent_reads_back_every_entity():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = InstructionAdherenceRuleMetric().compute(ctx)

    assert score.status is Status.PASS
    assert score.score == 1.0
    assert score.gating is Gating.GATE
    assert score.kind is MetricKind.DETERMINISTIC


def test_fails_when_agent_never_says_a_critical_entity():
    transcript = [
        Turn(speaker="agent", t_start=0.0, t_end=1.0, text="I've booked you with Doctor Lee."),
    ]
    ctx = _make_ctx(transcript, critical_entities=["Lee", "confirmation four eight two one three"])

    score = InstructionAdherenceRuleMetric().compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["missing_entities"] == ["confirmation four eight two one three"]


def test_entity_said_only_by_caller_does_not_count():
    transcript = [
        Turn(speaker="caller", t_start=0.0, t_end=1.0, text="I want Doctor Lee please."),
        Turn(speaker="agent", t_start=1.0, t_end=2.0, text="One moment."),
    ]
    ctx = _make_ctx(transcript, critical_entities=["Lee"])

    score = InstructionAdherenceRuleMetric().compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["missing_entities"] == ["Lee"]


def test_entity_match_is_case_insensitive():
    transcript = [Turn(speaker="agent", t_start=0.0, t_end=1.0, text="booked with dr. lee")]
    ctx = _make_ctx(transcript, critical_entities=["Lee"])

    score = InstructionAdherenceRuleMetric().compute(ctx)

    assert score.status is Status.PASS


def test_no_critical_entities_defined_passes_trivially():
    ctx = _make_ctx(transcript=[], critical_entities=[])

    score = InstructionAdherenceRuleMetric().compute(ctx)

    assert score.status is Status.PASS
    assert score.score == 1.0
