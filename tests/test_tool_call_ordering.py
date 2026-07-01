from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, ToolEvent, build_metric_context
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.semantic.tool_call_ordering import ToolCallOrderingMetric

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(tool_events, expected, scenario_db=None):
    return MetricContext(
        call_id="call-1",
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=[],
        tool_events=tool_events,
        events=[],
        expected=expected,
        scenario_db=scenario_db or {},
    )


def test_passes_on_happy_path_fixture():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = ToolCallOrderingMetric().compute(ctx)

    assert score.status is Status.PASS
    assert score.score == 1.0
    assert score.gating is Gating.GATE
    assert score.kind is MetricKind.DETERMINISTIC


def test_reschedule_trap_fixture_fails_via_zero_appointment_invariant():
    fixture = load_fixture(FIXTURES_DIR / "reschedule_trap")
    ctx = build_metric_context(fixture)

    score = ToolCallOrderingMetric().compute(ctx)

    assert score.status is Status.FAIL
    violations = score.details["invariants"]["never_zero_appointments_violations"]
    assert len(violations) == 1
    assert violations[0]["patient_id"] == "P100"


def test_out_of_order_tool_calls_are_flagged_even_with_no_invariants():
    tool_events = [
        ToolEvent(name="book_appointment", args={}, result={"status": "booked"}, t=1.0),
        ToolEvent(name="check_availability", args={}, result={"available": True}, t=2.0),
    ]
    expected = {
        "tool_sequence": [
            {"name": "check_availability", "required_args": {}},
            {"name": "book_appointment", "required_args": {}},
        ],
        "invariants": [],
    }
    ctx = _make_ctx(tool_events, expected)

    score = ToolCallOrderingMetric().compute(ctx)

    assert score.status is Status.FAIL
    missing = score.details["sequence_order"]["missing_or_out_of_order"]
    assert any(step["name"] == "book_appointment" for step in missing)


def test_required_args_mismatch_is_flagged():
    tool_events = [
        ToolEvent(name="book_appointment", args={"provider": "Smith"}, result={"status": "booked"}, t=1.0),
    ]
    expected = {
        "tool_sequence": [{"name": "book_appointment", "required_args": {"provider": "Lee"}}],
        "invariants": [],
    }
    ctx = _make_ctx(tool_events, expected)

    score = ToolCallOrderingMetric().compute(ctx)

    assert score.status is Status.FAIL


def test_invariant_violation_detected_independent_of_sequence_check():
    tool_events = [
        ToolEvent(name="cancel_appointment", args={"appointment_id": "A1"}, result={"status": "cancelled"}, t=1.0),
        ToolEvent(name="book_appointment", args={"patient_id": "P1"}, result={"status": "booked"}, t=2.0),
    ]
    expected = {"tool_sequence": [], "invariants": ["never_zero_appointments"]}
    scenario_db = {"patients": {"P1": {"appointments": ["A1"]}}}
    ctx = _make_ctx(tool_events, expected, scenario_db)

    score = ToolCallOrderingMetric().compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["sequence_order"]["missing_or_out_of_order"] == []
    assert len(score.details["invariants"]["never_zero_appointments_violations"]) == 1
