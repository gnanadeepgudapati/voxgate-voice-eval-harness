"""Known-answer correctness tests for tool_call_ordering's reschedule-trap
invariant (assessment.md line 41: "the new slot must be secured before the
old one is released, and a mid-sequence failure must not leave the caller
with zero appointments"). A metric merely existing in the registry doesn't
prove it returns the RIGHT answer for the one scenario the whole assignment
is built around -- these feed a synthetic tool-call record with a KNOWN
ground-truth outcome and assert the exact result, per assessment.md rubric
lines 75-78 ("evaluating the evaluators"). Does not depend on fixtures/."""
from eval_system.context.metric_context import MetricContext, ToolEvent
from eval_system.metrics.base import Status
from eval_system.metrics.semantic.tool_call_ordering import ToolCallOrderingMetric

SCENARIO_DB = {"patients": {"P1": {"appointments": ["A1"]}}}


def _ctx(tool_events):
    return MetricContext(
        call_id="known-answer-reschedule",
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=[],
        tool_events=tool_events,
        events=[],
        expected={"tool_sequence": [], "invariants": ["never_zero_appointments"]},
        scenario_db=SCENARIO_DB,
    )


def test_reschedule_trap_fails_when_old_released_first():
    # cancel (old) succeeds BEFORE book (new) succeeds -> a real zero-appointment window.
    tool_events = [
        ToolEvent(name="cancel_appointment", args={"appointment_id": "A1"}, result={"status": "cancelled"}, t=1.0),
        ToolEvent(name="book_appointment", args={"patient_id": "P1"}, result={"status": "booked"}, t=2.0),
    ]

    score = ToolCallOrderingMetric().compute(_ctx(tool_events))

    assert score.status is Status.FAIL
    violations = score.details["invariants"]["never_zero_appointments_violations"]
    assert len(violations) == 1
    assert violations[0]["patient_id"] == "P1"
    assert violations[0]["tool_event_t"] == 1.0  # flagged at the moment the count hit zero


def test_reschedule_trap_passes_when_new_secured_first():
    # book (new) succeeds BEFORE cancel (old) -- count goes 1 -> 2 -> 1, never zero.
    tool_events = [
        ToolEvent(name="book_appointment", args={"patient_id": "P1"}, result={"status": "booked"}, t=1.0),
        ToolEvent(name="cancel_appointment", args={"appointment_id": "A1"}, result={"status": "cancelled"}, t=2.0),
    ]

    score = ToolCallOrderingMetric().compute(_ctx(tool_events))

    assert score.status is Status.PASS
    assert score.details["invariants"]["never_zero_appointments_violations"] == []


def test_reschedule_midsequence_failure_not_left_empty():
    # old released, then the new-slot booking FAILS -- caller is left with
    # zero appointments and no successful recovery. Must fail.
    tool_events = [
        ToolEvent(name="cancel_appointment", args={"appointment_id": "A1"}, result={"status": "cancelled"}, t=1.0),
        ToolEvent(
            name="book_appointment", args={"patient_id": "P1"},
            result={"status": "error", "reason": "slot_unavailable"}, t=2.0,
        ),
    ]

    score = ToolCallOrderingMetric().compute(_ctx(tool_events))

    assert score.status is Status.FAIL
    violations = score.details["invariants"]["never_zero_appointments_violations"]
    assert len(violations) == 1  # the failed book never recovers the count
