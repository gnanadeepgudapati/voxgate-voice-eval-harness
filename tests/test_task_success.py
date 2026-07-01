from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, ToolEvent, build_metric_context
from eval_system.metrics.base import Gating, MetricKind, Status
from eval_system.metrics.semantic.task_success import TaskSuccessMetric

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def _make_ctx(tool_events, expected):
    return MetricContext(
        call_id="call-1",
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=[],
        tool_events=tool_events,
        events=[],
        expected=expected,
        scenario_db={},
    )


def test_passes_when_final_tool_and_result_match_real_fixture():
    fixture = load_fixture(FIXTURES_DIR / "happy_path_book")
    ctx = build_metric_context(fixture)

    score = TaskSuccessMetric().compute(ctx)

    assert score.status is Status.PASS
    assert score.score == 1.0
    assert score.gating is Gating.GATE
    assert score.kind is MetricKind.DETERMINISTIC


def test_fails_when_last_tool_call_is_not_the_expected_final_tool():
    tool_events = [
        ToolEvent(name="check_availability", args={}, result={"available": True}, t=1.0),
    ]
    expected = {
        "success_criteria": {
            "final_tool": "book_appointment",
            "result_contains": {"status": "booked"},
        }
    }
    ctx = _make_ctx(tool_events, expected)

    score = TaskSuccessMetric().compute(ctx)

    assert score.status is Status.FAIL
    assert score.score == 0.0


def test_fails_when_final_tool_result_missing_expected_fields():
    tool_events = [
        ToolEvent(name="book_appointment", args={}, result={"status": "error"}, t=1.0),
    ]
    expected = {
        "success_criteria": {
            "final_tool": "book_appointment",
            "result_contains": {"status": "booked", "confirmation": "48213"},
        }
    }
    ctx = _make_ctx(tool_events, expected)

    score = TaskSuccessMetric().compute(ctx)

    assert score.status is Status.FAIL
    assert score.details["mismatched_result_fields"] == {"status": "booked", "confirmation": "48213"}


def test_skipped_when_fixture_defines_no_success_criteria():
    fixture = load_fixture(FIXTURES_DIR / "reschedule_trap")
    ctx = build_metric_context(fixture)

    score = TaskSuccessMetric().compute(ctx)

    assert score.status is Status.SKIPPED
    assert score.score is None
