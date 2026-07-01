"""Known-answer tests for the gate/ship invariants (assessment.md rubric
lines 75-78: "evaluating the evaluators"). A flaky judge or a noisy advisory
metric must never fail a good deploy, and an evaluator crash (ERROR) must
never be conflated with an actual behavioral FAIL."""
from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import compute_ship_verdict


def _score(metric, kind, status, gating):
    return MetricScore(call_id="call-1", metric=metric, kind=kind, status=status, gating=gating, score=None)


def test_advisory_failure_does_not_block_ship():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("tool_call_ordering", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE),
        _score("pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY),  # a noisy advisory metric fails
    ]

    verdict = compute_ship_verdict(scores)

    assert verdict["ship"] is True
    assert verdict["gate_failures"] == []


def test_error_status_is_not_failure():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("faithfulness", MetricKind.JUDGE, Status.ERROR, Gating.ADVISORY),  # judge call broke
        _score("entity_intelligibility", MetricKind.SIGNAL, Status.ERROR, Gating.GATE),  # even a gate metric erroring
    ]

    verdict = compute_ship_verdict(scores)

    assert verdict["ship"] is True  # ERROR never flips ship
    assert verdict["gate_failures"] == []
    assert verdict["advisory_failures"] == []
    # ERROR is visible via aggregate.error_rate (computed in build_report), not here --
    # this function only ever sees FAIL/PASS/SKIPPED/ERROR status and deliberately
    # ignores ERROR for ship purposes (CLAUDE.md: "ERROR != FAIL").
