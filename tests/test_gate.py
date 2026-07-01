import pytest

from eval_system.gating.gate import evaluate_gate
from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status


def _score(metric, kind, status, gating, score=1.0):
    return MetricScore(
        call_id="call-1", metric=metric, kind=kind, status=status, gating=gating, score=score,
    )


def test_ships_when_all_gate_metrics_pass():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE),
        _score("pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY),
    ]

    verdict = evaluate_gate(scores)

    assert verdict.ship is True
    assert verdict.failures == []


def test_holds_when_a_gate_metric_fails():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
    ]

    verdict = evaluate_gate(scores)

    assert verdict.ship is False
    assert verdict.failures == [{"metric": "tool_call_ordering", "status": "fail"}]


def test_advisory_fail_never_blocks_ship():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("emotional_appropriateness", MetricKind.JUDGE, Status.FAIL, Gating.ADVISORY),
    ]

    verdict = evaluate_gate(scores)

    assert verdict.ship is True


def test_judge_only_gates_once_trusted():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("faithfulness", MetricKind.JUDGE, Status.FAIL, Gating.ADVISORY),
    ]

    untrusted_verdict = evaluate_gate(scores)
    trusted_verdict = evaluate_gate(scores, trusted_judge_metrics=frozenset({"faithfulness"}))

    assert untrusted_verdict.ship is True
    assert trusted_verdict.ship is False
    assert trusted_verdict.failures == [{"metric": "faithfulness", "status": "fail"}]


def test_error_on_gate_metric_holds_fail_closed():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("tool_call_ordering", MetricKind.DETERMINISTIC, Status.ERROR, Gating.GATE),
    ]

    verdict = evaluate_gate(scores)

    assert verdict.ship is False
    assert verdict.failures == [{"metric": "tool_call_ordering", "status": "error"}]


def test_skipped_gate_metric_excluded_from_conjunction():
    scores = [
        _score("task_success", MetricKind.DETERMINISTIC, Status.SKIPPED, Gating.GATE, score=None),
        _score("tool_call_ordering", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
    ]

    verdict = evaluate_gate(scores)

    assert verdict.ship is True


def test_raises_on_empty_scores():
    with pytest.raises(ValueError):
        evaluate_gate([])
