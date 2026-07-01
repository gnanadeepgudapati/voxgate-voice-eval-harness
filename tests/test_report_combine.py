from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import build_report, upsert_scores


def _score(call_id, metric, kind, status, gating, evaluator_version="1", judge_prompt_version=None, score=1.0):
    return MetricScore(
        call_id=call_id, metric=metric, kind=kind, status=status, gating=gating, score=score,
        evaluator_version=evaluator_version, judge_prompt_version=judge_prompt_version,
    )


def test_upsert_overwrites_same_key_not_duplicates():
    store = {}
    first = _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE, score=1.0)
    store = upsert_scores(store, [first])

    rerun = _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE, score=0.0)
    store = upsert_scores(store, [rerun])

    assert len(store) == 1
    assert store[rerun.key].status is Status.FAIL


def test_build_report_groups_by_call_and_computes_verdict():
    store = {}
    store = upsert_scores(store, [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-1", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
        _score("call-2", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
    ])

    report = build_report(store)

    assert set(report.per_call.keys()) == {"call-1", "call-2"}
    assert report.per_call["call-1"]["verdict"].ship is False
    assert report.per_call["call-2"]["verdict"].ship is True


def test_aggregate_counts_ships_and_holds():
    store = upsert_scores({}, [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-2", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
    ])

    report = build_report(store)

    assert report.aggregate["total_calls"] == 2
    assert report.aggregate["ships"] == 1
    assert report.aggregate["holds"] == 1


def test_aggregate_kind_counts_and_error_rate():
    store = upsert_scores({}, [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-1", "faithfulness", MetricKind.JUDGE, Status.ERROR, Gating.ADVISORY),
        _score("call-1", "barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE),
    ])

    report = build_report(store)

    assert report.aggregate["kind_counts"] == {"deterministic": 1, "judge": 1, "signal": 1}
    assert report.aggregate["error_rate"] == 1 / 3


def test_trusted_judge_metrics_recorded_in_aggregate_and_affect_verdict():
    store = upsert_scores({}, [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-1", "faithfulness", MetricKind.JUDGE, Status.FAIL, Gating.ADVISORY),
    ])

    untrusted = build_report(store)
    trusted = build_report(store, trusted_judge_metrics=frozenset({"faithfulness"}))

    assert untrusted.per_call["call-1"]["verdict"].ship is True
    assert trusted.per_call["call-1"]["verdict"].ship is False
    assert trusted.aggregate["trusted_judge_metrics"] == ["faithfulness"]
