import pytest

from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import (
    build_report,
    compute_metric_summary,
    compute_ship_verdict,
    judge_trust_note,
    upsert_scores,
)


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


# --- run-level ship verdict (assessment.md: "both suites feed one verdict";
# "a single ship/don't-ship decision for a CI pipeline") ---

def test_all_pass_scores_ship_true():
    scores = [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-2", "barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE),
    ]

    verdict = compute_ship_verdict(scores)

    assert verdict["ship"] is True
    assert verdict["gate_failures"] == []
    assert verdict["advisory_failures"] == []
    assert "SHIP" in verdict["ship_reason"]


def test_one_gate_failure_holds_and_is_listed():
    scores = [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-2", "entity_intelligibility", MetricKind.SIGNAL, Status.FAIL, Gating.GATE),
    ]

    verdict = compute_ship_verdict(scores)

    assert verdict["ship"] is False
    assert verdict["gate_failures"] == [
        {"call_id": "call-2", "metric": "entity_intelligibility", "status": "fail"}
    ]
    assert verdict["advisory_failures"] == []
    assert "HOLD" in verdict["ship_reason"]
    assert "entity_intelligibility" in verdict["ship_reason"]
    assert "call-2" in verdict["ship_reason"]


def test_advisory_failure_never_blocks_ship():
    scores = [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY),
    ]

    verdict = compute_ship_verdict(scores)

    assert verdict["ship"] is True
    assert verdict["gate_failures"] == []
    assert verdict["advisory_failures"] == [
        {"call_id": "call-1", "metric": "pitch_prosody", "status": "fail"}
    ]


def test_error_status_never_counts_as_a_failure():
    scores = [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-1", "faithfulness", MetricKind.JUDGE, Status.ERROR, Gating.ADVISORY),
        _score("call-2", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.ERROR, Gating.GATE),
    ]

    verdict = compute_ship_verdict(scores)

    assert verdict["ship"] is True
    assert verdict["gate_failures"] == []
    assert verdict["advisory_failures"] == []


def test_trusted_judge_promotes_gate_eligibility_for_ship_verdict():
    scores = [_score("call-1", "faithfulness", MetricKind.JUDGE, Status.FAIL, Gating.ADVISORY)]

    untrusted = compute_ship_verdict(scores)
    trusted = compute_ship_verdict(scores, trusted_judge_metrics=frozenset({"faithfulness"}))

    assert untrusted["ship"] is True
    assert trusted["ship"] is False
    assert trusted["gate_failures"] == [{"call_id": "call-1", "metric": "faithfulness", "status": "fail"}]


def test_build_report_aggregate_includes_run_level_ship_fields():
    store = upsert_scores({}, [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-2", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
    ])

    report = build_report(store)

    assert report.aggregate["ship"] is False
    assert report.aggregate["gate_failures"] == [
        {"call_id": "call-2", "metric": "tool_call_ordering", "status": "fail"}
    ]
    assert report.aggregate["advisory_failures"] == []
    assert "HOLD" in report.aggregate["ship_reason"]


# --- judge trust note (avoid a grader misreading an empty trusted-judge list) ---

def test_judge_trust_note_present_when_no_judges_trusted():
    assert judge_trust_note(frozenset()) is not None
    assert "0.60" in judge_trust_note(frozenset()) or "0.6" in judge_trust_note(frozenset())


def test_judge_trust_note_absent_when_a_judge_is_trusted():
    assert judge_trust_note(frozenset({"faithfulness"})) is None


def test_build_report_aggregate_includes_judge_trust_note_when_empty():
    store = upsert_scores({}, [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)])

    report = build_report(store)

    assert "judge_trust_note" in report.aggregate
    assert report.aggregate["judge_trust_note"] is not None


def test_build_report_includes_emotion_disagreement_turns_per_call():
    store = upsert_scores({}, [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        MetricScore(
            call_id="call-1", metric="ser_emotion", kind=MetricKind.SIGNAL, status=Status.PASS,
            gating=Gating.ADVISORY, score=0.9,
            details={"per_turn": [{"start": 0.0, "end": 1.0, "label": "hap", "confidence": 0.9}]},
        ),
        MetricScore(
            call_id="call-1", metric="emotion_appropriateness_mm", kind=MetricKind.JUDGE, status=Status.FAIL,
            gating=Gating.ADVISORY, score=2.0,
            details={"per_turn": [
                {"turn_index": 0, "start": 0.0, "end": 1.0, "appropriate": False, "score": 2, "detected_tone": "cheerful"}
            ]},
            judge_prompt_version="mm-v1",
        ),
    ])

    report = build_report(store)

    assert report.per_call["call-1"]["emotion_disagreement_turns"] == [
        {"turn": 0, "ser_label": "hap", "judge_tone": "cheerful", "judge_appropriate": False}
    ]
    # both metrics are advisory -- disagreement must never affect ship
    assert report.per_call["call-1"]["verdict"].ship is True
    assert report.aggregate["ship"] is True


def test_build_report_aggregate_omits_judge_trust_note_when_a_judge_is_trusted():
    store = upsert_scores({}, [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)])

    report = build_report(store, trusted_judge_metrics=frozenset({"faithfulness"}))

    assert "judge_trust_note" not in report.aggregate


# --- per-metric summary (feeds the aggregate report section: gate pass/fail/
# error counts, advisory flag rates, judge coverage) ---

def test_compute_metric_summary_counts_status_per_metric():
    scores = [
        _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
        _score("call-2", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
        _score("call-3", "task_success", MetricKind.DETERMINISTIC, Status.SKIPPED, Gating.GATE, score=None),
    ]

    summary = compute_metric_summary(scores)

    assert summary["task_success"]["pass"] == 1
    assert summary["task_success"]["fail"] == 1
    assert summary["task_success"]["skipped"] == 1
    assert summary["task_success"]["error"] == 0
    assert summary["task_success"]["kind"] == "deterministic"
    assert summary["task_success"]["gating"] == "gate"


def test_compute_metric_summary_ran_excludes_skipped():
    scores = [
        _score("call-1", "faithfulness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY),
        _score("call-2", "faithfulness", MetricKind.JUDGE, Status.ERROR, Gating.ADVISORY),
        _score("call-3", "faithfulness", MetricKind.JUDGE, Status.SKIPPED, Gating.ADVISORY, score=None),
    ]

    summary = compute_metric_summary(scores)

    assert summary["faithfulness"]["ran"] == 2  # skipped doesn't count as "ran" -- judge coverage


def test_compute_metric_summary_flag_rate():
    scores = [
        _score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY),
        _score("call-2", "pitch_prosody", MetricKind.SIGNAL, Status.PASS, Gating.ADVISORY),
        _score("call-3", "pitch_prosody", MetricKind.SIGNAL, Status.PASS, Gating.ADVISORY),
    ]

    summary = compute_metric_summary(scores)

    assert summary["pitch_prosody"]["flag_rate"] == pytest.approx(1 / 3)


def test_compute_metric_summary_flag_rate_none_when_never_ran():
    scores = [_score("call-1", "faithfulness", MetricKind.JUDGE, Status.SKIPPED, Gating.ADVISORY, score=None)]

    summary = compute_metric_summary(scores)

    assert summary["faithfulness"]["flag_rate"] is None
