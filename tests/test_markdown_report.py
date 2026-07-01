from eval_system.gating.gate import evaluate_gate
from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import Report, compute_ship_verdict
from eval_system.report.markdown_report import render_markdown_report


def _score(call_id, metric, kind, status, gating, score=1.0, details=None):
    return MetricScore(
        call_id=call_id, metric=metric, kind=kind, status=status, gating=gating, score=score,
        details=details or {},
    )


def _report(scores_by_call: dict[str, list[MetricScore]], trusted=frozenset(), disagreements=None) -> Report:
    disagreements = disagreements or {}
    per_call = {
        call_id: {
            "scores": scores,
            "verdict": evaluate_gate(scores, trusted),
            "emotion_disagreement_turns": disagreements.get(call_id, []),
        }
        for call_id, scores in scores_by_call.items()
    }
    all_scores = [s for scores in scores_by_call.values() for s in scores]
    aggregate = {"total_calls": len(scores_by_call), **compute_ship_verdict(all_scores, trusted)}
    return Report(per_call=per_call, aggregate=aggregate)


GATE_BREAKDOWN = [
    {"metric": "task_success", "kind": "deterministic", "default_gating": "gate", "rationale": "ground-truthed."},
    {"metric": "pitch_prosody", "kind": "signal", "default_gating": "advisory", "rationale": "a perceptual proxy."},
]


def test_top_line_shows_ship_when_aggregate_ship_true():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert md.splitlines()[0].strip() in ("# SHIP", "SHIP")
    assert "SHIP" in md.splitlines()[0]


def test_top_line_shows_hold_and_reason_when_ship_false():
    report = _report({
        "call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE)],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "HOLD" in md
    assert report.aggregate["ship_reason"] in md


def test_per_call_table_includes_call_id_and_barge_in_headline_status():
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE, details={"issue_count": 0}),
        ],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "call-1" in md
    assert "pass" in md.lower()


def test_gate_advisory_section_lists_metric_gating_and_rationale():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "task_success" in md
    assert "ground-truthed." in md
    assert "pitch_prosody" in md
    assert "a perceptual proxy." in md


def test_headline_barge_in_section_present():
    report = _report({
        "call-1": [_score("call-1", "barge_in", MetricKind.SIGNAL, Status.FAIL, Gating.GATE, details={"issue_count": 2})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "barge_in" in md.lower()
    assert "headline" in md.lower()


def test_call_with_no_barge_in_score_shows_not_available():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "n/a" in md.lower() or "not run" in md.lower()


def test_emotion_advisory_section_lists_disagreement_turns():
    report = _report(
        {"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]},
        disagreements={"call-1": [{"turn": 0, "ser_label": "hap", "judge_tone": "cheerful", "judge_appropriate": False}]},
    )

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "Emotion (advisory)" in md
    assert "call-1" in md.split("Emotion (advisory)")[1]
    assert "hap" in md
    assert "cheerful" in md


def test_emotion_advisory_section_notes_when_no_disagreements():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "Emotion (advisory)" in md
    section = md.split("Emotion (advisory)")[1]
    assert "no disagreement" in section.lower()
