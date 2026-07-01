from eval_system.gating.gate import evaluate_gate
from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import Report, compute_metric_summary, compute_ship_verdict
from eval_system.report.markdown_report import _one_line_reason, render_markdown_report


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
    aggregate = {
        "total_calls": len(scores_by_call),
        "kind_counts": {"deterministic": 0, "judge": 0, "signal": 0},
        "error_rate": (sum(1 for s in all_scores if s.status is Status.ERROR) / len(all_scores)) if all_scores else 0.0,
        "metric_summary": compute_metric_summary(all_scores),
        **compute_ship_verdict(all_scores, trusted),
    }
    for s in all_scores:
        aggregate["kind_counts"][s.kind.value] += 1
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


# --- Gap 1: per-call metric breakdown (every MetricScore, status/score/reason) ---

def test_per_call_details_lists_every_metric_with_status_and_score():
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE, score=1.0),
            _score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY, score=0.0,
                   details={"issues": ["monotone_pitch"]}),
        ],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "task_success" in md
    assert "pitch_prosody" in md
    assert "monotone_pitch" in md


def test_gate_failure_is_highlighted_in_per_call_details():
    report = _report({
        "call-1": [_score("call-1", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE, score=0.0)],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "GATE" in md  # some explicit highlight marker for a gate-eligible failure


def test_one_line_reason_strips_newlines_from_judge_notes():
    # Real LLM judge responses often contain multi-line markdown (numbered
    # lists, etc.) -- embedding that raw in a table cell corrupts the table.
    multiline_notes = "Overall good.\n\n1. **Point one**:\n   - detail\n2. **Point two**"
    score = _score("call-1", "instruction_adherence_judge", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                    score=0.9, details={"notes": multiline_notes})

    reason = _one_line_reason(score)

    assert "\n" not in reason


def test_one_line_reason_strips_newlines_from_emotional_appropriateness_notes():
    multiline_notes = "Calm tone.\n\n1. Detail one\n2. Detail two"
    score = _score("call-1", "emotional_appropriateness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                    score=0.9, details={"notes": multiline_notes})

    reason = _one_line_reason(score)

    assert "\n" not in reason


def test_per_call_details_grouped_semantic_and_acoustic():
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE, details={"barge_ins": [], "issue_count": 0}),
        ],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "Semantic" in md
    assert "Acoustic" in md


# --- Gap 2: acoustic measured values ---

def test_acoustic_measured_values_shows_turn_taking_percentiles_in_ms():
    report = _report({
        "call-1": [_score("call-1", "turn_taking_latency", MetricKind.SIGNAL, Status.PASS, Gating.ADVISORY,
                           score=0.5, details={"gaps": [0.5], "n": 1, "p50": 0.5, "p90": 0.6, "p99": 0.65})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "500" in md  # p50 in ms
    assert "600" in md  # p90 in ms
    assert "650" in md  # p99 in ms


def test_acoustic_measured_values_shows_pitch_prosody_f0_and_rate():
    report = _report({
        "call-1": [_score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.PASS, Gating.ADVISORY, score=1.0,
                           details={"voiced_frames": 10, "pitch_mean_hz": 150.0, "pitch_std_hz": 20.0,
                                     "pitch_range_hz": 80.0, "monotone": False, "speech_rate_wps": 2.5, "issues": []})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "150" in md  # F0 mean
    assert "80" in md   # F0 range


def test_acoustic_measured_values_shows_latency_thresholds_first_token_latency():
    report = _report({
        "call-1": [_score("call-1", "latency_thresholds", MetricKind.DETERMINISTIC, Status.PASS, Gating.ADVISORY,
                           score=1.0, details={"gaps": [{"gap": 0.3}], "violations": [], "threshold_sec": 3.0,
                                                 "first_token_latency_sec": 0.3})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "300" in md  # first-token latency in ms


def test_acoustic_measured_values_shows_entity_intelligibility_per_entity_table():
    report = _report({
        "call-1": [_score("call-1", "entity_intelligibility", MetricKind.SIGNAL, Status.FAIL, Gating.GATE, score=0.0,
                           details={"missing_entities": ["Lee"], "wer": 0.2, "wer_band": "poor",
                                     "asr_engine": "faster-whisper",
                                     "critical_entity_locations": [
                                         {"entity": "Lee", "found": False},
                                         {"entity": "Tuesday", "found": True, "start": 1.0, "end": 1.5, "confidence": 0.9},
                                     ]})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "Lee" in md
    assert "Tuesday" in md


def test_acoustic_measured_values_shows_barge_in_time_to_yield_in_ms():
    report = _report({
        "call-1": [_score("call-1", "barge_in", MetricKind.SIGNAL, Status.PASS, Gating.GATE, score=1.0,
                           details={"barge_ins": [
                               {"t_onset": 4.2, "time_to_yield": 0.35, "false_yield": False, "fail_to_yield": False},
                           ], "issue_count": 0})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "350" in md  # time-to-yield in ms


# --- Gap 3: faithfulness findings ---

def test_faithfulness_findings_shown_with_ungrounded_claims():
    report = _report({
        "call-1": [_score("call-1", "faithfulness", MetricKind.JUDGE, Status.FAIL, Gating.ADVISORY, score=0.2,
                           details={"ungrounded_claims": ["agent claimed a discount not in tool results"],
                                     "rationale": "found one hallucinated claim"})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "agent claimed a discount not in tool results" in md
    assert "found one hallucinated claim" in md


def test_faithfulness_findings_shown_when_grounded():
    report = _report({
        "call-1": [_score("call-1", "faithfulness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY, score=1.0,
                           details={"ungrounded_claims": [], "rationale": "all claims grounded"})],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "all claims grounded" in md


def test_faithfulness_findings_note_when_not_run():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "did not run" in md.lower() or "not run" in md.lower()


# --- Gap 4: expanded aggregate section ---

def test_aggregate_section_shows_total_calls():
    report = _report({
        "call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)],
        "call-2": [_score("call-2", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert "Total calls" in md
    assert "2" in md.split("Aggregate")[1]


def test_aggregate_section_shows_gate_metric_pass_fail_error_counts():
    report = _report({
        "call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)],
        "call-2": [_score("call-2", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE)],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    aggregate_section = md.split("## Aggregate")[1]
    assert "task_success" in aggregate_section


def test_aggregate_section_shows_advisory_flag_rate():
    report = _report({
        "call-1": [_score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY)],
        "call-2": [_score("call-2", "pitch_prosody", MetricKind.SIGNAL, Status.PASS, Gating.ADVISORY)],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    aggregate_section = md.split("## Aggregate")[1]
    assert "pitch_prosody" in aggregate_section
    assert "50%" in aggregate_section


def test_aggregate_section_shows_judge_coverage():
    report = _report({
        "call-1": [_score("call-1", "faithfulness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY)],
        "call-2": [_score("call-2", "faithfulness", MetricKind.JUDGE, Status.ERROR, Gating.ADVISORY, score=None)],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    aggregate_section = md.split("## Aggregate")[1]
    assert "faithfulness" in aggregate_section
    assert "2" in aggregate_section  # ran on 2 calls (pass + error both count as "ran")


def test_aggregate_section_shows_deterministic_vs_judge_note():
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "faithfulness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY),
        ],
    })

    md = render_markdown_report(report, GATE_BREAKDOWN)

    aggregate_section = md.split("## Aggregate")[1]
    assert "deterministic" in aggregate_section.lower()
    assert "judge" in aggregate_section.lower()


def test_aggregate_section_placed_after_per_call_results():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    md = render_markdown_report(report, GATE_BREAKDOWN)

    assert md.index("## Per-call results") < md.index("## Aggregate")
