from eval_system.gating.gate import evaluate_gate
from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import Report, compute_metric_summary, compute_ship_verdict
from eval_system.report.html_report import _truncate, render_html_report

GATE_BREAKDOWN = [
    {"metric": "task_success", "kind": "deterministic", "default_gating": "gate",
     "rationale": "Deterministic: final tool call vs. the fixture's ground-truth success criteria."},
    {"metric": "pitch_prosody", "kind": "signal", "default_gating": "advisory",
     "rationale": "A" * 200},  # deliberately long, to test truncation
]


def _score(call_id, metric, kind, status, gating, score=1.0, details=None):
    return MetricScore(
        call_id=call_id, metric=metric, kind=kind, status=status, gating=gating, score=score,
        details=details or {},
    )


def _report(scores_by_call, trusted=frozenset(), disagreements=None) -> Report:
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


def test_truncate_short_text_unchanged():
    text, truncated = _truncate("short reason", max_len=140)

    assert text == "short reason"
    assert truncated is False


def test_truncate_long_text_gets_ellipsis():
    long_text = "x" * 200
    text, truncated = _truncate(long_text, max_len=140)

    assert truncated is True
    assert len(text) <= 141  # 140 chars + ellipsis char
    assert text.endswith("…")


def test_long_reason_gets_title_attribute_and_details_overflow():
    long_notes = "y" * 200
    report = _report({
        "call-1": [_score("call-1", "instruction_adherence_judge", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                           details={"notes": long_notes})],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert 'title="' in html
    assert "<details>" in html
    assert long_notes in html  # full text preserved somewhere (details block)


def test_title_attribute_is_capped_even_for_very_long_reasons():
    # xhtml2pdf's layout engine corrupts column boundaries on a row when a
    # title= attribute value is very long (empirically verified: ~1500 chars
    # breaks it, ~900 is safe) -- real judge rationale text easily exceeds
    # that. The tooltip must be capped independently of the full text, which
    # still lives in the trailing <details> block.
    long_notes = "z" * 2000
    report = _report({
        "call-1": [_score("call-1", "instruction_adherence_judge", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                           details={"notes": long_notes})],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    title_start = html.index('title="') + len('title="')
    title_end = html.index('"', title_start)
    assert title_end - title_start <= 500
    assert long_notes in html  # full text still preserved in the details block


def test_status_badges_have_distinct_css_classes():
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
        ],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "badge-pass" in html
    assert "badge-fail" in html


def test_gate_tag_only_appears_for_gate_eligible_failure():
    report = _report({
        "call-1": [
            _score("call-1", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE),
            _score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.FAIL, Gating.ADVISORY),
        ],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    per_call_html = html.split("Aggregate")[0]
    # one GATE tag rendered in a table cell (for tool_call_ordering); the
    # advisory pitch_prosody failure must not get one.
    assert per_call_html.count('class="gate-tag"') == 1


def test_no_unicode_symbols_in_output():
    report = _report({
        "call-1": [_score("call-1", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE)],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "⚠" not in html
    assert "▮" not in html


def test_zebra_striping_uses_inline_style_not_nth_child():
    # xhtml2pdf does not support :nth-child -- verified empirically. Alternating
    # rows must be styled inline so the PDF actually renders them.
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "faithfulness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY),
        ],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "background-color:#f7f7f7" in html


def test_metric_table_has_column_width_hints():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "<colgroup>" in html
    assert "width:45%" in html  # reason column


def test_gate_advisory_table_has_20_12_68_column_widths():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "width:20%" in html
    assert "width:12%" in html
    assert "width:68%" in html


def test_gate_advisory_rationale_truncated_with_overflow():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "A" * 200 in html  # full text preserved in a details block somewhere


def test_header_lists_every_gate_failure_not_plus_more():
    report = _report({
        "call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE)],
        "call-2": [_score("call-2", "tool_call_ordering", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE)],
        "call-3": [_score("call-3", "barge_in", MetricKind.SIGNAL, Status.FAIL, Gating.GATE)],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "more)" not in html
    assert "call-1" in html and "task_success" in html
    assert "call-2" in html and "tool_call_ordering" in html
    assert "call-3" in html and "barge_in" in html


def test_long_metric_names_get_real_break_points_in_table_cells():
    # xhtml2pdf does not honor word-break/overflow-wrap inside fixed-width
    # table cells -- verified empirically with real PyMuPDF-rendered bounding
    # boxes: an unbroken run like "emotion_appropriateness_mm" is painted as
    # a single line that overflows straight into the next column (e.g. onto
    # the Status badge), regardless of that CSS. xhtml2pdf DOES wrap on real
    # whitespace, so a real space is inserted after each underscore to give
    # its layout engine an actual break point.
    report = _report({
        "call-1": [_score("call-1", "emotion_appropriateness_mm", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY)],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "<td>emotion_appropriateness_mm</td>" not in html  # unbroken form must not appear in a table cell
    assert "<td>emotion_ appropriateness_ mm</td>" in html


def test_aggregate_tables_have_explicit_column_widths():
    report = _report({
        "call-1": [
            _score("call-1", "emotion_appropriateness_mm", MetricKind.JUDGE, Status.ERROR, Gating.ADVISORY, score=None),
        ],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    aggregate_section = html.split("Aggregate")[1].split("Gate vs. advisory")[0]
    assert aggregate_section.count("<colgroup>") == 3  # gate metrics, advisory metrics, judge coverage tables


def test_css_includes_page_break_avoid():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "page-break-inside" in html


def test_html_document_is_well_formed():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert html.strip().startswith("<!DOCTYPE html>") or html.strip().startswith("<html")
    assert "<style>" in html
    assert "<body>" in html
    assert "</html>" in html


def test_ship_banner_when_all_pass():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "banner-ship" in html
    assert ">SHIP<" in html


def test_hold_banner_when_gate_failure():
    report = _report({"call-1": [_score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE)]})

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "banner-hold" in html
    assert ">HOLD<" in html


def test_acoustic_measured_values_heading_removed():
    report = _report({
        "call-1": [_score("call-1", "pitch_prosody", MetricKind.SIGNAL, Status.PASS, Gating.ADVISORY,
                           details={"pitch_mean_hz": 150.0, "pitch_range_hz": 80.0, "issues": []})],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "Acoustic measured values" not in html


def test_faithfulness_findings_heading_removed():
    report = _report({
        "call-1": [_score("call-1", "faithfulness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                           details={"ungrounded_claims": [], "rationale": "all claims grounded"})],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "Faithfulness judge findings" not in html


def test_emotional_appropriateness_dropped_from_per_call_table():
    report = _report({
        "call-1": [
            _score("call-1", "task_success", MetricKind.DETERMINISTIC, Status.PASS, Gating.GATE),
            _score("call-1", "emotional_appropriateness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                   details={"notes": "calm and appropriate"}),
        ],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    per_call_section = html.split("Headline metric")[0]
    assert "emotional_ appropriateness" not in per_call_section  # breakable form of the metric name


def test_emotional_appropriateness_still_in_aggregate():
    report = _report({
        "call-1": [_score("call-1", "emotional_appropriateness", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY)],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    aggregate_section = html.split("Aggregate")[1]
    assert "emotional_ appropriateness" in aggregate_section


def test_html_escapes_special_characters_in_reason_text():
    report = _report({
        "call-1": [_score("call-1", "instruction_adherence_judge", MetricKind.JUDGE, Status.PASS, Gating.ADVISORY,
                           details={"notes": "uses <script>alert(1)</script> & other chars"})],
    })

    html = render_html_report(report, GATE_BREAKDOWN)

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
