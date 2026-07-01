"""Human-readable combined report, HTML (and PDF-via-HTML) rendering.
Separate from markdown_report.py (which still produces report.md) because
HTML needs real layout control -- fixed column widths, tooltips, badges --
that Markdown can't express, and because xhtml2pdf (used to turn this HTML
into report.pdf) has a limited CSS engine verified empirically before
writing this:

- `background-color` on inline `style=` attributes: works.
- Plain-text colored badges: render cleanly. Unicode symbols (⚠, block-
  drawing characters) do NOT -- they showed up as garbled boxes in the PDF,
  which is why every badge here is plain text in a colored <span>, never an
  emoji or Unicode glyph.
- `border-radius`: doesn't round corners in the PDF (renders as a plain
  rectangle) -- harmless, not load-bearing, left in the CSS since it still
  looks right when report.html is opened directly in a browser.
- `tr:nth-child(even)` zebra striping: does NOT render in the PDF at all.
  Every other row's background is therefore computed and applied as an
  INLINE style at generation time instead of relying on the CSS selector.
- `<colgroup><col style="width:...">`: respected, used for both the
  per-call metric tables and the gate-vs-advisory table's column widths.
- `vertical-align:top`, `border-bottom` row separators: both render fine.
- Long underscore-joined metric names (e.g. `emotion_appropriateness_mm`) have
  no natural break point and otherwise overflow past their column into the
  next cell -- `word-break: break-all` on `td`/`th` forces a wrap.

Long free-text (judge rationale/notes) that would otherwise blow out a
narrow table cell is truncated to ~80 chars at a word boundary with a
`title=` tooltip holding the full text, and every truncated value is also
listed in full in a trailing `<details>` block below its table -- both
mechanisms, not just one, since a PDF reader can't hover for a tooltip the
way a browser can.

Deliberately terse (mirrors markdown_report.py's structure exactly, same
reasoning): no standalone "Acoustic measured values" or "Faithfulness judge
findings" sections -- both folded into the per-call metrics table's Reason
column via the shared `_one_line_reason()`. `emotional_appropriateness` is
dropped from the per-call table (still in Aggregate) since it duplicates
the two-proxy emotion signal already tracked per call."""
from __future__ import annotations

import html as _html
from typing import Any

from eval_system.metrics.base import Gating, MetricScore, Status
from eval_system.report.combine import Report, compute_ship_verdict
from eval_system.report.markdown_report import (
    ACOUSTIC_METRICS,
    HEADLINE_METRIC,
    PER_CALL_TABLE_EXCLUDE,
    SEMANTIC_METRICS,
    _one_line_reason,
)

REASON_MAX_LEN = 80
# The gate-vs-advisory table's Rationale column is 68% wide (vs. 45% for the
# per-call metrics table) and, since gate.py's GATE_RATIONALE strings are
# already trimmed to one crisp sentence, most fit without truncating at all
# at this wider cap -- the 80-char cap is specifically for judge free-text
# in the per-call table, not this column.
GATE_RATIONALE_MAX_LEN = 140
# xhtml2pdf's layout engine corrupts a row's column boundaries when a title=
# attribute value is very long (verified empirically: ~1500 chars breaks it,
# ~900 is safe) -- capped well under that; the full text still lives in the
# trailing <details> block, so nothing is lost.
TITLE_MAX_LEN = 500
ZEBRA_STYLE = "background-color:#f7f7f7"

STATUS_BADGE_CLASS = {
    "pass": "badge-pass",
    "fail": "badge-fail",
    "error": "badge-error",
    "skipped": "badge-skipped",
}

CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #222; }
h1, h2, h3, h4 { margin: 12px 0; }
p, ul { margin: 12px 0; }
.banner { display: inline-block; padding: 8px 18px; border-radius: 6px; font-weight: bold;
          font-size: 15pt; margin: 12px 0; }
.banner-ship { background-color: #e6f4ea; color: #1e7e34; border: 1px solid #1e7e34; }
.banner-hold { background-color: #fdecea; color: #b52424; border: 1px solid #b52424; }
table { border-collapse: collapse; width: 100%; table-layout: fixed; margin: 12px 0; }
th, td { text-align: left; padding: 6px 8px; vertical-align: top; border-bottom: 1px solid #ddd;
         word-wrap: break-word; overflow-wrap: break-word; word-break: break-all; }
th { border-bottom: 2px solid #999; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; color: #ffffff;
         font-size: 9pt; font-weight: bold; }
.badge-pass { background-color: #2e7d32; }
.badge-fail { background-color: #c62828; }
.badge-error { background-color: #ef6c00; }
.badge-skipped { background-color: #757575; }
.gate-tag { display: inline-block; padding: 1px 6px; margin-left: 4px; border: 1px solid #c62828;
            color: #c62828; border-radius: 3px; font-size: 8pt; font-weight: bold; }
details { margin: 6px 0 12px 0; }
summary { cursor: pointer; color: #555; font-size: 9pt; }
tr { page-break-inside: avoid; }
@media print {
  tr { page-break-inside: avoid; }
}
"""


def _esc(value: Any) -> str:
    return _html.escape(str(value))


def _breakable_metric_name(metric: str) -> str:
    # xhtml2pdf does not honor word-break/overflow-wrap inside fixed-width
    # table cells -- verified empirically with PyMuPDF-rendered bounding
    # boxes: an unbroken run like "emotion_appropriateness_mm" is painted as
    # a single line that overflows straight into the next column (onto the
    # Status badge) regardless of that CSS. xhtml2pdf DOES wrap on real
    # whitespace, so a real space is inserted after each underscore to give
    # its layout engine an actual break point.
    return _esc(metric).replace("_", "_ ")


def _truncate(text: str, max_len: int = REASON_MAX_LEN) -> tuple[str, bool]:
    if len(text) <= max_len:
        return text, False
    cut = text[:max_len]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…", True


def _fmt_score(score: float | None) -> str:
    return f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"


def _overflow_details(items: list[tuple[str, str]], label: str) -> str:
    if not items:
        return ""
    body = "".join(f"<p><strong>{_esc(name)}</strong>: {_esc(text)}</p>" for name, text in items)
    return f"<details><summary>{_esc(label)}</summary>{body}</details>"


def _status_cell_html(score: MetricScore) -> str:
    css_class = STATUS_BADGE_CLASS[score.status.value]
    cell = f'<span class="badge {css_class}">{score.status.value.upper()}</span>'
    if score.status in (Status.FAIL, Status.ERROR) and score.gating is Gating.GATE:
        cell += ' <span class="gate-tag">GATE</span>'
    return cell


def _metrics_table_html(scores: list[MetricScore]) -> str:
    overflow: list[tuple[str, str]] = []
    rows = []
    for i, s in enumerate(sorted(scores, key=lambda s: s.metric)):
        # Ask for the full (whitespace-collapsed, never raw-multiline) text --
        # markdown_report's own 160-char cap would otherwise clip it before
        # this module's truncate-with-tooltip/details treatment ever sees it.
        reason = _one_line_reason(s, max_len=100_000)
        shown, truncated = _truncate(reason)
        title_attr = f' title="{_esc(reason[:TITLE_MAX_LEN])}"' if truncated else ""
        if truncated:
            overflow.append((s.metric, reason))
        row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
        rows.append(
            f"<tr{row_style}><td>{_breakable_metric_name(s.metric)}</td><td>{_status_cell_html(s)}</td>"
            f"<td>{_fmt_score(s.score)}</td><td{title_attr}>{_esc(shown)}</td></tr>"
        )

    table = (
        '<table class="metric-table">'
        '<colgroup><col style="width:20%"><col style="width:15%">'
        '<col style="width:10%"><col style="width:45%"></colgroup>'
        "<tr><th>Metric</th><th>Status</th><th>Score</th><th>Reason</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return table + _overflow_details(overflow, "Full reason text (truncated above)")


def _per_call_details_html(scores: list[MetricScore]) -> str:
    parts: list[str] = []
    scores = [s for s in scores if s.metric not in PER_CALL_TABLE_EXCLUDE]
    semantic = [s for s in scores if s.metric in SEMANTIC_METRICS]
    acoustic = [s for s in scores if s.metric in ACOUSTIC_METRICS]
    other = [s for s in scores if s.metric not in SEMANTIC_METRICS and s.metric not in ACOUSTIC_METRICS]

    if semantic:
        parts.append("<h4>Semantic metrics</h4>")
        parts.append(_metrics_table_html(semantic))
    if acoustic:
        parts.append("<h4>Acoustic metrics</h4>")
        parts.append(_metrics_table_html(acoustic))
    if other:
        parts.append("<h4>Other metrics</h4>")
        parts.append(_metrics_table_html(other))
    return "".join(parts)


def _header_html(agg: dict[str, Any]) -> str:
    ship = agg["ship"]
    banner_class = "banner-ship" if ship else "banner-hold"
    banner_text = "SHIP" if ship else "HOLD"
    parts = [f'<div class="banner {banner_class}">{banner_text}</div>']
    if ship:
        parts.append(f"<p>{_esc(agg['ship_reason'])}</p>")
    else:
        parts.append(f"<p>{len(agg['gate_failures'])} gate failure(s):</p><ul>")
        for f in agg["gate_failures"]:
            parts.append(f"<li>{_esc(f['call_id'])} — {_esc(f['metric'])} ({_esc(f['status'])})</li>")
        parts.append("</ul>")
    return "".join(parts)


def _emotion_section_html(report: Report) -> str:
    parts = [
        "<h2>Emotion (advisory)</h2>",
        "<p>Two permanently-advisory proxies — objective offline SER (<code>ser_emotion</code>) vs. a "
        "contextual multimodal judge (<code>emotion_appropriateness_mm</code>) — cross-checked per agent "
        "turn. Disagreement is the trust signal here (assessment.md Part 1 Q3): neither metric can gate, "
        "but a divergence flags the turn for human review.</p>",
    ]
    any_disagreements = any(report.per_call[c]["emotion_disagreement_turns"] for c in report.per_call)
    if not any_disagreements:
        parts.append("<p>No disagreement turns across any call.</p>")
    else:
        parts.append(
            '<table><colgroup><col style="width:20%"><col style="width:10%"><col style="width:20%">'
            '<col style="width:30%"><col style="width:20%"></colgroup>'
            "<tr><th>Call</th><th>Turn</th><th>SER label</th><th>Judge tone</th><th>Judge appropriate</th></tr>"
        )
        i = 0
        for call_id in sorted(report.per_call):
            for d in report.per_call[call_id]["emotion_disagreement_turns"]:
                row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
                parts.append(
                    f"<tr{row_style}><td>{_esc(call_id)}</td><td>{d['turn']}</td>"
                    f"<td>{_esc(d['ser_label'])}</td><td>{_esc(d['judge_tone'])}</td>"
                    f"<td>{d['judge_appropriate']}</td></tr>"
                )
                i += 1
        parts.append("</table>")
    return "".join(parts)


def _aggregate_section_html(agg: dict[str, Any]) -> str:
    parts = ["<h2>Aggregate</h2>", f"<p><strong>Total calls</strong>: {agg['total_calls']}</p>"]

    summary: dict[str, dict] = agg.get("metric_summary", {})
    gate_metrics = {m: s for m, s in summary.items() if s["gating"] == "gate"}
    advisory_metrics = {m: s for m, s in summary.items() if s["gating"] == "advisory"}
    judge_metrics = {m: s for m, s in summary.items() if s["kind"] == "judge"}

    parts.append("<p><strong>Gate metrics</strong> — pass/fail/error/skipped across all calls:</p>")
    parts.append(
        '<table><colgroup><col style="width:40%"><col style="width:15%"><col style="width:15%">'
        '<col style="width:15%"><col style="width:15%"></colgroup>'
        "<tr><th>Metric</th><th>Pass</th><th>Fail</th><th>Error</th><th>Skipped</th></tr>"
    )
    for i, m in enumerate(sorted(gate_metrics)):
        s = gate_metrics[m]
        row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
        parts.append(f"<tr{row_style}><td>{_breakable_metric_name(m)}</td><td>{s['pass']}</td><td>{s['fail']}</td>"
                      f"<td>{s['error']}</td><td>{s['skipped']}</td></tr>")
    parts.append("</table>")

    parts.append("<p><strong>Advisory metrics</strong> — flag rate across all calls (never blocks ship):</p>")
    parts.append(
        '<table><colgroup><col style="width:60%"><col style="width:20%"><col style="width:20%"></colgroup>'
        "<tr><th>Metric</th><th>Flag rate</th><th>Ran</th></tr>"
    )
    for i, m in enumerate(sorted(advisory_metrics)):
        s = advisory_metrics[m]
        flag_rate = f"{s['flag_rate'] * 100:.0f}%" if s["flag_rate"] is not None else "n/a"
        row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
        parts.append(f"<tr{row_style}><td>{_breakable_metric_name(m)}</td><td>{flag_rate}</td><td>{s['ran']}</td></tr>")
    parts.append("</table>")

    parts.append("<p><strong>Judge coverage</strong> — calls each judge metric actually ran on (sampling policy):</p>")
    parts.append(
        '<table><colgroup><col style="width:45%"><col style="width:15%"><col style="width:20%">'
        '<col style="width:20%"></colgroup>'
        "<tr><th>Judge metric</th><th>Ran</th><th>Total calls</th><th>Coverage</th></tr>"
    )
    for i, m in enumerate(sorted(judge_metrics)):
        s = judge_metrics[m]
        coverage = f"{(s['ran'] / agg['total_calls'] * 100):.0f}%" if agg["total_calls"] else "n/a"
        row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
        parts.append(f"<tr{row_style}><td>{_breakable_metric_name(m)}</td><td>{s['ran']}</td>"
                      f"<td>{agg['total_calls']}</td><td>{coverage}</td></tr>")
    parts.append("</table>")

    errored = sorted(m for m, s in summary.items() if s["error"] > 0)
    error_note = (
        f" (metrics with at least one ERROR: {_esc(', '.join(errored))} — "
        "ERROR means the evaluator broke, not that the call failed)" if errored else ""
    )
    parts.append(f"<p><strong>Error rate</strong>: {agg['error_rate'] * 100:.1f}%{error_note}</p>")

    kind_counts = agg.get("kind_counts", {})
    parts.append(
        f"<p><strong>Deterministic vs. judge split</strong> (C1(8)): {kind_counts.get('deterministic', 0)} "
        f"deterministic, {kind_counts.get('signal', 0)} signal, {kind_counts.get('judge', 0)} judge score(s) "
        "across all calls. Deterministic/signal metrics run unconditionally on every call; judge metrics "
        "run per the sampling policy (defaults to full coverage on a fixture set).</p>"
    )
    if agg.get("judge_trust_note"):
        parts.append(f"<p>{_esc(agg['judge_trust_note'])}</p>")
    return "".join(parts)


def _gate_advisory_table_html(gate_breakdown: list[dict[str, Any]]) -> str:
    overflow: list[tuple[str, str]] = []
    rows = []
    for i, row in enumerate(gate_breakdown):
        rationale = row["rationale"]
        shown, truncated = _truncate(rationale, max_len=GATE_RATIONALE_MAX_LEN)
        title_attr = f' title="{_esc(rationale[:TITLE_MAX_LEN])}"' if truncated else ""
        if truncated:
            overflow.append((row["metric"], rationale))
        row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
        rows.append(
            f"<tr{row_style}><td>{_breakable_metric_name(row['metric'])}</td><td>{_esc(row['default_gating'])}</td>"
            f"<td{title_attr}>{_esc(shown)}</td></tr>"
        )

    table = (
        '<table class="gate-table">'
        '<colgroup><col style="width:20%"><col style="width:12%"><col style="width:68%"></colgroup>'
        "<tr><th>Metric</th><th>Gating</th><th>Rationale</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return table + _overflow_details(overflow, "Full rationale text (truncated above)")


def render_html_report(report: Report, gate_breakdown: list[dict[str, Any]]) -> str:
    agg = report.aggregate
    body: list[str] = [_header_html(agg)]

    body.append("<h2>Per-call results</h2>")
    body.append(
        "<table><tr><th>Call</th><th>Ship</th><th>Gate failures</th>"
        "<th>Advisory flags</th><th>barge_in (headline)</th></tr>"
    )
    for i, call_id in enumerate(sorted(report.per_call)):
        entry = report.per_call[call_id]
        scores = entry["scores"]
        call_verdict = compute_ship_verdict(scores)
        headline = next((s for s in scores if s.metric == HEADLINE_METRIC), None)
        headline_cell = headline.status.value if headline is not None else "n/a (not run)"
        row_style = f' style="{ZEBRA_STYLE}"' if i % 2 == 1 else ""
        body.append(
            f"<tr{row_style}><td>{_esc(call_id)}</td><td>{'ship' if entry['verdict'].ship else 'hold'}</td>"
            f"<td>{len(call_verdict['gate_failures'])}</td><td>{len(call_verdict['advisory_failures'])}</td>"
            f"<td>{_esc(headline_cell)}</td></tr>"
        )
    body.append("</table>")

    body.append("<h2>Per-call details</h2>")
    for call_id in sorted(report.per_call):
        scores = report.per_call[call_id]["scores"]
        verdict_word = "SHIP" if report.per_call[call_id]["verdict"].ship else "HOLD"
        body.append(f"<h3>{_esc(call_id)} — {verdict_word}</h3>")
        body.append(_per_call_details_html(scores))

    body.append(f"<h2>Headline metric: <code>{_esc(HEADLINE_METRIC)}</code></h2>")
    body.append(
        "<p>Detects each point the caller starts speaking while the agent is still talking and measures "
        "time-to-yield — the flagship acoustic behavior (assessment.md line 54).</p><ul>"
    )
    for call_id in sorted(report.per_call):
        scores = report.per_call[call_id]["scores"]
        headline = next((s for s in scores if s.metric == HEADLINE_METRIC), None)
        if headline is None:
            body.append(f"<li><strong>{_esc(call_id)}</strong>: not run</li>")
        else:
            issue_count = headline.details.get("issue_count", "n/a")
            body.append(f"<li><strong>{_esc(call_id)}</strong>: {headline.status.value} (issues: {issue_count})</li>")
    body.append("</ul>")

    body.append(_emotion_section_html(report))
    body.append(_aggregate_section_html(agg))

    body.append("<h2>Gate vs. advisory</h2>")
    body.append(_gate_advisory_table_html(gate_breakdown))

    return (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
        f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    )
