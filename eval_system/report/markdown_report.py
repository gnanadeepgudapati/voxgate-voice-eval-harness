"""Human-readable combined report (assessment.md: "a combined report for a
fixture set" a human "would actually circulate internally" -- JSON alone
isn't that). Pure function over an already-built `Report` + the gate-vs-
advisory breakdown list -- no file I/O here, so it's directly testable;
`eval_system/run.py` writes the returned string to `out/report.md`."""
from __future__ import annotations

from typing import Any

from eval_system.report.combine import Report, compute_ship_verdict

HEADLINE_METRIC = "barge_in"


def _headline_cell(scores) -> str:
    headline = next((s for s in scores if s.metric == HEADLINE_METRIC), None)
    return headline.status.value if headline is not None else "n/a (not run)"


def render_markdown_report(report: Report, gate_breakdown: list[dict[str, Any]]) -> str:
    agg = report.aggregate
    lines: list[str] = []

    lines.append("# SHIP" if agg["ship"] else "# HOLD")
    lines.append("")
    lines.append(agg["ship_reason"])
    lines.append("")

    lines.append("## Per-call results")
    lines.append("")
    lines.append("| Call | Ship | Gate failures | Advisory flags | barge_in (headline) |")
    lines.append("|---|---|---|---|---|")
    for call_id in sorted(report.per_call):
        entry = report.per_call[call_id]
        scores = entry["scores"]
        call_verdict = compute_ship_verdict(scores)
        lines.append(
            f"| {call_id} | {'ship' if entry['verdict'].ship else 'hold'} "
            f"| {len(call_verdict['gate_failures'])} | {len(call_verdict['advisory_failures'])} "
            f"| {_headline_cell(scores)} |"
        )
    lines.append("")

    lines.append(f"## Headline metric: `{HEADLINE_METRIC}`")
    lines.append("")
    lines.append(
        "Detects each point the caller starts speaking while the agent is still talking and "
        "measures time-to-yield -- the flagship acoustic behavior (assessment.md line 54)."
    )
    lines.append("")
    for call_id in sorted(report.per_call):
        scores = report.per_call[call_id]["scores"]
        headline = next((s for s in scores if s.metric == HEADLINE_METRIC), None)
        if headline is None:
            lines.append(f"- **{call_id}**: not run")
            continue
        issue_count = headline.details.get("issue_count", "n/a")
        lines.append(f"- **{call_id}**: {headline.status.value} (issues: {issue_count})")
    lines.append("")

    lines.append("## Emotion (advisory)")
    lines.append("")
    lines.append(
        "Two permanently-advisory proxies -- objective offline SER (`ser_emotion`) vs. a "
        "contextual multimodal judge (`emotion_appropriateness_mm`) -- cross-checked per "
        "agent turn. Disagreement is the trust signal here (assessment.md Part 1 Q3): "
        "neither metric can gate, but a divergence flags the turn for human review."
    )
    lines.append("")
    any_disagreements = any(report.per_call[c]["emotion_disagreement_turns"] for c in report.per_call)
    if not any_disagreements:
        lines.append("No disagreement turns across any call.")
    else:
        lines.append("| Call | Turn | SER label | Judge tone | Judge appropriate |")
        lines.append("|---|---|---|---|---|")
        for call_id in sorted(report.per_call):
            for d in report.per_call[call_id]["emotion_disagreement_turns"]:
                lines.append(
                    f"| {call_id} | {d['turn']} | {d['ser_label']} | {d['judge_tone']} | {d['judge_appropriate']} |"
                )
    lines.append("")

    lines.append("## Gate vs. advisory")
    lines.append("")
    lines.append("| Metric | Gating | Rationale |")
    lines.append("|---|---|---|")
    for row in gate_breakdown:
        lines.append(f"| {row['metric']} | {row['default_gating']} | {row['rationale']} |")
    lines.append("")

    return "\n".join(lines)
