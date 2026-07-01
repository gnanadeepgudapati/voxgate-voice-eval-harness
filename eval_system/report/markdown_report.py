"""Human-readable combined report (assessment.md: "a combined report for a
fixture set" a human "would actually circulate internally" -- JSON alone
isn't that). Pure function over an already-built `Report` + the gate-vs-
advisory breakdown list -- no file I/O here, so it's directly testable;
`eval_system/run.py` writes the returned string to `out/report.md`.

Structure (a reviewer who hasn't seen the codebase should be able to follow
this top to bottom, and skim the whole thing in under a minute): top-line
verdict -> per-call summary table -> per-call details (every MetricScore,
grouped semantic/acoustic, gate failures highlighted) -> headline metric ->
emotion cross-check -> aggregate (totals, per-metric pass/fail/error, judge
coverage) -> the gate-vs-advisory rationale table.

Deliberately terse: the extra numbers that used to live in a separate
"Acoustic measured values" block (barge-in time-to-yield, turn-taking
percentiles, F0/rate, first-token latency) are folded into the per-call
metrics table's Reason column instead of duplicated in prose; a standalone
"Faithfulness judge findings" section was similarly folded into that
metric's own table row (verdict + score are already separate columns;
Reason carries the claim count + rationale, truncated like any other judge
free-text). `emotional_appropriateness` is dropped from the per-call table
entirely -- it duplicates the two-proxy emotion signal already tracked
per call (`ser_emotion` + `emotion_appropriateness_mm`) -- but still rolls
up in the Aggregate section like every other metric."""
from __future__ import annotations

import re
from typing import Any

from eval_system.metrics.base import Gating, MetricScore, Status
from eval_system.report.combine import Report, compute_ship_verdict

HEADLINE_METRIC = "barge_in"

SEMANTIC_METRICS = {
    "task_success", "tool_call_ordering", "instruction_adherence_rule",
    "instruction_adherence_judge", "faithfulness",
}
ACOUSTIC_METRICS = {
    "barge_in", "turn_taking_latency", "latency_thresholds", "pitch_prosody",
    "entity_intelligibility", "emotional_appropriateness", "double_talk",
    "ser_emotion", "emotion_appropriateness_mm", "naturalness_mos",
}


def _headline_cell(scores) -> str:
    headline = next((s for s in scores if s.metric == HEADLINE_METRIC), None)
    return headline.status.value if headline is not None else "n/a (not run)"


def _fmt_score(score: float | None) -> str:
    return f"{score:.2f}" if isinstance(score, (int, float)) else "n/a"


def _single_line(text: str, max_len: int = 160) -> str:
    """Collapses embedded newlines/markdown list syntax (real LLM judge
    responses are often multi-line) into one line BEFORE truncating, so a
    table cell never gets corrupted by a raw multi-line string. Truncation
    cuts at the last word boundary (never mid-word) and marks it with an
    ellipsis; full text is never available in Markdown (no tooltip/details
    mechanism there), so this is deliberately terser than html_report.py's
    truncate-with-tooltip-and-overflow-details treatment of the same text."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_len:
        return collapsed
    cut = collapsed[:max_len]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def _compress_error(exc: str, max_len: int) -> str:
    """A raw evaluator exception (a Gemini 429 quota payload can run to a
    thousand characters of nested JSON) is never useful inline -- the quota
    case is common enough on repeated CI runs to get its own fixed, short
    message; anything else is truncated like any other free-text reason."""
    lowered = exc.lower()
    if "429" in exc or "resource_exhausted" in lowered or "quota" in lowered:
        return "ERROR — quota exceeded (429)"
    return _single_line(f"ERROR — {exc}", max_len)


def _one_line_reason(score: MetricScore, max_len: int = 160) -> str:
    """`max_len` controls how aggressively free-text fields (judge notes,
    ERROR messages) get truncated. Markdown rendering uses the 160-char
    default; html_report.py calls this with a very large max_len to get the
    full (whitespace-collapsed, never raw-multiline) text for its own
    140-char truncate-with-tooltip/details treatment -- one shared "what does
    this metric's reason mean" source of truth, two different display caps."""
    d = score.details
    if score.status is Status.SKIPPED:
        return d.get("reason", "skipped")
    if score.status is Status.ERROR:
        return _compress_error(str(d.get("exc", "unknown")), max_len)

    metric = score.metric
    if metric == "task_success":
        return (
            f"final tool `{d.get('final_tool')}` matched success criteria" if score.status is Status.PASS
            else f"expected `{d.get('expected_final_tool')}`, got `{d.get('actual_final_tool')}`"
        )
    if metric == "tool_call_ordering":
        violations = d.get("invariants", {}).get("never_zero_appointments_violations", [])
        missing = d.get("sequence_order", {}).get("missing_or_out_of_order", [])
        if violations:
            return f"never_zero_appointments violated at {len(violations)} point(s)"
        if missing:
            return f"{len(missing)} tool call(s) missing/out of order"
        return "sequence + invariants OK"
    if metric == "instruction_adherence_rule":
        missing = d.get("missing_entities", [])
        return f"missing: {', '.join(missing)}" if missing else "all critical entities read back"
    if metric in ("instruction_adherence_judge", "emotional_appropriateness"):
        return _single_line(d.get("notes") or "", max_len) or "no notes"
    if metric == "faithfulness":
        claims = d.get("ungrounded_claims", [])
        count_str = f"{len(claims)} ungrounded claim(s)" if claims else "grounded"
        rationale = d.get("rationale", "")
        return _single_line(f"{count_str} — {rationale}", max_len) if rationale else count_str
    if metric == "barge_in":
        barge_ins = d.get("barge_ins", [])
        if not barge_ins:
            return "no barge-in issues"
        parts = []
        for b in barge_ins:
            flag = " FAIL" if b.get("fail_to_yield") else (" FALSE-YIELD" if b.get("false_yield") else "")
            parts.append(f"{b['time_to_yield'] * 1000:.0f}ms{flag}")
        return _single_line(f"{len(barge_ins)} event(s): " + ", ".join(parts), max_len)
    if metric == "turn_taking_latency":
        return (
            f"p50={d['p50'] * 1000:.0f}ms p90={d['p90'] * 1000:.0f}ms p99={d['p99'] * 1000:.0f}ms (n={d.get('n')})"
            if "p50" in d else d.get("reason", "n/a")
        )
    if metric == "latency_thresholds":
        v = d.get("violations", [])
        if "first_token_latency_sec" in d:
            return f"FTL={d['first_token_latency_sec'] * 1000:.0f}ms, {len(v)} violation(s)"
        return f"{len(v)} threshold violation(s)" if v else "within threshold"
    if metric == "pitch_prosody":
        issues = d.get("issues", [])
        issue_str = ", ".join(issues) if issues else "no issues"
        if d.get("pitch_mean_hz") is not None:
            rate = d.get("speech_rate_wps")
            rate_str = f"{rate * 60:.0f}wpm" if rate is not None else "n/a"
            return _single_line(f"F0={d['pitch_mean_hz']:.0f}Hz(±{d['pitch_range_hz']:.0f}) rate={rate_str}; {issue_str}", max_len)
        return issue_str
    if metric == "entity_intelligibility":
        missing = d.get("missing_entities", [])
        return f"missing: {', '.join(missing)}" if missing else "all critical entities survived STT"
    if metric == "double_talk":
        return f"overlap {d.get('overlap_seconds', 0):.2f}s / {d.get('call_duration_sec', 0):.2f}s"
    if metric == "ser_emotion":
        return f"dominant={d.get('dominant_label')}"
    if metric == "emotion_appropriateness_mm":
        per_turn = d.get("per_turn", [])
        inappropriate = sum(1 for t in per_turn if not t.get("appropriate", True))
        return f"{inappropriate}/{len(per_turn)} turn(s) flagged inappropriate" if inappropriate else f"all {len(per_turn)} turn(s) appropriate"
    if metric == "naturalness_mos":
        return f"mean_mos={d['mean_mos']:.2f} ({d.get('band')})" if d.get("mean_mos") is not None else "n/a"
    return "—"


def _metrics_table(scores: list[MetricScore]) -> list[str]:
    lines = ["| Metric | Status | Score | Reason |", "|---|---|---|---|"]
    for s in sorted(scores, key=lambda s: s.metric):
        status_cell = s.status.value.upper()
        if s.status in (Status.FAIL, Status.ERROR) and s.gating is Gating.GATE:
            status_cell = f"**{status_cell} ⚠️ GATE**"
        lines.append(f"| {s.metric} | {status_cell} | {_fmt_score(s.score)} | {_one_line_reason(s, max_len=80)} |")
    return lines


# Dropped from the per-call breakdown table: emotional_appropriateness is a
# text-only judge over a prosody summary that duplicates the two tracked
# emotion proxies (ser_emotion + emotion_appropriateness_mm) already shown
# per call for the disagreement signal -- still visible in the Aggregate
# section's per-metric rollup, just not repeated as a fourth emotion row
# on every call.
PER_CALL_TABLE_EXCLUDE = {"emotional_appropriateness"}


def _per_call_details(scores: list[MetricScore]) -> list[str]:
    lines: list[str] = []
    scores = [s for s in scores if s.metric not in PER_CALL_TABLE_EXCLUDE]
    semantic = [s for s in scores if s.metric in SEMANTIC_METRICS]
    acoustic = [s for s in scores if s.metric in ACOUSTIC_METRICS]
    other = [s for s in scores if s.metric not in SEMANTIC_METRICS and s.metric not in ACOUSTIC_METRICS]

    if semantic:
        lines.append("**Semantic metrics**")
        lines.append("")
        lines.extend(_metrics_table(semantic))
        lines.append("")
    if acoustic:
        lines.append("**Acoustic metrics**")
        lines.append("")
        lines.extend(_metrics_table(acoustic))
        lines.append("")
    if other:
        lines.append("**Other metrics**")
        lines.append("")
        lines.extend(_metrics_table(other))
        lines.append("")
    return lines


def _aggregate_section(agg: dict[str, Any]) -> list[str]:
    lines: list[str] = ["## Aggregate", ""]
    lines.append(f"- **Total calls**: {agg['total_calls']}")
    lines.append("")

    summary: dict[str, dict] = agg.get("metric_summary", {})
    gate_metrics = {m: s for m, s in summary.items() if s["gating"] == "gate"}
    advisory_metrics = {m: s for m, s in summary.items() if s["gating"] == "advisory"}
    judge_metrics = {m: s for m, s in summary.items() if s["kind"] == "judge"}

    lines.append("**Gate metrics** -- pass/fail/error/skipped across all calls:")
    lines.append("")
    lines.append("| Metric | Pass | Fail | Error | Skipped |")
    lines.append("|---|---|---|---|---|")
    for m in sorted(gate_metrics):
        s = gate_metrics[m]
        lines.append(f"| {m} | {s['pass']} | {s['fail']} | {s['error']} | {s['skipped']} |")
    lines.append("")

    lines.append("**Advisory metrics** -- flag rate across all calls (never blocks ship):")
    lines.append("")
    lines.append("| Metric | Flag rate | Ran |")
    lines.append("|---|---|---|")
    for m in sorted(advisory_metrics):
        s = advisory_metrics[m]
        flag_rate = f"{s['flag_rate'] * 100:.0f}%" if s["flag_rate"] is not None else "n/a"
        lines.append(f"| {m} | {flag_rate} | {s['ran']} |")
    lines.append("")

    lines.append("**Judge coverage** -- calls each judge metric actually ran on (sampling policy):")
    lines.append("")
    lines.append("| Judge metric | Ran | Total calls | Coverage |")
    lines.append("|---|---|---|---|")
    for m in sorted(judge_metrics):
        s = judge_metrics[m]
        coverage = f"{(s['ran'] / agg['total_calls'] * 100):.0f}%" if agg["total_calls"] else "n/a"
        lines.append(f"| {m} | {s['ran']} | {agg['total_calls']} | {coverage} |")
    lines.append("")

    lines.append(f"- **Error rate**: {agg['error_rate'] * 100:.1f}%")
    errored = sorted(m for m, s in summary.items() if s["error"] > 0)
    if errored:
        lines.append(f"  (metrics with at least one ERROR: {', '.join(errored)} -- ERROR means the evaluator broke, not that the call failed)")
    lines.append("")

    kind_counts = agg.get("kind_counts", {})
    lines.append(
        f"- **Deterministic vs. judge split** (C1(8)): {kind_counts.get('deterministic', 0)} deterministic, "
        f"{kind_counts.get('signal', 0)} signal, {kind_counts.get('judge', 0)} judge score(s) across all calls. "
        "Deterministic/signal metrics run unconditionally on every call; judge metrics run per the sampling "
        "policy (defaults to full coverage on a fixture set)."
    )
    if agg.get("judge_trust_note"):
        lines.append(f"- {agg['judge_trust_note']}")
    lines.append("")
    return lines


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

    lines.append("## Per-call details")
    lines.append("")
    for call_id in sorted(report.per_call):
        scores = report.per_call[call_id]["scores"]
        lines.append(f"### {call_id} — {'SHIP' if report.per_call[call_id]['verdict'].ship else 'HOLD'}")
        lines.append("")
        lines.extend(_per_call_details(scores))

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

    lines.extend(_aggregate_section(agg))

    lines.append("## Gate vs. advisory")
    lines.append("")
    lines.append("| Metric | Gating | Rationale |")
    lines.append("|---|---|---|")
    for row in gate_breakdown:
        lines.append(f"| {row['metric']} | {row['default_gating']} | {row['rationale']} |")
    lines.append("")

    return "\n".join(lines)
