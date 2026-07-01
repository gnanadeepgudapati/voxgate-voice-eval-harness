"""Fuses a fixture set's MetricScores into one report: per-call scores +
gate verdict, plus the aggregate split (deterministic/judge/signal counts,
error rate, trusted-judge set). Both suites feed this one report -- there is
no separate semantic/acoustic reporting path (CLAUDE.md: "one verdict").
Upserts by `MetricScore.key` (←C1(2)): re-scoring a call overwrites its prior
record rather than duplicating it."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from eval_system.gating.gate import evaluate_gate, is_gate_eligible
from eval_system.metrics.base import MetricKind, MetricScore, Status

ScoreStore = dict[tuple, MetricScore]

JUDGE_TRUST_KAPPA_THRESHOLD = 0.60


def upsert_scores(store: ScoreStore, new_scores: list[MetricScore]) -> ScoreStore:
    updated = dict(store)
    for s in new_scores:
        updated[s.key] = s
    return updated


def compute_ship_verdict(
    scores: list[MetricScore], trusted_judge_metrics: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """The single run-level ship/don't-ship decision (assessment.md: both
    suites "feed one verdict"; Part 1 Q4's "single ship / don't-ship decision
    for a CI pipeline"). Takes any list of MetricScore -- the whole run, or a
    single call's subset for a per-call breakdown -- so the same rule is used
    everywhere it's asked for.

    SHIP iff there are ZERO scores that are both (a) gate-eligible (GATE, or a
    JUDGE metric explicitly promoted via `trusted_judge_metrics`) and (b)
    status == FAIL. ADVISORY failures are recorded for visibility but never
    block. ERROR is deliberately excluded from both failure buckets here --
    ERROR means the evaluator broke, not that the call did (CLAUDE.md:
    "ERROR != FAIL"); it's already tracked in aggregate.error_rate."""
    gate_failures = []
    advisory_failures = []
    for s in scores:
        if s.status is not Status.FAIL:
            continue
        entry = {"call_id": s.call_id, "metric": s.metric, "status": s.status.value}
        if is_gate_eligible(s, trusted_judge_metrics):
            gate_failures.append(entry)
        else:
            advisory_failures.append(entry)

    ship = len(gate_failures) == 0
    if ship:
        note = f" ({len(advisory_failures)} advisory flag(s), non-blocking)" if advisory_failures else ""
        ship_reason = f"SHIP: no gate failures{note}"
    else:
        first = gate_failures[0]
        extra = f", +{len(gate_failures) - 1} more" if len(gate_failures) > 1 else ""
        ship_reason = f"HOLD: {len(gate_failures)} gate failure(s) ({first['metric']} on {first['call_id']}{extra})"

    return {
        "ship": ship,
        "gate_failures": gate_failures,
        "advisory_failures": advisory_failures,
        "ship_reason": ship_reason,
    }


def compute_metric_summary(scores: list[MetricScore]) -> dict[str, dict[str, Any]]:
    """Per-metric rollup across all calls: status counts (pass/fail/error/
    skipped), plus `ran` (attempted -- pass+fail+error, excluding skipped,
    since a skip means the metric legitimately didn't apply) and `flag_rate`
    (fail / ran) for quick scanning. Feeds the aggregate report section: gate
    metrics' pass/fail/error counts, advisory metrics' flag rates, and judge
    coverage (`ran` vs. total calls) all come from this one rollup."""
    by_metric: dict[str, dict[str, Any]] = {}
    for s in scores:
        entry = by_metric.setdefault(s.metric, {
            "kind": s.kind.value,
            "gating": s.gating.value,
            "pass": 0, "fail": 0, "error": 0, "skipped": 0,
        })
        entry[s.status.value] += 1

    for entry in by_metric.values():
        ran = entry["pass"] + entry["fail"] + entry["error"]
        entry["ran"] = ran
        entry["flag_rate"] = (entry["fail"] / ran) if ran else None

    return by_metric


def judge_trust_note(trusted_judge_metrics: frozenset[str]) -> str | None:
    """A grader (or CI operator) reading an empty `trusted_judge_metrics` list
    could easily misread it as a broken judge layer rather than what it is:
    no judge has cleared calibration yet, so every judge metric is correctly
    running advisory-only. Returns None once at least one judge is trusted,
    so the note doesn't linger in the report as stale noise."""
    if trusted_judge_metrics:
        return None
    return (
        f"no judge metric has cleared the calibration kappa bar (>={JUDGE_TRUST_KAPPA_THRESHOLD:.2f}); "
        "all judge metrics running as advisory"
    )


SER_METRIC_NAME = "ser_emotion"
MM_JUDGE_METRIC_NAME = "emotion_appropriateness_mm"


def _turns_disagree(ser_valence: str, judge_tone: str, judge_appropriate: bool) -> bool:
    """Two independent read-outs of the same audio moment -- SER's coarse
    valence from the waveform, and the multimodal judge's tone + verdict.
    Flags disagreement two ways: (1) the two proxies assign different coarse
    valence to the same turn, or (2) SER heard clear positive emotion but the
    judge said the tone was inappropriate anyway (the "chirpy during bad
    news" case: SER agrees the tone WAS upbeat, judge says that was wrong for
    the moment -- a real disagreement about appropriateness, not just labels).
    Either is worth a human's attention rather than trusting either proxy
    alone (assessment.md Part 1 Q3: "evaluate your own evaluators")."""
    from eval_system.metrics.acoustic.ser_emotion import ser_label_to_valence

    judge_valence = ser_label_to_valence(judge_tone)
    if ser_valence != judge_valence:
        return True
    return ser_valence == "positive" and not judge_appropriate


def compute_emotion_disagreement(scores: list[MetricScore]) -> list[dict[str, Any]]:
    """Cross-checks `ser_emotion` (objective, context-blind) against
    `emotion_appropriateness_mm` (contextual, but a noisy LLM judge) per agent
    turn. REPORTING ONLY -- both metrics are permanently advisory, so this
    never affects ship; it's the trust signal itself (disagreement -> flag
    for human review) rather than a verdict. If either metric didn't run or
    errored, there's nothing to compare, so returns []."""
    from eval_system.metrics.acoustic.ser_emotion import ser_label_to_valence

    ser_score = next((s for s in scores if s.metric == SER_METRIC_NAME), None)
    mm_score = next((s for s in scores if s.metric == MM_JUDGE_METRIC_NAME), None)
    if ser_score is None or mm_score is None:
        return []
    if ser_score.status is Status.ERROR or mm_score.status is Status.ERROR:
        return []

    ser_by_span = {(pt["start"], pt["end"]): pt for pt in ser_score.details.get("per_turn", [])}

    disagreements = []
    for mm_turn in mm_score.details.get("per_turn", []):
        ser_turn = ser_by_span.get((mm_turn["start"], mm_turn["end"]))
        if ser_turn is None:
            continue

        ser_valence = ser_label_to_valence(ser_turn["label"])
        if _turns_disagree(ser_valence, mm_turn["detected_tone"], mm_turn["appropriate"]):
            disagreements.append({
                "turn": mm_turn["turn_index"],
                "ser_label": ser_turn["label"],
                "judge_tone": mm_turn["detected_tone"],
                "judge_appropriate": mm_turn["appropriate"],
            })
    return disagreements


def _scores_by_call(store: ScoreStore) -> dict[str, list[MetricScore]]:
    by_call: dict[str, list[MetricScore]] = defaultdict(list)
    for s in store.values():
        by_call[s.call_id].append(s)
    return dict(by_call)


@dataclass
class Report:
    per_call: dict[str, dict[str, Any]]
    aggregate: dict[str, Any]


def build_report(store: ScoreStore, trusted_judge_metrics: frozenset[str] = frozenset()) -> Report:
    by_call = _scores_by_call(store)

    per_call = {}
    for call_id, scores in by_call.items():
        per_call[call_id] = {
            "scores": scores,
            "verdict": evaluate_gate(scores, trusted_judge_metrics),
            "emotion_disagreement_turns": compute_emotion_disagreement(scores),
        }

    all_scores = list(store.values())
    verdicts = [entry["verdict"] for entry in per_call.values()]
    aggregate = {
        "total_calls": len(by_call),
        "ships": sum(1 for v in verdicts if v.ship),
        "holds": sum(1 for v in verdicts if not v.ship),
        "kind_counts": {k.value: sum(1 for s in all_scores if s.kind is k) for k in MetricKind},
        "error_rate": (sum(1 for s in all_scores if s.status is Status.ERROR) / len(all_scores)) if all_scores else 0.0,
        "trusted_judge_metrics": sorted(trusted_judge_metrics),
        "metric_summary": compute_metric_summary(all_scores),
        **compute_ship_verdict(all_scores, trusted_judge_metrics),
    }

    note = judge_trust_note(trusted_judge_metrics)
    if note is not None:
        aggregate["judge_trust_note"] = note

    return Report(per_call=per_call, aggregate=aggregate)
