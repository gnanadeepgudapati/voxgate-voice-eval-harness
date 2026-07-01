"""Two-tier gate: fuses one call's MetricScore list into a ship/no-ship
verdict. Only deterministic metrics and calibration-TRUSTED judges may
hard-gate (CLAUDE.md's gating-trust rule) -- an ADVISORY score, however bad,
never blocks a ship.

"pass^k": shipping requires ALL k gate-eligible metrics to pass -- a
conjunction, not an average -- so one clear-cut violation holds the call
however good the rest look. (This is deliberately distinct from judge
self-consistency sampling -- repeating one judge call k times and requiring
self-agreement -- which CLAUDE.md scopes as a Category-2 design decision:
documented, not implemented, here.)

An ERROR on a gate-eligible metric is fail-closed (doesn't ship) but reported
distinctly from a FAIL: the evaluator broke, not necessarily the call itself.
SKIPPED gate metrics (this metric legitimately doesn't apply to this fixture,
e.g. task_success with no success_criteria) are excluded from the
conjunction entirely, rather than counted as a missing pass."""
from __future__ import annotations

from dataclasses import dataclass, field

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status

GATE_FAILING_STATUSES = (Status.FAIL, Status.ERROR)

# The DoD's "explicit gate-vs-advisory list with rationale" -- why each metric
# sits where it does, not just what it's set to. Kept alongside the gate logic
# rather than in report/combine.py since it's about gating policy, not output
# formatting.
GATE_RATIONALE: dict[str, str] = {
    "task_success": "Deterministic: final tool call vs. the fixture's ground-truth success criteria. No proxy involved.",
    "tool_call_ordering": "Deterministic state reducer incl. the reschedule-trap invariant -- catches a real correctness bug, not a style preference.",
    "instruction_adherence_rule": "Deterministic substring check that critical entities were read back to the caller -- objective and ground-truthed.",
    "instruction_adherence_judge": "LLM judge for conversational nuance a keyword check can't capture; starts advisory until judge_agreement clears the kappa threshold.",
    "faithfulness": "LLM judge; grounding matters but the judge itself is an unproven proxy until calibration trusts it -- advisory until promoted.",
    "barge_in": "VAD-derived but a deterministic decision once computed, and the headline correctness behavior for a voice agent -- a real barge-in miss is a real defect.",
    "turn_taking_latency": "Reports a distribution (p50/p90/p99), not a single call verdict -- advisory by nature; feeds `latency_thresholds` once a hard cutoff is chosen.",
    "latency_thresholds": "Deterministic timestamp arithmetic against a fixed threshold, but the threshold itself is a judgment call -- advisory until promoted (C1(7)).",
    "pitch_prosody": "F0/speech-rate are perceptual proxies for naturalness, not correctness -- advisory.",
    "entity_intelligibility": "Round-trip STT on ground-truthed critical entities -- if a real STT engine can't recover it, a caller likely couldn't either.",
    "emotional_appropriateness": "Text judge over a prosody *summary*, not true multimodal audio -- always advisory per CLAUDE.md's explicit invariant, never promoted.",
}


def gate_advisory_breakdown(registered_metrics: list[BaseMetric]) -> list[dict]:
    return [
        {
            "metric": m.name,
            "kind": m.kind.value,
            "default_gating": m.default_gating.value,
            "rationale": GATE_RATIONALE.get(m.name, ""),
        }
        for m in registered_metrics
    ]


@dataclass
class GateVerdict:
    call_id: str
    ship: bool
    gate_metrics: list[str] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)


def is_gate_eligible(score: MetricScore, trusted_judge_metrics: frozenset[str]) -> bool:
    if score.gating is Gating.GATE:
        return True
    return score.kind is MetricKind.JUDGE and score.metric in trusted_judge_metrics


def evaluate_gate(
    scores: list[MetricScore], trusted_judge_metrics: frozenset[str] = frozenset()
) -> GateVerdict:
    if not scores:
        raise ValueError("evaluate_gate requires at least one MetricScore")

    gate_scores = [s for s in scores if is_gate_eligible(s, trusted_judge_metrics)]
    counted = [s for s in gate_scores if s.status is not Status.SKIPPED]
    failures = [
        {"metric": s.metric, "status": s.status.value}
        for s in counted
        if s.status in GATE_FAILING_STATUSES
    ]

    return GateVerdict(
        call_id=scores[0].call_id,
        ship=len(failures) == 0,
        gate_metrics=[s.metric for s in gate_scores],
        failures=failures,
    )
