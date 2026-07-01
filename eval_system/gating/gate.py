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
    "task_success": "Deterministic: final tool call checked against the fixture's ground-truth success criteria.",
    "tool_call_ordering": "Deterministic state-machine check, including the reschedule-trap invariant -- catches a real correctness bug.",
    "instruction_adherence_rule": "Deterministic substring check that critical entities were read back to the caller.",
    "instruction_adherence_judge": "LLM judge for conversational nuance a keyword check can't capture -- advisory until calibration earns trust.",
    "faithfulness": "LLM judge for grounding -- advisory until calibration proves the judge itself trustworthy.",
    "barge_in": "Deterministic once computed from VAD, and the headline interruption-handling behavior -- a real miss is a real defect.",
    "turn_taking_latency": "Reports a latency distribution (p50/p90/p99), not a single verdict -- advisory by nature.",
    "latency_thresholds": "Deterministic arithmetic against a threshold that is itself a judgment call -- advisory until promoted.",
    "pitch_prosody": "F0 and speech rate are perceptual proxies for naturalness, not correctness.",
    "entity_intelligibility": "Round-trip STT on ground-truthed critical entities -- if a real STT engine can't recover it, a caller likely couldn't either.",
    "emotional_appropriateness": "Text judge over a prosody summary, not true multimodal audio -- always advisory, never promoted.",
    "double_talk": "[LIVE FOLLOW-UP] Overlap alone isn't necessarily a defect -- reports duration/ratio, advisory by nature.",
    "ser_emotion": "Objective classifier, but acted-emotion SER is noisier than human inter-rater agreement (IEMOCAP kappa ~0.3-0.5) -- can't gate.",
    "emotion_appropriateness_mm": "Multimodal judge that hears real audio and context, but LLM judges drift and are noisy -- always advisory, never promoted.",
    "naturalness_mos": "[BEYOND-SCOPE ADDITION] Non-intrusive MOS saturates above ~4 and can't separate 'good' from 'excellent' -- always advisory.",
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
