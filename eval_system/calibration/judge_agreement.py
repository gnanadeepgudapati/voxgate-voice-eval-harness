"""Cohen's kappa of a judge's verdicts against a small human-labeled set -->
trust tier. CLAUDE.md's gating rule: judges start ADVISORY and may only be
promoted to GATE once this clears a threshold -- a judge earns authority, it
isn't assumed."""
from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import cohen_kappa_score

TRUST_KAPPA_THRESHOLD = 0.6


@dataclass
class JudgeAgreementResult:
    metric_name: str
    kappa: float | None
    n: int
    trusted: bool


def compute_judge_agreement(
    metric_name: str,
    judge_labels: list[str],
    human_labels: list[str],
    *,
    kappa_threshold: float = TRUST_KAPPA_THRESHOLD,
) -> JudgeAgreementResult:
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must be the same length")

    n = len(judge_labels)
    if len(set(judge_labels) | set(human_labels)) < 2:
        # No variability in the label set -- kappa is mathematically
        # undefined (0/0 expected-agreement term). Never silently trust this.
        return JudgeAgreementResult(metric_name, kappa=None, n=n, trusted=False)

    kappa = cohen_kappa_score(judge_labels, human_labels)
    return JudgeAgreementResult(metric_name, kappa=kappa, n=n, trusted=kappa >= kappa_threshold)
