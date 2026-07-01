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

from eval_system.gating.gate import evaluate_gate
from eval_system.metrics.base import MetricKind, MetricScore, Status

ScoreStore = dict[tuple, MetricScore]


def upsert_scores(store: ScoreStore, new_scores: list[MetricScore]) -> ScoreStore:
    updated = dict(store)
    for s in new_scores:
        updated[s.key] = s
    return updated


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
        per_call[call_id] = {"scores": scores, "verdict": evaluate_gate(scores, trusted_judge_metrics)}

    all_scores = list(store.values())
    verdicts = [entry["verdict"] for entry in per_call.values()]
    aggregate = {
        "total_calls": len(by_call),
        "ships": sum(1 for v in verdicts if v.ship),
        "holds": sum(1 for v in verdicts if not v.ship),
        "kind_counts": {k.value: sum(1 for s in all_scores if s.kind is k) for k in MetricKind},
        "error_rate": (sum(1 for s in all_scores if s.status is Status.ERROR) / len(all_scores)) if all_scores else 0.0,
        "trusted_judge_metrics": sorted(trusted_judge_metrics),
    }
    return Report(per_call=per_call, aggregate=aggregate)
