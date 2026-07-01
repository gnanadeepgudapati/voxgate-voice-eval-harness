"""Entry point: scores a fixture set end-to-end and writes per-call +
aggregate reports.

    uv run python -m eval_system.run --fixtures fixtures/ --out out/
    uv run python -m eval_system.run --fixtures fixtures/ --out out/ --metrics faithfulness,barge_in
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import build_metric_context
from eval_system.gating.gate import gate_advisory_breakdown
from eval_system.metrics import registry
from eval_system.metrics.base import MetricScore
from eval_system.report.combine import Report, build_report, upsert_scores

# Import every metric module so its @register side effect populates REGISTRY --
# a fresh interpreter running `python -m eval_system.run` otherwise sees an
# empty registry (metric modules only get imported by tests that reference
# them directly).
from eval_system.metrics.semantic import (  # noqa: F401
    faithfulness,
    instruction_adherence,
    task_success,
    tool_call_ordering,
)
from eval_system.metrics.acoustic import (  # noqa: F401
    barge_in,
    emotional_appropriateness,
    entity_intelligibility,
    latency_thresholds,
    pitch_prosody,
    turn_taking_latency,
)


def discover_fixtures(fixtures_dir: Path) -> list[Path]:
    return sorted(p for p in Path(fixtures_dir).iterdir() if p.is_dir())


def score_fixture_set(
    fixtures_dir: Path,
    metrics_filter: set[str] | None = None,
    trusted_judge_metrics: frozenset[str] = frozenset(),
) -> Report:
    store: dict = {}
    for fixture_dir in discover_fixtures(fixtures_dir):
        fixture = load_fixture(fixture_dir)
        ctx = build_metric_context(fixture)
        scores = registry.run(ctx, metrics_filter=metrics_filter)
        store = upsert_scores(store, scores)
    return build_report(store, trusted_judge_metrics)


def _score_to_dict(score: MetricScore) -> dict:
    d = asdict(score)
    d["kind"] = score.kind.value
    d["status"] = score.status.value
    d["gating"] = score.gating.value
    return d


def write_report(report: Report, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for call_id, entry in report.per_call.items():
        payload = {
            "call_id": call_id,
            "ship": entry["verdict"].ship,
            "failures": entry["verdict"].failures,
            "scores": [_score_to_dict(s) for s in entry["scores"]],
        }
        (out_dir / f"{call_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (out_dir / "aggregate.json").write_text(json.dumps(report.aggregate, indent=2), encoding="utf-8")
    (out_dir / "gate_advisory_breakdown.json").write_text(
        json.dumps(gate_advisory_breakdown(registry.REGISTRY), indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a VoxGate fixture set")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=str, default=None, help="comma-separated metric names to run")
    args = parser.parse_args()

    metrics_filter = set(args.metrics.split(",")) if args.metrics else None
    report = score_fixture_set(args.fixtures, metrics_filter=metrics_filter)
    write_report(report, args.out)
    print(f"Scored {report.aggregate['total_calls']} call(s) -> {args.out}")


if __name__ == "__main__":
    main()
