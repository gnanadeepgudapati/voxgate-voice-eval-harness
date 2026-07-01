"""Entry point: scores a fixture set end-to-end and writes per-call +
aggregate reports.

    uv run python -m eval_system.run --fixtures fixtures/ --out out/
    uv run python -m eval_system.run --fixtures fixtures/ --out out/ --metrics faithfulness,barge_in
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import build_metric_context
from eval_system.gating.gate import gate_advisory_breakdown
from eval_system.metrics import registry
from eval_system.metrics.base import MetricScore
from eval_system.report.combine import Report, build_report, upsert_scores
from eval_system.report.markdown_report import HEADLINE_METRIC, render_markdown_report
from eval_system.report.pdf_report import markdown_to_pdf_bytes

REPORT_FILENAME_RE = re.compile(r"^report_(\d+)\.md$")

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
    double_talk,
    emotion_appropriateness_mm,
    emotional_appropriateness,
    entity_intelligibility,
    latency_thresholds,
    pitch_prosody,
    ser_emotion,
    turn_taking_latency,
)


TEMPLATE_FIXTURE_DIR_NAME = "TEMPLATE"


def discover_fixtures(fixtures_dir: Path) -> list[Path]:
    # fixtures/TEMPLATE/ is a copyable authoring skeleton (see fixtures/TEMPLATE/README.md),
    # not a real scenario -- it must never be scored as part of a real eval run.
    return sorted(
        p for p in Path(fixtures_dir).iterdir()
        if p.is_dir() and p.name != TEMPLATE_FIXTURE_DIR_NAME
    )


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


def next_report_number(out_dir: Path) -> int:
    """Each write_report() call gets its own report_<n>.md/.pdf rather than
    overwriting a static report.md -- every run's report is kept, not just
    the latest. Scans existing report_<n>.md files and returns the next
    integer (1 if none exist yet)."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 1
    existing = [
        int(match.group(1))
        for p in out_dir.iterdir()
        if (match := REPORT_FILENAME_RE.match(p.name))
    ]
    return max(existing, default=0) + 1


def write_report(report: Report, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for call_id, entry in report.per_call.items():
        scores = entry["scores"]
        headline_score = next((s for s in scores if s.metric == HEADLINE_METRIC), None)
        payload = {
            "call_id": call_id,
            "ship": entry["verdict"].ship,
            "failures": entry["verdict"].failures,
            "headline": _score_to_dict(headline_score) if headline_score is not None else None,
            "emotion_disagreement_turns": entry["emotion_disagreement_turns"],
            "scores": [_score_to_dict(s) for s in scores],
        }
        (out_dir / f"{call_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (out_dir / "aggregate.json").write_text(json.dumps(report.aggregate, indent=2), encoding="utf-8")

    breakdown = gate_advisory_breakdown(registry.REGISTRY)
    (out_dir / "gate_advisory_breakdown.json").write_text(json.dumps(breakdown, indent=2), encoding="utf-8")

    report_n = next_report_number(out_dir)
    report_markdown = render_markdown_report(report, breakdown)
    (out_dir / f"report_{report_n}.md").write_text(report_markdown, encoding="utf-8")
    (out_dir / f"report_{report_n}.pdf").write_bytes(markdown_to_pdf_bytes(report_markdown))


def run_cli(argv: list[str] | None = None) -> int:
    """Scores the fixture set and writes the report; returns the process exit
    code (0 == ship, 1 == hold) so CI can gate on it (assessment.md Part 1
    Q4: "a single ship / don't-ship decision for a CI pipeline")."""
    parser = argparse.ArgumentParser(description="Score a VoxGate fixture set")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metrics", type=str, default=None, help="comma-separated metric names to run")
    args = parser.parse_args(argv)

    metrics_filter = set(args.metrics.split(",")) if args.metrics else None
    report = score_fixture_set(args.fixtures, metrics_filter=metrics_filter)
    write_report(report, args.out)
    print(f"Scored {report.aggregate['total_calls']} call(s) -> {args.out}")
    print(report.aggregate["ship_reason"])

    return 0 if report.aggregate["ship"] else 1


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
