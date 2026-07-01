import json
from pathlib import Path

from eval_system.run import discover_fixtures, score_fixture_set, write_report

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FAST_METRICS = {"task_success", "tool_call_ordering", "instruction_adherence_rule"}


def test_discover_fixtures_finds_known_fixture_dirs():
    dirs = discover_fixtures(FIXTURES_DIR)

    assert {d.name for d in dirs} == {"happy_path_book", "reschedule_trap", "barge_in_basic"}


def test_score_fixture_set_produces_a_report_over_all_fixtures():
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    assert set(report.per_call.keys()) == {"happy_path_book", "reschedule_trap", "barge_in_basic"}
    assert report.aggregate["total_calls"] == 3
    for entry in report.per_call.values():
        assert {s.metric for s in entry["scores"]} <= FAST_METRICS


def test_write_report_produces_per_call_and_aggregate_json(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    write_report(report, tmp_path)

    assert (tmp_path / "happy_path_book.json").exists()
    assert (tmp_path / "aggregate.json").exists()
    assert (tmp_path / "gate_advisory_breakdown.json").exists()

    payload = json.loads((tmp_path / "happy_path_book.json").read_text(encoding="utf-8"))
    assert payload["call_id"] == "happy_path_book"
    assert isinstance(payload["ship"], bool)
    assert isinstance(payload["scores"], list)
    assert payload["scores"][0]["kind"] in {"deterministic", "judge", "signal"}

    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["total_calls"] == 3

    breakdown = json.loads((tmp_path / "gate_advisory_breakdown.json").read_text(encoding="utf-8"))
    assert any(row["metric"] == "task_success" for row in breakdown)
