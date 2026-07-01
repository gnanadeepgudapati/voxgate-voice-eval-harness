import json
from pathlib import Path

from eval_system.run import discover_fixtures, run_cli, score_fixture_set, write_report

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
FAST_METRICS = {"task_success", "tool_call_ordering", "instruction_adherence_rule"}


def test_discover_fixtures_finds_known_fixture_dirs():
    dirs = discover_fixtures(FIXTURES_DIR)

    assert {d.name for d in dirs} == {"happy_path_book", "reschedule_trap", "barge_in_basic"}


def test_discover_fixtures_excludes_template():
    # fixtures/TEMPLATE/ is a copyable authoring skeleton, not a real scenario --
    # it must never be swept into a real eval run's aggregate.
    dirs = discover_fixtures(FIXTURES_DIR)

    assert "TEMPLATE" not in {d.name for d in dirs}


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


def test_aggregate_json_has_run_level_ship_fields(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    write_report(report, tmp_path)

    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert "ship" in aggregate
    assert "gate_failures" in aggregate
    assert "advisory_failures" in aggregate
    assert "ship_reason" in aggregate
    # reschedule_trap fails tool_call_ordering (a gate metric) -> run-level HOLD
    assert aggregate["ship"] is False


def test_per_call_file_has_headline_field_with_barge_in_score(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter={"task_success", "barge_in"})

    write_report(report, tmp_path)

    payload = json.loads((tmp_path / "happy_path_book.json").read_text(encoding="utf-8"))
    assert payload["headline"]["metric"] == "barge_in"


def test_per_call_file_headline_is_none_when_barge_in_did_not_run(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    write_report(report, tmp_path)

    payload = json.loads((tmp_path / "happy_path_book.json").read_text(encoding="utf-8"))
    assert payload["headline"] is None


def test_per_call_file_has_emotion_disagreement_turns_field(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    write_report(report, tmp_path)

    payload = json.loads((tmp_path / "happy_path_book.json").read_text(encoding="utf-8"))
    assert payload["emotion_disagreement_turns"] == []


def test_write_report_writes_versioned_markdown_and_pdf(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    write_report(report, tmp_path)

    report_md = (tmp_path / "report_1.md").read_text(encoding="utf-8")
    assert "HOLD" in report_md  # reschedule_trap's tool_call_ordering failure holds the run
    assert "barge_in" in report_md
    assert "Gate vs. advisory" in report_md

    pdf_bytes = (tmp_path / "report_1.pdf").read_bytes()
    assert pdf_bytes.startswith(b"%PDF")


def test_write_report_increments_filename_on_each_call(tmp_path):
    report = score_fixture_set(FIXTURES_DIR, metrics_filter=FAST_METRICS)

    write_report(report, tmp_path)
    write_report(report, tmp_path)
    write_report(report, tmp_path)

    assert (tmp_path / "report_1.md").exists()
    assert (tmp_path / "report_1.pdf").exists()
    assert (tmp_path / "report_2.md").exists()
    assert (tmp_path / "report_2.pdf").exists()
    assert (tmp_path / "report_3.md").exists()
    assert (tmp_path / "report_3.pdf").exists()
    # the old static filename is retired entirely, not just superseded
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "report.pdf").exists()


def test_run_cli_exit_code_0_when_ship_true(tmp_path):
    # task_success alone: PASS on happy_path_book, SKIPPED (no final_tool defined)
    # on the other two -- nothing fails, so the run-level verdict ships.
    exit_code = run_cli(["--fixtures", str(FIXTURES_DIR), "--out", str(tmp_path), "--metrics", "task_success"])

    assert exit_code == 0
    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["ship"] is True


def test_run_cli_exit_code_1_when_ship_false(tmp_path):
    # tool_call_ordering fails on reschedule_trap (the reschedule-trap invariant) --
    # a gate metric failure, so the run-level verdict holds.
    exit_code = run_cli(["--fixtures", str(FIXTURES_DIR), "--out", str(tmp_path), "--metrics", "tool_call_ordering"])

    assert exit_code == 1
    aggregate = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate["ship"] is False
