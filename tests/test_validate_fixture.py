import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval_system.validate_fixture import run_cli, validate_fixture

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SR = 8000


def _write_fixture(
    fixture_dir: Path,
    duration_sec: float = 2.0,
    events=None,
    transcript=None,
    scenario_db=None,
    expected=None,
):
    fixture_dir.mkdir(parents=True, exist_ok=True)
    n = round(duration_sec * SR)
    stereo = np.zeros((n, 2), dtype=np.float32)
    sf.write(fixture_dir / "call.wav", stereo, SR, subtype="PCM_16")

    (fixture_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in (events or [])), encoding="utf-8"
    )
    (fixture_dir / "transcript.jsonl").write_text(
        "\n".join(json.dumps(t) for t in (transcript or [])), encoding="utf-8"
    )
    (fixture_dir / "scenario_db.json").write_text(
        json.dumps(scenario_db if scenario_db is not None else {"patients": {}}), encoding="utf-8"
    )
    (fixture_dir / "expected.json").write_text(
        json.dumps(expected if expected is not None else {
            "tool_sequence": [], "invariants": [], "critical_entities": [],
        }),
        encoding="utf-8",
    )


def _levels(results, check=None):
    return {r.level for r in results if check is None or r.check == check}


def test_valid_real_fixture_has_no_hard_failures():
    results = validate_fixture(FIXTURES_DIR / "happy_path_book")

    assert "fail" not in _levels(results)


def test_reschedule_trap_real_fixture_has_no_hard_failures():
    results = validate_fixture(FIXTURES_DIR / "reschedule_trap")

    assert "fail" not in _levels(results)


def test_missing_required_file_fails_with_load_check(tmp_path):
    fixture_dir = tmp_path / "broken"
    _write_fixture(fixture_dir)
    (fixture_dir / "scenario_db.json").unlink()

    results = validate_fixture(fixture_dir)

    load_results = [r for r in results if r.check == "load"]
    assert len(load_results) == 1
    assert load_results[0].level == "fail"


def test_event_beyond_audio_duration_is_hard_fail(tmp_path):
    fixture_dir = tmp_path / "bad_events"
    _write_fixture(fixture_dir, duration_sec=1.0, events=[{"name": "cough", "t": 50.0, "meta": {}}])

    results = validate_fixture(fixture_dir)

    assert any(r.check == "events_timeline" and r.level == "fail" for r in results)


def test_unknown_tool_name_is_hard_fail(tmp_path):
    fixture_dir = tmp_path / "bad_tool"
    _write_fixture(fixture_dir, expected={
        "tool_sequence": [{"name": "delete_all_patients", "required_args": {}}],
        "invariants": [], "critical_entities": [],
    })

    results = validate_fixture(fixture_dir)

    assert any(r.check == "tool_names" and r.level == "fail" for r in results)


def test_misaligned_interrupt_marker_is_warn_not_fail(tmp_path):
    fixture_dir = tmp_path / "misaligned"
    _write_fixture(
        fixture_dir,
        duration_sec=5.0,
        transcript=[{"speaker": "agent", "t_start": 0.0, "t_end": 1.0, "text": "hi", "asr_confidence": None}],
        events=[{"name": "barge_in_start", "t": 3.0, "meta": {"channel": "caller"}}],  # outside [0,1]
    )

    results = validate_fixture(fixture_dir)

    assert any(r.check == "semantic_alignment" and r.level == "warn" for r in results)
    assert "fail" not in _levels(results)  # must warn, not hard-fail


def test_aligned_interrupt_marker_passes():
    # barge_in_basic's real markers land inside real agent turns (see docs/ERRORS.md).
    results = validate_fixture(FIXTURES_DIR / "barge_in_basic")

    assert any(r.check == "semantic_alignment" and r.level == "pass" for r in results)


def test_unknown_event_name_is_warn_not_fail(tmp_path):
    fixture_dir = tmp_path / "typo_event"
    _write_fixture(fixture_dir, events=[{"name": "cuogh", "t": 0.1, "meta": {}}])

    results = validate_fixture(fixture_dir)

    assert any(r.check == "event_vocabulary" and r.level == "warn" for r in results)
    assert "fail" not in _levels(results)


def test_missing_critical_entities_for_transactional_scenario_is_warn(tmp_path):
    fixture_dir = tmp_path / "no_entities"
    _write_fixture(fixture_dir, expected={
        "tool_sequence": [{"name": "book_appointment", "required_args": {}}],
        "invariants": [], "critical_entities": [],
    })

    results = validate_fixture(fixture_dir)

    assert any(r.check == "critical_entities" and r.level == "warn" for r in results)


def test_run_cli_returns_0_for_valid_fixture():
    exit_code = run_cli([str(FIXTURES_DIR / "happy_path_book")])

    assert exit_code == 0


def test_run_cli_returns_1_for_fixture_with_hard_failure(tmp_path):
    fixture_dir = tmp_path / "broken"
    _write_fixture(fixture_dir)
    (fixture_dir / "scenario_db.json").unlink()

    exit_code = run_cli([str(fixture_dir)])

    assert exit_code == 1
