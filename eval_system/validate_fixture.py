"""Fixture-authoring validator: `python -m eval_system.validate_fixture
fixtures/<scenario>/`.

Distinct from validators/preflight.py's scoring-time checks (channels,
clipping, transcript timeline -- run silently on every eval pass): this is
an authoring-time preflight an author runs ONCE, by hand, before trusting a
new fixture. It reuses preflight_check() and check_events_timeline() rather
than duplicating them, and adds checks that only make sense at authoring
time -- unknown tool names, an unrecognized event name (typo?), and the
semantic-alignment check that catches the #1 real failure mode this system
has: an interrupt marker placed where the agent isn't actually speaking,
which makes barge_in run without error but measure nothing real.

Two constants below (KNOWN_TOOL_NAMES, KNOWN_EVENT_NAMES) are NOT enforced
anywhere else in the runtime -- eval_system/tools/ is an empty stub (no real
tool registry exists), and EventModel.name is an unconstrained str. These
are the vocabularies actually observed across the 3 real fixtures, used only
to catch likely typos; update them when a fixture legitimately needs
something new."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from eval_system.context.fixture_loader import load_fixture
from eval_system.validators.preflight import check_events_timeline, preflight_check

KNOWN_TOOL_NAMES = {"check_availability", "book_appointment", "cancel_appointment"}
KNOWN_EVENT_NAMES = {"cough", "barge_in_start", "agent_yield", "zero_appointments_start", "zero_appointments_end"}
MIN_REASONABLE_SAMPLE_RATE_HZ = 8000


@dataclass
class ValidationResult:
    check: str
    level: str  # "pass" | "warn" | "fail"
    message: str


def validate_fixture(fixture_dir: Path) -> list[ValidationResult]:
    fixture_dir = Path(fixture_dir)
    results: list[ValidationResult] = []

    try:
        fixture = load_fixture(fixture_dir)
    except Exception as e:
        results.append(ValidationResult("load", "fail", f"could not load fixture: {e!r}"))
        return results
    results.append(ValidationResult(
        "load", "pass", "call.wav, scenario_db.json, expected.json all present and parseable"
    ))

    preflight_issues = preflight_check(fixture)
    for issue in preflight_issues:
        results.append(ValidationResult(issue.check, "fail", issue.message))
    if not preflight_issues:
        results.append(ValidationResult("preflight", "pass", "channels/clipping/transcript timeline OK"))

    if fixture.sr < MIN_REASONABLE_SAMPLE_RATE_HZ:
        results.append(ValidationResult(
            "sample_rate", "warn",
            f"{fixture.sr}Hz is unusually low -- metrics resample to 16kHz regardless, but content "
            "below ~8kHz may already be missing before that resampling happens",
        ))
    else:
        results.append(ValidationResult("sample_rate", "pass", f"{fixture.sr}Hz"))

    events_issues = check_events_timeline(fixture)
    for issue in events_issues:
        results.append(ValidationResult(issue.check, "fail", issue.message))
    if not events_issues:
        results.append(ValidationResult("events_timeline", "pass", "all event timestamps in range and monotonic"))

    unknown_events = sorted({e.name for e in fixture.events} - KNOWN_EVENT_NAMES)
    if unknown_events:
        results.append(ValidationResult(
            "event_vocabulary", "warn",
            f"event name(s) not in the observed vocabulary {sorted(KNOWN_EVENT_NAMES)}: {unknown_events} "
            "-- typo, or a legitimate new event type?",
        ))
    else:
        results.append(ValidationResult("event_vocabulary", "pass", "all event names recognized"))

    agent_turns = [t for t in fixture.transcript if t.speaker == "agent"]
    interrupt_events = [e for e in fixture.events if e.meta.get("channel") == "caller"]
    misaligned = [e for e in interrupt_events if not any(t.t_start <= e.t <= t.t_end for t in agent_turns)]
    if misaligned:
        for e in misaligned:
            results.append(ValidationResult(
                "semantic_alignment", "warn",
                f"interrupt marker '{e.name}' at t={e.t:.3f}s does not fall inside any agent-speaking "
                "window from transcript.jsonl -- this barge_in marker will not measure a real yield.",
            ))
    elif interrupt_events:
        results.append(ValidationResult(
            "semantic_alignment", "pass", f"{len(interrupt_events)} interrupt marker(s) align with agent speech"
        ))
    else:
        results.append(ValidationResult("semantic_alignment", "pass", "no caller-side interrupt markers to check"))

    tool_names = {step["name"] for step in fixture.expected.get("tool_sequence", [])}
    final_tool = fixture.expected.get("success_criteria", {}).get("final_tool")
    if final_tool:
        tool_names.add(final_tool)
    unknown_tools = sorted(tool_names - KNOWN_TOOL_NAMES)
    if unknown_tools:
        results.append(ValidationResult(
            "tool_names", "fail",
            f"expected.json references unknown tool name(s) {unknown_tools} -- not in {sorted(KNOWN_TOOL_NAMES)}",
        ))
    else:
        results.append(ValidationResult("tool_names", "pass", "all referenced tools recognized"))

    is_transactional = bool(fixture.expected.get("tool_sequence")) or final_tool is not None
    if is_transactional and not fixture.expected.get("critical_entities"):
        results.append(ValidationResult(
            "critical_entities", "warn",
            "this scenario declares tool calls / a success criterion but no critical_entities -- "
            "entity_intelligibility has nothing to check for this fixture",
        ))
    else:
        results.append(ValidationResult("critical_entities", "pass", "critical_entities declared where expected"))

    return results


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a VoxGate fixture directory before trusting it")
    parser.add_argument("fixture_dir", type=Path)
    args = parser.parse_args(argv)

    results = validate_fixture(args.fixture_dir)
    for r in results:
        print(f"[{r.level.upper():4s}] {r.check}: {r.message}")

    hard_fails = [r for r in results if r.level == "fail"]
    warnings = [r for r in results if r.level == "warn"]
    if hard_fails:
        print(f"\n{len(hard_fails)} hard failure(s) -- fixture is NOT safe to trust.")
        return 1
    print(f"\nAll hard checks passed{f' ({len(warnings)} warning(s))' if warnings else ''}.")
    return 0


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
