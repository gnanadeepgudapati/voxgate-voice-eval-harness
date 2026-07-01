"""Preflight sanity checks on a loaded fixture BEFORE it's scored (channels,
timeline bounds, clipping) plus a retry-then-quarantine wrapper (←C1(4)):
a fixture that fails to load/validate gets a few attempts (transient I/O),
then is quarantined -- excluded from scoring entirely -- rather than
crashing the run or silently being scored against bad data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from eval_system.context.fixture_loader import RawFixture

CLIPPING_ABS_THRESHOLD = 0.999
MAX_RETRIES = 3


@dataclass
class ValidationIssue:
    check: str
    message: str


def preflight_check(fixture: "RawFixture") -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if fixture.audio_caller.shape != fixture.audio_agent.shape:
        issues.append(ValidationIssue(
            "channels",
            f"caller/agent channel length mismatch: {fixture.audio_caller.shape} vs {fixture.audio_agent.shape}",
        ))

    for name, audio in [("caller", fixture.audio_caller), ("agent", fixture.audio_agent)]:
        if audio.size and np.max(np.abs(audio)) >= CLIPPING_ABS_THRESHOLD:
            issues.append(ValidationIssue("clipping", f"{name} channel shows clipping"))

    duration = len(fixture.audio_caller) / fixture.sr if fixture.sr else 0.0
    for turn in fixture.transcript:
        if turn.t_end < turn.t_start:
            issues.append(ValidationIssue("timeline", f"turn t_end < t_start: {turn}"))
        elif turn.t_end > duration + 1e-3:
            issues.append(ValidationIssue("timeline", f"turn extends past call duration ({duration:.2f}s): {turn}"))

    return issues


def check_events_timeline(fixture: "RawFixture") -> list[ValidationIssue]:
    """Authoring-time check (not part of preflight_check(), which runs on
    every scoring pass): events.jsonl timestamps must be non-negative,
    strictly increasing, and within the call's audio duration. A fixture
    author's typo here (e.g. a marker placed after the audio ends) silently
    produces a meaningless metric rather than an error, so this is checked
    explicitly before a fixture is trusted."""
    issues: list[ValidationIssue] = []
    duration = len(fixture.audio_caller) / fixture.sr if fixture.sr else 0.0
    prev_t: float | None = None
    for evt in fixture.events:
        if evt.t < 0:
            issues.append(ValidationIssue("events_timeline", f"event '{evt.name}' has negative t={evt.t}"))
        if evt.t > duration + 1e-3:
            issues.append(ValidationIssue(
                "events_timeline",
                f"event '{evt.name}' at t={evt.t:.3f}s exceeds call duration ({duration:.2f}s)",
            ))
        if prev_t is not None and evt.t <= prev_t:
            issues.append(ValidationIssue(
                "events_timeline",
                f"event '{evt.name}' at t={evt.t} is not strictly after the previous event (t={prev_t})",
            ))
        prev_t = evt.t
    return issues


@dataclass
class ProcessingResult:
    call_id: str
    status: str  # "ok" | "quarantined"
    attempts: int
    fixture: "RawFixture | None" = None
    issues: list[ValidationIssue] = field(default_factory=list)


def process_fixture_with_retry(
    call_id: str,
    load_fn: Callable[[str], "RawFixture"],
    max_retries: int = MAX_RETRIES,
) -> ProcessingResult:
    last_issues: list[ValidationIssue] = []
    for attempt in range(1, max_retries + 1):
        try:
            fixture = load_fn(call_id)
        except Exception as e:
            last_issues = [ValidationIssue("load_error", repr(e))]
            continue

        issues = preflight_check(fixture)
        if not issues:
            return ProcessingResult(call_id=call_id, status="ok", attempts=attempt, fixture=fixture)
        last_issues = issues

    return ProcessingResult(call_id=call_id, status="quarantined", attempts=max_retries, fixture=None, issues=last_issues)
