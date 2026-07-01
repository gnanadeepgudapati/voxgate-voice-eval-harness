"""The alignment backbone. Phase 1: types only — the join logic (Phase 3) is the
highest-risk component in the system and gets its own tests before any metric
reads from it. CANONICAL CLOCK = audio sample index at `sr`; every time value on
every dataclass here is seconds on that one clock. Metrics must never re-derive
timing from raw audio — they read it from here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from eval_system.context.fixture_loader import RawFixture


@dataclass
class Turn:
    speaker: str          # "agent" | "caller"
    t_start: float        # seconds, canonical clock
    t_end: float          # seconds, canonical clock
    text: str
    asr_confidence: float | None = None  # optional ←C1(6); absent must not change behavior


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any]
    result: Any
    t: float               # seconds, canonical clock


@dataclass
class Event:
    """Authored timeline marker, e.g. {t: 4.2, event: "interrupt_start"}."""
    name: str
    t: float               # seconds, canonical clock
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricContext:
    call_id: str
    sr: int
    audio_agent: np.ndarray
    audio_caller: np.ndarray
    transcript: list[Turn]
    tool_events: list[ToolEvent]
    events: list[Event]
    expected: dict[str, Any]
    scenario_db: dict[str, Any]


def _correct_channel_offset(audio: np.ndarray, offset_sec: float, sr: int) -> np.ndarray:
    """Shifts a channel by a known systematic recording skew (e.g. differing
    per-leg codec buffering) onto the canonical clock. `offset_sec` > 0 means
    this channel's raw capture started that many seconds AFTER the canonical
    clock's t=0 (its content is delayed and must be shifted later to compensate);
    < 0 means it started early (shifted earlier)."""
    if offset_sec == 0.0:
        return audio
    offset_samples = round(offset_sec * sr)
    shifted = np.zeros_like(audio)
    if offset_samples > 0:
        if offset_samples < len(audio):
            shifted[offset_samples:] = audio[: len(audio) - offset_samples]
    else:
        n = -offset_samples
        if n < len(audio):
            shifted[: len(audio) - n] = audio[n:]
    return shifted


def build_metric_context(
    fixture: "RawFixture",
    *,
    agent_channel_offset_sec: float = 0.0,
    caller_channel_offset_sec: float = 0.0,
) -> MetricContext:
    """Builds the canonical MetricContext from a loaded fixture. Ground-truth
    timestamps (transcript/tool_events/events) are authored on the canonical
    clock already; only the raw channels can carry a known capture skew, which
    is corrected here so every array downstream shares that one clock."""
    return MetricContext(
        call_id=fixture.call_id,
        sr=fixture.sr,
        audio_agent=_correct_channel_offset(fixture.audio_agent, agent_channel_offset_sec, fixture.sr),
        audio_caller=_correct_channel_offset(fixture.audio_caller, caller_channel_offset_sec, fixture.sr),
        transcript=fixture.transcript,
        tool_events=fixture.tool_events,
        events=fixture.events,
        expected=fixture.expected,
        scenario_db=fixture.scenario_db,
    )
