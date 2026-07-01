"""The alignment backbone. Phase 1: types only — the join logic (Phase 3) is the
highest-risk component in the system and gets its own tests before any metric
reads from it. CANONICAL CLOCK = audio sample index at `sr`; every time value on
every dataclass here is seconds on that one clock. Metrics must never re-derive
timing from raw audio — they read it from here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
