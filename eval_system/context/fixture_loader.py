"""Reads a fixture directory into validated, in-memory records. This is
parsing + schema validation only — channel-offset correction and the actual
clock join are Phase 3's job in metric_context.py's context builder."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from eval_system.context.fixture_schema import (
    EventModel,
    ExpectedModel,
    ToolEventModel,
    TurnModel,
)
from eval_system.context.metric_context import Event, ToolEvent, Turn

CHANNEL_CALLER = 0
CHANNEL_AGENT = 1


@dataclass
class RawFixture:
    call_id: str
    sr: int
    audio_caller: np.ndarray
    audio_agent: np.ndarray
    transcript: list[Turn]
    tool_events: list[ToolEvent]
    events: list[Event]
    scenario_db: dict
    expected: dict


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_fixture(fixture_dir: str | Path) -> RawFixture:
    fixture_dir = Path(fixture_dir)
    call_id = fixture_dir.name

    audio, sr = sf.read(fixture_dir / "call.wav", always_2d=True)
    if audio.shape[1] != 2:
        raise ValueError(f"{fixture_dir}: call.wav must be 2-channel, got {audio.shape[1]}")

    transcript = [
        Turn(**TurnModel(**t).model_dump())
        for t in _read_jsonl(fixture_dir / "transcript.jsonl")
    ]
    tool_events = [
        ToolEvent(**ToolEventModel(**t).model_dump())
        for t in _read_jsonl(fixture_dir / "tool_log.jsonl")
    ]
    events = [
        Event(**EventModel(**e).model_dump())
        for e in _read_jsonl(fixture_dir / "events.jsonl")
    ]

    scenario_db = json.loads((fixture_dir / "scenario_db.json").read_text(encoding="utf-8"))
    expected_raw = json.loads((fixture_dir / "expected.json").read_text(encoding="utf-8"))
    expected = ExpectedModel(**expected_raw).model_dump()

    return RawFixture(
        call_id=call_id,
        sr=sr,
        audio_caller=audio[:, CHANNEL_CALLER],
        audio_agent=audio[:, CHANNEL_AGENT],
        transcript=transcript,
        tool_events=tool_events,
        events=events,
        scenario_db=scenario_db,
        expected=expected,
    )
