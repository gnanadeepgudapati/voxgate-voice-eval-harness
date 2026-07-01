"""On-disk fixture file schema (validation only). A fixture directory holds:

    call.wav          2-channel audio: channel 0 = caller, channel 1 = agent
    transcript.jsonl  authored ground-truth turns (speaker/t_start/t_end/text)
    tool_log.jsonl    tool calls as actually executed during the recorded call
    events.jsonl      authored timeline markers (barge-in onset, yield, cough, ...)
    scenario_db.json  initial ground-truth DB state (free-form, domain-specific)
    expected.json     expected tool sequence + invariants + critical entities

All timestamps are seconds on the canonical clock (audio sample index / sr) —
see eval_system/context/metric_context.py.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TurnModel(BaseModel):
    speaker: Literal["agent", "caller"]
    t_start: float
    t_end: float
    text: str
    asr_confidence: float | None = None


class ToolEventModel(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    t: float


class EventModel(BaseModel):
    name: str
    t: float
    meta: dict[str, Any] = Field(default_factory=dict)


class ExpectedToolCall(BaseModel):
    name: str
    required_args: dict[str, Any] = Field(default_factory=dict)


class SuccessCriteria(BaseModel):
    final_tool: str | None = None
    result_contains: dict[str, Any] = Field(default_factory=dict)


class ExpectedModel(BaseModel):
    tool_sequence: list[ExpectedToolCall] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list)
    critical_entities: list[str] = Field(default_factory=list)
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
