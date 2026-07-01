"""Core metric contract. Locked interface — see CLAUDE.md. Do not change shape
without updating every metric and the report/gate layers that depend on it."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

SCHEMA_VERSION = "1"


class MetricKind(str, Enum):
    DETERMINISTIC = "deterministic"
    JUDGE = "judge"
    SIGNAL = "signal"


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"      # evaluator itself broke — distinct from a metric FAIL
    SKIPPED = "skipped"  # e.g. not sampled this run


class Gating(str, Enum):
    GATE = "gate"
    ADVISORY = "advisory"


@dataclass
class MetricScore:
    call_id: str
    metric: str
    kind: MetricKind
    status: Status
    gating: Gating
    score: float | None
    details: dict[str, Any] = field(default_factory=dict)
    evaluator_version: str = "0"
    judge_prompt_version: str | None = None
    schema_version: str = SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, str, str, str | None]:
        """Idempotency key: re-running a call upserts this record, never duplicates it."""
        return (self.call_id, self.metric, self.evaluator_version, self.judge_prompt_version)


class BaseMetric:
    name: str
    version: str
    kind: MetricKind
    default_gating: Gating
    requires_ground_truth: bool

    def compute(self, ctx: "MetricContext") -> MetricScore:
        raise NotImplementedError
