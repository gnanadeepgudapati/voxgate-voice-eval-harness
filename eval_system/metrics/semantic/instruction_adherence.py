"""Followed the agent's rules? Split into a deterministic rule (gate) and an
LLM judge for nuance (advisory) -- two separate BaseMetric classes since a
MetricScore is one kind (deterministic xor judge), never both.

The rule half checks the one instruction every fixture already encodes: the
agent must read the caller's critical entities (provider, confirmation
number, etc.) back to them, using `expected.critical_entities`. The judge
half (below) assesses conversational nuance a substring match can't."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from eval_system.judges.client import JudgeClient
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


@register
class InstructionAdherenceRuleMetric(BaseMetric):
    name = "instruction_adherence_rule"
    version = "1"
    kind = MetricKind.DETERMINISTIC
    default_gating = Gating.GATE
    requires_ground_truth = True

    def compute(self, ctx: "MetricContext") -> MetricScore:
        agent_text = " ".join(
            turn.text for turn in ctx.transcript if turn.speaker == "agent"
        ).lower()
        critical_entities = ctx.expected.get("critical_entities", [])
        missing = [e for e in critical_entities if e.lower() not in agent_text]

        status = Status.FAIL if missing else Status.PASS
        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=status,
            gating=self.default_gating,
            score=0.0 if missing else 1.0,
            details={"missing_entities": missing},
            evaluator_version=self.version,
        )


class InstructionAdherenceJudgment(BaseModel):
    followed_rules: bool
    score: float
    notes: str = ""


@register
class InstructionAdherenceJudgeMetric(BaseMetric):
    name = "instruction_adherence_judge"
    version = "1"
    kind = MetricKind.JUDGE
    default_gating = Gating.ADVISORY
    requires_ground_truth = False
    prompt_version = "v1"

    def __init__(self, client: JudgeClient | None = None):
        self.client = client

    def _get_client(self) -> JudgeClient:
        if self.client is None:
            from eval_system.judges.anthropic_client import AnthropicJudgeClient

            self.client = AnthropicJudgeClient()
        return self.client

    def compute(self, ctx: "MetricContext") -> MetricScore:
        judgment = self._get_client().structured_complete(self._build_prompt(ctx), InstructionAdherenceJudgment)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS if judgment.followed_rules else Status.FAIL,
            gating=self.default_gating,
            score=judgment.score,
            details={"notes": judgment.notes},
            evaluator_version=self.version,
            judge_prompt_version=self.prompt_version,
        )

    def _build_prompt(self, ctx: "MetricContext") -> str:
        lines = "\n".join(f"[{turn.t_start:.1f}s] {turn.speaker}: {turn.text}" for turn in ctx.transcript)
        return (
            "You are grading a clinic-scheduling voice agent's adherence to good "
            "conversational practice (e.g. confirming details back, not overpromising, "
            "staying within scope) beyond simple keyword checks.\n\n"
            f"Transcript:\n{lines}\n"
        )
