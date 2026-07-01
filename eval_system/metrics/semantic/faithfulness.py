"""Are the agent's statements grounded in tool results? LLM-as-judge,
structured output. Advisory until calibration promotes it (see
CLAUDE.md gating-trust rule) -- a judge earns gate authority only once
its kappa vs a human set clears a threshold. Reads per-turn asr_confidence
(C1(6)): a low-confidence transcription is flagged for the judge to weigh
cautiously rather than treated as ground truth; absent confidence changes
nothing."""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from eval_system.judges.client import JudgeClient
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

LOW_ASR_CONFIDENCE_THRESHOLD = 0.6


class FaithfulnessJudgment(BaseModel):
    grounded: bool
    score: float
    ungrounded_claims: list[str] = Field(default_factory=list)
    rationale: str = ""


@register
class FaithfulnessMetric(BaseMetric):
    name = "faithfulness"
    version = "1"
    kind = MetricKind.JUDGE
    default_gating = Gating.ADVISORY
    requires_ground_truth = True
    prompt_version = "v1"

    def __init__(self, client: JudgeClient | None = None):
        self.client = client

    def _get_client(self) -> JudgeClient:
        if self.client is None:
            from eval_system.judges.factory import get_default_judge_client

            self.client = get_default_judge_client()
        return self.client

    def compute(self, ctx: "MetricContext") -> MetricScore:
        judgment = self._get_client().structured_complete(self._build_prompt(ctx), FaithfulnessJudgment)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS if judgment.grounded else Status.FAIL,
            gating=self.default_gating,
            score=judgment.score,
            details={"ungrounded_claims": judgment.ungrounded_claims, "rationale": judgment.rationale},
            evaluator_version=self.version,
            judge_prompt_version=self.prompt_version,
        )

    def _build_prompt(self, ctx: "MetricContext") -> str:
        tool_facts = "\n".join(f"{te.name}({te.args}) -> {te.result}" for te in ctx.tool_events)
        agent_lines = "\n".join(
            f"[{turn.t_start:.1f}s] {turn.text}" for turn in ctx.transcript if turn.speaker == "agent"
        )
        low_confidence = [
            turn
            for turn in ctx.transcript
            if turn.asr_confidence is not None and turn.asr_confidence < LOW_ASR_CONFIDENCE_THRESHOLD
        ]

        prompt = (
            "You are grading whether a clinic-scheduling voice agent's statements are "
            "grounded in the tool results below. Flag any claim not supported by a tool result.\n\n"
            f"Tool results:\n{tool_facts}\n\n"
            f"Agent statements:\n{agent_lines}\n"
        )
        if low_confidence:
            notes = "\n".join(
                f"- [{t.t_start:.1f}s] {t.speaker}: {t.text!r} (confidence {t.asr_confidence:.2f})"
                for t in low_confidence
            )
            prompt += (
                "\nThe following turns had low-confidence transcription and may not "
                f"reflect what was actually said -- weigh them cautiously:\n{notes}\n"
            )
        return prompt
