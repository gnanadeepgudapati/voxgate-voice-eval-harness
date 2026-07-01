"""Multimodal appropriateness judge -- the second of two complementary,
PERMANENTLY-advisory attacks on "emotional appropriateness" (assessment.md
line 57; see docs/design_writeup.md §3). This one answers the doc's ACTUAL
question -- "was the tone appropriate for the moment?" -- which needs
conversational context a classifier can't use. Unlike the original
`emotional_appropriateness` (text description of prosody stats), this sends
real audio bytes to a multimodal model, fixing that metric's honestly-
documented "never actually hears the audio" gap.

Hardcoded non-promotable: LLM judges drift and are noisy (Part 1 Q3) -- this
metric is never eligible for gate promotion regardless of what calibration
says, same explicit invariant as the original emotional_appropriateness.

Reproducibility (this is a CI metric, so results must be stable across
runs): temperature=0, a pinned `judge_prompt_version`, and a JSON cache
keyed by (call_id, turn_index, judge_prompt_version, model) under
`out/.judge_cache/` -- a cache hit makes zero network calls, so a given
fixture yields the SAME judgment every run. The cache path is a fixed
relative location (not wired through the CLI's --out flag, since compute()
only receives a MetricContext, not the run's output directory) but is
constructor-injectable for tests.

Segments the AGENT channel by the same authored transcript turn boundaries
`ser_emotion.py` uses, so their per-turn results line up 1:1 for the
cross-metric disagreement check in report/combine.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, Field

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

GEMINI_MODEL = "gemini-2.5-flash"
JUDGE_PROMPT_VERSION = "mm-v1"
MM_SAMPLE_RATE = 16000
DEFAULT_CACHE_PATH = Path("out/.judge_cache/emotion_mm_cache.json")
PRIOR_CONTEXT_TURNS = 2

T = TypeVar("T", bound=BaseModel)


class MultimodalAppropriatenessJudgment(BaseModel):
    appropriate: bool
    score: int = Field(ge=1, le=5, description="1 (very inappropriate) to 5 (perfectly appropriate)")
    detected_tone: str = ""
    rationale: str = ""


class MultimodalJudgeClient(Protocol):
    model: str

    def judge_audio(self, audio_bytes: bytes, context_text: str, response_model: type[T]) -> T: ...


class GeminiAudioJudgeClient:
    """Thin adapter over the google-genai SDK. Not unit-tested itself for the
    same reason the other real judge clients aren't (Anthropic/OpenAI) --
    nothing meaningful to assert without mocking the whole SDK; tests inject
    a fake implementing the same protocol instead."""

    def __init__(self, model: str = GEMINI_MODEL, api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _sdk_client(self):
        if self._client is None:
            from eval_system.judges.factory import load_dotenv_if_available

            load_dotenv_if_available()

            import os

            api_key = self._api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set (env var or .env) -- required for emotion_appropriateness_mm"
                )

            from google import genai

            self._client = genai.Client(api_key=api_key)
        return self._client

    def judge_audio(self, audio_bytes: bytes, context_text: str, response_model: type[T]) -> T:
        from google.genai import types

        client = self._sdk_client()
        response = client.models.generate_content(
            model=self.model,
            contents=[context_text, types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        return response_model.model_validate_json(response.text)


def _slice_wav_bytes(audio_agent, sr: int, t_start: float, t_end: float) -> bytes:
    import io

    import librosa
    import numpy as np
    import soundfile as sf

    start_sample = max(0, round(t_start * sr))
    end_sample = min(len(audio_agent), round(t_end * sr))
    segment = audio_agent[start_sample:end_sample]
    resampled = librosa.resample(np.asarray(segment, dtype=np.float32), orig_sr=sr, target_sr=MM_SAMPLE_RATE)
    buf = io.BytesIO()
    sf.write(buf, resampled, MM_SAMPLE_RATE, format="WAV")
    return buf.getvalue()


@register
class MultimodalEmotionAppropriatenessMetric(BaseMetric):
    name = "emotion_appropriateness_mm"
    version = "1"
    kind = MetricKind.JUDGE
    default_gating = Gating.ADVISORY
    requires_ground_truth = False
    prompt_version = JUDGE_PROMPT_VERSION

    def __init__(self, client: MultimodalJudgeClient | None = None, cache_path: Path | None = None):
        self.client = client
        self.cache_path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE_PATH

    def _get_client(self) -> MultimodalJudgeClient:
        if self.client is None:
            self.client = GeminiAudioJudgeClient()
        return self.client

    def compute(self, ctx: "MetricContext") -> MetricScore:
        if ctx.audio_agent is None or len(ctx.audio_agent) == 0:
            raise ValueError("emotion_appropriateness_mm: no agent audio available")

        agent_turns = [(i, t) for i, t in enumerate(ctx.transcript) if t.speaker == "agent"]
        if not agent_turns:
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.SKIPPED,
                gating=self.default_gating,
                score=None,
                details={"reason": "no agent turns in transcript"},
                evaluator_version=self.version,
                judge_prompt_version=self.prompt_version,
            )

        per_turn = []
        for turn_index, turn in agent_turns:
            context_text = self._build_context(ctx, turn_index)
            audio_bytes = _slice_wav_bytes(ctx.audio_agent, ctx.sr, turn.t_start, turn.t_end)
            judgment = self._judge_turn(ctx.call_id, turn_index, audio_bytes, context_text)
            per_turn.append({
                "turn_index": turn_index,
                "start": turn.t_start,
                "end": turn.t_end,
                "appropriate": judgment.appropriate,
                "score": judgment.score,
                "detected_tone": judgment.detected_tone,
                "rationale": judgment.rationale,
            })

        overall_appropriate = all(p["appropriate"] for p in per_turn)
        mean_score = sum(p["score"] for p in per_turn) / len(per_turn)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.PASS if overall_appropriate else Status.FAIL,
            gating=self.default_gating,
            score=mean_score,
            details={"per_turn": per_turn, "model": self._get_client().model},
            evaluator_version=self.version,
            judge_prompt_version=self.prompt_version,
        )

    def _build_context(self, ctx: "MetricContext", turn_index: int) -> str:
        prior = ctx.transcript[max(0, turn_index - PRIOR_CONTEXT_TURNS):turn_index]
        prior_text = "\n".join(f"{t.speaker}: {t.text}" for t in prior) or "(this is the start of the call)"
        return (
            f"This is a clinic-scheduling phone call (call_id={ctx.call_id}). "
            f"Immediately before the attached audio clip:\n{prior_text}\n\n"
            "Judge whether the AGENT's vocal tone in the attached audio clip was "
            "appropriate for this moment in the conversation (e.g. calm and warm with an "
            "anxious caller, not chirpy when delivering bad news). Respond with the "
            "requested structured judgment."
        )

    def _judge_turn(self, call_id: str, turn_index: int, audio_bytes: bytes, context_text: str):
        client = self._get_client()
        key = f"{call_id}::{turn_index}::{self.prompt_version}::{client.model}"

        cache = self._load_cache()
        if key in cache:
            return MultimodalAppropriatenessJudgment.model_validate(cache[key])

        judgment = client.judge_audio(audio_bytes, context_text, MultimodalAppropriatenessJudgment)
        cache[key] = judgment.model_dump()
        self._save_cache(cache)
        return judgment

    def _load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(self, cache: dict) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
