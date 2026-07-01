"""Non-intrusive MOS (Mean Opinion Score) naturalness prediction --
assessment.md line 61 explicitly points at the "non-intrusive-MOS family";
no naturalness metric existed before this one. "Non-intrusive" means it
needs no reference/clean audio to compare against (unlike PESQ/POLQA),
which fits this system since there's no "clean" ground-truth waveform for a
real synthesized call.

Engine note: the spec for this metric named UTMOS via the `speechmos` PyPI
package (`from speechmos import utmos22_strong`) as primary. That import
does not exist in the published package -- only `dnsmos`/`aecmos`/`plcmos`
ship (verified empirically before writing this). Uses DNSMOS (P.808)
instead, explicitly named as an acceptable fallback in the same spec, and
it's already bundled (ONNX models ship with the package -- no extra
download or network call needed). `details["mos_engine"]` records which
engine actually ran, so a future real-UTMOS swap-in stays visible.

Segments the AGENT channel by the same authored transcript turn boundaries
`ser_emotion.py`/`emotion_appropriateness_mm.py` use, for consistent
cross-metric alignment.

GATING is always advisory, hardcoded -- MOS is a perceptual proxy that
saturates above ~4 and can't distinguish "good" from "excellent" well (see
docs/design_writeup.md §3). Status still reflects a dissatisfaction flag
(mean MOS < 3.5) so a real quality problem is visible in the report, exactly
like pitch_prosody's Status.FAIL/Gating.ADVISORY split -- it just can never
block a ship."""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext

MOS_MODEL = "dnsmos"
MOS_SAMPLE_RATE = 16000
GOOD_MOS_THRESHOLD = 4.0
ACCEPTABLE_MOS_THRESHOLD = 3.5

MosFn = Callable[[np.ndarray, int], float]


def _dnsmos_score(audio: np.ndarray, sr: int) -> float:
    import librosa
    from speechmos import dnsmos

    resampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=MOS_SAMPLE_RATE)
    result = dnsmos.run(resampled, sr=MOS_SAMPLE_RATE)
    return float(result["p808_mos"])


def mos_band(mean_mos: float | None) -> str | None:
    if mean_mos is None:
        return None
    if mean_mos >= GOOD_MOS_THRESHOLD:
        return "good"
    if mean_mos >= ACCEPTABLE_MOS_THRESHOLD:
        return "acceptable"
    return "dissatisfaction"


def _slice_turn_audio(audio_agent: np.ndarray, sr: int, t_start: float, t_end: float) -> np.ndarray:
    start_sample = max(0, round(t_start * sr))
    end_sample = min(len(audio_agent), round(t_end * sr))
    return audio_agent[start_sample:end_sample]


@register
class NaturalnessMosMetric(BaseMetric):
    name = "naturalness_mos"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.ADVISORY
    requires_ground_truth = False
    mos_engine = MOS_MODEL

    def __init__(self, mos_fn: MosFn | None = None):
        self.mos_fn = mos_fn or _dnsmos_score

    def compute(self, ctx: "MetricContext") -> MetricScore:
        if ctx.audio_agent is None or len(ctx.audio_agent) == 0:
            raise ValueError("naturalness_mos: no agent audio available")

        agent_turns = [t for t in ctx.transcript if t.speaker == "agent"]
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
            )

        per_turn = []
        for turn in agent_turns:
            segment = _slice_turn_audio(ctx.audio_agent, ctx.sr, turn.t_start, turn.t_end)
            mos = self.mos_fn(segment, ctx.sr)
            per_turn.append({"start": turn.t_start, "end": turn.t_end, "mos": mos})

        mean_mos = sum(pt["mos"] for pt in per_turn) / len(per_turn)
        band = mos_band(mean_mos)

        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=Status.FAIL if band == "dissatisfaction" else Status.PASS,
            gating=self.default_gating,
            score=mean_mos,
            details={"per_turn": per_turn, "mean_mos": mean_mos, "mos_engine": self.mos_engine, "band": band},
            evaluator_version=self.version,
        )
