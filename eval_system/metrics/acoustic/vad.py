"""Shared 2-channel VAD used by every acoustic metric that reasons over
speech/silence segments (barge_in, turn_taking_latency, ...). One
implementation, one swappable seam (`VadFn`) -- metrics inject a fake for
unit tests and fall back to the real model for the one smoke test each runs
against real fixture audio."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

VAD_SAMPLE_RATE = 16000


@dataclass
class SpeechSegment:
    t_start: float
    t_end: float


VadFn = Callable[[np.ndarray, int], list[SpeechSegment]]


def silero_vad_segments(audio: np.ndarray, sr: int) -> list[SpeechSegment]:
    import librosa
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    resampled = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=VAD_SAMPLE_RATE)
    model = load_silero_vad()
    timestamps = get_speech_timestamps(
        torch.from_numpy(resampled), model, sampling_rate=VAD_SAMPLE_RATE,
        return_seconds=True, time_resolution=3,
    )
    return [SpeechSegment(t["start"], t["end"]) for t in timestamps]
