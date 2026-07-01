from pathlib import Path

import numpy as np

from eval_system.context.fixture_loader import load_fixture
from eval_system.context.metric_context import MetricContext, build_metric_context
from eval_system.metrics.acoustic.barge_in import (
    BargeInMetric,
    SpeechSegment,
    find_barge_ins,
)
from eval_system.metrics.base import Gating, MetricKind, Status

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


# --- pure decision-logic tests: no VAD model, synthetic segments only ---

def test_genuine_overlap_measures_time_to_yield():
    caller = [SpeechSegment(t_start=2.0, t_end=3.5)]
    agent = [SpeechSegment(t_start=0.0, t_end=2.45)]

    barge_ins = find_barge_ins(caller, agent)

    assert len(barge_ins) == 1
    assert barge_ins[0]["t_onset"] == 2.0
    assert round(barge_ins[0]["time_to_yield"], 2) == 0.45
    assert barge_ins[0]["fail_to_yield"] is False
    assert barge_ins[0]["false_yield"] is False


def test_fail_to_yield_flagged_when_agent_keeps_talking_past_threshold():
    caller = [SpeechSegment(t_start=1.0, t_end=4.0)]
    agent = [SpeechSegment(t_start=0.0, t_end=3.0)]  # yields 2s after onset

    barge_ins = find_barge_ins(caller, agent, fail_to_yield_threshold_sec=1.0)

    assert barge_ins[0]["fail_to_yield"] is True


def test_short_noise_burst_overlap_is_flagged_as_false_yield_not_genuine():
    caller = [SpeechSegment(t_start=1.5, t_end=1.75)]  # 0.25s: a cough, not real speech
    agent = [SpeechSegment(t_start=0.3, t_end=1.8)]

    barge_ins = find_barge_ins(caller, agent, min_genuine_speech_duration_sec=0.3)

    assert len(barge_ins) == 1
    assert barge_ins[0]["false_yield"] is True
    assert barge_ins[0]["fail_to_yield"] is False  # false yields aren't scored for fail-to-yield


def test_overlap_exactly_on_the_boundary_still_counts():
    # VAD segment boundaries are rounded (silero's time_resolution), so a real
    # overlap can land exactly on the boundary between two segments -- must
    # still be treated as a barge-in, not silently dropped by a strict "<".
    caller = [SpeechSegment(t_start=8.5, t_end=9.2)]
    agent = [SpeechSegment(t_start=8.0, t_end=8.5)]

    barge_ins = find_barge_ins(caller, agent)

    assert len(barge_ins) == 1
    assert barge_ins[0]["time_to_yield"] == 0.0


def test_caller_onset_just_after_agent_end_within_tolerance_still_counts():
    # Real VAD has onset/offset latency (padding, silence thresholds), so a
    # genuine overlap can show up as the caller "starting" a few tens of ms
    # after the agent's segment "ends" per VAD's own boundaries -- must still
    # count as a barge-in within a small tolerance, not a missed overlap.
    caller = [SpeechSegment(t_start=8.514, t_end=9.182)]
    agent = [SpeechSegment(t_start=7.97, t_end=8.478)]

    barge_ins = find_barge_ins(caller, agent, overlap_tolerance_sec=0.2)

    assert len(barge_ins) == 1
    assert barge_ins[0]["time_to_yield"] == 0.0  # clamped, agent had already yielded


def test_caller_onset_well_after_agent_end_is_not_an_overlap():
    caller = [SpeechSegment(t_start=9.0, t_end=9.5)]
    agent = [SpeechSegment(t_start=7.97, t_end=8.478)]

    barge_ins = find_barge_ins(caller, agent, overlap_tolerance_sec=0.2)

    assert barge_ins == []


def test_no_overlap_yields_no_barge_ins():
    caller = [SpeechSegment(t_start=5.0, t_end=5.5)]  # after agent already finished
    agent = [SpeechSegment(t_start=0.0, t_end=2.0)]

    barge_ins = find_barge_ins(caller, agent)

    assert barge_ins == []


# --- metric-level tests: inject a fake VAD so no model runs ---

def _fake_vad(segments_by_audio_id):
    def vad(audio, sr):
        return segments_by_audio_id[id(audio)]

    return vad


def test_metric_passes_when_no_barge_in_issues():
    caller_audio = np.zeros(10)
    agent_audio = np.zeros(10)
    vad = _fake_vad({
        id(caller_audio): [SpeechSegment(2.0, 3.5)],
        id(agent_audio): [SpeechSegment(0.0, 2.45)],
    })
    ctx = MetricContext(
        call_id="call-1", sr=16000, audio_agent=agent_audio, audio_caller=caller_audio,
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )

    score = BargeInMetric(vad_fn=vad).compute(ctx)

    assert score.status is Status.PASS
    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.GATE
    assert score.details["barge_ins"][0]["fail_to_yield"] is False


def test_metric_fails_when_fail_to_yield_present():
    caller_audio = np.zeros(10)
    agent_audio = np.zeros(10)
    vad = _fake_vad({
        id(caller_audio): [SpeechSegment(1.0, 4.0)],
        id(agent_audio): [SpeechSegment(0.0, 3.0)],
    })
    ctx = MetricContext(
        call_id="call-1", sr=16000, audio_agent=agent_audio, audio_caller=caller_audio,
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )

    score = BargeInMetric(vad_fn=vad).compute(ctx)

    assert score.status is Status.FAIL


# --- one integration smoke test: real VAD, real audio, shape/type only ---

def test_real_fixture_runs_end_to_end_with_real_vad():
    fixture = load_fixture(FIXTURES_DIR / "barge_in_basic")
    ctx = build_metric_context(fixture)

    score = BargeInMetric().compute(ctx)

    assert score.kind is MetricKind.SIGNAL
    assert score.gating is Gating.GATE
    assert score.status in (Status.PASS, Status.FAIL)
    assert isinstance(score.details["barge_ins"], list)
