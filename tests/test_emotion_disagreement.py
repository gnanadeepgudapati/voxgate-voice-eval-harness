from eval_system.metrics.base import Gating, MetricKind, MetricScore, Status
from eval_system.report.combine import compute_emotion_disagreement


def _ser_score(per_turn, status=Status.PASS):
    return MetricScore(
        call_id="call-1", metric="ser_emotion", kind=MetricKind.SIGNAL, status=status,
        gating=Gating.ADVISORY, score=0.8, details={"per_turn": per_turn},
    )


def _mm_score(per_turn, status=Status.PASS):
    return MetricScore(
        call_id="call-1", metric="emotion_appropriateness_mm", kind=MetricKind.JUDGE, status=status,
        gating=Gating.ADVISORY, score=4.0, details={"per_turn": per_turn}, judge_prompt_version="mm-v1",
    )


def test_no_disagreement_when_valence_matches_and_appropriate():
    ser = _ser_score([{"start": 0.0, "end": 1.0, "label": "neu", "confidence": 0.9}])
    mm = _mm_score([{"turn_index": 0, "start": 0.0, "end": 1.0, "appropriate": True, "score": 5, "detected_tone": "calm"}])

    disagreements = compute_emotion_disagreement([ser, mm])

    assert disagreements == []


def test_disagreement_when_valence_mismatch():
    ser = _ser_score([{"start": 0.0, "end": 1.0, "label": "hap", "confidence": 0.9}])
    mm = _mm_score([{"turn_index": 0, "start": 0.0, "end": 1.0, "appropriate": True, "score": 4, "detected_tone": "sad"}])

    disagreements = compute_emotion_disagreement([ser, mm])

    assert disagreements == [{"turn": 0, "ser_label": "hap", "judge_tone": "sad", "judge_appropriate": True}]


def test_disagreement_when_ser_positive_but_judge_marks_inappropriate():
    # The exact "not chirpy when delivering bad news" scenario: SER hears
    # happy, the judge (with context) says that tone was wrong for the moment.
    ser = _ser_score([{"start": 0.0, "end": 1.0, "label": "hap", "confidence": 0.85}])
    mm = _mm_score([
        {"turn_index": 0, "start": 0.0, "end": 1.0, "appropriate": False, "score": 2, "detected_tone": "cheerful"}
    ])

    disagreements = compute_emotion_disagreement([ser, mm])

    assert disagreements == [{"turn": 0, "ser_label": "hap", "judge_tone": "cheerful", "judge_appropriate": False}]


def test_returns_empty_when_either_metric_missing():
    ser = _ser_score([{"start": 0.0, "end": 1.0, "label": "hap", "confidence": 0.9}])

    assert compute_emotion_disagreement([ser]) == []
    assert compute_emotion_disagreement([]) == []


def test_returns_empty_when_either_metric_errored():
    ser = _ser_score([{"start": 0.0, "end": 1.0, "label": "hap", "confidence": 0.9}], status=Status.ERROR)
    mm = _mm_score([{"turn_index": 0, "start": 0.0, "end": 1.0, "appropriate": False, "score": 2, "detected_tone": "cheerful"}])

    assert compute_emotion_disagreement([ser, mm]) == []


def test_turn_with_no_matching_ser_span_is_skipped_not_crashed():
    ser = _ser_score([{"start": 5.0, "end": 6.0, "label": "neu", "confidence": 0.9}])
    mm = _mm_score([{"turn_index": 0, "start": 0.0, "end": 1.0, "appropriate": True, "score": 5, "detected_tone": "calm"}])

    assert compute_emotion_disagreement([ser, mm]) == []
