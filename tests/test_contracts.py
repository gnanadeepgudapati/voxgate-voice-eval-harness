from eval_system.metrics.base import (
    BaseMetric,
    Gating,
    MetricKind,
    MetricScore,
    Status,
)


def test_enum_values_locked():
    assert {k.value for k in MetricKind} == {"deterministic", "judge", "signal"}
    assert {s.value for s in Status} == {"pass", "fail", "error", "skipped"}
    assert {g.value for g in Gating} == {"gate", "advisory"}


def test_metric_score_defaults():
    score = MetricScore(
        call_id="call-1",
        metric="task_success",
        kind=MetricKind.DETERMINISTIC,
        status=Status.PASS,
        gating=Gating.GATE,
        score=1.0,
    )
    assert score.details == {}
    assert score.judge_prompt_version is None
    assert score.schema_version == "1"


def test_key_idempotency_tuple_identifies_a_rerun():
    first = MetricScore(
        call_id="call-1",
        metric="faithfulness",
        kind=MetricKind.JUDGE,
        status=Status.PASS,
        gating=Gating.ADVISORY,
        score=0.9,
        evaluator_version="1",
        judge_prompt_version="v2",
    )
    rerun = MetricScore(
        call_id="call-1",
        metric="faithfulness",
        kind=MetricKind.JUDGE,
        status=Status.PASS,
        gating=Gating.ADVISORY,
        score=0.95,  # score changed on rerun
        evaluator_version="1",
        judge_prompt_version="v2",
    )
    different_version = MetricScore(
        call_id="call-1",
        metric="faithfulness",
        kind=MetricKind.JUDGE,
        status=Status.PASS,
        gating=Gating.ADVISORY,
        score=0.9,
        evaluator_version="2",
        judge_prompt_version="v2",
    )
    assert first.key == rerun.key  # same key despite differing score -> upsert target
    assert first.key != different_version.key


def test_base_metric_compute_is_abstract_contract():
    class Stub(BaseMetric):
        name = "stub"
        version = "1"
        kind = MetricKind.DETERMINISTIC
        default_gating = Gating.GATE
        requires_ground_truth = False

    stub = Stub()
    try:
        stub.compute(ctx=None)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
