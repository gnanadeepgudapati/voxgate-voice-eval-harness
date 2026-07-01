import pytest

from eval_system.context.metric_context import MetricContext
from eval_system.metrics import registry
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.sampling import StratifiedJudgeSampler


@pytest.fixture(autouse=True)
def clean_registry():
    saved = registry.REGISTRY[:]
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY[:] = saved


def make_ctx(call_id="call-1") -> MetricContext:
    return MetricContext(
        call_id=call_id, sr=16000, audio_agent=None, audio_caller=None,
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )


class DummyJudge(BaseMetric):
    name = "dummy_judge"
    version = "1"
    kind = MetricKind.JUDGE
    default_gating = Gating.ADVISORY
    requires_ground_truth = False

    def compute(self, ctx):
        return MetricScore(ctx.call_id, self.name, self.kind, Status.PASS, self.default_gating, 1.0)


def test_full_coverage_always_runs():
    sampler = StratifiedJudgeSampler(sample_rate=1.0)

    assert sampler.should_run(DummyJudge(), make_ctx(), []) is True


def test_zero_coverage_never_runs_absent_a_flag():
    sampler = StratifiedJudgeSampler(sample_rate=0.0)

    assert sampler.should_run(DummyJudge(), make_ctx(), []) is False


def test_flagged_calls_always_sampled_regardless_of_rate():
    sampler = StratifiedJudgeSampler(sample_rate=0.0, always_sample_calls=frozenset({"call-1"}))

    assert sampler.should_run(DummyJudge(), make_ctx("call-1"), []) is True


def test_deterministic_gate_failure_forces_judge_sampling():
    sampler = StratifiedJudgeSampler(sample_rate=0.0)
    prior_scores = [
        MetricScore("call-1", "task_success", MetricKind.DETERMINISTIC, Status.FAIL, Gating.GATE, 0.0)
    ]

    assert sampler.should_run(DummyJudge(), make_ctx(), prior_scores) is True


def test_sampling_decision_is_deterministic_given_seed():
    sampler_a = StratifiedJudgeSampler(sample_rate=0.5, seed=42)
    sampler_b = StratifiedJudgeSampler(sample_rate=0.5, seed=42)

    decisions_a = [sampler_a.should_run(DummyJudge(), make_ctx(f"call-{i}"), []) for i in range(20)]
    decisions_b = [sampler_b.should_run(DummyJudge(), make_ctx(f"call-{i}"), []) for i in range(20)]

    assert decisions_a == decisions_b
    assert any(decisions_a) and not all(decisions_a)  # a real mix, not degenerate


def test_full_coverage_sampler_matches_no_sampler_verdict():
    @registry.register
    class Det(BaseMetric):
        name = "det"
        version = "1"
        kind = MetricKind.DETERMINISTIC
        default_gating = Gating.GATE
        requires_ground_truth = False

        def compute(self, ctx):
            return MetricScore(ctx.call_id, self.name, self.kind, Status.PASS, self.default_gating, 1.0)

    registry.register(DummyJudge)

    no_sampler_scores = registry.run(make_ctx())
    full_coverage_scores = registry.run(make_ctx(), sampler=StratifiedJudgeSampler(sample_rate=1.0))

    assert {s.metric for s in no_sampler_scores} == {s.metric for s in full_coverage_scores} == {"det", "dummy_judge"}
