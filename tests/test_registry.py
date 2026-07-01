import pytest

from eval_system.metrics import registry
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.context.metric_context import MetricContext


@pytest.fixture(autouse=True)
def clean_registry():
    """Each test declares its own metrics; don't leak between tests."""
    saved = registry.REGISTRY[:]
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY[:] = saved


def make_ctx(call_id="call-1") -> MetricContext:
    return MetricContext(
        call_id=call_id,
        sr=16000,
        audio_agent=None,
        audio_caller=None,
        transcript=[],
        tool_events=[],
        events=[],
        expected={},
        scenario_db={},
    )


def test_register_drop_in_adds_metric_with_zero_runner_edits():
    @registry.register
    class AlwaysPass(BaseMetric):
        name = "always_pass"
        version = "1"
        kind = MetricKind.DETERMINISTIC
        default_gating = Gating.GATE
        requires_ground_truth = False

        def compute(self, ctx):
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.PASS,
                gating=self.default_gating,
                score=1.0,
            )

    assert any(isinstance(m, AlwaysPass) for m in registry.REGISTRY)
    scores = registry.run(make_ctx())
    assert len(scores) == 1
    assert scores[0].status is Status.PASS
    assert scores[0].metric == "always_pass"


def test_crashing_metric_yields_error_and_suite_continues():
    @registry.register
    class Crashes(BaseMetric):
        name = "crashes"
        version = "1"
        kind = MetricKind.DETERMINISTIC
        default_gating = Gating.GATE
        requires_ground_truth = False

        def compute(self, ctx):
            raise ValueError("boom")

    @registry.register
    class Survives(BaseMetric):
        name = "survives"
        version = "1"
        kind = MetricKind.DETERMINISTIC
        default_gating = Gating.ADVISORY
        requires_ground_truth = False

        def compute(self, ctx):
            return MetricScore(
                call_id=ctx.call_id,
                metric=self.name,
                kind=self.kind,
                status=Status.PASS,
                gating=self.default_gating,
                score=1.0,
            )

    scores = registry.run(make_ctx())
    by_metric = {s.metric: s for s in scores}
    assert by_metric["crashes"].status is Status.ERROR
    assert "boom" in by_metric["crashes"].details["exc"]
    assert by_metric["survives"].status is Status.PASS  # suite continued


def test_judge_metrics_run_after_deterministic_and_respect_sampler():
    order: list[str] = []

    @registry.register
    class Det(BaseMetric):
        name = "det"
        version = "1"
        kind = MetricKind.DETERMINISTIC
        default_gating = Gating.GATE
        requires_ground_truth = False

        def compute(self, ctx):
            order.append(self.name)
            return MetricScore(ctx.call_id, self.name, self.kind, Status.PASS, self.default_gating, 1.0)

    @registry.register
    class Judge(BaseMetric):
        name = "judge"
        version = "1"
        kind = MetricKind.JUDGE
        default_gating = Gating.ADVISORY
        requires_ground_truth = False

        def compute(self, ctx):
            order.append(self.name)
            return MetricScore(ctx.call_id, self.name, self.kind, Status.PASS, self.default_gating, 1.0)

    class RejectAll:
        def should_run(self, metric, ctx, scores):
            return False

    scores = registry.run(make_ctx(), sampler=RejectAll())
    assert order == ["det"]  # judge never ran
    assert len(scores) == 1

    order.clear()
    scores = registry.run(make_ctx())  # no sampler -> full coverage
    assert order == ["det", "judge"]
    assert len(scores) == 2
