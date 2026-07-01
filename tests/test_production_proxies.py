import pytest

from eval_system.context.metric_context import MetricContext
from eval_system.metrics import registry
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.monitoring.production_proxies import ground_truth_free_metrics, run_production_monitoring


@pytest.fixture
def clean_registry():
    saved = registry.REGISTRY[:]
    registry.REGISTRY.clear()
    yield
    registry.REGISTRY[:] = saved


def make_ctx() -> MetricContext:
    return MetricContext(
        call_id="call-1", sr=16000, audio_agent=None, audio_caller=None,
        transcript=[], tool_events=[], events=[], expected={}, scenario_db={},
    )


class NeedsGroundTruth(BaseMetric):
    name = "needs_gt"
    version = "1"
    kind = MetricKind.DETERMINISTIC
    default_gating = Gating.GATE
    requires_ground_truth = True

    def compute(self, ctx):
        return MetricScore(ctx.call_id, self.name, self.kind, Status.PASS, self.default_gating, 1.0)


class LiveSafe(BaseMetric):
    name = "live_safe"
    version = "1"
    kind = MetricKind.SIGNAL
    default_gating = Gating.ADVISORY
    requires_ground_truth = False

    def compute(self, ctx):
        return MetricScore(ctx.call_id, self.name, self.kind, Status.PASS, self.default_gating, 1.0)


def test_ground_truth_free_metrics_excludes_flagged_metrics():
    metrics = [NeedsGroundTruth(), LiveSafe()]

    filtered = ground_truth_free_metrics(metrics)

    assert [m.name for m in filtered] == ["live_safe"]


def test_run_production_monitoring_only_runs_ground_truth_free_subset(clean_registry):
    registry.register(NeedsGroundTruth)
    registry.register(LiveSafe)

    scores = run_production_monitoring(make_ctx())

    assert {s.metric for s in scores} == {"live_safe"}


def test_real_registry_flags_align_with_monitoring_intent():
    from eval_system.metrics.acoustic import barge_in  # noqa: F401
    from eval_system.metrics.semantic import task_success  # noqa: F401

    names = {m.name: m.requires_ground_truth for m in registry.REGISTRY}

    assert names["barge_in"] is False  # safe on live production audio
    assert names["task_success"] is True  # needs expected.json, fixtures-only
