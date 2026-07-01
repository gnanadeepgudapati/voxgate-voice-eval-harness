from eval_system.gating.gate import gate_advisory_breakdown
from eval_system.metrics import registry

# Import every metric module so it's registered -- side effect of @register.
from eval_system.metrics.semantic import task_success, tool_call_ordering, instruction_adherence, faithfulness  # noqa: F401
from eval_system.metrics.acoustic import barge_in, turn_taking_latency, latency_thresholds, pitch_prosody  # noqa: F401
from eval_system.metrics.acoustic import entity_intelligibility, emotional_appropriateness, double_talk  # noqa: F401


def test_breakdown_covers_every_registered_metric_with_a_rationale():
    breakdown = gate_advisory_breakdown(registry.REGISTRY)
    by_metric = {row["metric"]: row for row in breakdown}

    for expected_metric in [
        "task_success", "tool_call_ordering", "instruction_adherence_rule",
        "instruction_adherence_judge", "faithfulness", "barge_in",
        "turn_taking_latency", "latency_thresholds", "pitch_prosody",
        "entity_intelligibility", "emotional_appropriateness", "double_talk",
    ]:
        assert expected_metric in by_metric, f"missing {expected_metric}"
        assert by_metric[expected_metric]["rationale"], f"no rationale for {expected_metric}"


def test_breakdown_matches_claude_md_taxonomy_for_a_few_spot_checks():
    breakdown = gate_advisory_breakdown(registry.REGISTRY)
    by_metric = {row["metric"]: row for row in breakdown}

    assert by_metric["task_success"]["default_gating"] == "gate"
    assert by_metric["faithfulness"]["default_gating"] == "advisory"
    assert by_metric["barge_in"]["default_gating"] == "gate"
    assert by_metric["emotional_appropriateness"]["default_gating"] == "advisory"
    assert by_metric["entity_intelligibility"]["default_gating"] == "gate"
