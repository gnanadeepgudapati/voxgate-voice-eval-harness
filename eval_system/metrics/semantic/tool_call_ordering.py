"""Deterministic state reducer: right tool/args/sequence, plus the
reschedule-trap invariant (never_zero_appointments). The invariant is checked
by replaying tool_events against the initial scenario_db appointment counts,
so it catches the INTERMEDIATE zero-appointment window (e.g. old released
before new is secured), not just the final state."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval_system.context.metric_context import ToolEvent
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

if TYPE_CHECKING:
    from eval_system.context.metric_context import MetricContext


def _tool_succeeded(te: ToolEvent) -> bool:
    return not (isinstance(te.result, dict) and te.result.get("status") == "error")


def _initial_appointment_counts(scenario_db: dict) -> dict[str, int]:
    return {
        patient_id: len(patient.get("appointments", []))
        for patient_id, patient in scenario_db.get("patients", {}).items()
    }


def _resolve_patient(scenario_db: dict, te: ToolEvent) -> str | None:
    patient_id = te.args.get("patient_id")
    if patient_id is not None:
        return patient_id
    appointment_id = te.args.get("appointment_id")
    for pid, patient in scenario_db.get("patients", {}).items():
        if appointment_id in patient.get("appointments", []):
            return pid
    return None


@register
class ToolCallOrderingMetric(BaseMetric):
    name = "tool_call_ordering"
    version = "1"
    kind = MetricKind.DETERMINISTIC
    default_gating = Gating.GATE
    requires_ground_truth = True

    def compute(self, ctx: "MetricContext") -> MetricScore:
        order_ok, order_details = self._check_sequence_order(ctx)
        invariants_ok, invariant_details = self._check_invariants(ctx)

        status = Status.PASS if order_ok and invariants_ok else Status.FAIL
        return MetricScore(
            call_id=ctx.call_id,
            metric=self.name,
            kind=self.kind,
            status=status,
            gating=self.default_gating,
            score=1.0 if status is Status.PASS else 0.0,
            details={"sequence_order": order_details, "invariants": invariant_details},
            evaluator_version=self.version,
        )

    def _check_sequence_order(self, ctx: "MetricContext") -> tuple[bool, dict[str, Any]]:
        expected_sequence = ctx.expected.get("tool_sequence", [])
        cursor = 0
        missing = []
        for step in expected_sequence:
            required_args = step.get("required_args", {})
            match_index = None
            for i in range(cursor, len(ctx.tool_events)):
                te = ctx.tool_events[i]
                if te.name == step["name"] and all(
                    te.args.get(k) == v for k, v in required_args.items()
                ):
                    match_index = i
                    break
            if match_index is None:
                missing.append(step)
            else:
                cursor = match_index + 1
        return (not missing), {"missing_or_out_of_order": missing}

    def _check_invariants(self, ctx: "MetricContext") -> tuple[bool, dict[str, Any]]:
        invariants = ctx.expected.get("invariants", [])
        if "never_zero_appointments" not in invariants:
            return True, {}

        counts = _initial_appointment_counts(ctx.scenario_db)
        violations = []
        for te in ctx.tool_events:
            if not _tool_succeeded(te):
                continue
            patient_id = _resolve_patient(ctx.scenario_db, te)
            if patient_id is None:
                continue
            if te.name == "book_appointment":
                counts[patient_id] = counts.get(patient_id, 0) + 1
            elif te.name == "cancel_appointment":
                counts[patient_id] = counts.get(patient_id, 0) - 1
                if counts[patient_id] <= 0:
                    violations.append({"tool_event_t": te.t, "patient_id": patient_id})
        return (not violations), {"never_zero_appointments_violations": violations}
