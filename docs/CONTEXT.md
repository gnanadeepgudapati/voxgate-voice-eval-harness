# CONTEXT — VoxGate compressed snapshot

> ≤1 page. REWRITE (don't append) to keep short. A cold session should resume from this alone.
> Read order each session: CONTEXT.md → PROGRESS.md → ERRORS.md.

## What this is
Post-hoc eval system for an EXISTING inbound clinic scheduling voice agent. Scores recorded
calls on two axes — **semantic** (tool calls, faithfulness, task success, adherence) and
**acoustic** (barge-in, latency, prosody, emotion, intelligibility) — and fuses them into a
two-tier **ship / don't-ship** verdict for CI, plus per-call + aggregate reports.

## What exists now
- `CLAUDE.md` (decisions/constraints), `docs/assessment.md` (requirements), `docs/architecture.md` (structure).
- `docs/PROGRESS.md`, `docs/ERRORS.md`, `docs/CONTEXT.md` (this file) — working memory.
- Build plan approved. On branch `build/eval-system`. `uv` venv on Python 3.13.5
  (repo Python was 3.14, no `uv` installed — installed uv via pip, pinned venv to 3.13.5
  for ML wheel compatibility). Core deps only in the default install; `acoustic`/`judge`/
  `stats` are optional extras so core+semantic run without heavy libs.
- **Phase 1 done:** `eval_system/metrics/base.py` (MetricKind/Status/Gating/MetricScore/
  BaseMetric), `eval_system/context/metric_context.py` (Turn/ToolEvent/Event/MetricContext
  — types only, join logic is Phase 3), `eval_system/metrics/registry.py` (REGISTRY,
  @register, two-phase run(), _safe() isolation). 7/7 tests passing
  (`tests/test_contracts.py`, `tests/test_registry.py`).

## Key decisions (locked)
- **Open-loop, fixed-clock fixture replay only.** No closed-loop bot-to-bot runner.
- **No `agent/` module** — agent is external; we replay/score, never build it.
- **Canonical clock = audio sample index @ sr.** Every acoustic metric reads times from
  `MetricContext`, never re-derives. Clock-join is the backbone → test first & hardest.
- **One contract:** every metric returns `MetricScore` (status/gating/versions/idempotency key).
- **Gating trust rule:** only deterministic + calibration-trusted metrics may hard-gate.
  Judges start ADVISORY until kappa ≥ threshold. Emotion is advisory always.
- **Category-1 features are additive only** — never rewrite a metric, the clock-join, or the gate.
- **Category-2/3** (DLQ, distributed workers, key rotation, judge self-consistency, PHI/BAA)
  are writeup-only. No fake infra.
- **Reschedule trap:** `tool_call_ordering` must catch the INTERMEDIATE zero-appointment state
  (new slot secured before old released), not just final order.

## Build order (see PROGRESS.md for the live checklist)
Contracts → fixtures → clock-join(+tests) → semantic (det→judge) → acoustic (barge_in first)
→ calibration → gating+report → validators/sampling/monitoring → docs.

## Next 3 steps
1. Phase 2 — fixture format (caller.wav, events.jsonl, scenario_db.json, expected.json)
   + `happy_path_book`, `reschedule_trap`, `barge_in_basic` fixtures + schema-load test.
2. Phase 3 — clock-join backbone in `metric_context.py` (riskiest; test hardest).
3. Phase 4 — semantic suite, deterministic metrics first (task_success, tool_call_ordering
   incl. reschedule-trap invariant), then faithfulness judge.

## Open questions
- Judge provider (Anthropic vs OpenAI) for faithfulness/emotion — default Anthropic per CLAUDE.md tech list.
- Whether to vendor tiny real audio clips or generate synthetic tones for fixtures (must be REAL 2-ch audio + timestamps per constraint; synthetic-but-real waveforms OK, fabricated alignment data NOT).
