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
  @register, two-phase run(), _safe() isolation).
- **Phase 2 done:** fixture schema (`eval_system/context/fixture_schema.py`, pydantic) +
  loader (`eval_system/context/fixture_loader.py` → `RawFixture`, channels split but NOT
  yet clock-joined — that's Phase 3). Fixture layout: `call.wav` (2-ch: ch0=caller,
  ch1=agent), `transcript.jsonl`, `tool_log.jsonl`, `events.jsonl`, `scenario_db.json`
  (initial DB state), `expected.json` (tool_sequence/invariants/critical_entities/
  success_criteria). 3 fixtures generated via `scripts/generate_fixtures.py` using
  real Windows SAPI TTS speech (22050Hz) — timestamps are the ACTUAL rendered-clip
  boundaries, never invented independently (per CLAUDE.md no-fabricated-alignment rule):
  `happy_path_book` (clean booking), `reschedule_trap` (release-old-before-secure-new +
  failed rebook → real zero-appointment window, marked with zero_appointments_start/end
  events, for tool_call_ordering to catch), `barge_in_basic` (authored false-yield-on-cough
  + prompt genuine-barge-in-yield, both marked via cough/barge_in_start/agent_yield events).
  13/13 tests passing across `tests/test_contracts.py`, `tests/test_registry.py`,
  `tests/test_fixtures.py`.

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
1. Phase 3 — clock-join backbone: build `RawFixture` → `MetricContext` in
   `metric_context.py` (riskiest; test hardest — known-offset alignment tolerance test).
2. Phase 4 — semantic suite, deterministic metrics first (task_success, tool_call_ordering
   incl. reschedule-trap invariant using the zero_appointments_start/end markers), then
   faithfulness judge.
3. Phase 5 — acoustic suite, barge_in first (headline metric; barge_in_basic fixture
   already encodes both a false-yield and a prompt-yield case to test against).

## Open questions
- Judge provider (Anthropic vs OpenAI) for faithfulness/emotion — default Anthropic per CLAUDE.md tech list.
- Whether to vendor tiny real audio clips or generate synthetic tones for fixtures (must be REAL 2-ch audio + timestamps per constraint; synthetic-but-real waveforms OK, fabricated alignment data NOT).
