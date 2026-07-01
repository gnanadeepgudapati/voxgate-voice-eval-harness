# PROGRESS — VoxGate build checklist

> Legend: `[ ]` todo · `[~]` in progress · `[x]` done.
> Source of truth for "where we are." Tick after each completed step and move the ▶ pointer.
> Seeded from CLAUDE.md "Recommended build order" + "Definition of Done".

## ▶ Current step
**All phases (0-9) complete.** 134/134 tests passing. Both suites, calibration, gating,
report, validators/sampling/monitoring, CLI runner (`eval_system/run.py`), README, and
design writeup are all in place. See `docs/CONTEXT.md` for the compressed current-state
snapshot and remaining polish ideas (none blocking).

---

## Phase 0 — Planning & working memory
- [x] Read CLAUDE.md, docs/assessment.md, docs/architecture.md
- [x] Create docs/PROGRESS.md, docs/ERRORS.md, docs/CONTEXT.md
- [x] Draft ordered build plan (files + tests per step)
- [x] **GATE:** user approves build plan ← blocks all source code

## Phase 1 — Core contracts (build FIRST, lock interfaces)
- [x] `metrics/base.py` — `MetricKind`, `Status`, `Gating` enums; `MetricScore`; `BaseMetric`
- [x] `context/metric_context.py` — `MetricContext`, `Turn`, `ToolEvent`, `Event` dataclasses (types only, join logic later)
- [x] `metrics/registry.py` — `REGISTRY`, `@register`, two-phase `run()`, `_safe()` isolation
- [x] Test: contract shape / enum values / `MetricScore.key` idempotency tuple
- [x] Test: registry `@register` drop-in adds a metric with zero runner edits
- [x] Test: `_safe()` turns a crashing metric into `ERROR`, suite continues (C1(1))

## Phase 2 — Fixture format + synthetic fixtures
- [x] Define fixture schema (call.wav 2-ch, transcript/tool_log/events .jsonl, scenario_db/expected .json)
- [x] Fixture: `happy_path_book` (clean booking)
- [x] Fixture: `reschedule_trap` (release-old before secure-new + failed rebook → zero-appointment window; catches the trap)
- [x] Fixture: `barge_in_basic` (false yield on cough + prompt yield on genuine barge-in)
- [x] Test: fixtures load + validate against the schema (`tests/test_fixtures.py`, 6 tests)

## Phase 3 — Clock-join backbone (riskiest — test hardest)
- [x] `context/metric_context.py` — build join: 2-ch audio + transcript + tool log + markers onto ONE clock (audio sample index @ sr)
- [x] Test: known channel offset → alignment within tolerance
- [x] Test: every time is seconds on the canonical clock; metrics never re-derive
- [x] Test: optional `asr_confidence` absent → unchanged behavior (C1(6))

## Phase 4 — Semantic suite (deterministic before judge)
- [x] `metrics/semantic/task_success.py` — deterministic (final DB state vs expected)
- [x] `metrics/semantic/tool_call_ordering.py` — deterministic reducer + reschedule-trap invariant
- [x] Test: reschedule_trap mid-sequence failure → "never zero appointments" fires (intermediate state)
- [x] `metrics/semantic/instruction_adherence.py` — deterministic rule (gate) + judge (advisory), two classes
- [x] `metrics/semantic/faithfulness.py` — LLM judge (advisory; reads asr_confidence)
- [x] Test: task_success / ordering pass+fail cases
- [x] Test: faithfulness judge structured-output shape (via `JudgeClient` protocol + fake in tests)

## Phase 5 — Acoustic suite (barge_in first — headline)
- [x] `metrics/acoustic/barge_in.py` — 2-ch VAD (silero) + overlap logic → time-to-yield; fail-to-yield & false-yield
- [x] Test: synthetic overlap → correct time-to-yield; injected cough → flagged false yield, not a yield
- [x] `metrics/acoustic/turn_taking_latency.py` — gap distribution p50/p90/p99 (not just mean)
- [x] `metrics/acoustic/latency_thresholds.py` — deterministic FTL + silence, advisory (C1(7))
- [x] `metrics/acoustic/pitch_prosody.py` — F0 contour (range/monotone) + speech rate
- [x] `metrics/acoustic/entity_intelligibility.py` — round-trip STT (faster-whisper) + WER (jiwer) on critical entities
- [x] `metrics/acoustic/emotional_appropriateness.py` — text judge over transcript + prosody summary (no multimodal-audio client wired), always advisory
- [x] Test: turn_taking distribution; latency advisory does not move gate (C1(7)); entity WER on critical tokens

## Phase 6 — Calibration
- [x] `calibration/judge_agreement.py` — Cohen's kappa vs small human set → trust tier
- [x] `calibration/drift.py` — frozen golden set + KS test (fed by version stamps)
- [x] Test: low-kappa judge must NOT be able to hard-gate (degenerate/chance-level data → `trusted=False`)

## Phase 7 — Gating + report
- [x] `gating/gate.py` — two-tier; hard-gate deterministic+trusted; pass^k; tail thresholds; `gate_advisory_breakdown()` (DoD rationale list)
- [x] `report/combine.py` — per-call + aggregate + det/judge/error-rate/agreement split; upsert on key
- [x] Test: flaky judge cannot fail a good deploy; C1(2) re-run overwrites not duplicates
- [x] Test: semantic + acoustic emit identical MetricScore schema into one report (via shared `evaluate_gate`/`build_report`)

## Phase 8 — Validators, sampling, monitoring
- [x] `validators/` — preflight (channels, timeline parse, no clipping) + retry→quarantine (C1(4))
- [x] `metrics/sampling.py` — stratified judge coverage; defaults 100% on fixtures (C1(5))
- [x] `monitoring/production_proxies.py` — ground-truth-free subset (requires_ground_truth flag)
- [x] Test: coverage 100% → verdict identical to pre-sampling (C1(5))

## Phase 9 — Docs & deliverables
- [x] `README.md` — how to run + the *why* behind the structure
- [x] Design writeup (~2pp) — `docs/design_writeup.md`, all 5 Part-1 questions, opinionated
- [x] Category-2 items as documented design decisions (no fake infra) — design_writeup.md §4/§5
- [x] Category-3 PHI/BAA paragraph after faithfulness deterministic-vs-judge justification — design_writeup.md §3
- [x] Live-follow-up ready: `double_talk` drop-in (registered with zero runner edits, reuses the VAD seam)

## Definition of Done (rubric mirror)
- [x] Suite A runnable (task_success, tool_call_ordering catches trap, faithfulness, instruction_adherence)
- [x] Suite B runnable (barge_in aligned, turn_taking distribution, pitch_prosody, emotion honest, entity_intelligibility)
- [x] Registry: new metric drops in with zero runner edits (proven by `double_talk` + sampling/breakdown tests)
- [x] Calibration: judge_agreement (kappa) + drift (golden + KS)
- [x] Gating: two-tier pass^k; flaky judge cannot fail a good deploy
- [x] Combined report + explicit gate-vs-advisory list with rationale
- [x] README + design writeup + Category-2/3 notes
