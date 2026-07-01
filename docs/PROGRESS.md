# PROGRESS — VoxGate build checklist

> Legend: `[ ]` todo · `[~]` in progress · `[x]` done.
> Source of truth for "where we are." Tick after each completed step and move the ▶ pointer.
> Seeded from CLAUDE.md "Recommended build order" + "Definition of Done".

## ▶ Current step
**STEP 0 — Planning.** Working-memory files created. Build plan drafted and awaiting user
approval. **No source code may be written until the build plan is approved.**

---

## Phase 0 — Planning & working memory
- [x] Read CLAUDE.md, docs/assessment.md, docs/architecture.md
- [x] Create docs/PROGRESS.md, docs/ERRORS.md, docs/CONTEXT.md
- [~] Draft ordered build plan (files + tests per step)
- [ ] **GATE:** user approves build plan ← blocks all source code

## Phase 1 — Core contracts (build FIRST, lock interfaces)
- [ ] `metrics/base.py` — `MetricKind`, `Status`, `Gating` enums; `MetricScore`; `BaseMetric`
- [ ] `context/metric_context.py` — `MetricContext`, `Turn`, `ToolEvent`, `Event` dataclasses (types only, join logic later)
- [ ] `metrics/registry.py` — `REGISTRY`, `@register`, two-phase `run()`, `_safe()` isolation
- [ ] Test: contract shape / enum values / `MetricScore.key` idempotency tuple
- [ ] Test: registry `@register` drop-in adds a metric with zero runner edits
- [ ] Test: `_safe()` turns a crashing metric into `ERROR`, suite continues (C1(1))

## Phase 2 — Fixture format + synthetic fixtures
- [ ] Define fixture schema (caller.wav, events.jsonl, scenario_db.json, expected.json)
- [ ] Fixture: `happy_path_book` (clean booking)
- [ ] Fixture: `reschedule_trap` (hold-new → confirm → release-old + injected mid-sequence failure)
- [ ] Fixture: `barge_in_basic` (caller overlaps agent; + injected cough for false-yield test)
- [ ] Test: fixtures load + validate against the schema

## Phase 3 — Clock-join backbone (riskiest — test hardest)
- [ ] `context/metric_context.py` — build join: 2-ch audio + transcript + tool log + markers onto ONE clock (audio sample index @ sr)
- [ ] Test: known channel offset → alignment within tolerance
- [ ] Test: every time is seconds on the canonical clock; metrics never re-derive
- [ ] Test: optional `asr_confidence` absent → unchanged behavior (C1(6))

## Phase 4 — Semantic suite (deterministic before judge)
- [ ] `metrics/semantic/task_success.py` — deterministic (final DB state vs expected)
- [ ] `metrics/semantic/tool_call_ordering.py` — deterministic reducer + reschedule-trap invariant
- [ ] Test: reschedule_trap mid-sequence failure → "never zero appointments" fires (intermediate state)
- [ ] `metrics/semantic/instruction_adherence.py` — deterministic rule (gate) + judge stub (advisory)
- [ ] `metrics/semantic/faithfulness.py` — LLM judge (advisory; reads asr_confidence)
- [ ] Test: task_success / ordering pass+fail cases; faithfulness judge structured-output shape

## Phase 5 — Acoustic suite (barge_in first — headline)
- [ ] `metrics/acoustic/barge_in.py` — 2-ch VAD + markers → time-to-yield; fail-to-yield & false-yield
- [ ] Test: synthetic overlap → correct time-to-yield; injected cough → flagged false yield, not a yield
- [ ] `metrics/acoustic/turn_taking_latency.py` — gap distribution p50/p90/p99 (not just mean)
- [ ] `metrics/acoustic/latency_thresholds.py` — deterministic FTL + silence, advisory (C1(7))
- [ ] `metrics/acoustic/pitch_prosody.py` — F0 contour (range/monotone) + speech rate
- [ ] `metrics/acoustic/entity_intelligibility.py` — round-trip STT (faster-whisper) + WER (jiwer) on critical entities
- [ ] `metrics/acoustic/emotional_appropriateness.py` — multimodal judge, always advisory
- [ ] Test: turn_taking distribution; latency advisory does not move gate (C1(7)); entity WER on critical tokens

## Phase 6 — Calibration
- [ ] `calibration/judge_agreement.py` — Cohen's kappa vs small human set → trust tier
- [ ] `calibration/drift.py` — frozen golden set + KS test (fed by version stamps)
- [ ] Test: low-kappa judge must NOT be able to hard-gate

## Phase 7 — Gating + report
- [ ] `gating/gate.py` — two-tier; hard-gate deterministic+trusted; pass^k; tail thresholds
- [ ] `report/combine.py` — per-call + aggregate + det/judge/error-rate/agreement split; upsert on key
- [ ] Test: flaky judge cannot fail a good deploy; C1(2) re-run overwrites not duplicates
- [ ] Test: semantic + acoustic emit identical MetricScore schema into one report

## Phase 8 — Validators, sampling, monitoring
- [ ] `validators/` — preflight (channels, timeline parse, no clipping) + retry→quarantine (C1(4))
- [ ] `metrics/sampling.py` — stratified judge coverage; defaults 100% on fixtures (C1(5))
- [ ] `monitoring/production_proxies.py` — ground-truth-free subset (requires_ground_truth flag)
- [ ] Test: coverage 100% → verdict identical to pre-sampling (C1(5))

## Phase 9 — Docs & deliverables
- [ ] `README.md` — how to run + the *why* behind the structure
- [ ] Design writeup (~2pp) — all 5 Part-1 questions, opinionated
- [ ] Category-2 items as documented design decisions (no fake infra)
- [ ] Category-3 PHI/BAA paragraph after faithfulness deterministic-vs-judge justification
- [ ] Live-follow-up ready: `double_talk` drop-in OR emotion calibration vs 3 human clips

## Definition of Done (rubric mirror)
- [ ] Suite A runnable (task_success, tool_call_ordering catches trap, faithfulness, instruction_adherence)
- [ ] Suite B runnable (barge_in aligned, turn_taking distribution, pitch_prosody, emotion honest, entity_intelligibility)
- [ ] Registry: new metric drops in with zero runner edits
- [ ] Calibration: judge_agreement (kappa) + drift (golden + KS)
- [ ] Gating: two-tier pass^k; flaky judge cannot fail a good deploy
- [ ] Combined report + explicit gate-vs-advisory list with rationale
- [ ] README + design writeup + Category-2/3 notes
