# CONTEXT — VoxGate compressed snapshot

> ≤1 page. REWRITE (don't append) to keep short. A cold session should resume from this alone.
> Read order each session: CONTEXT.md → PROGRESS.md → ERRORS.md.

## What this is
Post-hoc eval system for an EXISTING inbound clinic scheduling voice agent. Scores recorded
calls on two axes — **semantic** (tool calls, faithfulness, task success, adherence) and
**acoustic** (barge-in, latency, prosody, emotion, intelligibility) — and fuses them into a
two-tier **ship / don't-ship** verdict for CI, plus per-call + aggregate reports.

## Status: feature-complete
All 9 build phases are done, 134/134 tests passing. On branch `main` (build branch
`build/eval-system` fast-forward-merged in; both pushed). `uv` venv on Python 3.13.5, all
four extras (`acoustic`/`judge`/`stats`/`dev`) installed together.

- **Contracts + fixtures + clock-join (Phases 1-3):** `metrics/base.py`, `metrics/registry.py`
  (two-phase run + `_safe()` isolation + optional `metrics_filter`/`sampler`),
  `context/metric_context.py` (`build_metric_context()` — canonical clock = audio sample
  index @ sr; corrects a *known* per-channel capture skew, ground-truth timestamps already
  authored on that clock by the fixture loader). 3 fixtures (`happy_path_book`,
  `reschedule_trap`, `barge_in_basic`) with real Windows-SAPI TTS audio.
- **Semantic suite / Suite A (Phase 4):** `task_success`, `tool_call_ordering` (state
  reducer + `never_zero_appointments` invariant — catches the reschedule trap),
  `instruction_adherence` (rule + judge, two classes), `faithfulness` (judge). Judges depend
  on `judges/client.py`'s `JudgeClient` protocol; tests inject a fake, no network needed.
- **Acoustic suite / Suite B (Phase 5):** `barge_in` (headline; VAD via `metrics/acoustic/
  vad.py`'s shared `silero_vad_segments`), `turn_taking_latency` (p50/p90/p99), `latency_
  thresholds` (deterministic, advisory/C1(7)), `pitch_prosody` (F0 + rate), `entity_
  intelligibility` (round-trip STT + digit-normalized matching), `emotional_appropriateness`
  (text judge over transcript + prosody summary, always advisory), `double_talk`
  (live-follow-up drop-in).
- **Calibration (Phase 6):** `judge_agreement` (Cohen's kappa, degenerate data never
  "trusted"), `drift` (KS test vs. a frozen golden set).
- **Gating + report (Phase 7):** `gating/gate.py`'s `evaluate_gate()` (pass^k conjunction
  over gate-eligible metrics; ERROR fail-closed but distinct from FAIL; SKIPPED excluded from
  the conjunction) + `gate_advisory_breakdown()` (rationale list). `report/combine.py`
  upserts by `MetricScore.key` (C1(2)) and rolls up the aggregate split.
- **Validators/sampling/monitoring (Phase 8):** `validators/preflight.py` (channel/clipping/
  timeline checks + retry-then-quarantine, C1(4)), `metrics/sampling.py`
  (`StratifiedJudgeSampler`, defaults to full coverage, C1(5)), `monitoring/
  production_proxies.py` (ground-truth-free subset for live traffic).
- **CLI + docs (Phase 9):** `eval_system/run.py` (`python -m eval_system.run --fixtures
  fixtures/ --out out/ [--metrics a,b]`), `README.md`, `docs/design_writeup.md` (the graded
  Part-1 argument — all 5 questions, Category-2/3 notes, PHI/BAA paragraph after the
  faithfulness justification).

## Two real bugs this system caught in itself (see docs/ERRORS.md)
1. `barge_in_basic`'s "genuine barge-in" scenario had events authored at absolute times but
   no actual overlapping caller audio (rendered sequentially after the agent's cutoff, not
   concurrently) — fixed by adding `CallBuilder.say_at()` for absolute-time placement and
   regenerating the fixture. A live example of "never fabricate alignment data."
2. `entity_intelligibility`'s substring match false-failed on spoken-digit vs. numeral
   mismatches ("four eight two one three" vs. STT's "4-8213") on a real end-to-end CLI run —
   fixed with number-word normalization before comparing.

## Key decisions (locked)
- Open-loop, fixed-clock fixture replay only. No closed-loop bot-to-bot runner, no `agent/`.
- Canonical clock = audio sample index @ sr; metrics never re-derive it.
- One `MetricScore` contract; one `BaseMetric` = one `kind` (deterministic xor judge), never
  mixed in a single class — hence `instruction_adherence`'s two-class split.
- Gating trust rule: only deterministic + calibration-trusted judges may hard-gate.
  `emotional_appropriateness` is hardcoded never-eligible, not just defaulted advisory.
- Category-1 features are additive only. Category-2/3 (DLQ, distributed workers, key
  rotation, judge self-consistency, PHI/BAA) are writeup-only, documented in
  `design_writeup.md` §3-5 — no fake infra.

## Open items (non-blocking polish, not required by the rubric)
- No OpenAI judge client built (Anthropic only) — extra installed, unused; would only matter
  for a second-judge cross-check in calibration.
- `emotional_appropriateness`'s "no multimodal audio judge" gap is a deliberate, documented
  limitation (see design_writeup.md §3), not a TODO — fixing it would need a multimodal-
  audio-capable `JudgeClient` implementation, out of scope here.
