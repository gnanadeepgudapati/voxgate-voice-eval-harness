# CONTEXT — VoxGate compressed snapshot

> ≤1 page. REWRITE (don't append) to keep short. A cold session should resume from this alone.
> Read order each session: CONTEXT.md → PROGRESS.md → ERRORS.md.

## What this is
Post-hoc eval system for an EXISTING inbound clinic scheduling voice agent. Scores recorded
calls on two axes — **semantic** and **acoustic** — and fuses them into a two-tier
**ship / don't-ship** verdict for CI, plus per-call + aggregate + human-readable reports.

## Status: feature-complete, actively extended beyond the original 9-phase build
Full suite passing (`uv run pytest -q`, no failures). Currently on `build/eval-system` branch
(per gitStatus at session start; earlier CONTEXT said "single branch now" — verify with
`git branch` before assuming).
`uv` venv, Python 3.13.5, all extras (`acoustic`/`judge`/`stats`/`dev`) installed together.
`docs/PROGRESS.md` still shows the original Phase 0-9 checklist (all done) — everything
below is what's been added since that checklist was last accurate.

## In progress right now: report presentation + README (Deliverable 2) polish pass
Explicit constraint for this whole pass: **no metric logic, thresholds, gating, or
MetricScore schema changes — presentation + docs only.**
- **Done:** `report/html_report.py` rewritten — static `report.md`/`.html`/`.pdf` (versioned
  `report_<n>` fully removed), clean CSS badges (no Unicode — xhtml2pdf renders Unicode
  glyphs as garbled boxes), inline-style zebra striping (`tr:nth-child` doesn't render in
  xhtml2pdf), `<colgroup>` column widths on every table incl. the 3 aggregate tables, title=
  tooltip + trailing `<details>` overflow for truncated reason/rationale text, header lists
  every gate failure (no "+N more"). Real bug found + fixed this session: xhtml2pdf ignores
  `word-break`/`overflow-wrap` inside fixed-width table cells — long unbroken metric names
  (`instruction_adherence_judge`, `emotion_appropriateness_mm`) overflowed straight into the
  next column. Fixed via `_breakable_metric_name()` (inserts a real space after each
  underscore — xhtml2pdf DOES wrap on real whitespace). See docs/ERRORS.md 2026-07-01 for the
  two false leads (title-length, then a flawed "word-break works" test) before finding the
  real cause via PyMuPDF bounding-box inspection. Visually verified all pages of a fresh
  `out/report.pdf` render correctly (no overlap, no empty gaps, no garbled badges).
- **Not started:** README.md rewrite for Deliverable 2 (two-suite quickstart, registry
  example, fixture format incl. TEMPLATE exclusion, gate-vs-advisory table, honest
  limitations section) — must actually run the quickstart commands after writing to verify.
- Task tracker has this as task #13, pending.

## Registered metrics (14 total)
Semantic: `task_success`, `tool_call_ordering`, `instruction_adherence_rule`,
`instruction_adherence_judge`, `faithfulness`.
Acoustic: `barge_in` (headline), `turn_taking_latency`, `latency_thresholds`, `pitch_prosody`,
`entity_intelligibility`, `emotional_appropriateness`, `double_talk` (live-follow-up),
`ser_emotion`, `emotion_appropriateness_mm` (multimodal Gemini judge, hears real audio),
`naturalness_mos` (beyond-scope addition, DNSMOS).

## What's been added since the original Phase 9 "done" snapshot
1. **OpenAI judge support**: `judges/openai_client.py` + `judges/factory.py`
   (`get_default_judge_client()`, env var `VOXGATE_JUDGE_PROVIDER=anthropic|openai`) so any
   judge metric can swap providers via one env var. `.env`/`.env.example` at repo root
   (gitignored) hold `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GEMINI_API_KEY`; loaded
   automatically via `python-dotenv`.
2. **Judge score range bug (real, found via live OpenAI run)**: GPT-4o returned scores on a
   0-10 scale for `instruction_adherence_judge`/`emotional_appropriateness` since nothing
   constrained the pydantic schema. Fixed with `Field(ge=0.0, le=1.0)` on all three judgment
   models (also `faithfulness`'s).
3. **Run-level ship verdict**: `report/combine.py`'s `compute_ship_verdict()` — SHIP iff zero
   FAIL among gate-eligible scores; ERROR never counts (only visible via `error_rate`).
   `run_cli()` returns process exit code 0/1 for CI gating.
4. **Two-proxy emotion attack** (`ser_emotion.py` + `emotion_appropriateness_mm.py`,
   objective offline SER vs. contextual multimodal judge that hears real audio bytes) with
   `compute_emotion_disagreement()` as the cross-metric trust signal (Part 1 Q3) — reporting
   only, never gates. Gemini judge caches responses (`out/.judge_cache/`, keyed by
   call_id/turn_index/prompt_version/model) for CI reproducibility.
5. **Known-answer tests** for the reschedule trap and `barge_in` timing (exact math via
   injected `SpeechSegment`s, since synthetic tones don't trigger real VAD — verified
   empirically) — no bugs found, confirmed existing behavior is correct.
6. **`fixtures/TEMPLATE/`** (copyable valid fixture skeleton, excluded from real eval runs by
   name in `discover_fixtures()`) + `eval_system/validate_fixture.py` (authoring-time
   preflight CLI, exit 0/1).
7. **Versioned Markdown+PDF report**: every `write_report()` call produces a NEW
   `report_<n>.md`/`report_<n>.pdf` (`next_report_number()` scans `out/`, never overwrites).
   PDF via pure-Python `markdown` → `xhtml2pdf` (no native/GTK deps).
8. **`entity_intelligibility` word-level entity location**: WhisperX was tried and rejected
   (broke `ser_emotion` via a forced dependency downgrade — reverted); uses
   `faster-whisper`'s own `word_timestamps=True` instead. `critical_entity_locations` now
   pinpoints WHERE a critical entity survived/was mangled, not just whether.
9. **Combined report rewrite** (this session): per-call metric breakdown (every MetricScore,
   grouped semantic/acoustic, gate failures marked "⚠️ GATE"), acoustic measured-values
   section (turn_taking_latency ms percentiles, pitch_prosody F0 mean/range/rate, latency_
   thresholds FTL, entity_intelligibility per-entity table, barge_in per-event time-to-yield
   ms), faithfulness findings per call, and an expanded Aggregate section (gate metric
   pass/fail/error counts, advisory flag rates, judge coverage, error rate, deterministic-
   vs-judge split). `double_talk`/`naturalness_mos` labeled in the gate-vs-advisory table as
   "live follow-up" / "beyond-scope addition" respectively. Found and fixed a real rendering
   bug: judge free-text notes with embedded newlines corrupted table rows — `_single_line()`
   now collapses whitespace before truncating (see docs/ERRORS.md, 2026-07-01).

## Key decisions (locked, unchanged from original build)
- Open-loop, fixed-clock fixture replay only. No closed-loop bot-to-bot runner, no `agent/`.
- Canonical clock = audio sample index @ sr; metrics never re-derive it.
- One `MetricScore` contract; one `BaseMetric` = one `kind`, never mixed in a single class.
- Gating trust rule: only deterministic + calibration-trusted judges may hard-gate.
  `emotional_appropriateness`/`emotion_appropriateness_mm`/`ser_emotion` are hardcoded
  never-eligible (SER structurally, via kind=signal; the two judges by explicit invariant).
- Category-1 features additive only. Category-2/3 writeup-only (design_writeup.md).

## Open items (non-blocking)
- Gemini free-tier rate limit (5 req/min) means `emotion_appropriateness_mm` can hit
  `Status.ERROR` on some calls in a real run if run back-to-back too fast — by design this
  never affects `ship` (ERROR ≠ FAIL), just visible in `error_rate`.
- `docs/PROGRESS.md`'s checklist reflects only the original Phase 0-9 build; items 1-9 above
  aren't tracked there as discrete checklist entries (tracked here instead).
