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
- **No source code yet.** Awaiting approval of the build plan before any implementation.

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
1. **GATE:** get user approval of the build plan (blocks all source).
2. Phase 1 — scaffold + lock core contracts: `base.py`, `metric_context.py` (types),
   `registry.py`; prove with shape/idempotency/registry-drop-in/ERROR-isolation tests.
3. Phase 2 — fixture format + `happy_path_book`, `reschedule_trap`, `barge_in_basic`.

## Open questions
- Judge provider (Anthropic vs OpenAI) for faithfulness/emotion — default Anthropic per CLAUDE.md tech list.
- Whether to vendor tiny real audio clips or generate synthetic tones for fixtures (must be REAL 2-ch audio + timestamps per constraint; synthetic-but-real waveforms OK, fabricated alignment data NOT).
