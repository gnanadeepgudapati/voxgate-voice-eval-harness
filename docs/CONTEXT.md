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
- On branch `build/eval-system`. `uv` venv on Python 3.13.5. Core deps only in the default
  install; `acoustic`/`judge`/`stats` are optional extras so core+semantic run without heavy libs.
- **Phases 1-3 done (contracts + fixtures + clock-join):** `metrics/base.py`
  (MetricKind/Status/Gating/MetricScore/BaseMetric), `metrics/registry.py` (REGISTRY,
  @register, two-phase run(), `_safe()` isolation), `context/metric_context.py`
  (Turn/ToolEvent/Event/MetricContext + `build_metric_context()` joining a `RawFixture`
  onto the canonical clock — corrects a *known* per-channel capture skew via sample-shift;
  ground-truth timestamps are already authored on that clock by the loader). Fixture format
  (`fixture_schema.py`/`fixture_loader.py`): `call.wav` (2-ch: ch0=caller, ch1=agent),
  transcript/tool_log/events `.jsonl`, scenario_db/expected `.json`. 3 fixtures generated via
  `scripts/generate_fixtures.py` with real Windows SAPI TTS speech (22050Hz), timestamps
  from actual rendered-clip boundaries (never fabricated): `happy_path_book`,
  `reschedule_trap` (real zero-appointment window via cancel-before-secure + failed rebook),
  `barge_in_basic` (false-yield-on-cough + prompt genuine-barge-in-yield).
- **Phase 4 done (semantic suite / Suite A):** all in `metrics/semantic/`.
  - `task_success.py` — PASS/FAIL from `expected.success_criteria` (final_tool +
    result_contains) vs the actual last tool call; SKIPPED when a fixture defines no
    final_tool (e.g. reschedule_trap — that fixture exists for ordering, not this).
  - `tool_call_ordering.py` — subsequence order check against `expected.tool_sequence`
    PLUS a `never_zero_appointments` state reducer that replays tool_events against
    scenario_db's initial appointment counts — catches the reschedule trap's
    INTERMEDIATE zero-appointment window, not just final order.
  - `instruction_adherence.py` — two classes: `InstructionAdherenceRuleMetric`
    (deterministic/gate — checks `expected.critical_entities` were verbalized back by
    the agent) + `InstructionAdherenceJudgeMetric` (judge/advisory — conversational
    nuance). A `MetricScore`/`BaseMetric` is one kind, never both, so "rule + judge" in
    CLAUDE.md's taxonomy table means two registered metrics, not one.
  - `faithfulness.py` — judge/advisory; reads per-turn `asr_confidence` (C1(6)) and flags
    low-confidence turns in the prompt for the judge to weigh cautiously; absent
    confidence changes nothing.
  - Both judges depend on `eval_system/judges/client.py`'s `JudgeClient` Protocol
    (`structured_complete(prompt, response_model) -> response_model`), constructor-injected
    and lazily defaulting to `judges/anthropic_client.py`'s `AnthropicJudgeClient` (thin
    SDK adapter, not unit-tested — no meaningful unit test without mocking the whole SDK).
    Tests inject a fake client, so the full suite runs with no network/API key.
  - 39/39 tests passing.

## Key decisions (locked)
- **Open-loop, fixed-clock fixture replay only.** No closed-loop bot-to-bot runner.
- **No `agent/` module** — agent is external; we replay/score, never build it.
- **Canonical clock = audio sample index @ sr.** Every acoustic metric reads times from
  `MetricContext`, never re-derives.
- **One contract:** every metric returns `MetricScore` (status/gating/versions/idempotency key).
  One `BaseMetric` = one `kind` (deterministic xor judge) — never mixed in a single class.
- **Gating trust rule:** only deterministic + calibration-trusted metrics may hard-gate.
  Judges start ADVISORY until kappa ≥ threshold. Emotion is advisory always.
- **Category-1 features are additive only** — never rewrite a metric, the clock-join, or the gate.
- **Category-2/3** (DLQ, distributed workers, key rotation, judge self-consistency, PHI/BAA)
  are writeup-only. No fake infra. (This does NOT cover the `JudgeClient` seam itself — that's
  a real, necessary component, just built swappable/testable via dependency injection.)
- **Reschedule trap:** `tool_call_ordering` must catch the INTERMEDIATE zero-appointment state
  (new slot secured before old released), not just final order. Done.

## Build order (see PROGRESS.md for the live checklist)
Contracts → fixtures → clock-join(+tests) → semantic (det→judge) → acoustic (barge_in first)
→ calibration → gating+report → validators/sampling/monitoring → docs.

## Next 3 steps
1. Phase 5 — acoustic suite, `barge_in` first (headline metric; `barge_in_basic` fixture
   already encodes both a false-yield and a prompt-yield case to test against). Needs VAD
   (`silero-vad`/`webrtcvad`) over the 2-ch audio + the authored cough/barge_in_start/
   agent_yield events from `metric_context`.
2. Phase 5 (cont.) — `turn_taking_latency` (p50/p90/p99 distribution), `latency_thresholds`
   (deterministic, advisory), `pitch_prosody`, `entity_intelligibility` (STT round-trip +
   WER on `expected.critical_entities`), `emotional_appropriateness` (judge, always advisory).
3. Phase 6 — calibration (`judge_agreement` kappa, `drift` KS test) so judges can earn gate
   trust; currently every judge metric here is hardcoded ADVISORY.

## Open questions
- Judge provider (Anthropic vs OpenAI) — using Anthropic (`AnthropicJudgeClient`) per
  CLAUDE.md tech list; OpenAI extra installed but no client built yet (not needed unless
  we want a second-judge cross-check for calibration).
- Whether `acoustic` extras (librosa/parselmouth/faster-whisper/webrtcvad) install cleanly
  on this Python 3.13.5 venv — not yet attempted; do this at the start of Phase 5.
