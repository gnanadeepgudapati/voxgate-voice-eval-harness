# Fixture authoring guide

This directory is a real, valid fixture (a copy of `happy_path_book` — a clean provider
booking) that passes `validate_fixture` out of the box. Copy it wholesale to start a new
scenario:

```bash
cp -r fixtures/TEMPLATE fixtures/my_new_scenario
uv run python -m eval_system.validate_fixture fixtures/my_new_scenario/   # validate first
uv run python -m eval_system.run --fixtures fixtures/ --out out/          # then the full eval
```

The fixture's `call_id` is just the directory name — nothing inside the files needs to
change for the copy to work as its own independent scenario.

JSON can't hold comments, so this file is the field-by-field explanation the files
themselves can't carry.

## The six files, one paragraph each

**`call.wav`** — the actual audio, 2-channel. **Channel 0 = caller, channel 1 = agent.**
There's no fixed required sample rate (every metric resamples internally), but keep it
real, rendered audio — never fabricate silence and pretend it's speech; `validate_fixture`
can't catch that, but every downstream metric (VAD, STT, F0) will produce meaningless
results against it.

**`transcript.jsonl`** — one JSON object per line, per speaker turn: `{"speaker": "agent"|
"caller", "t_start": float, "t_end": float, "text": str, "asr_confidence": float|null}`.
This is the ground-truth turn timing the rest of the system treats as authoritative —
`pitch_prosody`'s speech rate, `ser_emotion`/`emotion_appropriateness_mm`'s per-turn
segmentation, and this validator's semantic-alignment check all key off these `t_start`/
`t_end` windows, not off `events.jsonl`.

**`tool_log.jsonl`** — one JSON object per line, per tool call as actually executed:
`{"name": str, "args": dict, "result": Any, "t": float}`. This is what `task_success` and
`tool_call_ordering` score against. **Only these tool names exist in this codebase today:
`check_availability`, `book_appointment`, `cancel_appointment`** — there's no dynamic tool
registry (`eval_system/tools/` is an empty stub), so `validate_fixture` hard-fails on
anything else. If your scenario genuinely needs a new tool, add it to
`KNOWN_TOOL_NAMES` in `eval_system/validate_fixture.py`.

**`events.jsonl`** — one JSON object per line: `{"name": str, "t": float, "meta": dict}`.
Authored timeline markers for the acoustic suite. See "Placing an interrupt" below for the
one part of this that's easy to get subtly wrong.

**`scenario_db.json`** — free-form initial DB state (no schema is enforced). In practice,
only `scenario_db["patients"][patient_id]["appointments"]` (a list of appointment IDs) is
actually read by any metric (`tool_call_ordering`'s reschedule-trap invariant) — everything
else (`appointments`, `providers`) is convention for readability, not consumed by code.

**`expected.json`** — what "correct" means for this call:
```json
{
  "tool_sequence": [{"name": "check_availability", "required_args": {"provider": "Lee"}}, ...],
  "invariants": ["never_zero_appointments"],
  "critical_entities": ["Lee", "Tuesday", "ten AM", "four eight two one three"],
  "success_criteria": {"final_tool": "book_appointment", "result_contains": {"status": "booked"}}
}
```

## Placing an interrupt so a barge_in test is meaningful

`events.jsonl` has no enforced vocabulary (any `name` string is technically legal), but the
observed set across real fixtures is `cough`, `barge_in_start`, `agent_yield`,
`zero_appointments_start`, `zero_appointments_end`. `barge_in_start` and `cough` mark a
caller-side interrupt — tag them with `"meta": {"channel": "caller"}` so the validator can
find them.

**The one rule that matters:** an interrupt marker's `t` must fall inside an *agent* turn's
`[t_start, t_end]` window in `transcript.jsonl`. If it lands during silence or a caller turn,
`barge_in` will run without error but measure nothing real — there's no agent speech there
to yield on. `validate_fixture` warns on exactly this ("this barge_in marker will not
measure a real yield"), but it's a warning, not a hard failure, since it can't tell whether
you *meant* it that way — check the message.

A `cough` marker should be short (well under 300ms) and NOT represent real caller speech —
it exists to test that the agent does *not* falsely yield to background noise. A
`barge_in_start` marker should represent genuine, sustained caller speech that actually
starts inside the agent's turn — and critically, the caller audio must **actually overlap**
the agent's audio in `call.wav` at that timestamp, not just claim to via the event (a
fabricated-timestamp bug like this was caught and fixed once already in this repo's
history — see `docs/ERRORS.md`, 2026-06-30).

## Declaring invariants and critical entities

- **`invariants`**: today only `"never_zero_appointments"` is implemented (the reschedule
  trap — the new appointment must be secured before the old one is released; a mid-sequence
  booking failure must never leave the caller with zero appointments).
- **`critical_entities`**: the high-stakes tokens (provider names, dates/times, confirmation
  numbers) that `entity_intelligibility` round-trips through STT to verify they survive the
  audio. Declare these for any scenario where the agent says something a caller needs to
  actually understand — `validate_fixture` warns if a scenario has tool calls or a success
  criterion but no critical entities to check.
