# VoxGate — Voice Agent Evaluation Harness

Post-hoc evaluation system for an existing inbound clinic-scheduling voice agent. Scores
recorded calls on two axes — **semantic** (task success, tool-call ordering, faithfulness,
instruction adherence) and **acoustic** (barge-in, turn-taking latency, prosody, emotion,
entity intelligibility) — and fuses them into a two-tier ship / don't-ship verdict for CI,
plus per-call and aggregate reports.

This is a *post-hoc scorer*, not the voice agent itself, and it replays **fixed-clock,
pre-rendered fixtures** rather than driving a live bot-to-bot conversation — see
`docs/design_writeup.md` §1 for why that's the reproducibility-preserving choice.

- **`docs/design_writeup.md`** — the design argument (the graded deliverable): reproducibility,
  taxonomy, proxy validity, gating, offline vs. online, plus the Category-2/3 design notes.
- **`docs/architecture.md`** — the full component/directory map.
- **`CLAUDE.md`** — locked decisions and constraints this build follows.
- **`docs/PROGRESS.md` / `docs/ERRORS.md` / `docs/CONTEXT.md`** — build working-memory (what's
  done, what broke and why, a compressed resumable snapshot).

## Quickstart

```bash
uv sync --extra dev                              # core contracts + semantic suite + tests
uv run pytest -q                                 # run the test suite (134+ tests)

uv run python -m eval_system.run \                # score every fixture -> reports
    --fixtures fixtures/ --out out/
uv run python -m eval_system.run \                # subset, for fast iteration
    --fixtures fixtures/ --out out/ --metrics faithfulness,barge_in
```

Optional extras (installed together, not one at a time — see `docs/ERRORS.md` for why
sequential `--extra` syncs can uninstall packages from extras not listed in that call):

```bash
uv sync --extra acoustic --extra judge --extra stats --extra dev
```

- `acoustic` — librosa, parselmouth (Praat), faster-whisper, silero-vad, jiwer. Note: this
  extra pulls in `torch`/`torchaudio` (silero-vad's dependency) and pins `numba>=0.61`
  (librosa's own floor doesn't build on Python 3.13+ here — see `docs/ERRORS.md`). `webrtcvad`
  was dropped in favor of `silero-vad` (both are CLAUDE.md-sanctioned VAD choices) since it
  needs a C compiler this machine doesn't have.
- `judge` — Anthropic + OpenAI SDKs, for the real judge client. **Without an API key
  configured, judge metrics (`faithfulness`, `instruction_adherence_judge`,
  `emotional_appropriateness`) report `Status.ERROR`, not a crash** — the registry's
  per-evaluator isolation (`_safe()`) means the rest of the report still comes out clean.

  Provider defaults to Anthropic. `judges/factory.py`'s `get_default_judge_client()` is what
  every judge metric falls back to when no client is injected — switch it with one env var.

  **Easiest: drop your key in `.env`** (repo root, already gitignored — copy `.env.example`
  if you deleted it):
  ```
  VOXGATE_JUDGE_PROVIDER=openai
  OPENAI_API_KEY=sk-...
  ```
  `get_default_judge_client()` loads `.env` automatically on every call (via `python-dotenv`,
  a core dependency) — no shell exports needed, and an already-set shell/OS environment
  variable always takes priority over the file. `OPENAI_JUDGE_MODEL` optionally overrides the
  model (default `gpt-4o`); set `VOXGATE_JUDGE_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`
  to switch back. `judges/openai_client.py` / `judges/anthropic_client.py` are both thin
  adapters over each SDK's structured-output parsing — swap providers without touching a
  single metric.

  (You can still set these as real environment variables instead of using `.env` if you
  prefer — `$env:OPENAI_API_KEY = "sk-..."` in PowerShell, `export OPENAI_API_KEY=sk-...`
  in bash.)
- `stats` — scipy + scikit-learn, for calibration (`judge_agreement` kappa, `drift` KS test).

Core contracts and the full semantic suite run with `--extra dev` alone; nothing in
`eval_system/metrics/base.py`, `context/metric_context.py`, or `metrics/registry.py` depends
on the heavy optional libraries.

## Output

`--out out/` writes one JSON per call (`<call_id>.json`: `ship` verdict, `failures`, and every
`MetricScore` — semantic and acoustic emit the identical schema), an `aggregate.json` (total
calls, ships/holds, deterministic/judge/signal counts, error rate, trusted-judge set), and a
`gate_advisory_breakdown.json` — the explicit gate-vs-advisory list with a one-line rationale
per metric (also produced directly by `gating.gate.gate_advisory_breakdown()`).

Every run also writes the human-readable combined report as **`report_<n>.md` + `report_<n>.pdf`**
— a new number each call (`report_1`, `report_2`, ...; `run.py`'s `next_report_number()` scans
`out/` for the highest existing `report_<n>.md` and increments), so every run's report is kept
rather than overwritten. The PDF is rendered from the same Markdown via a pure-Python pipeline
(`markdown` → `xhtml2pdf`/reportlab — no native compiler or system GTK needed, unlike
weasyprint) in `report/pdf_report.py`.

## Gate vs. advisory, at a glance

| Gates | Advisory |
|---|---|
| `task_success`, `tool_call_ordering`, `instruction_adherence` (rule), `barge_in`, `entity_intelligibility` | `instruction_adherence` (judge), `faithfulness` (until calibration-trusted), `turn_taking_latency`, `latency_thresholds` (until promoted), `pitch_prosody`, `emotional_appropriateness` (always), `double_talk` |

Full rationale per metric: `docs/design_writeup.md` §2–3, or `gate_advisory_breakdown.json`
after a run.

## Why this structure (short version)

- **One contract.** Every metric — semantic or acoustic, deterministic or judge — returns the
  same `MetricScore` (`metrics/base.py`) into the same registry and report. There's no
  separate "acoustic report."
- **Registry, not a hardcoded pipeline.** Dropping a metric file with `@register` on it is
  the entire integration step (`metrics/registry.py`) — proven directly by the sampling and
  gate-breakdown tests, and by `double_talk.py` (the live-follow-up addition) needing zero
  runner edits.
- **One canonical clock.** `context/metric_context.py`'s `build_metric_context()` is the
  clock-join backbone: audio sample index at a fixed `sr`. Every acoustic metric reads times
  from `MetricContext`; none re-derive them from raw audio. This was built and tested first,
  deliberately, because a silent misalignment produces confidently wrong numbers with no
  error (see `docs/architecture.md`'s "highest-risk component" note).
- **Judges are a swappable seam, not a hardcoded SDK call.** `judges/client.py`'s
  `JudgeClient` protocol means every judge metric's tests inject a fake and never need
  network access or an API key; `AnthropicJudgeClient` is the one real, thin, deliberately
  untested (nothing to assert without mocking the whole SDK) adapter.
- **Trust is earned, not assumed.** Every judge starts advisory. `calibration/
  judge_agreement.py` (kappa vs. a human set) is what lets `gating/gate.py`'s
  `trusted_judge_metrics` promote one to gate-eligible — and `emotional_appropriateness`
  is hardcoded to never be eligible, because it's honestly a text-over-a-prosody-summary
  proxy, not true multimodal audio judgment.

See `docs/design_writeup.md` for the full argument, including the two real bugs this system
caught in itself while being built (a fabricated-audio-overlap fixture, and a digit-word vs.
numeral STT mismatch) — both documented in `docs/ERRORS.md` with symptom/root-cause/fix.

## Extending VoxGate: two different things, don't confuse them

**Adding a fixture** tests the *agent* — a new scenario (a new reschedule pattern, a new
interrupt timing) scored by the *existing* metrics.

```bash
cp -r fixtures/TEMPLATE fixtures/my_new_scenario
uv run python -m eval_system.validate_fixture fixtures/my_new_scenario/   # validate first
uv run python -m eval_system.run --fixtures fixtures/ --out out/          # then the full eval
```

`fixtures/TEMPLATE/` is a real, valid, copyable fixture (`fixtures/TEMPLATE/README.md` is the
full field-by-field authoring guide — what each of the six files is for, the event
vocabulary, and — the part that's easy to get subtly wrong — how to place an interrupt
marker so a `barge_in` test actually measures something real). `validate_fixture` is a
fast, standalone preflight (channels, timeline bounds, unknown tool names, and a
semantic-alignment check that catches the #1 real failure mode: an interrupt placed where
the agent isn't speaking, which runs without error but measures nothing) — run it before
trusting a new fixture, and it's cheap enough for a pre-commit hook or CI gate (exit 0/1).
`fixtures/TEMPLATE/` itself is excluded from real eval runs (`discover_fixtures()` in
`run.py` skips it by name) — it's a skeleton to copy, not a scenario to score.

**Adding a metric** tests a new *dimension* of ALL existing (and future) calls — drop a file
in `metrics/semantic/` or `metrics/acoustic/`, decorate the class with `@register`, done; zero
runner edits (proven by `double_talk.py`, the live-follow-up addition, and by `ser_emotion.py`/
`emotion_appropriateness_mm.py`). See "Why this structure" above and `docs/design_writeup.md`
§2 for the deterministic-vs-signal-vs-judge decision each new metric should defend.
