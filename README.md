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
uv sync --extra acoustic --extra judge --extra stats --extra dev  # everything, one shot
uv run pytest -q                                 # run the test suite (284 tests)

uv run python -m eval_system.run \                # score every fixture -> reports
    --fixtures fixtures/ --out out/
echo $?                                           # 0 = ship, 1 = hold -- this is the CI gate

uv run python -m eval_system.run \                # subset, for fast iteration
    --fixtures fixtures/ --out out/ --metrics faithfulness,barge_in
```

Install every extra **together, in one `uv sync` call** — always all four (`acoustic`,
`judge`, `stats`, `dev`), never one at a time (verified: a later `uv sync --extra dev` alone
silently *uninstalls* packages from extras not named in that call, and `calibration/drift.py`
/ `calibration/judge_agreement.py` import `scipy`/`scikit-learn` unconditionally at module
level, so `uv run pytest -q` fails to even *collect* tests without `--extra stats` present —
see `docs/ERRORS.md`).

That single `eval_system.run` command scores **both suites** (semantic + acoustic) over
every fixture in `fixtures/` and writes everything under `--out`: per-call JSON, an
aggregate, the gate-vs-advisory breakdown, and the combined human-readable report
(`report.md` / `report.html` / `report.pdf`, overwritten each run — see "Output" below).
The process exit code (`0`/`1`) is the thing a CI pipeline should actually gate on.

If you genuinely only want the core contracts + semantic suite running (no acoustic/judge
extras), `uv sync --extra dev` alone is enough to run `eval_system.run` itself — judge
metrics will report `Status.ERROR` (not a crash) rather than a `FAIL` — but skip the
`stats`-dependent test files (`test_drift.py`, `test_judge_agreement.py`) since they don't
degrade gracefully at collection time:

```bash
uv sync --extra dev
uv run pytest -q --ignore=tests/test_drift.py --ignore=tests/test_judge_agreement.py
```

- `acoustic` — librosa, parselmouth (Praat), faster-whisper, silero-vad, jiwer, speechmos.
  Pulls in `torch`/`torchaudio` (silero-vad's dependency) and pins `numba>=0.61` (librosa's
  own floor doesn't build on Python 3.13+ here — see `docs/ERRORS.md`). `webrtcvad` was
  dropped in favor of `silero-vad` (both are CLAUDE.md-sanctioned VAD choices) since it needs
  a C compiler this machine doesn't have.
- `judge` — Anthropic + OpenAI SDKs (text judges) and `google-genai` (the multimodal emotion
  judge). **Without an API key configured, judge metrics report `Status.ERROR`, not a
  crash** — the registry's per-evaluator isolation (`_safe()`) means the rest of the report
  still comes out clean, and `ERROR` never fails the ship verdict (see "Single verdict"
  below).

  - `faithfulness`, `instruction_adherence_judge`, `emotional_appropriateness` need
    `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (provider selected by `VOXGATE_JUDGE_PROVIDER`).
  - `emotion_appropriateness_mm` (the multimodal emotion judge that hears real audio bytes)
    needs `GEMINI_API_KEY` **specifically** — it's independent of `VOXGATE_JUDGE_PROVIDER`
    and will report `Status.ERROR` on every call if that key is missing, while every other
    metric runs unaffected.

  **Easiest: drop your key(s) in `.env`** (repo root, already gitignored — copy
  `.env.example` if you deleted it):
  ```
  VOXGATE_JUDGE_PROVIDER=openai
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=...
  ```
  `get_default_judge_client()` (in `judges/factory.py`) loads `.env` automatically on every
  call (via `python-dotenv`, a core dependency) — no shell exports needed, and an
  already-set shell/OS environment variable always takes priority over the file.
  `OPENAI_JUDGE_MODEL` optionally overrides the model (default `gpt-4o`); set
  `VOXGATE_JUDGE_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` to switch back.
  `judges/openai_client.py` / `judges/anthropic_client.py` are both thin adapters over each
  SDK's structured-output parsing — swap providers without touching a single metric.

  (You can still set these as real environment variables instead of using `.env` if you
  prefer — `$env:OPENAI_API_KEY = "sk-..."` in PowerShell, `export OPENAI_API_KEY=sk-...`
  in bash.)
- `stats` — scipy + scikit-learn, for calibration (`judge_agreement` kappa, `drift` KS test).

Core contracts and the full semantic suite run with `--extra dev` alone; nothing in
`eval_system/metrics/base.py`, `context/metric_context.py`, or `metrics/registry.py` depends
on the heavy optional libraries.

## The two suites

Both suites feed the same registry and emit the identical `MetricScore` shape
(`metrics/base.py`) — there's no separate "acoustic report." Every evaluator makes an
explicit, defended deterministic-vs-signal-vs-judge choice (assessment.md Parts 2 & 3):

### Suite A — semantic (`metrics/semantic/`)

| Metric | Kind | Why |
|---|---|---|
| `task_success` | deterministic | Final tool call vs. the fixture's ground-truth success criteria — no proxy, no judge needed. |
| `tool_call_ordering` | deterministic | A state-reducer walk over the tool log, including the **reschedule-trap invariant** (new slot secured before the old one releases; the caller must never sit at zero appointments mid-sequence) — this is a correctness bug, not a style preference, so it's asserted in code. |
| `instruction_adherence_rule` | deterministic | Substring check that critical entities were actually read back to the caller — objective and ground-truthed. |
| `instruction_adherence_judge` | LLM judge | Conversational nuance ("was this handled gracefully") a keyword check can't capture — starts advisory until `judge_agreement` clears the kappa bar. |
| `faithfulness` | LLM judge | Hallucination detection needs semantic understanding of whether a claim is grounded in the tool results, not just present in them — advisory until calibration trusts it. |

### Suite B — acoustic (`metrics/acoustic/`)

| Metric | Kind | Why |
|---|---|---|
| `barge_in` **(headline)** | signal (VAD) | Detects every point the caller speaks over the agent and measures time-to-yield, flagging both fail-to-yield and false-yield — the flagship interruption-handling behavior (assessment.md line 54). VAD-derived but deterministic once computed. |
| `turn_taking_latency` | signal | Gap between caller end-of-speech and agent response onset, reported as a **distribution** (p50/p90/p99), not a mean — a single flaky pause shouldn't drown a normally-fast agent. |
| `latency_thresholds` | deterministic | Timestamp arithmetic against a fixed cutoff — deterministic math, but the cutoff itself is a judgment call, so it stays advisory until promoted. |
| `pitch_prosody` | signal (F0) | Praat F0 extraction + speech rate — a genuine signal-processing measurement, but "pleasant pitch" is a perceptual proxy, so it's advisory. |
| `emotional_appropriateness` | LLM judge | Text judge over a prosody *summary* (not real audio) — always advisory, never promoted, because it's honestly not true multimodal judgment. |
| `entity_intelligibility` | signal (round-trip STT + WER) | Re-transcribes the agent's own audio and checks critical entities (names, dates, confirmation numbers) survive — if a real STT engine can't recover it, a caller likely couldn't either, so this gates. |
| `ser_emotion` *(two-proxy emotion, half A)* | signal (SER classifier) | Objective wav2vec2 classifier on the raw waveform — a real signal, but IEMOCAP shows even *humans* only agree with each other at Fleiss' kappa ~0.27–0.48 on acted emotion, so a noisier-than-human-agreement proxy can't gate. |
| `emotion_appropriateness_mm` *(two-proxy emotion, half B)* | multimodal LLM judge | Hears the real audio bytes + conversational context — the closest thing here to true multimodal judgment, but LLM judges drift and are noisy, so always advisory regardless of calibration. |
| `double_talk` *(live-follow-up)* | signal | Overlap duration/ratio between channels — overlap alone isn't a defect (natural backchannels overlap constantly), so it reports, it doesn't gate. |
| `naturalness_mos` *(beyond-scope addition)* | signal (DNSMOS/P.808) | Non-intrusive MOS estimate — saturates above ~4.0 and can't reliably separate "good" from "excellent," same class of limitation as `pitch_prosody`. |

`ser_emotion` + `emotion_appropriateness_mm` are a deliberate **two-proxy attack** on the
hardest metric in the assessment (emotional appropriateness): one objective classifier, one
contextual multimodal judge, cross-checked per agent turn (`compute_emotion_disagreement()`
in `report/combine.py`). Neither can gate on its own, but a *disagreement* between them is a
free trust signal — flag the turn for human review without spending any labeling budget on
it (assessment.md Part 1 Q3).

## Registry: adding a metric touches one file

Drop a class in `metrics/semantic/` or `metrics/acoustic/`, decorate it `@register`, and
it's live — zero edits to `registry.py`'s `run()` or to `run.py`. Every metric returns the
same `MetricScore`; the two-phase runner (`run(ctx, sampler=None)`) executes all
deterministic/signal metrics first, then judges per the sampling policy, and wraps every
`compute()` call in `_safe()` so one evaluator's exception becomes a `Status.ERROR` score,
never a crashed run.

Minimal shape (this is `double_talk.py`, the actual live-follow-up drop-in added this way —
see `docs/ERRORS.md`/`docs/CONTEXT.md` for how it was verified to need zero runner edits):

```python
from eval_system.metrics.base import BaseMetric, Gating, MetricKind, MetricScore, Status
from eval_system.metrics.registry import register

@register
class MyNewMetric(BaseMetric):
    name = "my_new_metric"
    version = "1"
    kind = MetricKind.SIGNAL          # or DETERMINISTIC / JUDGE -- pick one, never mixed
    default_gating = Gating.ADVISORY  # GATE only if you can defend it -- see below
    requires_ground_truth = False     # False => can also run on live traffic, not just fixtures

    def compute(self, ctx) -> MetricScore:
        ...
        return MetricScore(
            call_id=ctx.call_id, metric=self.name, kind=self.kind,
            status=Status.PASS, gating=self.default_gating, score=1.0,
            details={...}, evaluator_version=self.version,
        )
```

Then add one line to `GATE_RATIONALE` in `eval_system/gating/gate.py` (the DoD's explicit
"why this gates or not" list — see "Gate vs. advisory" below) and import the module from
`run.py` so its `@register` side effect actually fires (import-for-side-effect is the one
place the runner needs to know a new metric module exists at all).

## Fixture format: fixed-clock, open-loop replay

Every fixture is a directory of **6 files** describing one already-recorded call — VoxGate
never drives a live conversation (no bot-to-bot closed loop; see
`docs/design_writeup.md` §1 for why open-loop fixed-clock replay is what makes a barge-in
test mean the same thing every run):

| File | Contents |
|---|---|
| `call.wav` | 2-channel real audio — channel 0 caller, channel 1 agent. No enforced sample rate (every metric resamples internally). |
| `transcript.jsonl` | One JSON object per turn: `speaker`, `t_start`, `t_end`, `text`, `asr_confidence?` — the ground-truth turn timing every acoustic metric keys off. |
| `tool_log.jsonl` | One JSON object per executed tool call: `name`, `args`, `result`, `t` — what `task_success`/`tool_call_ordering` score against. |
| `events.jsonl` | Authored timeline markers (`barge_in_start`, `cough`, `agent_yield`, ...) for the acoustic suite. |
| `scenario_db.json` | Initial ground-truth DB state (only `patients[id].appointments` is actually read, by the reschedule-trap invariant). |
| `expected.json` | What "correct" means: `tool_sequence`, `invariants`, `critical_entities`, `success_criteria`. |

**Canonical clock:** every timestamp across every file is seconds on one clock — the audio
sample index at the fixture's `sr`. `context/metric_context.py`'s `build_metric_context()`
is the single clock-join backbone; acoustic metrics read times from `MetricContext` and
never re-derive them from raw audio. This was built and tested first, deliberately, because
a silent misalignment produces confidently wrong numbers with no error.

### Adding a fixture

```bash
cp -r fixtures/TEMPLATE fixtures/my_new_scenario
uv run python -m eval_system.validate_fixture fixtures/my_new_scenario/   # validate first
uv run python -m eval_system.run --fixtures fixtures/ --out out/          # then the full eval
```

`fixtures/TEMPLATE/` is a real, valid, copyable fixture — `fixtures/TEMPLATE/README.md` is
the full field-by-field authoring guide, including the one thing that's easy to get subtly
wrong: an interrupt marker must fall inside an *agent* turn's window AND the caller audio
must genuinely overlap the agent audio at that timestamp in `call.wav` (a fabricated-
timestamp bug like this was caught and fixed once already in this repo — see
`docs/ERRORS.md`, 2026-06-30). `validate_fixture` (channels, timeline bounds, unknown tool
names, and that same semantic-alignment check) is a fast standalone preflight, cheap enough
for a pre-commit hook or CI gate — exit 0/1. `TEMPLATE/` itself is excluded from real eval
runs (`discover_fixtures()` in `run.py` skips it by name) — it's a skeleton to copy, not a
scenario to score.

### Running tests

```bash
uv run pytest -q               # 284 tests
uv run pytest tests/test_barge_in_known_answer.py tests/test_known_answer_reschedule_trap.py -q
```

Beyond ordinary unit tests, there are **known-answer tests** with exact expected values for
the two hardest things to get right: the reschedule-trap invariant (tool-ordering state
machine) and `barge_in` timing (exact time-to-yield math via injected speech segments, plus
one real-audio alignment test against an actual fixture). These were written to *fail loudly*
if a metric's logic silently drifts, not just to hit a coverage number.

## Output

`--out out/` writes:
- One JSON per call (`<call_id>.json`): `ship` verdict, `failures`, and every `MetricScore` —
  semantic and acoustic emit the identical schema.
- `aggregate.json` — total calls, ships/holds, deterministic/judge/signal counts, error rate,
  trusted-judge set, and the run-level `ship`/`gate_failures`/`advisory_failures`/
  `ship_reason` fields (see "Single verdict" below).
- `gate_advisory_breakdown.json` — the explicit gate-vs-advisory list with a one-line
  rationale per metric (also producible directly via `gating.gate.gate_advisory_breakdown()`).
- The combined human-readable report, written **fresh every run** (overwritten, not
  versioned) as `report.md`, `report.html`, and `report.pdf` — per-call metric breakdown
  (grouped semantic/acoustic), acoustic measured values (barge-in per-event time-to-yield,
  turn-taking-latency percentiles, F0/rate, entity-by-entity STT survival), faithfulness
  judge findings, the emotion two-proxy disagreement table, and the full aggregate +
  gate-vs-advisory sections. The PDF is rendered directly from the HTML
  (`report/html_report.py` → `report/pdf_report.py`'s `html_to_pdf_bytes()`, via
  `xhtml2pdf`/reportlab — pure Python, no native compiler or system GTK needed, unlike
  weasyprint).

## Single verdict: how a pile of scores becomes ship/hold

**SHIP iff zero gate-eligible `FAIL`/`ERROR` across the whole fixture set** — advisory
scores, however bad, never block a ship (`compute_ship_verdict()` in `report/combine.py`;
per-call gating is `evaluate_gate()`'s "pass^k" conjunction in `gating/gate.py` — ALL
gate-eligible metrics must pass, not an average). `run_cli()` returns that verdict as a
process exit code: **0 = ship, 1 = hold** — that's the literal CI gate.

- **`ERROR` ≠ `FAIL`.** An `ERROR` means the evaluator itself broke (bad API key, model
  crash, rate limit) — it's fail-closed on a gate-eligible metric (doesn't ship) but reported
  distinctly, and it's exactly why the registry wraps every `compute()` in `_safe()`: one
  broken evaluator never corrupts the rest of the report.
- **`SKIPPED` gate metrics are excluded from the conjunction entirely**, not counted as a
  missing pass — e.g. `task_success` with no `success_criteria` defined for that fixture.
- **A judge only becomes gate-eligible once trusted.** Every judge starts advisory;
  `calibration/judge_agreement.py` (Cohen's kappa vs. a small human-labeled set, threshold
  0.6) is what lets `gating/gate.py`'s `trusted_judge_metrics` promote one. Two metrics are
  **hardcoded never-eligible** regardless of kappa: `emotional_appropriateness` and
  `emotion_appropriateness_mm` (see the design writeup — text-over-a-prosody-summary and
  LLM-judge drift respectively). `calibration/drift.py` (KS-test vs. a frozen golden-set
  baseline) is the tripwire for a judge that *was* trusted starting to score differently
  after a model/prompt change.

## Gate vs. advisory, with rationale

The exact rationale VoxGate defends for every registered metric (mirrors
`gate_advisory_breakdown.json` / the report's own "Gate vs. advisory" section —
`gating/gate.py`'s `GATE_RATIONALE` is the single source of truth this table is generated
from):

| Metric | Gating | Rationale |
|---|---|---|
| `task_success` | **gate** | Deterministic: final tool call checked against the fixture's ground-truth success criteria. |
| `tool_call_ordering` | **gate** | Deterministic state-machine check, including the reschedule-trap invariant — catches a real correctness bug. |
| `instruction_adherence_rule` | **gate** | Deterministic substring check that critical entities were read back to the caller. |
| `barge_in` | **gate** | Deterministic once computed from VAD, and the headline interruption-handling behavior — a real miss is a real defect. |
| `entity_intelligibility` | **gate** | Round-trip STT on ground-truthed critical entities — if a real STT engine can't recover it, a caller likely couldn't either. |
| `instruction_adherence_judge` | advisory | LLM judge for conversational nuance a keyword check can't capture — advisory until calibration earns trust. |
| `faithfulness` | advisory | LLM judge for grounding — advisory until calibration proves the judge itself trustworthy. |
| `turn_taking_latency` | advisory | Reports a latency distribution (p50/p90/p99), not a single verdict — advisory by nature. |
| `latency_thresholds` | advisory | Deterministic arithmetic against a threshold that is itself a judgment call — advisory until promoted. |
| `pitch_prosody` | advisory | F0 and speech rate are perceptual proxies for naturalness, not correctness. |
| `emotional_appropriateness` | advisory (always) | Text judge over a prosody summary, not true multimodal audio — always advisory, never promoted. |
| `ser_emotion` | advisory (never-promotable) | Objective classifier, but acted-emotion SER is noisier than human inter-rater agreement (IEMOCAP kappa ~0.3–0.5) — can't gate. |
| `emotion_appropriateness_mm` | advisory (always) | Multimodal judge that hears real audio and context, but LLM judges drift and are noisy — always advisory, never promoted. |
| `double_talk` | advisory | Overlap alone isn't necessarily a defect — reports duration/ratio, advisory by nature. Live-follow-up drop-in. |
| `naturalness_mos` | advisory | Non-intrusive MOS saturates above ~4 and can't separate "good" from "excellent." Beyond-scope addition. |

## Known limitations — honesty about what's not solved

- **Every acoustic/emotion score is a perceptual proxy, not ground truth.** F0/prosody,
  SER, and multimodal-judge tone are all stand-ins for what a human would actually perceive.
  None of the "always advisory" metrics above are expected to ever earn gate status — that's
  a deliberate position, not a TODO.
- **IEMOCAP-level human agreement on acted emotion is only ~0.27–0.48 kappa.** This is the
  actual argument for why `ser_emotion` can never be promoted: a proxy noisier than human
  inter-rater agreement has no business gating a deploy, no matter how good its own
  self-consistency looks.
- **LLM judges drift.** A judge that clears the kappa bar today isn't guaranteed to still be
  well-calibrated after a silent model update or prompt change — `calibration/drift.py`'s
  KS-test against a frozen golden set is the detection mechanism, but it's a tripwire, not a
  guarantee; there's no automatic re-calibration loop here (a Category-2 design decision,
  documented not implemented — see the design writeup).
- **Non-intrusive MOS (DNSMOS) saturates.** Above roughly 4.0/5.0 it stops reliably
  separating "good" from "excellent" — useful for catching genuinely bad audio, not for
  fine-grained naturalness ranking. Same limitation class as `pitch_prosody`.
- **No judge self-consistency sampling.** Repeating one judge call k times and requiring
  self-agreement is scoped as a Category-2 design decision (documented in
  `docs/design_writeup.md`), not implemented — `gating/gate.py`'s "pass^k" is a conjunction
  across *different* gate-eligible metrics, not repeated sampling of the same judge.
- **Small fixture set.** Three synthetic fixtures (`happy_path_book`, `reschedule_trap`,
  `barge_in_basic`) exercise the known-answer paths deliberately, not a representative
  production distribution — see "Offline vs. online" in the design writeup for what only
  real production traffic would actually teach.

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
  from `MetricContext`; none re-derive them from raw audio.
- **Judges are a swappable seam, not a hardcoded SDK call.** `judges/client.py`'s
  `JudgeClient` protocol means every judge metric's tests inject a fake and never need
  network access or an API key; the real SDK adapters are thin and deliberately untested
  beyond that seam.
- **Trust is earned, not assumed.** Every judge starts advisory; calibration is what
  promotes one, and two emotion metrics are hardcoded to never be eligible regardless.

See `docs/design_writeup.md` for the full argument, including the two real bugs this system
caught in itself while being built (a fabricated-audio-overlap fixture, and a digit-word vs.
numeral STT mismatch) — both documented in `docs/ERRORS.md` with symptom/root-cause/fix.
