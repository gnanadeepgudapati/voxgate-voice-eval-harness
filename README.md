# VoxGate — Voice Agent Evaluation Harness

An **offline, fixture-driven evaluation system** that decides, automatically and reproducibly, whether a given build of an inbound clinic-scheduling voice agent is **good enough to ship**.

It does not build a voice agent. It is the system that judges one — scoring both **what the agent says and does** (semantic / behavioral) and **how it sounds and behaves** (acoustic / paralinguistic), then collapsing everything into a single **SHIP / HOLD** verdict a CI pipeline can consume.

> This README is the "why behind the structure" (assessment.md line 69). For the full design argument see the Part 1 write-up (`docs/design_writeup.pdf`).

---

## Table of contents

1. [Quickstart — how to run](#1-quickstart--how-to-run)
2. [The one idea behind every decision](#2-the-one-idea-behind-every-decision)
3. [Architecture overview](#3-architecture-overview)
4. [Instruction flow — what happens on a run](#4-instruction-flow--what-happens-on-a-run)
5. [The fixture format](#5-the-fixture-format)
6. [The two suites & every evaluator](#6-the-two-suites--every-evaluator)
7. [Technologies involved](#7-technologies-involved)
8. [Gate vs. advisory — the ship decision](#8-gate-vs-advisory--the-ship-decision)
9. [Evaluating the evaluators (calibration & drift)](#9-evaluating-the-evaluators-calibration--drift)
10. [The registry — adding a new evaluator](#10-the-registry--adding-a-new-evaluator)
11. [Adding your own fixtures](#11-adding-your-own-fixtures)
12. [Tests](#12-tests)
13. [Honesty about uncertainty](#13-honesty-about-uncertainty)
14. [Offline vs. online](#14-offline-vs-online)

---

## 1. Quickstart — how to run

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) enable the multimodal emotion judge
export GEMINI_API_KEY=<your-key>     # only needed for emotion_appropriateness_mm;
                                     # without it that one metric returns ERROR (not FAIL)
                                     # and the run continues — ERROR never blocks a ship.

# 3. Run BOTH suites over a fixture set → combined report + CI verdict
python -m runners.open_loop --fixtures fixtures/ --out out/

#   • writes out/report.md, out/report.html, out/report.json
#   • prints the single SHIP/HOLD verdict
#   • exits 0 on SHIP, 1 on HOLD  ← this is the CI contract
```

Run a single fixture:

```bash
python -m runners.open_loop --fixtures fixtures/reschedule_trap/ --out out/
```

Run the test suite:

```bash
pytest -q            # 20+ known-answer unit tests
```

**Requirements:** Python 3.10+. Core deps: `numpy`, `librosa`, `parselmouth` (Praat), `silero-vad` (via `torch`), `faster-whisper`, `jiwer`, `pyannote.audio`, `google-genai` (for the Gemini judge), `scipy` (KS drift test), `pytest`. See `requirements.txt` for pinned versions.

---

## 2. The one idea behind every decision

**An evaluation is only useful if you can trust its numbers and reproduce them exactly.**

Every architectural choice below follows from that single constraint:

- **Determinism over realism** — we replay a fixed timeline instead of chatting live, so a barge-in test means the same thing every run.
- **ERROR ≠ FAIL** — a crashed metric must never masquerade as a quality regression, nor block a good deploy.
- **A hard wall between gate and advisory** — only objective, calibrated, reproducible metrics may block a release; everything perceptual is reported, never blocking.

---

## 3. Architecture overview

```
                              ┌──────────────────────────────────────────────┐
                              │                 FIXTURE (4 files)             │
                              │  caller.wav · events.jsonl ·                   │
                              │  scenario_db.json · expected.json              │
                              └───────────────────────┬──────────────────────┘
                                                      │
                                                      ▼
                           ┌────────────────────────────────────────────┐
                           │        validators/  (retry → quarantine)     │
                           │   rejects malformed fixtures before a run     │
                           └───────────────────────┬──────────────────────┘
                                                    │
                                                    ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │            runners/open_loop.py                             │
                    │   FIXED-CLOCK OPEN-LOOP REPLAY                              │
                    │   drives the agent off events.jsonl; the interrupt at        │
                    │   t=4.2s fires at exactly 4.2s on EVERY run                  │
                    └───────────────────────┬───────────────────────────────────┘
                                            │  produces run artifacts:
                                            │  agent audio · transcript ·
                                            │  tool-call/tool-result log · timestamps
                                            ▼
                    ┌───────────────────────────────────────────────────────────┐
                    │         context/metric_context.py   (THE CLOCK-JOIN)        │
                    │   aligns the agent output timeline with the fixture event    │
                    │   timeline. Highest-risk component: bad alignment =          │
                    │   wrong numbers with NO error raised.                        │
                    └───────────────┬───────────────────────────┬────────────────┘
                                    │                           │
                     ┌──────────────▼─────────────┐ ┌───────────▼──────────────────┐
                     │  SUITE A — SEMANTIC         │ │  SUITE B — ACOUSTIC            │
                     │  (transcript + tool record) │ │  (audio + event timeline)      │
                     │  · task_success       (det) │ │  · barge_in        (signal★)   │
                     │  · tool_call_ordering (det) │ │  · turn_taking_latency (signal)│
                     │  · faithfulness     (judge) │ │  · pitch_prosody      (signal) │
                     │  · instruction_adherence    │ │  · entity_intelligibility(sig) │
                     │        (rule + judge)       │ │  · emotional_appropriateness   │
                     │                             │ │        (SER + multimodal judge)│
                     │                             │ │  · double_talk        (signal) │
                     │                             │ │  · naturalness_mos    (signal) │
                     └──────────────┬─────────────┘ └───────────┬──────────────────┘
                                    │      every metric returns the same           │
                                    │      MetricScore(status, gating, score,       │
                                    │      details, versions)                       │
                                    └───────────────┬───────────────────────────────┘
                                                    ▼
                              ┌──────────────────────────────────────────────┐
                              │   metrics/registry.py   (plug-in registry)    │
                              │   ENFORCES: ERROR ≠ FAIL                        │
                              └───────────────────────┬──────────────────────┘
                                                      ▼
                    ┌──────────────────────┐   ┌──────────────────────────────────┐
                    │  calibration/        │   │       gating/gate.py              │
                    │  judge_agreement (κ) │──▶│  SHIP iff ZERO gate failures      │
                    │  drift (KS test)     │   │  advisory never blocks · exit 0/1 │
                    └──────────────────────┘   └───────────────┬──────────────────┘
                                                               ▼
                              ┌──────────────────────────────────────────────┐
                              │   report/combine.py  →  out/report.{md,html,json}│
                              │   single verdict + per-call + aggregate +      │
                              │   gate/advisory statement + emotion disagreement│
                              └──────────────────────────────────────────────┘

                    monitoring/production_proxies.py  ── reuses the metric library
                                                          for live monitoring (see §14)

                    ★ barge_in = the HEADLINE metric (assessment.md line 54)
```

### Component responsibilities

| Component | Responsibility |
|---|---|
| `runners/open_loop.py` | Fixed-clock open-loop replay engine. Drives the agent off the fixture timeline; produces run artifacts. |
| `context/metric_context.py` | **The clock-join.** Aligns agent-output timeline with fixture-event timeline. Highest-risk component — tested first and hardest. |
| `validators/` | Validates fixtures before a run; retries transient issues, quarantines malformed fixtures. |
| `metrics/registry.py` | Plug-in registry. New evaluators register here without touching the runner. Enforces ERROR ≠ FAIL. |
| `metrics/base.py` | Defines the shared `MetricScore` object every evaluator returns. |
| `metrics/semantic/` | Suite A evaluators (transcript + tool-call record). |
| `metrics/acoustic/` | Suite B evaluators (audio + timeline). |
| `sampling.py` | Judge sampling policy (defaults to full coverage on a fixture set). |
| `calibration/` | `judge_agreement` (Cohen's/Fleiss' κ) + drift detection (KS test, version stamps). |
| `gating/gate.py` | Collapses all `MetricScore`s into one SHIP/HOLD boolean + CI exit code. |
| `report/combine.py` | Renders the combined report and the gate-vs-advisory statement. |
| `monitoring/production_proxies.py` | Reuses the metric library against production traffic (online monitoring). |

---

## 4. Instruction flow — what happens on a run

1. **Load & validate** — `validators/` reads the fixture's four files. Malformed fixtures are quarantined; the run doesn't score garbage.
2. **Fixed-clock replay** — `runners/open_loop.py` plays `caller.wav` and fires each event from `events.jsonl` at its exact timestamp against the agent. The caller audio does **not** react to the agent (open loop) — that's what makes the run reproducible.
3. **Capture artifacts** — the runner records the agent's audio, transcript, tool-call/tool-result log, and every timestamp.
4. **Clock-join** — `context/metric_context.py` aligns the agent output timeline against the fixture event timeline so every metric measures against the same, correct clock.
5. **Score both suites** — the registry runs Suite A (semantic) and Suite B (acoustic). Every evaluator returns a `MetricScore(status, gating, score, details, versions)`. Deterministic/signal metrics run on every call; judge metrics run per the sampling policy.
6. **Gate** — `gating/gate.py` reads the `gating` flag on each score: **SHIP iff zero gating failures across all calls.** Advisory results and ERRORs never block.
7. **Report** — `report/combine.py` writes `out/report.{md,html,json}` with the verdict, per-call detail, aggregate, the emotion-disagreement signal, and the gate/advisory statement.
8. **Exit** — process exits `0` on SHIP, `1` on HOLD. That's the entire CI contract.

---

## 5. The fixture format

A fixture fully specifies **one scenario** as four decoupled files, so a test author can change one axis without regenerating the others:

```
fixtures/reschedule_trap/
├── caller.wav          # the human audio (the simulated caller)
├── events.jsonl        # timed ground-truth events, one per line:
│                       #   {"t": 4.2, "event": "interrupt_start"}
│                       #   {"t": 6.0, "event": "interrupt_end"}
├── scenario_db.json    # the world state the agent's tools read from
│                       #   (providers, existing appointments, availability)
└── expected.json       # ground truth to score against
                        #   (expected tool sequence, critical entities,
                        #    success criteria, expected emotional context)
```

**Why this design (Q1 — reproducibility):** a real voice agent's output changes run to run and depends on *when* things happen. By pinning the interruption timing in `events.jsonl` and replaying it on a fixed clock, a barge-in test **means the same thing across runs**. The tradeoff — we give up emergent multi-turn behavior a live human would produce — is deliberate: a non-reproducible number nobody can debug is worse than no number.

`fixtures/TEMPLATE/` is a scaffold for authoring new fixtures and is **excluded from real eval runs** (the runner skips any fixture directory named `TEMPLATE`).

---

## 6. The two suites & every evaluator

Every evaluator makes an **explicit, defended choice** of deterministic vs. LLM-judge vs. signal-processing (assessment.md line 46).

### Suite A — Semantic / behavioral (transcript + tool-call + tool-result)

| Evaluator | Kind | What it checks | Why this kind |
|---|---|---|---|
| `tool_call_ordering` | **Deterministic** | Right tool, right args, right sequence. Includes the **reschedule trap**: the new slot must be secured *before* the old one is released, and a mid-sequence failure must never leave the caller with zero appointments (`never_zero_appointments` invariant). | Correctness is exactly assertable in code — no proxy needed. |
| `task_success` | **Deterministic** | Did the final tool call match the fixture's ground-truth success criteria? | Objective; a plain state comparison. |
| `faithfulness` | **LLM judge** | Did the agent surface only providers, times, and confirmation numbers that actually appear in tool results (no hallucination)? | Requires semantic comparison of free-form speech against structured tool output — a keyword check can't do it. |
| `instruction_adherence` | **Rule + judge** | Did it follow stated rules (e.g. always read the appointment back to the caller)? | The read-back check is a deterministic rule (**gate**); conversational nuance is a judge (**advisory**). |

### Suite B — Acoustic / paralinguistic (audio + event timeline)

| Evaluator | Kind | What it checks | Why this kind |
|---|---|---|---|
| `barge_in` ★ | **Signal (HEADLINE)** | Every point the caller starts speaking while the agent is talking, and the agent's **time-to-yield**. Flags fail-to-yield (talks over) and false-yields (stops for a cough). | VAD-derived but a deterministic decision once computed; the flagship correctness behavior. |
| `turn_taking_latency` | **Signal** | Gap between caller end-of-speech and agent response onset. Reports the **distribution (p50/p90/p99)**, not just a mean. | A distribution, not a single verdict — advisory feed into the latency gate. |
| `pitch_prosody` | **Signal** | F0 contour: out-of-natural-range pitch, flat monotone, plus speech rate. | Perceptual proxy for naturalness, not correctness. |
| `entity_intelligibility` | **Signal** | Round-trip STT on the agent's audio: do high-stakes tokens (provider/medication names, dates, confirmation numbers) survive? | If a real STT engine can't recover the token, a caller likely couldn't either. |
| `emotional_appropriateness` | **SER + multimodal judge** | Was the delivered tone appropriate for the moment (calm with an anxious caller, not chirpy on bad news)? Two proxies cross-checked. | See §13 — emotion is perceptual and permanently advisory. |
| `double_talk` | **Signal** | Sustained overlapping speech between channels. | Live-follow-up drop-in; natural backchannels overlap, so advisory. |
| `naturalness_mos` | **Signal** | Non-intrusive MOS estimate of audio quality. | Beyond-scope addition; a saturating perceptual proxy — advisory. |

Both suites emit a **structured per-call scored report plus an aggregate**, and both feed **one** verdict.

---

## 7. Technologies involved

| Layer | Technology | Used for |
|---|---|---|
| Language / runtime | **Python 3.10+** | Whole harness |
| Voice activity detection | **Silero VAD** (via `torch`; `webrtcvad` fallback) | `barge_in`, `turn_taking_latency` segment boundaries |
| Pitch / prosody | **Praat via `parselmouth`** (`librosa.pyin` fallback) | F0 contour, monotone detection, speech rate (De Jong syllable-nuclei) |
| Round-trip STT | **`faster-whisper`** with native `word_timestamps=True` | `entity_intelligibility` re-transcription + word-level localization |
| Word-error rate | **`jiwer`** | Entity-survival / WER scoring |
| Speech emotion recognition | **wav2vec2 SER** — `superb/wav2vec2-base-superb-er` (Hugging Face `transformers`) | `ser_emotion` (objective, offline emotion proxy) |
| Multimodal audio judge | **Gemini 2.5-flash** (`google-genai`, audio bytes via `Part.from_bytes`) | `emotion_appropriateness_mm` (contextual emotion proxy) |
| LLM judges | **Gemini / LLM-as-judge** | `faithfulness`, `instruction_adherence` (judge portion) |
| Overlap / diarization | **`pyannote.audio`** | `double_talk` overlap detection |
| Non-intrusive MOS | **UTMOS / DNSMOS (P.808)** family | `naturalness_mos` |
| Numerics | **`numpy`** | Signal math, percentiles |
| Calibration / drift | **Cohen's/Fleiss' κ + KS test (`scipy.stats`)** | `judge_agreement`, drift detection |
| Testing | **`pytest`** | 20+ known-answer unit tests |

### Two engineering decisions worth calling out

- **We rejected WhisperX** even though it's the "recommended" word-alignment tool: integrating it **force-downgraded `transformers`** and broke the already-working, tested SER emotion metric. We switched to **`faster-whisper` with native `word_timestamps=True`**, which gives sufficient word-level alignment for entity localization without destabilizing a working metric. *The "best" library isn't best if it breaks your suite.*
- **The judge is cached for determinism.** LLM-judge results are cached at `out/.judge_cache/` keyed by `fixture::turn::promptversion::model` at `temperature=0`, so a network judge is reproducible inside CI.

---

## 8. Gate vs. advisory — the ship decision

**SHIP iff there are zero gating failures across all calls. Advisory results never block. ERROR never blocks.** CI reads exit code `0` (SHIP) / `1` (HOLD).

A metric may **gate** only if it is objective, calibrated, and reproducible. Everything perceptual is **advisory** — reported, never blocking. We also **gate on the tails (p95/p99), not the mean**: a healthy average latency hides a long tail where a handful of callers wait three seconds, and those are the calls that get abandoned.

| Metric | Gating | Why |
|---|---|---|
| `tool_call_ordering` | **gate** | Deterministic state reducer incl. the reschedule-trap invariant — a real correctness bug, not a style preference. |
| `task_success` | **gate** | Deterministic: final tool call vs. ground-truth success criteria. No proxy. |
| `instruction_adherence_rule` | **gate** | Deterministic read-back check on ground-truthed critical entities. |
| `barge_in` | **gate** | The headline behavior; a real barge-in miss is a real defect. |
| `entity_intelligibility` | **gate** | Round-trip STT on ground-truthed entities — if STT can't recover it, a caller likely couldn't either. |
| `faithfulness` | advisory | LLM judge; unproven proxy until calibration (κ) trusts it. |
| `instruction_adherence_judge` | advisory | LLM judge for conversational nuance; advisory until κ clears the bar. |
| `turn_taking_latency` | advisory | A distribution, not a per-call verdict; feeds the latency gate once a cutoff is fixed. |
| `latency_thresholds` | advisory | Deterministic arithmetic, but the threshold itself is a judgment call. |
| `pitch_prosody` | advisory | Perceptual proxy for naturalness, not correctness. |
| `ser_emotion` | advisory | Objective SER, but a noisy proxy — IEMOCAP humans agree only at Fleiss' κ≈0.27–0.48. Non-promotable. |
| `emotion_appropriateness_mm` | advisory | Multimodal judge; LLM judges drift and are noisy. Permanently advisory (§13). |
| `double_talk` | advisory | Natural backchannels overlap; reports duration/ratio. |
| `naturalness_mos` | advisory | Non-intrusive MOS saturates above ~4; can't separate "good" from "excellent." |

---

## 9. Evaluating the evaluators (calibration & drift)

**Trusting a metric enough to gate:** a judge is promotable only if it agrees with human labels at **Cohen's/Fleiss' κ ≥ 0.6 (we prefer ≥ 0.8)**. Calibration effectively *gates the gate* — until a judge clears the bar it runs advisory.

**Without a giant labeling budget:** we calibrate against a **small frozen golden set** of human-labeled clips and compute κ (`calibration/judge_agreement.py`). The live follow-up (calibrating emotion against three hand-labeled clips) plugs straight into this.

**Judge drift:** LLM judges change under us. We defend with (1) the `temperature=0` reproducibility cache, (2) **version stamps** on every `MetricScore`, and (3) a **KS test** comparing the judge's current score distribution to the frozen golden distribution — if it drifts, we detect it instead of trusting stale numbers.

---

## 10. The registry — adding a new evaluator

New evaluators drop in **without touching the runner** (assessment.md line 38). Implement `MetricScore`-returning callable and register it:

```python
# metrics/acoustic/my_metric.py
from metrics.base import MetricScore, Status
from metrics.registry import register

@register(name="my_metric", suite="acoustic", kind="signal", gating=False)
def my_metric(ctx) -> MetricScore:
    value = compute_something(ctx.agent_audio, ctx.events)   # your logic
    ok = value < THRESHOLD
    return MetricScore(
        status=Status.PASS if ok else Status.FAIL,
        gating=False,                 # advisory
        score=value,
        details={"threshold": THRESHOLD},
        versions={"my_metric": "1.0.0"},
    )
```

The runner discovers it automatically. `kind` is one of `deterministic` / `signal` / `judge`; `gating` declares whether it can block a ship. **ERROR ≠ FAIL** is enforced by the registry: if your metric raises, it's recorded as `ERROR` and never blocks the ship.

---

## 11. Adding your own fixtures

1. Copy `fixtures/TEMPLATE/` to `fixtures/<your_scenario>/`.
2. Drop in `caller.wav`, fill `events.jsonl` (timed events), `scenario_db.json` (tool world), and `expected.json` (ground truth).
3. Run `python -m runners.open_loop --fixtures fixtures/<your_scenario>/`.

The validator will reject the fixture (with a clear message) if any file is missing or malformed, so a bad fixture never silently produces wrong numbers.

---

## 12. Tests

```bash
pytest -q
```

20+ **known-answer** unit tests. Two conventions worth knowing:

- **Tests derive inputs from the real code constants** (e.g. `FAIL_TO_YIELD_THRESHOLD_SEC = 1.0`), never from hard-coded guesses — so the test and the code can't drift apart.
- **`barge_in` is tested two ways**: an *exact-math* test with injected VAD segments (synthetic tones don't trigger the speech-trained Silero VAD), plus a *real-fixture* test with ±50 ms tolerance for end-to-end behavior.

---

## 13. Honesty about uncertainty

- **Emotion is permanently advisory.** We run two proxies — an offline wav2vec2 SER classifier (objective, calibratable) and a Gemini multimodal judge (contextual) — and treat their **disagreement as a trust signal** that flags a turn for human review. But even human annotators only reach Fleiss' κ≈0.27–0.48 on emotion (IEMOCAP), so **no machine judge can honestly gate a release on emotion.** Neither is ever promoted.
- **Perceptual proxies stay advisory.** Pitch, speech rate, and MOS approximate "naturalness," not correctness. MOS in particular saturates above ~4.0 and can't reliably separate "good" from "excellent."
- **Judges drift.** Caching, version stamps, and the KS drift test keep them honest; when they move, we detect it rather than trust them.

---

## 14. Offline vs. online

**Offline (this harness) is a gate:** deterministic, blocking, run in CI before merge on fixed fixtures with known answers. It answers *"is this build correct?"* with authority.

**Online monitoring is a smoke alarm:** production has no ground truth, so `monitoring/production_proxies.py` reuses the same measurement code but can only track **proxies** — latency distributions, VAD-derived barge-in behavior, emotion-disagreement rates — and **alert**, never block. It answers *"has live behavior drifted from what we validated?"* Things you can only learn live: real interruption patterns, real caller emotional range, tail-latency under production load, and STT intelligibility on real acoustic conditions.

---

*Design write-up (Part 1): `docs/design_writeup.pdf`. Combined sample report: `out/report.html`.*
