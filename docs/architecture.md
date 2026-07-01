# Voice Agent Eval System — Architecture

> Repo location: `docs/architecture.md`.

A post-hoc scoring system over recorded call artifacts (2-channel audio + transcript +
tool-call/result log + event timeline). Borrows the *scoring* layer of ServiceNow's EVA
(registry, MetricContext, MetricScore, metrics) — **not** its live-generation harness.

## Scope decisions baked in
- **No `agent/`** — the agent-under-test is the given system; integrate/replay, don't build it.
- **No closed-loop runner** — only open-loop fixed-clock fixture replay. Closed-loop survives as one contrast sentence in the Part 1 writeup.
- **Category-1 production-readiness features added** — tagged `←C1(n)`; all additive (see guarantee below).
- **Category-2/3 features are writeup-only** — DLQ, distributed workers, key rotation, judge self-consistency (C2); PHI/BAA note (C3). No fake infra.

## Directory layout
```
eval-system/
├── runners/
│   └── open_loop.py          # fixed-clock fixture replay (reproducibility)
├── fixtures/
│   ├── <scenario>/
│   │   ├── caller.wav         # pre-rendered caller audio
│   │   ├── events.jsonl       # {t: 4.2, event: interrupt_start} — authored timeline
│   │   ├── scenario_db.json   # ground-truth state for the tool executor
│   │   └── expected.json      # expected tool sequence + invariants + critical entities
│   └── reschedule_trap/       # hold-new → confirm → release-old; injected failure
├── tools/                     # deterministic tool executor + scenario DB
├── context/
│   └── metric_context.py      # joins 2-ch audio + logs + markers onto ONE clock
│                              # + carries optional per-turn asr_confidence   ←C1(6)
├── validators/                # trust gate: malformed/bad runs excluded, not scored
│                              # + retry counter → QUARANTINED after N fails   ←C1(4)
├── metrics/
│   ├── registry.py            # plug-in point; new metrics need no runner edits
│   │                          # + per-evaluator try/except (ERROR ≠ FAIL)    ←C1(1)
│   │                          # + two-phase: deterministic on all → sample judges ←C1(5)
│   ├── base.py                # BaseMetric.compute(ctx) -> MetricScore
│   │                          # + status{PASS,FAIL,ERROR,SKIPPED}            ←C1(1)
│   │                          # + idempotency key + version stamps           ←C1(2,3)
│   ├── sampling.py            # NEW — stratified judge-coverage policy        ←C1(5)
│   ├── semantic/              # ── Suite A ──
│   │   ├── task_success.py            # deterministic (DB state hash)
│   │   ├── tool_call_ordering.py      # deterministic (+ reschedule trap)
│   │   ├── faithfulness.py            # LLM judge (+ reads asr_confidence)    ←C1(6)
│   │   └── instruction_adherence.py   # deterministic rule + judge for nuance
│   └── acoustic/             # ── Suite B ──
│       ├── barge_in.py               # signal (VAD 2-ch + markers) — HEADLINE
│       ├── turn_taking_latency.py    # signal (gap distribution p50/p90/p99)
│       ├── latency_thresholds.py     # NEW — determ. FTL + silence, advisory ←C1(7)
│       ├── pitch_prosody.py          # signal (librosa/parselmouth F0 + rate)
│       ├── emotional_appropriateness.py  # multimodal judge — always advisory
│       ├── entity_intelligibility.py # round-trip STT (faster-whisper + jiwer)
│       └── double_talk.py            # ← live follow-up drops in here
├── calibration/
│   ├── judge_agreement.py     # Cohen's kappa vs small human set → trust tier
│   └── drift.py               # frozen golden set + KS test (fed by C1(3) versions)
├── gating/
│   └── gate.py                # two-tier: hard-gate deterministic+trusted;
│                              #   advisory otherwise; pass^k; tail thresholds
├── report/
│   └── combine.py             # per-call + aggregate + verdict
│                              # + split: deterministic / judge / error-rate / agreement ←C1(8)
├── monitoring/
│   └── production_proxies.py  # ground-truth-free subset on live traffic
└── README.md
```

## Runtime flow (one run)
The **open-loop runner** drives the agent-under-test (external) on a fixed clock through a
fixture; the **tool executor** answers calls deterministically from the scenario DB;
artifacts land on disk (2-ch audio + transcript + tool log + event timeline). The **context
builder** joins them onto one clock (plus optional ASR confidence). **Validators** confirm
the recording is well-formed; bad runs are retried, then quarantined, never scored. The
**registry** runs deterministic metrics on every call, consults the **sampling policy** for
judge coverage, and wraps each evaluator so a crash yields `ERROR`, not `FAIL`.
**Calibration** decides which judges may gate; **gating** fuses the trusted set into a
two-tier verdict; the **report** writes per-call + aggregate + the split. In parallel,
**monitoring** runs the ground-truth-free subset on production traffic.

## Category-1 additions (additive, non-invasive)
| # | Feature | Touches | Why it doesn't change functionality |
|---|---|---|---|
| 1 | Exception isolation, `ERROR` ≠ `FAIL` | `registry.py` + `base.py` status | Metrics unchanged; ERROR only where a metric would have crashed the run |
| 2 | Idempotent score keys | key on `base.py`; `combine.py` upserts | Write-path keying; re-runs overwrite, not duplicate |
| 3 | Version stamping | fields on every `MetricScore` | Metadata; makes `drift.py` trend lines falsifiable |
| 4 | Retry-then-quarantine | `validators/` | Only affects records already failing to process |
| 5 | Stratified sampling | `metrics/sampling.py` | Deterministic stays 100%; **defaults to full judge coverage** on the fixture set |
| 6 | ASR-confidence into faithfulness | transcript → `metric_context` → `faithfulness.py` | Optional field; absent → unchanged behavior |
| 7 | Deterministic latency thresholds | `metrics/acoustic/latency_thresholds.py` | Drop-in; **advisory by default** |
| 8 | Aggregate split | `report/combine.py` | Presentation-layer breakdown; scoring/gating math unchanged |

**Non-invasive guarantee:** on a clean run with full coverage, the core loop
(`metric_context` → metrics → `gate` → verdict) produces the **same verdict** as before.

## Three load-bearing decisions
1. **Reproducibility** — open-loop runner cuts the feedback loop; fixed-clock caller → barge-in lands identically every run.
2. **Trust** — calibration gates the gate; a judge earns hard-gate authority only after its kappa clears a threshold.
3. **One verdict** — every metric emits the same `MetricScore` into the same registry and report.

## Highest-risk component — test first and hardest
`context/metric_context.py` (the clock-join). Every acoustic metric is downstream; bad
alignment produces wrong numbers with no error. Canonical clock = audio sample index at a
known rate; watch codec delay and channel offset.

## Component reference — function + technology
### Input & recording
| Component | Function | Tech |
|---|---|---|
| Fixtures | Authored inputs: caller audio, interrupt timeline, ground-truth DB, expected sequence/invariants | WAV, JSON, JSONL |
| open_loop runner | Replays a fixture against the external agent on a fixed clock; records the call | Python asyncio, WebSocket, soundfile/pydub |
| tools executor | Answers the agent's tool calls deterministically from the scenario DB | Python state machine over JSON |
| Recorded artifacts | The eval input: 2-ch audio, transcript, tool log, event timeline | stereo WAV, JSONL |

### Alignment · trust · framework
| Component | Function | Tech |
|---|---|---|
| metric_context | Backbone: joins channels, transcript, tool log, markers onto one clock (+ ASR confidence) | Python, numpy |
| validators | Preflight sanity (channels, timeline parse, no clipping) + retry→quarantine | Python, soundfile, jsonschema |
| registry | Plug-in discovery; ERROR isolation; deterministic-then-sample execution | Python registry |
| base / MetricScore | The one contract every metric returns (status, versions, key) | Python, pydantic |
| sampling | Stratified judge coverage: 100% on flagged calls, sample the rest | Python |

### Suite A · semantic
| Component | Function | Tech |
|---|---|---|
| task_success | Did the call reach the caller's goal? Final DB state vs expected | deterministic; hash/set compare |
| tool_call_ordering | Right tool/args/sequence; reschedule-trap invariant (never zero appts) | deterministic state reducer |
| faithfulness | Only facts grounded in tool results? | LLM-as-judge (GPT/Claude), structured output |
| instruction_adherence | Followed the agent's rules (e.g. confirm back)? | deterministic rules + LLM judge for nuance |

### Suite B · acoustic
| Component | Function | Tech |
|---|---|---|
| barge_in *(headline)* | Detect caller-over-agent overlap; measure time-to-yield; flag fail-to-yield & false yields | 2-ch VAD (Silero/webrtcvad) + numpy |
| turn_taking_latency | Caller-end → agent-onset gap distribution (p50/p90/p99) | VAD + numpy |
| latency_thresholds | Deterministic FTL + inter-turn silence vs thresholds; advisory | timestamp arithmetic |
| pitch_prosody | F0 contour (out-of-range / monotone) + speech rate | librosa / parselmouth (Praat) |
| emotional_appropriateness | Tone appropriate for the moment? Advisory | audio emotion classifier / multimodal audio-LLM |
| entity_intelligibility | Round-trip STT; verify critical tokens survive | faster-whisper + jiwer (WER) |
| double_talk | Overlapping-speech detection (live follow-up) | overlap VAD / pyannote |

### Calibration · gating · output · monitoring
| Component | Function | Tech |
|---|---|---|
| judge_agreement | Kappa of each judge vs a small human set → trust tier | scikit-learn, numpy |
| drift | Golden set re-run + KS test on score distribution | scipy.stats + version stamps |
| gate | Two-tier decision; hard-gate trusted+deterministic; pass^k; tail thresholds | Python policy |
| report/combine | Per-call + aggregate + det/judge/error-rate/agreement split → verdict | Python, JSON |
| production_proxies | Ground-truth-free subset on live traffic (time-to-yield drift, abandonment, transfer rate) | Python; telephony/CDR |

## Deliverable map
- **D1 (writeup):** Part-1 argument; Category-2 items as documented design decisions; Category-3 PHI/BAA paragraph after the faithfulness justification.
- **D2 (two suites + README):** `metrics/semantic/`, `metrics/acoustic/`, `registry.py`, `README.md`.
- **D3 (combined report + gate-vs-advisory):** `report/combine.py` + `gating/gate.py`.

## Test-first targets
- **Clock-join:** known offsets → assert alignment within tolerance.
- **barge_in:** synthetic overlap → correct time-to-yield; injected cough → flagged false yield, not a yield.
- **tool_call_ordering:** reschedule_trap mid-sequence failure → "never zero appointments" invariant fires.
- **gating:** a low-kappa judge must NOT be able to hard-gate.
- **registry:** drop a new metric in a folder → runs with zero runner edits.
- **report:** semantic + acoustic emit the identical `MetricScore` schema.
- **C1(1):** inject a crashing metric → `ERROR`, suite continues.
- **C1(2):** re-run a call → record overwritten, not duplicated.
- **C1(5):** flagged call always judged; coverage 100% → verdict identical to pre-sampling.
- **C1(7):** advisory latency score present but does not move the gate until promoted.
