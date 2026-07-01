# CLAUDE.md — VoxGate (Voice Agent Eval System)

> **Repo name:** voxgate-voice-eval
> **This file is auto-loaded by Claude Code every session.** It is the source of truth for
> DECISIONS and CONSTRAINTS. Requirements live in `docs/assessment.md`; full structure in
> `docs/architecture.md`; live state in `docs/PROGRESS.md`, `docs/CONTEXT.md`,
> `docs/ERRORS.md`. Read the working-memory files first every session (see protocol below).

---

## What this is
A **post-hoc evaluation system** for an EXISTING inbound clinic scheduling voice agent
(books / reschedules / cancels appointments by phone). It decides, automatically and
reproducibly, whether a given version of that agent is good enough to ship. It scores two
axes and fuses them into one verdict:

- **Semantic / behavioral** — what the agent *says and does*: tool calls, faithfulness, task success, instruction adherence.
- **Acoustic / paralinguistic** — how it *sounds*: barge-in, turn-taking latency, prosody, emotion, intelligibility.

**Output:** a two-tier ship / don't-ship verdict for CI, plus per-call and aggregate reports.

## What we are NOT building (hard scope boundaries)
- **NOT the voice agent.** It is the system under test — integrate/replay against it, never build or ship it. (No `agent/` module.)
- **NO closed-loop bot-to-bot runner.** Only open-loop, fixed-clock fixture replay. Closed-loop is one contrast sentence in the writeup, not code.
- **NO fake infra.** Real broker-backed DLQ, distributed workers, multi-key rotation, and judge self-consistency sampling are *documented design decisions only* (assessment Category 2). Explain them in the README; do not stub/mock them.
- **Assume fixtures contain REAL 2-channel audio + event timestamps** (assessment Part 3 states this). Never fabricate audio-timestamp data to feed the alignment code.

## Design philosophy (this is the graded part — treat as first-class)
1. **Reproducibility** of a non-deterministic real-time system → open-loop fixed-clock replay; a barge-in test means the same thing every run.
2. **Taxonomy** → semantic vs acoustic, each split into deterministic / model-judge / signal-processing; defend every placement.
3. **Proxy validity** → pitch/naturalness/emotion are perceptual proxies. Per metric decide: trustworthy enough to **gate**, or **advisory** only. Evaluate your own evaluators (judge vs a small human set) without a big labeling budget. Detect judge **drift**.
4. **Thresholds & gating** → turn noisy scores into one decision; a flaky judge must not fail a good deploy (or pass a bad one).
5. **Offline vs online** → what only production traffic teaches; what to monitor live.

## Core contracts — build these FIRST and keep them stable
```python
# metrics/base.py
class MetricKind(str, Enum):  DETERMINISTIC; JUDGE; SIGNAL
class Status(str, Enum):      PASS; FAIL; ERROR; SKIPPED   # ERROR = evaluator broke, ≠ FAIL
class Gating(str, Enum):      GATE; ADVISORY

@dataclass
class MetricScore:
    call_id: str
    metric: str
    kind: MetricKind
    status: Status
    gating: Gating
    score: float | None          # 0..1 or metric-specific; None if N/A
    details: dict                 # evidence for the verdict
    evaluator_version: str
    judge_prompt_version: str | None
    schema_version: str
    @property
    def key(self):                # idempotency: re-runs upsert, not duplicate
        return (self.call_id, self.metric, self.evaluator_version, self.judge_prompt_version)

class BaseMetric:
    name: str; version: str
    kind: MetricKind
    default_gating: Gating
    requires_ground_truth: bool   # False → can also run on live traffic (monitoring)
    def compute(self, ctx: "MetricContext") -> MetricScore: ...

# context/metric_context.py  — THE ALIGNMENT BACKBONE
@dataclass
class MetricContext:
    call_id: str
    sr: int
    audio_agent: "np.ndarray"     # agent channel
    audio_caller: "np.ndarray"    # caller channel
    transcript: list["Turn"]      # Turn: speaker, t_start, t_end, text, asr_confidence?
    tool_events: list["ToolEvent"]# name, args, result, t
    events: list["Event"]         # authored markers, e.g. interrupt_start @ t
    expected: dict                # expected.json: tool sequence + invariants + critical entities
    scenario_db: dict             # initial ground-truth DB state
    # CANONICAL CLOCK = audio sample index at `sr`; ALL times are seconds on this clock.

# metrics/registry.py
REGISTRY: list[BaseMetric] = []
def register(cls):  REGISTRY.append(cls()); return cls        # drop-in; no runner edits

def run(ctx, sampler=None) -> list[MetricScore]:
    scores = []
    for m in REGISTRY:                                        # phase 1: deterministic + signal on ALL
        if m.kind is MetricKind.JUDGE: continue
        scores.append(_safe(m, ctx))
    for m in REGISTRY:                                        # phase 2: judges per sampling policy
        if m.kind is not MetricKind.JUDGE: continue
        if sampler and not sampler.should_run(m, ctx, scores): continue
        scores.append(_safe(m, ctx))
    return scores

def _safe(m, ctx):                                            # per-evaluator isolation
    try: return m.compute(ctx)
    except Exception as e: return MetricScore(status=Status.ERROR, details={"exc": repr(e)}, ...)
```

## Metric taxonomy (kind + default gating)
| Metric | Suite | Kind | Default gating | Ground truth? |
|---|---|---|---|---|
| task_success | semantic | deterministic | **gate** | yes (expected + scenario_db) |
| tool_call_ordering | semantic | deterministic | **gate** | yes |
| faithfulness | semantic | judge | advisory → gate iff trusted | uses tool results |
| instruction_adherence | semantic | rule + judge | gate (rule) / advisory (judge) | partial |
| barge_in *(headline)* | acoustic | signal | **gate** | no |
| turn_taking_latency | acoustic | signal | advisory (report distribution) | no |
| latency_thresholds | acoustic | deterministic | advisory (until promoted) | no |
| pitch_prosody | acoustic | signal | advisory | no |
| emotional_appropriateness | acoustic | judge | **advisory always** | no |
| entity_intelligibility | acoustic | signal (STT+WER) | **gate** | yes (critical entities) |
| double_talk | acoustic | signal | advisory | no |

## Invariants that MUST hold
- **Reschedule trap:** new slot secured BEFORE old released; a mid-sequence failure must NEVER leave the caller with zero appointments. `tool_call_ordering` must check the *intermediate* zero-appointment state, not just final order.
- **Clock-join is the backbone:** one canonical clock (audio sample index). Every acoustic metric reads times from `MetricContext`; it never re-derives them. Test alignment first and hardest.
- **Advisory ≠ gate:** only deterministic + calibration-trusted metrics may hard-gate. Judges start advisory until their kappa ≥ threshold.
- **Category-1 features are additive only:** optional fields, runner wrappers that fire on error, one advisory evaluator, report breakdowns. Never rewrite a metric, the clock-join, or the gate. Sampling defaults to 100% coverage on the fixture set; `latency_thresholds` is advisory.

## Tech / libraries
- Python 3.11+, **uv** for deps, **pydantic** for data models, **pytest** for tests.
- Audio/DSP: `numpy`, `soundfile`, `librosa`, `parselmouth` (Praat F0), `silero-vad` or `webrtcvad`, `pyannote.audio` (optional, double-talk).
- STT round-trip: `faster-whisper`; WER: `jiwer`.
- Judges: OpenAI / Anthropic SDK with structured (pydantic) outputs.
- Stats: `scipy.stats` (KS test), `scikit-learn` (`cohen_kappa_score`).

## Commands
```bash
uv sync                                             # install
uv run pytest -q                                    # tests
uv run python -m eval_system.run \                  # score a fixture set → reports
    --fixtures fixtures/ --out out/
uv run python -m eval_system.run --metrics faithfulness,barge_in   # subset
```

## Conventions
- New metric = drop a file in `metrics/semantic/` or `metrics/acoustic/` + `@register`; zero runner edits.
- Every metric returns the same `MetricScore`. Both suites feed **one** report.
- Report = per-call `metrics.json` + an aggregate + the deterministic / judge / error-rate / judge-agreement split + gate-vs-advisory verdict.

## Working memory — session protocol (READ + UPDATE every session)
State that outlives the context window lives in three files. They are the source of truth
for "where we are." Do not re-derive from scratch.

- `docs/PROGRESS.md` — build checklist. `[ ]` todo · `[~]` in progress · `[x]` done. Keep a "▶ Current step" pointer at the top.
- `docs/ERRORS.md` — append-only: Symptom → Root cause → Fix → Prevention. Check it before retrying anything that broke.
- `docs/CONTEXT.md` — a COMPRESSED snapshot (≤1 page): architecture state, decisions, what's built, open questions, next 3 steps. REWRITE it (don't append) to keep it short.

Protocol:
1. **Session start:** read `CONTEXT.md` → `PROGRESS.md` → `ERRORS.md` before acting.
2. **After each completed step:** tick `PROGRESS.md` and move the ▶ pointer.
3. **On any error:** append to `ERRORS.md` before moving on.
4. **When context feels full, before `/compact`, or when I say "checkpoint":** rewrite `CONTEXT.md` so a cold session can resume from it alone.
5. Never let these drift from reality — they are how you recover after compaction.

## Recommended build order
1. Contracts: `base.py`, `MetricContext`, `registry.py`, `MetricScore`.
2. Fixture format + 2–3 synthetic fixtures incl. `reschedule_trap`.
3. `context/metric_context.py` clock-join **+ tests** ← riskiest, do first.
4. Semantic suite (deterministic first, then `faithfulness` judge).
5. Acoustic suite (`barge_in` first — headline).
6. Calibration (`judge_agreement`, `drift`).
7. Gating + report.
8. Validators, sampling, monitoring hook (`requires_ground_truth` flag).
9. README + design writeup + Category-2/3 design notes.

## Definition of Done (from the rubric + deliverables)
- [ ] **Design writeup (~2pp)** covering all 5 Part-1 questions — opinionated, circulate-worthy.
- [ ] **Suite A runnable:** task_success, tool_call_ordering (**catches reschedule trap**), faithfulness, instruction_adherence; each with a defended deterministic-vs-judge choice; per-call + aggregate report.
- [ ] **Suite B runnable:** barge_in (**correct audio↔event alignment**), turn_taking_latency (distribution, not just mean), pitch_prosody (F0 + rate), emotional_appropriateness (honest about trust), entity_intelligibility (round-trip STT on critical entities); identical `MetricScore` shape.
- [ ] **Registry:** a new metric drops in with zero runner edits.
- [ ] **Calibration:** judge_agreement (kappa vs a small human set), drift (golden set + KS).
- [ ] **Gating:** two-tier, `pass^k`; a flaky judge cannot fail a good deploy.
- [ ] **Combined report** over a fixture set + explicit **gate-vs-advisory** list with rationale.
- [ ] **README:** how to run + the *why* behind the structure.
- [ ] **Category-2** items documented as design decisions (no fake infra).
- [ ] **Category-3** PHI/BAA paragraph inserted after the faithfulness deterministic-vs-judge justification.
- [ ] **Live-follow-up ready:** `double_talk` drop-in OR emotion calibration against 3 human-labeled clips.
