# Take-Home — Voice Agent Evaluation System

**Role:** Voice AI Engineer (with strong general SWE) · **Time:** ~1 focused day.

> Repo location: `docs/assessment.md`. This is the requirements source of truth.
> `CLAUDE.md` holds our decisions; `docs/architecture.md` holds the structure we settled on.

## The premise
You are **not** building a voice agent. We already have one — an inbound scheduling agent
for a clinic that books, reschedules, and cancels appointments over the phone. Your job is
the harder and more valuable one: **build the system that tells us, automatically and
reproducibly, whether a given version of that agent is good enough to ship.**

A good agent has to get two very different things right at once:

- **What it says and does** — correct tool calls, no hallucinated availability or confirmation numbers, accomplishes the caller's actual goal, follows its instructions.
- **How it sounds and behaves** — yields promptly when interrupted, responds without awkward latency, speaks with natural pitch and pacing, and carries the right emotional tone for the moment (calm and warm with an anxious caller, not chirpy when delivering bad news).

Most teams only test the first. We want both, and we want you to treat the second as a
real engineering problem rather than a vibe check.

---

## Part 1 — Design (write ~2 pages — this is the most important deliverable)
Don't give us a feature list. Give us an opinionated argument. Address:

1. **Reproducibility of a real-time, non-deterministic system.** A voice agent's output changes run to run, and its behavior depends on when things happen (when the caller interrupts, how long a pause runs). How do you make evaluation deterministic and repeatable anyway? In particular, how would you design a simulated caller / fixture format that lets you replay the exact same interruption timing and conversational pressure every time, so a barge-in test means the same thing across runs?
2. **A taxonomy of what's worth measuring.** Lay out the categories. We expect at minimum a split into semantic/behavioral vs acoustic/paralinguistic, and within each, which checks are deterministic (assertable in code), which need a model judge, and which need a signal-processing metric. Defend where each thing lands.
3. **The validity problem — the part we care about most.** Pitch range, "naturalness," and especially emotion are perceptual. You will end up using proxies: an F0 estimator stands in for "pleasant pitch," an emotion classifier or multimodal judge stands in for "right tone." Every one of these proxies has imperfect correlation with what a human actually perceives. So:
   - How do you decide whether a given metric is trustworthy enough to gate a deploy versus merely advisory?
   - How do you evaluate your own evaluators — i.e., measure how well your automated emotion/quality scores agree with human judgment — without a giant labeling budget?
   - LLM-as-judge and emotion models are themselves noisy and drift over time. How do you keep a judge stable and detect when it has drifted?
4. **Thresholds and gating.** How do you turn a pile of noisy scores into a single ship / don't-ship decision for a CI pipeline? How do you avoid a flaky judge failing a deploy that's actually fine (and vice versa)?
5. **Offline vs online.** What can you only learn from production traffic that batch fixtures can't tell you, and what would you monitor live?

## Part 2 — Implementation A: the semantic / LLM-output eval suite
Build a runnable suite that scores the **transcript + tool-call + tool-result record** of
each call. Make it **registry-based** so new evaluators drop in without touching the runner.
At minimum, implement evaluators for:

- **Tool-call correctness & ordering** — right tool, right arguments, right sequence. Include the **reschedule trap**: the new slot must be secured before the old one is released, and a mid-sequence failure must not leave the caller with zero appointments.
- **Faithfulness / no hallucination** — did the agent surface only providers, times, and confirmation details that actually appear in tool results? (Good candidate for a model judge — say why.)
- **Task success** — did the call accomplish the caller's stated goal?
- **Instruction adherence** — did it follow the agent's stated rules (e.g., always confirm the appointment back to the caller)?

For every evaluator, make an explicit, defended choice of **deterministic vs LLM-as-judge**,
and emit a structured, per-call scored report plus an aggregate.

## Part 3 — Implementation B: the acoustic / voice-quality eval suite
Build a second suite that scores the **audio** of each call. The two channels and the
event-log timestamps let you align the audio timeline with what was said and done.
Implement, at minimum:

- **Interruption / barge-in handling.** Detect each point where the caller starts speaking while the agent is talking, and measure the agent's time-to-yield (how long until its audio stops). Flag both failures to yield (talks over the caller) and false yields (stops for a cough or background noise). **This is the headline metric — get the timeline alignment right.**
- **Turn-taking latency.** Gap between caller end-of-speech and the agent's response onset. Report the distribution, not just a mean.
- **Pitch / prosody.** Extract an F0 contour and flag problems — out-of-natural-range pitch, or a flat monotone delivery. Also report speech rate.
- **Emotional appropriateness.** Given the call context, was the agent's delivered tone appropriate for the moment? Use an audio emotion classifier or a multimodal judge — your call — and be honest in your writeup about how much you trust it.
- **Intelligibility of critical entities.** Re-transcribe the agent's audio (round-trip STT) and check that high-stakes tokens — provider names, medication names, dates/times, confirmation numbers — survive. A booking the caller can't make out is a failed call even if the tool call was perfect.

You're free to reach for whatever libraries fit (e.g. things in the librosa / Praat /
non-intrusive-MOS / diarization / Whisper families) — choosing well is part of what we're
assessing. Emit a structured per-call report consistent with Part 2 so both suites can feed
one verdict.

---

## Deliverables
1. The design writeup (Part 1).
2. Two runnable suites (Parts 2 & 3) with a README: how to run them, and the *why* behind your structure.
3. A combined report for a fixture set, plus a clear statement of which metrics you'd let **gate** a deploy versus keep **advisory**, and why.

## What we're assessing
| Dimension | Strong looks like |
|---|---|
| Eval philosophy | Treats reproducibility, proxy validity, and "evaluating the evaluators" as first-class problems |
| Semantic evals | Correct tool/ordering checks; principled deterministic-vs-judge calls; catches the reschedule trap |
| Acoustic evals | Correct audio↔event timeline alignment; defensible interruption, pitch, and emotion metrics |
| Honesty about uncertainty | Knows which metrics to trust, calibrates them, and says where they break down |
| Engineering | Clean, extensible (registry) design; runnable; consistent reporting across both layers |
| Communication | A writeup we'd actually circulate internally |

## Live follow-up (~30 min — extend your own submission)
Add a new acoustic evaluator (e.g. detect overlapping speech / double-talk), **or** calibrate
your emotion metric against three human-labeled clips we hand you and report the agreement.
This confirms you own what you built.
