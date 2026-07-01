# VoxGate — Voice Agent Evaluation Harness

Post-hoc evaluation system for an existing inbound clinic-scheduling voice agent.
Scores recorded calls on two axes — **semantic** (task success, tool-call ordering,
faithfulness, instruction adherence) and **acoustic** (barge-in, turn-taking latency,
prosody, emotion, entity intelligibility) — and fuses them into a two-tier
ship / don't-ship verdict.

See `CLAUDE.md` for the design constraints and `docs/architecture.md` for the full
component map. This file will be expanded with run instructions and the design
rationale as the build progresses (Phase 9 of `docs/PROGRESS.md`).

## Quickstart

```bash
uv sync --extra dev            # core contracts + tests only
uv run pytest -q               # run tests
uv run python -m eval_system.run --fixtures fixtures/ --out out/
```

Optional extras: `--extra acoustic` (librosa/parselmouth/whisper/vad),
`--extra judge` (Anthropic/OpenAI SDKs), `--extra stats` (scipy/scikit-learn for
calibration). Core contracts and the semantic suite run without them.
