# ERRORS — VoxGate incident log

> Append-only. On ANY error, add an entry BEFORE moving on. Check here before retrying
> anything that broke. Never delete entries — they are how a cold session avoids repeats.

## Template (copy for each entry)
```
### <YYYY-MM-DD> — <short title>
- **Symptom:** what was observed (message, wrong output, failing test).
- **Root cause:** the actual underlying reason (not the surface symptom).
- **Fix:** what resolved it.
- **Prevention:** the guardrail/test/convention that stops a recurrence.
```

---

### 2026-06-30 — acoustic extras fail to install on Python 3.13.5 / Windows
- **Symptom:** `uv sync --extra acoustic` fails twice: (1) `librosa` pulls in
  `numba==0.53.1` which requires `llvmlite==0.36.0`, whose sdist build refuses
  Python >=3.10 ("Cannot install on Python version 3.13.5"); (2) after pinning
  a modern `numba`, `webrtcvad==2.0.10` fails to build — "Microsoft Visual C++
  14.0 or greater is required" (no C compiler on this machine, no prebuilt
  Windows wheel for this Python).
- **Root cause:** librosa's declared numba floor predates Python 3.13 support;
  webrtcvad is a C extension with no wheel for this interpreter/OS combo and
  this machine has no MSVC Build Tools installed.
- **Fix:** added an explicit `numba>=0.61` pin to the `acoustic` extra in
  `pyproject.toml` (forces a build that supports 3.13). Replaced `webrtcvad`
  with `silero-vad` (CLAUDE.md already lists it as an equally acceptable VAD
  choice) — pure Python + ONNX via `onnxruntime`, no compiler needed. Pulled in
  `torch`/`torchaudio` as a side effect of `silero-vad`.
- **Prevention:** when running `uv sync` with extras, always pass every extra
  needed together (e.g. `--extra acoustic --extra dev`) — passing one alone
  uninstalls packages from extras not listed (it dropped `pytest` once).

### 2026-06-30 — barge_in_basic fixture's "genuine barge-in" had no real overlapping audio
- **Symptom:** Real silero-VAD run against `barge_in_basic` found zero barge-ins at all
  (`BargeInMetric().compute(ctx)` → `barge_ins: []`), even though the fixture is meant to
  encode two scenarios. Direct inspection (`fixture.audio_caller[8.0*sr:9.0*sr]`) showed
  literal all-zero samples in the window the `barge_in_start`(@8.43s)/`agent_yield`(@8.88s)
  events claimed the caller was speaking over the agent.
- **Root cause:** In `scripts/generate_fixtures.py::build_barge_in_basic`, the "genuine
  barge-in" events were authored at absolute times, but the caller's actual line
  ("Actually, never mind...") was rendered afterward via sequential `b.say(..., gap_before=0.05)`
  off `b.cursor = t_yield2` — i.e. placed AFTER the agent's clip ended, not overlapping it.
  This is exactly the fabricated-alignment failure mode CLAUDE.md's scope note warns
  against: an authored event timestamp with no real audio behind it. (The cough/false-yield
  scenario was fine — that noise burst genuinely overlaps the agent audio; VAD legitimately
  just doesn't classify it as speech, which is a real proxy-validity finding, not a bug.)
- **Fix:** added `CallBuilder.say_at(speaker, text, t_start, ...)` — renders and places a
  clip at an ABSOLUTE time (mirrors the existing `inject_noise` pattern) instead of
  cursor-relative. `build_barge_in_basic` now places the caller's barge-in line via
  `say_at(..., t_barge_in)`, so it genuinely overlaps the last 0.45s of the agent's cut-short
  utterance. Regenerated the fixture; `BargeInMetric` now detects the real overlap.
- **Prevention:** when authoring a fixture event that claims two channels overlap, verify
  it by inspecting the actual samples (not just trusting the event's `t` value) before
  treating the fixture as ground truth for a metric's test.

### 2026-07-01 — installing whisperx broke the already-working SER metric
- **Symptom:** Asked to swap `entity_intelligibility`'s ASR backend to WhisperX for
  word-level timestamps. `uv pip install whisperx` resolved cleanly, but afterward
  `tests/test_ser_emotion.py::test_real_fixture_runs_end_to_end_with_real_model` failed:
  `RuntimeError: Could not load libtorchcodec` (DLL not found).
- **Root cause:** WhisperX's dependency tree force-downgrades `transformers` (5.12.1 →
  4.57.6), `torch` (2.12.1 → 2.8.0), and `torchaudio` in the same shared venv `ser_emotion.py`
  depends on — verified via `uv pip install whisperx --dry-run` first, which should have been
  enough warning, but the actual breakage was confirmed by running the test.
- **Fix:** reverted (`uv sync --extra acoustic --extra judge --extra stats --extra dev`
  restores the pinned versions from `uv.lock`/`pyproject.toml`, since `uv pip install` never
  touched either file). Verified all 8 SER tests passed again before proceeding. Used
  faster-whisper's own `word_timestamps=True` instead (same model, zero new deps) —
  delivers the same word-level-timestamp capability without touching the shared environment.
- **Prevention:** before adding any new heavy dependency, check `uv pip install <pkg>
  --dry-run` for downgrades of packages OTHER metrics already depend on, then actually run
  those other metrics' tests after a real install — a clean dependency resolution does not
  guarantee the newly-pinned versions are still compatible with existing code.

### 2026-07-01 — speechmos package doesn't ship the UTMOS entry point it advertises
- **Symptom:** Asked to add `naturalness_mos.py` using UTMOS via `from speechmos import
  utmos22_strong`. That import raises `ImportError: cannot import name 'utmos22_strong'`.
- **Root cause:** the published `speechmos` PyPI package (0.0.1.1) only ships
  `dnsmos`/`aecmos`/`plcmos` submodules; UTMOS was never actually released in it.
- **Fix:** used DNSMOS (`speechmos.dnsmos.run(...)`, `p808_mos` field) instead — already
  named as an acceptable fallback, needs no extra download (ONNX models bundled), verified
  working on real fixture audio. `details["mos_engine"]` records which engine actually ran.
- **Prevention:** verify an unfamiliar package's advertised API actually exists (`dir()` the
  installed module, don't trust an assumed import path) before writing code against it —
  same lesson as the silero-vad/google-genai API checks earlier in this build.

### 2026-07-01 — judge free-text notes with embedded newlines corrupted Markdown tables
- **Symptom:** After adding the per-call metric breakdown table to `report/markdown_report.py`,
  a real end-to-end run showed `instruction_adherence_judge`/`emotional_appropriateness` table
  rows visually broken across multiple lines in `report_1.md` — the judge's own multi-line,
  numbered-list-formatted response text was embedded raw inside a table cell.
- **Root cause:** `_one_line_reason()` truncated `details["notes"]` to 160 characters
  (`text[:160]`) but never stripped embedded `\n` characters first, so a multi-paragraph judge
  response broke the single-line table-row assumption Markdown tables require.
- **Fix:** added `_single_line()` (collapses all whitespace/newlines to single spaces via
  `re.sub(r"\s+", " ", text)`, then truncates) and applied it to every free-text field placed
  in a table cell (judge notes, ERROR exception messages). Caught by re-running the real CLI
  end-to-end and reading the actual generated `report_1.md`, not just the unit tests (the unit
  tests used single-line fake judge notes, so they didn't exercise this path — a reminder that
  real-artifact review remains necessary even with full test coverage).
- **Prevention:** any free-text field from an LLM judge that lands inside a Markdown table
  cell must go through `_single_line()` (or equivalent) before insertion — never trust
  external text to already be single-line.
