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
