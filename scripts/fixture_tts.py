"""Windows SAPI TTS wrapper used only to author fixture audio. This is a
dev-time generator tool, not part of the runtime eval_system package or its
dependency surface -- it never runs during scoring."""
from __future__ import annotations

import subprocess
from pathlib import Path

VOICES = {
    "agent": "Microsoft David Desktop",
    "caller": "Microsoft Zira Desktop",
}


def synthesize(text: str, out_path: Path, voice: str) -> None:
    escaped = text.replace("'", "''")
    ps_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SelectVoice('{voice}'); "
        "$s.Rate = 0; "
        f"$s.SetOutputToWaveFile('{out_path}'); "
        f"$s.Speak('{escaped}'); "
        "$s.Dispose()"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_script],
        check=True,
        capture_output=True,
    )
