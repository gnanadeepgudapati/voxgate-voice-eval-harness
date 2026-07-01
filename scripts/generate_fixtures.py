"""Generates the synthetic fixture set under fixtures/. Run once with:

    uv run python scripts/generate_fixtures.py

Output (call.wav + transcript.jsonl + tool_log.jsonl + events.jsonl +
scenario_db.json + expected.json per scenario) is committed to the repo, same
as any other test fixture.

Audio is REAL synthesized speech (Windows SAPI TTS via System.Speech), not
fabricated tones -- every timestamp below is derived from the ACTUAL sample
length of the clip it describes. Alignment is never invented independently of
the audio it describes (see CLAUDE.md scope note on fixtures).
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixture_tts import VOICES, synthesize  # noqa: E402

SR = 22050
MAX_SECONDS = 90
FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class CallBuilder:
    call_id: str
    sr: int = SR
    caller_buf: np.ndarray = field(default_factory=lambda: np.zeros(int(MAX_SECONDS * SR), dtype=np.float32))
    agent_buf: np.ndarray = field(default_factory=lambda: np.zeros(int(MAX_SECONDS * SR), dtype=np.float32))
    cursor: float = 0.0
    end_of_call: float = 0.0
    transcript: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    tool_events: list[dict] = field(default_factory=list)

    def _buf(self, speaker: str) -> np.ndarray:
        return self.caller_buf if speaker == "caller" else self.agent_buf

    def _render(self, speaker: str, text: str) -> np.ndarray:
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "clip.wav"
            synthesize(text, wav_path, VOICES[speaker])
            clip, clip_sr = sf.read(wav_path, dtype="float32")
        if clip.ndim > 1:
            clip = clip.mean(axis=1)
        assert clip_sr == self.sr, f"expected TTS output at {self.sr} Hz, got {clip_sr}"
        return clip

    def say(self, speaker: str, text: str, gap_before: float = 0.3,
             asr_confidence: float | None = None, cut_after: float | None = None,
             log_transcript: bool = True) -> tuple[float, float]:
        """Render `text` as `speaker` and place it `gap_before` seconds after
        the current cursor. If `cut_after` is given, the clip is truncated
        that many seconds into its own audio (models the agent yielding /
        being interrupted mid-utterance) and the cursor advances only to the
        truncation point, since that's when the speaker actually stopped."""
        clip = self._render(speaker, text)
        t_start = self.cursor + gap_before
        start_sample = round(t_start * self.sr)
        full_end_sample = start_sample + len(clip)

        if cut_after is not None:
            cut_sample = start_sample + round(cut_after * self.sr)
            cut_sample = min(cut_sample, full_end_sample)
            placed = clip[: cut_sample - start_sample]
            end_sample = cut_sample
        else:
            placed = clip
            end_sample = full_end_sample

        buf = self._buf(speaker)
        buf[start_sample:end_sample] = placed
        t_end = end_sample / self.sr

        self.cursor = t_end
        self.end_of_call = max(self.end_of_call, t_end)
        if log_transcript:
            self.transcript.append({
                "speaker": speaker, "t_start": t_start, "t_end": t_end,
                "text": text, "asr_confidence": asr_confidence,
            })
        return t_start, t_end

    def inject_noise(self, speaker: str, t_start: float, duration: float,
                      amplitude: float = 0.08, seed: int = 0) -> tuple[float, float]:
        """Places a short noise burst at an ABSOLUTE time (used for coughs /
        background noise, independent of the sequential cursor)."""
        rng = np.random.default_rng(seed)
        n = round(duration * self.sr)
        noise = (rng.standard_normal(n) * amplitude).astype(np.float32)
        # short fade in/out so it doesn't click
        fade = min(n // 6, 200)
        if fade > 0:
            noise[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            noise[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        start_sample = round(t_start * self.sr)
        end_sample = start_sample + n
        buf = self._buf(speaker)
        buf[start_sample:end_sample] += noise
        t_end = end_sample / self.sr
        self.end_of_call = max(self.end_of_call, t_end)
        return t_start, t_end

    def add_event(self, name: str, t: float, meta: dict | None = None) -> None:
        self.events.append({"name": name, "t": t, "meta": meta or {}})

    def add_tool_event(self, name: str, args: dict, result, t: float) -> None:
        self.tool_events.append({"name": name, "args": args, "result": result, "t": t})

    def write(self, scenario_db: dict, expected: dict) -> None:
        out_dir = FIXTURES_DIR / self.call_id
        out_dir.mkdir(parents=True, exist_ok=True)

        tail_silence = 0.5
        total_samples = round((self.end_of_call + tail_silence) * self.sr)
        stereo = np.stack([self.caller_buf[:total_samples], self.agent_buf[:total_samples]], axis=1)
        sf.write(out_dir / "call.wav", stereo, self.sr, subtype="PCM_16")

        (out_dir / "transcript.jsonl").write_text(
            "\n".join(json.dumps(t) for t in self.transcript) + "\n", encoding="utf-8"
        )
        (out_dir / "tool_log.jsonl").write_text(
            "\n".join(json.dumps(t) for t in self.tool_events) + "\n" if self.tool_events else "",
            encoding="utf-8",
        )
        (out_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(e) for e in self.events) + "\n" if self.events else "",
            encoding="utf-8",
        )
        (out_dir / "scenario_db.json").write_text(json.dumps(scenario_db, indent=2), encoding="utf-8")
        (out_dir / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
        print(f"wrote fixture: {out_dir} ({self.end_of_call:.2f}s)")


def build_happy_path_book() -> None:
    b = CallBuilder(call_id="happy_path_book")

    b.say("agent", "Thank you for calling Example Clinic. How can I help you today?", gap_before=0.3)
    b.say("caller", "Hi, I'd like to book an appointment with Doctor Lee next Tuesday at ten AM.", gap_before=0.4)
    _, t_end3 = b.say("agent", "Sure, one moment while I check that availability.", gap_before=0.3)

    t_check = t_end3 + 0.15
    b.add_tool_event(
        "check_availability",
        {"provider": "Lee", "date": "2026-07-07", "time": "10:00"},
        {"available": True},
        t_check,
    )
    t_book = t_end3 + 0.35
    b.add_tool_event(
        "book_appointment",
        {"provider": "Lee", "date": "2026-07-07", "time": "10:00", "patient_id": "P100"},
        {"status": "booked", "confirmation": "48213"},
        t_book,
    )

    b.say(
        "agent",
        "I found an opening with Doctor Lee on Tuesday at ten AM. "
        "I've booked that appointment for you. Your confirmation number is four eight two one three.",
        gap_before=0.6,
    )
    b.say("caller", "Great, thank you so much.", gap_before=0.4)
    b.say("agent", "You're welcome. Goodbye.", gap_before=0.3)

    scenario_db = {
        "patients": {"P100": {"name": "Jane Doe", "appointments": []}},
        "providers": {"Lee": {"name": "Dr. Lee", "availability": [
            {"date": "2026-07-07", "time": "10:00", "available": True},
        ]}},
    }
    expected = {
        "tool_sequence": [
            {"name": "check_availability", "required_args": {"provider": "Lee"}},
            {"name": "book_appointment", "required_args": {"provider": "Lee"}},
        ],
        "invariants": [],
        "critical_entities": ["Lee", "Tuesday", "ten AM", "four eight two one three"],
        "success_criteria": {
            "final_tool": "book_appointment",
            "result_contains": {"confirmation": "48213", "status": "booked"},
        },
    }
    b.write(scenario_db, expected)


def build_reschedule_trap() -> None:
    """Encodes the TRAP being hit: old appointment released before the new
    one is secured, and the rebooking attempt fails -- leaving the caller
    with zero appointments for a real span of the call. This fixture exists
    for tool_call_ordering to catch, not to model correct behavior."""
    b = CallBuilder(call_id="reschedule_trap")

    b.say("agent", "Thank you for calling Example Clinic. How can I help you today?", gap_before=0.3)
    b.say(
        "caller",
        "I need to reschedule my appointment with Doctor Lee from Tuesday to Wednesday at two PM.",
        gap_before=0.4,
    )
    _, t_end3 = b.say("agent", "Sure, let me look into that for you.", gap_before=0.3)

    t_cancel = t_end3 + 0.2
    b.add_tool_event(
        "cancel_appointment",
        {"appointment_id": "A1", "confirmation": "48213"},
        {"status": "cancelled"},
        t_cancel,
    )
    t_rebook_fail = t_end3 + 0.5
    b.add_tool_event(
        "book_appointment",
        {"provider": "Lee", "date": "2026-07-08", "time": "14:00", "patient_id": "P100"},
        {"status": "error", "reason": "slot_unavailable"},
        t_rebook_fail,
    )
    b.add_event("zero_appointments_start", t_cancel)

    b.say(
        "agent",
        "I'm sorry, it looks like that time is no longer available. Let me find another option.",
        gap_before=0.5,
    )
    b.say("caller", "Wait, so do I even have an appointment right now?", gap_before=0.4)
    _, t_end6 = b.say(
        "agent",
        "Let me check. I apologize, there seems to have been an issue. "
        "Let me rebook that Tuesday slot for you right away.",
        gap_before=0.4,
    )

    t_rebook_ok = t_end6 + 0.3
    b.add_tool_event(
        "book_appointment",
        {"provider": "Lee", "date": "2026-07-07", "time": "10:00", "patient_id": "P100"},
        {"status": "booked", "confirmation": "48213b"},
        t_rebook_ok,
    )
    b.add_event("zero_appointments_end", t_rebook_ok)

    b.say("caller", "Okay, thank you.", gap_before=0.5)
    b.say("agent", "You're welcome, and sorry about the confusion.", gap_before=0.3)

    scenario_db = {
        "patients": {"P100": {"name": "Jane Doe", "appointments": ["A1"]}},
        "appointments": {"A1": {"provider": "Lee", "date": "2026-07-07", "time": "10:00", "confirmation": "48213"}},
        "providers": {"Lee": {"name": "Dr. Lee", "availability": [
            {"date": "2026-07-08", "time": "14:00", "available": False},
        ]}},
    }
    expected = {
        "tool_sequence": [
            {"name": "book_appointment", "required_args": {"provider": "Lee"}},
            {"name": "cancel_appointment", "required_args": {}},
        ],
        "invariants": ["never_zero_appointments"],
        "critical_entities": ["Lee", "Wednesday", "two PM"],
        "success_criteria": {"final_tool": None, "result_contains": {}},
    }
    b.write(scenario_db, expected)


def build_barge_in_basic() -> None:
    b = CallBuilder(call_id="barge_in_basic")

    # --- False yield: agent stops for a cough, which it should NOT do ---
    t_start1, _ = b.say(
        "agent",
        "Let me pull up your file and check what appointments you currently have on the schedule right now.",
        gap_before=0.3,
        cut_after=1.8,  # models the agent stopping 1.8s into this line
    )
    t_cough_start = t_start1 + 1.5
    t_cough_start, t_cough_end = b.inject_noise("caller", t_cough_start, duration=0.25, amplitude=0.08, seed=1)
    t_yield1 = t_start1 + 1.8
    b.add_event("cough", t_cough_start, {"channel": "caller"})
    b.add_event("agent_yield", t_yield1, {"trigger": "cough", "expected": "false_yield"})
    b.cursor = t_yield1

    b.say("caller", "Hello? Are you still there?", gap_before=1.0)
    _, t_end5 = b.say(
        "agent",
        "Sorry about that, yes, I'm still here. Let me continue checking your appointments now, one moment please.",
        gap_before=0.3,
        cut_after=2.45,  # genuine barge-in below, agent yields promptly
    )
    t_start5 = t_end5 - 2.45  # recover this segment's own t_start for the barge-in math
    t_barge_in = t_start5 + 2.0
    t_yield2 = t_start5 + 2.45
    b.add_event("barge_in_start", t_barge_in, {"channel": "caller"})
    b.add_event("agent_yield", t_yield2, {"trigger": "caller_speech", "expected": "prompt_yield"})
    b.cursor = t_yield2

    b.say("caller", "Actually, never mind, I found the confirmation email.", gap_before=0.05)
    b.say("agent", "Sounds good, glad you found it. Have a great day.", gap_before=0.3)

    scenario_db = {"patients": {"P100": {"name": "Jane Doe", "appointments": ["A1"]}}}
    expected = {
        "tool_sequence": [],
        "invariants": [],
        "critical_entities": ["confirmation email"],
        "success_criteria": {"final_tool": None, "result_contains": {}},
    }
    b.write(scenario_db, expected)


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    build_happy_path_book()
    build_reschedule_trap()
    build_barge_in_basic()
