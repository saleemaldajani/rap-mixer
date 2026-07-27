from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from rap_mixer.performance.cadence import cadence_plan


def speech_safe_text(text: str) -> str:
    """Remove internal analysis IDs from performer-facing speech without changing stored bars."""
    cleaned = re.sub(r"\bH\s*\d+\b\s*[:—-]?", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:human\s+)?bar\s+#?\d+\b\s*[:—-]?", "that line", cleaned,
                     flags=re.IGNORECASE)
    # Commas create exaggerated pauses in macOS speech; bar/beat timing supplies the phrasing.
    cleaned = re.sub(r"[,;]+", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def beat_phrases(text: str, beats: int = 4) -> list[str]:
    """Split a lyric line into balanced, ordered phrases for beat-level entrances."""
    words = text.split()
    if not words:
        return [""] * beats
    groups = []
    for index in range(beats):
        start = round(index * len(words) / beats)
        stop = round((index + 1) * len(words) / beats)
        groups.append(" ".join(words[start:stop]))
    return groups


class TextOnlyVoiceProvider:
    def synthesize_verse(self, text: str) -> None:
        return None

    def capabilities(self) -> dict:
        return {"audio": False, "label": "Text-only fallback"}


class LocalSyntheticVoiceProvider:
    def capabilities(self) -> dict:
        return {"audio": bool(Path("/usr/bin/say").exists()), "label": "AI-generated synthetic voice"}

    def _say_line(self, text: str, rate: int, pitch: int) -> tuple[np.ndarray, int]:
        source = tempfile.NamedTemporaryFile(
            prefix="rap-ai-source-", suffix=".aiff", delete=False
        ).name
        wav = tempfile.NamedTemporaryFile(prefix="rap-ai-line-", suffix=".wav", delete=False).name
        try:
            # Keep the contour subtle; strong global emphasis makes system TTS sound robotic.
            marked = f"[[pbas {pitch:+d}]] {text} [[slnc 35]]"
            subprocess.run(
                ["/usr/bin/say", "-v", "Samantha", "-r", str(rate), "-o", source, marked],
                check=True, timeout=30, capture_output=True,
            )
            subprocess.run(
                ["/usr/bin/afconvert", "-f", "WAVE", "-d", "LEI16", source, wav],
                check=True, timeout=30, capture_output=True,
            )
            audio, sample_rate = sf.read(wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sample_rate
        finally:
            Path(source).unlink(missing_ok=True)
            Path(wav).unlink(missing_ok=True)

    def synthesize_performance(
        self, text: str, bpm: float = 90, energy: float = 60, aggression: float = 50,
    ) -> str | None:
        if not self.capabilities()["audio"]:
            return None
        clean = speech_safe_text(text)
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if not lines:
            return None
        target = tempfile.NamedTemporaryFile(prefix="rap-ai-", suffix=".wav", delete=False).name
        sample_rate = 22050
        bar_seconds = 4 * 60 / max(30, float(bpm))
        rendered: list[tuple[int, np.ndarray]] = []
        next_available = 0
        for index, line in enumerate(lines):
            phrases = beat_phrases(line)
            for beat_index, phrase in enumerate(phrases):
                if not phrase:
                    continue
                plan = cadence_plan(phrase, bpm, beats=1)
                density = float(plan["syllables_per_second"])
                rate = int(np.clip(125 + density * 6 + energy * .10, 145, 210))
                contour = 1 if beat_index in {0, 3} else 0
                pitch = int(np.clip((energy - 50) / 30 + contour - index % 2, -2, 3))
                audio, sample_rate = self._say_line(phrase, rate, pitch)
                beat_samples = max(1, round(sample_rate * 60 / max(30, float(bpm))))
                bar_samples = beat_samples * 4
                grid_onset = index * bar_samples + beat_index * beat_samples
                # Never stack synthetic phrases. If a phrase runs long, enter on the next
                # available beat instead of talking over the previous words.
                onset = max(grid_onset, next_available)
                rendered.append((onset, audio))
                next_available = onset + len(audio) + round(.010 * sample_rate)
        minimum_length = max(1, round(len(lines) * bar_seconds * sample_rate))
        total_length = max(minimum_length, max(onset + len(audio) for onset, audio in rendered))
        verse = np.zeros(total_length, dtype="float32")
        for onset, audio in rendered:
            stop = onset + len(audio)
            verse[onset:stop] += audio
        # Gentle saturation controls rare overlaps without flattening natural dynamics.
        verse = np.tanh(verse * 1.05).astype("float32")
        peak = float(np.max(np.abs(verse))) if verse.size else 0
        if peak > .98:
            verse *= .98 / peak
        sf.write(target, verse, sample_rate, subtype="PCM_16")
        return target

    def synthesize_verse(self, text: str) -> str | None:
        """Backward-compatible neutral performance."""
        return self.synthesize_performance(text)
