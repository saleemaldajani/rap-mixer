from __future__ import annotations

import io
import os
import time

import httpx
import numpy as np
import soundfile as sf


class MagentaRealtimeProvider:
    """Optional adapter for a separately running Magenta RealTime service."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("MAGENTA_REALTIME_URL", "")).rstrip("/")

    def capabilities(self) -> dict:
        midi_outputs = []
        audio_inputs = []
        try:
            import mido
            import sounddevice as sd

            midi_outputs = mido.get_output_names()
            audio_inputs = [
                device["name"] for device in sd.query_devices()
                if device["max_input_channels"] > 0
            ]
        except (ImportError, OSError):
            pass
        local_ready = (
            any("MRT2 - Collider Input" in name for name in midi_outputs)
            and "ZoomAudioDevice" in audio_inputs
        )
        return {
            "available": bool(self.base_url) or local_ready,
            "configured_url": bool(self.base_url),
            "local_mrt2_ready": local_ready,
            "midi_outputs": [x for x in midi_outputs if "MRT2" in x],
            "capture_device": "ZoomAudioDevice" if "ZoomAudioDevice" in audio_inputs else None,
            "fallback": "uploaded instrumental or metronome",
            "note": (
                "Local MRT2 capture requires Collider/Jam Audio Output = ZoomAudioDevice at 48 kHz."
            ),
        }

    def continue_audio(self, bpm: float, bars: int, engine: str = "Collider"):
        if self.base_url:
            response = httpx.post(
                f"{self.base_url}/continue", json={"bpm": bpm, "bars": bars}, timeout=45
            )
            response.raise_for_status()
            audio, sample_rate = sf.read(io.BytesIO(response.content), dtype="float32")
            return sample_rate, audio
        return self._capture_local_mrt2(bpm, bars, engine)

    @staticmethod
    def _capture_local_mrt2(bpm: float, bars: int, engine: str):
        try:
            import mido
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("Install the MRT2 bridge audio and MIDI dependencies.") from exc
        midi_name = f"MRT2 - {'Jam' if engine.lower() == 'jam' else 'Collider'} Input"
        matching = [name for name in mido.get_output_names() if midi_name in name]
        if not matching:
            raise RuntimeError(f"{midi_name} is unavailable; open the MRT2 app first.")
        sample_rate = 48000
        beat_seconds = 60 / max(30, min(300, bpm))
        duration = bars * 4 * beat_seconds + 1.0
        recording = sd.rec(
            int(duration * sample_rate), samplerate=sample_rate, channels=2,
            dtype="float32", device="ZoomAudioDevice",
        )
        chords = [(48, 55, 60), (46, 53, 58), (43, 50, 55), (45, 52, 57)]
        with mido.open_output(matching[0]) as port:
            for beat in range(bars * 4):
                chord = chords[(beat // 4) % len(chords)]
                notes = chord if beat % 4 == 0 else (chord[-1] + 12,)
                for note in notes:
                    port.send(mido.Message("note_on", note=note, velocity=82))
                time.sleep(beat_seconds * 0.82)
                for note in notes:
                    port.send(mido.Message("note_off", note=note, velocity=0))
                time.sleep(beat_seconds * 0.18)
        sd.wait()
        audio = np.asarray(recording, dtype=np.float32)
        if float(np.sqrt(np.mean(audio**2))) < 1e-5:
            raise RuntimeError(
                "MRT2 produced no loopback audio. In Collider/Jam set Audio Output to "
                "ZoomAudioDevice (48 kHz), then retry."
            )
        return sample_rate, audio

    def stop(self) -> None:
        if self.base_url:
            try:
                httpx.post(f"{self.base_url}/stop", timeout=5)
            except httpx.HTTPError:
                pass
