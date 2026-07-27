from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from openai import OpenAI
from pydantic import SecretStr

from rap_mixer.providers.credentials import Settings, resolve_key
from rap_mixer.security.consent import require_cloud_consent


def credential_source(ui_value: str) -> str:
    return {
        "Local/open-source—no API key": "local",
        "Use my own API key": "user_supplied",
    }.get(ui_value, "local")


def openai_client(source_ui: str, session_key: SecretStr | None, consent: bool) -> OpenAI:
    require_cloud_consent("openai", consent)
    source = credential_source(source_ui)
    if source == "local":
        raise ValueError("Select an OpenAI credential source for this cloud operation.")
    key = resolve_key("openai", source, session_key, Settings.from_env())
    return OpenAI(api_key=key.get_secret_value(), timeout=45, max_retries=1)


def transcribe_numpy(client: OpenAI, audio: tuple[int, np.ndarray], model: str) -> str:
    sr, data = audio
    samples = np.asarray(data)
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = float(max(abs(info.min), info.max))
        y = samples.astype(np.float32) / scale
    else:
        y = samples.astype(np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    path = tempfile.NamedTemporaryFile(prefix="rap-asr-", suffix=".wav", delete=False).name
    try:
        sf.write(path, y, sr, subtype="PCM_16")
        with open(path, "rb") as handle:
            result = client.audio.transcriptions.create(model=model, file=handle)
        text = (result.text or "").strip()
        if not text:
            raise ValueError(
                "No speech was detected in the recording. Try recording closer to the "
                "microphone, or enter the lyrics manually."
            )
        return text
    finally:
        Path(path).unlink(missing_ok=True)


def generate_structured_battle(client: OpenAI, model: str, transcript: str, strategy: dict,
                               bar_count: int, disallowed: str,
                               previous_responses: list[str] | None = None) -> list[dict]:
    prior = "\n---\n".join(previous_responses or []) or "none"
    prompt = f"""You are writing a consensual battle-rap response. Attack claims, framing, and craft—not protected identity. No threats or private data. Produce exactly {bar_count} short responsive punchlines. No filler, narration, generic setup bars, or repeated sentence frames. Each bar must cite one actual human bar ID and remain performable. Disallowed topics: {disallowed or 'none'}.

Human transcript with IDs:
{transcript}

Validated strategy:
{json.dumps(strategy)}

Previous responses that must not be repeated or closely paraphrased:
{prior}

The text performed aloud must never contain IDs such as H1/H2 or phrases such as "bar 1"; keep
those references only in addressed_human_bar_ids. Return JSON only as
{{"bars":[{{"text":"...","addressed_human_bar_ids":["H1"],"function":"answer","delivery_note":"..."}}]}}."""
    response = client.responses.create(model=model, input=prompt)
    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    payload = json.loads(text)
    bars = payload.get("bars", [])
    if len(bars) != bar_count:
        raise ValueError("Model returned the wrong bar count.")
    return bars


def test_openai(client: OpenAI) -> str:
    try:
        client.models.list()
        return "Connected"
    except Exception as exc:
        name = type(exc).__name__.lower()
        if "auth" in name:
            return "Authentication failed"
        if "rate" in name:
            return "Rate limited"
        return "Provider unavailable"
