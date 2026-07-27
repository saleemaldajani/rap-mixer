from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from google import genai
from pydantic import BaseModel, Field, SecretStr

from rap_mixer.providers.credentials import Settings, resolve_key
from rap_mixer.providers.openai_audio import credential_source
from rap_mixer.security.consent import require_cloud_consent


class YouTubeAnalysis(BaseModel):
    transcript: str
    bars: list[str]
    bpm_estimate: float | None = Field(default=None, ge=30, le=300)
    time_signature: str = "4/4"
    summary: str
    confidence: float = Field(default=0.5, ge=0, le=1)
    limitations: list[str] = []
    energy: float = Field(default=0.5, ge=0, le=1)
    drum_density: float = Field(default=0.5, ge=0, le=1)
    groove: str = "straight"
    bass_style: str = "sustained"
    key_estimate: str | None = None
    instrumentation: list[str] = []


def validate_youtube_url(url: str) -> str:
    parsed = urlparse(url.strip())
    host = parsed.hostname.lower() if parsed.hostname else ""
    if parsed.scheme != "https" or host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        raise ValueError("Enter a public HTTPS YouTube or youtu.be URL.")
    if host == "youtu.be" and not parsed.path.strip("/"):
        raise ValueError("The YouTube URL has no video ID.")
    if "youtube.com" in host and parsed.path != "/watch" and not parsed.path.startswith(("/shorts/", "/live/")):
        raise ValueError("The URL must point directly to a YouTube video, Short, or live recording.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif parsed.path == "/watch":
        video_id = parse_qs(parsed.query).get("v", [""])[0]
    else:
        video_id = parsed.path.rstrip("/").split("/")[-1]
    if not video_id:
        raise ValueError("The YouTube URL has no video ID.")
    # Gemini's video fetcher can treat playlist/radio parameters as an HTML URL. Always send a
    # canonical, video-only URL regardless of the user's copied YouTube link shape.
    return f"https://www.youtube.com/watch?v={video_id}"


def gemini_client(source_ui: str, session_key: SecretStr | None, consent: bool):
    require_cloud_consent("gemini", consent)
    source = credential_source(source_ui)
    if source == "local":
        raise ValueError("Select a Gemini credential source for YouTube analysis.")
    key = resolve_key("gemini", source, session_key, Settings.from_env())
    return genai.Client(api_key=key.get_secret_value())


def analyze_youtube(client, url: str, model: str, purpose: str = "rap performance") -> YouTubeAnalysis:
    url = validate_youtube_url(url)
    instrumental_only = "instrumental" in purpose.lower()
    if instrumental_only:
        task = """This is an instrumental-analysis request. Do not transcribe vocals, samples, titles,
or descriptions. Return transcript as an empty string and bars as an empty array. Estimate BPM,
time signature, and concisely describe groove, arrangement, energy, and structural changes."""
    else:
        task = """Transcribe the performed rap where intelligible and split completed lyrics into bars.
Do not invent inaudible lyrics; put uncertain words in brackets."""
    prompt = f"""Analyze the audio performance in this public YouTube video for the Rap Mixer. Purpose: {purpose}.
{task}
Return JSON only with: transcript, bars, bpm_estimate, time_signature, summary, confidence,
limitations, energy (0-1), drum_density (0-1), groove, bass_style, key_estimate, and
instrumentation. Do not invent inaudible lyrics. Put uncertain words in brackets."""
    response = client.interactions.create(
        model=model,
        input=[{"type": "text", "text": prompt}, {"type": "video", "uri": url}],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": YouTubeAnalysis.model_json_schema(),
        },
        generation_config={"max_output_tokens": 1800 if instrumental_only else 16000},
        timeout=300,
    )
    text = (response.output_text or "").strip()
    if not text:
        raise ValueError("Gemini returned an empty YouTube analysis.")
    return YouTubeAnalysis.model_validate_json(text)


def youtube_error_message(exc: Exception) -> str:
    message = str(exc).lower()
    name = type(exc).__name__.lower()
    if "401" in message or "403" in message or "auth" in name or "permission" in name:
        return "Gemini authentication or permission failed. Test the Gemini key above."
    if "429" in message or "quota" in message or "rate" in message:
        return "Gemini quota or rate limit reached. Wait briefly or use another Gemini key."
    if "timeout" in message or "timed out" in message:
        return "Gemini timed out while reading the video. Try a shorter public video."
    if any(term in message for term in ("private", "unlisted", "youtube", "video", "not found")):
        return "Gemini could not access that video. It must be public (not private or unlisted)."
    if "json" in name or "validation" in name or "empty youtube analysis" in message:
        return "Gemini returned an incomplete analysis. Try again or use a shorter video."
    return "Gemini could not analyze this video. Test the key, then try a shorter public video."
