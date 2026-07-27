---
title: The Rap Mixer
emoji: 🎤
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 6.20.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
suggested_hardware: cpu-basic
short_description: Context-aware rap analysis, revision, freestyle, battles
---

# The Rap Mixer

The Rap Mixer is a Gradio application for inspecting how artist-controlled performance choices (A) interact with a selected cultural context (B) to produce several output scores—not one universal rating. It supports pre-recorded analysis, a bounded live four-bar stream, transparent calculation traces, and constrained revision recommendations.

> Scores explain a configuration in a selected context. They are not objective artistic truth.

## Quick start

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`. The default URL is `http://127.0.0.1:7860`.

The base install is CPU-only and requires no API key. For local transcription, install `pip install -e '.[whisper]'`; faster-whisper downloads `small.en` on first use. On NVIDIA systems, install the PyTorch wheel matching the CUDA version first, then the whisper extra. For Ollama, install Ollama, run `ollama pull llama3.2`, and set `OLLAMA_BASE_URL`.

## What works

- Pre-recorded upload or microphone recording with manual/corrected lyrics, BPM, context blending, complete bar table, plots, scoring, traces, inverse suggestions, and JSON export.
- Live microphone streaming via Gradio's `.stream()` event every second. Audio uses a 120-second ring buffer; the complete estimated bar history remains visible and the newest four bars are active.
- Audio checks for empty, silent, clipped, and noisy recordings. Transcript edits immediately recalculate lexical and semantic heuristics without retranscription.
- Six A layers: person, words, voice, music, interaction, culture. Manual values are explicit controls; audio/text adjustments are calculated locally.
- Twelve independent outputs, uncertainty intervals, contribution traces, explicit interactions, context weights, and versioned YAML coefficients.
- The inverse search holds B fixed, respects protected A dimensions, limits change magnitude, and rejects candidates that damage protected outputs beyond tolerance.

## Scoring and provenance

For each output, the engine evaluates `100 × sigmoid(bias + Σ(context × coefficient × normalized A) + Σ interaction coefficient × interaction)`. Final arithmetic is always deterministic and local. `config/scoring.yaml` contains every coefficient; `config/contexts.yaml` contains context profiles. Both are labeled prototype assumptions and should be replaced or calibrated through community-led empirical work.

Direct audio features include duration, energy, dynamic range, clipping, noise contrast, and onset proxy. Transcript-derived features include syllables, lexical novelty, rhyme-ending proxies, repetition, and clarity proxies. Semantic roles and cultural values are local heuristic inferences, never direct measurements. Manually moved A controls are user-supplied. Missing audio/transcript evidence increases uncertainty.

## Credentials, consent, and privacy

This public edition contains no shared API keys and deliberately ignores provider-key environment variables. Choose the free local path, or choose **Use my own API key** and hold OpenAI, Gemini, and/or Anthropic credentials for the current session. User keys are wrapped in `SecretStr` and held in a process-memory vault keyed by Gradio session hash, never in serialized `gr.State`, exports, URLs, logs, or exception text. Clear Credentials and Clear Session erase the vault entry.

Cloud operations require session-specific consent. A missing or failed user key never falls back to a server credential. Provider errors are reduced to one of: Connected, Authentication failed, Provider unavailable, Rate limited, Model unavailable. Raw lyrics, audio, prompts, responses, and keys are not logged. Temporary JSON exports are created only when the user runs analysis and chooses to download them; production deployments should add an expiry worker.

Set frontier model identifiers in `.env`; none is treated as permanently “latest.” Optional SDKs install with `pip install -e '.[cloud]'`. The deterministic path remains available if any adapter fails. Remote semantic adapters are safe extension points in this prototype; see Known limitations.

## Configuration

- `config/scoring.yaml`: biases, A coefficients, interaction coefficients, version.
- `config/contexts.yaml`: blendable context profiles.
- `config/models.yaml`: configurable economical model defaults.
- `config/limits.yaml`: timeouts, retry, rate, and stream bounds.

Run checks:

```bash
pip install -e '.[dev]'
pytest -q
ruff check .
```

## Deploy on Railway

The repository includes a production `Dockerfile` and `railway.json`. In Railway:

1. Choose **New Project → Deploy from GitHub repo** and select this repository.
2. Keep the generated service at one replica because session credentials live only in that process's memory.
3. Do not add OpenAI, Gemini, or Anthropic keys as Railway variables. Visitors provide their own keys in the app, or use the local/free modes.
4. Generate a Railway domain after the deployment becomes healthy. HTTPS is required for browser microphone access.

Railway supplies `PORT`; `app.py` binds that port on `0.0.0.0` inside the container. The image installs `ffmpeg` and `libsndfile1`. Its filesystem is ephemeral, so downloaded exports should be treated as temporary. The base image does not install the large optional Whisper models; manual lyrics and deterministic local analysis work without API keys. To offer local Whisper in a larger deployment, add the `whisper` optional dependencies and budget for model download time, RAM, and persistent caching.

For local development, the app remains at `127.0.0.1:7860` unless the Gradio environment variables are changed.

## Deploy on Hugging Face Spaces

The YAML front matter at the top of this README configures a Gradio Space (CPU Basic). The Space
installs `requirements.txt` plus the system packages in `packages.txt` (`ffmpeg`, `libsndfile1`)
and runs `app.py`, which exposes the Blocks object as `demo`. As on Railway, do not add provider
API keys as Space secrets: this edition ignores server-side provider keys, and visitors either use
the free local path or hold their own key for the session. Native MRT2/MIDI probing stays disabled
(`ENABLE_LOCAL_MRT2_PROBE` unset), and heavyweight local Whisper models are not installed on the
free CPU tier—manual lyrics and deterministic local analysis remain the no-key path.

## Known limitations and roadmap

The included semantic cloud adapter is deliberately non-operational until provider-specific structured-output calls and mock/real integration tests are configured; deterministic analysis survives that failure. Local Whisper classes are available but the UI currently treats corrected/manual lyrics as authoritative and does not trigger heavyweight model downloads automatically. Live bar detection uses BPM timing and phrase allocation rather than full beat/downbeat tracking; tap tempo, boundary split/merge/lock, word timestamps, pitch/spectrogram/rhyme graph, CSV/report export, and screenshot automation remain roadmap work. Browser microphone verification requires a human permission gesture, so automated validation simulates streaming chunks.

This is a transparent research prototype, not a scientifically validated assessment instrument.

## Live Battle Rapper AI

The integrated top-level **Live Battle Rapper AI** tab adds seven panels: Battle Response,
Instrumental Freestyle, Creative Loop, Mixer Dashboard, Battle Analysis, Session History, and
Model and Voice Setup. The stable production path is turn-based:

```text
Human turn → corrected transcript → completed bars → argument graph
           → validated strategy → constrained bar drafts → cadence checks
           → synthetic voice or text fallback → transparent scoring → history
```

The local battle generator directly cites human bar IDs, plans a strategy before writing,
checks disallowed material, respects the requested response length, and exposes intended beat
entries, rhyme endings, syllable targets, delivery notes, warnings, score uncertainty, and a
latency waterfall. Immediate Stop invalidates the active generation and returns the state machine
to `IDLE`.

Instrumental Freestyle accepts uploaded or recorded audio without Magenta. It locally checks the
audio, accepts BPM and creative constraints, creates editable bar/cadence plans, and degrades to
metronome/text development mode. Magenta RealTime is capability-detected and optional; it is not
treated as an argument, lyric, cadence, or battle model.

To enable Magenta in **Live Battle Rapper AI → Instrumental Freestyle**, choose `Magenta
RealTime`, enable its checkbox, and configure `MAGENTA_REALTIME_URL` to a separately running
adapter service. The service contract is `POST /continue` with `{bpm, bars}` returning WAV audio,
and `POST /stop`. If it is missing, unreachable, or returns invalid audio, the interface explicitly
reports the failure and produces a local metronome backing track. The Magenta repository and its
hardware/runtime dependencies are intentionally not bundled into the base Rap Mixer install.
Direct MRT2 hardware probing is disabled in this public/headless build. Local developers may opt
in with `ENABLE_LOCAL_MRT2_PROBE=true`; Railway deployments should leave it disabled and use the
adapter URL or uploaded instrumentals.

On macOS, **Performed response** uses the system `say` command as a clearly labeled synthetic
development voice. Battle and freestyle performances render each line into one BPM-sized bar,
varying speaking rate from syllable density and varying pitch, entry timing, downbeat emphasis,
and breath space from the selected Voice, intensity, aggression, or looseness controls. This is
still ordinary expressive TTS, not human-level rap flow or precision. Other systems use
the text-only fallback until a local or consent-gated cloud voice adapter is configured. No voice
is named after or intended to imitate a living performer.

### Battle architecture

```text
Existing credential/consent panel ───────┐
Existing Bank A + context B ─────────────┼─→ candidate scoring + inverse model
                                         │
Human audio/text → bars → argument graph ├─→ strategy → generated bars → cadence plan
Uploaded beat ─→ instrumental adapter ───┘                         │
                                                                   ├─→ text fallback
                                                                   └─→ synthetic voice
All turns → in-memory history → explicit JSON export
```

Safety boundaries focus counters on claims, contradictions, technique, framing, and consensually
provided facts. Credible threats, private-data attacks, protected-characteristic attacks,
self-harm encouragement, and configured disallowed topics are rejected or redirected.

FastRTC full duplex, non-OpenAI cloud battle adapters, bar-level audio splicing, persisted
Creative Loop presets, automatic live Whisper inside the battle callback, richer audio mixing,
and Magenta RealTime are documented extension points, not claimed as working integrations.

### OpenAI transcription and battle generation

OpenAI is now a working optional path. In the **API keys, models & data sharing** panel at the
top (shared by every tab), paste the key into the OpenAI box — it is held automatically in the
server-side session vault and the credential source switches to `Use my own API key` — then
enable session cloud consent and click **Test OpenAI**. The key remains only in the server-side
session vault. It is never placed in `gr.State` or returned to the browser.

The transcription and battle-generation model fields accept custom compatible model IDs. Defaults
come from `DEFAULT_OPENAI_TRANSCRIPTION_MODEL` and `DEFAULT_OPENAI_MODEL`. Pre-recorded mode has an
explicit Transcribe button. Live four-bar mode sends a bounded overlapping window approximately
every three seconds and deduplicates overlap. Battle Response can transcribe an audio-only turn and
use the Responses API to generate bars from the validated local strategy; if the provider fails,
the deterministic responsive result remains visible with a fallback notice.

Every performance workflow now exposes an explicit lyrics source. Choose **Use supplied lyrics**
to paste/edit text without an ASR call, or **Transcribe recording/microphone** to send recorded
audio to the selected transcription provider after consent. The resulting transcript always
appears in the editable transcript box and is the exact text passed to bar segmentation and
scoring. Instrumental Freestyle does not transcribe its backing track as lyrics; it generates new
lyrics from the topic, message, and creative constraints.

### Public YouTube input with Gemini

Pre-recorded analysis, Battle Response, and Instrumental Freestyle accept public YouTube video
URLs. Store your Gemini key in the server-side session vault, choose `Use my own API key`,
enable cloud consent, and select a compatible Gemini video model.
The default comes from `DEFAULT_GEMINI_MODEL`.

Gemini receives the public YouTube URL directly and returns a validated transcript/bar list,
summary, confidence, and estimated BPM/time signature. Review model-produced transcripts before
scoring. This route does not download the original waveform, so pitch, RMS, clipping, masking, and
other direct audio measurements remain unavailable unless audio is also uploaded or recorded.
Only public videos are supported; private and unlisted videos are rejected by the provider. The
YouTube URL feature is currently a Gemini preview capability, so availability, pricing, and limits
may change.

### Performance Tuner integration

**Performance Tuner** is a top-level view inside the existing `app.py` Gradio application. The
single `live_audio.stream(...)` callback continues to own microphone ingestion, the bounded audio
ring buffer, transcription, bar segmentation, semantic estimates, forward scoring, interactions,
and inverse recommendations. After those calculations finish, it writes one normalized snapshot
to `LiveSession.latest_analysis`; both Live Feedback and the tuner render that same snapshot.
Opening the tuner or changing display controls only renders cached state and makes no provider or
inference request. Context controls in the two live views share the same value.

The Master Lock aggregates only the selected Bank A dimensions against the chosen creative target.
Six Bank A meters, scored outputs, interaction readings, the latest-four-bar timeline, and one
confidence-aware coaching cue are available. Exponential smoothing, outlier bounds, dwell-time
locking, hysteresis, stale-state detection, and different semantic/audio response rates reduce
jitter. Target profiles in `config/tuner_targets.yaml` are configurable creative guides, never
claims of universal artistic quality.

Accessibility controls include Reduced Motion, a static display, High Contrast, visible numeric
values and direction text, motion-speed adjustment, animation pause, keyboard-native Gradio
controls, and screen-reader labels. The CSS also honors the operating system's
`prefers-reduced-motion` setting and does not use full-screen flashing. **Global Stop** cancels the
one live stream event and pauses tuner animation. The current limitation is that a saved-window
reference is session-local; it is not persisted across app restarts.
