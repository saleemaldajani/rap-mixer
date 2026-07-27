from __future__ import annotations

import json
import os
import re
import tempfile
import time
from html import escape
from pathlib import Path

import gradio as gr
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from rap_mixer.analysis.commercial import frames as commercial_frames
from rap_mixer.analysis.feature_estimation import estimate_features
from rap_mixer.audio.bars import enrich_bars_audio, latest_four_completed, segment_text
from rap_mixer.audio.buffering import deduplicate_text
from rap_mixer.providers.credentials import Settings, resolve_key
from rap_mixer.providers.gemini_youtube import analyze_youtube, gemini_client, youtube_error_message
from rap_mixer.providers.openai_audio import (
    credential_source as normalize_credential_source,
)
from rap_mixer.providers.openai_audio import (
    openai_client,
    test_openai,
    transcribe_numpy,
)
from rap_mixer.providers.semantic_analysis import analyze_semantics
from rap_mixer.scoring.explanations import trace_text
from rap_mixer.scoring.forward import DeterministicScoringEngine
from rap_mixer.scoring.inverse import RecommendationEngine
from rap_mixer.security.consent import require_cloud_consent
from rap_mixer.security.redaction import safe_provider_error
from rap_mixer.state import LiveSession
from rap_mixer.transcription.router import transcribe_selected
from rap_mixer.tuner.renderer import render_tuner
from rap_mixer.tuner.targets import PROFILES
from rap_mixer.tuner.view_model import live_analysis_to_tuner_view
from rap_mixer.ui.battle import build_battle_content
from rap_mixer.ui.credentials import clear_key, clear_keys, get_key, store_key
from rap_mixer.ui.human_battle import build_human_battle_tab
from rap_mixer.ui.plots import output_plot, radar, waveform

load_dotenv()

LAYERS = ["person", "words", "voice", "music", "interaction", "culture"]
SCORER = DeterministicScoringEngine()
INVERSE = RecommendationEngine(SCORER)
CONTEXTS = __import__("yaml").safe_load((Path(__file__).parents[2] / "config/contexts.yaml").read_text())["contexts"]
CSS = """
.gradio-container {max-width: 1500px !important}
.hero {padding: 1rem 0}
.disclaimer {padding: 0.5rem 0}
.pt-root {--pt-bg:var(--block-background-fill);--pt-fg:var(--body-text-color);color:var(--pt-fg)}
.pt-master,.pt-card,.pt-cue,.pt-timeline>div {background:var(--block-background-fill);border:1px solid var(--border-color-primary);border-radius:14px;padding:1rem}
.pt-master {position:relative;text-align:center;overflow:hidden;min-height:210px;border-width:3px}.pt-master h2{font-size:clamp(1.7rem,5vw,3.5rem);margin:.25rem}.pt-big{font-size:2rem;font-weight:800}.pt-master-direction{display:inline-block;padding:.5rem 1rem;border-radius:999px;background:#005f73;color:#fff;font-weight:900;letter-spacing:.05em}
.pt-phase {height:20px;background:repeating-linear-gradient(90deg,#0072b2 0 9px,transparent 9px 18px);animation:pt-drift var(--drift-speed) linear infinite}.state-locked .pt-master{border-color:#009e73;box-shadow:0 0 24px color-mix(in srgb,#009e73 32%,transparent)}.state-close .pt-master{border-color:#0072b2}.state-drifting .pt-master,.state-off-target .pt-master{border-color:#e69f00;box-shadow:0 0 24px color-mix(in srgb,#e69f00 28%,transparent)}.state-insufficient-evidence .pt-master,.state-building-window .pt-master{border-color:#777}
.pt-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:.75rem;margin:.75rem 0}.pt-grid.compact{grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.pt-card header{font-size:1.05rem;font-weight:750;margin-bottom:.55rem}.pt-meter{height:24px;background:var(--neutral-200);border-radius:12px;position:relative;overflow:hidden;margin:.5rem 0}.pt-zone{position:absolute;left:38%;width:24%;height:100%;background:#56b4e9;opacity:.52;border-left:2px solid #0072b2;border-right:2px solid #0072b2}.pt-marker{position:absolute;width:6px;height:100%;background:#111;z-index:2;border:1px solid #fff}.pt-stripes{position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent 0 12px,currentColor 13px 15px);opacity:.28;animation:pt-drift var(--metric-speed) linear infinite}.pt-guidance{margin:.4rem 0;padding:.35rem;border-radius:8px;text-align:center;font-weight:900;font-size:.82rem;letter-spacing:.025em}.pt-how{min-height:4.2em;margin:.45rem 0;padding:.5rem;border-left:4px solid #0072b2;background:color-mix(in srgb,#56b4e9 12%,transparent);line-height:1.35}.pt-master-action{max-width:720px;margin:.7rem auto!important;font-size:1.05rem}.dir-increase,.dir-decrease{border-color:#e69f00}.dir-increase .pt-guidance,.dir-decrease .pt-guidance{background:#e69f00;color:#171717}.dir-decrease .pt-stripes{animation-direction:reverse}.dir-hold{border-color:#009e73;box-shadow:inset 0 0 16px color-mix(in srgb,#009e73 14%,transparent)}.dir-hold .pt-guidance{background:#009e73;color:#fff}.dir-hold .pt-stripes{animation:none;opacity:.08}.dir-unknown .pt-guidance{background:#777;color:#fff}
.pt-cue{text-align:center;border-width:2px}.pt-cue h2{margin:.25rem;font-size:clamp(1.4rem,4vw,2.6rem)}.pt-timeline{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem}.pt-timeline>div{text-align:center}
.pt-transcript{margin:.75rem 0;padding:1rem 1.2rem;border:2px solid #2f80ed;border-radius:14px;background:var(--block-background-fill);font-size:1.08rem}.pt-transcript>b{display:block;color:#2f80ed;letter-spacing:.04em;margin-bottom:.5rem}.pt-transcript p{margin:.45rem 0;line-height:1.45}
.pt-root.contrast .pt-card,.pt-root.contrast .pt-master{border:3px solid currentColor}.pt-root.reduced *,.pt-root.contrast.reduced *{animation:none!important}
@keyframes pt-drift{from{background-position:0 0}to{background-position:72px 0}}
@media(max-width:640px){.pt-timeline{grid-template-columns:repeat(2,1fr)}}
@media(prefers-reduced-motion:reduce){.pt-root *{animation:none!important}}
.hvh-banner{padding:1rem;border:2px solid #c98a00;border-radius:18px;background:linear-gradient(135deg,#fff8e7,var(--block-background-fill));color:var(--body-text-color)}.hvh-banner h2{text-align:center;margin:.2rem 0 1rem}.hvh-cycle{display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem}.hvh-step{display:flex;align-items:center;gap:.5rem;padding:.75rem;border:1px solid var(--border-color-primary);border-radius:999px;opacity:.55}.hvh-step strong{display:grid;place-items:center;width:2rem;height:2rem;border-radius:50%;background:#777;color:#fff}.hvh-step.active{opacity:1;border:3px solid #d89a00;font-weight:800}.hvh-step.active strong{background:#d89a00}.hvh-step.done{opacity:.85;border-color:#168466}.hvh-step.done strong{background:#168466}.hvh-banner.results{display:flex;justify-content:center;gap:1rem;align-items:center;border-color:#168466}.hvh-banner.results b{font-size:1.5rem}
@media(max-width:700px){.hvh-cycle{grid-template-columns:repeat(2,1fr)}}
"""


def _context(names: list[str] | str, familiarity: float) -> dict[str, float]:
    names = [names] if isinstance(names, str) else (names or ["Cypher"])
    result = {k: 0.0 for k in next(iter(CONTEXTS.values()))}
    for name in names:
        for key, value in CONTEXTS[name].items():
            result[key] += value / len(names)
    result["familiarity"] = familiarity / 100
    return result


def _features(values, text, audio=None, mode="Auto-estimate, then edit"):
    return estimate_features(values, text, audio, mode)


def analyze(audio, lyrics, bpm, contexts, familiarity, mode, *values,
            semantic_estimates=None, semantic_notice=""):
    try:
        if not (lyrics or "").strip():
            raise ValueError(
                "No text is available to score. Enter supplied lyrics, or transcribe the "
                "recording and select Use transcribed audio."
            )
        duration = len(audio[1]) / audio[0] if audio else max(4, len((lyrics or "").split()) / 2.5)
        features, confidence, warning = _features(values, lyrics or "", audio, mode)
        if semantic_estimates:
            for name, estimate in semantic_estimates.items():
                features[name] = 0.5 * features[name] + 0.5 * estimate
        ctx = _context(contexts, familiarity)
        bundle = SCORER.score(features, ctx, confidence)
        bars = segment_text(lyrics, duration, bpm)
        if audio:
            enrich_bars_audio(bars, audio[0], audio[1])
        frame = pd.DataFrame([x.model_dump() for x in bars])
        details = "\n\n---\n\n".join(trace_text(x) for x in bundle.outputs[:6])
        status = f"Analysis complete · semantic: {semantic_notice or 'deterministic local'}. {warning}".strip()
        export = _export(bundle, features, bars)
        scored_text = _scored_bars_markdown(bars)
        return status, scored_text, output_plot(bundle), radar(features), waveform(audio, bars), frame, details, bundle.model_dump(), export
    except ValueError as exc:
        return str(exc), "", None, None, waveform(None), pd.DataFrame(), "", None, None


def analyze_with_lyrics_source(
    audio, supplied_lyrics, transcript, lyrics_source, bpm, contexts, familiarity, mode, stt_provider,
    semantic_provider, credential_source, cloud_consent, transcription_model, person, words, voice, music,
    interaction, culture,
    request: gr.Request,
):
    resolved = supplied_lyrics or ""
    transcription_notice = ""
    if lyrics_source == "Use transcribed audio":
        resolved = (transcript or "").strip()
        transcription_notice = " Scored the visible transcribed-audio text."
    semantic_estimates, semantic_notice = _selected_semantics(
        semantic_provider, resolved, credential_source, cloud_consent, request
    )
    result = list(analyze(
        audio, resolved, bpm, contexts, familiarity, mode,
        person, words, voice, music, interaction, culture,
        semantic_estimates=semantic_estimates, semantic_notice=semantic_notice,
    ))
    result[0] += transcription_notice
    return transcript or "", *result


def _selected_semantics(provider, text, credential_source, cloud_consent, request):
    if provider == "Deterministic local baseline":
        return None, "deterministic local"
    openai_semantic = None
    gemini_semantic = None
    try:
        if provider == "OpenAI":
            openai_semantic = openai_client(
                credential_source, get_key(request.session_hash, "openai"), cloud_consent
            )
        elif provider == "Google Gemini":
            gemini_semantic = gemini_client(
                credential_source, get_key(request.session_hash, "gemini"), cloud_consent
            )
        anthropic_key = None
        if provider == "Anthropic Claude":
            require_cloud_consent("anthropic", cloud_consent)
            resolved = resolve_key(
                "anthropic", normalize_credential_source(credential_source),
                get_key(request.session_hash, "anthropic"), Settings.from_env(),
            )
            anthropic_key = resolved.get_secret_value()
        return analyze_semantics(
            provider, text, openai=openai_semantic, gemini=gemini_semantic,
            anthropic_key=anthropic_key,
        )
    except Exception as exc:
        return None, f"{safe_provider_error(exc)}; deterministic fallback"
    finally:
        if openai_semantic is not None:
            openai_semantic.close()
        if gemini_semantic is not None:
            gemini_semantic.close()


def recommend_with_lyrics_source(
    bundle_data, targets, protected_features, protected_outputs, desired, max_change,
    supplied_lyrics, transcript, lyrics_source, *values,
):
    evidence = transcript if lyrics_source == "Use transcribed audio" else supplied_lyrics
    return recommend(
        bundle_data, targets, protected_features, protected_outputs, desired, max_change,
        evidence, *values,
    )


def recommend(bundle_data, targets, protected_features, protected_outputs, desired, max_change,
              evidence_text, *values):
    if not bundle_data:
        return "Run forward analysis first.", None, pd.DataFrame()
    features = dict(zip(LAYERS, values, strict=True))
    result = INVERSE.recommend(features, bundle_data["context"], targets or ["Intelligibility"], desired,
                               set(protected_features or []), set(protected_outputs or []), max_change,
                               evidence=evidence_text or "", workflow="prerecorded")
    rows = [x.model_dump() for x in result.recommendations]
    text = ("Feasible within configured approximation." if result.feasible else "Partial improvement found; requested target may exceed one-revision constraints.")
    return text, _before_after(result), pd.DataFrame(rows)


def _before_after(result):
    class B:
        pass
    b = B()
    b.outputs = [type("O", (), {"name": k, "score": v, "uncertainty": result.uncertainty}) for k, v in result.before.items()]
    return output_plot(b, result.after)


def _export(bundle, features, bars):
    payload = {"scores": bundle.model_dump(), "A": features, "bars": [x.model_dump() for x in bars]}
    path = tempfile.NamedTemporaryFile(prefix="rap-mixer-", suffix=".json", delete=False, mode="w")
    json.dump(payload, path, indent=2)
    path.close()
    return path.name


def _scored_bars_markdown(bars, active_only: bool = False) -> str:
    selected = [bar for bar in bars if bar.active] if active_only else bars
    if not selected:
        return "_No completed text bars yet._"
    rows = []
    for bar in selected:
        marker = "🟣 ACTIVE" if bar.active else "⚪ history"
        rows.append(
            f"**Bar {bar.number} · {bar.start:.1f}–{bar.end:.1f}s · {marker}**  \n"
            f"> {bar.transcript or '[no transcript text]'}  \n"
            f"{bar.syllables_per_second:.1f} syllables/s · {bar.words_per_second:.1f} words/s "
            f"· role: {bar.semantic_role}"
        )
    return "\n\n".join(rows)


def _latest_two_transcript(bars, transcript: str = "") -> str:
    latest = list(bars)[-2:]
    if not latest:
        text = (transcript or "").strip()
        if not text:
            return "<div class='pt-transcript'><b>LIVE TRANSCRIPTION</b><p>Waiting for speech…</p></div>"
        lines = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+", text) if part.strip()]
        visible = lines[-2:]
        return (
            "<div class='pt-transcript'><b>LATEST TRANSCRIPTION · PROVISIONAL</b>"
            + "".join(f"<p>{escape(item)}</p>" for item in visible)
            + "</div>"
        )
    return (
        "<div class='pt-transcript'><b>LATEST TWO TRANSCRIBED BARS</b>"
        + "".join(
            f"<p><strong>BAR {bar.number}</strong><br>{escape(bar.transcript or '[transcription pending]')}</p>"
            for bar in latest
        )
        + "</div>"
    )


def live_update(chunk, state, bpm, transcript, contexts, familiarity, targets,
                protected_features, protected_outputs, desired, max_change, lyrics_source,
                stt_provider, semantic_provider, credential_source, cloud_consent,
                transcription_model, person, words, voice,
                music, interaction, culture, tuner_profile, tuner_selected, tuner_display,
                tuner_reduced, tuner_contrast, tuner_numbers, tuner_speed, tuner_smoothing,
                tuner_sensitivity, tuner_frozen, target_person, target_words, target_voice,
                target_music, target_interaction, target_culture, request: gr.Request):
    values = (person, words, voice, music, interaction, culture)
    start = time.perf_counter()
    state = state or LiveSession()
    if chunk is None:
        rendered = render_cached_tuner(
            state, tuner_profile, tuner_selected, tuner_display, tuner_reduced,
            tuner_contrast, tuner_numbers, tuner_speed, tuner_smoothing, tuner_sensitivity,
            tuner_frozen, target_person, target_words, target_voice, target_music,
            target_interaction, target_culture,
        )
        return (state, transcript, "Waiting for microphone audio…", "", pd.DataFrame(),
                waveform(None), None, "", pd.DataFrame(), None,
                "_Waiting for completed transcribed bars…_", *rendered)
    data = state.buffer.append(chunk)
    state.generation += 1
    duration = len(data) / state.buffer.sample_rate
    asr_warning = ""
    if (lyrics_source == "Transcribe microphone" and
            duration - state.last_transcription_duration >= 3):
        client = None
        try:
            if stt_provider == "OpenAI transcription API":
                client = openai_client(
                    credential_source, get_key(request.session_hash, "openai"), cloud_consent
                )
            overlap_samples = int(state.buffer.sample_rate * 12)
            new_text, provenance = transcribe_selected(
                stt_provider, (state.buffer.sample_rate, data[-overlap_samples:]), client,
                transcription_model,
            )
            state.stable_transcript = deduplicate_text(state.stable_transcript, new_text)
            transcript = state.stable_transcript
            state.last_transcription_duration = duration
            asr_warning = f" · transcript: {provenance}"
        except Exception as exc:
            detail = str(exc).lower()
            if "no speech was detected" in detail:
                asr_warning = " · no speech detected in the latest live window"
            else:
                asr_warning = f" · {safe_provider_error(exc)} ({stt_provider})"
        finally:
            if client is not None:
                client.close()
    bars = segment_text(transcript or state.stable_transcript or "[provisional audio bar]", duration, bpm)
    enrich_bars_audio(bars, state.buffer.sample_rate, data)
    state.bar_history = bars
    active = latest_four_completed(bars)
    features, confidence, warning = _features(values, transcript or "", (state.buffer.sample_rate, data))
    semantic_text = (transcript or state.stable_transcript or "").strip()
    if semantic_text and semantic_text != state.semantic_text:
        state.semantic_features, state.semantic_provider = _selected_semantics(
            semantic_provider, semantic_text, credential_source, cloud_consent, request
        )
        state.semantic_text = semantic_text
    if state.semantic_features:
        for name, estimate in state.semantic_features.items():
            features[name] = 0.5 * features[name] + 0.5 * estimate
    context = _context(contexts, familiarity)
    bundle = SCORER.score(features, context, confidence)
    inverse = INVERSE.recommend(
        features, context, targets or ["Intelligibility"], desired,
        set(protected_features or []), set(protected_outputs or []), max_change,
        evidence="Active bars: " + " | ".join(
            f"bar {x.number} ({x.syllables_per_second:.1f} syllables/s): {x.transcript}"
            for x in active
        ),
        workflow="live",
    )
    state.latency_ms = (time.perf_counter() - start) * 1000
    frame = pd.DataFrame([x.model_dump() for x in bars])
    status = (
        f"Latest {len(active)} completed/estimated bars active · {state.latency_ms:.0f} ms "
        f"· semantic: {state.semantic_provider} · {warning}{asr_warning}"
    )
    scoring_trace = "\n\n---\n\n".join(trace_text(x) for x in bundle.outputs[:6])
    recommendations = pd.DataFrame([x.model_dump() for x in inverse.recommendations])
    first_recommendation = (
        inverse.recommendations[0].model_dump() if inverse.recommendations else {}
    )
    state.latest_analysis = {
        "features": dict(features), "confidence": confidence,
        "context_name": (contexts[0] if isinstance(contexts, list) and contexts else contexts or "Cypher"),
        "active_bars": [x.model_dump() for x in active],
        "completed_bar_count": len(active),
        "active_window_id": "-".join(str(x.number) for x in active) or None,
        "outputs": {x.name: x.score for x in bundle.outputs},
        "interactions": dict(bundle.interactions), "recommendation": first_recommendation,
        "latency_ms": state.latency_ms, "transcript": transcript or state.stable_transcript,
        "provenance": f"shared live analysis · semantic {state.semantic_provider}",
        "updated_at": time.time(),
    }
    rendered = render_cached_tuner(
        state, tuner_profile, tuner_selected, tuner_display, tuner_reduced,
        tuner_contrast, tuner_numbers, tuner_speed, tuner_smoothing, tuner_sensitivity,
        tuner_frozen, target_person, target_words, target_voice, target_music,
        target_interaction, target_culture,
    )
    return (
        state, transcript, status, _scored_bars_markdown(active, active_only=False), frame,
        waveform((state.buffer.sample_rate, data), bars), output_plot(bundle), scoring_trace,
        recommendations, _before_after(inverse),
        _latest_two_transcript(active, transcript or state.stable_transcript), *rendered,
    )


def render_cached_tuner(state, profile, selected, display, reduced, contrast, numbers,
                        speed, smoothing, sensitivity, frozen, person, words, voice,
                        music, interaction, culture):
    state = state or LiveSession()
    manual = dict(zip(LAYERS, (person, words, voice, music, interaction, culture), strict=True))
    view = live_analysis_to_tuner_view(
        state, profile, selected, manual, sensitivity, smoothing, frozen,
    )
    if frozen and not state.frozen_tuner_view:
        state.frozen_tuner_view = view.model_dump()
    elif not frozen:
        state.frozen_tuner_view = None
    state.previous_tuner_view = view.model_dump()
    return render_tuner(
        view, reduced_motion=reduced or display == "Static accessible",
        high_contrast=contrast, show_numbers=numbers, paused=state.tuner_paused,
        motion_speed=speed, full=display == "Full Mixer",
    )


def toggle_tuner_pause(state):
    state = state or LiveSession()
    state.tuner_paused = not state.tuner_paused
    return state, "Resume animation" if state.tuner_paused else "Pause animation"


def save_tuner_target(state):
    state = state or LiveSession()
    features = (state.latest_analysis or {}).get("features", {})
    if not features:
        return state, "No live window is available to save."
    state.saved_tuner_target = {name: float(features[name]) for name in LAYERS if name in features}
    return state, "Saved the current live window as a session reference."


def reset_tuner_display():
    rendered = render_cached_tuner(
        LiveSession(), "Clear and direct", LAYERS, "Focus", False, False, True,
        40, 68, 78, False, 65, 65, 65, 65, 65, 65,
    )
    transcript = _latest_two_transcript([], "")
    return transcript, *rendered, "Pause animation", ""


def save_session_credential(provider: str, value: str, request: gr.Request):
    """Hold a key in the session vault and switch the credential source with it.

    Returns a status message plus an update for the Credential source radio so
    a pasted key is immediately usable without extra clicks.
    """
    name = provider.lower().split()[0]
    if not (value or "").strip():
        clear_key(request.session_hash, name)
        return f"{provider} key box is empty — no key held.", gr.update()
    store_key(request.session_hash, name, value.strip())
    return (
        f"{provider} key held in server-side session memory and selected for use. "
        "Cloud calls also need the consent box checked.",
        gr.update(value="Use my own API key"),
    )


def transcribe_once(audio, source, consent, model, request: gr.Request):
    if audio is None:
        return "", "Record or upload audio first."
    try:
        client = openai_client(source, get_key(request.session_hash, "openai"), consent)
        transcript = transcribe_numpy(client, audio, model)
        return transcript, "Transcribed with OpenAI. Review the visible text before scoring."
    except Exception as exc:
        return "", f"{safe_provider_error(exc)}. Check consent, credential source, key, and model access."


def test_provider_connection(provider, source, consent, request: gr.Request):
    try:
        normalized = provider.lower()
        if normalized == "openai":
            client = openai_client(source, get_key(request.session_hash, "openai"), consent)
            return test_openai(client)
        if normalized == "gemini":
            client = gemini_client(source, get_key(request.session_hash, "gemini"), consent)
            next(iter(client.models.list(config={"page_size": 1})), None)
            client.close()
            return "Connected"
        return "Provider unavailable"
    except Exception as exc:
        return safe_provider_error(exc)


def analyze_youtube_ui(url, source, consent, model, current_bpm, request: gr.Request):
    client = None
    try:
        client = gemini_client(source, get_key(request.session_hash, "gemini"), consent)
        result = analyze_youtube(client, url, model)
        transcript = result.transcript or "\n".join(result.bars)
        bpm = result.bpm_estimate or current_bpm
        notice = (
            f"Gemini YouTube analysis complete · confidence {result.confidence:.0%}. "
            f"{result.summary} Audio-only measurements remain unavailable until audio is supplied."
        )
        return transcript, bpm, notice
    except Exception as exc:
        return "", current_bpm, youtube_error_message(exc)
    finally:
        if client is not None:
            client.close()


def commercial_youtube_ui(url, source, consent, model, current_bpm, request: gr.Request):
    client = None
    try:
        client = gemini_client(source, get_key(request.session_hash, "gemini"), consent)
        result = analyze_youtube(client, url, model, "commercial rap-song analysis")
        transcript = result.transcript or "\n".join(result.bars)
        if not transcript.strip():
            raise ValueError("Gemini did not detect lyrics in this video.")
        return transcript, result.bpm_estimate or current_bpm, (
            f"YouTube lyrics imported · confidence {result.confidence:.0%}. "
            f"{result.summary} Review the transcript before commercial analysis."
        )
    except Exception as exc:
        return "", current_bpm, youtube_error_message(exc)
    finally:
        if client is not None:
            client.close()


def analyze_commercial_ui(
    audio, supplied_lyrics, transcript, lyrics_source, bpm, semantic_provider, stt_provider,
    credential_source_value, cloud_consent, transcription_model, request: gr.Request,
):
    resolved = supplied_lyrics or ""
    provenance = "supplied lyrics"
    transcription_client = None
    if lyrics_source == "Use imported / transcribed text":
        resolved = transcript or ""
        provenance = "visible imported/transcribed text"
    elif lyrics_source == "Transcribe recording":
        if audio is None:
            return transcript or "", "Record or upload audio first.", "", None, None, None, ""
        try:
            if stt_provider == "OpenAI transcription API":
                transcription_client = openai_client(
                    credential_source_value, get_key(request.session_hash, "openai"),
                    cloud_consent,
                )
            resolved, provenance = transcribe_selected(
                stt_provider, audio, transcription_client, transcription_model
            )
        except Exception as exc:
            return transcript or "", safe_provider_error(exc), "", None, None, None, ""
        finally:
            if transcription_client is not None:
                transcription_client.close()
    resolved = (resolved or "").strip()
    if not resolved:
        return transcript or "", "No lyrics or transcript are available to analyze.", "", None, None, None, ""
    features, confidence, warning = estimate_features(
        [55] * 6, resolved, audio, "Auto-estimate from performance"
    )
    semantic_values, semantic_notice = _selected_semantics(
        semantic_provider, resolved, credential_source_value, cloud_consent, request
    )
    if semantic_values:
        for name, estimate in semantic_values.items():
            features[name] = 0.5 * features[name] + 0.5 * estimate
    indicators, advice, plot = commercial_frames(features, resolved, confidence)
    overall = indicators["score"].mean()
    status = (
        f"Commercial-readiness analysis complete · overall directional index {overall:.1f}/100 "
        f"· text: {provenance} · semantic: {semantic_notice}. {warning}"
    )
    explanation = (
        "These indicators estimate observable readiness factors—not sales, streams, label interest, "
        "or guaranteed success. The green bars are modeled effects of one controlled revision, not "
        "market forecasts. Audio affects Voice/Production; text affects hook, replay, connection, and "
        f"distinctiveness. BPM used for alignment context: {bpm:.0f}."
    )
    return resolved, status, f"> {resolved}", indicators, advice, plot, explanation


def clear_session_credentials(request: gr.Request) -> str:
    return clear_keys(request.session_hash)


def clear_one_credential(provider: str, request: gr.Request) -> str:
    clear_key(request.session_hash, provider.lower())
    return f"{provider} credential cleared."


def hold_openai_key(value: str, request: gr.Request):
    return save_session_credential("OpenAI", value, request)


def hold_gemini_key(value: str, request: gr.Request):
    return save_session_credential("Gemini", value, request)


def hold_anthropic_key(value: str, request: gr.Request):
    return save_session_credential("Anthropic", value, request)


def test_openai_key(source, consent, request: gr.Request) -> str:
    return test_provider_connection("OpenAI", source, consent, request)


def test_gemini_key(source, consent, request: gr.Request) -> str:
    return test_provider_connection("Gemini", source, consent, request)


def test_anthropic_key(source, consent, request: gr.Request) -> str:
    try:
        require_cloud_consent("anthropic", consent)
        key = resolve_key(
            "anthropic", normalize_credential_source(source),
            get_key(request.session_hash, "anthropic"), Settings.from_env(),
        )
        model = os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key.get_secret_value(), "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "Reply OK"}]},
            timeout=30,
        )
        if response.status_code == 400:
            message = response.json().get("error", {}).get("message", "")
            if "credit balance" in message.lower():
                return "Key accepted · Anthropic credit balance is too low"
        response.raise_for_status()
        return f"Connected · {model}"
    except Exception as exc:
        return safe_provider_error(exc)


def clear_openai_key(request: gr.Request) -> str:
    return clear_one_credential("OpenAI", request)


def clear_gemini_key(request: gr.Request) -> str:
    return clear_one_credential("Gemini", request)


def clear_anthropic_key(request: gr.Request) -> str:
    return clear_one_credential("Anthropic", request)


def reset_session(request: gr.Request):
    return None, None, None, clear_keys(request.session_hash)


def build_app():
    initial_tuner = render_cached_tuner(
        LiveSession(), "Clear and direct", LAYERS, "Focus", False, False, True,
        40, 68, 78, False, 65, 65, 65, 65, 65, 65,
    )
    with gr.Blocks(title="The Rap Mixer") as demo:
        gr.Markdown("""<div class='hero'><h1>The Rap Mixer</h1><h3>What the artist makes × what the context rewards → several outcomes, not one universal score</h3><p class='disclaimer'>Scores explain a configuration in a selected context. They are not objective artistic truth.</p></div>""")
        bundle_state = gr.State(None)
        live_state = gr.State(None)
        with gr.Accordion("API keys, models & data sharing — this one panel is used by every tab", open=True):
            with gr.Row():
                stt = gr.Dropdown(["faster-whisper", "transformers-whisper", "OpenAI transcription API"], value="faster-whisper", label="Speech-to-text")
                semantic = gr.Dropdown(["Deterministic local baseline", "Ollama", "OpenAI", "Anthropic Claude", "Google Gemini"], value="Deterministic local baseline", label="Semantic analysis")
                source = gr.Radio(
                    ["Local/open-source—no API key", "Use my own API key"],
                    value="Local/open-source—no API key",
                    label="Credential source",
                )
            gr.Markdown(
                "This public app has no shared or owner API keys. Use the free local modes, "
                "or paste your own provider key below — it is held automatically in "
                "server-side session memory and used by all tabs. Cloud calls also require "
                "the consent box."
            )
            gr.Markdown("Cloud sharing notice: depending on the provider, audio, transcript text, or extracted features may leave this server. Local modes send nothing.")
            consent = gr.Checkbox(label="I consent to this session's described cloud data transfer")
            gr.Markdown(
                "Enter providers independently. Both keys can be held at the same time; "
                "they never enter serialized Gradio state."
            )
            with gr.Row():
                with gr.Column():
                    openai_key = gr.Textbox(
                        type="password", label="OpenAI API key",
                        placeholder="Session memory only", interactive=True,
                    )
                    openai_status = gr.Textbox(
                        label="OpenAI status", interactive=False
                    )
                    with gr.Row():
                        save_openai = gr.Button("Hold OpenAI key")
                        test_openai_button = gr.Button("Test OpenAI")
                        clear_openai = gr.Button("Clear OpenAI key")
                with gr.Column():
                    gemini_key = gr.Textbox(
                        type="password", label="Gemini API key",
                        placeholder="Session memory only", interactive=True,
                    )
                    gemini_status = gr.Textbox(
                        label="Gemini status", interactive=False
                    )
                    with gr.Row():
                        save_gemini = gr.Button("Hold Gemini key")
                        test_gemini_button = gr.Button("Test Gemini")
                        clear_gemini = gr.Button("Clear Gemini key")
                with gr.Column():
                    anthropic_key = gr.Textbox(
                        type="password", label="Anthropic API key",
                        placeholder="Session memory only", interactive=True,
                    )
                    anthropic_status = gr.Textbox(
                        label="Anthropic status", interactive=False
                    )
                    with gr.Row():
                        save_anthropic = gr.Button("Hold Anthropic key")
                        test_anthropic_button = gr.Button("Test Anthropic")
                        clear_anthropic = gr.Button("Clear Anthropic key")
            with gr.Row():
                transcription_model = gr.Textbox(
                    value=os.getenv("DEFAULT_OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe"),
                    label="Transcription model ID",
                )
                generation_model = gr.Textbox(
                    value=os.getenv("DEFAULT_OPENAI_MODEL", "gpt-5.4-mini"),
                    label="Battle/text generation model ID",
                )
                gemini_model = gr.Textbox(
                    value=os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.5-flash"),
                    label="Gemini YouTube model ID",
                )
            credential_status = gr.Textbox(label="Session credential status", interactive=False)
            save_openai.click(
                hold_openai_key, openai_key, [openai_status, source],
            )
            save_gemini.click(
                hold_gemini_key, gemini_key, [gemini_status, source],
            )
            save_anthropic.click(
                hold_anthropic_key, anthropic_key, [anthropic_status, source],
            )
            # Pasting or typing a key holds it immediately; the buttons stay as an
            # explicit alternative. `.input` only fires on user edits, so session
            # resets that clear these boxes do not re-hold anything.
            openai_key.input(
                hold_openai_key, openai_key, [openai_status, source],
            )
            gemini_key.input(
                hold_gemini_key, gemini_key, [gemini_status, source],
            )
            anthropic_key.input(
                hold_anthropic_key, anthropic_key, [anthropic_status, source],
            )
            test_openai_button.click(
                test_openai_key, [source, consent], openai_status,
            )
            test_gemini_button.click(
                test_gemini_key, [source, consent], gemini_status,
            )
            test_anthropic_button.click(
                test_anthropic_key, [source, consent], anthropic_status,
            )
            clear_openai.click(
                clear_openai_key, None, openai_status
            )
            clear_gemini.click(
                clear_gemini_key, None, gemini_status
            )
            clear_anthropic.click(
                clear_anthropic_key, None, anthropic_status
            )
            gr.Button("Clear all credentials").click(
                clear_session_credentials, None, credential_status
            )

        with gr.Tabs():
            with gr.Tab("Pre-recorded bars"):
                with gr.Row():
                    audio = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Performance (upload or complete recording)")
                    instrumental = gr.Audio(sources=["upload"], type="numpy", label="Optional instrumental")
                with gr.Row():
                    supplied_lyrics = gr.Textbox(
                        lines=8,
                        label="Supplied lyrics",
                        placeholder="Type or paste the lyrics you want scored.",
                    )
                    transcript = gr.Textbox(
                        lines=8,
                        label="Transcribed audio / YouTube transcript",
                        placeholder="Speech-to-text appears here and remains visible for review.",
                    )
                lyrics_source = gr.Radio(
                    ["Use supplied lyrics", "Use transcribed audio"],
                    value="Use supplied lyrics",
                    label="Lyrics source",
                    info="Transcribe first, review the visible text, then select it for scoring.",
                )
                with gr.Row():
                    youtube_url = gr.Textbox(label="Public YouTube URL")
                    youtube_button = gr.Button("Analyze YouTube with Gemini")
                transcribe_status = gr.Markdown()
                gr.Button("Transcribe recording with selected OpenAI model").click(
                    transcribe_once,
                    [audio, source, consent, transcription_model],
                    [transcript, transcribe_status],
                )
                with gr.Row():
                    bpm = gr.Number(value=90, minimum=30, maximum=300, label="BPM")
                    time_sig = gr.Dropdown(["4/4", "3/4", "6/8"], value="4/4", label="Time signature")
                    contexts = gr.Dropdown(list(CONTEXTS), value=["Cypher"], multiselect=True, label="Blend contexts")
                    familiarity = gr.Slider(0, 100, 50, label="Audience familiarity")
                youtube_button.click(
                    analyze_youtube_ui,
                    [youtube_url, source, consent, gemini_model, bpm],
                    [transcript, bpm, transcribe_status],
                )
                gr.Markdown("### Bank A — artist-controlled performance")
                sliders = []
                with gr.Row():
                    for name in LAYERS:
                        sliders.append(gr.Slider(0, 100, 55, label=name.title()))
                mode = gr.Radio(["Auto-estimate from performance", "Manually configure", "Auto-estimate, then edit"], value="Auto-estimate, then edit", label="A input mode")
                run = gr.Button("Run forward analysis", variant="primary")
                status = gr.Markdown()
                scored_text = gr.Markdown(label="Transcript being scored")
                with gr.Row():
                    outplot = gr.Plot(label="Output meters with uncertainty")
                    aplot = gr.Plot(label="A parameter radar")
                wave = gr.Plot()
                table = gr.Dataframe(label="Bar analysis table", interactive=False, wrap=True)
                with gr.Accordion("How was this calculated?", open=False):
                    trace = gr.Markdown()
                export = gr.File(label="Explicit JSON export")
                run.click(
                    analyze_with_lyrics_source,
                    [audio, supplied_lyrics, transcript, lyrics_source, bpm, contexts, familiarity,
                     mode, stt, semantic, source,
                     consent, transcription_model, *sliders],
                    [transcript, status, scored_text, outplot, aplot, wave, table, trace,
                     bundle_state, export],
                )
                gr.Markdown("### Inverse analysis — improve outcomes while protecting identity")
                outputs = list(SCORER.config["outputs"])
                with gr.Row():
                    targets = gr.Dropdown(
                        outputs, value=outputs, multiselect=True,
                        label="Targets (all selected by default)",
                    )
                    protected_a = gr.Dropdown(LAYERS, multiselect=True, label="Protected A dimensions")
                    protected_o = gr.Dropdown(outputs, multiselect=True, label="Protected outputs")
                    desired = gr.Slider(1, 20, 5, label="Desired improvement")
                    max_change = gr.Slider(1, 25, 12, label="Maximum A change")
                inv = gr.Button("Find specific revisions")
                inv_status = gr.Markdown()
                inv_plot = gr.Plot()
                inv_table = gr.Dataframe(label="Prioritized recommendations", wrap=True)
                inv.click(recommend_with_lyrics_source,
                          [bundle_state, targets, protected_a, protected_o, desired,
                           max_change, supplied_lyrics, transcript, lyrics_source, *sliders],
                          [inv_status, inv_plot, inv_table])

            with gr.Tab("Live four-bar analysis"):
                gr.Markdown("Audio is held in a bounded 120-second ring buffer. The newest four estimated completed bars are active; full bar history stays below.")
                live_audio = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Live microphone")
                live_transcript = gr.Textbox(lines=7, label="Complete transcript / live corrections", placeholder="Optional local transcript or corrections; never reset by the stream.")
                with gr.Row():
                    live_lyrics_source = gr.Radio(
                        ["Use supplied lyrics", "Transcribe microphone"],
                        value="Transcribe microphone",
                        label="Lyrics source",
                    )
                    live_stt = gr.Dropdown(
                        ["faster-whisper", "transformers-whisper", "OpenAI transcription API"],
                        value="OpenAI transcription API",
                        label="Live transcription provider",
                        info="For OpenAI, select an API credential source and consent above.",
                    )
                with gr.Row():
                    live_bpm = gr.Number(value=90, label="BPM")
                    live_context = gr.Dropdown(list(CONTEXTS), value=["Cypher"], multiselect=True, label="Context")
                    live_familiarity = gr.Slider(0, 100, 50, label="Audience familiarity")
                live_sliders = []
                with gr.Row():
                    for name in LAYERS:
                        live_sliders.append(gr.Slider(0, 100, 55, label=name.title()))
                live_status = gr.Markdown("Waiting for microphone audio…")
                gr.Markdown("### Active transcribed/corrected text being scored")
                live_active_text = gr.Markdown("_No completed text bars yet._")
                with gr.Row():
                    live_wave = gr.Plot()
                    live_outputs = gr.Plot()
                live_table = gr.Dataframe(label="Complete bar history (active latest four highlighted in active column)", wrap=True)
                with gr.Accordion("Live scoring calculation — how is this text being scored?", open=True):
                    live_trace = gr.Markdown()
                gr.Markdown("### Real-time inverse suggestions")
                with gr.Row():
                    live_targets = gr.Dropdown(outputs, value=["Intelligibility"], multiselect=True, label="Live targets")
                    live_protected_a = gr.Dropdown(LAYERS, multiselect=True, label="Protected A dimensions")
                    live_protected_o = gr.Dropdown(outputs, multiselect=True, label="Protected outputs")
                    live_desired = gr.Slider(1, 20, 5, label="Desired improvement")
                    live_max_change = gr.Slider(1, 25, 12, label="Maximum A change")
                live_recommendations = gr.Dataframe(label="Suggestions refreshed with the active window", wrap=True)
                live_inverse_plot = gr.Plot(label="Predicted current versus proposed outputs")
                gr.Markdown(
                    "## Performance Tuner\nA glanceable view of the same Live Feedback "
                    "session. It does not open another microphone or rerun transcription, scoring, "
                    "semantic analysis, or recommendations. Amber motion shows the correction direction; "
                    "the cyan corridor is the target; green means hold. This is visual coaching, not a "
                    "medical biofeedback device. Targets are creative guides—not universal standards."
                )
                with gr.Row():
                    tuner_profile = gr.Dropdown(
                        [*PROFILES, "Custom"], value="Clear and direct", label="Creative target profile"
                    )
                    tuner_context = gr.Dropdown(
                        list(CONTEXTS), value=["Cypher"], multiselect=True,
                        label="Shared live context",
                    )
                    tuner_selected = gr.Dropdown(
                        LAYERS, value=LAYERS, multiselect=True,
                        label="Metrics included in Master Lock",
                    )
                    tuner_display = gr.Radio(
                        ["Focus", "Full Mixer", "Static accessible"], value="Focus",
                        label="Display mode",
                    )
                gr.Markdown("### Live words feeding the tuner")
                tuner_latest_two = gr.HTML(_latest_two_transcript([], ""))
                tuner_master = gr.HTML(initial_tuner[0])
                tuner_cue = gr.HTML(initial_tuner[3])
                tuner_bank = gr.HTML(initial_tuner[1])
                tuner_full = gr.HTML(initial_tuner[2])
                gr.Markdown("### Latest four completed bars")
                tuner_timeline = gr.HTML(initial_tuner[4])
                with gr.Accordion("Targets, comparison and accessibility", open=False):
                    with gr.Row():
                        tuner_reduced = gr.Checkbox(label="Reduced Motion")
                        tuner_contrast = gr.Checkbox(label="High Contrast")
                        tuner_numbers = gr.Checkbox(value=True, label="Show numeric values")
                        tuner_frozen = gr.Checkbox(label="Freeze current tuner view")
                    with gr.Row():
                        tuner_speed = gr.Slider(0, 100, 40, label="Motion speed")
                        tuner_smoothing = gr.Slider(8, 95, 68, label="Responsiveness")
                        tuner_sensitivity = gr.Slider(0, 100, 78, label="Target sensitivity")
                    tuner_targets = []
                    with gr.Row():
                        for name in LAYERS:
                            tuner_targets.append(gr.Slider(0, 100, 65, label=f"{name.title()} target"))
                    tuner_details = gr.Markdown(initial_tuner[5] or "_No live analysis yet._")
                with gr.Row():
                    tuner_pause = gr.Button("Pause animation")
                    tuner_save = gr.Button("Save current window target")
                    tuner_stop = gr.Button("Global Stop", variant="stop")
                tuner_save_status = gr.Markdown()

                # User input synchronizes the context controls; cached tuner renders never invoke inference.
                tuner_context.input(lambda value: value, tuner_context, live_context)
                live_context.input(lambda value: value, live_context, tuner_context)

                tuner_settings = [
                    tuner_profile, tuner_selected, tuner_display, tuner_reduced, tuner_contrast,
                    tuner_numbers, tuner_speed, tuner_smoothing, tuner_sensitivity, tuner_frozen,
                    *tuner_targets,
                ]
                tuner_render_outputs = [
                    tuner_master, tuner_bank, tuner_full, tuner_cue, tuner_timeline, tuner_details,
                ]
                for control in tuner_settings:
                    control.input(
                        render_cached_tuner, [live_state, *tuner_settings], tuner_render_outputs,
                        queue=False,
                    )
                tuner_pause.click(toggle_tuner_pause, live_state, [live_state, tuner_pause], queue=False)
                tuner_save.click(save_tuner_target, live_state, [live_state, tuner_save_status], queue=False)
            with gr.Tab("Commercial Success Indicators"):
                gr.Markdown(
                    "## Commercial readiness and revision advice\n"
                    "Directional evidence only: this does not predict or guarantee sales, streams, "
                    "virality, playlisting, or label interest."
                )
                with gr.Row():
                    commercial_audio = gr.Audio(
                        sources=["upload", "microphone"], type="numpy",
                        label="Song/performance — upload or record with the microphone",
                    )
                    with gr.Column():
                        commercial_supplied = gr.Textbox(
                            lines=6, label="Supplied lyrics",
                            placeholder="Paste lyrics here.",
                        )
                        commercial_transcript = gr.Textbox(
                            lines=6, label="Imported / transcribed lyrics",
                            placeholder="Recording or YouTube transcription appears here.",
                        )
                commercial_source = gr.Radio(
                    ["Use supplied lyrics", "Use imported / transcribed text", "Transcribe recording"],
                    value="Use supplied lyrics", label="Evidence source",
                )
                with gr.Row():
                    commercial_youtube = gr.Textbox(label="Public YouTube song URL")
                    commercial_youtube_button = gr.Button("Import song with Gemini")
                    commercial_bpm = gr.Number(value=90, minimum=30, maximum=300, label="BPM")
                commercial_status = gr.Markdown()
                commercial_youtube_button.click(
                    commercial_youtube_ui,
                    [commercial_youtube, source, consent, gemini_model, commercial_bpm],
                    [commercial_transcript, commercial_bpm, commercial_status],
                ).then(lambda: "Use imported / transcribed text", None, commercial_source)
                commercial_run = gr.Button(
                    "Analyze commercial readiness and find improvements", variant="primary"
                )
                commercial_scored = gr.Markdown(label="Lyrics being analyzed")
                commercial_plot = gr.Plot(label="Current versus modeled revision")
                with gr.Row():
                    commercial_table = gr.Dataframe(
                        label="Commercial-readiness indicators", interactive=False, wrap=True
                    )
                    commercial_advice = gr.Dataframe(
                        label="Prioritized commercial revisions", interactive=False, wrap=True
                    )
                commercial_explanation = gr.Markdown()
                commercial_run.click(
                    analyze_commercial_ui,
                    [commercial_audio, commercial_supplied, commercial_transcript,
                     commercial_source, commercial_bpm, semantic, stt, source, consent,
                     transcription_model],
                    [commercial_transcript, commercial_status, commercial_scored,
                     commercial_table, commercial_advice, commercial_plot,
                     commercial_explanation],
                )
            with gr.Tab("Live Battle Rapper AI", id="live-battle-rapper-ai"):
                battle_state, battle_clearable = build_battle_content(
                    SCORER, CONTEXTS, stt, source, consent, transcription_model,
                    generation_model, gemini_model,
                )
            with gr.Tab("Human vs Human Battle"):
                human_battle_state, human_battle_clearable = build_human_battle_tab(
                    SCORER, CONTEXTS, stt, source, consent, transcription_model,
                )
        # One microphone event fans the same computed live state into detailed and tuner views.
        live_event = live_audio.stream(
            live_update,
            [live_audio, live_state, live_bpm, live_transcript, live_context,
             live_familiarity, live_targets, live_protected_a, live_protected_o,
             live_desired, live_max_change, live_lyrics_source, live_stt, semantic,
             source, consent, transcription_model, *live_sliders, *tuner_settings],
            [live_state, live_transcript, live_status, live_active_text, live_table,
             live_wave, live_outputs, live_trace, live_recommendations, live_inverse_plot,
             tuner_latest_two, *tuner_render_outputs],
            stream_every=1.0,
            time_limit=600,
        )
        tuner_stop.click(
            toggle_tuner_pause, live_state, [live_state, tuner_pause],
            cancels=[live_event], queue=False,
        )
        clearable = [
            openai_key, gemini_key, anthropic_key, openai_status, gemini_status,
            anthropic_status, audio, instrumental,
            youtube_url, supplied_lyrics, transcript, lyrics_source, transcribe_status,
            status, scored_text,
            outplot, aplot, wave, table, trace, export, inv_status, inv_plot, inv_table,
            live_audio, live_transcript, live_lyrics_source, live_stt, live_status, live_active_text,
            live_wave,
            live_outputs, live_table, live_trace, live_recommendations, live_inverse_plot,
            tuner_save_status, tuner_master, tuner_bank, tuner_full, tuner_cue, tuner_timeline,
            tuner_details, tuner_latest_two,
            commercial_audio, commercial_supplied, commercial_transcript, commercial_source,
            commercial_youtube, commercial_status, commercial_scored, commercial_plot,
            commercial_table, commercial_advice, commercial_explanation,
            human_battle_state, *human_battle_clearable, *battle_clearable,
        ]
        clear = gr.ClearButton(clearable, value="Clear session")
        clear.click(
            reset_session, None,
            [bundle_state, live_state, battle_state, credential_status],
        ).then(
            reset_tuner_display, None,
            [tuner_latest_two, *tuner_render_outputs, tuner_pause, tuner_save_status],
            queue=False,
        )
        gr.Markdown("Prototype coefficients and cultural interpretations are configurable assumptions, not empirically validated truth. No audio or transcripts are retained unless you explicitly export.")
    return demo
