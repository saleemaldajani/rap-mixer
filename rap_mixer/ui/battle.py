from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from rap_mixer.analysis.feature_estimation import estimate_features
from rap_mixer.analysis.lyrics import lyric_features
from rap_mixer.battle.engine import run_battle_round
from rap_mixer.battle.safety import validate_boundaries
from rap_mixer.battle.state_machine import BattleSession
from rap_mixer.battle.verse_plan import generate_bars
from rap_mixer.instrumentals.magenta_rt import MagentaRealtimeProvider
from rap_mixer.instrumentals.metronome import metronome
from rap_mixer.instrumentals.procedural import generated_beat
from rap_mixer.instrumentals.uploaded import UploadedInstrumentalProvider
from rap_mixer.performance.cadence import cadence_plan
from rap_mixer.performance.mixing import mix_voice_and_backing
from rap_mixer.performance.voice import LocalSyntheticVoiceProvider
from rap_mixer.providers.freestyle_generation import generate_freestyle
from rap_mixer.providers.gemini_youtube import (
    analyze_youtube,
    gemini_client,
    youtube_error_message,
)
from rap_mixer.providers.openai_audio import (
    generate_structured_battle,
    openai_client,
    transcribe_numpy,
)
from rap_mixer.security.redaction import safe_provider_error
from rap_mixer.transcription.router import transcribe_selected
from rap_mixer.ui.credentials import get_key
from rap_mixer.ui.plots import output_plot

LAYERS = ["person", "words", "voice", "music", "interaction", "culture"]
FREESTYLE_RUNS = itertools.count()


def _context(contexts, names):
    names = names or ["Battle"]
    result = {k: 0.0 for k in next(iter(contexts.values()))}
    for name in names:
        base = contexts.get(name, contexts.get("Battle"))
        for key, value in base.items():
            result[key] += value / len(names)
    return result


def _latency_plot(latency):
    fig = go.Figure(go.Bar(x=list(latency.values()), y=list(latency), orientation="h"))
    fig.update_layout(template="plotly_dark", title="Pipeline latency (ms)", height=320)
    return fig


def _comparison(human, ai):
    hs = {x.name: x.score for x in human.outputs[:6]}
    fig = output_plot(ai, hs)
    fig.data[0].name = "AI"
    fig.data[1].name = "Human"
    return fig


def _performer_lyrics(generated) -> str:
    if not generated:
        return "_Generate a response to see performer lyrics._"
    lines = [f"> **{index}.** {bar.text}" for index, bar in enumerate(generated, 1)]
    return "### Your lyrics — rap these over the playback\n\n" + "\n\n".join(lines)


def perform_round(scorer, contexts, session, transcript, audio, bpm, human_count, response_count, agreement,
                  intensity, humor, aggression, directness, allowed, disallowed, profanity,
                  output_mode, context_names, lyrics_source, stt_provider, provider_mode,
                  backing_provider, backing_audio, enable_magenta, credential_source, cloud_consent,
                  transcription_model, generation_model, *values, request: gr.Request):
    provider_notice = "deterministic local"
    client = None
    if lyrics_source == "Transcribe recording":
        if audio is None:
            return (session, "**Record or upload the human round first.**", "", pd.DataFrame(),
                    "", pd.DataFrame(), None, None, "", None, None, pd.DataFrame(), "")
        try:
            transcription_client = None
            if stt_provider == "OpenAI transcription API":
                transcription_client = openai_client(
                    credential_source, get_key(request.session_hash, "openai"), cloud_consent
                )
            transcript, transcript_provider = transcribe_selected(
                stt_provider, audio, transcription_client, transcription_model
            )
            provider_notice = f"transcript: {transcript_provider} · deterministic local"
        except Exception as exc:
            return (session, f"**{safe_provider_error(exc)} while transcribing.**", "",
                    pd.DataFrame(), "", pd.DataFrame(), None, None, "", None, None, pd.DataFrame(), "")
    if provider_mode == "OpenAI":
        try:
            client = openai_client(
                credential_source, get_key(request.session_hash, "openai"), cloud_consent
            )
            provider_notice = f"OpenAI · {generation_model}"
        except Exception:
            provider_notice = "OpenAI unavailable; deterministic fallback"
    features, evidence_confidence, feature_warning = estimate_features(
        values, transcript or "", audio, "Auto-estimate, then edit"
    )
    features["person"] = max(0, min(100, (features["person"] + intensity) / 2))
    features["voice"] = max(0, min(100, (features["voice"] + intensity + aggression) / 3))
    features["interaction"] = max(0, min(100, (features["interaction"] + directness) / 2))
    features["words"] = max(0, min(100, (features["words"] + humor + directness) / 3))
    tone = f"{agreement.lower()}, intensity {intensity:.0f}, humor {humor:.0f}, aggression {aggression:.0f}, directness {directness:.0f}, {profanity.lower()}"
    try:
        result = run_battle_round(
            session, transcript, audio, bpm, int(human_count), int(response_count), tone,
            allowed, disallowed, output_mode, features, _context(contexts, context_names), scorer,
        )
    except ValueError as exc:
        return (session, f"**{exc}**", transcript, pd.DataFrame(), "", pd.DataFrame(),
                None, None, "", None, None, pd.DataFrame(), "")
    session, bars, graph, strategy, generated, human, ai, audio_path, latency = result
    if client is not None:
        try:
            human_with_ids = "\n".join(f"H{i}: {bar.transcript}" for i, bar in enumerate(bars, 1))
            cloud_bars = generate_structured_battle(
                client, generation_model, human_with_ids, strategy.model_dump(),
                int(response_count), disallowed,
                previous_responses=[
                    turn.transcript for turn in session.turns[:-1] if turn.speaker == "AI"
                ][-3:],
            )
            for generated_bar, cloud_bar in zip(generated, cloud_bars, strict=True):
                safe, reason = validate_boundaries(cloud_bar["text"], disallowed)
                if not safe:
                    raise ValueError(f"Cloud bar rejected by safety boundary: {reason}")
                generated_bar.text = cloud_bar["text"]
                generated_bar.addressed_human_bar_ids = cloud_bar["addressed_human_bar_ids"]
                generated_bar.function = cloud_bar.get("function", generated_bar.function)
                generated_bar.delivery_note = cloud_bar.get(
                    "delivery_note", generated_bar.delivery_note
                )
                plan = cadence_plan(generated_bar.text, bpm)
                generated_bar.target_syllables = plan["target_syllables"]
                generated_bar.stress_pattern = plan["stress_pattern"]
                generated_bar.warnings = plan["warnings"]
            verse = "\n".join(x.text for x in generated)
            cloud_features = dict(features)
            cloud_lyrics = lyric_features(verse)
            cloud_features["words"] = sum(cloud_lyrics.values()) / len(cloud_lyrics)
            cloud_features["interaction"] = min(
                100, 60 + 5 * len({bar_id for x in generated for bar_id in x.addressed_human_bar_ids})
            )
            ai = scorer.score(cloud_features, _context(contexts, context_names), 0.72)
            session.turns[-1].transcript = verse
            session.turns[-1].bar_texts = [x.text for x in generated]
            session.turns[-1].generated_bars = generated
            session.turns[-1].score_bundle = ai.model_dump()
            session.turns[-1].provider = provider_notice
        except Exception:
            provider_notice = "OpenAI generation failed; deterministic fallback"
    verse = "\n".join(x.text for x in generated)
    performed_mode = output_mode != "Text only"
    if performed_mode:
        audio_path = LocalSyntheticVoiceProvider().synthesize_performance(
            verse, bpm=bpm, energy=intensity, aggression=aggression,
        )
    active = "\n\n".join(
        f"**H{x.number} {'🟣' if x.active else '⚪'}** — {x.transcript}" for x in bars
    )
    strategy_md = (
        f"**Primary angle:** {strategy.primary_angle}\n\n"
        f"**Moves:** {', '.join(strategy.response_moves)}\n\n"
        f"**Human bars addressed:** {', '.join(strategy.human_lines_addressed)}\n\n"
        f"**Round arc:** {' → '.join(strategy.round_arc)}"
    )
    generated_frame = pd.DataFrame([x.model_dump() for x in generated])
    backing_output = None
    if performed_mode:
        backing = None
        backing_notice = "vocals only"
        if backing_provider.startswith(("MRT2", "Magenta")) and enable_magenta:
            try:
                engine = "Jam" if "Jam" in backing_provider else "Collider"
                backing = MagentaRealtimeProvider().continue_audio(
                    bpm, int(response_count), engine=engine
                )
                backing_notice = f"MRT2 {engine} live backing mixed under the voice"
            except Exception as exc:
                backing = generated_beat(
                    bpm, int(response_count), seed_text=verse,
                    profile={"energy": intensity / 100, "drum_density": aggression / 100,
                             "groove": "battle boom-bap"},
                )
                backing_notice = f"{exc} Local generated-beat fallback mixed under the voice"
        elif backing_provider == "Uploaded instrumental" and backing_audio is not None:
            backing = backing_audio
            backing_notice = "uploaded instrumental backing"
        elif backing_provider == "Metronome / click track":
            backing = metronome(bpm, int(response_count))
            backing_notice = "metronome backing"
        elif backing_provider == "Generated beat (local)":
            backing = generated_beat(
                bpm, int(response_count), seed_text=verse,
                profile={"energy": intensity / 100, "drum_density": aggression / 100,
                         "groove": "battle boom-bap" if directness > 55 else "laid-back"},
            )
            backing_notice = "local generated beat"
        backing_output = backing
        if backing is not None and audio_path:
            audio_path = mix_voice_and_backing(audio_path, backing)
        provider_notice = f"{provider_notice} · {backing_notice}"
    playback_label = "synthetic vocal performance" if audio_path else "text-only fallback"
    status = (
        f"State: **{session.phase}** · {provider_notice} · evidence confidence "
        f"{evidence_confidence:.0%} · {feature_warning} · AI text ready · voice: "
        f"{playback_label}"
    )
    return (session, status, transcript, pd.DataFrame(graph), strategy_md, generated_frame,
            backing_output, audio_path, _performer_lyrics(generated), _comparison(human, ai),
            _latency_plot(latency), _history(session), active)


def _history(session):
    return pd.DataFrame([{"turn": x.turn_id, "speaker": x.speaker, "transcript": x.transcript,
                          "provider": x.provider, "safety": x.safety_result,
                          "latency_ms": x.latency_ms.get("total", 0)} for x in (session.turns if session else [])])


def stop_battle(session):
    session = session or BattleSession()
    session.stop()
    return session, "State: **IDLE** · generation and queued performance stopped."


def export_history(session):
    path = tempfile.NamedTemporaryFile(prefix="rap-battle-", suffix=".json", delete=False, mode="w")
    json.dump({"turns": [x.model_dump() for x in (session.turns if session else [])]}, path, indent=2)
    path.close()
    return path.name


def freestyle(instrumental, instrumental_provider, enable_magenta, topic, message, seed_words,
              required, forbidden, bpm, bar_count, looseness, coherence, rhyme, *values):
    run_number = next(FREESTYLE_RUNS)
    analysis = UploadedInstrumentalProvider().analyze(instrumental)
    fake_nodes_text = [f"{topic}: {message}. {seed_words}. Required references: {required}"]
    from rap_mixer.battle.argument_graph import build_argument_graph
    from rap_mixer.battle.strategy import plan_strategy
    nodes, _ = build_argument_graph(fake_nodes_text)
    strategy = plan_strategy(nodes, int(bar_count), f"freestyle looseness {looseness}, coherence {coherence}", required, forbidden, bpm)
    bars = generate_bars(strategy, nodes, bpm, forbidden)
    mixer = dict(zip(LAYERS, values, strict=True))
    seeds = [x.strip() for x in seed_words.split(",") if x.strip()] or ["motion", "room"]
    references = [x.strip() for x in required.split(",") if x.strip()]
    rhyme_endings = (
        ["night", "flame", "flow", "air", "light", "name", "glow", "there"]
        if rhyme >= 50 else
        ["move", "clear", "ground", "time", "pace", "sound", "line", "space"]
    )
    templates = [
        "{topic} through the {seed}, I {message}, with {reference} in sight",
        "From {seed} to the stage, {reference} frames the way I {message}",
        "I turn {seed} into proof: {message}, while {reference} holds the frame",
        "Hear {topic} in the pulse of {seed}; I {message} and sharpen {reference}",
        "No copied route—through {seed} I {message}, then point back to {reference}",
        "The room shifts with {seed}; on {topic}, I {message} beside {reference}",
        "I plant {reference} on the downbeat, let {seed} move while I {message}",
        "New angle on {topic}: {seed} carries the moment as I {message}",
    ]
    for index, bar in enumerate(bars):
        seed = seeds[index % len(seeds)]
        reference = references[index % len(references)] if references else topic
        if coherence >= 50 or index in {0, len(bars) - 1}:
            template = templates[(run_number + index) % len(templates)]
            line = template.format(
                topic=topic, seed=seed, message=message, reference=reference
            )
        else:
            loose_images = ["neon", "backspin", "side street", "open sky", "train smoke"]
            image = loose_images[(run_number + index) % len(loose_images)]
            line = f"Loose association: {seed} meets {image}, then {topic} enters the frame"
        if looseness > 70 and index % 2:
            turns = ["swerve", "double back", "break pattern", "change lanes"]
            turn = turns[(run_number + index) % len(turns)]
            line = f"Off the cuff, {seed} makes me {turn}, then I return to {topic}"
        if rhyme >= 65:
            ending = rhyme_endings[(run_number + index) % len(rhyme_endings)]
            line = f"{line.rsplit(' ', 1)[0]} {ending}"
        forbidden_terms = [x.strip().lower() for x in forbidden.split(",") if x.strip()]
        if any(term in line.lower() for term in forbidden_terms):
            line = "I keep the message clear and move the chosen topic forward"
        plan = cadence_plan(line, bpm)
        bar.text = line
        bar.target_syllables = plan["target_syllables"]
        bar.stress_pattern = plan["stress_pattern"]
        bar.warnings = plan["warnings"]
        bar.delivery_note = (
            f"Mixer cadence {mixer['voice']:.0f}/100; land on beat 4; {plan['pause']}"
        )
    frame = pd.DataFrame([x.model_dump() for x in bars])
    backing = instrumental
    if instrumental_provider.startswith(("MRT2", "Magenta")) and enable_magenta:
        magenta = MagentaRealtimeProvider()
        try:
            engine = "Jam" if "Jam" in instrumental_provider else "Collider"
            backing = magenta.continue_audio(bpm, int(bar_count), engine=engine)
            notice = f"MRT2 {engine} live continuation captured and aligned."
        except Exception as exc:
            backing = generated_beat(bpm, int(bar_count))
            notice = f"{exc} Using local generated-beat fallback."
    elif instrumental_provider == "Metronome / click track":
        backing = metronome(bpm, int(bar_count))
        notice = "Generated local metronome development track."
    elif instrumental_provider == "Generated beat (local)":
        backing = generated_beat(bpm, int(bar_count))
        notice = "Generated a local drum, bass, and hat backing track."
    elif analysis.get("available"):
        notice = "Uploaded instrumental analyzed locally."
    else:
        backing = metronome(bpm, int(bar_count))
        notice = "No uploaded instrumental; using metronome fallback."
    notice = f"{notice} · lyric variation {run_number + 1}"
    return notice, frame, "\n".join(x.text for x in bars), backing


def creative_loop(plan, person, lyrics, voice, music, interaction, culture, understood,
                  felt, changed, replay, effect):
    layers = {"Person": person, "Lyrics": lyrics, "Voice": voice, "Music": music,
              "Interaction": interaction, "Culture": culture}
    weakest = min(layers, key=layers.get)
    measures = {"understood": understood, "felt": felt, "interaction changed": changed,
                "survives replay": replay, "effect achieved": effect}
    weak_measure = min(measures, key=measures.get)
    return (f"PLAN: {plan}\n\nIDENTIFY: weakest causal link is **{weakest} ({layers[weakest]:.0f})**.\n\n"
            f"MEASURE: weakest signal is **{weak_measure} ({measures[weak_measure]:.0f})**.\n\n"
            f"LEARN → PLAN: keep the strongest signal; revise one observable cause in {weakest}, then rerun `{plan}`.")


def youtube_for_battle(url, source, consent, model, current_bpm, request: gr.Request):
    client = None
    try:
        client = gemini_client(source, get_key(request.session_hash, "gemini"), consent)
        result = analyze_youtube(client, url, model, "battle-rap human round")
        text = result.transcript or "\n".join(result.bars)
        if not text.strip():
            raise ValueError("Gemini did not detect intelligible lyrics in this battle round.")
        return text, result.bpm_estimate or current_bpm, (
            f"YouTube round imported with Gemini · confidence {result.confidence:.0%}. "
            "Review the transcript before generating the counter."
        ), "Use supplied lyrics"
    except Exception as exc:
        return "", current_bpm, youtube_error_message(exc), "Use supplied lyrics"
    finally:
        if client is not None:
            client.close()


def youtube_for_freestyle(url, source, consent, model, current_bpm, request: gr.Request):
    client = None
    try:
        client = gemini_client(source, get_key(request.session_hash, "gemini"), consent)
        result = analyze_youtube(client, url, model, "instrumental tempo and structure")
        bpm = result.bpm_estimate or current_bpm
        metadata = {
            "url": url, "bpm": bpm, "time_signature": result.time_signature,
            "summary": result.summary, "confidence": result.confidence,
            "energy": result.energy, "drum_density": result.drum_density,
            "groove": result.groove, "bass_style": result.bass_style,
            "key_estimate": result.key_estimate,
            "instrumentation": result.instrumentation,
        }
        return bpm, result.summary, (
            f"YouTube instrumental analyzed · confidence {result.confidence:.0%}. "
            "Tempo is model-estimated; no waveform was downloaded."
        ), metadata
    except Exception as exc:
        return current_bpm, "", youtube_error_message(exc), None
    finally:
        if client is not None:
            client.close()


def freestyle_with_youtube(
    instrumental, instrumental_provider, enable_magenta, topic, message, seed_words,
    required, forbidden, bpm, bar_count, looseness, coherence, rhyme, youtube_analysis, *values,
):
    provider = instrumental_provider
    if youtube_analysis and instrumental is None and provider == "Uploaded / recorded instrumental":
        provider = "Generated beat (local)"
    notice, frame, text, backing = freestyle(
        instrumental, provider, enable_magenta, topic, message, seed_words, required, forbidden,
        bpm, bar_count, looseness, coherence, rhyme, *values,
    )
    if youtube_analysis:
        confidence = float(youtube_analysis.get("confidence", 0))
        structure = youtube_analysis.get("summary", "")[:240]
        notice = (
            f"YouTube instrumental analysis used for BPM/cadence alignment "
            f"({confidence:.0%} confidence). {notice} Structure: {structure}"
        )
    performance = None
    voice_path = LocalSyntheticVoiceProvider().synthesize_performance(
        text, bpm=bpm, energy=65, aggression=35,
    )
    if voice_path:
        try:
            performance = mix_voice_and_backing(voice_path, backing)
        finally:
            Path(voice_path).unlink(missing_ok=True)
    return notice, frame, text, backing, performance


def freestyle_with_model(
    instrumental, instrumental_provider, enable_magenta, topic, message, seed_words,
    required, forbidden, bpm, bar_count, looseness, coherence, rhyme, youtube_analysis,
    text_provider, credential_source, cloud_consent, openai_model, gemini_model,
    person, words, voice, music, interaction, culture,
    request: gr.Request,
):
    values = (person, words, voice, music, interaction, culture)
    provider = instrumental_provider
    if youtube_analysis and instrumental is None and provider == "Uploaded / recorded instrumental":
        provider = "Generated beat (local)"
    notice, frame, text, backing = freestyle(
        instrumental, provider, enable_magenta, topic, message, seed_words, required, forbidden,
        bpm, bar_count, looseness, coherence, rhyme, *values,
    )
    model_notice = "improved deterministic local"
    client = None
    try:
        if text_provider == "OpenAI":
            client = openai_client(
                credential_source, get_key(request.session_hash, "openai"), cloud_consent
            )
            model = openai_model
        elif text_provider == "Google Gemini":
            client = gemini_client(
                credential_source, get_key(request.session_hash, "gemini"), cloud_consent
            )
            model = gemini_model
        else:
            model = "local templates"
        if client is not None:
            lines = generate_freestyle(
                client, text_provider, model, topic=topic, message=message,
                seed_words=seed_words, required=required, forbidden=forbidden, bpm=bpm,
                bar_count=int(bar_count),
                structure=(youtube_analysis or {}).get("summary", ""),
            )
            for index, line in enumerate(lines):
                plan = cadence_plan(line, bpm)
                frame.at[index, "text"] = line
                frame.at[index, "target_syllables"] = plan["target_syllables"]
                frame.at[index, "stress_pattern"] = plan["stress_pattern"]
                frame.at[index, "warnings"] = plan["warnings"]
            text = "\n".join(lines)
            model_notice = f"{text_provider} · {model}"
    except Exception as exc:
        model_notice = f"{safe_provider_error(exc)}; improved deterministic local fallback"
    finally:
        if client is not None:
            client.close()
    if youtube_analysis:
        confidence = float(youtube_analysis.get("confidence", 0))
        notice = f"YouTube structure/BPM used ({confidence:.0%} confidence). {notice}"
    if provider == "Generated beat (local)":
        profile = youtube_analysis or {
            "energy": 0.35 + looseness / 200,
            "drum_density": 0.25 + rhyme / 180,
            "groove": "swung" if looseness > 55 else "straight",
        }
        backing = generated_beat(
            bpm, int(bar_count), profile=profile,
            seed_text=f"{text}|{topic}|{message}|{seed_words}",
        )
        notice = (
            f"{notice} · beat tuned to energy {float(profile.get('energy', .5)):.0%}, "
            f"density {float(profile.get('drum_density', .5)):.0%}, "
            f"groove {profile.get('groove', 'straight')}"
        )
    notice = f"{notice} · lyrics: {model_notice}"
    performance = None
    voice_path = LocalSyntheticVoiceProvider().synthesize_performance(
        text, bpm=bpm, energy=voice, aggression=looseness,
    )
    if voice_path:
        try:
            performance = mix_voice_and_backing(voice_path, backing)
        finally:
            Path(voice_path).unlink(missing_ok=True)
    return notice, frame, text, backing, performance


def build_battle_content(scorer, contexts, stt_provider, credential_source, cloud_consent,
                         transcription_model, generation_model, gemini_model):
    battle_state = gr.State(None)
    with gr.Column():
        gr.Markdown("## Live Battle Rapper AI\nReliable turn-based local battle and freestyle. Scores are prototype context models—not objective winners.")
        with gr.Tabs():
            with gr.Tab("Battle Response"):
                with gr.Row():
                    start = gr.Button("Start round", variant="primary")
                    stop = gr.Button("■ Immediate stop", variant="stop")
                human_audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Human round / push-to-talk")
                human_text = gr.Textbox(lines=7, label="Complete human transcript / corrections")
                battle_lyrics_source = gr.Radio(
                    ["Use supplied lyrics", "Transcribe recording"],
                    value="Transcribe recording",
                    label="Lyrics source",
                )
                with gr.Row():
                    battle_youtube = gr.Textbox(label="Public YouTube battle-round URL")
                    battle_youtube_button = gr.Button("Import round with Gemini")
                with gr.Row():
                    human_bars = gr.Dropdown([4, 8, 16], value=4, label="Human round bars")
                    response_bars = gr.Dropdown([4, 8, 16], value=4, label="AI response bars")
                    bpm = gr.Number(value=90, label="BPM")
                    agreement = gr.Dropdown(["Friendly", "Competitive", "Comedy", "Practice"], value="Friendly", label="Battle agreement")
                    output_mode = gr.Radio(
                        ["Performed response + separate beat and lyrics", "Text only"],
                        value="Performed response + separate beat and lyrics",
                        label="AI response output",
                        info="Performed mode returns the beat, mixed synthetic vocal, and readable lyrics.",
                    )
                with gr.Row():
                    battle_backing_provider = gr.Dropdown(
                        ["MRT2 Collider (live)", "MRT2 Jam (live)",
                         "Magenta RealTime service", "Uploaded instrumental",
                         "Generated beat (local)", "Metronome / click track", "Vocals only"],
                        value="MRT2 Collider (live)",
                        label="AI response backing provider",
                    )
                    battle_enable_magenta = gr.Checkbox(
                        value=True,
                        label="Generate and mix live MRT2 backing",
                        info="Set the selected MRT2 app Audio Output to ZoomAudioDevice at 48 kHz.",
                    )
                battle_backing_audio = gr.Audio(
                    sources=["upload", "microphone"], type="numpy",
                    label="Optional uploaded AI-response instrumental",
                )
                with gr.Row():
                    intensity = gr.Slider(0, 100, 55, label="Intensity")
                    humor = gr.Slider(0, 100, 45, label="Humor")
                    aggression = gr.Slider(0, 100, 35, label="Aggression ceiling")
                    directness = gr.Slider(0, 100, 70, label="Directness")
                allowed = gr.Textbox(label="Allowed battle facts", placeholder="Comma-separated consensual facts")
                disallowed = gr.Textbox(label="Disallowed topics / personal facts forbidden")
                profanity = gr.Dropdown(["Clean", "Mild", "Unrestricted within safety boundaries"], value="Clean", label="Profanity")
                battle_context = gr.Dropdown(list(contexts), value=["Battle"], multiselect=True, label="Bank B context")
                provider_mode = gr.Radio(
                    ["Deterministic local", "OpenAI"], value="Deterministic local",
                    label="Battle generation provider",
                )
                battle_sliders = []
                with gr.Row():
                    for layer in LAYERS:
                        battle_sliders.append(gr.Slider(0, 100, 55, label=f"AI {layer.title()}"))
                battle_status = gr.Markdown("State: **IDLE**")
                battle_youtube_button.click(
                    youtube_for_battle,
                    [battle_youtube, credential_source, cloud_consent, gemini_model, bpm],
                    [human_text, bpm, battle_status, battle_lyrics_source],
                )
                gr.Markdown("### Latest four human bars")
                active_bars = gr.Markdown()
                graph = gr.Dataframe(label="Performed argument graph")
                strategy = gr.Markdown(label="Validated AI strategy")
                generated = gr.Dataframe(label="Generated bars, addressed human bar IDs, cadence and warnings", interactive=True, wrap=True)
                battle_generated_backing = gr.Audio(label="Generated/selected battle beat — play this to perform yourself")
                ai_audio = gr.Audio(label="Synthetic battle response mixed with the beat")
                performer_lyrics = gr.Markdown(
                    "_Generate a response to see performer lyrics._",
                    label="Lyrics to perform over the beat",
                )
                comparison = gr.Plot(label="Human versus AI outputs with uncertainty")
                latency = gr.Plot(label="Latency trace")
                history = gr.Dataframe(label="Turn timeline")
                inputs = [battle_state, human_text, human_audio, bpm, human_bars, response_bars,
                          agreement, intensity, humor, aggression, directness, allowed, disallowed,
                          profanity, output_mode, battle_context, battle_lyrics_source,
                          stt_provider, provider_mode, battle_backing_provider,
                          battle_backing_audio, battle_enable_magenta, credential_source,
                          cloud_consent, transcription_model, generation_model, *battle_sliders]
                outputs = [battle_state, battle_status, human_text, graph, strategy, generated,
                           battle_generated_backing, ai_audio, performer_lyrics, comparison,
                           latency, history, active_bars]

                def battle_callback(
                    session, transcript, audio, selected_bpm, human_count, response_count,
                    selected_agreement, selected_intensity, selected_humor, selected_aggression,
                    selected_directness, selected_allowed, selected_disallowed,
                    selected_profanity, selected_output_mode, context_names, selected_lyrics_source,
                    selected_stt_provider, provider, selected_backing_provider,
                    selected_backing_audio, selected_enable_magenta,
                    selected_source, selected_consent, selected_transcription_model,
                    selected_generation_model, person, words, voice, music, interaction, culture,
                    request: gr.Request,
                ):
                    return perform_round(
                        scorer, contexts, session, transcript, audio, selected_bpm, human_count,
                        response_count, selected_agreement, selected_intensity, selected_humor,
                        selected_aggression, selected_directness, selected_allowed,
                        selected_disallowed, selected_profanity, selected_output_mode,
                        context_names, selected_lyrics_source, selected_stt_provider, provider,
                        selected_backing_provider, selected_backing_audio, selected_enable_magenta,
                        selected_source, selected_consent,
                        selected_transcription_model, selected_generation_model,
                        person, words, voice, music, interaction, culture, request=request,
                    )

                start.click(lambda s: (s or BattleSession(), "State: **LISTENING** — record or enter the round, then Generate response."), battle_state, [battle_state, battle_status])
                generate = gr.Button("Stop human turn → Generate responsive counter", variant="primary")
                generation_event = generate.click(
                    battle_callback, inputs, outputs
                )
                stop.click(
                    stop_battle, battle_state, [battle_state, battle_status],
                    cancels=[generation_event],
                )

            with gr.Tab("Instrumental Freestyle"):
                youtube_instrumental_state = gr.State(None)
                instrumental = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Instrumental")
                with gr.Row():
                    instrumental_provider = gr.Dropdown(
                        ["Uploaded / recorded instrumental", "Metronome / click track",
                         "Generated beat (local)",
                         "MRT2 Collider (live)", "MRT2 Jam (live)",
                         "Magenta RealTime service"],
                        value="Uploaded / recorded instrumental",
                        label="Instrumental provider",
                    )
                    enable_magenta = gr.Checkbox(
                        label="Enable Magenta / MRT2 provider",
                        info="Local MRT2 capture uses ZoomAudioDevice at 48 kHz.",
                    )
                with gr.Row():
                    freestyle_youtube = gr.Textbox(label="Public YouTube instrumental URL")
                    freestyle_youtube_button = gr.Button("Analyze YouTube instrumental with Gemini")
                instrumental_summary = gr.Textbox(
                    lines=4, label="Gemini instrumental structure",
                    interactive=False,
                    placeholder="Tempo and arrangement analysis appears here; it is not used as lyrics.",
                )
                with gr.Row():
                    topic = gr.Textbox(value="building something under pressure", label="Topic")
                    message = gr.Textbox(value="turn constraint into momentum", label="Message / intended effect")
                    seed = gr.Textbox(label="Seed words")
                    required = gr.Textbox(label="Required references")
                    forbidden = gr.Textbox(label="Forbidden words")
                with gr.Row():
                    free_bpm = gr.Number(value=90, label="BPM")
                    free_count = gr.Dropdown([4, 8, 16], value=4, label="Bars")
                    looseness = gr.Slider(0, 100, 50, label="Freestyle looseness")
                    coherence = gr.Slider(0, 100, 70, label="Coherence")
                    rhyme = gr.Slider(0, 100, 60, label="Rhyme density")
                freestyle_text_provider = gr.Dropdown(
                    ["OpenAI", "Google Gemini", "Deterministic local"],
                    value="OpenAI",
                    label="Freestyle lyric model",
                )
                free_sliders = [gr.Slider(0, 100, 55, label=x.title()) for x in LAYERS]
                free_run = gr.Button("Analyze instrumental and generate aligned freestyle")
                free_status = gr.Markdown()
                freestyle_youtube_button.click(
                    youtube_for_freestyle,
                    [freestyle_youtube, credential_source, cloud_consent, gemini_model, free_bpm],
                    [free_bpm, instrumental_summary, free_status, youtube_instrumental_state],
                )
                free_bars = gr.Dataframe(label="Editable bar/cadence plan", interactive=True, wrap=True)
                free_text = gr.Textbox(lines=8, label="Text-only performance")
                generated_backing = gr.Audio(label="Generated/selected backing track")
                freestyle_performance = gr.Audio(
                    label="Synthetic freestyle voice mixed with aligned backing"
                )
                free_run.click(
                    freestyle_with_model,
                    [instrumental, instrumental_provider, enable_magenta, topic, message, seed,
                     required, forbidden, free_bpm, free_count, looseness, coherence, rhyme,
                     youtube_instrumental_state, freestyle_text_provider, credential_source,
                     cloud_consent, generation_model, gemini_model,
                     *free_sliders],
                    [free_status, free_bars, free_text, generated_backing,
                     freestyle_performance],
                )

            with gr.Tab("Creative Loop"):
                plan = gr.Dropdown(["Challenge or defeat", "Belief or truth", "Feeling or intimacy",
                                    "Motion or energy", "Reflection or insight", "Identity or status"],
                                   value="Challenge or defeat", label="PLAN — what should change?")
                loop_layers = [gr.Slider(0, 100, 55, label=f"IDENTIFY — {x}") for x in ["Person", "Lyrics", "Voice", "Music", "Interaction", "Culture"]]
                measures = [gr.Slider(0, 100, 50, label=f"MEASURE — {x}") for x in ["Understood", "Felt", "Changed interaction", "Survives replay", "Achieved effect"]]
                loop_run = gr.Button("LEARN and feed revision back to PLAN")
                loop_result = gr.Markdown()
                loop_run.click(creative_loop, [plan, *loop_layers, *measures], loop_result)

            with gr.Tab("Mixer Dashboard"):
                gr.Markdown("The same six Bank A controls and configured Bank B context weights drive battle analysis, generation candidate scoring, and inverse revision. Adjust them in Battle Response or Instrumental Freestyle before generation.")
                gr.JSON(value={"Bank A": LAYERS, "Bank B": list(contexts), "scoring_config": scorer.config["version"]})

            with gr.Tab("Battle Analysis"):
                gr.Markdown("Argument nodes, strategy moves, addressed human bar IDs, cadence warnings, score comparison, uncertainty, and latency update in Battle Response. The rubric is a configurable prototype.")

            with gr.Tab("Session History"):
                session_history = gr.Dataframe(label="Current in-memory session")
                refresh = gr.Button("Refresh history")
                refresh.click(_history, battle_state, session_history)
                export = gr.File(label="Explicit JSON export")
                gr.Button("Export session JSON").click(export_history, battle_state, export)

            with gr.Tab("Model and Voice Setup"):
                gr.Markdown("**Battle model:** deterministic local responsive baseline. Cloud adapters reuse the global credential panel when enabled.\n\n**Voice:** macOS synthetic `say` adapter or text-only fallback. It is clearly AI-generated and does not imitate an artist.\n\n**Transport:** Gradio turn-based audio is active. FastRTC full duplex is optional roadmap work.\n\n**Magenta RealTime:** capability-detected optional provider; uploaded audio/metronome fallback remains available.")
                gr.JSON(value={"local_voice": True, "magenta": MagentaRealtimeProvider().capabilities(),
                               "fastrtc": False, "text_fallback": True})
    clearable = [
        human_audio, battle_youtube, human_text, battle_lyrics_source, battle_status, active_bars,
        battle_backing_audio,
        graph, strategy, generated,
        battle_generated_backing, ai_audio, performer_lyrics, comparison, latency, history,
        instrumental, free_status, free_bars,
        freestyle_youtube, instrumental_summary, freestyle_text_provider, free_text, generated_backing,
        freestyle_performance, loop_result, session_history, export,
    ]
    return battle_state, clearable
