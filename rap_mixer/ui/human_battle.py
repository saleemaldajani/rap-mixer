from __future__ import annotations

import copy

import gradio as gr
import pandas as pd

from rap_mixer.analysis.feature_estimation import estimate_features
from rap_mixer.providers.openai_audio import openai_client
from rap_mixer.security.redaction import safe_provider_error
from rap_mixer.transcription.router import transcribe_selected
from rap_mixer.ui.credentials import get_key

LAYERS = ["person", "words", "voice", "music", "interaction", "culture"]


def _empty_match(round_number: int = 1, history: list | None = None) -> dict:
    return {
        "round": round_number, "performer": 1, "phase": "recording",
        "names": ["Person 1", "Person 2"], "players": {"1": {}, "2": {}},
        "history": list(history or []),
    }


def _banner(state: dict) -> str:
    if state["phase"] == "results":
        return (
            "<div class='hvh-banner results'><b>ROUND COMPLETE</b>"
            "<span>Both turns were transcribed and analyzed. Review the decision below.</span></div>"
        )
    name = state["names"][state["performer"] - 1]
    steps = "".join(
        f"<div class='hvh-step {'active' if index == state['performer'] - 1 else 'done' if index < state['performer'] - 1 else ''}'>"
        f"<strong>{index + 1}</strong><span>{label}</span></div>"
        for index, label in enumerate(state["names"])
    )
    return (
        f"<div class='hvh-banner'><h2>ROUND {state['round']} · {name} RECORDS NOW</h2>"
        f"<div class='hvh-cycle two'>{steps}</div>"
        "<p>Record the complete turn, stop the microphone, then submit it. Analysis is automatic.</p></div>"
    )


def _score_match(state: dict) -> tuple[pd.DataFrame, str]:
    rows, totals = [], []
    for number, name in enumerate(state["names"], 1):
        player = state["players"][str(number)]
        outputs = player["outputs"]
        modeled = sum(outputs.get(key, 0) for key in (
            "Battle effectiveness", "Potency", "Intelligibility", "Audience fit",
        )) / 4
        totals.append(modeled)
        rows.append({
            "Performer": name,
            "Battle effectiveness": outputs.get("Battle effectiveness", 0),
            "Potency": outputs.get("Potency", 0),
            "Intelligibility": outputs.get("Intelligibility", 0),
            "Audience fit": outputs.get("Audience fit", 0),
            "Round score": modeled, "Evidence confidence": player["confidence"],
        })
    margin = abs(totals[0] - totals[1])
    if margin < 1:
        heading = "### Round result: Draw"
    else:
        heading = f"### Round winner: {state['names'][0 if totals[0] > totals[1] else 1]}"
    explanation = (
        f"{heading}\n\nDecision margin: **{margin:.1f} points**. Both performers were judged "
        "with the same Battle-context model. This is a configurable contextual decision—not "
        "universal artistic worth."
    )
    return pd.DataFrame(rows), explanation


def _analysis_rows(state: dict) -> pd.DataFrame:
    rows = []
    for number, name in enumerate(state["names"], 1):
        player = state["players"][str(number)]
        if not player:
            continue
        features = player["features"]
        strongest = max(features, key=features.get)
        weakest = min(features, key=features.get)
        rows.append({
            "Performer": name, "PLAN": "Challenge / defeat",
            "MESSAGE": player["transcript"],
            "MEASURE": f"battle {player['outputs'].get('Battle effectiveness', 0):.1f} · "
                       f"potency {player['outputs'].get('Potency', 0):.1f}",
            "LEARN — keep": strongest, "LEARN — repair": weakest,
            "Transcription": player["transcription_provider"],
        })
    return pd.DataFrame(rows)


def process_human_turn(state, audio, person_1, person_2, stt_provider, credential_source,
                       cloud_consent, transcription_model, scorer, contexts, request: gr.Request):
    state = copy.deepcopy(state or _empty_match())
    state["names"] = [(person_1 or "Person 1").strip(), (person_2 or "Person 2").strip()]
    if state["phase"] == "results":
        state = _empty_match(state["round"] + 1, state["history"])
        state["names"] = [(person_1 or "Person 1").strip(), (person_2 or "Person 2").strip()]
        return (
            state, _banner(state), "Next round ready. Record Person 1.", None, "", "",
            pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame(state["history"]),
            "Finish Person 1 turn → Transcribe and analyze",
        )
    if audio is None:
        return (
            state, _banner(state), "Record the complete turn before submitting.", gr.skip(),
            gr.skip(), gr.skip(), _analysis_rows(state), pd.DataFrame(), "",
            pd.DataFrame(state["history"]), gr.skip(),
        )
    if stt_provider == "OpenAI transcription API" and not cloud_consent:
        return (
            state, _banner(state),
            "Enable **I consent to send this battle recording to OpenAI**, then submit again.",
            gr.skip(), gr.skip(), gr.skip(), _analysis_rows(state), pd.DataFrame(), "",
            pd.DataFrame(state["history"]), gr.skip(),
        )
    client = None
    try:
        if stt_provider == "OpenAI transcription API":
            client = openai_client(
                credential_source, get_key(request.session_hash, "openai"), cloud_consent
            )
        transcript, provenance = transcribe_selected(
            stt_provider, audio, client, transcription_model,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "consent" in message:
            error = "Enable cloud consent for this battle recording."
        elif "credential" in message or "api key" in message:
            error = "OpenAI credential unavailable. Hold and test your own key above."
        elif "no speech" in message:
            error = "No speech was detected. Record closer to the microphone and try again."
        else:
            error = safe_provider_error(exc)
        return (
            state, _banner(state), f"{error} while transcribing this turn.",
            gr.skip(), gr.skip(), gr.skip(), _analysis_rows(state), pd.DataFrame(), "",
            pd.DataFrame(state["history"]), gr.skip(),
        )
    finally:
        if client is not None:
            client.close()
    try:
        features, confidence, warning = estimate_features(
            (55, 55, 55, 55, 55, 55), transcript, audio, "Auto-estimate from performance"
        )
        context = dict(contexts["Battle"])
        context["familiarity"] = .5
        bundle = scorer.score(features, context, confidence)
    except Exception as exc:
        return (
            state, _banner(state), f"{safe_provider_error(exc)} while analyzing this turn.",
            gr.skip(), gr.skip(), gr.skip(), _analysis_rows(state), pd.DataFrame(), "",
            pd.DataFrame(state["history"]), gr.skip(),
        )
    player_number = state["performer"]
    state["players"][str(player_number)] = {
        "transcript": transcript, "features": features, "confidence": confidence,
        "outputs": {item.name: item.score for item in bundle.outputs},
        "transcription_provider": provenance,
    }
    transcripts = [
        state["players"][str(number)].get("transcript", "") for number in (1, 2)
    ]
    if player_number == 1:
        state["performer"] = 2
        status = f"Person 1 analyzed ({confidence:.0%} confidence). {warning} Record Person 2 now."
        score, winner = pd.DataFrame(), ""
        button = "Finish Person 2 turn → Transcribe, analyze, and judge"
    else:
        state["phase"] = "results"
        score, winner = _score_match(state)
        state["history"].extend(score.assign(Round=state["round"]).to_dict("records"))
        status = f"Person 2 analyzed ({confidence:.0%} confidence). {warning} Round complete."
        button = "Start next round"
    return (
        state, _banner(state), status, None, transcripts[0], transcripts[1],
        _analysis_rows(state), score, winner, pd.DataFrame(state["history"]), button,
    )


def build_human_battle_tab(scorer, contexts, stt_provider, credential_source,
                           cloud_consent, transcription_model):
    state = gr.State(_empty_match())
    gr.Markdown(
        "## Automatic Live Human vs Human Battle\n"
        "One shared recorder. Person 1 performs, then Person 2. Each completed turn is "
        "automatically transcribed and analyzed with the same Battle-context rubric."
    )
    with gr.Row():
        person_1 = gr.Textbox(value="Person 1", label="First performer")
        person_2 = gr.Textbox(value="Person 2", label="Second performer")
    banner = gr.HTML(_banner(_empty_match()))
    status = gr.Markdown("Record Person 1's complete turn.")
    human_stt = gr.Dropdown(
        ["OpenAI transcription API", "faster-whisper", "transformers-whisper"],
        value="OpenAI transcription API", label="Battle transcription provider",
        info="OpenAI uses the key/consent panel above; local modes may take longer to load.",
    )
    human_consent = gr.Checkbox(
        label="I consent to send this battle recording to OpenAI for transcription",
        info="Required only for the OpenAI transcription provider. Resets with the session.",
    )
    audio = gr.Audio(
        sources=["microphone", "upload"], type="numpy",
        label="Current performer — record, then stop",
    )
    submit = gr.Button("Finish Person 1 turn → Transcribe and analyze", variant="primary")
    with gr.Row():
        transcript_1 = gr.Textbox(lines=7, label="Person 1 AI transcript", interactive=False)
        transcript_2 = gr.Textbox(lines=7, label="Person 2 AI transcript", interactive=False)
    analysis = gr.Dataframe(
        label="Automatic Plan → Message → Measure → Learn analysis",
        interactive=False, wrap=True,
    )
    winner = gr.Markdown()
    score = gr.Dataframe(label="Head-to-head score breakdown", interactive=False, wrap=True)
    history = gr.Dataframe(label="Match history", interactive=False, wrap=True)
    outputs = [
        state, banner, status, audio, transcript_1, transcript_2, analysis, score,
        winner, history, submit,
    ]
    def submit_turn(current_state, current_audio, first_name, second_name, selected_stt,
                    selected_source, selected_consent, selected_model,
                    request: gr.Request):
        return process_human_turn(
            current_state, current_audio, first_name, second_name, selected_stt,
            selected_source, selected_consent, selected_model, scorer, contexts, request,
        )

    event = submit.click(
        lambda: ("Analyzing the completed turn…", gr.update(interactive=False)),
        None, [status, submit], queue=False,
    ).then(
        submit_turn,
        [state, audio, person_1, person_2, human_stt, credential_source,
         human_consent, transcription_model],
        outputs,
    )
    event.then(lambda: gr.update(interactive=True), None, submit, queue=False)
    return state, [
        audio, human_stt, human_consent, transcript_1, transcript_2, status, analysis, winner,
        score, history,
    ]
