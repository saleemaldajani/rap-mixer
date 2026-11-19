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
# Offered as an editable starting point; the performers decide which outcomes count.
BATTLE_OUTCOMES = ["Battle effectiveness", "Potency", "Intelligibility", "Audience fit"]
NO_CONTEXT = (
    "Choose the context for this round first. Nothing is preselected: both performers agree "
    "on the game before Person 1 records."
)


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
            "<span>Both turns were transcribed and analyzed. Review the comparison below.</span></div>"
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


def _blend(contexts: dict, names: list[str]) -> dict[str, float]:
    keys = next(iter(contexts.values()))
    blended = {key: sum(contexts[name][key] for name in names) / len(names) for key in keys}
    blended["familiarity"] = .5
    return blended


def _reading(difference: float, combined_band: float, labels: list[str]) -> str:
    if abs(difference) <= combined_band:
        return "inside the bands"
    return f"{labels[0 if difference > 0 else 1]} ahead"


def _score_match(state: dict) -> tuple[pd.DataFrame, str, list[dict]]:
    """Compare the two turns outcome by outcome, each with its band.

    A winner is named only when the round-score margin exceeds both bands added together, that is,
    when the two displayed bands do not overlap. Otherwise the app reports no decision.
    """
    names = state["names"]
    labels = names if names[0] != names[1] else [f"{names[0]} (1)", f"{names[1]} (2)"]
    outcomes = state.get("outcomes") or BATTLE_OUTCOMES
    players = [state["players"][str(number)] for number in (1, 2)]

    def value(player, key):
        return float(player["outputs"].get(key, 0))

    def band(player, key):
        return float(player.get("bands", {}).get(key, 0))

    rows = []
    for key in outcomes:
        difference = value(players[0], key) - value(players[1], key)
        rows.append({
            "Outcome": key,
            labels[0]: f"{value(players[0], key):.1f} ± {band(players[0], key):.1f}",
            labels[1]: f"{value(players[1], key):.1f} ± {band(players[1], key):.1f}",
            "Difference": round(abs(difference), 1),
            "Reading": _reading(difference, band(players[0], key) + band(players[1], key), labels),
        })
    totals = [sum(value(player, key) for key in outcomes) / len(outcomes) for player in players]
    bands = [sum(band(player, key) for key in outcomes) / len(outcomes) for player in players]
    margin = abs(totals[0] - totals[1])
    combined = bands[0] + bands[1]
    decided = margin > combined
    winner = labels[0 if totals[0] > totals[1] else 1] if decided else None
    rows.append({
        "Outcome": "Round score (mean)",
        labels[0]: f"{totals[0]:.1f} ± {bands[0]:.1f}",
        labels[1]: f"{totals[1]:.1f} ± {bands[1]:.1f}",
        "Difference": round(margin, 1),
        "Reading": _reading(totals[0] - totals[1], combined, labels),
    })
    context_name = " + ".join(state.get("context_names") or ["Battle"])
    if decided:
        heading = f"### Round winner: {winner}"
        reading = "The margin exceeds both bands, so the two estimates are separated."
    else:
        heading = "### Round result: No decision"
        reading = (
            "The margin is inside the bands, so the estimates cannot separate these two turns "
            "and the app names no winner. The performers and the room decide."
        )
    explanation = (
        f"{heading}\n\nDecision margin: **{margin:.1f} points**. Bands: **±{bands[0]:.1f}** and "
        f"**±{bands[1]:.1f}** (combined {combined:.1f}). {reading} Both turns were scored under "
        f"the context the performers chose ({context_name}), on {len(outcomes)} outcomes. This is "
        "a contextual estimate—not universal artistic worth."
    )
    history = [{
        "Round": state["round"], "Performer": labels[index],
        "Round score": round(totals[index], 1), "Band": round(bands[index], 1),
        "Evidence confidence": round(players[index]["confidence"], 2),
        "Result": "no decision" if not decided else ("winner" if labels[index] == winner else ""),
    } for index in (0, 1)]
    return pd.DataFrame(rows), explanation, history


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
                       cloud_consent, transcription_model, scorer, contexts, request: gr.Request,
                       selected_contexts=None, selected_outcomes=None):
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
    if state["performer"] == 1:
        # The round's context and outcomes are fixed when Person 1 submits, so both turns are
        # scored the same way. Neither is chosen by the app.
        if not selected_contexts:
            return (
                state, _banner(state), NO_CONTEXT, gr.skip(), gr.skip(), gr.skip(),
                _analysis_rows(state), pd.DataFrame(), "", pd.DataFrame(state["history"]),
                gr.skip(),
            )
        state["context_names"] = list(selected_contexts)
        state["outcomes"] = list(selected_outcomes or BATTLE_OUTCOMES)
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
        context = _blend(contexts, state.get("context_names") or ["Battle"])
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
        "bands": {item.name: item.uncertainty for item in bundle.outputs},
        "transcription_provider": provenance,
    }
    transcripts = [
        state["players"][str(number)].get("transcript", "") for number in (1, 2)
    ]
    if player_number == 1:
        state["performer"] = 2
        status = f"Person 1 analyzed ({confidence:.0%} confidence). {warning} Record Person 2 now."
        score, winner = pd.DataFrame(), ""
        button = "Finish Person 2 turn → Transcribe, analyze, and compare"
    else:
        state["phase"] = "results"
        score, winner, history_rows = _score_match(state)
        state["history"].extend(history_rows)
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
        "automatically transcribed and scored under the context that the two performers choose. "
        "A winner is named only when the margin exceeds both displayed bands."
    )
    with gr.Row():
        battle_context = gr.Dropdown(
            list(contexts), value=None, multiselect=True, label="Context for this round",
            info="Required. Nothing is preselected: agree on the game before Person 1 records.",
        )
        battle_outcomes = gr.Dropdown(
            list(scorer.config["outputs"]), value=BATTLE_OUTCOMES, multiselect=True,
            label="Outcomes that count",
            info="An editable starting point. Both turns are compared on these outcomes.",
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
    score = gr.Dataframe(
        label="Head-to-head comparison by outcome, with bands", interactive=False, wrap=True,
    )
    history = gr.Dataframe(label="Match history", interactive=False, wrap=True)
    outputs = [
        state, banner, status, audio, transcript_1, transcript_2, analysis, score,
        winner, history, submit,
    ]
    def submit_turn(current_state, current_audio, first_name, second_name, selected_stt,
                    selected_source, selected_consent, selected_model, chosen_contexts,
                    chosen_outcomes, request: gr.Request):
        return process_human_turn(
            current_state, current_audio, first_name, second_name, selected_stt,
            selected_source, selected_consent, selected_model, scorer, contexts, request,
            selected_contexts=chosen_contexts, selected_outcomes=chosen_outcomes,
        )

    event = submit.click(
        lambda: ("Analyzing the completed turn…", gr.update(interactive=False)),
        None, [status, submit], queue=False,
    ).then(
        submit_turn,
        [state, audio, person_1, person_2, human_stt, credential_source,
         human_consent, transcription_model, battle_context, battle_outcomes],
        outputs,
    )
    event.then(lambda: gr.update(interactive=True), None, submit, queue=False)
    return state, [
        audio, human_stt, human_consent, transcript_1, transcript_2, status, analysis, winner,
        score, history,
    ]
