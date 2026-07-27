from __future__ import annotations

import time

from rap_mixer.analysis.lyrics import lyric_features
from rap_mixer.audio.bars import segment_text
from rap_mixer.battle.argument_graph import build_argument_graph, graph_rows
from rap_mixer.battle.safety import validate_boundaries
from rap_mixer.battle.state_machine import BattleSession
from rap_mixer.battle.strategy import plan_strategy
from rap_mixer.battle.verse_plan import generate_bars
from rap_mixer.performance.voice import LocalSyntheticVoiceProvider, TextOnlyVoiceProvider
from rap_mixer.schemas.battle import BattlePhase, BattleTurn


def run_battle_round(session: BattleSession, transcript: str, audio, bpm: float, human_bars: int,
                     response_bars: int, tone: str, allowed: str, disallowed: str,
                     output_mode: str, features: dict[str, float], context: dict[str, float], scorer):
    session = session or BattleSession()
    generation = session.start()
    latency = {}
    started = time.perf_counter()
    session.advance(BattlePhase.FINALIZING, generation)
    if not transcript.strip():
        transcript = "The human round has audio but no corrected transcript yet." if audio else ""
    if not transcript:
        raise ValueError("Record a human round or enter its transcript before generating a response.")
    session.advance(BattlePhase.TRANSCRIBING, generation)
    latency["turn_finalization"] = (time.perf_counter() - started) * 1000
    t = time.perf_counter()
    duration = len(audio[1]) / audio[0] if audio else human_bars * 4 * 60 / bpm
    bars = segment_text(transcript, duration, bpm)[:human_bars]
    texts = [x.transcript for x in bars]
    nodes, edges = build_argument_graph(texts)
    latency["argument_analysis"] = (time.perf_counter() - t) * 1000
    safe, safety_reason = validate_boundaries(transcript, disallowed)
    session.advance(BattlePhase.PLANNING, generation)
    t = time.perf_counter()
    strategy = plan_strategy(nodes, response_bars, tone, allowed, disallowed, bpm)
    latency["strategy"] = (time.perf_counter() - t) * 1000
    session.advance(BattlePhase.WRITING, generation)
    t = time.perf_counter()
    generated = generate_bars(strategy, nodes, bpm, disallowed, variation=session.generation)
    latency["verse_generation"] = (time.perf_counter() - t) * 1000
    session.advance(BattlePhase.ALIGNING, generation)
    verse = "\n".join(x.text for x in generated)
    lyric = lyric_features(verse)
    candidate = dict(features)
    candidate["words"] = sum(lyric.values()) / len(lyric)
    candidate["interaction"] = min(100, 60 + 5 * len({x.addressed_human_bar_ids[0] for x in generated}))
    ai_score = scorer.score(candidate, context, 0.72)
    human_score = scorer.score(features, context, 0.65)
    session.advance(BattlePhase.RENDERING, generation)
    t = time.perf_counter()
    # The integrated UI owns rhythmic rendering/mixing; retain legacy direct-engine behavior only.
    voice = LocalSyntheticVoiceProvider() if output_mode == "Performed response" else TextOnlyVoiceProvider()
    audio_path = voice.synthesize_verse(verse)
    latency["voice_rendering"] = (time.perf_counter() - t) * 1000
    session.advance(BattlePhase.SCORING, generation)
    human_turn = BattleTurn(turn_id=len(session.turns) + 1, speaker="Human", transcript=transcript,
                            bar_texts=texts, claims=[x.text for x in nodes],
                            score_bundle=human_score.model_dump(), safety_result=safety_reason,
                            latency_ms=latency)
    ai_turn = BattleTurn(turn_id=len(session.turns) + 2, speaker="AI", transcript=verse,
                         bar_texts=[x.text for x in generated], strategy=strategy,
                         generated_bars=generated, score_bundle=ai_score.model_dump(),
                         safety_result="passed" if safe else f"bounded: {safety_reason}", latency_ms=latency)
    session.turns.extend([human_turn, ai_turn])
    session.advance(BattlePhase.WAITING, generation)
    latency["total"] = (time.perf_counter() - started) * 1000
    return session, bars, graph_rows(nodes, edges), strategy, generated, human_score, ai_score, audio_path, latency
