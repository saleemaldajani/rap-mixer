from __future__ import annotations

import re

from rap_mixer.analysis.lyrics import lyric_features
from rap_mixer.audio.features import AudioFeatureExtractor

LAYERS = ["person", "words", "voice", "music", "interaction", "culture"]
EXTRACTOR = AudioFeatureExtractor()


def estimate_features(values, text: str, audio=None, mode="Auto-estimate, then edit"):
    result = dict(zip(LAYERS, values, strict=True))
    if mode == "Manually configure":
        return result, 1.0, "Manual A values used without automatic feature adjustment."
    lf = lyric_features(text or "")
    if text:
        tokens = re.findall(r"[A-Za-z0-9']+", text)
        lower = [x.lower() for x in tokens]
        first_person = sum(x in {"i", "i'm", "my", "me", "mine"} for x in lower)
        second_person = sum(x in {"you", "your", "you're", "yours"} for x in lower)
        concrete = sum(x in {
            "room", "street", "door", "hand", "face", "night", "light", "city",
            "train", "stage", "mic", "snare", "kick", "crowd",
        } for x in lower)
        named_or_numeric = sum(x[:1].isupper() or any(ch.isdigit() for ch in x) for x in tokens)
        repeated = 1 - len(set(lower)) / max(1, len(lower))
        person_estimate = min(100, 30 + first_person * 5 + concrete * 7 + named_or_numeric * 3)
        interaction_estimate = min(100, 25 + second_person * 7 + text.count("?") * 8)
        culture_estimate = min(100, 25 + named_or_numeric * 5 + concrete * 2)
        words_estimate = max(0, min(100,
            0.35 * lf["semantic_clarity"] + 0.3 * lf["lexical_novelty"]
            + 0.2 * lf["narrative_coherence"] + 0.15 * lf["image_density"]
            - repeated * 18
        ))
        weight = 1.0 if mode == "Auto-estimate from performance" else 0.7
        for name, estimate in {
            "person": person_estimate, "words": words_estimate,
            "interaction": interaction_estimate, "culture": culture_estimate,
        }.items():
            result[name] = weight * estimate + (1 - weight) * result[name]
    warning = ""
    confidence = 0.72 if text else 0.5
    if audio:
        measured = EXTRACTOR.analyze_audio(*audio)
        dynamic_score = min(100, measured["dynamic_range"] / 35 * 100)
        clarity_score = 100 * (1 - measured["noise_ratio"])
        clipping_score = 100 * (1 - min(1, measured["clipping"] * 20))
        voice_estimate = 0.4 * dynamic_score + 0.35 * clarity_score + 0.25 * clipping_score
        rhythm_score = min(100, measured["onset_rate"] * 12)
        spectral_score = min(100, measured["spectral_centroid"] / 4000 * 100)
        music_estimate = 0.45 * rhythm_score + 0.3 * dynamic_score + 0.25 * spectral_score
        weight = 1.0 if mode == "Auto-estimate from performance" else 0.7
        result["voice"] = weight * voice_estimate + (1 - weight) * result["voice"]
        result["music"] = weight * music_estimate + (1 - weight) * result["music"]
        warning = measured["warning"]
        confidence -= measured["noise_ratio"] * 0.2
    return result, max(0.25, confidence), warning
