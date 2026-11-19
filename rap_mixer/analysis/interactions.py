def calculate(features: dict[str, float], context: dict[str, float]) -> dict[str, float]:
    def g(key: str) -> float:
        return features.get(key, 50.0) / 100

    def c(key: str) -> float:
        # A context weight is a multiplier with 1.0 as neutral (config/contexts.yaml).
        return context.get(key, 1.0)
    return {
        "words_voice": 2 * min(g("words"), g("voice")) - 1,
        "voice_music": 2 * min(g("voice"), g("music")) - 1,
        "person_culture": 2 * min(g("person"), g("culture")) - 1,
        "words_culture": 2 * min(g("words"), g("culture")) - 1,
        "music_context": g("music") * c("groove") - 0.5,
        "interaction_context": g("interaction") * c("response") - 0.5,
        "person_words": 2 * min(g("person"), g("words")) - 1,
        "voice_person": 2 * min(g("voice"), g("person")) - 1,
    }
