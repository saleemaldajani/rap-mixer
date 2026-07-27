def calculate(features: dict[str, float], context: dict[str, float]) -> dict[str, float]:
    def g(key: str) -> float:
        return features.get(key, 50.0) / 100

    def c(key: str) -> float:
        return context.get(key, 50.0) / 100
    return {
        "words_voice": 2 * min(g("words"), g("voice")) - 1,
        "voice_music": 2 * min(g("voice"), g("music")) - 1,
        "person_culture": 2 * min(g("person"), g("culture")) - 1,
        "words_culture": 2 * min(g("words"), g("culture")) - 1,
        "music_context": 2 * g("music") * c("groove") - 0.5,
        "interaction_context": 2 * g("interaction") * c("response") - 0.5,
        "person_words": 2 * min(g("person"), g("words")) - 1,
        "voice_person": 2 * min(g("voice"), g("person")) - 1,
    }
