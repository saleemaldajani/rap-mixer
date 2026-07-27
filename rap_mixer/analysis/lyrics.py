import re

VOWELS = re.compile(r"[aeiouy]+", re.I)


def syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    n = len(VOWELS.findall(word))
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def lyric_features(text: str) -> dict[str, float]:
    words = re.findall(r"[\w']+", text.lower())
    unique = len(set(words)) / max(1, len(words))
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    endings = [re.findall(r"[a-z]+", x.lower())[-1][-3:] for x in lines if re.findall(r"[a-z]+", x.lower())]
    rhyme = 0 if len(endings) < 2 else 100 * (1 - len(set(endings)) / len(endings))
    concrete = sum(w in {"street", "room", "door", "hand", "face", "night", "light", "city", "train"} for w in words)
    return {
        "semantic_clarity": min(100, 45 + 55 * unique),
        "rhyme_density": min(100, 25 + rhyme),
        "lexical_novelty": 100 * unique,
        "image_density": min(100, concrete * 12),
        "narrative_coherence": min(100, 45 + 5 * len(lines)),
        "repetition": max(0, 100 * (1 - unique)),
    }
