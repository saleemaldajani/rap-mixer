def cultural_baseline(text: str) -> float:
    references = sum(token.startswith("#") or token[:1].isupper() for token in text.split())
    return max(20, min(80, 40 + references * 3))
