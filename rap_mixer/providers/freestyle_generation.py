from __future__ import annotations

from pydantic import BaseModel


class FreestyleResult(BaseModel):
    bars: list[str]


def generate_freestyle(client, provider: str, model: str, *, topic: str, message: str,
                       seed_words: str, required: str, forbidden: str, bpm: float,
                       bar_count: int, structure: str) -> list[str]:
    prompt = f"""Write exactly {bar_count} performable freestyle bars aligned to {bpm:.0f} BPM.
Topic: {topic}
Intended message: {message}
Optional seed words: {seed_words or 'none'}
Required references: {required or 'none'}
Forbidden words/topics: {forbidden or 'none'}
Instrumental structure: {structure or 'unknown'}

Requirements:
- Develop an arc across the bars; each line must add a new image, claim, or turn.
- Do not repeat the full topic or intended-message phrase more than once.
- Never use a seed word as both a setting and an object in the same line.
- Use natural syntax and meaningful end rhymes; never replace a final word merely to force rhyme.
- Keep each bar concise enough to perform at the stated BPM.
- Return JSON only as {{"bars":["...", "..."]}}."""
    if provider == "OpenAI":
        response = client.responses.create(model=model, input=prompt)
        result = FreestyleResult.model_validate_json(response.output_text)
    else:
        response = client.interactions.create(
            model=model, input=prompt,
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": FreestyleResult.model_json_schema()},
            timeout=90,
        )
        result = FreestyleResult.model_validate_json(response.output_text)
    if len(result.bars) != bar_count:
        raise ValueError(f"Model returned {len(result.bars)} bars; expected {bar_count}.")
    return [line.strip() for line in result.bars if line.strip()]
