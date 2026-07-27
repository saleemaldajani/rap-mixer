from __future__ import annotations

import json
import os

import httpx
from pydantic import BaseModel, Field


class SemanticScores(BaseModel):
    person: float = Field(ge=0, le=100)
    words: float = Field(ge=0, le=100)
    interaction: float = Field(ge=0, le=100)
    culture: float = Field(ge=0, le=100)


def _prompt(text: str) -> str:
    return f"""Assess this rap transcript as evidence, not artistic truth. Return JSON scores from 0-100
for person (specific identity/stakes), words (clarity/craft), interaction (direct response/audience
address), and culture (context-legible references). Do not score voice or music from text.

Transcript:
{text[:12000]}"""


def analyze_semantics(provider: str, text: str, *, openai=None, gemini=None,
                      anthropic_key: str | None = None) -> tuple[dict[str, float], str]:
    prompt = _prompt(text)
    if provider == "OpenAI":
        model = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-5.4-mini")
        response = openai.responses.create(model=model, input=prompt)
        scores = SemanticScores.model_validate_json(response.output_text)
        return scores.model_dump(), f"OpenAI · {model}"
    if provider == "Google Gemini":
        model = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-3.5-flash")
        response = gemini.interactions.create(
            model=model, input=prompt,
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": SemanticScores.model_json_schema()},
            timeout=90,
        )
        scores = SemanticScores.model_validate_json(response.output_text)
        return scores.model_dump(), f"Gemini · {model}"
    if provider == "Ollama":
        model = os.getenv("OLLAMA_SEMANTIC_MODEL", "llama3.2")
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        response = httpx.post(
            f"{base}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=90,
        )
        response.raise_for_status()
        scores = SemanticScores.model_validate_json(response.json()["response"])
        return scores.model_dump(), f"Ollama · {model}"
    if provider == "Anthropic Claude":
        if not anthropic_key:
            raise ValueError("Enter and hold your Anthropic API key for this session.")
        model = os.getenv("DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-5")
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 500,
                  "messages": [{"role": "user", "content": prompt + " Return JSON only."}]},
            timeout=90,
        )
        response.raise_for_status()
        raw = response.json()["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        scores = SemanticScores.model_validate(json.loads(raw))
        return scores.model_dump(), f"Anthropic · {model}"
    raise ValueError(f"Unsupported semantic provider: {provider}")
