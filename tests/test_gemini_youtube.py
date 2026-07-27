from types import SimpleNamespace

import pytest

from rap_mixer.providers.gemini_youtube import (
    analyze_youtube,
    validate_youtube_url,
    youtube_error_message,
)


class FakeInteractions:
    def create(self, **kwargs):
        assert kwargs["model"] == "custom-gemini-model"
        assert kwargs["input"][0]["type"] == "text"
        video = kwargs["input"][1]
        assert video == {"type": "video", "uri": "https://www.youtube.com/watch?v=abc123"}
        assert kwargs["response_format"]["mime_type"] == "application/json"
        assert kwargs["generation_config"]["max_output_tokens"] == 16000
        return SimpleNamespace(
            output_text='{"transcript":"first bar\\nsecond bar","bars":["first bar",'
            '"second bar"],"bpm_estimate":94,"time_signature":"4/4",'
            '"summary":"two-bar rap performance","confidence":0.8,"limitations":[]}'
        )


def test_youtube_url_validation_rejects_non_youtube_and_non_video_urls():
    canonical = "https://www.youtube.com/watch?v=abc123"
    assert validate_youtube_url("https://youtu.be/abc123") == canonical
    assert validate_youtube_url("https://www.youtube.com/shorts/abc123") == canonical
    assert validate_youtube_url(
        "https://www.youtube.com/watch?v=abc123&list=RDabc123&start_radio=1"
    ) == canonical
    with pytest.raises(ValueError):
        validate_youtube_url("https://example.com/watch?v=abc123")
    with pytest.raises(ValueError):
        validate_youtube_url("https://www.youtube.com/")


def test_gemini_youtube_analysis_is_structured_and_model_selectable():
    client = SimpleNamespace(interactions=FakeInteractions())
    result = analyze_youtube(
        client, "https://www.youtube.com/watch?v=abc123", "custom-gemini-model"
    )
    assert result.bpm_estimate == 94
    assert result.bars == ["first bar", "second bar"]


def test_youtube_errors_are_specific_without_echoing_provider_details():
    assert "public" in youtube_error_message(Exception("video is private"))
    assert "quota" in youtube_error_message(Exception("429 quota exceeded")).lower()
    assert "shorter" in youtube_error_message(TimeoutError("timed out"))
