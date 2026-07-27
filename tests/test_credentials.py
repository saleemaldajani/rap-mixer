import logging

import pytest
from pydantic import SecretStr

from rap_mixer.providers.credentials import Settings, resolve_key
from rap_mixer.security.redaction import safe_provider_error
from rap_mixer.ui.credentials import clear_keys, get_key, store_key


def test_explicit_credential_resolution(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-environment-value")
    assert resolve_key("openai", "user_supplied", SecretStr("user-secret"), Settings()).get_secret_value() == "user-secret"
    with pytest.raises(ValueError):
        resolve_key("openai", "user_supplied", None, Settings())
    assert resolve_key("openai", "local", None, Settings()) is None
    with pytest.raises(ValueError, match="Unsupported credential source"):
        resolve_key("openai", "server_environment", None, Settings())


def test_session_vault_clear_and_no_log(caplog):
    caplog.set_level(logging.DEBUG)
    store_key("session", "openai", "very-secret")
    assert get_key("session", "openai").get_secret_value() == "very-secret"
    assert "very-secret" not in caplog.text
    clear_keys("session")
    assert get_key("session", "openai") is None
    assert safe_provider_error(RuntimeError("very-secret raw body")) == "Provider unavailable"


def test_openai_and_gemini_keys_coexist_in_one_session():
    store_key("dual", "openai", "openai-test-value")
    store_key("dual", "gemini", "gemini-test-value")
    assert get_key("dual", "openai").get_secret_value() == "openai-test-value"
    assert get_key("dual", "gemini").get_secret_value() == "gemini-test-value"
    clear_keys("dual")
