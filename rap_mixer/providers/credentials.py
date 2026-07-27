from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr

CredentialSource = Literal["local", "user_supplied"]


class Settings(BaseModel):
    @classmethod
    def from_env(cls):
        # Kept as a compatibility boundary for provider adapters. Public builds
        # deliberately do not read API credentials from the server environment.
        return cls()


def resolve_key(provider: str, source: CredentialSource, session_key: SecretStr | None, settings: Settings) -> SecretStr | None:
    if source == "local":
        return None
    if source == "user_supplied":
        if session_key is None or not session_key.get_secret_value():
            raise ValueError("Selected user credential is unavailable.")
        return session_key
    raise ValueError("Unsupported credential source. Use a local provider or your own API key.")
