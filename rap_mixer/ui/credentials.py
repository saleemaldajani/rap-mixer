from __future__ import annotations

from threading import RLock

from pydantic import SecretStr

_VAULT: dict[str, dict[str, SecretStr]] = {}
_LOCK = RLock()


def store_key(session_hash: str, provider: str, value: str) -> str:
    with _LOCK:
        _VAULT.setdefault(session_hash, {})[provider] = SecretStr(value)
    return "Credential held in server-side session memory."


def get_key(session_hash: str, provider: str) -> SecretStr | None:
    return _VAULT.get(session_hash, {}).get(provider)


def clear_keys(session_hash: str) -> str:
    with _LOCK:
        _VAULT.pop(session_hash, None)
    return "Credentials cleared."


def clear_key(session_hash: str, provider: str) -> None:
    with _LOCK:
        providers = _VAULT.get(session_hash, {})
        providers.pop(provider, None)
        if not providers:
            _VAULT.pop(session_hash, None)
