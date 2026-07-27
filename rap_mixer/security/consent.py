class ConsentRequired(PermissionError):
    pass


def require_cloud_consent(provider: str, consent: bool) -> None:
    if provider not in {"deterministic", "faster-whisper", "transformers-whisper", "ollama"} and not consent:
        raise ConsentRequired("Cloud data sharing requires session-specific consent.")

