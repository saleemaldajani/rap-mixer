def safe_provider_error(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    # Setup mistakes raised by our own guards carry no secrets; surface them as
    # actionable guidance instead of a generic provider failure.
    if "consent" in name:
        return "Consent required — check the cloud data-sharing box in the API keys panel"
    if "credential is unavailable" in message:
        return "No API key is held for this session — paste your key in the API keys panel at the top"
    if "credential source" in message:
        return "Set Credential source to 'Use my own API key' in the API keys panel at the top"
    if "credit balance" in message or "billing" in message or "purchase credits" in message:
        return "Provider key accepted, but billing credits are unavailable"
    if "auth" in name or "permission" in name:
        return "Authentication failed"
    if "rate" in name:
        return "Rate limited"
    if "model" in name:
        return "Model unavailable"
    return "Provider unavailable"
