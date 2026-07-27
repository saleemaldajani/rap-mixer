def safe_provider_error(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "credit balance" in message or "billing" in message or "purchase credits" in message:
        return "Provider key accepted, but billing credits are unavailable"
    if "auth" in name or "permission" in name:
        return "Authentication failed"
    if "rate" in name:
        return "Rate limited"
    if "model" in name:
        return "Model unavailable"
    return "Provider unavailable"
