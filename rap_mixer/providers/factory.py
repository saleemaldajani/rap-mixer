from pydantic import SecretStr

from rap_mixer.providers.credentials import CredentialSource, Settings, resolve_key
from rap_mixer.providers.deterministic import DeterministicSemanticAnalyzer
from rap_mixer.security.consent import require_cloud_consent


def create_provider(provider_name: str, credential_source: CredentialSource,
                    session_key: SecretStr | None, settings: Settings, consent: bool = False):
    if provider_name in {"deterministic", "ollama"}:
        return DeterministicSemanticAnalyzer()
    require_cloud_consent(provider_name, consent)
    key = resolve_key(provider_name, credential_source, session_key, settings)
    # Import adapters lazily; keys never enter Gradio state or provider errors.
    from rap_mixer.providers.remote import OpenAICompatibleSemanticAnalyzer
    return OpenAICompatibleSemanticAnalyzer(provider_name, key)

