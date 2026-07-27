import pytest

from rap_mixer.security.consent import ConsentRequired, require_cloud_consent


def test_cloud_requires_consent_local_does_not():
    require_cloud_consent("deterministic", False)
    with pytest.raises(ConsentRequired):
        require_cloud_consent("openai", False)

