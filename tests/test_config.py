"""
Tests for configuration base models.
"""

import pytest
from pydantic import SecretStr, ValidationError

from grounding.config import (
    GroundingSettings,
    LocalProviderSettings,
    RemoteProviderSettings,
)

# ---------------------------------------------------------------------------
# GroundingSettings
# ---------------------------------------------------------------------------


def test_grounding_settings_defaults():
    settings = GroundingSettings()

    assert settings.default_provider is None
    assert settings.request_timeout == 30.0
    assert settings.connect_timeout == 10.0
    assert settings.provider_max_retries == 2
    assert settings.user_agent == "grounding-framework/1.0"


def test_grounding_settings_custom_values():
    settings = GroundingSettings(
        default_provider="vlmrun",
        request_timeout=15.0,
        connect_timeout=5.0,
        provider_max_retries=4,
        user_agent="pytest-agent",
    )

    assert settings.default_provider == "vlmrun"
    assert settings.request_timeout == 15.0
    assert settings.connect_timeout == 5.0
    assert settings.provider_max_retries == 4
    assert settings.user_agent == "pytest-agent"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_timeout", 0),
        ("request_timeout", -1),
        ("connect_timeout", 0),
        ("connect_timeout", -10),
    ],
)
def test_timeout_validation(
    field: str,
    value: float,
):
    with pytest.raises(ValidationError):
        GroundingSettings(**{field: value})


@pytest.mark.parametrize(
    "value",
    [-1, -5],
)
def test_provider_max_retries_validation(
    value: int,
):
    with pytest.raises(ValidationError):
        GroundingSettings(
            provider_max_retries=value,
        )


def test_grounding_settings_are_frozen():
    settings = GroundingSettings()

    with pytest.raises(ValidationError):
        settings.request_timeout = 5.0


def test_grounding_settings_environment_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "GROUNDING_DEFAULT_PROVIDER",
        "vlmrun",
    )
    monkeypatch.setenv(
        "GROUNDING_REQUEST_TIMEOUT",
        "12",
    )
    monkeypatch.setenv(
        "GROUNDING_PROVIDER_MAX_RETRIES",
        "5",
    )

    settings = GroundingSettings()

    assert settings.default_provider == "vlmrun"
    assert settings.request_timeout == 12.0
    assert settings.provider_max_retries == 5


def test_grounding_settings_model_dump():
    settings = GroundingSettings(
        default_provider="fake",
        user_agent="pytest-agent",
        provider_max_retries=3,
        request_timeout=5.0,
        connect_timeout=5.0,
    )

    dumped = settings.model_dump()

    assert dumped["default_provider"] == "fake"
    assert dumped["user_agent"] == "pytest-agent"
    assert dumped["provider_max_retries"] == 3
    assert dumped["request_timeout"] == 5.0
    assert dumped["connect_timeout"] == 5.0


def test_grounding_settings_model_copy():
    settings = GroundingSettings(
        request_timeout=15.0,
    )

    copied = settings.model_copy()

    assert copied == settings
    assert copied is not settings


def test_grounding_settings_equality():
    first = GroundingSettings(
        request_timeout=20.0,
    )
    second = GroundingSettings(
        request_timeout=20.0,
    )

    assert first == second


def test_grounding_settings_inequality():
    first = GroundingSettings(
        request_timeout=20.0,
    )
    second = GroundingSettings(
        request_timeout=30.0,
    )

    assert first != second


# ---------------------------------------------------------------------------
# RemoteProviderSettings
# ---------------------------------------------------------------------------


def test_remote_provider_defaults():
    settings = RemoteProviderSettings(
        base_url="https://example.com",
    )

    assert settings.base_url == "https://example.com"
    assert settings.api_key is None
    assert settings.verify_ssl is True
    assert settings.transport_max_retries == 2
    assert settings.extra_headers == {}


def test_remote_provider_custom_values():
    settings = RemoteProviderSettings(
        api_key=SecretStr("secret"),
        base_url="https://example.com",
        verify_ssl=False,
        transport_max_retries=5,
        extra_headers={
            "Authorization": "Bearer token",
        },
    )

    assert isinstance(settings.api_key, SecretStr)
    assert settings.api_key.get_secret_value() == "secret"
    assert settings.base_url == "https://example.com"
    assert settings.verify_ssl is False
    assert settings.transport_max_retries == 5
    assert settings.extra_headers == {
        "Authorization": "Bearer token",
    }


@pytest.mark.parametrize(
    "value",
    [-1, -2],
)
def test_transport_retry_validation(
    value: int,
):
    with pytest.raises(ValidationError):
        RemoteProviderSettings(
            base_url="https://example.com",
            transport_max_retries=value,
        )


# ---------------------------------------------------------------------------
# LocalProviderSettings
# ---------------------------------------------------------------------------


def test_local_provider_defaults():
    settings = LocalProviderSettings()

    assert settings.device == "auto"
    assert settings.cache_dir is None
    assert settings.num_threads is None
    assert settings.model_config_overrides == {}


def test_local_provider_custom_values():
    settings = LocalProviderSettings(
        device="cuda",
        cache_dir="/tmp/models",
        num_threads=8,
        model_config_overrides={
            "dtype": "float16",
        },
    )

    assert settings.device == "cuda"
    assert settings.cache_dir == "/tmp/models"
    assert settings.num_threads == 8
    assert settings.model_config_overrides == {
        "dtype": "float16",
    }
