"""
Tests for VLMRunSettings configuration model.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.types import SecretStr

from grounding.providers.vlmrun import VLMRunSettings


def test_vlmrun_defaults():
    settings = VLMRunSettings(_env_file=None)

    assert settings.base_url == ("https://api.vlm.run/v1/openai")

    assert settings.model == ("vlmrun-orion-2:auto")

    assert settings.api_key is None

    assert settings.temperature == 0.0
    assert settings.max_tokens is None

    assert settings.request_timeout == 30.0
    assert settings.transport_max_retries == 2
    assert settings.verify_ssl is True

    assert settings.system_prompt_filepath.name == (
        "orion_grounding_system_prompt.md"
    )

    assert settings.grounding_schema_filepath.name == (
        "orion_grounding.schema.json"
    )


def test_vlmrun_custom_values():
    settings = VLMRunSettings(
        api_key="secret-key",  # type: ignore[arg-type]
        model="custom-model",
        base_url="https://example.com",
        temperature=0.25,
        max_tokens=1024,
        request_timeout=15.0,
        transport_max_retries=5,
    )

    assert isinstance(settings.api_key, SecretStr)
    assert settings.api_key.get_secret_value() == "secret-key"

    assert settings.model == "custom-model"
    assert settings.base_url == "https://example.com"

    assert settings.temperature == 0.25
    assert settings.max_tokens == 1024

    assert settings.request_timeout == 15.0
    assert settings.transport_max_retries == 5


def test_vlmrun_settings_are_frozen():
    settings = VLMRunSettings()

    with pytest.raises(ValidationError):
        settings.temperature = 0.5


def test_vlmrun_environment_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "VLMRUN_API_KEY",
        "env-secret",
    )
    monkeypatch.setenv(
        "VLMRUN_MODEL",
        "env-model",
    )
    monkeypatch.setenv(
        "VLMRUN_TEMPERATURE",
        "0.7",
    )

    settings = VLMRunSettings()

    assert settings.api_key is not None
    assert isinstance(settings.api_key, SecretStr)
    assert settings.api_key.get_secret_value() == ("env-secret")

    assert settings.model == "env-model"
    assert settings.temperature == 0.7


def test_vlmrun_prompt_paths_are_paths():
    settings = VLMRunSettings()

    assert isinstance(
        settings.system_prompt_filepath,
        Path,
    )
    assert isinstance(
        settings.grounding_schema_filepath,
        Path,
    )

    assert settings.system_prompt_filepath.is_absolute()
    assert settings.grounding_schema_filepath.is_absolute()


def test_vlmrun_secret_not_exposed_in_repr():
    settings = VLMRunSettings(
        api_key="super-secret",  # type: ignore[arg-type]
    )

    representation = repr(settings)

    assert "super-secret" not in representation


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -100,
    ],
)
def test_vlmrun_request_timeout_validation(
    value: float,
):
    with pytest.raises(ValidationError):
        VLMRunSettings(
            request_timeout=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        -5,
    ],
)
def test_vlmrun_transport_retry_validation(
    value: int,
):
    with pytest.raises(ValidationError):
        VLMRunSettings(
            transport_max_retries=value,
        )
