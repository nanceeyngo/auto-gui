"""
Configuration models for the grounding framework.
"""

from dotenv import find_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GroundingSettings(BaseSettings):
    """
    Shared configuration for grounding providers.

    Values may be supplied through environment variables,
    a .env file, or directly when constructing the settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="GROUNDING_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    default_provider: str | None = None

    request_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Default request timeout in seconds.",
    )

    connect_timeout: float = Field(
        default=10.0,
        gt=0,
        description="Connection timeout in seconds.",
    )

    provider_max_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum number of retry attempts due to framework-related "
            "failures (fallback logic, provider unavailable, parsing "
            "failures, rate limits, etc.)."
        ),
    )

    user_agent: str = Field(
        default="grounding-framework/1.0",
        description="Default HTTP User-Agent.",
    )


class RemoteProviderSettings(GroundingSettings):
    """
    Base configuration for HTTP-based providers.
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    api_key: SecretStr | None = None
    base_url: str
    verify_ssl: bool = True
    transport_max_retries: int = Field(
        default=2,
        ge=0,
        description=(
            "Maximum number of retry attempts due to transport-related"
            "failures (connection reset, transient network failures, "
            "HTTP 502/503/504)"
        ),
    )
    extra_headers: dict[str, str] = Field(default_factory=dict)


class LocalProviderSettings(GroundingSettings):
    """
    Base configuration for local model providers.
    """

    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    device: str = "auto"

    cache_dir: str | None = None

    num_threads: int | None = None

    model_config_overrides: dict[str, object] = Field(default_factory=dict)


__all__ = [
    "GroundingSettings",
    "LocalProviderSettings",
    "RemoteProviderSettings",
]
