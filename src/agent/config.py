"""
Configuration models for the GUI automation agent.
"""

from pathlib import Path

from dotenv import find_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class AgentSettings(BaseSettings):
    """
    Configuration for the GUI automation agent.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    model: str = "qwen/qwen3.6-27b"

    api_key: SecretStr | None = None

    base_url: str | None = None

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the language model.",
    )

    max_iterations: int = Field(
        default=30,
        gt=0,
        description=(
            "Maximum number of reasoning/tool-use iterations "
            "before the agent stops."
        ),
    )

    post_action_delay: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Delay (seconds) after GUI actions before taking "
            "the next screenshot."
        ),
    )

    screenshot_directory: Path = Field(
        default=Path(".screenshots"),
        description="Directory used to store temporary screenshots.",
    )

    keep_screenshots: bool = Field(
        default=False,
        description=(
            "Whether screenshots should be retained after execution completes."
        ),
    )

    request_timeout: float = Field(
        default=120.0,
        gt=0,
        description="Timeout for LLM requests.",
    )

    max_tool_calls: int = Field(
        default=100,
        gt=0,
    )

    seed: int | None = Field(
        default=None,
        description="Random seed for deterministic sampling if supported.",
    )

    verbose: bool = Field(
        default=False,
        description="Enable verbose logging.",
    )


settings = AgentSettings()


__all__ = [
    "AgentSettings",
    "settings",
]
