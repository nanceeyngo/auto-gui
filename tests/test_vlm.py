"""
Tests for `agent.vlm.create_vlm`.
"""

from typing import Any, cast

import pytest
from pydantic import SecretStr

from agent.config import AgentSettings
from agent.vlm import create_vlm


class TestCreateVlm:
    def test_passes_configured_max_tokens_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Regression coverage: `create_vlm()` previously constructed
        `ChatOpenAI` with no `max_tokens` at all. Left unbounded, some
        OpenAI-compatible backends (e.g. OpenRouter) interpret that as
        "reserve the model's full context window" for every request,
        which can fail with a 402 insufficient-credits error even on
        an account with a real, non-zero balance.
        """
        monkeypatch.setattr(
            "agent.vlm.settings",
            AgentSettings(max_tokens=2048, api_key=SecretStr("test-dummy-key")),
        )

        model = create_vlm()

        assert cast(Any, model).max_tokens == 2048

    def test_max_tokens_can_be_disabled_explicitly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "agent.vlm.settings",
            AgentSettings(max_tokens=None, api_key=SecretStr("test-dummy-key")),
        )

        model = create_vlm()

        assert cast(Any, model).max_tokens is None

    def test_default_max_tokens_is_bounded(self) -> None:
        settings = AgentSettings()

        assert settings.max_tokens is not None
        assert settings.max_tokens > 0
