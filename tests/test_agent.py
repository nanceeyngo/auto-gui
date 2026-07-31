"""
Tests for `agent.agent.GUIAutomationAgent`.

Exercises the main run loop -- including the AgentExecutionError
wrapping added to make backend/model failures observable instead of
crashing opaquely -- using a fake `BaseChatModel` so no real model
backend is required.
"""

from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from PIL import Image

from agent.agent import GUIAutomationAgent
from agent.config import settings
from agent.context import AgentServices
from agent.exceptions import AgentExecutionError
from agent.tools.action.actions import ActionManager
from agent.tools.coordinates import IdentityCoordinateMapper
from agent.tools.grounding import GroundingManager
from agent.tools.screenshot.backend import ScreenshotBackend
from agent.tools.screenshot.screenshot import ScreenshotManager
from grounding.client import GroundingClient
from grounding.config import GroundingSettings
from grounding.registry import GroundingRegistry
from tests.fakes import FakeGroundingProvider


class FakeScreenshotBackend(ScreenshotBackend):
    def capture(self) -> Image.Image:
        return Image.new("RGB", (100, 100))

    def capture_region(
        self, *, left: int, top: int, width: int, height: int
    ) -> Image.Image:
        return Image.new("RGB", (width, height))


class RespondImmediatelyChatModel(BaseChatModel):
    """
    A fake chat model that always responds with a plain text answer
    and never calls a tool, so the ReAct loop terminates in one turn.
    """

    reply: str = "Task complete."

    @property
    def _llm_type(self) -> str:
        return "fake-immediate"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RespondImmediatelyChatModel":
        # create_agent always calls bind_tools(); our fake doesn't
        # need real tool-calling support, so just return self.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = AIMessage(content=self.reply)
        return ChatResult(generations=[ChatGeneration(message=message)])


class BoomingChatModel(BaseChatModel):
    """
    A fake chat model that always raises, to exercise
    GUIAutomationAgent's exception-wrapping behavior.
    """

    @property
    def _llm_type(self) -> str:
        return "fake-booming"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "BoomingChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("simulated model backend failure")


def make_services(tmp_path: Path) -> AgentServices:
    registry = GroundingRegistry()
    registry.register(FakeGroundingProvider)

    return AgentServices(
        grounding=GroundingManager(
            client=GroundingClient(
                registry=registry,
                settings=GroundingSettings(default_provider="fake"),
            )
        ),
        screenshots=ScreenshotManager(
            directory=tmp_path / "shots",
            backend=FakeScreenshotBackend(),
        ),
        actions=ActionManager(
            post_action_delay=0.0,
            coordinate_mapper=IdentityCoordinateMapper(),
        ),
        config=settings,
    )


class TestGUIAutomationAgentRun:
    def test_run_returns_final_message_content(self, tmp_path: Path) -> None:
        agent = GUIAutomationAgent(
            vlm=RespondImmediatelyChatModel(),
            services=make_services(tmp_path),
        )

        result = agent.run("Click the OK button.")

        assert result == "Task complete."

    def test_run_wraps_backend_failures_in_agent_execution_error(
        self, tmp_path: Path
    ) -> None:
        agent = GUIAutomationAgent(
            vlm=BoomingChatModel(),
            services=make_services(tmp_path),
        )

        with pytest.raises(AgentExecutionError) as exc_info:
            agent.run("Click the OK button.")

        assert exc_info.value.goal == "Click the OK button."
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.asyncio
    async def test_arun_returns_final_message_content(self, tmp_path: Path) -> None:
        agent = GUIAutomationAgent(
            vlm=RespondImmediatelyChatModel(),
            services=make_services(tmp_path),
        )

        result = await agent.arun("Click the OK button.")

        assert result == "Task complete."

    @pytest.mark.asyncio
    async def test_arun_wraps_backend_failures(self, tmp_path: Path) -> None:
        agent = GUIAutomationAgent(
            vlm=BoomingChatModel(),
            services=make_services(tmp_path),
        )

        with pytest.raises(AgentExecutionError):
            await agent.arun("Click the OK button.")


class TestGUIAutomationAgentLifecycle:
    def test_context_raises_before_first_run(self, tmp_path: Path) -> None:
        agent = GUIAutomationAgent(
            vlm=RespondImmediatelyChatModel(),
            services=make_services(tmp_path),
        )

        with pytest.raises(RuntimeError):
            agent.context

    def test_context_available_after_run(self, tmp_path: Path) -> None:
        agent = GUIAutomationAgent(
            vlm=RespondImmediatelyChatModel(),
            services=make_services(tmp_path),
        )

        agent.run("Do something.")

        assert agent.context.goal == "Do something."

    def test_reset_clears_context(self, tmp_path: Path) -> None:
        agent = GUIAutomationAgent(
            vlm=RespondImmediatelyChatModel(),
            services=make_services(tmp_path),
        )
        agent.run("Do something.")

        agent.reset()

        with pytest.raises(RuntimeError):
            agent.context

    def test_context_manager_closes_services(self, tmp_path: Path) -> None:
        services = make_services(tmp_path)

        with GUIAutomationAgent(
            vlm=RespondImmediatelyChatModel(), services=services
        ) as agent:
            agent.run("Do something.")

        # closing should not raise, and directory should be cleaned up
        assert not services.screenshots.directory.exists()
