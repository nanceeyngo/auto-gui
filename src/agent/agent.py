"""
High-level GUI automation agent.
"""

import operator
from typing import Annotated, Any, Self, cast

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel

from .config import settings
from .context import AgentContext, AgentServices
from .exceptions import AgentExecutionError
from .logging_config import get_logger
from .prompts import (
    get_prompt,
    get_system_prompt,
)
from .tools.action.actions import ActionManager
from .tools.agent_tools import GUI_AGENT_TOOLS
from .tools.grounding import GroundingManager
from .tools.screenshot.screenshot import ScreenshotManager
from .vlm import create_vlm

logger = get_logger("agent.agent")


class GUIAgentState(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add]


class GUIAutomationAgent:
    __slots__ = (
        "_agent",
        "_context",
        "_services",
        "_vlm",
    )

    def __init__(
        self,
        *,
        vlm: BaseChatModel | None = None,
        services: AgentServices | None = None,
    ) -> None:
        self._vlm = vlm or create_vlm()

        self._services = services or AgentServices(
            grounding=GroundingManager(),
            screenshots=ScreenshotManager(directory=settings.screenshot_directory),
            actions=ActionManager(),
            config=settings,
        )

        self._context: AgentContext | None = None

        self._agent = create_agent(
            model=self._vlm,
            tools=GUI_AGENT_TOOLS,
            system_prompt=get_system_prompt(),
            context_schema=AgentContext,
        )

    @staticmethod
    def _build_state(task: str) -> GUIAgentState:
        prompt = get_prompt(
            "task",
            task=task,
        )

        return GUIAgentState(
            messages=[
                HumanMessage(content=prompt),
            ]
        )

    def _create_context(
        self,
        task: str,
    ) -> AgentContext:
        context = AgentContext(
            goal=task,
            services=self._services,
        )
        self._context = context
        context.reset_execution()

        return context

    def _begin_run(self, task: str) -> tuple[GUIAgentState, AgentContext]:
        context = self._create_context(task)
        state = self._build_state(task)
        return state, context

    async def arun(
        self,
        task: str,
    ) -> str:
        state, context = self._begin_run(task)

        logger.info(
            "Agent run starting",
            extra={"context": {"goal": task}},
        )

        try:
            result = await self._agent.ainvoke(
                cast(Any, state),
                context=context,
            )
        except Exception as exc:
            logger.error(
                "Agent run failed",
                extra={
                    "context": {
                        "goal": task,
                        "exception_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )
            raise AgentExecutionError(
                f"Agent run failed while executing task {task!r}: {exc}",
                goal=task,
            ) from exc

        last_message_content: str = result["messages"][-1].content

        logger.info(
            "Agent run completed",
            extra={
                "context": {
                    "goal": task,
                    "iterations": context.iteration,
                }
            },
        )

        return last_message_content

    def run(
        self,
        task: str,
    ) -> str:
        state, context = self._begin_run(task)

        logger.info(
            "Agent run starting",
            extra={"context": {"goal": task}},
        )

        try:
            result = self._agent.invoke(
                cast(Any, state),
                context=context,
            )
        except Exception as exc:
            logger.error(
                "Agent run failed",
                extra={
                    "context": {
                        "goal": task,
                        "exception_type": type(exc).__name__,
                    }
                },
                exc_info=True,
            )
            raise AgentExecutionError(
                f"Agent run failed while executing task {task!r}: {exc}",
                goal=task,
            ) from exc

        last_message_content: str = result["messages"][-1].content

        logger.info(
            "Agent run completed",
            extra={
                "context": {
                    "goal": task,
                    "iterations": context.iteration,
                }
            },
        )

        return last_message_content

    async def aclose(self) -> None:
        self._services.screenshots.close()
        self._services.actions.close()
        await self._services.grounding.aclose()

    def close(self) -> None:
        self._services.screenshots.close()
        self._services.actions.close()
        self._services.grounding.close()

    def reset(self) -> None:
        """
        Reset any execution-specific state.
        """
        self._context = None

    @property
    def context(self) -> AgentContext:
        if self._context is None:
            raise RuntimeError("Agent is not currently executing.")
        return self._context

    @property
    def services(self) -> AgentServices:
        return self._services

    @property
    def goal(self) -> str:
        return self.context.goal

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
