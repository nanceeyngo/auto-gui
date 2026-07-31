"""
Regression tests for tool JSON-schema generation.

Bug this covers
----------------
`capture_screen` was the only tool in `GUI_AGENT_TOOLS` registered
without an explicit `args_schema`. LangChain fell back to
auto-inferring one from the function signature, which -- because the
function's only parameter is the injected `Runtime[AgentContext]` --
walked into `AgentContext` and hit `ScreenshotResult.image:
PIL.Image.Image`, an arbitrary type Pydantic cannot render as JSON
Schema. This broke `model.bind_tools(GUI_AGENT_TOOLS)` for *every*
tool the moment `capture_screen` was included, which only surfaces
when a real chat model's `bind_tools()` is exercised.

The unit tests in `tests/test_agent.py` did not catch this because
the fake chat models there override `bind_tools()` to skip real
schema conversion entirely (by design, to avoid needing a real model
backend). These tests instead call the real
`langchain_core.utils.function_calling.convert_to_openai_tool`
conversion used internally by `bind_tools()`, with no chat model
involved, so this class of bug can't hide behind a test double again.
"""

from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.tools.agent_tools import GUI_AGENT_TOOLS


class TestToolSchemasAreJsonSerializable:
    def test_every_tool_converts_to_openai_schema(self) -> None:
        failures: dict[str, str] = {}

        for tool in GUI_AGENT_TOOLS:
            try:
                convert_to_openai_tool(tool)
            except Exception as exc:
                failures[tool.name] = f"{type(exc).__name__}: {exc}"

        assert failures == {}, (
            f"{len(failures)} tool(s) failed OpenAI schema conversion: " f"{failures}"
        )

    def test_capture_screen_has_an_empty_but_explicit_schema(self) -> None:
        capture_screen = next(t for t in GUI_AGENT_TOOLS if t.name == "capture_screen")

        schema = convert_to_openai_tool(capture_screen)

        assert schema["function"]["name"] == "capture_screen"
        assert schema["function"]["parameters"]["properties"] == {}

    def test_all_tools_have_an_explicit_args_schema(self) -> None:
        """
        Guards against this regressing again for a *future* tool: any
        tool relying on auto-inferred schema is one `Runtime`-typed
        parameter away from this exact failure mode.
        """
        missing = [t.name for t in GUI_AGENT_TOOLS if t.args_schema is None]

        assert missing == []
