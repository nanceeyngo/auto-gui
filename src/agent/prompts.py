"""
Prompt loading utilities for the GUI automation agent.
"""

from functools import cache
from pathlib import Path
from string import Formatter

_PROMPTS_DIRECTORY = Path(__file__).parent / "prompts"


@cache
def _load_prompt(path: Path) -> str:
    """
    Load a prompt template from disk.

    Prompt files are cached after the first read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")

    if not path.is_file():
        raise IsADirectoryError(f"Prompt path is not a file: {path}")

    return path.read_text(
        encoding="utf-8",
    )


def available_prompts() -> tuple[str, ...]:
    """
    Return the available prompt names.
    """
    return tuple(sorted(path.stem for path in _PROMPTS_DIRECTORY.glob("*.md")))


def prompt_path(name: str) -> Path:
    """
    Return the path of a prompt template.
    """
    path = (_PROMPTS_DIRECTORY / f"{name}.md").resolve()

    try:
        path.relative_to(_PROMPTS_DIRECTORY.resolve())
    except ValueError as exc:
        raise ValueError(f"Invalid prompt name: {name!r}") from exc

    return path


def get_prompt(
    name: str,
    **kwargs: object,
) -> str:
    """
    Load and format a prompt template.

    Examples
    --------
    get_prompt("system")

    get_prompt(
        "grounding",
        element="Search box",
    )
    """
    template = _load_prompt(
        prompt_path(name),
    )

    formatter = Formatter()

    required_fields = {
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name
    }

    missing = required_fields - kwargs.keys()

    if missing:
        missing_names = ", ".join(sorted(missing))

        raise ValueError(f"Missing prompt template variables: {missing_names}")

    unused = kwargs.keys() - required_fields

    if unused:
        raise ValueError(
            f"Unused prompt template variables: {', '.join(sorted(unused))}"
        )

    return template.format(**kwargs)


def get_system_prompt() -> str:
    """
    Return the agent system prompt.
    """
    return get_prompt("system")


def clear_prompt_cache() -> None:
    """
    Clear the prompt cache.
    """
    _load_prompt.cache_clear()


def reload_prompt(name: str) -> str:
    """
    Reload a single prompt from disk.
    """
    clear_prompt_cache()
    return get_prompt(name)


__all__ = [
    "available_prompts",
    "clear_prompt_cache",
    "get_prompt",
    "get_system_prompt",
    "prompt_path",
    "reload_prompt",
]
