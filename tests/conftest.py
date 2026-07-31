"""
Shared fixtures for the `agent` test suite.

The root-level `conftest.py` installs fake `pyautogui`/`pywinctl`
modules into `sys.modules` before collection. This module exposes
fixtures for reaching into those fakes from individual tests.
"""

import sys
from collections.abc import Iterator

import pytest

from tests.agent_fakes import CallRecorder, MouseState


@pytest.fixture
def fake_pyautogui_recorder() -> Iterator[CallRecorder]:
    """
    The shared `CallRecorder` backing the fake `pyautogui` module,
    reset before and after each test.
    """
    import conftest as root_conftest

    recorder: CallRecorder = root_conftest._fake_recorder
    recorder.calls.clear()

    yield recorder

    recorder.calls.clear()


@pytest.fixture
def fake_mouse() -> Iterator[MouseState]:
    """
    The shared `MouseState` backing the fake `pyautogui` module,
    reset to the origin before and after each test.
    """
    import conftest as root_conftest

    mouse: MouseState = root_conftest._fake_mouse
    mouse.x, mouse.y = 0, 0

    yield mouse

    mouse.x, mouse.y = 0, 0


@pytest.fixture
def fake_pyautogui() -> Iterator[object]:
    """
    The fake `pyautogui` module object itself, as installed in
    `sys.modules`. Useful for monkeypatching individual functions
    (e.g. `size`, `screenshot`) within a single test.
    """
    yield sys.modules["pyautogui"]
