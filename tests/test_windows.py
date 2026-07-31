"""
Unit tests for `agent.tools.windows.WindowManager`.
"""

import sys
from typing import Any

import pytest

from agent.tools.windows import WindowManager


@pytest.fixture(autouse=True)
def clean_windows():
    """
    Ensure the fake pywinctl window registry starts empty for every
    test and is cleaned up afterward.
    """
    fake_pywinctl = sys.modules["pywinctl"]
    fake_pywinctl._windows.clear()
    yield
    fake_pywinctl._windows.clear()


def add_window(title: str, **kwargs: Any) -> Any:
    fake_pywinctl = sys.modules["pywinctl"]
    window = fake_pywinctl.Window(title, **kwargs)
    fake_pywinctl._windows.append(window)
    return window


@pytest.fixture
def manager() -> WindowManager:
    return WindowManager()


class TestListWindows:
    def test_lists_visible_titled_windows(self, manager: WindowManager) -> None:
        add_window("Notepad")
        add_window("Calculator")

        titles = {w.title for w in manager.list_windows()}

        assert titles == {"Notepad", "Calculator"}

    def test_skips_windows_with_blank_titles(self, manager: WindowManager) -> None:
        add_window("   ")
        add_window("Real Window")

        titles = [w.title for w in manager.list_windows()]

        assert titles == ["Real Window"]

    def test_skips_invisible_windows(self, manager: WindowManager) -> None:
        window = add_window("Hidden")
        window.isVisible = False
        add_window("Visible")

        titles = [w.title for w in manager.list_windows()]

        assert titles == ["Visible"]


class TestFindAndGetWindow:
    def test_find_windows_substring_match(self, manager: WindowManager) -> None:
        add_window("Google Chrome - Tab 1")

        found = manager.find_windows("Chrome")

        assert len(found) == 1

    def test_find_windows_exact_match_filters_substrings(
        self, manager: WindowManager
    ) -> None:
        add_window("Chrome")
        add_window("Chrome Extended")

        found = manager.find_windows("Chrome", exact=True)

        assert len(found) == 1
        assert found[0].title == "Chrome"

    def test_get_window_raises_when_no_match(self, manager: WindowManager) -> None:
        with pytest.raises(RuntimeError):
            manager.get_window("Nonexistent")

    def test_get_window_raises_when_multiple_match(
        self, manager: WindowManager
    ) -> None:
        add_window("Dup")
        add_window("Dup")

        with pytest.raises(RuntimeError):
            manager.get_window("Dup")

    def test_get_window_returns_single_match(self, manager: WindowManager) -> None:
        add_window("Unique Window")

        window = manager.get_window("Unique Window")

        assert window.title == "Unique Window"


class TestActiveWindow:
    def test_get_active_window_returns_active(self, manager: WindowManager) -> None:
        window = add_window("Active One")
        window.isActive = True

        info = manager.get_active_window()

        assert info.title == "Active One"

    def test_get_active_window_raises_when_none_active(
        self, manager: WindowManager
    ) -> None:
        window = add_window("Inactive")
        window.isActive = False

        with pytest.raises(RuntimeError):
            manager.get_active_window()


class TestWaitForWindow:
    def test_returns_immediately_when_window_exists(
        self, manager: WindowManager
    ) -> None:
        add_window("Ready")

        window = manager.wait_for_window("Ready", timeout=1.0)

        assert window.title == "Ready"

    def test_times_out_when_window_never_appears(self, manager: WindowManager) -> None:
        with pytest.raises(TimeoutError):
            manager.wait_for_window("Never", timeout=0.2, poll_interval=0.05)

    def test_raises_on_ambiguous_match_while_waiting(
        self, manager: WindowManager
    ) -> None:
        add_window("Ambiguous")
        add_window("Ambiguous")

        with pytest.raises(RuntimeError):
            manager.wait_for_window("Ambiguous", timeout=1.0)


class TestWindowOperations:
    def test_activate_window_activates_and_restores(
        self, manager: WindowManager
    ) -> None:
        window = add_window("App")
        window.isMinimized = True

        info = manager.activate_window("App")

        assert info.is_active
        assert not info.is_minimized

    def test_activate_window_logs_but_does_not_raise_on_restore_failure(
        self, manager: WindowManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window = add_window("Flaky")

        def boom() -> None:
            raise RuntimeError("cannot restore")

        monkeypatch.setattr(window, "restore", boom)

        info = manager.activate_window("Flaky")

        assert info.title == "Flaky"

    def test_minimize_window(self, manager: WindowManager) -> None:
        add_window("ToMinimize")

        info = manager.minimize_window("ToMinimize")

        assert info.is_minimized

    def test_maximize_window(self, manager: WindowManager) -> None:
        add_window("ToMaximize")

        info = manager.maximize_window("ToMaximize")

        assert info.is_maximized

    def test_restore_window(self, manager: WindowManager) -> None:
        window = add_window("ToRestore")
        window.isMaximized = True

        info = manager.restore_window("ToRestore")

        assert not info.is_maximized

    def test_close_window(self, manager: WindowManager) -> None:
        window = add_window("ToClose")

        manager.close_window("ToClose")

        assert window.isVisible is False
