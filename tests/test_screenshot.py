"""
Unit tests for `agent.tools.screenshot.screenshot`.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

from agent.tools.screenshot.backend import ScreenshotBackend
from agent.tools.screenshot.screenshot import ScreenshotManager, ScreenshotResult


class FakeScreenshotBackend(ScreenshotBackend):
    """
    Deterministic in-memory screenshot backend for tests.
    """

    def __init__(self, size: tuple[int, int] = (200, 100)) -> None:
        self.size = size
        self.capture_calls = 0
        self.region_calls: list[tuple[int, int, int, int]] = []

    def capture(self) -> Image.Image:
        self.capture_calls += 1
        return Image.new("RGB", self.size, color=(10, 20, 30))

    def capture_region(
        self, *, left: int, top: int, width: int, height: int
    ) -> Image.Image:
        self.region_calls.append((left, top, width, height))
        return Image.new("RGB", (width, height), color=(1, 2, 3))


@pytest.fixture
def manager(tmp_path: Path) -> ScreenshotManager:
    return ScreenshotManager(
        directory=tmp_path / "shots",
        backend=FakeScreenshotBackend(),
    )


class TestScreenshotResult:
    def test_populates_dimensions_from_image(self) -> None:
        image = Image.new("RGB", (640, 480))

        result = ScreenshotResult(image=image)

        assert result.width == 640
        assert result.height == 480
        assert result.size == (640, 480)

    def test_explicit_width_must_match_image_width(self) -> None:
        image = Image.new("RGB", (640, 480))

        with pytest.raises(ValueError):
            ScreenshotResult(image=image, width=999)

    def test_missing_source_raises(self) -> None:
        with pytest.raises(ValueError):
            ScreenshotResult()

    def test_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ScreenshotResult(path=tmp_path / "missing.png")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            ScreenshotResult(path=tmp_path)

    def test_image_or_load_from_path(self, tmp_path: Path) -> None:
        path = tmp_path / "shot.png"
        Image.new("RGB", (50, 50)).save(path)

        result = ScreenshotResult(path=path)
        loaded = result.image_or_load

        assert loaded.size == (50, 50)

    def test_region_property(self) -> None:
        result = ScreenshotResult(image=Image.new("RGB", (10, 10)), left=5, top=7)

        assert result.region == (5, 7, 10, 10)


class TestScreenshotManagerCapture:
    def test_capture_stores_result_and_writes_file(
        self, manager: ScreenshotManager
    ) -> None:
        result = manager.capture()

        assert result.path is not None
        assert result.path.exists()
        assert result.left == 0
        assert result.top == 0
        assert len(manager) == 1

    def test_capture_region_passes_coordinates_to_backend(
        self, manager: ScreenshotManager
    ) -> None:
        backend = manager._backend
        assert isinstance(backend, FakeScreenshotBackend)

        result = manager.capture_region(left=10, top=20, width=300, height=150)

        assert backend.region_calls == [(10, 20, 300, 150)]
        assert result.width == 300
        assert result.height == 150
        assert result.left == 10
        assert result.top == 20


class TestScreenshotManagerCleanup:
    def test_cleanup_single_screenshot_removes_file(
        self, manager: ScreenshotManager
    ) -> None:
        result = manager.capture()
        path = result.path
        assert path is not None

        manager.cleanup(result)

        assert not path.exists()
        assert len(manager) == 0

    def test_cleanup_all_screenshots(self, manager: ScreenshotManager) -> None:
        manager.capture()
        manager.capture()

        manager.cleanup()

        assert len(manager) == 0

    def test_cleanup_unknown_screenshot_raises(
        self, manager: ScreenshotManager, tmp_path: Path
    ) -> None:
        with pytest.raises(RuntimeError):
            manager.cleanup(tmp_path / "unmanaged.png")

    def test_cleanup_removes_directory_when_not_keeping_screenshots(
        self, manager: ScreenshotManager
    ) -> None:
        manager.capture()
        directory = manager.directory

        manager.cleanup()

        assert not directory.exists()

    def test_keep_screenshots_preserves_directory(self, tmp_path: Path) -> None:
        manager = ScreenshotManager(
            directory=tmp_path / "kept",
            keep_screenshots=True,
            backend=FakeScreenshotBackend(),
        )
        manager.capture()

        manager.cleanup()

        assert manager.directory.exists()

    def test_close_calls_cleanup(self, manager: ScreenshotManager) -> None:
        manager.capture()

        manager.close()

        assert len(manager) == 0


class TestScreenshotManagerWindowCapture:
    def test_capture_window_requires_exactly_one_of_window_or_title(
        self, manager: ScreenshotManager
    ) -> None:
        with pytest.raises(ValueError):
            manager.capture_window()

    def test_capture_window_by_title(
        self, manager: ScreenshotManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pywinctl = sys.modules["pywinctl"]
        window = fake_pywinctl.Window("My App", left=1, top=2, width=300, height=400)

        def fake_get_windows_with_title(title: str) -> list[object]:
            return [window]

        monkeypatch.setattr(
            fake_pywinctl, "getWindowsWithTitle", fake_get_windows_with_title
        )

        result = manager.capture_window(title="My App")

        assert result.left == 1
        assert result.top == 2
        assert result.width == 300
        assert result.height == 400

    def test_capture_window_no_match_raises(
        self, manager: ScreenshotManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pywinctl = sys.modules["pywinctl"]

        def fake_get_windows_with_title(title: str) -> list[object]:
            return []

        monkeypatch.setattr(
            fake_pywinctl, "getWindowsWithTitle", fake_get_windows_with_title
        )

        with pytest.raises(RuntimeError):
            manager.capture_window(title="Nonexistent")

    def test_capture_window_multiple_matches_raises(
        self, manager: ScreenshotManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pywinctl = sys.modules["pywinctl"]
        windows = [
            fake_pywinctl.Window("Dup"),
            fake_pywinctl.Window("Dup"),
        ]

        def fake_get_windows_with_title(title: str) -> list[object]:
            return windows

        monkeypatch.setattr(
            fake_pywinctl, "getWindowsWithTitle", fake_get_windows_with_title
        )

        with pytest.raises(RuntimeError):
            manager.capture_window(title="Dup")

    def test_capture_active_window_no_active_window_raises(
        self, manager: ScreenshotManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pywinctl = sys.modules["pywinctl"]

        def fake_get_active_window() -> None:
            return None

        monkeypatch.setattr(fake_pywinctl, "getActiveWindow", fake_get_active_window)

        with pytest.raises(RuntimeError):
            manager.capture_active_window()
