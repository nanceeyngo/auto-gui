"""
Screenshot capture utilities for the GUI automation agent.
"""

import shutil
import uuid
from pathlib import Path
from time import perf_counter
from typing import Self

import pywinctl
from langchain_core.tools import tool
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ...config import settings
from ...logging_config import get_logger
from ..windows import WindowInfo
from .backend import ScreenshotBackend
from .pyautogui_backend import PyAutoGuiScreenshotBackend

logger = get_logger("agent.tools.screenshot")


class ScreenshotResult(BaseModel):
    """
    Metadata describing a captured screenshot.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    _cached_image: Image.Image | None = PrivateAttr(None)

    path: Path | None = None

    image: Image.Image | None = None

    width: int | None = Field(
        default=None,
        ge=1,
    )

    height: int | None = Field(
        default=None,
        ge=1,
    )

    left: int = 0

    top: int = 0

    @model_validator(mode="after")
    def validate_and_populate(self) -> Self:
        """
        Validate the screenshot source and populate missing metadata.
        """

        image_width: int | None = None
        image_height: int | None = None

        if self.image is not None:
            image_width, image_height = self.image.size
        elif self.path is not None:
            path = self.path.expanduser().resolve()

            object.__setattr__(
                self,
                "path",
                path,
            )
            if not path.exists():
                raise FileNotFoundError(f"Screenshot path does not exist: {path}")

            if not path.is_file():
                raise IsADirectoryError(f"Screenshot path is not a file: {path}")

            try:
                with Image.open(path) as image:
                    image_width, image_height = image.size
            except OSError as exc:
                raise ValueError(f"Unable to read image: {path}") from exc
        else:
            raise ValueError("Either 'path' or 'image' must be provided.")

        width = image_width if self.width is None else self.width

        height = image_height if self.height is None else self.height

        if width != image_width:
            raise ValueError(
                f"width ({width}) does not match image width ({image_width})."
            )

        if height != image_height:
            raise ValueError(
                f"height ({height}) does not match " f"image height ({image_height})."
            )

        object.__setattr__(
            self,
            "width",
            width,
        )

        object.__setattr__(
            self,
            "height",
            height,
        )

        return self

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def has_path(self) -> bool:
        return self.path is not None

    @property
    def image_or_load(self) -> Image.Image:
        if self.image is not None:
            return self.image

        if self._cached_image is None:
            if self.path is None:
                raise RuntimeError("No screenshot path is available.")

            with Image.open(self.path) as image:
                image.load()
                self._cached_image = image.copy()

        return self._cached_image

    @property
    def size(self) -> tuple[int | None, int | None]:
        return self.width, self.height

    @property
    def region(self) -> tuple[int, int, int | None, int | None]:
        return (
            self.left,
            self.top,
            self.width,
            self.height,
        )

    def close(self) -> None:
        image = self._cached_image

        if image is None:
            return

        image.close()
        self._cached_image = None


class ScreenshotManager:
    """
    Manages screenshot capture and cleanup.
    """

    __slots__ = (
        "_backend",
        "_directory",
        "_keep_screenshots",
        "_results",
    )

    def __init__(
        self,
        *,
        directory: Path,
        keep_screenshots: bool = False,
        backend: ScreenshotBackend | None = None,
    ) -> None:
        self._directory = directory.expanduser().resolve()
        self._keep_screenshots = keep_screenshots

        self._results: dict[
            Path,
            ScreenshotResult,
        ] = {}

        self._backend = backend if backend is not None else PyAutoGuiScreenshotBackend()

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _ensure_directory(self) -> Path:
        self._directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self._directory

    def _generate_filename(self) -> Path:
        return self._ensure_directory() / f"{uuid.uuid4().hex}.png"

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _store_image(
        self,
        image: Image.Image,
        *,
        left: int,
        top: int,
    ) -> ScreenshotResult:
        """
        Persist an image and create its metadata.
        """
        path = self._generate_filename()

        image.save(path)

        result = ScreenshotResult(
            path=path,
            image=image,
            width=image.width,
            height=image.height,
            left=left,
            top=top,
        )

        self._results[path] = result

        return result

    # ------------------------------------------------------------------
    # Capture methods
    # ------------------------------------------------------------------

    def capture(self) -> ScreenshotResult:
        """
        Capture the entire desktop.
        """
        started = perf_counter()
        image = self._backend.capture()
        result = self._store_image(
            image,
            left=0,
            top=0,
        )

        logger.debug(
            "Captured full-desktop screenshot",
            extra={
                "context": {
                    "width": result.width,
                    "height": result.height,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                }
            },
        )

        return result

    def capture_region(
        self,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> ScreenshotResult:
        """
        Capture a rectangular region.
        """
        started = perf_counter()
        image = self._backend.capture_region(
            left=left,
            top=top,
            width=width,
            height=height,
        )
        result = self._store_image(
            image,
            left=left,
            top=top,
        )

        logger.debug(
            "Captured region screenshot",
            extra={
                "context": {
                    "left": left,
                    "top": top,
                    "width": width,
                    "height": height,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                }
            },
        )

        return result

    def capture_window(
        self,
        window: pywinctl.Window | WindowInfo | None = None,
        *,
        title: str | None = None,
        exact: bool = False,
    ) -> ScreenshotResult:
        """
        Capture a specific application window.

        Exactly one of ``window`` or ``title`` must be supplied.

        Parameters
        ----------
        window
            Existing pywinctl window, or WindowInfo, object.

        title
            Window title to search for.

        exact
            Whether the title must match exactly.

        Raises
        ------
        ValueError
            If neither or both of ``window`` and ``title`` are
            supplied.

        RuntimeError
            If no matching window is found or multiple windows
            match the supplied title.
        """
        if (window is None) == (title is None):
            raise ValueError("Exactly one of 'window' or 'title' must be provided.")

        if window is None and title is not None:
            windows = pywinctl.getWindowsWithTitle(title)

            if exact:
                windows = [w for w in windows if w.title == title]

            if not windows:
                raise RuntimeError(f"No window found with title '{title}'.")

            if len(windows) > 1:
                raise RuntimeError(f"Multiple windows match title '{title}'.")

            window = windows[0]

        assert window is not None

        return self.capture_region(
            left=window.left,
            top=window.top,
            width=window.width,
            height=window.height,
        )

    def capture_active_window(self) -> ScreenshotResult:
        """
        Capture the currently active window.
        """
        windows = pywinctl.getActiveWindow()

        if windows is None:
            raise RuntimeError("No active window.")

        return self.capture_window(windows)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(
        self,
        screenshot: ScreenshotResult | Path | str | None = None,
    ) -> None:
        """
        Delete one screenshot or every tracked screenshot.
        """
        if screenshot is None:
            paths = tuple(self._results.keys())
        else:
            if isinstance(screenshot, ScreenshotResult):
                path = screenshot.path
            elif isinstance(screenshot, str):
                path = Path(screenshot)
            else:
                path = screenshot

            if path not in self._results:
                raise RuntimeError(f'No managed screenshot at the path: "{path}".')

            paths = (path,)

        for path in paths:
            try:
                self._results[path].close()
                path.unlink()
            except FileNotFoundError:
                pass

            self._results.pop(path)

        if not self._keep_screenshots and not self._results:
            self.cleanup_directory()

    def cleanup_directory(self) -> None:
        """
        Remove the screenshot directory.
        """
        if self._directory.exists():
            shutil.rmtree(
                self._directory,
                ignore_errors=True,
            )

    def close(self) -> None:
        self.cleanup()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def screenshots(self) -> tuple[Path, ...]:
        """
        Return tracked screenshots.
        """
        return tuple(sorted(self._results.keys()))

    def __len__(self) -> int:
        return len(self._results)

    @property
    def directory(self) -> Path:
        return self._directory


_default_manager = ScreenshotManager(
    directory=settings.screenshot_directory,
    keep_screenshots=settings.keep_screenshots,
)


@tool
def take_screenshot() -> ScreenshotResult:
    """
    Capture the entire desktop.

    Returns
    -------
    ScreenshotResult
        object containing  metadata on the screenshot.
    """
    return _default_manager.capture()


@tool
def take_region_screenshot(
    left: int,
    top: int,
    width: int,
    height: int,
) -> ScreenshotResult:
    """
    Capture a rectangular screen region.
    """
    return _default_manager.capture_region(
        left=left,
        top=top,
        width=width,
        height=height,
    )


@tool
def take_active_window_screenshot() -> ScreenshotResult:
    """
    Capture the currently active window.
    """
    return _default_manager.capture_active_window()


__all__ = [
    "ScreenshotManager",
    "ScreenshotResult",
    "take_active_window_screenshot",
    "take_region_screenshot",
    "take_screenshot",
]
