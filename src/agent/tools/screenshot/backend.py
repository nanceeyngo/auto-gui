from abc import ABC, abstractmethod

from PIL import Image


class ScreenshotBackend(ABC):
    """
    Interface for screenshot capture backends.
    """

    @abstractmethod
    def capture(self) -> Image.Image:
        """
        Capture the entire desktop.
        """
        raise NotImplementedError

    @abstractmethod
    def capture_region(
        self,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> Image.Image:
        """
        Capture a rectangular screen region.
        """
        raise NotImplementedError
