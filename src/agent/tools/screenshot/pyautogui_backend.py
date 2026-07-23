import pyautogui
from PIL import Image

from .backend import ScreenshotBackend


class PyAutoGuiScreenshotBackend(ScreenshotBackend):
    """
    Screenshot backend implemented using PyAutoGUI.
    """

    def capture(self) -> Image.Image:
        image: Image.Image = pyautogui.screenshot()
        return image

    def capture_region(
        self,
        *,
        left: int,
        top: int,
        width: int,
        height: int,
    ) -> Image.Image:
        image: Image.Image = pyautogui.screenshot(
            region=(
                left,
                top,
                width,
                height,
            )
        )
        return image
