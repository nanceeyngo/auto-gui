# ------------------------------------------------------------------
# Image helpers
# ------------------------------------------------------------------

from pathlib import Path

from PIL import Image
from PIL.Image import Image as PILImage


def load_image(
    image: str | Path | PILImage,
) -> PILImage:
    """
    Load an image into a PIL Image.
    """

    if isinstance(image, PILImage):
        return image

    if isinstance(image, str):
        image = Path(image)

    if not image.exists():
        raise FileNotFoundError(image)

    if not image.is_file():
        raise IsADirectoryError(image)

    return Image.open(image)
