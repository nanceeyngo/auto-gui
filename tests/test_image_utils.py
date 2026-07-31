"""
Unit tests for `grounding.utils.images.load_image`.
"""

from pathlib import Path

import pytest
from PIL import Image

from grounding.utils.images import load_image


class TestLoadImage:
    def test_returns_pil_image_as_is(self) -> None:
        image = Image.new("RGB", (10, 10))

        assert load_image(image) is image

    def test_loads_from_path_object(self, tmp_path: Path) -> None:
        path = tmp_path / "img.png"
        Image.new("RGB", (20, 20)).save(path)

        loaded = load_image(path)

        assert loaded.size == (20, 20)

    def test_loads_from_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "img.png"
        Image.new("RGB", (30, 30)).save(path)

        loaded = load_image(str(path))

        assert loaded.size == (30, 30)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "missing.png")

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IsADirectoryError):
            load_image(tmp_path)
