"""
Coordinate mapping between screenshot pixel space and OS input space.

Bug this module fixes
----------------------
Screenshots (and therefore every bounding box a grounding provider
returns) are expressed in *physical* image pixels. On High-DPI /
Retina / scaled displays, the physical pixel resolution of a
screenshot is a multiple of the *logical* coordinate space that the
OS mouse/keyboard input APIs (and therefore ``pyautogui.moveTo`` /
``pyautogui.click``) operate in.

For example, on a 2x Retina display, ``pyautogui.screenshot()``
commonly returns an image that is 2x the width/height reported by
``pyautogui.size()``. If a grounding-derived bounding box center of
``(2400, 1200)`` (screenshot pixel space) is passed directly to
``pyautogui.click(x=2400, y=1200)`` (which expects logical
coordinates), the click lands far outside the intended element -- or
off-screen entirely -- because it is silently treated as a logical
coordinate on a 2x-too-small canvas.

``CoordinateMapper`` fixes this by computing the scale factor between
the physical screenshot resolution and the logical screen size
reported by the input backend, and converting screenshot-space pixel
coordinates into logical OS coordinates before they are dispatched to
``pyautogui``.

On displays with no DPI scaling (the common case on Linux/X11 and
most non-Retina Windows configurations), the scale factor is exactly
``1.0`` and this module is a no-op passthrough.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from ..logging_config import get_logger

logger = get_logger("agent.tools.coordinates")


class Scale(NamedTuple):
    """
    Ratio of physical (screenshot) pixels to logical (input) pixels.
    """

    x: float
    y: float

    @property
    def is_identity(self) -> bool:
        return self.x == 1.0 and self.y == 1.0


class CoordinateMapper:
    """
    Converts screenshot-space pixel coordinates to OS logical
    coordinates, accounting for display scaling (High-DPI/Retina).

    The scale factor is computed lazily from a physical resolution
    provider (typically a screenshot backend) and a logical resolution
    provider (typically ``pyautogui.size``), and cached until
    ``refresh()`` is called or the providers report a different size
    (e.g. after a display configuration change).
    """

    __slots__ = (
        "_logical_size_fn",
        "_physical_size_fn",
        "_scale",
    )

    def __init__(
        self,
        *,
        physical_size_fn: Callable[[], tuple[int, int]],
        logical_size_fn: Callable[[], tuple[int, int]],
    ) -> None:
        """
        Parameters
        ----------
        physical_size_fn
            Zero-argument callable returning the ``(width, height)``
            of a full-desktop screenshot, in physical pixels.

        logical_size_fn
            Zero-argument callable returning the ``(width, height)``
            of the logical coordinate space used by the OS input
            backend (e.g. ``pyautogui.size``).
        """
        self._physical_size_fn = physical_size_fn
        self._logical_size_fn = logical_size_fn
        self._scale: Scale | None = None

    # ------------------------------------------------------------------
    # Scale computation
    # ------------------------------------------------------------------

    def refresh(self) -> Scale:
        """
        Recompute (and cache) the scale factor.
        """
        physical_width, physical_height = self._physical_size_fn()
        logical_width, logical_height = self._logical_size_fn()

        if logical_width <= 0 or logical_height <= 0:
            raise ValueError(
                "Logical screen size must be positive; got "
                f"({logical_width}, {logical_height})."
            )

        scale = Scale(
            x=physical_width / logical_width,
            y=physical_height / logical_height,
        )

        if scale != self._scale:
            logger.info(
                "Computed display scale factor",
                extra={
                    "context": {
                        "physical_size": (physical_width, physical_height),
                        "logical_size": (logical_width, logical_height),
                        "scale_x": round(scale.x, 4),
                        "scale_y": round(scale.y, 4),
                    }
                },
            )

        self._scale = scale

        return scale

    @property
    def scale(self) -> Scale:
        if self._scale is None:
            return self.refresh()

        return self._scale

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_logical(self, x: int, y: int) -> tuple[int, int]:
        """
        Convert a screenshot-space pixel coordinate to a logical OS
        coordinate suitable for ``pyautogui`` mouse/keyboard input.
        """
        scale = self.scale

        if scale.is_identity:
            return x, y

        logical_x = round(x / scale.x)
        logical_y = round(y / scale.y)

        logger.debug(
            "Mapped screenshot coordinate to logical coordinate",
            extra={
                "context": {
                    "screenshot_xy": (x, y),
                    "logical_xy": (logical_x, logical_y),
                    "scale_x": round(scale.x, 4),
                    "scale_y": round(scale.y, 4),
                }
            },
        )

        return logical_x, logical_y

    def to_physical(self, x: int, y: int) -> tuple[int, int]:
        """
        Convert a logical OS coordinate back to screenshot pixel
        space. Primarily useful for tests/diagnostics.
        """
        scale = self.scale

        if scale.is_identity:
            return x, y

        return round(x * scale.x), round(y * scale.y)


class IdentityCoordinateMapper(CoordinateMapper):
    """
    A coordinate mapper that never scales.

    Useful as an explicit opt-out, or in test environments where
    computing a real screenshot resolution is undesirable.
    """

    def __init__(self) -> None:
        super().__init__(
            physical_size_fn=lambda: (1, 1),
            logical_size_fn=lambda: (1, 1),
        )
        self._scale = Scale(x=1.0, y=1.0)

    def refresh(self) -> Scale:
        assert self._scale is not None
        return self._scale


__all__ = [
    "CoordinateMapper",
    "IdentityCoordinateMapper",
    "Scale",
]
