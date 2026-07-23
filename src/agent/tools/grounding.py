"""
Grounding helpers for the GUI automation agent.
"""

from pathlib import Path

from grounding.client import GroundingClient
from grounding.models import (
    GroundingRequest,
    GroundingResponse,
)

from .screenshot.screenshot import ScreenshotResult


class GroundingManager:
    """
    Coordinates GUI element grounding using a GroundingClient.

    This class does not own screenshot capture. It only accepts
    existing screenshots or images and performs grounding.
    """

    __slots__ = ("_client",)

    def __init__(
        self,
        *,
        client: GroundingClient | None = None,
    ) -> None:
        self._client = client if client is not None else GroundingClient()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def client(self) -> GroundingClient:
        """
        Return the underlying GroundingClient.
        """
        return self._client

    # ------------------------------------------------------------------
    # Core grounding
    # ------------------------------------------------------------------

    def locate(
        self,
        *,
        screenshot: ScreenshotResult,
        query: str,
        provider: str | None = None,
    ) -> GroundingResponse:
        """
        Locate GUI elements within an existing screenshot.
        """
        request = GroundingRequest(
            image=screenshot.image_or_load,
            query=query,
        )

        return self._client.locate(
            request=request,
            provider=provider,
        )

    async def alocate(
        self,
        *,
        screenshot: ScreenshotResult,
        query: str,
        provider: str | None = None,
    ) -> GroundingResponse:
        """
        Asynchronously locate GUI elements within a screenshot.
        """
        request = GroundingRequest(
            image=screenshot.image_or_load,
            query=query,
        )

        return await self._client.alocate(
            request=request,
            provider=provider,
        )

    # ------------------------------------------------------------------
    # Existing image files
    # ------------------------------------------------------------------

    def locate_image(
        self,
        *,
        image: str | Path,
        query: str,
        provider: str | None = None,
    ) -> GroundingResponse:
        """
        Locate GUI elements within an existing image file.
        """
        return self.locate(
            screenshot=ScreenshotResult(
                path=Path(image),
            ),
            query=query,
            provider=provider,
        )

    async def alocate_image(
        self,
        *,
        image: str | Path,
        query: str,
        provider: str | None = None,
    ) -> GroundingResponse:
        """
        Asynchronously locate GUI elements within an existing image file.
        """
        return await self.alocate(
            screenshot=ScreenshotResult(
                path=Path(image),
            ),
            query=query,
            provider=provider,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Release owned resources.
        """
        self._client.close()

    async def aclose(self) -> None:
        """
        Asynchronously release owned resources.
        """
        await self._client.aclose()


grounding_manager = GroundingManager()


def locate(
    *,
    screenshot: ScreenshotResult,
    query: str,
    provider: str | None = None,
) -> GroundingResponse:
    """
    Locate GUI elements within a screenshot.
    """
    return grounding_manager.locate(
        screenshot=screenshot,
        query=query,
        provider=provider,
    )


async def alocate(
    *,
    screenshot: ScreenshotResult,
    query: str,
    provider: str | None = None,
) -> GroundingResponse:
    """
    Asynchronously locate GUI elements within a screenshot.
    """
    return await grounding_manager.alocate(
        screenshot=screenshot,
        query=query,
        provider=provider,
    )


def locate_image(
    *,
    image: str | Path,
    query: str,
    provider: str | None = None,
) -> GroundingResponse:
    """
    Locate GUI elements within an existing image.
    """
    return grounding_manager.locate_image(
        image=image,
        query=query,
        provider=provider,
    )


async def alocate_image(
    *,
    image: str | Path,
    query: str,
    provider: str | None = None,
) -> GroundingResponse:
    """
    Asynchronously locate GUI elements within an existing image.
    """
    return await grounding_manager.alocate_image(
        image=image,
        query=query,
        provider=provider,
    )


def close() -> None:
    """
    Release all shared grounding resources.
    """
    grounding_manager.close()


async def aclose() -> None:
    """
    Asynchronously release all shared grounding resources.
    """
    await grounding_manager.aclose()


__all__ = [
    "GroundingManager",
    "aclose",
    "alocate",
    "alocate_image",
    "close",
    "grounding_manager",
    "locate",
    "locate_image",
]
