"""
High-level client for interacting with grounding providers.
"""

import asyncio
from collections.abc import Iterator
from types import TracebackType
from typing import Self, final, overload

from .config import GroundingSettings
from .exceptions import NoDefaultGroundingProviderError
from .interfaces import GroundingEngine
from .models import (
    GroundingRequest,
    GroundingResponse,
    ImageLike,
)
from .registry import (
    GroundingProviderType,
    GroundingRegistry,
)


class GroundingClient:
    """
    High-level entry point for visual grounding.

    The client owns provider instances, reusing them across requests.
    Provider classes are obtained from the registry and lazily
    instantiated on first use.
    """

    __slots__ = (
        "_providers",
        "_registry",
        "_settings",
    )

    def __init__(
        self,
        *,
        registry: GroundingRegistry | None = None,
        settings: GroundingSettings | None = None,
    ) -> None:
        if registry is None:
            from .defaults import registry as default_registry

            registry = default_registry

        self._registry = registry
        self._settings = settings or GroundingSettings()

        self._providers: dict[str, GroundingEngine] = {}

    # ------------------------------------------------------------------
    # Provider management
    # ------------------------------------------------------------------

    @property
    def registry(self) -> GroundingRegistry:
        return self._registry

    @property
    def settings(self) -> GroundingSettings:
        return self._settings

    @property
    def available_providers(self) -> tuple[str, ...]:
        return self._registry.registered_ids()

    def register(
        self,
        provider: GroundingProviderType,
        *,
        overwrite: bool = False,
    ) -> Self:
        self._registry.register(
            provider,
            overwrite=overwrite,
        )
        return self

    def unregister(
        self,
        provider_id: str,
    ) -> None:
        self._providers.pop(provider_id, None)
        self._registry.unregister(provider_id)

    def clear_cache(self) -> None:
        """
        Forget all instantiated providers.

        Already-created providers are closed first.
        """
        self.close()

    # ------------------------------------------------------------------
    # Provider lookup
    # ------------------------------------------------------------------

    def get_provider(
        self,
        provider_id: str | None = None,
    ) -> GroundingEngine:
        """
        Return a provider instance.

        Providers are instantiated lazily and then cached.
        """

        provider_id = self._resolve_provider(provider_id)

        provider = self._providers.get(provider_id)

        if provider is None:
            provider = self._registry.create(provider_id)
            self._providers[provider_id] = provider

        return provider

    def _resolve_provider(
        self,
        provider_id: str | None,
    ) -> str:
        provider_id = provider_id or self._settings.default_provider

        if provider_id is None:
            raise NoDefaultGroundingProviderError(
                "No grounding provider was specified and "
                "GROUNDING_DEFAULT_PROVIDER is not configured."
            )

        return provider_id

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    @staticmethod
    def _build_request(
        *,
        request: GroundingRequest | None = None,
        image: ImageLike | None = None,
        query: str | None = None,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> GroundingRequest:
        """
        Construct a GroundingRequest.

        Either provide an existing request, or provide
        image + query.
        """

        if request is not None:
            return request

        if image is None:
            raise ValueError(
                "'image' is required when 'request' is not supplied."
            )

        if query is None:
            raise ValueError(
                "'query' is required when 'request' is not supplied."
            )

        return GroundingRequest(
            image=image,
            query=query,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

    @overload
    def locate(
        self,
        *,
        request: GroundingRequest,
        provider: str | None = ...,
    ) -> GroundingResponse: ...

    @overload
    def locate(
        self,
        *,
        image: ImageLike,
        query: str,
        provider: str | None = ...,
        top_k: int = ...,
        confidence_threshold: float | None = ...,
    ) -> GroundingResponse: ...

    @final
    def locate(
        self,
        *,
        request: GroundingRequest | None = None,
        provider: str | None = None,
        image: ImageLike | None = None,
        query: str | None = None,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> GroundingResponse:
        request = self._build_request(
            request=request,
            image=image,
            query=query,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        return self.get_provider(provider).locate(request)

    @overload
    async def alocate(
        self,
        *,
        request: GroundingRequest,
        provider: str | None = ...,
    ) -> GroundingResponse: ...

    @overload
    async def alocate(
        self,
        *,
        image: ImageLike,
        query: str,
        provider: str | None = ...,
        top_k: int = ...,
        confidence_threshold: float | None = ...,
    ) -> GroundingResponse: ...

    @final
    async def alocate(
        self,
        *,
        request: GroundingRequest | None = None,
        provider: str | None = None,
        image: ImageLike | None = None,
        query: str | None = None,
        top_k: int = 1,
        confidence_threshold: float | None = None,
    ) -> GroundingResponse:
        request = self._build_request(
            request=request,
            image=image,
            query=query,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        return await self.get_provider(provider).alocate(request)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> dict[str, bool]:
        return {
            provider_id: self.get_provider(provider_id).health_check()
            for provider_id in self._registry
        }

    async def ahealth_check(self) -> dict[str, bool]:
        provider_ids = tuple(self._registry)

        results = await asyncio.gather(
            *(self.get_provider(pid).ahealth_check() for pid in provider_ids)
        )

        return dict(zip(provider_ids, results, strict=False))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()

        self._providers.clear()

    async def aclose(self) -> None:
        provider_ids = tuple(self._registry)

        await asyncio.gather(
            *(self.get_provider(pid).aclose() for pid in provider_ids)
        )

        self._providers.clear()

    # ------------------------------------------------------------------
    # Context managers
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def __contains__(
        self,
        provider_id: object,
    ) -> bool:
        if not isinstance(provider_id, str):
            return False

        return provider_id in self._registry

    def __bool__(self) -> bool:
        return bool(self._registry)

    def __len__(self) -> int:
        return len(self._registry)

    def __iter__(self) -> Iterator[str]:
        return iter(self._registry)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"default_provider={self._settings.default_provider!r}, "
            f"registered={len(self._registry)}, "
            f"cached={len(self._providers)}"
            f")"
        )
