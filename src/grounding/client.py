"""
High-level client for interacting with grounding providers.
"""

import asyncio
from collections.abc import Iterator, Sequence
from types import TracebackType
from typing import Self, final, overload

from .config import GroundingSettings
from .exceptions import (
    GroundingError,
    GroundingFallbackError,
    GroundingProviderError,
    GroundingRequestError,
    NoDefaultGroundingProviderError,
)
from .interfaces import GroundingEngine
from .logging_utils import get_logger
from .models import (
    GroundingRequest,
    GroundingResponse,
    ImageLike,
)
from .registry import (
    GroundingProviderType,
    GroundingRegistry,
)

logger = get_logger("grounding.client")


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
            raise ValueError("'image' is required when 'request' is not supplied.")

        if query is None:
            raise ValueError("'query' is required when 'request' is not supplied.")

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
    # Fallback strategy
    # ------------------------------------------------------------------
    #
    # Tries a sequence of providers in order (e.g. a fast/cheap model
    # first, falling back to a higher-capacity model), moving to the
    # next provider whenever the current one raises, returns no
    # detections, or returns a best-detection confidence below
    # `min_confidence`. Raises GroundingFallbackError, carrying every
    # per-provider failure, if none of the providers succeed.

    def _provider_acceptable(
        self,
        provider_id: str,
        response: GroundingResponse,
        min_confidence: float | None,
    ) -> GroundingError | None:
        """
        Returns None if the response is acceptable, otherwise the
        reason it was rejected.
        """
        if not response.success:
            return GroundingRequestError(
                f"Provider '{provider_id}' returned no usable detections "
                f"(status={response.status})."
            )

        best = response.best_detection

        if (
            min_confidence is not None
            and best is not None
            and best.confidence is not None
            and best.confidence < min_confidence
        ):
            return GroundingRequestError(
                f"Provider '{provider_id}' best detection confidence "
                f"{best.confidence:.3f} is below the required threshold "
                f"{min_confidence:.3f}."
            )

        return None

    @final
    def locate_with_fallback(
        self,
        *,
        providers: Sequence[str],
        request: GroundingRequest | None = None,
        image: ImageLike | None = None,
        query: str | None = None,
        top_k: int = 1,
        confidence_threshold: float | None = None,
        min_confidence: float | None = None,
    ) -> GroundingResponse:
        """
        Locate a target, trying each provider in ``providers`` in
        order until one returns an acceptable detection.

        A provider's result is rejected -- causing the next provider
        to be tried -- when the provider raises a
        ``GroundingProviderError``, returns no detections, or its
        best detection's confidence falls below ``min_confidence``.

        Raises
        ------
        ValueError
            If ``providers`` is empty.

        GroundingFallbackError
            If every provider fails or is rejected. The exception
            carries a mapping of provider id -> failure reason.
        """
        if not providers:
            raise ValueError("'providers' must contain at least one provider id.")

        built_request = self._build_request(
            request=request,
            image=image,
            query=query,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        failures: dict[str, GroundingError] = {}

        for provider_id in providers:
            try:
                response = self.get_provider(provider_id).locate(built_request)
            except GroundingProviderError as exc:
                logger.warning(
                    "Grounding provider failed; trying next fallback",
                    extra={
                        "context": {
                            "provider": provider_id,
                            "exception_type": type(exc).__name__,
                        }
                    },
                )
                failures[provider_id] = exc
                continue

            rejection = self._provider_acceptable(provider_id, response, min_confidence)

            if rejection is None:
                if provider_id != providers[0]:
                    logger.info(
                        "Fallback grounding provider succeeded",
                        extra={
                            "context": {
                                "provider": provider_id,
                                "attempt": providers.index(provider_id) + 1,
                            }
                        },
                    )
                return response

            logger.info(
                "Grounding provider result rejected; trying next fallback",
                extra={
                    "context": {
                        "provider": provider_id,
                        "reason": str(rejection),
                    }
                },
            )
            failures[provider_id] = rejection

        raise GroundingFallbackError(
            "All configured grounding providers failed to produce an "
            "acceptable detection.",
            failures=failures,
        )

    @final
    async def alocate_with_fallback(
        self,
        *,
        providers: Sequence[str],
        request: GroundingRequest | None = None,
        image: ImageLike | None = None,
        query: str | None = None,
        top_k: int = 1,
        confidence_threshold: float | None = None,
        min_confidence: float | None = None,
    ) -> GroundingResponse:
        """
        Asynchronous counterpart to ``locate_with_fallback``.
        """
        if not providers:
            raise ValueError("'providers' must contain at least one provider id.")

        built_request = self._build_request(
            request=request,
            image=image,
            query=query,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )

        failures: dict[str, GroundingError] = {}

        for provider_id in providers:
            try:
                response = await self.get_provider(provider_id).alocate(built_request)
            except GroundingProviderError as exc:
                logger.warning(
                    "Grounding provider failed; trying next fallback",
                    extra={
                        "context": {
                            "provider": provider_id,
                            "exception_type": type(exc).__name__,
                        }
                    },
                )
                failures[provider_id] = exc
                continue

            rejection = self._provider_acceptable(provider_id, response, min_confidence)

            if rejection is None:
                return response

            failures[provider_id] = rejection

        raise GroundingFallbackError(
            "All configured grounding providers failed to produce an "
            "acceptable detection.",
            failures=failures,
        )

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

        await asyncio.gather(*(self.get_provider(pid).aclose() for pid in provider_ids))

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
