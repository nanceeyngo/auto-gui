"""
Registry for grounding providers.
"""

from collections.abc import Iterable, Iterator
from typing import Self

from .exceptions import UnknownGroundingProviderError
from .interfaces import GroundingEngine

type GroundingProviderType = type[GroundingEngine]


class GroundingRegistry:
    """
    Registry of available grounding providers.

    Providers are registered by their provider_id.

    Example:
        registry = GroundingRegistry()

        registry.register(VLMRunGrounder)
        registry.register(OmniParserGrounder)

        engine = registry.create("vlmrun")
    """

    def __init__(self) -> None:
        self._providers: dict[str, GroundingProviderType] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        provider: GroundingProviderType,
        *,
        overwrite: bool = False,
    ) -> Self:
        """
        Register a grounding provider.

        Args:
            provider:
                Provider class.

            overwrite:
                Whether an existing registration may be replaced.

        Raises:
            ValueError:
                If the provider is already registered and
                overwrite=False.
        """

        identifier = provider.provider_id

        if not overwrite and identifier in self._providers:
            raise ValueError(
                f"Grounding provider '{identifier}' is already registered."
            )

        self._providers[identifier] = provider

        return self

    def register_many(
        self,
        providers: Iterable[GroundingProviderType],
        *,
        overwrite: bool = False,
    ) -> Self:
        """
        Register multiple providers.
        """

        for provider in providers:
            self.register(
                provider,
                overwrite=overwrite,
            )

        return self

    def unregister(
        self,
        provider_id: str,
    ) -> None:
        """
        Remove a provider registration.

        Raises:
            UnknownGroundingProviderError
                If the provider is not registered.
        """

        try:
            del self._providers[provider_id]
        except KeyError as exc:
            raise UnknownGroundingProviderError(
                f"Unknown grounding provider '{provider_id}'."
            ) from exc

    def clear(self) -> None:
        """
        Remove all registered providers.
        """

        self._providers.clear()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self,
        provider_id: str,
    ) -> GroundingProviderType:
        """
        Return the registered provider class.

        Raises:
            UnknownGroundingProviderError
        """

        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise UnknownGroundingProviderError(
                f"Unknown grounding provider '{provider_id}'."
            ) from exc

    def create(
        self,
        provider_id: str,
    ) -> GroundingEngine:
        """
        Construct a provider instance.

        Args:
            provider_id:
                Identifier of the registered provider.

        Raises:
            UnknownGroundingProviderError:
                If no provider with the given identifier is registered.
        """

        provider = self.get(provider_id)

        return provider()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def contains(
        self,
        provider_id: str,
    ) -> bool:
        """
        Whether a provider is registered.
        """

        return provider_id in self

    def registered_ids(self) -> tuple[str, ...]:
        """
        Return registered provider identifiers.
        """

        return tuple(sorted(self._providers))

    def registered_providers(
        self,
    ) -> tuple[GroundingProviderType, ...]:
        """
        Return registered provider classes.
        """

        return tuple(
            self._providers[provider_id]
            for provider_id in self.registered_ids()
        )

    def __bool__(self) -> bool:
        return bool(self._providers)

    def __contains__(
        self,
        item: object,
    ) -> bool:
        if not isinstance(item, str):
            return False

        return item in self._providers

    def __getitem__(
        self,
        provider_id: str,
    ) -> GroundingProviderType:
        return self.get(provider_id)

    def __len__(self) -> int:
        return len(self._providers)

    def __iter__(self) -> Iterator[str]:
        return iter(self.registered_ids())

    def __repr__(self) -> str:
        registered_ids = self.registered_ids()

        return (
            f"{type(self).__name__}("
            f"count={len(registered_ids)}, "
            f"providers={list(registered_ids)!r}"
            f")"
        )


__all__ = [
    "GroundingProviderType",
    "GroundingRegistry",
]
