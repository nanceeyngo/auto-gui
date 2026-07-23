"""
Abstract interfaces for grounding providers.
"""

import asyncio
import inspect
import re
from abc import ABC, abstractmethod
from inspect import isabstract
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, cast, get_args

from .models import GroundingRequest, GroundingResponse

GroundingCapability = Literal[
    # Capabilities exposed by grounding providers.
    "confidence_scores",
    "icon_detection",
    "local_model",
    "multi_detection",
    "natural_language_query",
    "ocr",
    "screen_parsing",
    "semantic_grounding",
    "remote_api",
]


class GroundingEngine(ABC):
    """
    Base class for all grounding providers.

    Subclasses must define a unique provider_id.

    Conventions:
        - lowercase
        - ASCII letters, digits and underscores only
        - stable across releases
        - never derived dynamically

    Examples:
        "vlmrun"
        "omniparser"
        "groundingdino"

    Providers are intentionally lightweight and stateless.

    Responsibilities:
        - accept a GroundingRequest
        - perform grounding
        - return a GroundingResponse
    """

    _IDENTIFIER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^[a-z][a-z0-9_]*$"
    )
    _VALID_CAPABILITIES: ClassVar[frozenset[GroundingCapability]] = frozenset(
        cast(tuple[GroundingCapability, ...], get_args(GroundingCapability))
    )

    provider_id: ClassVar[str]
    version: ClassVar[str] = "unknown"

    capabilities: ClassVar[frozenset[GroundingCapability]] = frozenset()
    requires_initialization: ClassVar[bool] = False

    def __init__(self) -> None:
        """
        Providers must be default-constructible.

        Perform provider-specific setup in initialize() instead.
        """
        super().__init__()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        # Abstract base classes are not concrete providers,
        # and therefore, are exempt from provider metadata validation.
        if isabstract(cls):
            return

        provider_id = getattr(cls, "provider_id", None)

        if not isinstance(provider_id, str):
            raise TypeError(f"{cls.__name__}.provider_id must be a string.")

        provider_id = provider_id.strip()

        if not provider_id:
            raise TypeError(f"{cls.__name__}.provider_id cannot be empty.")

        if not cls._IDENTIFIER_PATTERN.fullmatch(provider_id):
            raise TypeError(
                f"{cls.__name__}.provider_id must be a lowercase identifier "
                "(letters, digits and underscores)."
            )

        version = getattr(cls, "version", None)

        if not isinstance(version, str):
            raise TypeError(f"{cls.__name__}.version must be a string.")

        if not version.strip():
            raise TypeError(f"{cls.__name__}.version cannot be empty.")

        invalid = cls.capabilities - cls._VALID_CAPABILITIES

        if invalid:
            raise TypeError(
                f"{cls.__name__} defines invalid capabilities: "
                f"{sorted(invalid)}."
            )

        signature = inspect.signature(cls.__init__)

        parameters = list(signature.parameters.values())[1:]  # Skip 'self'

        if parameters:
            raise TypeError(
                f"{cls.__name__}.__init__() must not declare constructor "
                "parameters. Use initialize() instead."
            )

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Optional synchronous initialization.

        Examples:
            - load local models
            - create HTTP sessions
            - warm caches
        """
        return None

    async def ainitialize(self) -> None:
        """
        Optional asynchronous initialization.
        """

        await asyncio.to_thread(self.initialize)

    def close(self) -> None:
        """
        Optional synchronous cleanup.
        """

        return None

    async def aclose(self) -> None:
        """
        Optional asynchronous cleanup.
        """

        await asyncio.to_thread(self.close)

    # ------------------------------------------------------------------
    # Grounding API
    # ------------------------------------------------------------------

    @abstractmethod
    def locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        """
        Locate one or more UI elements.

        Returns:
            GroundingResponse

        Must never return None.
        """
        raise NotImplementedError

    @abstractmethod
    async def alocate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        """
        Async equivalent of locate().
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    @property
    def provider(self) -> str:
        return type(self).provider_id

    @property
    def provider_version(self) -> str:
        return type(self).version

    def supports(
        self,
        capability: GroundingCapability,
    ) -> bool:
        """
        Returns whether the provider supports
        a specific capability.
        """

        return capability in type(self).capabilities

    def supports_all(
        self,
        *capabilities: GroundingCapability,
    ) -> bool:
        return all(
            capability in type(self).capabilities
            for capability in capabilities
        )

    def supports_any(
        self,
        *capabilities: GroundingCapability,
    ) -> bool:
        return any(
            capability in type(self).capabilities
            for capability in capabilities
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Optional provider health check.

        Override in remote providers.
        """

        return True

    async def ahealth_check(self) -> bool:
        """
        Async health check.
        """

        return await asyncio.to_thread(self.health_check)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        await self.ainitialize()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        capability_names: list[str] = sorted(type(self).capabilities)

        return (
            f"{type(self).__name__}("
            f"provider='{type(self).provider_id}', "
            f"version='{type(self).version}', "
            f"capabilities={capability_names!r}"
            f")"
        )
