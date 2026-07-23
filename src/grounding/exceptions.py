"""
Exceptions used throughout the grounding framework.
"""

from collections.abc import Mapping
from types import MappingProxyType


class GroundingError(Exception):
    """
    Base exception for all grounding-related errors.
    """


# ----------------------------------------------------------------------
# Configuration / lifecycle
# ----------------------------------------------------------------------


class GroundingInitializationError(GroundingError):
    """
    Raised when a grounding provider cannot be initialized.
    """


class GroundingShutdownError(GroundingError):
    """
    Raised when a grounding provider cannot be shut down cleanly.
    """


# ----------------------------------------------------------------------
# Requests
# ----------------------------------------------------------------------


class GroundingRequestError(GroundingError):
    """
    Raised when a grounding request is invalid.
    """


class UnsupportedCapabilityError(GroundingRequestError):
    """
    Raised when a provider does not support a requested capability.
    """


# ----------------------------------------------------------------------
# Provider execution
# ----------------------------------------------------------------------


class GroundingProviderError(GroundingError):
    """
    Raised when a provider fails while processing a request.
    """


class GroundingConnectionError(GroundingProviderError):
    """
    Raised when communication with a remote grounding provider fails.

    Examples:
        - DNS resolution failure
        - Connection refused
        - TLS handshake failure
        - HTTP transport error
    """


class GroundingTimeoutError(GroundingProviderError):
    """
    Raised when a grounding operation exceeds its timeout.
    """


class GroundingCancelledError(GroundingProviderError):
    """
    Raised when a grounding operation is cancelled.
    """


# ----------------------------------------------------------------------
# Responses
# ----------------------------------------------------------------------


class GroundingResponseError(GroundingError):
    """
    Raised when a provider returns an invalid response.
    """


class NoGroundingResultError(GroundingResponseError):
    """
    Raised when no grounding result could be produced.
    """


class InvalidGroundingResultError(GroundingResponseError):
    """
    Raised when a provider returns malformed grounding data.
    """


# ----------------------------------------------------------------------
# Registry / client
# ----------------------------------------------------------------------


class UnknownGroundingProviderError(GroundingError):
    """
    Raised when a provider identifier is not registered.
    """


class GroundingFallbackError(GroundingError):
    """
    Raised when all configured providers fail.
    """

    def __init__(
        self,
        message: str,
        failures: Mapping[str, GroundingError],
    ):
        super().__init__(message)
        self.message: str = message
        self.failures: Mapping[str, GroundingError] = MappingProxyType(
            dict(failures)
        )

    def __str__(self) -> str:
        providers = ", ".join(sorted(self.failures))
        return f"{super().__str__()} Failed providers: {providers}"


class NoDefaultGroundingProviderError(GroundingError):
    """
    Raised when no default grounding provider is configured,
    and provider retrieval is attempted without specifying a provider id.
    """


__all__ = [
    "GroundingCancelledError",
    "GroundingConnectionError",
    "GroundingError",
    "GroundingFallbackError",
    "GroundingInitializationError",
    "GroundingProviderError",
    "GroundingRequestError",
    "GroundingResponseError",
    "GroundingShutdownError",
    "GroundingTimeoutError",
    "InvalidGroundingResultError",
    "NoDefaultGroundingProviderError",
    "NoGroundingResultError",
    "UnknownGroundingProviderError",
    "UnsupportedCapabilityError",
]
