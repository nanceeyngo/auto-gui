"""
Grounding framework.

A provider-agnostic interface for visual grounding backends.
"""

from .client import GroundingClient
from .config import (
    GroundingSettings,
    LocalProviderSettings,
    RemoteProviderSettings,
)
from .defaults import registry
from .exceptions import (
    GroundingCancelledError,
    GroundingConnectionError,
    GroundingError,
    GroundingFallbackError,
    GroundingInitializationError,
    GroundingProviderError,
    GroundingRequestError,
    GroundingResponseError,
    GroundingShutdownError,
    GroundingTimeoutError,
    InvalidGroundingResultError,
    NoDefaultGroundingProviderError,
    NoGroundingResultError,
    UnknownGroundingProviderError,
    UnsupportedCapabilityError,
)
from .models import (
    BoundingBox,
    DetectionStatus,
    GroundingDetection,
    GroundingFailure,
    GroundingRequest,
    GroundingResponse,
    ImageLike,
)
from .providers import (
    BaseGroundingProvider,
    VLMRunGroundingProvider,
    VLMRunSettings,
)
from .registry import GroundingRegistry

__version__ = "1.0.0"


__all__ = [
    "BaseGroundingProvider",
    "BoundingBox",
    "DetectionStatus",
    "GroundingCancelledError",
    "GroundingClient",
    "GroundingConnectionError",
    "GroundingDetection",
    "GroundingError",
    "GroundingFailure",
    "GroundingFallbackError",
    "GroundingInitializationError",
    "GroundingProviderError",
    "GroundingRegistry",
    "GroundingRequest",
    "GroundingRequestError",
    "GroundingResponse",
    "GroundingResponseError",
    "GroundingSettings",
    "GroundingShutdownError",
    "GroundingTimeoutError",
    "ImageLike",
    "InvalidGroundingResultError",
    "LocalProviderSettings",
    "NoDefaultGroundingProviderError",
    "NoGroundingResultError",
    "RemoteProviderSettings",
    "UnknownGroundingProviderError",
    "UnsupportedCapabilityError",
    "VLMRunGroundingProvider",
    "VLMRunSettings",
    "__version__",
    "registry",
]
