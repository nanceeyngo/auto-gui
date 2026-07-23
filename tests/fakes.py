"""
Reusable test doubles for the grounding framework.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any

from grounding.config import GroundingSettings
from grounding.exceptions import GroundingProviderError
from grounding.models import (
    BoundingBox,
    GroundingRequest,
    GroundingResponse,
)
from grounding.providers.base import BaseGroundingProvider


class ThreadedAsyncMixin(ABC):
    __slots__ = ()

    @abstractmethod
    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse: ...

    async def _alocate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return await asyncio.to_thread(
            self._locate,
            request,
        )


class FakeGroundingProvider(ThreadedAsyncMixin, BaseGroundingProvider):
    """
    Simple provider that always returns one detection.

    Used for testing the registry and client.
    """

    __slots__ = ()

    provider_id = "fake"
    version = "1.0.0"

    requires_initialization = False

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return self.make_response(
            request=request,
            detections=[
                self.make_detection(
                    bounding_box=BoundingBox(
                        x1=10,
                        y1=20,
                        x2=110,
                        y2=120,
                    ),
                    confidence=0.95,
                    label=request.query,
                )
            ],
        )


class EmptyGroundingProvider(ThreadedAsyncMixin, BaseGroundingProvider):
    """
    Provider that always returns no detections.
    """

    __slots__ = ()

    provider_id = "empty"
    version = "1.0.0"

    requires_initialization = False

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return self.empty_response(request)


class InitializableGroundingProvider(
    ThreadedAsyncMixin, BaseGroundingProvider
):
    """
    Provider used to verify initialization and shutdown hooks.
    """

    __slots__ = ("close_calls", "initialize_calls")

    provider_id = "initializable"
    version = "1.0.0"

    requires_initialization = True

    def __init__(self) -> None:
        super().__init__()
        self.initialize_calls: int = 0
        self.close_calls: int = 0

    def _initialize(self) -> None:
        self.initialize_calls += 1

    def _close(self) -> None:
        self.close_calls += 1

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return self.empty_response(request)


class FailingGroundingProvider(ThreadedAsyncMixin, BaseGroundingProvider):
    """
    Provider that always fails.

    Useful for testing exception translation.
    """

    __slots__ = ()

    provider_id = "failing"
    version = "1.0.0"

    requires_initialization = False

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        raise RuntimeError("Simulated provider failure.")


class SlowGroundingProvider(ThreadedAsyncMixin, BaseGroundingProvider):
    """
    Provider that intentionally sleeps before returning.

    Useful for testing timeout handling and latency measurement.
    """

    __slots__ = ("_timeout_duration",)

    provider_id = "slow"
    version = "1.0.0"

    requires_initialization = False

    def __init__(self) -> None:
        super().__init__()
        self._timeout_duration: float = 0.2
        self._should_return_empty_response: bool = True

    @property
    def timeout_duration(self) -> float:
        return self._timeout_duration

    def configure_timeout_duration(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("timeout_duration must be positive")

        self._timeout_duration = value

    @property
    def should_return_empty_response(self) -> bool:
        return self._should_return_empty_response

    def configure_should_return_empty_response(self, value: bool) -> None:
        self._should_return_empty_response = value

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        time.sleep(self._timeout_duration)
        if self.should_return_empty_response:
            return self.empty_response(request)

        return self.make_response(
            request=request,
            detections=[
                self.make_detection(
                    bounding_box=BoundingBox(
                        x1=10,
                        y1=20,
                        x2=110,
                        y2=120,
                    ),
                    confidence=0.95,
                    label=request.query,
                )
            ],
        )


class ConfigurableGroundingProvider(ThreadedAsyncMixin, BaseGroundingProvider):
    """
    Provider that always returns a preconfigured response.

    Useful for testing client behavior under arbitrary provider
    responses.
    """

    __slots__ = ("_response",)

    provider_id = "configurable"
    version = "1.0.0"

    requires_initialization = False

    def __init__(self) -> None:
        super().__init__()
        self._response: GroundingResponse | None = None

    @property
    def response(self) -> GroundingResponse | None:
        return self._response

    def configure_response(
        self,
        response: GroundingResponse,
    ) -> None:
        self._response = response

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        if self.response is None:
            raise RuntimeError("No response has been configured.")

        return self.response


class FlakyGroundingProvider(
    ThreadedAsyncMixin,
    BaseGroundingProvider,
):
    """
    Provider that fails a configurable number of times before returning
    a configured or default successful response.

    Useful for testing retry logic.
    """

    __slots__ = (
        "_calls",
        "_exception",
        "_failures_before_success",
        "_response",
        "_settings",
    )

    provider_id = "flaky"
    version = "1.0.0"
    retriable_errors = (GroundingProviderError,)
    retry_backoff = 0.1

    requires_initialization = False

    def __init__(self) -> None:
        super().__init__()

        self._failures_before_success: int = 1
        self._calls: int = 0
        self._settings: GroundingSettings = GroundingSettings()

        self._response: GroundingResponse | None = None
        self._exception: Exception = GroundingProviderError(
            "Simulated transient provider failure."
        )

    @property
    def failures_before_success(self) -> int:
        return self._failures_before_success

    def configure_failures_before_success(self, value: int) -> None:
        if value < 0:
            raise ValueError("`failures_before_success` must be non-negative.")

        self._failures_before_success = value

    @property
    def provider_max_retries(self) -> int:
        return self._settings.provider_max_retries

    def configure_provider_max_retries(self, value: int) -> None:
        if value < 0:
            raise ValueError("`provider_max_retries` must be non-negative.")

        current_settings: dict[str, Any] = self._settings.model_dump()
        current_settings.update(provider_max_retries=value)
        self._settings = GroundingSettings(**current_settings)

    @property
    def response(self) -> GroundingResponse | None:
        return self._response

    def configure_response(
        self,
        response: GroundingResponse,
    ) -> None:
        self._response = response

    def configure_success_response(
        self,
        request: GroundingRequest,
    ) -> None:
        self._response = self.make_response(
            request=request,
            detections=[
                self.make_detection(
                    bounding_box=BoundingBox(
                        x1=10,
                        y1=20,
                        x2=110,
                        y2=120,
                    ),
                    confidence=0.95,
                    label=request.query,
                )
            ],
        )

    @property
    def exception(self) -> Exception:
        return self._exception

    def configure_exception(
        self,
        exception: Exception,
    ) -> None:
        self._exception = exception

    @property
    def calls(self) -> int:
        return self._calls

    def reset(self) -> None:
        self._calls = 0

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        self._calls += 1

        if self._calls <= self.failures_before_success:
            raise self.exception

        if self.response is not None:
            return self.response

        return self.make_response(
            request=request,
            detections=[
                self.make_detection(
                    bounding_box=BoundingBox(
                        x1=10,
                        y1=20,
                        x2=110,
                        y2=120,
                    ),
                    confidence=0.95,
                    label=request.query,
                )
            ],
        )


class CountingGroundingProvider(
    ThreadedAsyncMixin,
    BaseGroundingProvider,
):
    """
    Provider that counts the number of times _locate() is invoked.

    Useful for verifying retry behavior, ensuring that each attempt
    results in exactly one call to _locate().
    """

    __slots__ = ("_calls",)

    provider_id = "counting"
    version = "1.0.0"

    requires_initialization = False

    def __init__(self) -> None:
        super().__init__()
        self._calls: int = 0

    @property
    def calls(self) -> int:
        return self._calls

    def reset(self) -> None:
        """
        Reset the call counter.
        """
        self._calls = 0

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        self._calls += 1

        return self.make_response(
            request=request,
            detections=[
                self.make_detection(
                    bounding_box=BoundingBox(
                        x1=10,
                        y1=20,
                        x2=110,
                        y2=120,
                    ),
                    confidence=1.0,
                    label=request.query,
                )
            ],
        )


class FailingInitializationProvider(
    ThreadedAsyncMixin,
    BaseGroundingProvider,
):
    """
    Provider that always raises an exception when the `initialize()` method
    is called.

    Useful for verifying initialization failure handling,
    including failure error wrapping and surfacing
    """

    provider_id = "failing_init"
    version = "1.0.0"

    requires_initialization = True

    def _initialize(self) -> None:
        raise RuntimeError("Boom! Initialization failed.")

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return self.empty_response(request)


__all__ = [
    "ConfigurableGroundingProvider",
    "CountingGroundingProvider",
    "EmptyGroundingProvider",
    "FailingGroundingProvider",
    "FailingInitializationProvider",
    "FakeGroundingProvider",
    "FlakyGroundingProvider",
    "InitializableGroundingProvider",
    "SlowGroundingProvider",
]
