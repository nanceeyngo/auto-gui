"""
Shared base implementation for grounding providers.
"""

import asyncio
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from time import perf_counter, sleep
from typing import Any, ClassVar, final

from ..config import GroundingSettings
from ..exceptions import (
    GroundingInitializationError,
    GroundingProviderError,
    GroundingRequestError,
    GroundingShutdownError,
)
from ..interfaces import GroundingEngine
from ..models import (
    BoundingBox,
    DetectionStatus,
    GroundingDetection,
    GroundingRequest,
    GroundingResponse,
)

_MS_PER_SECOND = 1000.0


class BaseGroundingProvider(GroundingEngine):
    """
    Base implementation shared by all grounding providers.

    Responsibilities:
        - lifecycle management
        - request validation
        - automatic initialization
        - execution timing
        - response helpers
        - exception translation

    Subclasses should implement _locate() and _alocate().
    Other lifecycle hooks may also be overridden when required.
    """

    __slots__ = (
        "_initialized",
        "_settings",
    )

    retry_backoff: ClassVar[float] = 0.0
    retriable_errors: ClassVar[tuple[type[GroundingProviderError], ...]] = ()

    def __init__(self) -> None:
        super().__init__()
        self._settings: GroundingSettings = GroundingSettings()
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._initialized

    @final
    def initialize(self) -> None:
        if self._initialized:
            return

        try:
            self._initialize()
        except GroundingInitializationError:
            raise
        except Exception as exc:
            raise GroundingInitializationError(
                f"Failed to initialize provider '{self.provider}': {exc}."
            ) from exc

        self._initialized = True

    @final
    def close(self) -> None:
        if not self._initialized:
            return

        try:
            self._close()
        except GroundingShutdownError:
            raise
        except Exception as exc:
            raise GroundingShutdownError(
                f"Failed to shut down provider '{self.provider}': {exc}."
            ) from exc

        self._initialized = False

    # ------------------------------------------------------------------
    # Public Grounding API (Template Method)
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if type(self).requires_initialization and not self.initialized:
            self.initialize()

    async def _aensure_initialized(self) -> None:
        await asyncio.to_thread(self._ensure_initialized)

    @staticmethod
    def _record_elapsed_time(
        response: GroundingResponse,
        started: float,
    ) -> None:
        response.metadata.setdefault(
            "elapsed_ms",
            (perf_counter() - started) * _MS_PER_SECOND,
        )

    @final
    def locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        self._validate_request(request)

        self._ensure_initialized()

        started = perf_counter()

        try:
            response = self._execute(lambda: self._locate(request))
        except GroundingProviderError:
            raise
        except Exception as exc:
            raise GroundingProviderError(
                f"{self.provider} failed while processing the request.: {exc}"
            ) from exc

        self._record_elapsed_time(response, started)
        return response

    @final
    async def alocate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        self._validate_request(request)

        await self._aensure_initialized()

        started = perf_counter()

        try:
            response = await self._aexecute(lambda: self._alocate(request))
        except GroundingProviderError:
            raise
        except Exception as exc:
            raise GroundingProviderError(
                f"{self.provider} failed while processing the request: {exc}."
            ) from exc

        self._record_elapsed_time(response, started)
        return response

    # ------------------------------------------------------------------
    # Retry configuration hooks
    # ------------------------------------------------------------------

    def _is_retryable(self, exc: GroundingProviderError) -> bool:
        return isinstance(exc, type(self).retriable_errors)

    def _sleep_before_retry(self) -> None:
        """
        Wait before the next retry attempt.

        The default implementation uses a fixed backoff.
        Subclasses may override to implement exponential
        backoff, jitter, etc.
        """
        if type(self).retry_backoff > 0:
            sleep(type(self).retry_backoff)

    async def _asleep_before_retry(self) -> None:
        if type(self).retry_backoff > 0:
            await asyncio.sleep(type(self).retry_backoff)

    def _execute(
        self, operation: Callable[[], GroundingResponse]
    ) -> GroundingResponse:
        attempts = self._settings.provider_max_retries + 1

        for attempt in range(attempts):
            try:
                return operation()

            except GroundingProviderError as exc:
                if attempt + 1 == attempts or not self._is_retryable(exc):
                    raise

                self._sleep_before_retry()

        raise AssertionError("unreachable")

    async def _aexecute(
        self, operation: Callable[[], Awaitable[GroundingResponse]]
    ) -> GroundingResponse:
        attempts = self._settings.provider_max_retries + 1

        for attempt in range(attempts):
            try:
                return await operation()

            except GroundingProviderError as exc:
                if attempt + 1 == attempts or not self._is_retryable(exc):
                    raise

                await self._asleep_before_retry()

        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    @final
    def health_check(self) -> bool:
        return self._health_check()

    # ------------------------------------------------------------------
    # Extension hooks for subclasses
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """
        Optional synchronous initialization hook.
        """
        return None

    async def _ainitialize(self) -> None:
        return await asyncio.to_thread(self._initialize)

    def _close(self) -> None:
        """
        Optional synchronous shutdown hook.
        """
        return None

    @abstractmethod
    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        """
        Provider-specific synchronous grounding implementation.
        """

    @abstractmethod
    async def _alocate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        """
        Provider-specific asynchronous grounding implementation.
        """

    @staticmethod
    def _health_check() -> bool:
        return True

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: GroundingRequest) -> None:
        if not request.query.strip():
            raise GroundingRequestError("Grounding query cannot be empty.")

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
    @final
    def make_bounding_box(
        bbox: Sequence[float],
        image_width: int,
        image_height: int,
    ) -> BoundingBox:
        if len(bbox) != 4:
            raise GroundingRequestError(
                "Bounding box must have 4 coordinates."
            )
        x, y, width, height = bbox

        x1 = round(x * image_width)
        y1 = round(y * image_height)
        x2 = round((x + width) * image_width)
        y2 = round((y + height) * image_height)

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    @final
    def make_detection(
        *,
        bounding_box: BoundingBox,
        confidence: float | None = None,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GroundingDetection:
        return GroundingDetection(
            bbox=bounding_box,
            confidence=confidence,
            label=label,
            metadata=metadata or {},
        )

    @final
    def make_response(
        self,
        *,
        request: GroundingRequest,
        detections: list[GroundingDetection],
        metadata: dict[str, Any] | None = None,
    ) -> GroundingResponse:
        num_detections = len(detections)
        if metadata and "error" in metadata:
            response_status = DetectionStatus.ERROR
        elif request.confidence_threshold is not None and any(
            detection.confidence < request.confidence_threshold
            for detection in detections
            if detection.confidence is not None
        ):
            response_status = DetectionStatus.AMBIGUOUS
        elif num_detections > 0:
            response_status = DetectionStatus.SUCCESS
        else:
            response_status = DetectionStatus.NO_MATCH

        response = GroundingResponse(
            provider=self.provider,
            provider_version=self.provider_version,
            request_query=request.query,
            detections=detections,
            status=response_status,
            metadata=metadata or {},
        )

        return response

    @final
    def empty_response(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return self.make_response(
            request=request,
            detections=[],
        )

    @property
    def settings(self) -> GroundingSettings:
        return self._settings
