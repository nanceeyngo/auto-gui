"""
VLM Run grounding provider.
"""

import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

from dotenv import find_dotenv
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict
from vlmrun.client import VLMRun
from vlmrun.client.exceptions import (
    APIError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    VLMRunError,
)
from vlmrun.client.types import CreditUsage, PredictionResponse
from vlmrun.common.image import encode_image

from ..config import GroundingSettings, RemoteProviderSettings
from ..exceptions import (
    GroundingProviderError,
    GroundingRequestError,
    GroundingTimeoutError,
    InvalidGroundingResultError,
)
from ..logging_utils import get_logger
from ..models import (
    GroundingDetection,
    GroundingRequest,
    GroundingResponse,
    ImageLike,
)
from .base import BaseGroundingProvider

logger = get_logger("grounding.providers.vlmrun")

type DataUrlString = str | bytes
type PromptMessages = list[ChatCompletionMessageParam]
StreamingImageFormats = Literal["PNG", "JPEG", "binary"]
_DEFAULT_IMAGE_FORMAT: StreamingImageFormats = "PNG"


class VLMRunSettings(RemoteProviderSettings):
    model_config = SettingsConfigDict(
        env_prefix="VLMRUN_",
        env_file=find_dotenv(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    base_url: str = "https://api.vlm.run/v1/openai"
    model: str = "vlmrun-orion-2:auto"
    api_key: SecretStr | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    system_prompt_filepath: Path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "orion_grounding_system_prompt.md"
    )
    grounding_schema_filepath: Path = (
        Path(__file__).resolve().parent.parent
        / "schemas"
        / "orion_grounding.schema.json"
    )


class VLMRunGroundingProvider(BaseGroundingProvider):
    """
    Grounding provider backed by VLM Run.
    """

    __slots__ = (
        "_client",
        "_settings",
    )

    provider_id = "vlmrun"
    version = version("vlmrun")

    capabilities = frozenset(
        {
            "confidence_scores",
            "icon_detection",
            "multi_detection",
            "natural_language_query",
            "remote_api",
            "semantic_grounding",
            "screen_parsing",
        }
    )

    requires_initialization = True

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._settings: GroundingSettings = VLMRunSettings()
        self._client: VLMRun | None = None

    def _initialize(self) -> None:
        self._client = VLMRun(
            api_key=(
                self.settings.api_key.get_secret_value()
                if self.settings.api_key is not None
                else None
            ),
            base_url=self.settings.base_url,
            timeout=self.settings.request_timeout,
            max_retries=self.settings.transport_max_retries,
        )

    def _close(self) -> None:
        self._client = None

    @property
    def client(self) -> VLMRun:
        assert self._client is not None
        return self._client

    @staticmethod
    def _load_prompt(prompt_filepath: str | Path) -> str:
        if not isinstance(prompt_filepath, Path):
            prompt_filepath = Path(prompt_filepath)

        try:
            prompt = prompt_filepath.read_text()
        except (
            FileNotFoundError,
            IsADirectoryError,
            PermissionError,
            UnicodeError,
        ) as exc:
            raise GroundingRequestError(
                f"Failed to load prompt from {prompt_filepath}: {exc}"
            ) from exc

        return prompt

    @staticmethod
    def _inject_schema_into_prompt(
        schema_path: str | Path, prompt_template: str, **kwargs: Any
    ) -> str:
        """
        Reads a JSON schema file, minifies it to minimize token bloat,
        and injects it into a prompt template using a dynamically named
        placeholder argument.
        """
        try:
            # Load and minify the schema
            with open(schema_path) as f:
                schema_data = json.load(f)
            minified_schema = json.dumps(schema_data, separators=(",", ":"))

            template_args = {**kwargs}

            # Validate that the user provided exactly one placeholder key
            # for the schema
            if not template_args or len(template_args) != 1:
                raise ValueError(
                    "Schema placeholder name must be provided as a "
                    "keyword argument (e.g., output_schema=...)"
                )

            # Extract the placeholder name from `template_args`
            placeholder_name = next(iter(template_args.keys()))

            # Safely replace only the specified placeholder in
            # the prompt template without touching any other ones
            pattern = r"\{" + re.escape(placeholder_name) + r"\}"
            final_prompt, substitution_count = re.subn(
                pattern, minified_schema.replace("\\", "\\\\"), prompt_template
            )

            # Verify that the schema was actually injected
            if substitution_count == 0:
                raise ValueError(
                    "Injection failed: The placeholder "
                    f"'{{{placeholder_name}}}' was not found in the "
                    "provided prompt template."
                )

            return final_prompt

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Schema file not found at: {schema_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON syntax in schema file: {exc}") from exc

    @staticmethod
    def _build_user_prompt(
        request: GroundingRequest,
    ) -> str:
        parts = [
            f"Locate the UI element described as '{request.query}'.",
            f"Return no more than {request.top_k} matching bounding boxes.",
        ]

        if request.confidence_threshold is not None:
            parts.append(
                "Discard any detection with confidence below "
                f"{request.confidence_threshold}."
            )

        parts.append("Return only the JSON object specified by the system prompt.")

        return " ".join(parts)

    @staticmethod
    def _get_image_size(image: ImageLike) -> tuple[int, int]:
        """
        Returns the image dimensions in pixels.
        """

        if isinstance(image, Image.Image):
            return image.size

        try:
            with Image.open(image) as opened_image:
                return opened_image.size
        except OSError as exc:
            raise GroundingRequestError(
                f"Failed to load image from {image!r}: {exc}"
            ) from exc

    @staticmethod
    def _encode_image(
        image: ImageLike,
        *,
        image_format: StreamingImageFormats = _DEFAULT_IMAGE_FORMAT,
    ) -> DataUrlString:
        """
        Returns a complete `data:image/<format>;base64,<payload>` URI.

        NOTE: `vlmrun.common.image.encode_image` already returns a
        full data URL (not a bare base64 payload) when `image_format`
        is "PNG" or "JPEG". Callers must use the return value as-is;
        re-prefixing it with a data URL scheme produces a corrupted,
        doubly-prefixed URL that the model cannot decode.
        """
        if isinstance(image, Image.Image):
            data_url: DataUrlString = encode_image(image, format=image_format)
        else:
            try:
                with Image.open(image) as open_image:
                    data_url = encode_image(open_image, format=image_format)
            except OSError as exc:
                raise GroundingRequestError(
                    f"Failed to load image from {image!r}: {exc}"
                ) from exc

        return data_url

    def _build_messages(
        self,
        request: GroundingRequest,
    ) -> PromptMessages:
        system_prompt_template = self._load_prompt(self.settings.system_prompt_filepath)
        system_prompt = self._inject_schema_into_prompt(
            self.settings.grounding_schema_filepath,
            system_prompt_template,
            output_json_schema="",
        )
        user_prompt = self._build_user_prompt(request)
        data_url = self._encode_image(request.image)

        # `data_url` is already a complete `data:image/...;base64,...`
        # URI (see `_encode_image`'s docstring) and must be used
        # as-is. Two bugs were previously stacked here: (1) the SDK's
        # already-complete data URL was concatenated onto a second,
        # hardcoded `data:image/png;base64,` prefix, and (2) the
        # payload was embedded via `!r` (Python repr), which wraps it
        # in quotes/escapes. Either bug alone corrupts every image
        # sent to the model; together they produced a doubly-prefixed,
        # repr-quoted, completely undecodable URL.
        if isinstance(data_url, bytes):
            data_url = data_url.decode("ascii")

        messages: PromptMessages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(type="text", text=user_prompt),
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={"url": data_url},
                    ),
                ],
            ),
        ]

        return messages

    def _submit_prediction(
        self,
        messages: PromptMessages,
    ) -> PredictionResponse:
        started = perf_counter()

        try:
            raw_response = self.client.agent.completions.create(
                model=self.settings.model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                response_format={"type": "json_object"},
            )

            response_content = raw_response.choices[0].message.content

            prediction = PredictionResponse(
                id=raw_response.id,
                status="completed",
                created_at=datetime.fromtimestamp(raw_response.created),
                completed_at=datetime.now(tz=UTC),
                usage=CreditUsage(element_type="image"),
                response=response_content,
            )

            elapsed_ms = (perf_counter() - started) * 1000.0
            logger.debug(
                "vlmrun prediction completed",
                extra={
                    "context": {
                        "backend": self.provider_id,
                        "model": self.settings.model,
                        "prediction_id": raw_response.id,
                        "elapsed_ms": round(elapsed_ms, 2),
                    }
                },
            )

            return prediction

        except RequestTimeoutError as exc:
            logger.warning(
                "vlmrun request timed out",
                extra={
                    "context": {
                        "backend": self.provider_id,
                        "model": self.settings.model,
                    }
                },
            )
            raise GroundingTimeoutError(
                "Timed out while communicating with VLM Run."
            ) from exc

        except (APIError, RateLimitError, ServerError, VLMRunError) as exc:
            logger.error(
                "vlmrun API error",
                extra={
                    "context": {
                        "backend": self.provider_id,
                        "model": self.settings.model,
                        "exception_type": type(exc).__name__,
                    }
                },
            )
            raise GroundingProviderError(f"VLM Run API error: {exc}") from exc

    def _wait_for_prediction(
        self, prediction: PredictionResponse
    ) -> PredictionResponse:
        if prediction.status == "completed":
            return prediction

        return self.client.predictions.wait(prediction.id)

    async def _await_prediction(
        self, prediction: PredictionResponse
    ) -> PredictionResponse:
        return await asyncio.to_thread(self._wait_for_prediction, prediction)

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        if value is None:
            return None

        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None

        if not 0.0 <= confidence <= 1.0:
            return None

        return confidence

    def _parse_prediction(
        self, *, request: GroundingRequest, prediction: PredictionResponse
    ) -> list[GroundingDetection]:
        result = prediction.response

        if not isinstance(result, Mapping):
            raise InvalidGroundingResultError(
                "Prediction response is not a mapping object."
            )

        raw_detections = result.get("detections")

        if raw_detections is None:
            raise InvalidGroundingResultError(
                "Prediction response does not contain a 'detections' field."
            )
        if not isinstance(raw_detections, list):
            raise InvalidGroundingResultError("'detections' must be a list.")

        image_width, image_height = self._get_image_size(request.image)

        detections: list[GroundingDetection] = []

        skipped = 0

        for item in raw_detections:
            if not isinstance(item, Mapping):
                skipped += 1
                continue

            bbox = item.get("bbox")

            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes))
                or len(bbox) != 4
            ):
                skipped += 1
                continue

            if not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in bbox):
                skipped += 1
                continue

            bounding_box = self.make_bounding_box(bbox, image_width, image_height)

            confidence = self._parse_confidence(item.get("confidence"))

            label = item.get("label")

            if not isinstance(label, str):
                label = None

            detections.append(
                self.make_detection(
                    bounding_box=bounding_box,
                    confidence=confidence,
                    label=label,
                )
            )

        if skipped:
            logger.warning(
                "vlmrun response contained malformed detection payloads",
                extra={
                    "context": {
                        "backend": self.provider_id,
                        "skipped_count": skipped,
                        "accepted_count": len(detections),
                    }
                },
            )

        return detections

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        messages = self._build_messages(request)
        prediction = self._submit_prediction(messages)
        prediction = self._wait_for_prediction(prediction)
        detections = self._parse_prediction(request=request, prediction=prediction)

        return self.make_response(
            request=request,
            detections=detections,
        )

    async def _alocate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        return await asyncio.to_thread(
            self._locate,
            request,
        )

    @property
    def settings(self) -> VLMRunSettings:
        return cast(VLMRunSettings, self._settings)
