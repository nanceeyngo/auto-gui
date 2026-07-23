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
from ..models import (
    GroundingDetection,
    GroundingRequest,
    GroundingResponse,
    ImageLike,
)
from .base import BaseGroundingProvider

type Base64String = str | bytes
type PromptMessages = list[ChatCompletionMessageParam]
StreamingImageFormats = Literal["PNG", "JPEG", "binary"]
_IMAGE_DATA_URL_PREFIX = "data:image/png;base64,"
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
            raise FileNotFoundError(
                f"Schema file not found at: {schema_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON syntax in schema file: {exc}"
            ) from exc

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

        parts.append(
            "Return only the JSON object specified by the system prompt."
        )

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
    ) -> Base64String:
        if isinstance(image, Image.Image):
            base64_str: Base64String = encode_image(image, format=image_format)
        else:
            try:
                with Image.open(image) as open_image:
                    base64_str = encode_image(open_image, format=image_format)
            except OSError as exc:
                raise GroundingRequestError(
                    f"Failed to load image from {image!r}: {exc}"
                ) from exc

        return base64_str

    def _build_messages(
        self,
        request: GroundingRequest,
    ) -> PromptMessages:
        system_prompt_template = self._load_prompt(
            self.settings.system_prompt_filepath
        )
        system_prompt = self._inject_schema_into_prompt(
            self.settings.grounding_schema_filepath,
            system_prompt_template,
            output_json_schema="",
        )
        user_prompt = self._build_user_prompt(request)
        base64_str = self._encode_image(request.image)

        messages: PromptMessages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(
                        type="text", text=user_prompt
                    ),
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={
                            "url": f"{_IMAGE_DATA_URL_PREFIX}{base64_str!r}"
                        },
                    ),
                ],
            ),
        ]

        return messages

    def _submit_prediction(
        self,
        messages: PromptMessages,
    ) -> PredictionResponse:
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

            return prediction

        except RequestTimeoutError as exc:
            raise GroundingTimeoutError(
                "Timed out while communicating with VLM Run."
            ) from exc

        except (APIError, RateLimitError, ServerError, VLMRunError) as exc:
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

        for item in raw_detections:
            if not isinstance(item, Mapping):
                continue

            bbox = item.get("bbox")

            if (
                not isinstance(bbox, Sequence)
                or isinstance(bbox, (str, bytes))
                or len(bbox) != 4
            ):
                continue

            if not all(
                isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in bbox
            ):
                continue

            bounding_box = self.make_bounding_box(
                bbox, image_width, image_height
            )

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

        return detections

    def _locate(
        self,
        request: GroundingRequest,
    ) -> GroundingResponse:
        messages = self._build_messages(request)
        prediction = self._submit_prediction(messages)
        prediction = self._wait_for_prediction(prediction)
        detections = self._parse_prediction(
            request=request, prediction=prediction
        )

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
