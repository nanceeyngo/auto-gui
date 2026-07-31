"""
Regression tests for bug fixes in the VLM Run grounding provider's
image data-URL construction.

Two stacked bugs were found and fixed here:

1. `vlmrun.common.image.encode_image()` already returns a complete
   `data:image/png;base64,<payload>` URI (not a bare base64 payload),
   but the provider concatenated it onto a second, hardcoded
   `data:image/png;base64,` prefix -- producing a doubly-prefixed,
   undecodable URL.

2. The (doubly-prefixed) payload was then embedded via `!r`
   (`f"...{value!r}"`), which wraps it in Python repr() quoting and
   escaping -- corrupting it further.

Either bug alone breaks every grounding request; stacked together
they made the resulting "image" undecodable by any client.
"""

import base64
import io
import re
from typing import Any, cast

from PIL import Image
from pydantic import SecretStr

from grounding.models import GroundingRequest
from grounding.providers.vlmrun import (
    PromptMessages,
    VLMRunGroundingProvider,
    VLMRunSettings,
)


def make_provider() -> VLMRunGroundingProvider:
    provider = VLMRunGroundingProvider()
    provider._settings = VLMRunSettings(api_key=SecretStr("test-dummy-api-key"))
    return provider


def make_request() -> GroundingRequest:
    return GroundingRequest(
        image=Image.new("RGB", (64, 64), color=(255, 0, 0)),
        query="Submit button",
    )


class TestImageDataUrlEncoding:
    def _extract_image_url(self, messages: PromptMessages) -> str:
        user_message = messages[1]
        assert user_message["role"] == "user"

        content = cast(list[dict[str, Any]], user_message["content"])
        image_part = content[1]
        image_url = cast(dict[str, str], image_part["image_url"])
        return image_url["url"]

    def test_data_url_has_exactly_one_scheme_prefix(self) -> None:
        provider = make_provider()
        request = make_request()

        messages = provider._build_messages(request)

        url = self._extract_image_url(messages)

        assert url.count("data:image/") == 1
        assert url.count(";base64,") == 1

    def test_data_url_does_not_contain_repr_quoting(self) -> None:
        provider = make_provider()
        request = make_request()

        messages = provider._build_messages(request)

        url = self._extract_image_url(messages)

        assert not url.startswith("data:image/png;base64,'")
        assert not url.startswith('data:image/png;base64,b"')
        assert not url.startswith("data:image/png;base64,b'")
        assert "\\x" not in url

    def test_data_url_payload_is_valid_base64(self) -> None:
        provider = make_provider()
        request = make_request()

        messages = provider._build_messages(request)

        url = self._extract_image_url(messages)

        match = re.match(r"^data:image/(png|jpeg);base64,(.+)$", url)
        assert match is not None

        payload = match.group(2)

        decoded = base64.b64decode(payload, validate=True)
        assert decoded.startswith(b"\x89PNG") or decoded[:2] == b"\xff\xd8"

    def test_decoded_image_round_trips(self) -> None:
        provider = make_provider()
        request = make_request()

        messages = provider._build_messages(request)
        url = self._extract_image_url(messages)
        payload = url.split(",", 1)[1]

        decoded_bytes = base64.b64decode(payload, validate=True)
        image = Image.open(io.BytesIO(decoded_bytes))

        assert image.size == (64, 64)
