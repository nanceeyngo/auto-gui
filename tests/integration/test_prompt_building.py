"""
Integration tests for VLMRunGroundingProvider prompt construction.
"""

import pytest

from grounding.models import GroundingRequest
from grounding.providers.vlmrun import (
    VLMRunGroundingProvider,
    VLMRunSettings,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def provider() -> VLMRunGroundingProvider:
    provider = VLMRunGroundingProvider()

    provider._settings = VLMRunSettings(
        api_key="dummy",  # type: ignore[arg-type]
    )

    return provider


@pytest.fixture
def grounding_request() -> GroundingRequest:
    return GroundingRequest(
        image="dummy.png",
        query="Login button",
    )


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


def test_prompt_contains_query(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    prompt = provider._build_user_prompt(grounding_request)

    assert grounding_request.query in prompt


def test_prompt_mentions_bounding_box(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    prompt = provider._build_user_prompt(grounding_request)

    assert "bounding" in prompt.lower()
    assert "box" in prompt.lower()


def test_prompt_mentions_confidence_when_confidence_threshold_is_not_none(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    grounding_request.confidence_threshold = 0.78
    prompt = provider._build_user_prompt(grounding_request)

    assert "confidence" in prompt.lower()


def test_prompt_does_not_mention_confidence_when_confidence_threshold_is_none(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    grounding_request.confidence_threshold = None
    prompt = provider._build_user_prompt(grounding_request)

    assert "confidence" not in prompt.lower()


def test_prompt_requests_json(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    prompt = provider._build_user_prompt(grounding_request)

    assert "json" in prompt.lower()


# ---------------------------------------------------------------------------
# Query handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "Login button",
        "Search icon",
        "Username field",
        "⚙ Settings",
        "Save",
        "こんにちは",
    ],
)
def test_prompt_embeds_query_verbatim(
    provider: VLMRunGroundingProvider,
    query: str,
):
    request = GroundingRequest(
        image="dummy.png",
        query=query,
    )

    prompt = provider._build_user_prompt(request)

    assert query in prompt


def test_prompt_contains_query_once(
    provider: VLMRunGroundingProvider,
):
    request = GroundingRequest(
        image="dummy.png",
        query="Login button",
    )

    prompt = provider._build_user_prompt(request)

    assert prompt.count("Login button") == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_prompt_is_deterministic(
    provider: VLMRunGroundingProvider,
    grounding_request: GroundingRequest,
):
    prompt1 = provider._build_user_prompt(grounding_request)
    prompt2 = provider._build_user_prompt(grounding_request)

    assert prompt1 == prompt2


def test_different_queries_produce_different_prompts(
    provider: VLMRunGroundingProvider,
):
    request1 = GroundingRequest(
        image="dummy.png",
        query="Login button",
    )

    request2 = GroundingRequest(
        image="dummy.png",
        query="Search box",
    )

    prompt1 = provider._build_user_prompt(request1)
    prompt2 = provider._build_user_prompt(request2)

    assert prompt1 != prompt2
