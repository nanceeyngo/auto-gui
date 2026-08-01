"""
Vision-language model (VLM) factory utilities.
"""

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import settings


def create_vlm() -> BaseChatModel:
    """
    Create a configured vision-language model.
    """
    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
        max_completion_tokens=settings.max_tokens,
        api_key=(settings.api_key if settings.api_key is not None else None),
        base_url=settings.base_url,
        seed=settings.seed,
    )


@lru_cache(maxsize=1)
def shared_vlm() -> BaseChatModel:
    """
    Return the shared VLM instance.
    """
    return create_vlm()


def clear_vlm_cache() -> None:
    """
    Clear the shared VLM cache.
    """
    shared_vlm.cache_clear()


__all__ = [
    "clear_vlm_cache",
    "create_vlm",
    "shared_vlm",
]
