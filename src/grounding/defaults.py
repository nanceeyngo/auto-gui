from .providers import VLMRunGroundingProvider
from .registry import GroundingRegistry

registry = GroundingRegistry()

registry.register_many(
    [
        VLMRunGroundingProvider,
    ]
)
