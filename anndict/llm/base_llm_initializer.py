"""
The base class for managing LLM provider initialization and rate limiting.

The module defines:
- ``LLMProviderConfig``: Configuration dataclass for LLM providers
- ``BaseLLMInitializer``: Abstract base class for provider initialization logic 
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type
from langchain_core.rate_limiters import InMemoryRateLimiter


@dataclass
class LLMProviderConfig:
    """Configuration for LLM provider"""

    class_name: str
    module_path: str
    init_class: Type["BaseLLMInitializer"]
    requests_per_minute: float = 40
    check_every_n_seconds: float = 0.1


class BaseLLMInitializer(ABC):
    """Base class for LLM initialization logic"""

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    def create_rate_limiter(
        self, constructor_args: dict[str, Any]
    ) -> InMemoryRateLimiter:
        """Create a rate limiter with given or default parameters"""
        requests_per_minute = float(
            constructor_args.pop("requests_per_minute", self.config.requests_per_minute)
        )
        requests_per_second = requests_per_minute / 60
        check_every_n_seconds = float(
            constructor_args.pop(
                "check_every_n_seconds", self.config.check_every_n_seconds
            )
        )
        max_bucket_size = float(
            constructor_args.pop("max_bucket_size", requests_per_minute)
        )

        return InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,
            max_bucket_size=max_bucket_size,
        )

    @abstractmethod
    def initialize(
        self, constructor_args: dict[str, Any], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Initialize the LLM provider with given arguments"""
