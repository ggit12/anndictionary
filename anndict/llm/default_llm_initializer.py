"""
This module defines the default LLM initializer.
- DefaultLLMInitializer: Default implementation with rate limiting

Rate limiting is handled via InMemoryRateLimiter with configurable parameters:
- requests_per_minute: Max requests allowed per minute (default 40)
- check_every_n_seconds: Rate check frequency (default 0.1s)
- max_bucket_size: Maximum number of requests that can be queued

Example

.. code-block:: python

   config = LLMProviderConfig(
       class_name="OpenAI",
       module_path="langchain.llms",
       init_class=DefaultLLMInitializer,
       requests_per_minute=60
   )
   initializer = DefaultLLMInitializer(config)
   constructor_args, kwargs = initializer.initialize({"api_key": "..."})
"""


from typing import Any

from .base_llm_initializer import BaseLLMInitializer


class DefaultLLMInitializer(BaseLLMInitializer):
    """Default initialization logic with rate limiting"""

    def initialize(
        self, constructor_args: dict[str, Any], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rate_limiter = self.create_rate_limiter(constructor_args)
        constructor_args["rate_limiter"] = rate_limiter
        return constructor_args, kwargs
