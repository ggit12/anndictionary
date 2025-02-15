"""
Manage interactions with LLMs across providers.
Provides functional configuration, initialization, 
and response processing/retry/fallback mechanisms for making LLM calls
through a unified interface that works with OpenAI, Anthropic, Amazon Bedrock, Google, and others.
"""
from functools import wraps

from typing import Any
from .llm_manager import LLMManager

@wraps(LLMManager.configure_llm_backend)
def configure_llm_backend(
    provider: str,
    model: str,
    **kwargs
) -> None:
    """Wrapper for LLMManager.configure_llm_backend"""
    LLMManager.configure_llm_backend(provider, model, **kwargs)


@wraps(LLMManager.get_llm_config)
def get_llm_config() -> dict[str, Any]:
    """Wrapper for LLMManager.get_llm_config"""
    return LLMManager.get_llm_config()


@wraps(LLMManager.call_llm)
def call_llm(messages: list[dict[str, str]], **kwargs) -> str:
    """Wrapper for LLMManager.call_llm"""
    return LLMManager.call_llm(messages, **kwargs)


@wraps(LLMManager.retry_call_llm)
def retry_call_llm(
    messages: list[dict[str, str]],
    process_response,
    failure_handler,
    max_attempts: int = 5,
    call_llm_kwargs: dict[str, Any] | None = None,
    process_response_kwargs: dict[str, Any] | None = None,
    failure_handler_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Wrapper for LLMManager.retry_call_llm"""
    return LLMManager.retry_call_llm(
        messages=messages,
        process_response=process_response,
        failure_handler=failure_handler,
        max_attempts=max_attempts,
        call_llm_kwargs=call_llm_kwargs,
        process_response_kwargs=process_response_kwargs,
        failure_handler_kwargs=failure_handler_kwargs,
    )
