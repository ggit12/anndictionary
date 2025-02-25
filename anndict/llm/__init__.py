"""
LLM Configuration Module
=========================

This module handles the configuration and use of LLMs from different providers through 
a unified interface. It manages provider configurations, initialization strategies, and 
rate limiting for each supported LLM provider.

The module supports dynamic configuration of various LLM backends including:
    - OpenAI
    - Anthropic
    - AWS Bedrock
    - Google AI
    - Azure OpenAI
    - Azure ML endpoints
    - Cohere
    - HuggingFace
    - Vertex AI
    - Ollama

Key Components:
    - Provider configuration using dataclasses
        - Provider-specific initialization strategies
        - Abstract base classes for provider initialization
    - LLM calling with
        - retry logic
        - rate limiting
        - customizable response processing and failure handling


The module is used internally by ``AnnDictionary`` shouldn't generally be imported directly
by end users. Instead, use the main package interface:

.. code-block:: python

    import anndict as adt
    adt.configure_llm_backend(...)
"""

from .llm_call import (  # type: ignore
    configure_llm_backend,
    get_llm_config,
    call_llm,
    retry_call_llm,

)

from .parse_llm_response import (
    extract_dictionary_from_ai_string,
    extract_list_from_ai_string,
    process_llm_category_mapping,

)

from .provider_initializer_mapping import LLMProviders

__all__ = [

    #llm_call.py
    "configure_llm_backend",
    "get_llm_config",
    "call_llm",
    "retry_call_llm",

    #parse_llm_response.py
    "extract_dictionary_from_ai_string",
    "extract_list_from_ai_string",
    "process_llm_category_mapping",

    #provider_initializer_mapping.py
    "LLMProviders",
]
