"""
unit tests for anndict.llm.base_llm_initializer
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

from anndict.llm.base_llm_initializer import LLMProviderConfig
from anndict.llm.default_llm_initializer import DefaultLLMInitializer

# Test LLMProviderConfig
def test_llm_provider_config_creation():
    """Test creation of LLMProviderConfig with default and custom values"""
    config = LLMProviderConfig(
        class_name="TestLLM",
        module_path="test.llms",
        init_class=DefaultLLMInitializer,
    )

    assert config.class_name == "TestLLM"
    assert config.module_path == "test.llms"
    assert config.init_class == DefaultLLMInitializer
    assert config.requests_per_minute == 40
    assert config.check_every_n_seconds == 0.1

# Test DefaultLLMInitializer
def test_default_initializer_initialize(default_config):
    """Test DefaultLLMInitializer's initialize method"""
    initializer = DefaultLLMInitializer(default_config)
    constructor_args = {"api_key": "test-key"}
    additional_kwargs = {"temperature": 0.7}

    result_args, result_kwargs = initializer.initialize(
        constructor_args.copy(), 
        **additional_kwargs
    )

    # Check rate_limiter was added
    assert "rate_limiter" in result_args
    assert isinstance(result_args["rate_limiter"], InMemoryRateLimiter)

    # Check original args were preserved
    assert result_args["api_key"] == "test-key"

    # Check additional kwargs were preserved
    assert result_kwargs == additional_kwargs

def test_default_initializer_with_custom_rate_limits(custom_config):
    """Test DefaultLLMInitializer with custom rate limiting parameters"""
    initializer = DefaultLLMInitializer(custom_config)
    constructor_args = {
        "requests_per_minute": 120,
        "check_every_n_seconds": 0.3,
    }

    result_args, _ = initializer.initialize(constructor_args.copy())
    rate_limiter = result_args["rate_limiter"]

    assert rate_limiter.requests_per_second == 120 / 60
    assert rate_limiter.check_every_n_seconds == 0.3

def test_default_initializer_preserve_unknown_args(default_config):
    """Test that unknown constructor arguments are preserved"""
    initializer = DefaultLLMInitializer(default_config)
    constructor_args = {
        "custom_param": "value",
        "another_param": 123,
    }

    result_args, _ = initializer.initialize(constructor_args.copy())

    assert "custom_param" in result_args
    assert "another_param" in result_args
    assert result_args["custom_param"] == "value"
    assert result_args["another_param"] == 123