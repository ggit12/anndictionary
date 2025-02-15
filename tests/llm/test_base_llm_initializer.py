"""
unit tests for anndict.llm.base_llm_initializer
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

from anndict.llm.base_llm_initializer import (
    BaseLLMInitializer,
)

# Test BaseLLMInitializer
class BaseLLMInitializerForTest(BaseLLMInitializer):
    """Concrete implementation of BaseLLMInitializer for testing"""
    def initialize(self, constructor_args, **kwargs):
        pass


def test_base_initializer_create_rate_limiter(default_config):
    """Test rate limiter creation with default parameters"""
    initializer = BaseLLMInitializerForTest(default_config)
    rate_limiter = initializer.create_rate_limiter({})

    assert isinstance(rate_limiter, InMemoryRateLimiter)
    assert rate_limiter.requests_per_second == default_config.requests_per_minute / 60
    assert rate_limiter.check_every_n_seconds == default_config.check_every_n_seconds
    assert rate_limiter.max_bucket_size == default_config.requests_per_minute

def test_base_initializer_create_rate_limiter_custom_params(default_config):
    """Test rate limiter creation with custom parameters"""
    initializer = BaseLLMInitializerForTest(default_config)
    custom_args = {
        "requests_per_minute": 100,
        "check_every_n_seconds": 0.2,
        "max_bucket_size": 150,
    }

    rate_limiter = initializer.create_rate_limiter(custom_args.copy())

    assert isinstance(rate_limiter, InMemoryRateLimiter)
    assert rate_limiter.requests_per_second == 100 / 60
    assert rate_limiter.check_every_n_seconds == 0.2
    assert rate_limiter.max_bucket_size == 150
