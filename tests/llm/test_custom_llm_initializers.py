"""
unit tests for the LLM provider implementations
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from langchain_community.chat_models.azureml_endpoint import (
    AzureMLEndpointApiType,
    LlamaChatContentFormatter,
)
from anndict.llm.custom_llm_initalizers import (
    BedrockLLMInitializer,
    AzureMLLLMInitializer,
    GoogleGenAILLMInitializer,
)


# BedrockLLMInitializer Tests
@pytest.mark.parametrize(
    "model_id,expected_support",
    [
        ("amazon.titan-text-express-v1", False),
        ("anthropic.claude-3", True),
    ],
)
def test_bedrock_system_message_support(bedrock_config, model_id, expected_support):
    """Test system message support detection for different models"""
    initializer = BedrockLLMInitializer(bedrock_config)

    with patch.dict(
        os.environ,
        {
            "LLM_REGION_NAME": "us-west-2",
            "LLM_AWS_ACCESS_KEY_ID": "test-key",
            "LLM_AWS_SECRET_ACCESS_KEY": "test-secret",
            "LLM_MODEL": model_id,
        },
    ), patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        _, kwargs = initializer.initialize({})
        assert kwargs["supports_system_messages"] == expected_support


def test_bedrock_missing_env_vars(bedrock_config):
    """Test error handling for missing environment variables"""
    initializer = BedrockLLMInitializer(bedrock_config)

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="LLM_REGION_NAME"):
            initializer.initialize({})


def test_bedrock_client_creation(bedrock_config):
    """Test Bedrock client creation and parameter filtering"""
    initializer = BedrockLLMInitializer(bedrock_config)
    test_args = {
        "model": "test-model",
        "streaming": True,
        "invalid_param": "should-be-removed",
    }

    with patch.dict(
        os.environ,
        {
            "LLM_REGION_NAME": "us-west-2",
            "LLM_AWS_ACCESS_KEY_ID": "test-key",
            "LLM_AWS_SECRET_ACCESS_KEY": "test-secret",
            "LLM_MODEL": "anthropic.claude-3",
        },
    ), patch("boto3.Session") as mock_session:
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        args, _ = initializer.initialize(test_args)
        assert "client" in args
        assert args["streaming"] is True
        assert "invalid_param" not in args
        assert "rate_limiter" in args


# AzureMLLLMInitializer Tests
def test_azure_url_construction(azure_config):
    """Test Azure endpoint URL construction"""
    initializer = AzureMLLLMInitializer(azure_config)
    constructor_args = {
        "endpoint_name": "test-endpoint",
        "region": "eastus",
        "api_key": "test-key",
    }

    args, _ = initializer.initialize(constructor_args)
    assert (
        args["endpoint_url"]
        == "https://test-endpoint.eastus.inference.ai.azure.com/v1/chat/completions"
    )
    assert args["endpoint_api_type"] == AzureMLEndpointApiType.serverless
    assert args["endpoint_api_key"] == "test-key"
    assert isinstance(args["content_formatter"], LlamaChatContentFormatter)


def test_azure_missing_params(azure_config):
    """Test error handling for missing required parameters"""
    initializer = AzureMLLLMInitializer(azure_config)

    with pytest.raises(ValueError, match="endpoint_name, region, and api_key"):
        initializer.initialize({"endpoint_name": "test"})


# GoogleGenAILLMInitializer Tests
def test_google_parameter_mapping(google_config):
    """Test parameter mapping and transformation"""
    initializer = GoogleGenAILLMInitializer(google_config)

    args, kwargs = initializer.initialize(
        constructor_args={}, max_tokens=100, temperature=0.7
    )

    assert args["max_output_tokens"] == 100
    assert args["temperature"] == 0.7
    assert "max_tokens" not in args
    assert kwargs == {}
    assert "rate_limiter" in args
    assert os.environ["GRPC_VERBOSITY"] == "ERROR"
