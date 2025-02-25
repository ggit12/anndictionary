"""
Unit tests for anndict.llm.llm_manager
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
#disabled pylint false positives

import os
from unittest.mock import patch, MagicMock, mock_open

import pytest

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from anndict.llm.llm_manager import LLMManager
from anndict.llm.default_llm_initializer import DefaultLLMInitializer


# Test configure_llm_backend
def test_configure_llm_backend_sets_environment(clean_env):
    """Test that configure_llm_backend correctly sets environment variables"""
    with patch("anndict.llm.llm_manager.LLMProviders.get_providers") as mock_get_providers:
        mock_get_providers.return_value = {"openai": MagicMock()}

        LLMManager.configure_llm_backend(
            "openai", "gpt-4", api_key="test-key", temperature=0.7
        )

        assert os.environ["LLM_PROVIDER"] == "openai"
        assert os.environ["LLM_MODEL"] == "gpt-4"
        assert os.environ["LLM_API_KEY"] == "test-key"
        assert os.environ["LLM_TEMPERATURE"] == "0.7"


def test_configure_llm_backend_cleans_old_vars(clean_env):
    """Test that old LLM_ variables are cleaned before setting new ones"""
    os.environ["LLM_OLD_VAR"] = "old-value"

    with patch("anndict.llm.llm_manager.LLMProviders.get_providers") as mock_get_providers:
        mock_get_providers.return_value = {"openai": MagicMock()}

        LLMManager.configure_llm_backend("openai", "gpt-4")

        assert "LLM_OLD_VAR" not in os.environ


def test_configure_llm_backend_invalid_provider(clean_env):
    """Test that invalid provider raises ValueError"""
    with pytest.raises(ValueError, match="Unsupported provider"):
        LLMManager.configure_llm_backend("invalid_provider", "model")


# Test get_llm_config
def test_get_llm_config_returns_correct_config(clean_env):
    """Test that get_llm_config returns the expected configuration"""
    with patch("anndict.llm.llm_manager.LLMProviders.get_providers") as mock_get_providers:
        mock_providers = {
            "openai": MagicMock(
                class_name="ChatOpenAI", module_path="langchain_openai.chat_models"
            )
        }
        mock_get_providers.return_value = mock_providers

        LLMManager.configure_llm_backend("openai", "gpt-4", api_key="test-key")

        config = LLMManager.get_llm_config()

        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4"
        assert config["api_key"] == "test-key"
        assert config["class"] == "ChatOpenAI"
        assert config["module"] == "langchain_openai.chat_models"


def test_get_llm_config_no_provider_configured(clean_env):
    """Test that get_llm_config raises error when no provider is configured"""
    with pytest.raises(ValueError, match="No LLM backend found"):
        LLMManager.get_llm_config()


# Test get_llm
def test_get_llm_initializes_new_instance(clean_env):
    """Test that get_llm correctly initializes a new LLM instance"""
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_llm_class = MagicMock()
        mock_module.ChatOpenAI = mock_llm_class
        mock_import.return_value = mock_module

        with patch("anndict.llm.llm_manager.LLMProviders.get_providers") as mock_get_providers:
            mock_providers = {
                "openai": MagicMock(
                    class_name="ChatOpenAI",
                    module_path="langchain_openai.chat_models",
                    init_class=DefaultLLMInitializer,
                )
            }
            mock_get_providers.return_value = mock_providers

            LLMManager.configure_llm_backend("openai", "gpt-4", api_key="test-key")
            instance, config = LLMManager.get_llm(None)

            assert instance == mock_llm_class.return_value
            assert config["provider"] == "openai"
            assert config["model"] == "gpt-4"


# Test call_llm
def test_call_llm_processes_messages_correctly(clean_env):
    """Test that call_llm correctly processes different message types"""
    # Create a mock response with proper string representation
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock_response.__str__.return_value = "Test response"  # This controls what gets written to file

    mock_llm = MagicMock()
    mock_llm.return_value = mock_response

    mock_initializer = MagicMock()
    mock_initializer.initialize.return_value = ({}, {"supports_system_messages": True})

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    with patch("anndict.llm.llm_manager.LLMProviders.get_providers") as mock_get_providers:
        mock_provider_config = MagicMock()
        mock_provider_config.init_class.return_value = mock_initializer
        mock_get_providers.return_value = {"openai": mock_provider_config}

        LLMManager.configure_llm_backend("openai", "gpt-4", api_key="test-key")

        with patch("anndict.llm.llm_manager.LLMManager.get_llm") as mock_get_llm, \
             patch("builtins.open", mock_open()) as mock_file:
            mock_get_llm.return_value = (mock_llm, {})

            response = LLMManager.call_llm(messages)

            # Verify message types
            calls = mock_llm.call_args[0][0]
            assert isinstance(calls[0], SystemMessage)
            assert isinstance(calls[1], HumanMessage)
            assert isinstance(calls[2], AIMessage)

            # Verify response handling
            assert response == "Test response"
            mock_file().write.assert_called_with("Test response\n")


# Test retry_call_llm
def test_retry_call_llm_succeeds_first_try(clean_env):
    """Test retry_call_llm when first attempt succeeds"""
    mock_process = MagicMock(return_value="Processed result")
    mock_failure = MagicMock()

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.return_value = "LLM response"

        result = LLMManager.retry_call_llm(
            messages=[{"role": "user", "content": "test"}],
            process_response=mock_process,
            failure_handler=mock_failure,
        )

        assert result == "Processed result"
        mock_call_llm.assert_called_once()
        mock_failure.assert_not_called()


def test_retry_call_llm_retries_on_failure(clean_env):
    """Test retry_call_llm retries on process_response failure"""
    mock_process = MagicMock(side_effect=[ValueError, ValueError, "Success"])
    mock_failure = MagicMock()

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.return_value = "LLM response"

        result = LLMManager.retry_call_llm(
            messages=[{"role": "user", "content": "test"}],
            process_response=mock_process,
            failure_handler=mock_failure,
            max_attempts=3,
        )

        assert result == "Success"
        assert mock_call_llm.call_count == 3
        mock_failure.assert_not_called()


def test_retry_call_llm_exhausts_attempts(clean_env):
    """Test retry_call_llm calls failure_handler after max attempts"""
    mock_process = MagicMock(side_effect=ValueError)
    mock_failure = MagicMock(return_value="Failure result")

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.return_value = "LLM response"

        result = LLMManager.retry_call_llm(
            messages=[{"role": "user", "content": "test"}],
            process_response=mock_process,
            failure_handler=mock_failure,
            max_attempts=2,
        )

        assert result == "Failure result"
        assert mock_call_llm.call_count == 2
        mock_failure.assert_called_once()
