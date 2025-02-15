"""
Unit tests for anndict.llm.llm_call
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=protected-access
#disabled pylint false positives

import unittest
from unittest.mock import patch, MagicMock

from anndict.llm.llm_call import (
    configure_llm_backend,
    get_llm_config,
    call_llm,
    retry_call_llm,
)
from anndict.llm.llm_manager import LLMManager
# from anndict.annotate.cells.de_novo.annotate_cell_type import ai_cell_type


def test_configure_llm_backend_passes_args():
    """Test that configure_llm_backend passes arguments correctly to LLMManager"""
    with patch('anndict.llm.llm_manager.LLMManager.configure_llm_backend') as mock_configure:
        configure_llm_backend(
            provider="openai",
            model="gpt-4",
            api_key="test-key",
        )

        mock_configure.assert_called_once_with(
            "openai",
            "gpt-4",
            api_key="test-key",
        )

class TestLLMClientCaching(unittest.TestCase):
    """Test caching behaviour of LLM client before and after reconfiguration"""
    def setUp(self):
        """Set up the test patches"""
        # Create and start the patches
        self.chat_openai_patcher = patch('langchain_openai.chat_models.ChatOpenAI', autospec=True)
        self.mock_chat_openai = self.chat_openai_patcher.start()

        # Create a factory that accepts constructor arguments
        def create_mock_instance(*args, **kwargs):
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Mocked response"
            mock_instance.invoke.return_value = mock_response
            return mock_instance

        # Set up the mock to create new instances each time it's called
        self.mock_chat_openai.side_effect = create_mock_instance

        # Configure initial LLM backend
        configure_llm_backend(
            provider='openai',
            model='gpt-4o',
            api_key='test-key'
        )

    def tearDown(self):
        """Stop all patches"""
        self.chat_openai_patcher.stop()

    def test_client_caching_and_reconfiguration(self):
        """Test that the LLM client is cached and properly recreated on reconfiguration"""

        # Sample messages for testing
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test message."},
        ]

        # First call - creates initial client
        _ = call_llm(test_messages)
        initial_client = LLMManager._llm_instance.client
        initial_client_id = id(initial_client)

        # Second call - should reuse the same client
        _ = call_llm(test_messages)
        second_client = LLMManager._llm_instance.client
        second_client_id = id(second_client)

        # Assert that the same client instance was reused
        self.assertEqual(
            initial_client_id, 
            second_client_id,
            "LLM client instance should be reused when configuration hasn't changed"
        )

        # Verify that ChatOpenAI was only created once
        self.mock_chat_openai.assert_called_once()

        # Reconfigure with different model
        configure_llm_backend(
            provider='openai',
            model='gpt-4',  # Different model
            api_key='test-key'
        )

        # Third call - should create new client due to reconfiguration
        _ = call_llm(test_messages)
        new_client = LLMManager._llm_instance.client
        new_client_id = id(new_client)

        # Assert that a new client instance was created
        self.assertNotEqual(
            initial_client_id,
            new_client_id,
            "LLM client instance should be different after reconfiguration"
        )

        # Verify that ChatOpenAI was created twice (once initially, once after reconfigure)
        self.assertEqual(self.mock_chat_openai.call_count, 2, 
                        "ChatOpenAI instance should be created exactly twice")


#this one works (tests caching behaviour, same as above test) if you insert real api keys
# class TestLLMClientCaching(unittest.TestCase):
#     def setUp(self):
#         """Set up initial configuration before each test"""
#         # Configure with initial settings
#         configure_llm_backend(
#             provider='openai',
#             model='gpt-4o',
#             api_key='test-openai-key'
#         )

#     def tearDown(self):
#         """Clean up after each test"""
#         # Reset any global state if necessary
#         pass

#     def test_client_caching_and_reconfiguration(self):
#         """Test that the LLM client is cached and properly recreated on reconfiguration"""

#         # First call - creates initial client
#         _ = ai_cell_type(['GAPDH', 'CD74', 'C3', 'C5'])
#         initial_client = LLMManager._llm_instance.client
#         initial_client_id = id(initial_client)

#         # Second call - should reuse the same client
#         _ = ai_cell_type(['GAPDH', 'CD74', 'C3', 'C5'])
#         second_client = LLMManager._llm_instance.client
#         second_client_id = id(second_client)

#         # Assert that the same client instance was reused
#         self.assertEqual(
#             initial_client_id, 
#             second_client_id,
#             "LLM client instance should be reused when configuration hasn't changed"
#         )

#         # Reconfigure with different model
#         configure_llm_backend(
#             provider='openai',
#             model='gpt-4',  # Different model
#             api_key='test-openai-key'
#         )

#         # Third call - should create new client due to reconfiguration
#         _ = ai_cell_type(['GAPDH', 'CD74', 'C3', 'C5'])
#         new_client = LLMManager._llm_instance.client
#         new_client_id = id(new_client)

#         # Assert that a new client instance was created
#         self.assertNotEqual(
#             initial_client_id,
#             new_client_id,
#             "LLM client instance should be different after reconfiguration"
#         )


def test_get_llm_config_passes_through():
    """Test that get_llm_config passes through to LLMManager"""
    expected_config = {"key": "value"}

    with patch('anndict.llm.llm_manager.LLMManager.get_llm_config', return_value=expected_config) as mock_get_config:
        config = get_llm_config()

        mock_get_config.assert_called_once_with()
        assert config == expected_config

def test_call_llm_passes_args():
    """Test that call_llm passes arguments correctly to LLMManager"""
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    kwargs = {"temperature": 0.7}

    with patch('anndict.llm.llm_manager.LLMManager.call_llm') as mock_call:
        call_llm(messages, **kwargs)

        # Assert with messages as a positional argument
        mock_call.assert_called_once_with(messages, temperature=0.7)

def test_retry_call_llm_passes_args():
    """Test that retry_call_llm passes all arguments correctly to LLMManager"""
    messages = [{"role": "user", "content": "Hello"}]
    process_response = lambda x: x
    failure_handler = lambda: None
    call_llm_kwargs = {"temperature": 0.7}
    process_response_kwargs = {"key1": "value1"}
    failure_handler_kwargs = {"key2": "value2"}

    with patch('anndict.llm.llm_manager.LLMManager.retry_call_llm') as mock_retry:
        retry_call_llm(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            max_attempts=3,
            call_llm_kwargs=call_llm_kwargs,
            process_response_kwargs=process_response_kwargs,
            failure_handler_kwargs=failure_handler_kwargs,
        )

        mock_retry.assert_called_once_with(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            max_attempts=3,
            call_llm_kwargs=call_llm_kwargs,
            process_response_kwargs=process_response_kwargs,
            failure_handler_kwargs=failure_handler_kwargs,
        )

def test_retry_call_llm_default_kwargs():
    """Test that retry_call_llm handles default kwargs correctly"""
    messages = [{"role": "user", "content": "Hello"}]
    process_response = lambda x: x
    failure_handler = lambda: None

    with patch('anndict.llm.llm_manager.LLMManager.retry_call_llm') as mock_retry:
        retry_call_llm(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler
        )

        # Verify defaults are passed correctly
        mock_retry.assert_called_once_with(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            max_attempts=5,
            call_llm_kwargs=None,
            process_response_kwargs=None,
            failure_handler_kwargs=None,
        )
