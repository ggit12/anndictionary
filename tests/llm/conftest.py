"""
Conftest for LLM module
"""

import os
from unittest.mock import MagicMock

import pytest

from anndict.llm.base_llm_initializer import LLMProviderConfig
from anndict.llm.default_llm_initializer import DefaultLLMInitializer
from anndict.llm.custom_llm_initalizers import (
    BedrockLLMInitializer,
    AzureMLLLMInitializer,
    GoogleGenAILLMInitializer,
)

#test fixtures for anndict.llm.llm_manager
@pytest.fixture
def clean_env():
    """Clear LLM-related environment variables before each test"""
    # Save original environment
    original_env = dict(os.environ)

    # Clean LLM variables
    for key in list(os.environ.keys()):
        if key.startswith("LLM_"):
            del os.environ[key]

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_providers():
    """Mock LLMProviders.get_providers"""
    return {
        "openai": MagicMock(
            class_name="ChatOpenAI",
            module_path="langchain_openai.chat_models",
            init_class=DefaultLLMInitializer,
        ),
        "anthropic": MagicMock(
            class_name="ChatAnthropic",
            module_path="langchain_anthropic.chat_models",
            init_class=DefaultLLMInitializer,
        ),
    }


#test fixtures for anndict.llm.llm_provider_implementations
@pytest.fixture
def bedrock_config():
    """Fixture for Bedrock LLM provider configuration"""
    return LLMProviderConfig(
        class_name="BedrockChat",
        module_path="langchain.chat_models",
        init_class=BedrockLLMInitializer,
    )

@pytest.fixture
def azure_config():
    """Fixture for Azure ML LLM provider configuration"""
    return LLMProviderConfig(
        class_name="AzureMLEndpoint",
        module_path="langchain.chat_models",
        init_class=AzureMLLLMInitializer,
    )

@pytest.fixture
def google_config():
    """Fixture for Google Generative AI LLM provider configuration"""
    return LLMProviderConfig(
        class_name="ChatGoogleGenerativeAI",
        module_path="langchain.chat_models",
        init_class=GoogleGenAILLMInitializer,
    )

#test fixtures for anndict.llm.llm_provider_base
@pytest.fixture
def default_config():
    return LLMProviderConfig(
        class_name="TestLLM",
        module_path="test.llms",
        init_class=DefaultLLMInitializer,
    )

@pytest.fixture
def custom_config():
    return LLMProviderConfig(
        class_name="TestLLM",
        module_path="test.llms",
        init_class=DefaultLLMInitializer,
        requests_per_minute=100,
        check_every_n_seconds=0.2,
    )


#test fixtures for anndict.llm.parse_llm_response
@pytest.fixture
def raw_llm_response_with_mapping():
    """
    Raw LLM response containing a mapping dictionary
    Contains typical LLM response elements like explanatory text
    """
    return '''
    Here's the mapping you requested:
    {
        "First Category": "Group A",
        "Second Category": "Group A",
        "Third Category": "Group B",
        "Fourth Category": "Group B",
        "Fifth Category": "Group C"
    }
    Let me know if you need anything else!
    '''


@pytest.fixture
def raw_llm_response_with_list():
    """
    Raw LLM response containing a category list
    Includes typical AI response patterns
    """
    return '''
    Here are the categories:
    [
        "Group A",
        "Group B",
        "Group C"
    ]
    Hope this helps!
    '''


@pytest.fixture
def sample_mapping():
    """
    Sample abstract mapping dictionary
    Maps specific items to their groups
    """
    return {
        "First Category": "Group A",
        "Second Category": "Group A",
        "Third Category": "Group B",
        "Fourth Category": "Group B",
        "Fifth Category": "Group C"
    }


@pytest.fixture
def input_categories():
    """
    List of input categories with variations in naming
    Includes duplicates, case variations, and unmapped items
    """
    return [
        "First Category",
        "first category",
        "SECOND CATEGORY",
        "Third Category",
        "Unmapped Item"
    ]


@pytest.fixture
def expected_mapping():
    """
    Expected mapping after processing input categories
    Handles case-matching and unmapped items
    """
    return {
        "First Category": "Group A",
        "first category": "Group A",
        "SECOND CATEGORY": "Group A",
        "Third Category": "Group B",
        "Unmapped Item": "Unmapped Item"
    }