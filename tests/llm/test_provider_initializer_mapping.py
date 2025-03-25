"""
unit tests for anndict.llm.provider_initializer_mapping
"""

from anndict.llm.provider_initializer_mapping import LLMProviders, LLMProviderConfig
from anndict.llm.default_llm_initializer import DefaultLLMInitializer
from anndict.llm.custom_llm_initalizers import (
    BedrockLLMInitializer,
    AzureMLLLMInitializer,
    GoogleGenAILLMInitializer,
    OpenAILLMInitializer,
)

def test_get_providers_returns_expected_providers():
    """Test that get_providers returns all expected provider configurations"""
    providers = LLMProviders.get_providers()

    expected_providers = {
        "openai", "anthropic", "bedrock", "google", "azureml_endpoint",
        "azure_openai", "cohere", "huggingface", "vertexai", "ollama"
    }

    assert set(providers.keys()) == expected_providers

def test_provider_config_structure():
    """Test that each provider config has the required attributes with correct types"""
    providers = LLMProviders.get_providers()

    for _, config in providers.items():
        assert isinstance(config, LLMProviderConfig)
        assert isinstance(config.class_name, str)
        assert isinstance(config.module_path, str)
        assert hasattr(config.init_class, '__call__')  # Verify it's a callable/class

def test_specific_provider_configurations():
    """Test specific provider configurations that have custom initializers"""
    providers = LLMProviders.get_providers()

    # Test Bedrock configuration
    bedrock_config = providers["bedrock"]
    assert bedrock_config.class_name == "ChatBedrockConverse"
    assert bedrock_config.module_path == "langchain_aws.chat_models.bedrock_converse"
    assert bedrock_config.init_class == BedrockLLMInitializer

    # Test Google configuration
    google_config = providers["google"]
    assert google_config.class_name == "ChatGoogleGenerativeAI"
    assert google_config.module_path == "langchain_google_genai.chat_models"
    assert google_config.init_class == GoogleGenAILLMInitializer

    # Test Azure ML Endpoint configuration
    azure_config = providers["azureml_endpoint"]
    assert azure_config.class_name == "AzureMLChatOnlineEndpoint"
    assert azure_config.module_path == "langchain_community.chat_models.azureml_endpoint"
    assert azure_config.init_class == AzureMLLLMInitializer

    # Test OpenAI configuration
    openai_config = providers["openai"]
    assert openai_config.class_name == "ChatOpenAI"
    assert openai_config.module_path == "langchain_openai.chat_models"
    assert openai_config.init_class == OpenAILLMInitializer

def test_default_initializer_providers():
    """Test that standard providers use DefaultLLMInitializer"""
    providers = LLMProviders.get_providers()
    default_providers = ["anthropic", "cohere", "huggingface", "vertexai", "ollama", "azure_openai"]

    for provider in default_providers:
        assert providers[provider].init_class == DefaultLLMInitializer, f"{provider} should use DefaultLLMInitializer"

def test_provider_module_paths():
    """Test that module paths follow expected patterns"""
    providers = LLMProviders.get_providers()

    # Check community providers are in correct module
    community_providers = ["azureml_endpoint", "azure_openai", "cohere", "huggingface", "vertexai", "ollama"]
    for provider in community_providers:
        assert providers[provider].module_path.startswith("langchain_community."), \
            f"{provider} should be in langchain_community"
