"""
unit tests for anndict.annotate.genes.make_cell_type_gene_list
"""

from unittest.mock import patch

import pytest

from anndict.llm import configure_llm_backend
from anndict.annotate.genes.make_cell_type_gene_list import ai_make_cell_type_gene_list


def test_basic_usage():
    """Test basic usage with just required parameters"""
    expected_genes = ["CD3E", "CD4", "CD8A"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for call_llm calls
        mock_call_llm.side_effect = [
            "T cells express various markers",  # First call response
            "Here's a list of markers",  # Second call response
            "['CD3E', 'CD4', 'CD8A']",  # Response during retry
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_make_cell_type_gene_list("T cell", "human")

        assert result == expected_genes
        assert mock_call_llm.call_count == 3


def test_with_list_length():
    """Test with optional list_length parameter"""
    expected_genes = ["CD3E", "CD4", "CD8A", "IL2RA"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for call_llm calls
        mock_call_llm.side_effect = [
            "T cells express various markers",  # First call response
            "Here's a longer list of markers",  # Second call (list_length request)
            "Here's the final list",  # Third call response
            "['CD3E', 'CD4', 'CD8A', 'IL2RA']",  # Response during retry
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_make_cell_type_gene_list("T cell", "human", list_length="long")

        assert result == expected_genes
        assert mock_call_llm.call_count == 4


def test_llm_error():
    """Test error handling when LLM calls fail"""
    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = Exception("API Error")

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        with pytest.raises(Exception):
            ai_make_cell_type_gene_list("T cell", "human")


def test_numeric_cell_type():
    """Test that cell type raises ValueError if it can be cast to int"""
    with pytest.raises(ValueError):
        ai_make_cell_type_gene_list("123", "human")
