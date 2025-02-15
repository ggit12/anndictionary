"""
unit tests for anndict.annotate.cells.de_novo.annotate_cell_type
"""
# pylint: disable=redefined-outer-name
#disable pylint false positives

from unittest.mock import patch

import pytest

import pandas as pd
from anndict.llm import configure_llm_backend

from anndict.annotate.cells.de_novo.annotate_cell_type import (
    ai_cell_type,
    ai_annotate_cell_type,
)


def test_ai_cell_type_single_gene():
    """Test ai_cell_type with a single gene"""
    gene_list = ["CD3E"]
    expected_response = "T cell receptor signaling"

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up response for call_llm
        mock_call_llm.side_effect = [expected_response]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_cell_type(gene_list)

        assert result == expected_response
        assert mock_call_llm.call_count == 1

def test_ai_cell_type_multiple_genes():
    """Test ai_cell_type with multiple genes"""
    gene_list = ["CD3E", "CD4", "CD8A"]
    expected_response = "T cell activation"

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up response for call_llm
        mock_call_llm.side_effect = [expected_response]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_cell_type(gene_list)

        assert result == expected_response
        assert mock_call_llm.call_count == 1

def test_ai_cell_type_empty_list():
    """Test ai_biological_process with empty list raises error"""
    with pytest.raises(ValueError):
        ai_cell_type([])

def test_ai_cell_type_invalid_genes():
    """Test ai_biological_process with invalid gene names raises error"""
    with pytest.raises(ValueError):
        ai_cell_type(["123", "456"])

def test_ai_annotate_cell_type(simple_adata_with_two_groups, expected_label_results_for_two_groups):
    """Test basic usage of ai_annotate_cell_type"""
    n_top_genes = 2

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for call_llm calls
        mock_call_llm.side_effect = [
            "ai_label_1",  # First call response
            "ai_label_2",  # Second call response
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_annotate_cell_type(
            adata=simple_adata_with_two_groups, groupby="cluster", 
            n_top_genes=n_top_genes, new_label_column='new_label_column'
        )
        print("result*********")
        print(result)
        print("expected_df*********")
        print(expected_label_results_for_two_groups)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected_label_results_for_two_groups)
        assert simple_adata_with_two_groups.obs['new_label_column'].tolist() == ['ai_label_1', 'ai_label_1', 'ai_label_2', 'ai_label_2']
