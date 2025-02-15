"""
unit tests for anndict.annotate.cells.de_novo.annotate_cell_type_by_comparison
"""
# pylint: disable=redefined-outer-name
#disable pylint false positives

from unittest.mock import patch

import pytest

import pandas as pd
from anndict.llm import configure_llm_backend

from anndict.annotate.cells.de_novo.annotate_cell_type_by_comparison import (
    ai_cell_types_by_comparison,
    ai_annotate_cell_type_by_comparison,
)


def test_ai_cell_types_by_comparison_single_gene():
    """Test ai_cell_type with a single gene"""
    gene_lists = [["CD3E"]]
    expected_response = ["T cell receptor signaling"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up response for call_llm
        mock_call_llm.side_effect = ["Initial Discussion",
                                     *expected_response
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_cell_types_by_comparison(gene_lists)

        assert result == expected_response

def test_ai_cell_types_by_comparison_multiple_genes():
    """Test ai_cell_type with multiple genes"""
    gene_lists = [["CD3E"], ["CD4", "CD8A"]]
    expected_response = ["T cell activation", "T cell receptor signaling"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up response for call_llm
        mock_call_llm.side_effect = ["Initial Discussion",
                                     *expected_response
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_cell_types_by_comparison(gene_lists)

        assert result == expected_response

def test_ai_cell_types_by_comparison_empty_list():
    """Test ai_biological_process with empty list raises error"""
    with pytest.raises(ValueError):
        ai_cell_types_by_comparison([])

def test_ai_cell_types_by_comparison_invalid_genes():
    """Test ai_biological_process with invalid gene names raises error"""
    with pytest.raises(ValueError):
        ai_cell_types_by_comparison([["123"], ["valid_gene"]])

def test_ai_annotate_cell_type_by_comparison(simple_adata_with_two_groups, expected_label_results_for_two_groups):
    """Test basic usage of ai_annotate_cell_type_by_comparison"""
    n_top_genes = 2

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for call_llm calls
        mock_call_llm.side_effect = [
            "Initial Discussion",
            *expected_label_results_for_two_groups['new_label_column'].tolist()
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_annotate_cell_type_by_comparison(
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

def test_ai_cell_types_by_comparison_with_tissues():
    """Test ai_cell_types_by_comparison with tissue information"""
    gene_lists = [["CD3E"], ["CD19"], ["CD14"]]  # Multiple gene lists
    tissues = ["blood", "blood", "blood"]  # One tissue per gene list
    cell_types = ["T cell", "B cell", "Monocyte"]  # One cell type per gene list
    expected_response = ["CD4+ T cell", "Memory B cell", "Classical monocyte"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up response for call_llm
        mock_call_llm.side_effect = [
            "Initial Discussion",  # First call for comparison
            *expected_response    # Subsequent calls for each gene list
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_cell_types_by_comparison(
            gene_lists,
            tissues=tissues,
            cell_types=cell_types,
            subtype=True
        )

        assert result == expected_response
        # Verify the number of LLM calls matches expected
        # (1 for initial comparison + 1 for each gene list)
        assert mock_call_llm.call_count == len(gene_lists) + 1