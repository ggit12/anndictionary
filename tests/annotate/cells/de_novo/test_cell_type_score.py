"""
unit tests for anndict.annotate.cells.de_novo.cell_type_score
"""

from unittest.mock import patch

import pytest
import numpy as np

from anndict.llm import configure_llm_backend

from anndict.annotate.cells.de_novo.cell_type_score import cell_type_marker_gene_score


def test_cell_type_marker_gene_score_basic(simple_adata_with_two_groups):
    """Test basic functionality of cell_type_marker_gene_score"""

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for call_llm calls
        mock_call_llm.side_effect = [
            "T cells express various markers",
            "Here's a list of markers",
            "['gene1', 'gene2']",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Test with explicit cell types
        cell_type_marker_gene_score(
            adata=simple_adata_with_two_groups,
            cell_types=["T cell"],
            species="Human",
            score_name="_score",
        )

        # Verify the score column was created
        assert "T cell_score" in simple_adata_with_two_groups.obs.columns

        # Verify genes were marked in var
        assert "T cell_score" in simple_adata_with_two_groups.var.columns
        assert (
            simple_adata_with_two_groups.var["T cell_score"].sum() == 2
        )  # Two genes should be marked


def test_cell_type_marker_gene_score_col(simple_adata_with_two_groups):
    """Test using cell type column"""

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for each cluster
        mock_call_llm.side_effect = [
            "Condition A markers",
            "Marker list for cluster 0",
            "['gene1']",
            "Condition B markers",
            "Marker list for cluster 1",
            "['gene2']",
        ]

        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Test with cell type column
        cell_type_marker_gene_score(
            adata=simple_adata_with_two_groups,
            cell_type_col="condition",
            species="Human",
        )

        # Should have created scores for both clusters
        assert "A_score" in simple_adata_with_two_groups.obs.columns
        assert "B_score" in simple_adata_with_two_groups.obs.columns

        # Verify genes were marked in var
        assert "A_score" in simple_adata_with_two_groups.var.columns
        assert "B_score" in simple_adata_with_two_groups.var.columns


def test_cell_type_marker_gene_score_no_valid_genes(simple_adata_with_two_groups):
    """Test behavior when LLM returns genes not in dataset"""

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses to return genes not in our dataset
        mock_call_llm.side_effect = [
            "Cell type markers",
            "List of markers",
            "['CD3', 'CD4', 'CD8']",  # Genes not in test dataset
        ]

        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Test with explicit cell type
        cell_type_marker_gene_score(
            adata=simple_adata_with_two_groups, cell_types=["T cell"], species="Human"
        )

        # Verify score column exists and contains NaN values
        assert "T cell_score" in simple_adata_with_two_groups.obs.columns
        assert np.all(np.isnan(simple_adata_with_two_groups.obs["T cell_score"]))

        # Verify genes were marked in var
        assert "T cell_score" in simple_adata_with_two_groups.var.columns
        assert (
            simple_adata_with_two_groups.var["T cell_score"].sum() == 0
        )  # No genes should be marked


def test_cell_type_marker_gene_score_validation(simple_adata_with_two_groups):
    """Test input validation"""

    # Test providing both cell_types and cell_type_col
    with pytest.raises(
        ValueError, match="Provide either 'cell_type_col' or 'cell_types', not both"
    ):
        cell_type_marker_gene_score(
            adata=simple_adata_with_two_groups,
            cell_types=["TypeA"],
            cell_type_col="cluster",
        )

    # Test providing neither cell_types nor cell_type_col
    with pytest.raises(
        ValueError, match="Either 'cell_type_col' or 'cell_types' must be provided"
    ):
        cell_type_marker_gene_score(adata=simple_adata_with_two_groups, species="Human")
