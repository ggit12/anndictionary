"""
unit tests for anndict.annotate.cells.de_novo.annotate_cell_subtype
"""

from unittest.mock import patch

import pandas as pd

from anndict.llm import configure_llm_backend
from anndict.annotate.cells.de_novo.annotate_cell_subtype import (
    ai_annotate_cell_type_by_comparison_adata_dict,
    ai_annotate_cell_sub_type,
)


def test_ai_annotate_cell_type_by_comparison_adata_dict(simple_adata_dict_more_cells):
    """Test the ai_annotate_cell_type_by_comparison_adata_dict function."""
    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:

        # Set up responses for call_llm
        mock_call_llm.side_effect = [
            "Gene set comparison analysis...",
            "NK cells",
            "Macrophages",
            "Gene set comparison analysis...",
            "B cells",
            "T cells",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Call the wrapper
        result = ai_annotate_cell_type_by_comparison_adata_dict(
            adata_dict=simple_adata_dict_more_cells,
            groupby="condition",
            use_multithreading=False,  # need False for consistent order of results
        )

        # Create expected DataFrames

        expected_sample1_df = pd.DataFrame(
            {
                "condition": ["ctrl", "treat"],  # Not categorical anymore
                "ai_cell_type_by_comparison": ["NK cells", "Macrophages"],
                "top_10_genes": [["gene2", "gene1"], ["gene2", "gene1"]],
            }
        )

        expected_sample2_df = pd.DataFrame(
            {
                "condition": ["ctrl", "treat"],  # Not categorical anymore
                "ai_cell_type_by_comparison": ["B cells", "T cells"],
                "top_10_genes": [["gene2", "gene1"], ["gene2", "gene1"]],
            }
        )

        pd.testing.assert_frame_equal(
            result[("sample1",)].reset_index(drop=True),
            expected_sample1_df,
            check_index_type=False,
        )

        # Compare with actual results
        pd.testing.assert_frame_equal(
            result[("sample2",)].reset_index(drop=True),
            expected_sample2_df,
            check_index_type=False,
        )


def test_ai_annotate_cell_sub_type(mock_adata_for_cell_subtyping):
    """Test the ai_annotate_cell_sub_type function."""
    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up responses for LLM calls
        mock_call_llm.side_effect = [
            "Gene set comparison for T cells...",
            "CD4+ T cell",
            "CD8+ T cell",
            "Gene set comparison for B cells...",
            "Memory B cell",
            "Plasma cell",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Call the function
        result_adata, _ = ai_annotate_cell_sub_type(
            adata=mock_adata_for_cell_subtyping,
            cell_type_col="cell_type",
            sub_cluster_col="subcluster",
            new_label_column="ai_subtype",
            tissue_of_origin_col="tissue",
            n_top_genes=10,
        )

        # Assert the results
        assert "ai_subtype" in result_adata.obs.columns
