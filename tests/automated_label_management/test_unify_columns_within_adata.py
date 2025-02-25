"""
unit tests for anndict.automated_label_management.unify_columns_within_adata
"""

from unittest.mock import patch

import pandas as pd

from anndict.automated_label_management.unify_columns_within_adata import (
    ensure_label_consistency_adata,
    ensure_label_consistency_main,
)
from anndict.llm import configure_llm_backend


def test_ensure_label_consistency_adata(simple_adata):
    """Test ensure_label_consistency_adata function."""
    cols = ["cell_type", "cell_id"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = [
            "Acknowledged",
            '{"A": "a", "B": "b", "cell1": "cell1", "cell2": "cell2"}',
        ]

        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        label_map = ensure_label_consistency_adata(simple_adata, cols)

        assert label_map == {"A": "a", "B": "b", "cell1": "cell1", "cell2": "cell2"}
        assert "consistent_cell_type" in simple_adata.obs.columns
        assert "consistent_cell_id" in simple_adata.obs.columns
        assert (simple_adata.obs["consistent_cell_type"] == ["a", "b"]).all()
        assert (simple_adata.obs["consistent_cell_id"] == ["cell1", "cell2"]).all()


def test_ensure_label_consistency_main():
    """Test ensure_label_consistency_main function."""
    df = pd.DataFrame(
        {
            "cell_type": ["CD8+ t cell", "cd8 T-cell"],
            "cell_id": ["CD8+ T Cell", "CD8-positive T cell"],
        }
    )

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = [
            "Acknowledged",
            '{"CD8+ t cell": "CD8+ T cell", "cd8 T-cell": "CD8+ T cell", "CD8+ T Cell": \
            "CD8+ T cell", "CD8-positive T cell": "CD8+ T cell"}',
        ]

        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result_df, label_map = ensure_label_consistency_main(df)

        assert (result_df["cell_type"] == ["CD8+ T cell", "CD8+ T cell"]).all()
        assert (result_df["cell_id"] == ["CD8+ T cell", "CD8+ T cell"]).all()
        assert label_map == {
            "CD8+ t cell": "CD8+ T cell",
            "cd8 T-cell": "CD8+ T cell",
            "CD8+ T Cell": "CD8+ T cell",
            "CD8-positive T cell": "CD8+ T cell",
        }
