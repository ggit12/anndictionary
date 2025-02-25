"""
unit tests for anndict.automated_label_management.unify_columns_between_adata
"""
# pylint: disable=use-implicit-booleaness-not-comparison
#disable pylint false positives

from unittest.mock import patch

import pytest

from anndict.automated_label_management.unify_columns_between_adata import (
    ai_unify_labels,
)
from anndict.llm import configure_llm_backend


def test_ai_unify_labels_basic(adata_dict_with_different_cell_type_cols):
    """Test basic functionality of ai_unify_labels."""
    label_columns = {("sample1",): "cell_type", ("sample2",): "labels"}

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up direct string responses for LLM
        mock_call_llm.side_effect = [
            "Acknowledged",  # Initial acknowledgment
            "{'CD8+ t cell': 'CD8+ T cell', 'cd8 T-cell': 'CD8+ T cell', \
            'CD8+ T Cell': 'CD8+ T cell', 'CD8-positive T cell': 'CD8+ T cell'}",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_unify_labels(
            adata_dict_with_different_cell_type_cols,
            label_columns,
            new_label_column="unified_type",
            simplification_level="unified, typo-fixed",
        )

        # Verify LLM was called correctly
        assert mock_call_llm.call_count == 2  # Initial + mapping call

        # Check if new column was added to each AnnData
        for key in adata_dict_with_different_cell_type_cols:
            assert "unified_type" in adata_dict_with_different_cell_type_cols[key].obs.columns
            # Verify all values in new column are unified
            assert all(
                val == "CD8+ T cell"
                for val in adata_dict_with_different_cell_type_cols[key].obs["unified_type"]
            )

        # Check returned mapping
        expected_mapping = {
            "CD8+ t cell": "CD8+ T cell",
            "cd8 T-cell": "CD8+ T cell",
            "CD8+ T Cell": "CD8+ T cell",
            "CD8-positive T cell": "CD8+ T cell",
        }
        assert result == expected_mapping


def test_ai_unify_labels_empty_data(adata_dict_with_empty_adata):
    """Test ai_unify_labels with empty AnnData objects."""

    label_columns = {("empty_sample1",): "cell_type", ("empty_sample2",): "labels"}

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up direct string responses for LLM
        mock_call_llm.side_effect = [
            "Acknowledged",  # Initial acknowledgment
            "{}",  # Empty mapping since no labels
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = ai_unify_labels(
            adata_dict_with_empty_adata, label_columns, "unified_type"
        )

    assert result == {}
    # Check new column was added even if empty
    for key in adata_dict_with_empty_adata:
        assert "unified_type" in adata_dict_with_empty_adata[key].obs.columns


def test_ai_unify_labels_missing_column(simple_adata_dict):
    """Test error handling for missing column."""

    label_columns = {
        ("sample1",): "nonexistent_column",
        ("sample2",): "condition",  # Assuming 'condition' exists in sample2
    }

    with pytest.raises(KeyError):
        ai_unify_labels(simple_adata_dict, label_columns, "unified_type")
