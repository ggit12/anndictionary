"""
unit tests for anndict.auto_label_management.clean_single_column.in_adata_var
"""

# pylint: disable=redefined-outer-name
# disable pylint false positives

from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd

from anndata import AnnData
from anndict.automated_label_management.clean_single_column.in_adata_var import (
    simplify_var_index,
    map_gene_labels_to_simplified_set,
)
from anndict.llm import configure_llm_backend


# Test simplify_var_index function
def test_simplify_var_index_basic(simple_adata_with_var):
    """Test basic functionality of simplify_var_index."""
    with patch(
        "anndict.automated_label_management.clean_single_column.in_adata_var.map_gene_labels_to_simplified_set"
    ) as mock_map:
        # Set up mock mapping
        mock_map.return_value = {
            "HSP90AA1": "Heat Shock",
            "HSPA1A": "Heat Shock",
            "CLOCK": "Circadian",
            "IL1A": "Interleukin",
        }

        result = simplify_var_index(
            simple_adata_with_var,
            "simplify",
            "functional_category",
            simplification_level="pathway",
        )

        # Check if mapping function was called correctly
        mock_map.assert_called_once()
        mock_genes = mock_map.call_args[0][0]
        assert set(mock_genes) == {"HSP90AA1", "HSPA1A", "CLOCK", "IL1A"}

        # Check if new column was created and mapped correctly
        assert "functional_category" in simple_adata_with_var.var.columns
        assert (
            simple_adata_with_var.var.loc["HSP90AA1", "functional_category"]
            == "Heat Shock"
        )
        assert (
            simple_adata_with_var.var.loc["APOD", "functional_category"] == "APOD"
        )  # Non-simplified gene

        assert result == mock_map.return_value


def test_simplify_var_index_non_boolean_column(simple_adata_with_var):
    """Test error handling for non-boolean column."""
    with pytest.raises(ValueError, match="must be a boolean column"):
        simplify_var_index(simple_adata_with_var, "other_col", "functional_category")


def test_simplify_var_index_too_many_genes():
    """Test error handling for too many genes."""
    # Create AnnData with >1000 genes
    var_df = pd.DataFrame(
        {"simplify": [True] * 1001}, index=[f"gene_{i}" for i in range(1001)]
    )

    adata = AnnData(X=np.random.rand(3, 1001), var=var_df)

    with pytest.raises(ValueError, match="Cannot simplify more than 1000 genes"):
        simplify_var_index(adata, "simplify", "functional_category")


# Test map_gene_labels_to_simplified_set function
def test_map_gene_labels_basic():
    """Test basic functionality of map_gene_labels_to_simplified_set."""
    labels = ["HSP90AA1", "HSPA1A", "CLOCK", "IL1A"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up direct string responses
        mock_call_llm.side_effect = [
            "Acknowledged",  # Initial acknowledgment
            "{'HSP90AA1': 'Heat Shock', 'HSPA1A': 'Heat Shock', 'CLOCK': 'Circadian', 'IL1A': 'Interleukin'}",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = map_gene_labels_to_simplified_set(
            labels, simplification_level="pathway", batch_size=5
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(labels)
        assert result["HSP90AA1"] == "Heat Shock"
        assert result["HSPA1A"] == "Heat Shock"
        assert result["CLOCK"] == "Circadian"
        assert result["IL1A"] == "Interleukin"

        # Verify the mock was called correctly
        assert mock_call_llm.call_count == 2


def test_map_gene_labels_batch_processing():
    """Test batch processing in map_gene_labels_to_simplified_set."""
    labels = ["HSP90AA1", "HSPA1A", "CLOCK", "IL1A", "PER1", "ARNTL"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = [
            "Acknowledged",  # Initial acknowledgment
            "{'HSP90AA1': 'Heat Shock', 'HSPA1A': 'Heat Shock', 'CLOCK': 'Circadian'}",  # First batch
            "{'IL1A': 'Interleukin', 'PER1': 'Circadian', 'ARNTL': 'Circadian'}",  # Second batch
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = map_gene_labels_to_simplified_set(labels, batch_size=3)

        assert mock_call_llm.call_count == 3  # Initial + 2 batches
        assert len(result) == len(labels)
        # Check specific mappings
        assert result["HSP90AA1"] == "Heat Shock"
        assert result["CLOCK"] == "Circadian"
        assert result["PER1"] == "Circadian"


def test_map_gene_labels_error_handling():
    """Test error handling in map_gene_labels_to_simplified_set."""
    labels = ["HSP90AA1", "INVALID_GENE"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = Exception("API Error")

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        with pytest.raises(Exception):
            map_gene_labels_to_simplified_set(labels)
