"""
unit tests for anndict.annotate.cells.benchmarking.ai_compare_cell_type_labels
"""

from unittest.mock import patch

import pandas as pd

from anndict.llm import configure_llm_backend
from anndict.annotate.cells.benchmarking.ai_compare_cell_type_labels import (
    ai_compare_cell_types_binary,
    ai_compare_cell_types_categorical,
    ai_compare_cell_type_labels_pairwise,
)


def test_ai_compare_cell_types_binary():
    """Test the LLM-based cell type comparison function"""

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up mock response for the LLM call
        mock_call_llm.side_effect = ["no", "yes"]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Test the comparison function
        result_no = ai_compare_cell_types_binary("T cell", "fibroblast")
        result_yes = ai_compare_cell_types_binary("T cell", "CD4+ T cell")

        # Assert the mock was called two times
        assert mock_call_llm.call_count == 2

        # Verify the result is a string
        assert isinstance(result_no, str)
        assert isinstance(result_yes, str)

        # Verify the result matches our mock response
        assert result_no == "no"
        assert result_yes == "yes"


def test_ai_compare_cell_types_binary_same_type():
    """Test the comparison function with the same exact cell type (i.e. a normalized string match)"""

    # Test with same cell type
    # this should short circuit the LLM call (so no mocking is needed)
    result = ai_compare_cell_types_binary("T cell", "t cell")

    # Verify the result is a string
    assert isinstance(result, str)

    # Verify the result matches our mock response
    assert result == "yes"


def test_ai_compare_cell_types_categorical():
    """Test the LLM-based cell type comparison function"""

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up mock response for the LLM call
        mock_call_llm.side_effect = [
            "no match",
            "partial match",
            "perfect match",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Test the comparison function
        result_no = ai_compare_cell_types_categorical("T cell", "fibroblast")
        result_partial = ai_compare_cell_types_categorical("B cell", "plasma cell")
        result_perfect = ai_compare_cell_types_categorical("CD8+ T cell", "T cell")

        # Assert the mock was called 3 times
        assert mock_call_llm.call_count == 3

        # Verify the result is a string
        assert isinstance(result_no, str)
        assert isinstance(result_partial, str)
        assert isinstance(result_perfect, str)

        # Verify the result matches our mock response
        assert result_no == "no match"
        assert result_partial == "partial match"
        assert result_perfect == "perfect match"


def test_ai_compare_cell_types_categorical_same_type():
    """Test the comparison function with the same exact cell type (i.e. a normalized string match)"""

    # Test with same cell type
    # this should short circuit the LLM call (so no mocking is needed)
    result = ai_compare_cell_types_categorical("T cell", "t cell")

    # Verify the result is a string
    assert isinstance(result, str)

    # Verify the result matches our mock response
    assert result == "perfect match"


def test_ai_compare_cell_type_labels_pairwise(simple_adata_with_many_obs_labels):
    """Test the pairwise comparison of cell type labels"""

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up mock responses for call_llm calls
        # We need 4 responses for the unique pairs:
        # - For cell_type_1 vs cell_type_2: (T, CD4), (B, CD8)
        # - For cell_type_1 vs cell_type_3: (T, naive), (B, memory)
        mock_call_llm.side_effect = [
            "yes",  # T vs CD4
            "no",  # B vs CD8
            "no",  # T vs naive
            "yes",  # B vs memory
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Test the comparison function
        result = ai_compare_cell_type_labels_pairwise(
            adata=simple_adata_with_many_obs_labels,
            cols1=["cell_type_1"],
            cols2=["cell_type_2", "cell_type_3"],
            comparison_level="binary",
        )

        # Assert the correct number of LLM calls
        assert mock_call_llm.call_count == 4

        # Assert the result has the correct structure
        assert isinstance(result, dict)
        assert list(result.keys()) == [
            ("cell_type_1", "cell_type_2"),
            ("cell_type_1", "cell_type_3"),
        ]

        # Check the structure and values of first comparison (cell_type_1 vs cell_type_2)
        df1 = result[("cell_type_1", "cell_type_2")]
        assert isinstance(df1, pd.DataFrame)
        assert list(df1.columns) == ["col1", "col2", "raw_agreement", "agreement"]
        assert df1.shape == (2, 4)
        assert df1.iloc[0].tolist() == ["T", "CD4", "yes", 1]
        assert df1.iloc[1].tolist() == ["B", "CD8", "no", 0]

        # Check the structure and values of second comparison (cell_type_1 vs cell_type_3)
        df2 = result[("cell_type_1", "cell_type_3")]
        assert isinstance(df2, pd.DataFrame)
        assert list(df2.columns) == ["col1", "col2", "raw_agreement", "agreement"]
        assert df2.shape == (2, 4)
        assert df2.iloc[0].tolist() == ["T", "naive", "no", 0]
        assert df2.iloc[1].tolist() == ["B", "memory", "yes", 1]
