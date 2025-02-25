"""
unit tests for anndict.auto_label_management.clean_single_column.in_adata_obs
"""
# pylint: disable=unused-argument
# pylint: disable=use-implicit-booleaness-not-comparison
#disable pylint false positives

from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd

from anndata import AnnData
from anndict.automated_label_management.clean_single_column.in_adata_obs import (
    simplify_obs_column,
    create_label_hierarchy,
    map_cell_type_labels_to_simplified_set,
)
from anndict.llm import configure_llm_backend


# Test fixtures for LLM setup
@pytest.fixture(autouse=True)
def mock_llm_setup():
    """Set up the LLM client mocking for all tests"""
    with patch(
        "langchain_openai.chat_models.ChatOpenAI", autospec=True
    ) as mock_chat_openai:
        # Create a factory that accepts constructor arguments
        def create_mock_instance(*args, **kwargs):
            mock_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "Mocked response"
            mock_instance.invoke.return_value = mock_response
            return mock_instance

        mock_chat_openai.side_effect = create_mock_instance

        # Configure initial LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        yield mock_chat_openai


# Test simplify_obs_column function
def test_simplify_obs_column_basic(simple_adata):
    """Test basic functionality of simplify_obs_column."""
    with patch(
        "anndict.automated_label_management.clean_single_column.in_adata_obs.map_cell_type_labels_to_simplified_set"
    ) as mock_map:
        mock_map.return_value = {"A": "Type1", "B": "Type2"}

        result = simplify_obs_column(
            simple_adata, "cell_type", "simplified_type", simplification_level="basic"
        )
        # Check if mapping function was called correctly
        mock_map.assert_called_once()
        assert set(mock_map.call_args[0][0]) == {"A", "B"}

        # Check if new column was created and mapped correctly
        assert "simplified_type" in simple_adata.obs.columns
        assert simple_adata.obs["simplified_type"].dtype.name == "category"
        assert set(simple_adata.obs["simplified_type"].unique()) == {"Type1", "Type2"}
        assert result == {"A": "Type1", "B": "Type2"}


def test_simplify_obs_column_empty_data():
    """Test simplify_obs_column with empty data."""
    empty_adata = AnnData(
        X=np.zeros((0, 0)),
        obs=pd.DataFrame({"cell_type": pd.Series(dtype="str")}, index=pd.Index([])),
    )

    with patch(
        "anndict.automated_label_management.clean_single_column.in_adata_obs.map_cell_type_labels_to_simplified_set" # pylint: disable=line-too-long
    ) as mock_map:
        mock_map.return_value = {}

        result = simplify_obs_column(empty_adata, "cell_type", "simplified_type")

        assert result == {}
        assert "simplified_type" in empty_adata.obs.columns


def test_simplify_obs_column_invalid_column(simple_adata):
    """Test simplify_obs_column with non-existent column."""
    with pytest.raises(KeyError):
        simplify_obs_column(simple_adata, "nonexistent_column", "simplified_type")


# Test create_label_hierarchy function
def test_create_label_hierarchy_basic(simple_adata):
    """Test basic functionality of create_label_hierarchy."""
    with patch(
        "anndict.automated_label_management.clean_single_column.in_adata_obs.simplify_obs_column"
    ) as mock_simplify:
        mock_simplify.side_effect = [
            {"A": "TypeA", "B": "TypeB"},
            {"TypeA": "Class1", "TypeB": "Class1"},
        ]

        levels = ["cell_type", "cell_class"]
        result = create_label_hierarchy(simple_adata, "cell_type", levels)

        assert mock_simplify.call_count == len(levels)
        assert len(result) == len(levels)
        assert "cell_type_cell_type" in result
        assert "cell_type_cell_class" in result


def test_create_label_hierarchy_single_level(simple_adata):
    """Test create_label_hierarchy with single level."""
    with patch(
        "anndict.automated_label_management.clean_single_column.in_adata_obs.simplify_obs_column"
    ) as mock_simplify:
        mock_simplify.return_value = {"A": "TypeA", "B": "TypeB"}

        result = create_label_hierarchy(simple_adata, "cell_type", ["basic"])

        mock_simplify.assert_called_once()
        assert len(result) == 1
        assert "cell_type_basic" in result


def test_create_label_hierarchy_empty_levels(simple_adata):
    """Test create_label_hierarchy with empty levels list."""
    result = create_label_hierarchy(simple_adata, "cell_type", [])
    assert result == {}


# Test map_cell_type_labels_to_simplified_set function
def test_map_cell_type_labels_basic():
    """Test basic functionality of map_cell_type_labels_to_simplified_set."""
    labels = ["CD8+ T cell", "CD4+ T cell", "B cell"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up direct string responses
        mock_call_llm.side_effect = [
            "Acknowledged",
            "{'CD8+ T cell': 'T cell', 'CD4+ T cell': 'T cell', 'B cell': 'B cell'}",
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = map_cell_type_labels_to_simplified_set(
            labels, simplification_level="cell_type", batch_size=5
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(labels)
        assert result["CD8+ T cell"] == "T cell"
        assert result["CD4+ T cell"] == "T cell"
        assert result["B cell"] == "B cell"

        # Verify the mock was called correctly
        assert mock_call_llm.call_count == 2


def test_map_cell_type_labels_batch_processing():
    """Test batch processing in map_cell_type_labels_to_simplified_set."""
    labels = ["CD8+ T cell", "CD4+ T cell", "B cell", "NK cell"]

    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        mock_call_llm.side_effect = [
            "Acknowledged",  # Initial acknowledgment
            "{'CD8+ T cell': 'T cell', 'CD4+ T cell': 'T cell'}",  # First batch
            "{'B cell': 'B cell', 'NK cell': 'NK cell'}",  # Second batch
        ]

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        result = map_cell_type_labels_to_simplified_set(labels, batch_size=2)

        assert mock_call_llm.call_count == 3  # Initial + 2 batches
        assert len(result) == len(labels)

        # Verify all labels were mapped correctly
        assert result["CD8+ T cell"] == "T cell"
        assert result["CD4+ T cell"] == "T cell"
        assert result["B cell"] == "B cell"
        assert result["NK cell"] == "NK cell"


def test_map_cell_type_labels_error_handling():
    """Test error handling in map_cell_type_labels_to_simplified_set."""
    labels = ["CD8+ T cell", "Invalid cell"]

    with patch("anndict.llm.llm_call.retry_call_llm") as mock_retry_llm:
        mock_retry_llm.side_effect = Exception("API Error")

        with pytest.raises(Exception):
            map_cell_type_labels_to_simplified_set(labels)


def test_map_cell_type_labels_empty_input():
    """Test map_cell_type_labels_to_simplified_set with empty input."""
    result = map_cell_type_labels_to_simplified_set([])
    assert result == {}
