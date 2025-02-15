"""
unit tests for anddict.annotate.cells.de_novo.automated_clustering
"""

import base64

from unittest.mock import patch

import numpy as np
import pytest
import matplotlib.pyplot as plt

from anndata import AnnData
from anndict.llm import configure_llm_backend
from anndict.annotate.cells.de_novo.automated_clustering import (
    encode_plot_for_openai,
    ai_resolution_interpretation,
    determine_sign_of_resolution_change,
    ai_determine_leiden_resolution,
)

def test_encode_plot_for_openai():
    """Test encode_plot_for_openai function"""

    # Create a simple test plot
    def test_plot():
        plt.plot([1, 2, 3], [1, 2, 3])

    # Get result
    result = encode_plot_for_openai(test_plot)

    # Test if it's valid base64
    try:
        base64.b64decode(result)
        is_valid = True
    except:  # pylint: disable=bare-except
        is_valid = False

    assert is_valid
    plt.close()


def test_ai_resolution_interpretation():
    """Test ai_resolution_interpretation function"""

    # Create a simple test plot
    def test_plot():
        plt.figure(figsize=(10, 6))
        plt.plot([1, 2, 3], [1, 2, 3])
        plt.title("Test Clustering Plot")
        plt.xlabel("X")
        plt.ylabel("Y")

    # Expected response from the LLM
    expected_response = "increased"

    # Mock the LLM call
    with patch("anndict.llm.llm_manager.LLMManager.call_llm") as mock_call_llm:
        # Set up response for call_llm
        mock_call_llm.return_value = expected_response

        # Configure LLM backend
        configure_llm_backend(provider="openai", model="gpt-4", api_key="test-key")

        # Call the function
        result = ai_resolution_interpretation(test_plot)

        # Verify the LLM was called with correct parameters
        assert mock_call_llm.called

        # Check the result
        assert result == expected_response

    # Clean up
    plt.close()


@pytest.mark.parametrize(
    "annotation, expected",
    [
        ("resolution decreased", -1),  # Tests "decreased" case
        ("decreased resolution", -1),  # Another "decreased" case
        ("resolution increased", 1),  # Tests "increased" case
        ("increased resolution", 1),  # Another "increased" case
        ("resolution unchanged", 0),  # Tests "unchanged" case
        ("unchanged resolution", 0),  # Another "unchanged" case
        ("no change", 0),  # Tests default case
        ("", 0),  # Tests empty string
        ("some random text", 0),  # Tests arbitrary text
    ],
)
def test_determine_sign_of_resolution_change(annotation, expected):
    """Test the determine_sign_of_resolution_change function with inputs that trigger all cases"""
    assert determine_sign_of_resolution_change(annotation) == expected


@pytest.mark.parametrize(
    "mock_responses, expected_calls, final_resolution",
    [
        (
            ["resolution unchanged"],
            1,
            1.0
        ),
        (
            ["resolution increased", "resolution decreased"],
            2,
            1
        ),
        (
            [" increased", "increased", "unchanged"],
            3,
            1.3
        ),
        (
            [" decreased", ""],
            2,
            0.85
        )
    ]
)
def test_ai_determine_leiden_resolution_sequences(mock_responses, expected_calls, final_resolution):
    """Test ai_determine_leiden_resolution with different response sequences"""
    # Create a mock AnnData object
    n_obs = 100
    n_vars = 50
    data = np.random.random((n_obs, n_vars))
    adata = AnnData(data)
    adata.obsm['X_umap'] = np.random.random((n_obs, 2))

    initial_resolution = 1.0

    # Create patches for both functions
    with patch('anndict.annotate.cells.de_novo.automated_clustering.ai_resolution_interpretation') as mock_ai_interp, \
         patch('scanpy.tl.leiden') as mock_leiden:
        
        # Configure mock for ai_resolution_interpretation
        mock_ai_interp.side_effect = mock_responses
        
        # Configure mock for sc.tl.leiden
        mock_leiden.return_value = None

        # Call the function
        result = ai_determine_leiden_resolution(adata, initial_resolution)

        # Verify the number of calls
        assert mock_ai_interp.call_count == expected_calls
        
        # Verify leiden was called with correct initial resolution
        first_leiden_call = mock_leiden.call_args_list[0]
        assert first_leiden_call[1]['resolution'] == initial_resolution

        # Verify exact final resolution
        assert result == final_resolution
