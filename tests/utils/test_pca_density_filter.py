"""
unit tests for anndict.utils.pca_density_filter
"""

import pytest
import numpy as np
import pandas as pd

from scipy import sparse

from anndict.utils.pca_density_filter import (
    pca_density_filter_adata,
    pca_density_filter_main,
)


# Tests for small adata (< 10 cells)
def test_pca_density_filter_basic(simple_adata):
    """Test basic functionality with a simple AnnData object."""
    # Make a copy to test non-inplace operation
    adata_copy = simple_adata.copy()

    # Test with default parameters
    _ = pca_density_filter_adata(adata_copy, drop_cells=True)

    # Check that a new column was added to original
    assert "density_filter" in adata_copy.obs.columns

    # Check that the filter column is boolean
    print(f'***Density filter dtype: {adata_copy.obs["density_filter"].dtype}')
    assert adata_copy.obs["density_filter"].dtype == pd.BooleanDtype()

    # Check that the var dataframe has the filter column
    assert "density_filter" in adata_copy.var.columns


def test_pca_density_filter_small_data(simple_adata):
    """Test behavior with very small data (< 10 cells)."""
    # Create a small subset
    small_adata = simple_adata[:1].copy()

    # Should not raise error and should mark all cells as True
    pca_density_filter_adata(small_adata)
    assert np.all(small_adata.obs["density_filter"])


def test_pca_density_filter_constant_data(simple_adata):
    """Test behavior with constant data."""
    # Create data with constant values
    constant_adata = simple_adata.copy()
    constant_adata.X = np.ones_like(constant_adata.X)

    # Should handle constant data gracefully
    pca_density_filter_adata(constant_adata)
    assert "density_filter" in constant_adata.obs.columns
    assert not np.any(constant_adata.var["density_filter"])


def test_pca_density_filter_layer(simple_adata):
    """Test using a specific layer for filtering."""
    # Add a test layer
    adata_with_layer = simple_adata.copy()
    adata_with_layer.layers["test_layer"] = adata_with_layer.X * 2

    # Filter using the layer
    pca_density_filter_adata(adata_with_layer, use_layer="test_layer")

    # Check that the filter was created
    assert "density_filter" in adata_with_layer.obs.columns


def test_pca_density_filter_custom_name(simple_adata):
    """Test using a custom name for the filter column."""
    custom_name = "custom_filter"

    pca_density_filter_adata(simple_adata, new_filter_column_name=custom_name)

    assert custom_name in simple_adata.obs.columns
    assert custom_name in simple_adata.var.columns


def test_pca_density_filter_main_basic():
    """Test the main filtering function directly."""
    # Create test data
    data = np.random.normal(0, 1, (100, 10))

    density, cutoff, variables = pca_density_filter_main(
        data, n_components=3, threshold=0.1
    )

    assert density.shape[0] == data.shape[0]
    assert isinstance(cutoff, float)
    assert len(variables) <= data.shape[1]


def test_pca_density_filter_main_edge_cases():
    """Test edge cases for the main filtering function."""
    # Test with constant data
    constant_data = np.ones((50, 5))
    density, _, variables = pca_density_filter_main(constant_data)
    assert len(variables) == 0

    # Test with small n_components
    data = np.random.normal(0, 1, (100, 10))
    density, _, variables = pca_density_filter_main(data, n_components=1)
    assert density.shape[0] == data.shape[0]


@pytest.mark.parametrize("threshold", [0.05, 0.1, 0.2])
def test_pca_density_filter_thresholds(simple_adata, threshold):
    """Test different density thresholds."""
    adata_copy = simple_adata.copy()

    pca_density_filter_adata(adata_copy, threshold=threshold)

    # Check the filter exists
    assert "density_filter" in adata_copy.obs.columns
    # Verify it's boolean
    assert adata_copy.obs["density_filter"].dtype == pd.BooleanDtype()


# Tests for large adata (> 10 cells)
def test_pca_density_filter_large_dataset(large_adata):
    """Test PCA density filtering on a larger dataset."""
    # Make a copy to test non-inplace operation
    adata_copy = large_adata.copy()

    # Run the filter
    filtered_adata = pca_density_filter_adata(
        adata_copy, threshold=0.1, drop_cells=True
    )

    # Basic checks
    assert "density_filter" in adata_copy.obs.columns
    assert adata_copy.obs["density_filter"].dtype == pd.BooleanDtype()

    # Check that we actually filtered some cells
    n_cells_before = large_adata.n_obs
    n_cells_after = filtered_adata.n_obs
    assert n_cells_after < n_cells_before, "No cells were filtered out"

    # Check that the filter is working as expected
    # For our simulated data, we expect most cells to be kept
    assert n_cells_after >= 0.8 * n_cells_before, "Too many cells were filtered out"


def test_pca_density_filter_sparse_data(sparse_large_adata):
    """Test PCA density filtering on sparse data."""
    # Make a copy
    adata_copy = sparse_large_adata.copy()

    # Run the filter
    filtered_adata = pca_density_filter_adata(
        adata_copy, threshold=0.1, drop_cells=True
    )

    # Check basic properties
    assert "density_filter" in adata_copy.obs.columns
    assert sparse_large_adata.n_obs > filtered_adata.n_obs

    # Verify that the output maintains sparsity
    assert sparse.issparse(filtered_adata.X), "Output lost sparsity"


def test_pca_density_filter_with_layers(large_adata):
    """Test PCA density filtering using different layers."""
    # Create a copy and add some layers
    adata_copy = large_adata.copy()

    # Add normalized and log-transformed layers
    adata_copy.layers["normalized"] = (
        adata_copy.X / np.sum(adata_copy.X, axis=1)[:, np.newaxis]
    )
    adata_copy.layers["log"] = np.log1p(adata_copy.X)

    # First run with default layer (None)
    pca_density_filter_adata(
        adata_copy,
        use_layer=None,
        new_filter_column_name="density_filter",
        drop_cells=False  # Don't drop cells so we can compare filters
    )

    # Then run with each layer and compare
    for layer in ["normalized", "log"]:
        pca_density_filter_adata(
            adata_copy,  # Use same object
            use_layer=layer,
            new_filter_column_name=f"density_filter_{layer}",
            drop_cells=False  # Don't drop cells so we can compare filters
        )

        # Now both filters should exist in the same object
        base_filter = adata_copy.obs["density_filter"]
        layer_filter = adata_copy.obs[f"density_filter_{layer}"]
        assert not np.array_equal(
            base_filter, layer_filter
        ), f"Filter results for layer {layer} identical to base filter"


def test_pca_density_filter_high_dimensional(large_adata):
    """Test PCA density filtering with different numbers of components."""
    # Test with different numbers of PCA components
    for n_components in [2, 5, 10]:
        adata_copy = large_adata.copy()

        filtered_adata = pca_density_filter_adata(
            adata_copy,
            n_components=n_components,
            new_filter_column_name=f"density_filter_{n_components}",
            drop_cells=True,
        )

        # Check that the filter exists and is boolean
        assert f"density_filter_{n_components}" in adata_copy.obs.columns
        assert (
            adata_copy.obs[f"density_filter_{n_components}"].dtype == pd.BooleanDtype()
        )

        # Verify that using more components doesn't fail
        assert filtered_adata is not None


def test_pca_density_filter_thresholds_large_data(large_adata):
    """Test different density thresholds on larger dataset."""
    results = {}

    for threshold in [0.05, 0.1, 0.2]:
        adata_copy = large_adata.copy()

        filtered_adata = pca_density_filter_adata(
            adata_copy,
            threshold=threshold,
            new_filter_column_name=f"density_filter_{threshold}",
            drop_cells=True,
        )

        results[threshold] = filtered_adata.n_obs

    # Check that higher thresholds retain fewer cells
    # (since threshold=0.2 means keep only cells above 20th percentile of density)
    assert (
        results[0.2] <= results[0.1] <= results[0.05]
    ), "Higher thresholds should remove more cells"
