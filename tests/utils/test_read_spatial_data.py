"""
Unit tests for spatial transcriptomics data reading functionality.
"""
# pylint: disable=redefined-outer-name
#disabled pylint false positives

import tempfile
import os

import pytest
import numpy as np
import pandas as pd
import anndata as ad

from scipy.sparse import csr_matrix
from anndict.utils.read_spatial_data import (
    read_transcript_coords,
    get_steps_and_coords,
    populate_sparse_array,
    create_anndata,
    add_blank_image_to_adata,
    build_adata_from_transcript_positions,
    build_adata_from_visium,
)


@pytest.fixture
def sample_merscope_data():
    """Create a sample Merscope DataFrame."""
    return pd.DataFrame(
        {
            "global_x": [0.0, 0.0, 16.0, 16.0],
            "global_y": [0.0, 16.0, 0.0, 16.0],
            "gene": ["GAPDH", "ACTB", "GAPDH", "ACTB"],
        }
    ).astype({"global_x": "float64", "global_y": "float64", "gene": "string"})


@pytest.fixture
def sample_xenium_data():
    """Create a sample Xenium DataFrame."""
    return pd.DataFrame(
        {
            "feature_name": ["GAPDH", "ACTB", "GAPDH", "MALAT1"],
            "x_location": [1.0, 2.0, 1.5, 3.0],
            "y_location": [1.0, 2.0, 1.5, 3.0],
        }
    )


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("global_x,global_y,gene\n")
        # Create a 2x2 grid of points with consistent spacing
        f.write("0.0,0.0,GAPDH\n")  # bottom left
        f.write("0.0,16.0,ACTB\n")  # top left
        f.write("16.0,0.0,GAPDH\n")  # bottom right
        f.write("16.0,16.0,ACTB\n")  # top right
    yield f.name
    os.unlink(f.name)


def test_read_transcript_coords_merscope(temp_csv_file):
    """Test reading Merscope data."""

    df = read_transcript_coords(temp_csv_file, platform="Merscope")
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ["global_x", "global_y", "gene"])


def test_read_transcript_coords_xenium(sample_xenium_data):
    """Test reading Xenium data."""

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample_xenium_data.to_csv(f.name, index=False)
        df = read_transcript_coords(f.name, platform="Xenium")
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in ["global_x", "global_y", "gene"])
    os.unlink(f.name)


def test_read_transcript_coords_invalid_platform(temp_csv_file):
    """Test reading data with invalid platform."""

    with pytest.raises(ValueError, match="Unsupported platform"):
        read_transcript_coords(temp_csv_file, platform="InvalidPlatform")


def test_get_steps_and_coords(sample_merscope_data):
    """Test computation of steps and coordinates."""

    x_steps, y_steps, coords = get_steps_and_coords(
        sample_merscope_data, box_size=16, step_size=16
    )
    assert isinstance(x_steps, int)
    assert isinstance(y_steps, int)
    assert isinstance(coords, list)
    assert all(isinstance(coord, list) and len(coord) == 2 for coord in coords)
    # Print debug information
    print(f"x_steps: {x_steps}, y_steps: {y_steps}")
    print(f"Number of coordinates: {len(coords)}")
    print(f"Coordinates: {coords}")


def test_get_steps_and_coords_invalid_box_size(sample_merscope_data):
    """Test steps and coordinates with invalid box size."""

    with pytest.raises(ValueError, match="box size is larger than image"):
        get_steps_and_coords(sample_merscope_data, box_size=1000, step_size=1)


def test_populate_sparse_array(sample_merscope_data):
    """Test creation of sparse array from data."""

    genes = np.array(["GAPDH", "ACTB"])
    sparse_mat = populate_sparse_array(sample_merscope_data, genes, step_size=16)
    assert isinstance(sparse_mat, csr_matrix)
    assert sparse_mat.shape[1] == len(genes)
    # Print shape and contents for debugging
    print(f"Sparse matrix shape: {sparse_mat.shape}")
    print(f"Sparse matrix data: {sparse_mat.toarray()}")


def test_create_anndata(sample_merscope_data):
    """Test creation of AnnData object."""

    genes = np.array(["GAPDH", "ACTB"])
    sparse_mat = populate_sparse_array(sample_merscope_data, genes, step_size=16)
    _, _, coords = get_steps_and_coords(sample_merscope_data, box_size=16, step_size=16)
    print(f"Sparse matrix shape: {sparse_mat.shape}")
    print(f"Number of coordinates: {len(coords)}")
    adata = create_anndata(sparse_mat, genes, coords)
    assert isinstance(adata, ad.AnnData)
    assert "gene_symbols" in adata.var.columns
    assert all(
        col in adata.obs.columns for col in ["global_x_topleft", "global_y_topleft"]
    )
    assert adata.n_obs == len(
        coords
    ), f"Number of observations {adata.n_obs} should match number of coordinates {len(coords)}"
    assert isinstance(adata, ad.AnnData)
    assert "gene_symbols" in adata.var.columns
    assert all(
        col in adata.obs.columns for col in ["global_x_topleft", "global_y_topleft"]
    )


def test_add_blank_image_to_adata():
    """Test adding blank image to AnnData object."""

    adata = ad.AnnData(
        X=np.array([[1, 2], [3, 4]]),
        obs=pd.DataFrame(
            {
                "global_x_topleft": [1.0, 2.0],
                "global_y_topleft": [1.0, 2.0],
            },
            index=["cell1", "cell2"],
        ),
    )
    adata = add_blank_image_to_adata(adata, platform="Merscope")
    assert "spatial" in adata.uns
    assert "spatial" in adata.obsm
    assert isinstance(adata.uns["spatial"]["library_id"]["images"]["hires"], np.ndarray)


def test_build_adata_from_transcript_positions(temp_csv_file):
    """Test end-to-end building of AnnData from transcript positions."""

    with tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False) as out_file:
        paths_dict = {temp_csv_file: out_file.name}
        # Use same box_size and step_size as the test data points spacing
        build_adata_from_transcript_positions(
            paths_dict, box_size=16, step_size=16, platform="Merscope"
        )
        assert os.path.exists(out_file.name)
        adata = ad.read_h5ad(out_file.name)
        assert isinstance(adata, ad.AnnData)
        # Check the dimensions make sense given our 2x2 grid input
        assert adata.n_vars == 2  # GAPDH and ACTB
        assert adata.obs.shape[0] >= 1  # Should have at least one spatial region
    os.unlink(out_file.name)


def test_build_adata_from_visium(tmp_path):
    """Test building AnnData from Visium data."""
    # Note: This test requires creating mock Visium data structure
    # You might want to use the fixtures from conftest.py or create specific Visium mock data

    # This is a placeholder - you'll need to create proper Visium directory structure
    visium_dir = tmp_path / "visium_data"
    visium_dir.mkdir()
    output_path = tmp_path / "output.h5ad"

    # Mock minimum Visium directory structure
    # You'll need to add necessary files based on what sc.read_visium expects

    with pytest.raises(Exception):  # Replace with specific exception
        build_adata_from_visium({str(visium_dir): str(output_path)})
