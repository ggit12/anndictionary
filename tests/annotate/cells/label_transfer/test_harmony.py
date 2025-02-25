"""
unit tests for anndict.annotate.cells.label_transfer.harmony
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
#disable pylint false positives


from unittest.mock import patch

import pytest
import numpy as np

from anndict.annotate.cells.label_transfer.harmony import harmony_label_transfer


@pytest.fixture
def scanpy_mocks():
    """Fixture to create all required scanpy mocks"""
    with patch("scanpy.tl.pca") as mock_pca, patch(
        "scanpy.pp.neighbors"
    ) as mock_neighbors, patch("scanpy.tl.umap") as mock_umap, patch(
        "scanpy.tl.leiden"
    ) as mock_leiden:

        def mock_pca_effect(adata, **kwargs):
            adata.obsm["X_pca"] = np.random.rand(adata.n_obs, 50)

        mock_pca.side_effect = mock_pca_effect

        yield {
            "pca": mock_pca,
            "neighbors": mock_neighbors,
            "umap": mock_umap,
            "leiden": mock_leiden,
        }

@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_harmony_label_transfer(simple_adata_with_two_groups, scanpy_mocks):
    """Test harmony label transfer function with identical source and destination data"""

    # Create copies of the test data for source and destination
    destination_adata = simple_adata_with_two_groups.copy()
    origin_adata = simple_adata_with_two_groups.copy()

    # Mock the harmony integration
    with patch("harmonypy.run_harmony") as mock_harmony:
        # Create a mock harmony result
        class MockHarmonyResult:
            def __init__(self):
                self.Z_corr = np.random.rand(
                    50, 8
                )  # Mock integrated data - for 8 total cells (4 from each dataset)

        mock_harmony.return_value = MockHarmonyResult()

        # Run the harmony label transfer
        harmony_label_transfer(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_subset_column="condition",
            label_column="cluster",
        )

        # Verify that harmony labels were added to the destination object
        assert "harmony_labels" in destination_adata.obs.columns

        # Verify that all cells received a label
        assert not destination_adata.obs["harmony_labels"].isna().any()

        # Verify that the labels are consistent with the original clustering
        # Since we're using the same data, the transferred labels should match
        # the original clusters
        assert all(
            destination_adata.obs["harmony_labels"] == destination_adata.obs["cluster"]
        )

        # Verify that the original data wasn't modified
        assert np.array_equal(destination_adata.X, simple_adata_with_two_groups.X)

        # Verify that all required functions were called
        mock_harmony.assert_called_once()
        scanpy_mocks["pca"].assert_called_once()
        scanpy_mocks["neighbors"].assert_called_once()
        scanpy_mocks["umap"].assert_called_once()
        scanpy_mocks["leiden"].assert_called_once()

@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_harmony_label_transfer_with_subset(simple_adata_with_two_groups, scanpy_mocks):
    """Test harmony label transfer with a subset of conditions"""

    # Create copies and modify the destination data to have only condition 'A'
    destination_adata = simple_adata_with_two_groups[
        simple_adata_with_two_groups.obs["condition"] == "A"
    ].copy()
    origin_adata = simple_adata_with_two_groups.copy()

    # Mock the harmony integration
    with patch("harmonypy.run_harmony") as mock_harmony:

        class MockHarmonyResult:
            """Mock class to simulate harmony integration results"""
            def __init__(self):
                self.Z_corr = np.random.rand(
                    50, 4
                )  # Mock integrated data for subset - 2 cells from destination + 2 matching cells from source

        mock_harmony.return_value = MockHarmonyResult()

        # Run the harmony label transfer
        harmony_label_transfer(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_subset_column="condition",
            label_column="cluster",
        )

        # Verify that harmony labels were added
        assert "harmony_labels" in destination_adata.obs.columns

        # Verify that all cells in the subset received a label
        assert not destination_adata.obs["harmony_labels"].isna().any()

        # Verify that only cells with condition 'A' are present
        assert all(destination_adata.obs["condition"] == "A")

        # Verify that all required functions were called
        mock_harmony.assert_called_once()
        scanpy_mocks["pca"].assert_called_once()
        scanpy_mocks["neighbors"].assert_called_once()
        scanpy_mocks["umap"].assert_called_once()
        scanpy_mocks["leiden"].assert_called_once()

@pytest.mark.filterwarnings("ignore:Observation names are not unique")
def test_harmony_label_transfer_invalid_column(simple_adata_with_two_groups, scanpy_mocks):
    """Test harmony label transfer with invalid column names"""

    destination_adata = simple_adata_with_two_groups.copy()
    origin_adata = simple_adata_with_two_groups.copy()

    # Test with invalid origin_subset_column
    with pytest.raises(KeyError):
        harmony_label_transfer(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_subset_column="invalid_column",
            label_column="cluster",
        )

    # Test with invalid label_column
    with pytest.raises(ValueError):
        harmony_label_transfer(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_subset_column="condition",
            label_column="invalid_column",
        )
