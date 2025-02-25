"""
unit tets for anndict.utils.scanpy_
"""

import pytest
import numpy as np

from anndict.adata_dict import AdataDict
from anndict.utils.scanpy_ import (
    sample_adata_dict,
    sample_and_drop,
)

def test_sample_adata_dict_n_obs(simple_adata_with_three_unbalanced_groups):
    """Test sampling with fixed n_obs"""
    adata_dict = AdataDict(
        {
            ("sample1",): simple_adata_with_three_unbalanced_groups.copy(),
            ("sample2",): simple_adata_with_three_unbalanced_groups.copy(),
        },
        hierarchy=("sample",),
    )

    # Original number of observations
    _ = {key: adata.n_obs for key, adata in adata_dict.items()}

    # Sample with n_obs = 3
    sample_adata_dict(adata_dict, n_obs=3)

    # Check new number of observations
    for key in adata_dict:
        assert adata_dict[key].n_obs == 3

        # Check the distribution of groups is maintained
        for group in ["0", "1", "2"]:
            orig_prop = (
                simple_adata_with_three_unbalanced_groups.obs["cluster"] == group
            ).mean()
            new_prop = (adata_dict[key].obs["cluster"] == group).mean()
            # Allow for some sampling variation
            assert abs(orig_prop - new_prop) < 0.4


def test_sample_adata_dict_fraction(simple_adata_with_three_unbalanced_groups):
    """Test sampling with fraction"""
    adata_dict = AdataDict(
        {
            ("sample1",): simple_adata_with_three_unbalanced_groups.copy(),
            ("sample2",): simple_adata_with_three_unbalanced_groups.copy(),
        },
        hierarchy=("sample",),
    )

    # Original number of observations
    original_n_obs = {key: adata.n_obs for key, adata in adata_dict.items()}

    # Sample with fraction = 0.5
    sample_adata_dict(adata_dict, fraction=0.5)

    # Check new number of observations
    for key in adata_dict:
        assert adata_dict[key].n_obs == original_n_obs[key] // 2

        # Check the distribution of groups is maintained
        for group in ["0", "1", "2"]:
            orig_prop = (
                simple_adata_with_three_unbalanced_groups.obs["cluster"] == group
            ).mean()
            new_prop = (adata_dict[key].obs["cluster"] == group).mean()
            # Allow for some sampling variation
            assert abs(orig_prop - new_prop) < 0.4


def test_sample_and_drop_basic(simple_adata_with_three_unbalanced_groups):
    """Test basic functionality of sample_and_drop with string key"""
    adata = simple_adata_with_three_unbalanced_groups.copy()

    # Test with strata_key as string
    result = sample_and_drop(adata, strata_keys="cluster")

    # Should maintain all groups as min_num_cells=0
    assert len(np.unique(result.obs["cluster"])) == 3
    assert result.n_obs == adata.n_obs

    # Test group sizes maintained approximately
    for group in ["0", "1", "2"]:
        orig_count = (adata.obs["cluster"] == group).sum()
        new_count = (result.obs["cluster"] == group).sum()
        assert new_count == orig_count


def test_sample_and_drop_list_key(simple_adata_with_three_unbalanced_groups):
    """Test sample_and_drop with list key"""
    adata = simple_adata_with_three_unbalanced_groups.copy()

    # Test with strata_key as list
    result = sample_and_drop(adata, strata_keys=["cluster"])

    # Should maintain all groups
    assert len(np.unique(result.obs["cluster"])) == 3
    assert result.n_obs == adata.n_obs


def test_sample_and_drop_min_cells(simple_adata_with_three_unbalanced_groups):
    """Test sample_and_drop with minimum cell filter"""
    adata = simple_adata_with_three_unbalanced_groups.copy()

    # Test with min_num_cells=2
    result = sample_and_drop(adata, strata_keys="cluster", min_num_cells=2)

    # All groups should still be present as all have >= 2 cells
    assert len(np.unique(result.obs["cluster"])) == 3

    # Create a modified version with one small group
    small_adata = adata.copy()
    small_adata = small_adata[
        ~small_adata.obs_names.isin(
            small_adata[small_adata.obs["cluster"] == "1"].obs_names[1:]
        )
    ]

    # Now test with min_num_cells=2
    result_small = sample_and_drop(small_adata, strata_keys="cluster", min_num_cells=2)

    # Group '1' should be dropped as it now has only 1 cell
    assert len(np.unique(result_small.obs["cluster"])) == 2
    assert "1" not in result_small.obs["cluster"].values


def test_sample_and_drop_n_largest(simple_adata_with_three_unbalanced_groups):
    """Test sample_and_drop with n_largest_groups parameter"""
    adata = simple_adata_with_three_unbalanced_groups.copy()

    # Test with n_largest_groups=2
    result = sample_and_drop(adata, strata_keys="cluster", n_largest_groups=2)

    # Should keep only the 2 largest groups
    unique_groups = np.unique(result.obs["cluster"])
    assert len(unique_groups) == 2

    # Group '2' (4 cells) and '0' (3 cells) should be kept, '1' (2 cells) should be dropped
    assert "2" in unique_groups
    assert "0" in unique_groups
    assert "1" not in unique_groups

def test_sample_and_drop_oversample(simple_adata_with_three_unbalanced_groups):
    """Test over-sampling with fixed n_obs"""
    adata = simple_adata_with_three_unbalanced_groups.copy()

    adata = sample_and_drop(adata, strata_keys="cluster", min_num_cells=0, n_obs=20) #more n_obs than any cluster has

    # Check new number of observations
    #in this case, should have same number of cells as you started with (for each cluster)
    for group in ["0", "1", "2"]:
        orig_count = (simple_adata_with_three_unbalanced_groups.obs["cluster"] == group).sum()
        new_count = (adata.obs["cluster"] == group).sum()
        assert new_count == orig_count


def test_sample_and_drop_invalid_key(simple_adata_with_three_unbalanced_groups):
    """Test sample_and_drop with invalid strata key"""
    adata = simple_adata_with_three_unbalanced_groups.copy()

    # Test with invalid strata key
    with pytest.raises(KeyError):
        _ = sample_and_drop(adata, strata_keys="invalid_key")
