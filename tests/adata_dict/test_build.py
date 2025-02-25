"""
unit tests for anndict.adata_dict.build
"""

import pytest
from anndict.adata_dict.build import build_adata_dict
from anndict.adata_dict import AdataDict

def test_build_adata_dict_basic(simple_adata):
    """Test basic functionality of build_adata_dict."""
    # Convert cell_type to category
    simple_adata.obs['cell_type'] = simple_adata.obs['cell_type'].astype('category')

    adata_dict = build_adata_dict(simple_adata, ['cell_type'])

    # Check structure
    assert isinstance(adata_dict, AdataDict)
    assert len(adata_dict) == 2
    assert ('A',) in adata_dict
    assert ('B',) in adata_dict

    # Check contents
    assert adata_dict[('A',)].n_obs == 1
    assert adata_dict[('B',)].n_obs == 1

    # Check hierarchy
    assert adata_dict.hierarchy == ('cell_type',)

def test_build_adata_dict_multiple_strata(simple_adata):
    """Test building with multiple stratification keys."""
    # Convert columns to category
    simple_adata.obs['cell_type'] = simple_adata.obs['cell_type'].astype('category')
    simple_adata.obs['cell_id'] = simple_adata.obs['cell_id'].astype('category')

    adata_dict = build_adata_dict(simple_adata, ['cell_type', 'cell_id'])

    # Check structure
    assert len(adata_dict) == 2  # Since each cell has unique combination
    assert ('A', 'cell1') in adata_dict
    assert ('B', 'cell2') in adata_dict

    # Check each stratum has correct number of cells
    assert all(adata.n_obs == 1 for adata in adata_dict.values())

def test_build_adata_dict_desired_strata_list(simple_adata):
    """Test build_adata_dict with desired strata as list."""
    simple_adata.obs['cell_type'] = simple_adata.obs['cell_type'].astype('category')

    desired = [('A',)]
    adata_dict = build_adata_dict(simple_adata, ['cell_type'], desired_strata=desired)

    # Check only desired combinations are present
    assert len(adata_dict) == 1
    assert set(adata_dict.keys()) == set(desired)
    assert adata_dict[('A',)].n_obs == 1

def test_build_adata_dict_desired_strata_dict(simple_adata):
    """Test build_adata_dict with desired strata as dictionary."""
    simple_adata.obs['cell_type'] = simple_adata.obs['cell_type'].astype('category')

    desired = {
        'cell_type': ['A']
    }
    adata_dict = build_adata_dict(simple_adata, ['cell_type'], desired_strata=desired)

    # Check structure
    assert len(adata_dict) == 1
    assert ('A',) in adata_dict
    assert ('B',) not in adata_dict

def test_build_adata_dict_invalid_strata(simple_adata):
    """Test error handling for invalid strata specifications."""
    simple_adata.obs['cell_type'] = simple_adata.obs['cell_type'].astype('category')

    # Test with single strings instead of tuples
    with pytest.raises(ValueError, match="should be a list of tuples"):
        build_adata_dict(simple_adata, ['cell_type'], desired_strata=['A'])

    # Test with invalid desired_strata type
    with pytest.raises(ValueError, match="must be either a list of tuples or a dictionary"):
        build_adata_dict(simple_adata, ['cell_type'], desired_strata=42)
