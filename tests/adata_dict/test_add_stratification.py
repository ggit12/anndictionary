"""
Unit tests for add_stratification module.
"""
import pytest

from anndict.adata_dict import AdataDict
from anndict.adata_dict.add_stratification import add_stratification


def test_add_stratification_basic(simple_adata_dict):
    """Test basic add_stratification functionality."""
    # Add condition stratification to sample-based dict
    adata_dict = add_stratification(simple_adata_dict, ['condition'])

    # Check structure
    assert isinstance(adata_dict, AdataDict)
    assert ('ctrl',) in adata_dict
    assert ('treat',) in adata_dict
    assert all(isinstance(v, AdataDict) for v in adata_dict.values())

    # Check data integrity
    assert adata_dict[('ctrl',)][('sample1',)].n_obs == 2
    assert adata_dict[('treat',)][('sample2',)].n_obs == 2

def test_add_stratification_with_desired_strata(simple_adata_dict):
    """Test add_stratification with specific desired strata."""
    desired = {'condition': ['ctrl']}
    adata_dict = add_stratification(simple_adata_dict, ['condition'], desired_strata=desired)

    # Check structure
    assert ('ctrl',) in adata_dict
    assert ('treat',) not in adata_dict
    assert all(isinstance(v, AdataDict) for v in adata_dict.values())

def test_add_stratification_redundant(simple_adata_dict):
    """Test error handling when trying to add redundant stratification."""
    with pytest.raises(ValueError, match="already stratified by sample"):
        add_stratification(simple_adata_dict, ['sample'])

def test_add_stratification_nested(nested_adata_dict):
    """Test add_stratification on nested dictionary."""
    # Add condition stratification to nested group/sample dict
    adata_dict = add_stratification(nested_adata_dict, ['condition'])

    # Check structure
    assert isinstance(adata_dict, AdataDict)
    assert ('ctrl',) in adata_dict
    assert ('treat',) in adata_dict

    # Check that original structure is preserved under new stratification
    assert ('group1',) in adata_dict[('ctrl',)]
    assert ('group2',) in adata_dict[('treat',)]
