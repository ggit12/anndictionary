"""
Unit tests for the ``AdataDict`` class.
"""

import pytest
import pandas as pd
import numpy as np
import anndata as ad

from anndict.adata_dict import AdataDict

def test_adata_dict_init(simple_adata_dict):
    """Test AdataDict initialization."""
    # Test empty initialization
    empty_dict = AdataDict()
    assert len(empty_dict) == 0
    assert empty_dict.hierarchy == ()

    # Test initialization with data and hierarchy
    assert len(simple_adata_dict) == 2
    assert simple_adata_dict.hierarchy == ('sample',)

    # Test that all values are AnnData objects
    for value in simple_adata_dict.values():
        assert isinstance(value, ad.AnnData)

def test_adata_dict_getitem(simple_adata_dict):
    """Test __getitem__ functionality."""
    # Test tuple key access
    assert isinstance(simple_adata_dict[('sample1',)], ad.AnnData)

    # Test single string key access
    assert isinstance(simple_adata_dict['sample1'], ad.AnnData)

    # Test key error
    with pytest.raises(KeyError):
        _ = simple_adata_dict['nonexistent']

def test_adata_dict_setitem(simple_adata):
    """Test __setitem__ functionality."""
    test_dict = AdataDict(hierarchy=('sample',))

    # Test tuple key setting
    test_dict[('sample1',)] = simple_adata
    assert isinstance(test_dict[('sample1',)], ad.AnnData)

    # Test single string key setting
    test_dict['sample2'] = simple_adata
    assert isinstance(test_dict['sample2'], ad.AnnData)

def test_adata_dict_flatten(nested_adata_dict):
    """Test flatten method."""
    # Flatten the dictionary in-place
    nested_adata_dict.flatten()

    # Check that all nested AdataDicts are flattened
    assert all(isinstance(v, ad.AnnData) for v in nested_adata_dict.values())

    # Check that keys are properly combined
    expected_keys = {
        ('group1', 'sample1'), ('group1', 'sample2'),
        ('group2', 'sample3'), ('group2', 'sample4')
    }
    assert set(nested_adata_dict.keys()) == expected_keys

def test_adata_dict_flatten_complex(complex_nested_adata_dict):
    """Test flatten method with complex nested structure including celltypes."""
    # Flatten the dictionary in-place
    complex_nested_adata_dict.flatten()

    # Check that all nested AdataDicts are flattened
    assert all(isinstance(v, ad.AnnData) for v in complex_nested_adata_dict.values())

    # Check that keys are properly combined
    expected_keys = {
        ('group1', 'sample1', 'celltype1'), 
        ('group1', 'sample2', 'celltype2'),
        ('group2', 'sample3', 'celltype3'), 
        ('group2', 'sample4', 'celltype4')
    }
    assert set(complex_nested_adata_dict.keys()) == expected_keys

    # Verify hierarchy is flattened
    assert complex_nested_adata_dict.hierarchy == ('group', 'sample', 'celltype')

    # Verify data integrity for a few key entries
    assert np.array_equal(complex_nested_adata_dict[('group1', 'sample1', 'celltype1')].X, 
                         np.array([[1, 2]]))
    assert complex_nested_adata_dict[('group2', 'sample3', 'celltype3')].obs['condition'].iloc[0] == 'treat'
    assert complex_nested_adata_dict[('group1', 'sample2', 'celltype2')].uns['sample'] == 'B'

def test_adata_dict_set_hierarchy_flat(simple_adata_dict):
    """Test set_hierarchy with flat structure."""
    # Test reorganizing hierarchy
    simple_adata_dict.set_hierarchy(['sample'])
    assert simple_adata_dict.hierarchy == ('sample',)
    assert all(isinstance(k, tuple) and len(k) == 1 for k in simple_adata_dict.keys())

def test_adata_dict_set_hierarchy_nested(nested_adata_dict):
    """Test set_hierarchy with nested structure."""
    # Test restructuring from ('group', 'sample') to nested ['sample', ['group']]
    nested_adata_dict.set_hierarchy(['sample', ['group']])

    # Check the new structure maintains the same data with new nesting
    assert isinstance(nested_adata_dict[('sample1',)], AdataDict)
    assert isinstance(nested_adata_dict[('sample2',)], AdataDict)
    assert isinstance(nested_adata_dict[('sample3',)], AdataDict)
    assert isinstance(nested_adata_dict[('sample4',)], AdataDict)
    assert isinstance(nested_adata_dict[('sample1',)][('group1',)], ad.AnnData)
    assert isinstance(nested_adata_dict[('sample2',)][('group1',)], ad.AnnData)
    assert isinstance(nested_adata_dict[('sample3',)][('group2',)], ad.AnnData)
    assert isinstance(nested_adata_dict[('sample4',)][('group2',)], ad.AnnData)

    # Verify the hierarchy property is updated correctly
    assert nested_adata_dict.hierarchy == ('sample', ('group',))

def test_adata_dict_property_passthrough(simple_adata_dict):
    """Test property passthrough to AnnData objects."""
    # Test property access
    obs_dfs = simple_adata_dict.obs
    assert isinstance(obs_dfs, dict)
    assert all(isinstance(df, pd.DataFrame) for df in obs_dfs.values())

    var_dfs = simple_adata_dict.var
    assert isinstance(var_dfs, dict)
    assert all(isinstance(df, pd.DataFrame) for df in var_dfs.values())

    # Test method calls still work
    copies = simple_adata_dict.copy()
    assert isinstance(copies, dict)
    assert all(isinstance(v, ad.AnnData) for v in copies.values())

def test_nested_adata_dict_property_passthrough(nested_adata_dict):
    """Test property passthrough in nested AdataDict."""
    # Test property access at each level
    for group_dict in nested_adata_dict.values():
        obs_dfs = group_dict.obs
        assert isinstance(obs_dfs, dict)
        assert all(isinstance(df, pd.DataFrame) for df in obs_dfs.values())

        var_dfs = group_dict.var
        assert isinstance(var_dfs, dict)
        assert all(isinstance(df, pd.DataFrame) for df in var_dfs.values())

def test_error_handling(error_prone_adata_dict):
    """Test error handling in method passthrough."""
    # Test that errors in individual AnnData objects don't break the whole structure
    with pytest.raises(ValueError):
        error_prone_adata_dict[('value_error',)].problematic_method()

    with pytest.raises(KeyError):
        error_prone_adata_dict[('key_error',)].problematic_method()

    # The good sample should still work
    assert error_prone_adata_dict[('good_sample',)].uns['status'] == 'ok'

def test_adata_dict_hierarchy_property(simple_adata_dict, nested_adata_dict):
    """Test hierarchy property."""
    assert simple_adata_dict.hierarchy == ('sample',)
    assert nested_adata_dict.hierarchy == ('group', ('sample',))

    # Test hierarchy after modification
    simple_adata_dict.set_hierarchy(['sample'])
    assert simple_adata_dict.hierarchy == ('sample',)

def test_flatten_nesting_list(nested_adata_dict):
    """Test flatten_nesting_list method."""
    nested_list = ['group', ['sample', 'condition']]
    flattened = nested_adata_dict.flatten_nesting_list(nested_list)
    assert flattened == ['group', 'sample', 'condition']

def test_get_levels(nested_adata_dict):
    """Test get_levels method."""
    nesting_list = ['group', ['sample']]
    levels = nested_adata_dict.get_levels(nesting_list)
    assert levels == [['group'], ['sample']]

    # Test more complex nesting
    complex_nesting = ['group', ['sample', ['condition']]]
    complex_levels = nested_adata_dict.get_levels(complex_nesting)
    assert complex_levels == [['group'], ['sample'], ['condition']]


# test for main methods called in from utils and fapply:
def add_param(adata, *, extra_param=None):
    """Simple test function that adds a parameter to adata.uns."""
    adata.uns['extra_param'] = extra_param

def return_func(adata, *, extra_param=None):  # pylint: disable=unused-argument
    """Simple test function that returns a value."""
    if extra_param:
        return extra_param
    return 'done'

def test_fapply_wrapper(simple_adata_dict):
    """Test fapply wrapper with a simple function."""
    simple_adata_dict.fapply(add_param, extra_param='test_value')

    # Check that function was applied to all AnnData objects
    for adata in simple_adata_dict.values():
        assert 'extra_param' in adata.uns
        assert adata.uns['extra_param'] == 'test_value'

def test_fapply_return_wrapper(simple_adata_dict):
    """Test fapply_return wrapper with a function that returns values."""
    # Test with default return (as dict)
    result = simple_adata_dict.fapply_return(return_func, extra_param='test_value')
    assert isinstance(result, dict)
    assert all(v == 'test_value' for v in result.values())

    # Test with return_as_adata_dict=True
    result = simple_adata_dict.fapply_return(return_func, return_as_adata_dict=True)
    assert isinstance(result, AdataDict)
    assert all(v == 'done' for v in result.values())

def test_set_var_index_wrapper(simple_adata_dict):
    """Test set_var_index wrapper."""
    # Store original gene_ids
    original_gene_ids = {key: adata.var['gene_id'].copy() for key, adata in simple_adata_dict.items()}

    # Set index using gene_id
    simple_adata_dict.set_var_index('gene_id')

    # Verify the index was set correctly in all AnnData objects
    for key, adata in simple_adata_dict.items():
        assert all(adata.var.index == original_gene_ids[key])

def test_set_obs_index_wrapper(simple_adata_dict):
    """Test set_obs_index wrapper."""
    # Store original cell_ids
    original_cell_ids = {key: adata.obs['cell_id'].copy() for key, adata in simple_adata_dict.items()}

    # Set index using cell_id
    simple_adata_dict.set_obs_index('cell_id')

    # Verify the index was set correctly in all AnnData objects
    for key, adata in simple_adata_dict.items():
        assert all(adata.obs.index == original_cell_ids[key])
