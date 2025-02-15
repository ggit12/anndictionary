"""
unit tests for :func:`concatenate_adata_dict`.
"""

import pytest
import numpy as np

from anndict.adata_dict.concatenate import concatenate_adata_dict

def test_concatenate_simple_adata_dict(simple_adata_dict):
    """Test concatenation of a simple AdataDict with two samples."""
    result = concatenate_adata_dict(simple_adata_dict, new_col_name='sample')

    # Check basic properties
    assert result.n_obs == 4  # Total cells from both samples
    assert result.n_vars == 2  # Should maintain same number of genes
    assert all(col in result.obs.columns for col in ['condition', 'cell_id', 'sample'])

    # Check sample assignment
    assert list(result.obs['sample'].unique()) == [('sample1',), ('sample2',)]
    assert sum(result.obs['sample'] == ('sample1',)) == 2
    assert sum(result.obs['sample'] == ('sample2',)) == 2

    # Verify data integrity
    assert np.array_equal(result[result.obs['sample'] == ('sample1',)].X, 
                         np.array([[1, 2], [3, 4]]))
    assert np.array_equal(result[result.obs['sample'] == ('sample2',)].X, 
                         np.array([[5, 6], [7, 8]]))

def test_concatenate_nested_adata_dict(nested_adata_dict):
    """Test concatenation of a nested AdataDict with group/sample hierarchy."""
    result = concatenate_adata_dict(nested_adata_dict, new_col_name='group')

    # Check dimensions
    assert result.n_obs == 4  # Total cells across all groups/samples
    assert result.n_vars == 2  # Should maintain same number of genes

    # Check group assignment - expecting full tuples
    expected_groups = {('group1', 'sample1'), ('group1', 'sample2'), 
                      ('group2', 'sample3'), ('group2', 'sample4')}
    assert set(result.obs['group'].unique()) == expected_groups
    assert sum(result.obs['group'].isin([('group1', 'sample1'), ('group1', 'sample2')])) == 2
    assert sum(result.obs['group'].isin([('group2', 'sample3'), ('group2', 'sample4')])) == 2

    # Check conditions are preserved
    group1_mask = result.obs['group'].isin([('group1', 'sample1'), ('group1', 'sample2')])
    group2_mask = result.obs['group'].isin([('group2', 'sample3'), ('group2', 'sample4')])
    assert all(result[group1_mask].obs['condition'] == 'ctrl')
    assert all(result[group2_mask].obs['condition'] == 'treat')

def test_concatenate_complex_nested_adata_dict(complex_nested_adata_dict):
    """Test concatenation of a complex nested AdataDict with group/sample/celltype hierarchy."""
    result = concatenate_adata_dict(complex_nested_adata_dict, new_col_name='group')

    # Check dimensions
    assert result.n_obs == 4  # Total cells across all groups/samples
    assert result.n_vars == 2  # Should maintain same number of genes

    # Check group assignment - expecting full tuples
    expected_groups = {
        ('group1', 'sample1', 'celltype1'), 
        ('group1', 'sample2', 'celltype2'),
        ('group2', 'sample3', 'celltype3'), 
        ('group2', 'sample4', 'celltype4')
    }
    assert set(result.obs['group'].unique()) == expected_groups

    # Check group counts
    group1_mask = result.obs['group'].isin([
        ('group1', 'sample1', 'celltype1'), 
        ('group1', 'sample2', 'celltype2')
    ])
    group2_mask = result.obs['group'].isin([
        ('group2', 'sample3', 'celltype3'), 
        ('group2', 'sample4', 'celltype4')
    ])
    assert sum(group1_mask) == 2
    assert sum(group2_mask) == 2

    # Check conditions are preserved
    assert all(result[group1_mask].obs['condition'] == 'ctrl')
    assert all(result[group2_mask].obs['condition'] == 'treat')

    # Check data integrity
    assert np.array_equal(
        result[result.obs['group'] == ('group1', 'sample1', 'celltype1')].X,
        np.array([[1, 2]])
    )
    assert np.array_equal(
        result[result.obs['group'] == ('group2', 'sample4', 'celltype4')].X,
        np.array([[7, 8]])
    )

def test_concatenate_single_adata(simple_adata_dict):
    """Test concatenation when AdataDict contains only one AnnData object."""
    # Create single-entry AdataDict
    single_adata_dict = type(simple_adata_dict)({
        ('sample1',): simple_adata_dict[('sample1',)]
    }, hierarchy=('sample',))

    result = concatenate_adata_dict(single_adata_dict, new_col_name='sample')

    # Should return the original AnnData object with added sample column
    assert result.n_obs == 2
    assert result.n_vars == 2
    assert all(result.obs['sample'] == ('sample1',))
    assert np.array_equal(result.X, np.array([[1, 2], [3, 4]]))

def test_concatenate_empty_adata_dict(simple_adata_dict):
    """Test concatenation with empty AdataDict raises ValueError."""
    empty_dict = type(simple_adata_dict)({}, hierarchy=('sample',))

    with pytest.raises(ValueError, match="adata_dict is empty"):
        concatenate_adata_dict(empty_dict)

def test_concatenate_without_new_col_name(simple_adata_dict):
    """Test concatenation without specifying new_col_name."""
    result = concatenate_adata_dict(simple_adata_dict)

    # Basic checks
    assert result.n_obs == 4
    assert result.n_vars == 2
    assert 'sample' not in result.obs.columns  # No sample column should be added

def test_concatenate_with_special_characters(adata_dict_with_special_character_keys):
    """Test concatenation with special characters in keys."""
    result = concatenate_adata_dict(adata_dict_with_special_character_keys, 
                                  new_col_name='level1')

    # Check if special characters are preserved in the column
    unique_keys = set(result.obs['level1'])
    assert ("key with spaces", "special#chars") in unique_keys
    assert ("number123", "symbols!@") in unique_keys

def test_concatenate_with_custom_join(simple_adata_dict):
    """Test concatenation with custom join parameter."""
    result = concatenate_adata_dict(simple_adata_dict, join='inner')

    # Check if the join parameter was respected
    assert result.n_obs == 4
    assert result.n_vars == 2

def test_index_uniqueness(simple_adata_dict):
    """Test that original indices are preserved during concatenation."""
    result = concatenate_adata_dict(simple_adata_dict, new_col_name='sample')

    # Check if original indices are preserved
    expected_indices = ['AAG', 'AAC', 'TTG', 'TTC']
    assert all(idx in result.obs.index for idx in expected_indices)
