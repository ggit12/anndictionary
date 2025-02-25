"""
unit tests for adata_dict.utils
"""
import pytest

from anndict.adata_dict.utils import check_and_create_stratifier

def test_single_strata_key(simple_adata):
    """Test function with a single stratification key."""
    result = check_and_create_stratifier(simple_adata, ['cell_type'])
    assert result == 'cell_type'
    assert result in simple_adata.obs.columns
    assert simple_adata.obs[result].dtype.name == 'category'
    assert list(simple_adata.obs[result]) == ['A', 'B']

def test_multiple_strata_keys(simple_adata):
    """Test function with multiple stratification keys."""
    result = check_and_create_stratifier(simple_adata, ['cell_type', 'cell_id'])
    assert result == 'cell_type_cell_id'
    assert result in simple_adata.obs.columns
    assert simple_adata.obs[result].dtype.name == 'category'
    assert list(simple_adata.obs[result]) == ['A_cell1', 'B_cell2']

def test_nonexistent_key(simple_adata):
    """Test function raises ValueError for nonexistent keys."""
    with pytest.raises(ValueError, match="one or more of your stratifying variables does not exist in adata.obs"):
        check_and_create_stratifier(simple_adata, ['nonexistent_key'])

def test_mixed_existing_nonexisting_keys(simple_adata):
    """Test function raises ValueError when some keys exist and others don't."""
    with pytest.raises(ValueError, match="one or more of your stratifying variables does not exist in adata.obs"):
        check_and_create_stratifier(simple_adata, ['cell_type', 'nonexistent_key'])

def test_empty_strata_keys(simple_adata):
    """Test function with empty list of keys."""
    with pytest.raises(ValueError, match="You specified strata_keys as an empty list. You must specify at least one stratifying variable"):
        check_and_create_stratifier(simple_adata, [])

def test_existing_combined_column(simple_adata):
    """Test function when combined column already exists."""
    # First create the combined column
    result1 = check_and_create_stratifier(simple_adata, ['cell_type', 'cell_id'])
    # Then call again with same keys
    result2 = check_and_create_stratifier(simple_adata, ['cell_type', 'cell_id'])

    assert result1 == result2
    assert result1 in simple_adata.obs.columns
    assert simple_adata.obs[result1].dtype.name == 'category'
    assert list(simple_adata.obs[result1]) == ['A_cell1', 'B_cell2']

def test_numeric_values(simple_adata):
    """Test function with numeric values in observation."""
    # Add numeric column
    simple_adata.obs['numeric_col'] = [1, 2]
    result = check_and_create_stratifier(simple_adata, ['cell_type', 'numeric_col'])
    assert result == 'cell_type_numeric_col'
    assert result in simple_adata.obs.columns
    assert simple_adata.obs[result].dtype.name == 'category'
    assert list(simple_adata.obs[result]) == ['A_1', 'B_2']

def test_null_values(simple_adata):
    """Test function with null values in observation."""
    # Add column with null value
    simple_adata.obs['null_col'] = ['value', None]
    result = check_and_create_stratifier(simple_adata, ['cell_type', 'null_col'])
    assert result == 'cell_type_null_col'
    assert result in simple_adata.obs.columns
    assert simple_adata.obs[result].dtype.name == 'category'
    assert list(simple_adata.obs[result]) == ['A_value', 'B_None']

def test_nan_values(simple_adata):
    """Test function with NaN values in observation."""
    # Add column with NaN value
    simple_adata.obs['nan_col'] = ['value', float('nan')]
    result = check_and_create_stratifier(simple_adata, ['cell_type', 'nan_col'])
    assert result == 'cell_type_nan_col'
    assert result in simple_adata.obs.columns
    assert simple_adata.obs[result].dtype.name == 'category'
    assert list(simple_adata.obs[result]) == ['A_value', 'B_nan']

def test_duplicate_keys(simple_adata):
    """Test function with duplicate keys in input."""
    result = check_and_create_stratifier(simple_adata, ['cell_type', 'cell_type'])
    assert result == 'cell_type_cell_type'
    assert result in simple_adata.obs.columns
    assert simple_adata.obs[result].dtype.name == 'category'
    assert list(simple_adata.obs[result]) == ['A_A', 'B_B']
