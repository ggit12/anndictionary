"""
unit tests for write_adata_dict.
"""
# pylint: disable=protected-access

import os
import shutil
import json
import pytest
import scanpy as sc

from anndict.adata_dict import AdataDict, write_adata_dict, to_nested_tuple

def test_basic_write(simple_adata_dict, tmp_path):
    """Test basic functionality of writing AdataDict to disk."""
    base_dir = tmp_path / "test_write"

    # Write the AdataDict to disk
    write_adata_dict(simple_adata_dict, str(base_dir))

    # Check that the hierarchy file exists and contains correct data
    hierarchy_file = base_dir / "adata_dict.hierarchy.json"
    assert hierarchy_file.exists()
    with open(hierarchy_file, "r", encoding="utf-8") as f:
        saved_hierarchy = json.load(f)
    assert to_nested_tuple(saved_hierarchy) == simple_adata_dict.hierarchy

    # Check that all files exist directly in base directory for flat hierarchy
    for key, adata in simple_adata_dict._flatten().items():
        expected_file = base_dir / f"{'_'.join(map(str, key))}.h5ad"
        assert expected_file.exists()

        # Load and verify the saved AnnData object
        loaded_adata = sc.read_h5ad(expected_file)
        assert loaded_adata.shape == adata.shape
        assert all(loaded_adata.obs_names == adata.obs_names)
        assert all(loaded_adata.var_names == adata.var_names)
        assert loaded_adata.uns['sample'] == adata.uns['sample']

def test_nested_write(nested_adata_dict, tmp_path):
    """Test writing nested AdataDict with multiple hierarchy levels."""
    base_dir = tmp_path / "test_nested"

    write_adata_dict(nested_adata_dict, str(base_dir))

    # Verify hierarchy
    with open(base_dir / "adata_dict.hierarchy.json", "r", encoding="utf-8") as f:
        saved_hierarchy = json.load(f)
    assert to_nested_tuple(saved_hierarchy) == nested_adata_dict.hierarchy

    # Check all files in the nested structure
    for key, adata in nested_adata_dict._flatten().items():
        # Only use outer keys for directory structure
        outer_keys = key[:-1]  # All but the last key tuple
        inner_keys = key[-1:]  # Just the last key tuple
        
        if outer_keys:
            path = base_dir.joinpath(*map(str, outer_keys))
            expected_file = path / f"{'_'.join(map(str, inner_keys))}.h5ad"
        else:
            # If no outer keys, file should be directly in base directory
            expected_file = base_dir / f"{'_'.join(map(str, inner_keys))}.h5ad"
            
        assert expected_file.exists()

        loaded_adata = sc.read_h5ad(expected_file)
        assert loaded_adata.shape == adata.shape
        assert loaded_adata.uns['group'] == adata.uns['group']
        assert loaded_adata.uns['sample'] == adata.uns['sample']

def test_complex_nested_write(complex_nested_adata_dict, tmp_path):
    """Test writing complex nested AdataDict with multiple indices at deepest level."""
    base_dir = tmp_path / "test_complex_nested"

    write_adata_dict(complex_nested_adata_dict, str(base_dir))

    # Verify hierarchy
    with open(base_dir / "adata_dict.hierarchy.json", "r", encoding="utf-8") as f:
        saved_hierarchy = json.load(f)
    assert to_nested_tuple(saved_hierarchy) == complex_nested_adata_dict.hierarchy

    # Check all files in the nested structure
    for key, adata in complex_nested_adata_dict._flatten().items():
        # The group is the outer key for directory structure
        outer_keys = key[:-2]  # Just the group
        inner_keys = key[-2:]  # Sample and celltype together
        
        if outer_keys:
            path = base_dir.joinpath(*map(str, outer_keys))
            expected_file = path / f"{'_'.join(map(str, inner_keys))}.h5ad"
        else:
            # If no outer keys, file should be directly in base directory
            expected_file = base_dir / f"{'_'.join(map(str, inner_keys))}.h5ad"

        assert expected_file.exists()

        loaded_adata = sc.read_h5ad(expected_file)
        assert loaded_adata.shape == adata.shape
        assert loaded_adata.uns['group'] == adata.uns['group']
        assert loaded_adata.uns['sample'] == adata.uns['sample']

def test_file_prefix(simple_adata_dict, tmp_path):
    """Test writing with a file prefix."""
    base_dir = tmp_path / "test_prefix"
    prefix = "prefix_"

    write_adata_dict(simple_adata_dict, str(base_dir), file_prefix=prefix)

    # Check files with prefix directly in base directory
    for key in simple_adata_dict._flatten().keys():
        expected_file = base_dir / f"{prefix}{'_'.join(map(str, key))}.h5ad"
        assert expected_file.exists()

def test_special_characters(adata_dict_with_special_character_keys, tmp_path):
    """Test handling of special characters in keys."""
    base_dir = tmp_path / "test_special"
    write_adata_dict(adata_dict_with_special_character_keys, str(base_dir))

    def safe_str(val):
        """Mirror the safe_str function from write_adata_dict"""
        s = str(val)
        return "".join(c if c not in r'\/:*?"<>|' else "_" for c in s)

    # Verify files exist and can be read
    for key in adata_dict_with_special_character_keys._flatten().keys():
        # Get file name using safe versions of the strings
        file_name = f"{'_'.join(map(safe_str, key))}.h5ad"
        # Files should be directly in base directory per docstring
        expected_file = base_dir / file_name

        assert expected_file.exists()
        assert sc.read_h5ad(expected_file) is not None

def test_existing_directory(simple_adata_dict, tmp_path):
    """Test that writing to existing directory raises an error."""
    base_dir = tmp_path / "test_existing"
    os.makedirs(base_dir)

    with pytest.raises(FileExistsError):
        write_adata_dict(simple_adata_dict, str(base_dir))

def test_empty_adata_dict(tmp_path):
    """Test handling of empty AdataDict."""
    empty_dict = AdataDict({}, hierarchy=("sample",))
    base_dir = tmp_path / "test_empty"

    write_adata_dict(empty_dict, str(base_dir))

    # Should still create directory and hierarchy file
    assert base_dir.exists()
    hierarchy_file = base_dir / "adata_dict.hierarchy.json"
    assert hierarchy_file.exists()
    with open(hierarchy_file, "r", encoding="utf-8") as f:
        saved_hierarchy = json.load(f)
    assert to_nested_tuple(saved_hierarchy) == empty_dict.hierarchy

@pytest.fixture(autouse=True)
def cleanup(tmp_path):
    """Cleanup temporary files after each test."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)
