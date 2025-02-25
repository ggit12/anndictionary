"""
This module contains the functions necessary to read :class:`AdataDict` objects from adata on disk.
"""

import os
import shutil

import pytest
import numpy as np
import pandas as pd

from anndict.adata_dict import read_adata_dict, write_adata_dict

def test_read_nonexistent_directory():
    """Test reading from a nonexistent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_adata_dict("/nonexistent/directory")

def test_read_missing_hierarchy_file(simple_adata_dict, tmp_path):
    """Test reading when hierarchy file is missing raises FileNotFoundError."""
    # Create a new subdirectory to ensure clean state
    test_dir = tmp_path / "test_dir"

    # Write the adata_dict
    write_adata_dict(simple_adata_dict, test_dir)

    # Remove hierarchy file
    os.remove(test_dir / "adata_dict.hierarchy.json")

    with pytest.raises(FileNotFoundError):
        read_adata_dict(test_dir)

def test_read_missing_db_file(simple_adata_dict, tmp_path):
    """Test reading when database file is missing raises FileNotFoundError."""
    test_dir = tmp_path / "test_dir"

    # Write the adata_dict
    write_adata_dict(simple_adata_dict, test_dir)

    # Remove db file
    os.remove(test_dir / "adata_dict.db.json")

    with pytest.raises(FileNotFoundError):
        read_adata_dict(test_dir)

def test_read_missing_adata_file(simple_adata_dict, tmp_path):
    """Test reading when an AnnData file is missing raises FileNotFoundError."""
    test_dir = tmp_path / "test_dir"

    # Write the adata_dict
    write_adata_dict(simple_adata_dict, test_dir)

    # Remove one of the .h5ad files
    h5ad_files = list(test_dir.glob("*.h5ad"))
    os.remove(h5ad_files[0])

    with pytest.raises(FileNotFoundError):
        read_adata_dict(test_dir)

def test_read_simple_adata_dict(simple_adata_dict, tmp_path):
    """Test reading a simple AdataDict with flat hierarchy."""
    test_dir = tmp_path / "test_dir"

    # Write then read the adata_dict
    write_adata_dict(simple_adata_dict, test_dir)
    read_dict = read_adata_dict(test_dir)

    # Check structure matches
    assert set(read_dict.keys()) == set(simple_adata_dict.keys())
    assert read_dict.hierarchy == simple_adata_dict.hierarchy

    # Check contents match for each AnnData object
    for key in simple_adata_dict.keys():
        original = simple_adata_dict[key]
        read = read_dict[key]

        # Compare basic attributes
        np.testing.assert_array_equal(original.X, read.X)
        pd.testing.assert_frame_equal(original.obs, read.obs)
        pd.testing.assert_frame_equal(original.var, read.var)
        assert original.uns == read.uns

def test_read_nested_adata_dict(nested_adata_dict, tmp_path):
    """Test reading a nested AdataDict with hierarchical structure."""
    test_dir = tmp_path / "test_dir"

    # Write then read the adata_dict
    write_adata_dict(nested_adata_dict, test_dir)
    read_dict = read_adata_dict(test_dir)

    # Check top-level structure
    assert set(read_dict.keys()) == set(nested_adata_dict.keys())
    assert read_dict.hierarchy == nested_adata_dict.hierarchy

    # Check nested structure and contents
    for group_key in nested_adata_dict.keys():
        original_group = nested_adata_dict[group_key]
        read_group = read_dict[group_key]

        assert set(read_group.keys()) == set(original_group.keys())

        for sample_key in original_group.keys():
            original = original_group[sample_key]
            read = read_group[sample_key]

            np.testing.assert_array_equal(original.X, read.X)
            pd.testing.assert_frame_equal(original.obs, read.obs)
            pd.testing.assert_frame_equal(original.var, read.var)
            assert original.uns == read.uns

def test_read_complex_nested_adata_dict(complex_nested_adata_dict, tmp_path):
    """Test reading a complex nested AdataDict with multiple indices at the deepest level."""
    test_dir = tmp_path / "test_dir"

    # Write then read the adata_dict
    write_adata_dict(complex_nested_adata_dict, test_dir)
    read_dict = read_adata_dict(test_dir)

    # Check hierarchy is preserved
    assert read_dict.hierarchy == complex_nested_adata_dict.hierarchy

    # Check top-level structure (groups)
    assert set(read_dict.keys()) == set(complex_nested_adata_dict.keys())

    # Check each group's contents
    for group_key in complex_nested_adata_dict.keys():
        original_group = complex_nested_adata_dict[group_key]
        read_group = read_dict[group_key]

        # Verify second level keys (sample, celltype combinations)
        assert set(read_group.keys()) == set(original_group.keys())

        # Check each AnnData object
        for key in original_group.keys():
            original = original_group[key]
            read = read_group[key]

            # Verify the key structure is preserved (should be a tuple of two elements)
            assert len(key) == 2  # (sample, celltype)

            # Compare contents
            np.testing.assert_array_equal(original.X, read.X)
            pd.testing.assert_frame_equal(original.obs, read.obs)
            pd.testing.assert_frame_equal(original.var, read.var)
            assert original.uns == read.uns


def test_read_special_characters(adata_dict_with_special_character_keys, tmp_path):
    """Test reading AdataDict with special characters in keys."""
    test_dir = tmp_path / "test_dir"

    # Write then read the adata_dict
    write_adata_dict(adata_dict_with_special_character_keys, test_dir)
    read_dict = read_adata_dict(test_dir)

    # Check structure matches
    assert set(read_dict.keys()) == set(adata_dict_with_special_character_keys.keys())
    assert read_dict.hierarchy == adata_dict_with_special_character_keys.hierarchy

    # Check contents match
    for key in adata_dict_with_special_character_keys.keys():
        original = adata_dict_with_special_character_keys[key]
        read = read_dict[key]
        np.testing.assert_array_equal(original.X, read.X)

def test_read_corrupted_hierarchy_file(simple_adata_dict, tmp_path):
    """Test reading with corrupted hierarchy.json raises ValueError."""
    test_dir = tmp_path / "test_dir"

    # Write the adata_dict
    write_adata_dict(simple_adata_dict, test_dir)

    # Corrupt the hierarchy file
    with open(test_dir / "adata_dict.hierarchy.json", "w", encoding="utf-8") as f:
        f.write("corrupted json{")

    with pytest.raises(ValueError):
        read_adata_dict(test_dir)

def test_read_corrupted_db_file(simple_adata_dict, tmp_path):
    """Test reading with corrupted db.json raises ValueError."""
    test_dir = tmp_path / "test_dir"

    # Write the adata_dict
    write_adata_dict(simple_adata_dict, test_dir)

    # Corrupt the db file
    with open(test_dir / "adata_dict.db.json", "w", encoding="utf-8") as f:
        f.write("corrupted json{")

    with pytest.raises(ValueError):
        read_adata_dict(test_dir)

@pytest.fixture(autouse=True)
def cleanup(tmp_path):
    """Cleanup temporary files after each test."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)
