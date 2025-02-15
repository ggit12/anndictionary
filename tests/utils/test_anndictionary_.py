"""
unit tests for anndict.utils.anndictionary_
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anndict.utils.anndictionary_ import (
    enforce_semantic_list,
    make_names,
    normalize_string,
    normalize_label,
    create_color_map,
    get_slurm_cores,
    summarize_metadata,
)


def test_enforce_semantic_list():
    """Test enforce_semantic_list function with various inputs."""
    # Valid cases
    assert enforce_semantic_list(["gene1", "gene2", "gene3"]) is True
    assert enforce_semantic_list(["CD4+ T cells", "B cells", "NK cells"]) is True

    # Invalid cases
    with pytest.raises(ValueError):
        enforce_semantic_list(["1", "2", "3"])

    with pytest.raises(ValueError):
        enforce_semantic_list([1, 2, 3])

    with pytest.raises(ValueError):
        enforce_semantic_list(["gene1", "2", "gene3"])


def test_make_names():
    """Test make_names function."""
    # Test basic name conversion
    assert make_names(["Hello World", "Test-123", "Sample.1"]) == [
        "Hello_World",
        "Test_123",
        "Sample_1",
    ]

    # Test duplicate handling
    assert make_names(["test", "test", "test"]) == ["test", "test.1", "test.2"]

    # Test special characters
    assert make_names(["$special@", "#chars%"]) == ["_special_", "_chars_"]


def test_normalize_string():
    """Test normalize_string function."""
    assert normalize_string("Hello, World!") == "hello world"
    assert normalize_string("Gene-1_2@3") == "gene1_23"
    assert normalize_string("  Spaces  ") == "  spaces  "
    assert normalize_string("") == ""


def test_normalize_label():
    """Test normalize_label function."""
    assert normalize_label("Hello, World!") == "hello world"
    assert normalize_label("  Spaces  ") == "spaces"
    assert normalize_label(np.nan) == "missing"
    assert normalize_label(pd.NA) == "missing"


def test_create_color_map(simple_adata):
    """Test create_color_map function using simple_adata fixture."""
    # Add both categorical and continuous data to test
    simple_adata.obs["continuous"] = [1.0, 2.0]
    simple_adata.obs["categorical"] = ["A", "B"]

    color_map = create_color_map(simple_adata, ["continuous", "categorical"])

    # Test continuous mapping
    assert isinstance(color_map["continuous"], plt.cm.ScalarMappable)
    assert color_map["continuous"].norm.vmin == 1.0
    assert color_map["continuous"].norm.vmax == 2.0

    # Test categorical mapping
    assert isinstance(color_map["categorical"], dict)
    assert len(color_map["categorical"]) == 2
    assert "A" in color_map["categorical"]
    assert "B" in color_map["categorical"]


def test_summarize_metadata(simple_adata):
    """Test summarize_metadata function using simple_adata fixture."""
    # Test single column summary
    summary = summarize_metadata(simple_adata, ["cell_type"])
    assert "cell_type" in summary
    assert len(summary["cell_type"]) == 2  # A and B categories

    # Add another column for joint frequency test
    simple_adata.obs["condition"] = ["ctrl", "treat"]
    summary = summarize_metadata(simple_adata, ["cell_type*condition"])
    assert "cell_type x condition" in summary
    assert summary["cell_type x condition"].shape == (2, 2)  # 2x2 contingency table


def test_summarize_metadata_missing_values(simple_adata):
    """Test summarize_metadata handling of missing values."""
    # Add column with missing values
    simple_adata.obs["missing_col"] = ["A", np.nan]
    summary = summarize_metadata(simple_adata, ["missing_col"])
    assert "missing_col" in summary
    assert pd.isna(
        summary["missing_col"].index
    ).any()  # Check if NaN is preserved in index


def test_summarize_metadata_multiple_columns(simple_adata_dict):
    """Test summarize_metadata with multiple columns using simple_adata_dict fixture."""
    adata = simple_adata_dict[("sample1",)]
    summary = summarize_metadata(adata, ["condition", "cell_id"])
    assert "condition" in summary
    assert "cell_id" in summary
    assert len(summary["condition"]) == 1  # All 'ctrl'
    assert len(summary["cell_id"]) == 2  # Two unique cell IDs


# Additional test for get_slurm_cores would depend on environment variables,
# so we'll need to mock those for testing
def test_get_slurm_cores(monkeypatch):
    """Test get_slurm_cores function with mocked environment variables."""
    # Test default values
    assert get_slurm_cores() == 1

    # Test with mocked environment variables
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    assert get_slurm_cores() == 8

    # Test with only one variable set
    monkeypatch.delenv("SLURM_NTASKS")
    assert get_slurm_cores() == 4
