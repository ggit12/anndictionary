"""
unit tets for anndict.utils.anndata_
"""

import pytest
import numpy as np
import pandas as pd

from anndict.utils.anndata_ import (
    remove_genes,
    add_col_to_pd_df,
    add_col_to_adata_obs,
    add_col_to_adata_var,
    convert_obs_col_to_category,
    convert_obs_col_to_string,
    convert_obs_index_to_str,
    get_adata_columns,
    filter_gene_list,
)


def test_remove_genes(simple_adata):
    """Test removing genes from AnnData object."""
    original_n_vars = simple_adata.n_vars
    genes_to_remove = ["gene1"]

    remove_genes(simple_adata, genes_to_remove)

    assert simple_adata.n_vars == original_n_vars - 1
    assert "gene1" not in simple_adata.var_names
    assert "gene2" in simple_adata.var_names


def test_remove_nonexistent_genes(simple_adata):
    """Test removing genes that don't exist in the dataset."""
    original_n_vars = simple_adata.n_vars
    genes_to_remove = ["nonexistent_gene"]

    remove_genes(simple_adata, genes_to_remove)

    assert simple_adata.n_vars == original_n_vars
    assert list(simple_adata.var_names) == ["gene1", "gene2"]

def test_add_col_to_pd_df_different_index_dtypes():
    """Test that the function works for differrent dtypes of index."""
    df = pd.DataFrame(index=["a", "b"])

    # Test string indices
    add_col_to_pd_df(df, indices=["a"], values=[42], new_col_name="str_idx_int")
    assert isinstance(df["str_idx_int"].dtype, pd.Int64Dtype)
    assert df.loc["a", "str_idx_int"] == 42
    assert pd.isna(df.loc["b", "str_idx_int"])

    add_col_to_pd_df(df, indices=["b"], values=[3.14], new_col_name="str_idx_float")
    assert df["str_idx_float"].dtype == np.float64
    assert df.loc["b", "str_idx_float"] == 3.14
    assert np.isnan(df.loc["a", "str_idx_float"])

    # Test integer indices
    add_col_to_pd_df(df, indices=[0], values=[True], new_col_name="int_idx_bool")
    assert isinstance(df["int_idx_bool"].dtype, pd.BooleanDtype)
    assert df.iloc[0]["int_idx_bool"] == True # pylint: disable=singleton-comparison
    assert pd.isna(df.iloc[1]["int_idx_bool"])

    # Test boolean mask
    bool_mask = [True, False]
    add_col_to_pd_df(df, indices=bool_mask, values=["test"], new_col_name="bool_idx_str")
    assert df["bool_idx_str"].dtype == object
    assert df.iloc[0]["bool_idx_str"] == "test"
    assert pd.isna(df.iloc[1]["bool_idx_str"])

    # Test multiple values for each index type
    add_col_to_pd_df(df, indices=["a", "b"], values=[1, 2], new_col_name="multi_str_idx")
    assert isinstance(df["multi_str_idx"].dtype, pd.Int64Dtype)
    assert df.loc["a", "multi_str_idx"] == 1
    assert df.loc["b", "multi_str_idx"] == 2

    add_col_to_pd_df(df, indices=[0, 1], values=[1.1, 2.2], new_col_name="multi_int_idx")
    assert df["multi_int_idx"].dtype == np.float64
    assert df.iloc[0]["multi_int_idx"] == 1.1
    assert df.iloc[1]["multi_int_idx"] == 2.2

    bool_mask = [True, True]
    add_col_to_pd_df(df, indices=bool_mask, values=[True, False], new_col_name="multi_bool_idx")
    assert isinstance(df["multi_bool_idx"].dtype, pd.BooleanDtype)
    assert df.iloc[0]["multi_bool_idx"] == True # pylint: disable=singleton-comparison
    assert df.iloc[1]["multi_bool_idx"] == False # pylint: disable=singleton-comparison


def test_add_col_to_pd_df_different_data_dtypes():
    """Test that dtypes are preserved for different input types."""
    df = pd.DataFrame(index=["a", "b"])
    indices = df.index

    # Test integer values - now using nullable integer type
    int_values = [1, 2]
    add_col_to_pd_df(df, indices, int_values, "int_col")
    assert df["int_col"].dtype == pd.Int64Dtype()
    pd.testing.assert_series_equal(
        df["int_col"],
        pd.Series(int_values, index=indices, dtype=pd.Int64Dtype(), name="int_col"),
    )

    # Test float values
    float_values = [1.1, 2.2]
    add_col_to_pd_df(df, indices, float_values, "float_col")
    assert df["float_col"].dtype == pd.Series(float_values[0]).dtype
    pd.testing.assert_series_equal(
        df["float_col"],
        pd.Series(
            float_values,
            index=indices,
            dtype=pd.Series(float_values[0]).dtype,
            name="float_col",
        ),
    )

    # Test string values
    str_values = ["A", "B"]
    add_col_to_pd_df(df, indices, str_values, "str_col")
    assert df["str_col"].dtype == pd.Series(str_values[0]).dtype
    pd.testing.assert_series_equal(
        df["str_col"],
        pd.Series(
            str_values,
            index=indices,
            dtype=pd.Series(str_values[0]).dtype,
            name="str_col",
        ),
    )

    # Test boolean values
    bool_values = [True, False]
    add_col_to_pd_df(df, indices, bool_values, "bool_col")
    assert df["bool_col"].dtype == pd.BooleanDtype()
    pd.testing.assert_series_equal(
        df["bool_col"],
        pd.Series(bool_values, index=indices, dtype=pd.BooleanDtype(), name="bool_col"),
    )


def test_add_col_to_pd_df_some_indices():
    """Test adding columns to a DataFrame with only some indices."""
    df = pd.DataFrame(index=["a", "b"])
    indices = df.index[:1]

    # Test integer values
    int_values = [1]
    add_col_to_pd_df(df, indices, int_values, "partial_int_col")
    assert df["partial_int_col"].dtype == pd.Int64Dtype()

    # Create expected series with pd.NA for the unspecified index
    expected = pd.Series(
        pd.NA, index=df.index, dtype=pd.Int64Dtype(), name="partial_int_col"
    )
    expected[indices] = int_values

    pd.testing.assert_series_equal(df["partial_int_col"], expected)


def test_add_col_to_adata_obs_and_var(simple_adata):
    """Test adding columns to both adata.obs and adata.var"""
    # Test adding to obs
    obs_indices = ["ATG"]  # Using actual index from fixture
    obs_values = [1]
    add_col_to_adata_obs(simple_adata, obs_indices, obs_values, "test_obs_col")

    # Verify obs modification
    assert "test_obs_col" in simple_adata.obs.columns
    expected_obs = pd.Series(
        pd.NA, index=simple_adata.obs.index, dtype=pd.Int64Dtype(), name="test_obs_col"
    )
    expected_obs[obs_indices] = obs_values
    pd.testing.assert_series_equal(simple_adata.obs["test_obs_col"], expected_obs)

    # Test adding to var
    var_indices = ["gene1"]  # Using actual index from fixture
    var_values = ["test"]
    add_col_to_adata_var(simple_adata, var_indices, var_values, "test_var_col")

    # Verify var modification
    assert "test_var_col" in simple_adata.var.columns
    expected_var = pd.Series(
        pd.NA, index=simple_adata.var.index, dtype=str, name="test_var_col"
    )
    expected_var[var_indices] = var_values
    pd.testing.assert_series_equal(simple_adata.var["test_var_col"], expected_var)


def test_add_col_to_adata_errors(simple_adata):
    """Test error handling in adata wrappers"""
    # Test index not in adata
    with pytest.raises((KeyError, IndexError)):
        add_col_to_adata_obs(simple_adata, ["invalid_index"], [1], "test_col")

    # Test mismatched lengths - use pandas' actual error message
    with pytest.raises(
        ValueError,
        match="cannot set using a list-like indexer with a different length than the value",
    ):
        add_col_to_adata_var(simple_adata, ["gene1", "gene2"], [1], "test_col")


def test_convert_obs_col_to_category(simple_adata):
    """Test converting obs columns to category dtype."""
    # Add test columns
    simple_adata.obs["test_str"] = ["A", "B"]
    simple_adata.obs["test_int"] = [1, 2]

    # Test single column conversion
    convert_obs_col_to_category(simple_adata, "test_str")
    assert isinstance(simple_adata.obs["test_str"].dtype, pd.CategoricalDtype)

    # Test multiple columns conversion
    convert_obs_col_to_category(simple_adata, ["test_int", "cell_type"])
    assert isinstance(simple_adata.obs["test_int"].dtype, pd.CategoricalDtype)
    assert isinstance(simple_adata.obs["cell_type"].dtype, pd.CategoricalDtype)

    # Test error for non-existent column
    with pytest.raises(ValueError):
        convert_obs_col_to_category(simple_adata, "nonexistent_col")


def test_convert_obs_col_to_string(simple_adata):
    """Test converting obs columns to string dtype."""
    # Add test columns
    simple_adata.obs["test_int"] = [1, 2]
    simple_adata.obs["test_float"] = [1.1, 2.2]

    # Test single column conversion
    convert_obs_col_to_string(simple_adata, "test_int")
    assert simple_adata.obs["test_int"].dtype == np.dtype("O")
    assert all(isinstance(x, str) for x in simple_adata.obs["test_int"])

    # Test multiple columns conversion
    convert_obs_col_to_string(simple_adata, ["test_float", "cell_type"])
    assert simple_adata.obs["test_float"].dtype == np.dtype("O")
    assert simple_adata.obs["cell_type"].dtype == np.dtype("O")

    # Test error for non-existent column
    with pytest.raises(ValueError):
        convert_obs_col_to_string(simple_adata, "nonexistent_col")


def test_convert_obs_index_to_str(simple_adata):
    """Test converting obs index to string dtype."""
    # Set numeric index
    simple_adata.obs.index = pd.Index([1, 2])

    convert_obs_index_to_str(simple_adata)
    assert simple_adata.obs.index.dtype == np.dtype("O")
    assert all(isinstance(x, str) for x in simple_adata.obs.index)


def test_get_adata_columns_input_strings(simple_adata):
    """Test getting columns based on string patterns using string inputs."""
    # Add test columns
    simple_adata.obs["test_a"] = [1, 2]
    simple_adata.obs["test_b"] = [3, 4]
    simple_adata.obs["other_a"] = [5, 6]
    simple_adata.obs["other_b"] = [7, 8]

    # Test starts_with
    assert set(get_adata_columns(simple_adata, starts_with="test_")) == {
        "test_a",
        "test_b",
    }

    # Test ends_with
    assert set(get_adata_columns(simple_adata, ends_with="_a")) == {"test_a", "other_a"}

    # Test contains
    assert set(get_adata_columns(simple_adata, contains="other")) == {
        "other_a",
        "other_b",
    }

    # Test not_starts_with
    assert set(
        get_adata_columns(simple_adata, contains="", not_starts_with="test_")
    ) == {"cell_type", "cell_id", "other_a", "other_b"}

    # Test multiple conditions
    assert set(
        get_adata_columns(simple_adata, starts_with="test_", not_ends_with="_b")
    ) == {"test_a"}


def test_get_adata_columns_input_lists(simple_adata):
    """Test getting columns based on string patterns."""
    # Add test columns
    simple_adata.obs["test_a"] = [1, 2]
    simple_adata.obs["test_b"] = [3, 4]
    simple_adata.obs["other_a"] = [5, 6]
    simple_adata.obs["other_b"] = [7, 8]

    # Test starts_with
    assert set(get_adata_columns(simple_adata, starts_with=["test_"])) == {
        "test_a",
        "test_b",
    }

    # Test ends_with
    assert set(get_adata_columns(simple_adata, ends_with=["_a"])) == {
        "test_a",
        "other_a",
    }

    # Test contains
    assert set(get_adata_columns(simple_adata, contains=["other"])) == {
        "other_a",
        "other_b",
    }

    # Test not_starts_with
    all_cols = set(simple_adata.obs.columns)
    test_cols = set(get_adata_columns(simple_adata, starts_with=["test_"]))
    assert (
        set(get_adata_columns(simple_adata, contains=[""], not_starts_with=["test_"]))
        == all_cols - test_cols
    )

    # Test multiple conditions
    assert set(
        get_adata_columns(simple_adata, starts_with=["test_"], not_ends_with=["_b"])
    ) == {"test_a"}


def test_filter_gene_list(simple_adata):
    """Test filtering gene list based on presence in adata."""
    # Test exact matches
    gene_list = ["gene1", "gene2", "nonexistent_gene"]
    filtered_genes = filter_gene_list(simple_adata, gene_list)
    assert set(filtered_genes) == {"gene1", "gene2"}

    # Test case insensitive matching
    gene_list = ["GENE1", "Gene2"]
    filtered_genes = filter_gene_list(simple_adata, gene_list)
    assert set(filtered_genes) == {"gene1", "gene2"}

    # Test empty result
    gene_list = ["nonexistent_gene"]
    filtered_genes = filter_gene_list(simple_adata, gene_list)
    assert len(filtered_genes) == 0
