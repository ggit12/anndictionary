"""
This module contains core functions for de novo annotation of cells based on marker genes and LLMs.
The functions in this module are called by other annotation functions.
We include these functions in the docs for reference, but you should not generally use them directly.
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
#disable pylint false positives

import pytest
import pandas as pd

from anndict.annotate.cells.de_novo.base import ai_annotate, ai_annotate_by_comparison

# Mock function to simulate cell type annotation based on marker genes
def mock_annotate_function(gene_list, tissue=None):
    """Mock function that returns a cell type based on marker genes"""
    if "CD3" in gene_list:
        return "T cell"
    elif "CD19" in gene_list:
        return "B cell"
    else:
        return "Unknown"

def test_ai_annotate(mock_adata):
    """Test basic functionality of ai_annotate."""
    result_df = ai_annotate(
        func=mock_annotate_function,
        adata=mock_adata,
        groupby="clusters",
        n_top_genes=10,
        new_label_column="cell_types",
        tissue_of_origin_col="tissue",
    )

    # Test assertions
    assert (
        "cell_types" in mock_adata.obs.columns
    ), "New label column not added to adata.obs"
    assert isinstance(result_df, pd.DataFrame), "Function should return a DataFrame"
    assert (
        "clusters" in result_df.columns
    ), "Result DataFrame should contain groupby column"
    assert (
        "cell_types" in result_df.columns
    ), "Result DataFrame should contain new label column"
    assert (
        "top_10_genes" in result_df.columns
    ), "Result DataFrame should contain top genes column"

    # Test categorical dtype conversion
    assert isinstance(
        mock_adata.obs["cell_types"].dtype, pd.CategoricalDtype
    ), "New label column should be categorical"

@pytest.mark.filterwarnings('ignore::pandas.errors.PerformanceWarning') #the mock_adata_many_clusters fixture triggers an irrelevant fragmented df pandas warning
def test_ai_annotate_warnings(mock_adata_many_clusters):
    """Test warnings for too many categories."""
    with pytest.warns(UserWarning, match="may result in slow runtimes"):
        ai_annotate(
            func=mock_annotate_function,
            adata=mock_adata_many_clusters,
            groupby="clusters",
            n_top_genes=10,
            new_label_column="cell_types",
        )

def test_ai_annotate_missing_tissue(mock_adata_no_tissue_or_cell_type):
    """Test behavior when tissue_of_origin_col doesn't exist."""
    with pytest.warns(UserWarning, match="Tissue of origin column .* not found"):
        ai_annotate(
            func=mock_annotate_function,
            adata=mock_adata_no_tissue_or_cell_type,
            groupby="clusters",
            n_top_genes=10,
            new_label_column="cell_types",
            tissue_of_origin_col="nonexistent_tissue",
        )

def test_ai_annotate_rerun_diffexp(mock_adata_no_diffexp):
    """Test automatic rerunning of rank_genes_groups."""
    _ = ai_annotate(
        func=mock_annotate_function,
        adata=mock_adata_no_diffexp,
        groupby="clusters",
        n_top_genes=10,
        new_label_column="cell_types",
    )

    assert (
        "rank_genes_groups" in mock_adata_no_diffexp.uns
    ), "rank_genes_groups should be run automatically"
    assert (
        mock_adata_no_diffexp.uns["rank_genes_groups"]["params"]["groupby"]
        == "clusters"
    )





#Functions for ai_annotate_by_comparison# Mock function to simulate cell type annotation based on marker genes
def mock_annotate_function_comparison(gene_lists, tissues=None, cell_types=None):
    """Mock function that returns cell types based on marker genes for multiple clusters"""
    annotations = []
    for gene_list in gene_lists:
        if "CD3" in gene_list:
            annotations.append("T cell")
        elif "CD19" in gene_list:
            annotations.append("B cell")
        else:
            annotations.append("Unknown")
    return annotations

def test_ai_annotate_by_comparison(mock_adata):
    """Test basic functionality of ai_annotate_by_comparison."""
    result_df = ai_annotate_by_comparison(
        func=mock_annotate_function_comparison,
        adata=mock_adata,
        groupby="clusters",
        n_top_genes=10,
        new_label_column="cell_types",
        tissue_of_origin_col="tissue",
        cell_type_of_origin_col="cell_type_of_origin"
    )

    # Test assertions
    assert "cell_types" in mock_adata.obs.columns, "New label column not added to adata.obs"
    assert isinstance(result_df, pd.DataFrame), "Function should return a DataFrame"
    assert "clusters" in result_df.columns, "Result DataFrame should contain groupby column"
    assert "cell_types" in result_df.columns, "Result DataFrame should contain new label column"
    assert "top_10_genes" in result_df.columns, "Result DataFrame should contain top genes column"

    # Test categorical dtype conversion
    assert isinstance(
        mock_adata.obs["cell_types"].dtype, pd.CategoricalDtype
    ), "New label column should be categorical"

def test_ai_annotate_by_comparison_missing_tissue(mock_adata_no_tissue_or_cell_type):
    """Test behavior when tissue_of_origin_col doesn't exist."""
    with pytest.warns(UserWarning, match="Tissue of origin column .* not found"):
        ai_annotate_by_comparison(
            func=mock_annotate_function_comparison,
            adata=mock_adata_no_tissue_or_cell_type,
            groupby="clusters",
            n_top_genes=10,
            new_label_column="cell_types",
            tissue_of_origin_col="nonexistent_tissue"
        )

def test_ai_annotate_by_comparison_missing_cell_type(mock_adata_no_tissue_or_cell_type):
    """Test behavior when cell_type_of_origin_col doesn't exist."""
    with pytest.warns(UserWarning, match="Cell type of origin column .* not found"):
        ai_annotate_by_comparison(
            func=mock_annotate_function_comparison,
            adata=mock_adata_no_tissue_or_cell_type,
            groupby="clusters",
            n_top_genes=10,
            new_label_column="cell_types",
            cell_type_of_origin_col="nonexistent_cell_type"
        )

def test_ai_annotate_by_comparison_rerun_diffexp(mock_adata_no_diffexp):
    """Test automatic rerunning of rank_genes_groups."""
    _ = ai_annotate_by_comparison(
        func=mock_annotate_function_comparison,
        adata=mock_adata_no_diffexp,
        groupby="clusters",
        n_top_genes=10,
        new_label_column="cell_types"
    )

    assert "rank_genes_groups" in mock_adata_no_diffexp.uns, "rank_genes_groups should be run automatically"
    assert mock_adata_no_diffexp.uns["rank_genes_groups"]["params"]["groupby"] == "clusters"
