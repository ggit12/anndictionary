"""
Common test fixtures for de novo cell type annotation tests.
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc

from anndata import AnnData


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create mock data
    n_cells = 100
    n_genes = 50

    # Generate random expression data
    data = np.abs(np.random.normal(10, 2, (n_cells, n_genes)))

    # Create gene names including known marker genes
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    gene_names[0] = "CD3"  # Known T cell marker
    gene_names[1] = "CD19"  # Known B cell marker

    # Create cell groups/clusters
    clusters = np.array(["0", "1", "2"] * (n_cells // 3 + 1))[:n_cells]

    # Create mock tissue of origin data
    tissue_origin = np.array(["blood", "lymph node"] * (n_cells // 2 + 1))[:n_cells]
    cell_type_origin = np.array(["Macrophage", "Monocyte"] * (n_cells // 2 + 1))[:n_cells]

    # Create AnnData object
    adata = AnnData(
        X=data,
        obs=pd.DataFrame(
            {"clusters": clusters, "tissue": tissue_origin, "cell_type_of_origin": cell_type_origin},
            index=[f"Cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )

    # Run rank_genes_groups
    sc.tl.rank_genes_groups(adata, "clusters", method="t-test")

    return adata

@pytest.fixture
def mock_adata_for_cell_subtyping():
    """A mock AnnData for cell subtyping."""
    return AnnData(
            X=np.array(
                [
                    # T cells subcluster 0
                    [1, 2],
                    [2, 3],
                    # T cells subcluster 1
                    [3, 4],
                    [4, 5],
                    # B cells subcluster 0
                    [5, 6],
                    [6, 7],
                    # B cells subcluster 1
                    [7, 8],
                    [8, 9],
                ]
            ),
            obs=pd.DataFrame(
                {
                    "cell_type": pd.Categorical(
                        [
                            "T cell",
                            "T cell",
                            "T cell",
                            "T cell",
                            "B cell",
                            "B cell",
                            "B cell",
                            "B cell",
                        ]
                    ),
                    "subcluster": pd.Categorical(["0", "0", "1", "1", "0", "0", "1", "1"]),
                    "condition": pd.Categorical(
                        ["ctrl", "treat", "ctrl", "treat", "ctrl", "treat", "ctrl", "treat"]
                    ),  # Add condition for comparison
                    "tissue": ["blood"] * 8,
                }, index=[f"Cell_{i}" for i in range(8)]
            ),
            var=pd.DataFrame(
                {"gene_id": ["gene1", "gene2"], "gene_name": ["Gene_1", "Gene_2"]},
                index=["gene1", "gene2"],
            ))

@pytest.fixture
def mock_adata_no_tissue_or_cell_type():
    """Create a mock AnnData object without tissue information."""
    n_cells = 100
    n_genes = 50
    data = np.abs(np.random.normal(10, 2, (n_cells, n_genes)))
    clusters = np.array(["0", "1", "2"] * (n_cells // 3 + 1))[:n_cells]

    return AnnData(
        X=data,
        obs=pd.DataFrame(
            {"clusters": clusters}, index=[f"Cell_{i}" for i in range(n_cells)]
        ),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)]),
    )

@pytest.fixture
def mock_adata_no_diffexp():
    """Create a mock AnnData object without pre-computed differential expression."""
    n_cells = 100
    n_genes = 50
    data = np.abs(np.random.normal(10, 2, (n_cells, n_genes)))
    clusters = np.array(["0", "1", "2"] * (n_cells // 3 + 1))[:n_cells]

    return AnnData(
        X=data,
        obs=pd.DataFrame(
            {"clusters": clusters}, index=[f"Cell_{i}" for i in range(n_cells)]
        ),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)]),
    )

@pytest.fixture
def mock_adata_many_clusters():
    """Create a mock AnnData object with many clusters for testing warnings."""
    n_clusters = 51
    min_cells_per_cluster = 3  # Minimum cells needed for t-test
    n_cells = n_clusters * min_cells_per_cluster
    n_genes = 50

    # Generate random expression data
    data = np.abs(np.random.normal(10, 2, (n_cells, n_genes)))

    # Create clusters ensuring each has at least min_cells_per_cluster cells
    clusters = np.repeat(np.arange(n_clusters).astype(str), min_cells_per_cluster)

    return AnnData(
        X=data,
        obs=pd.DataFrame(
            {"clusters": clusters}, index=[f"Cell_{i}" for i in range(n_cells)]
        ),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)]),
    )

@pytest.fixture
def expected_label_results_for_two_groups(n_top_genes=2):
    """Expected results for two groups test"""
    return pd.DataFrame(
        {
            "cluster": ["0", "1"],
            "new_label_column": ["ai_label_1", "ai_label_2"],
            f"top_{n_top_genes}_genes": [["gene1", "gene2"], ["gene3", "gene2"]],
        }
    )
