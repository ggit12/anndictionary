"""
Shared test fixtures for the ``AnnDictionary`` package.
This module provides mock objects and fixtures that simulate :class:`AnnData`.
behavior for testing purposes.
"""

import pytest
import numpy as np
import pandas as pd
import anndata as ad

from scipy import sparse
from anndict.adata_dict import AdataDict


# AnnData fixtures
@pytest.fixture
def simple_adata():
    """Single AnnData object with basic attributes."""
    return ad.AnnData(
        X=np.array([[1, 2], [3, 4]]),
        obs=pd.DataFrame(
            {"cell_type": ["A", "B"], "cell_id": ["cell1", "cell2"]},
            index=["ATG", "ATC"],
        ),
        var=pd.DataFrame(
            {"gene_id": ["gene1", "gene2"], "gene_name": ["Gene_1", "Gene 2"]},
            index=["gene1", "gene2"],
        ),
        uns={"experiment": "test"},
    )

@pytest.fixture
def simple_adata_with_two_groups():
    """Fixture for creating test AnnData object with multiple samples per group"""
    obs = pd.DataFrame(
        {
            "cluster": ["0", "0", "1", "1"],
            "condition": ["A", "A", "B", "B"],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
    X = np.array([[1, 1, 1, 1], [5, 4, 5, 4], [10, 6, 10, 8]])
    return ad.AnnData(X=X.T, obs=obs, var=var)

@pytest.fixture
def simple_adata_with_many_obs_labels():
    """Fixture for creating test AnnData object with multiple samples per group"""
    obs = pd.DataFrame(
        {
            "cluster": ["0", "0", "1", "1"],
            "condition": ["A", "A", "B", "B"],
            "cell_type_1": ["T", "B", "T", "B"],
            "cell_type_2": ["CD4", "CD8", "CD4", "CD8"],
            "cell_type_3": ["naive", "memory", "naive", "memory"],
            "cell_type_4": ["activated", "activated", "resting", "resting"],
        },
        index=["cell1", "cell2", "cell3", "cell4"],
    )
    var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
    X = np.array([[1, 1, 1, 1], [5, 4, 5, 4], [10, 6, 10, 8]])
    return ad.AnnData(X=X.T, obs=obs, var=var)


@pytest.fixture
def simple_adata_with_var():
    """Create a simple AnnData object with gene labels."""
    var_df = pd.DataFrame(
        {
            "simplify": [True, True, True, True, False, False],
            "other_col": ["a", "b", "c", "d", "e", "f"],
        },
        index=["HSP90AA1", "HSPA1A", "CLOCK", "IL1A", "APOD", "CFD"],
    )

    return ad.AnnData(X=np.random.rand(3, 6), var=var_df)  # 3 cells, 6 genes


@pytest.fixture
def large_adata():
    """AnnData object with 50 cells and 20 genes."""
    n_cells, n_genes = 50, 20

    # Create realistic-looking expression data with some structure
    np.random.seed(42)
    X = np.zeros((n_cells, n_genes))

    # Create two distinct cell populations
    X[:25] = np.random.normal(5, 1, (25, n_genes))
    X[25:] = np.random.normal(2, 0.5, (25, n_genes))

    # Add some noise
    X += np.random.normal(0, 0.1, X.shape)

    # Create cell metadata
    cell_types = ["TypeA"] * 25 + ["TypeB"] * 25
    cell_ids = [f"cell_{i}" for i in range(n_cells)]
    obs = pd.DataFrame(
        {"cell_type": cell_types, "cell_id": cell_ids},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    # Create gene metadata
    var = pd.DataFrame(
        {
            "gene_id": [f"gene_{i}" for i in range(n_genes)],
            "gene_name": [f"Gene_{i}" for i in range(n_genes)],
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    return ad.AnnData(X=X, obs=obs, var=var, uns={"experiment": "large_test"})


@pytest.fixture
def sparse_large_adata():
    """AnnData object with 100 cells and 30 genes, using sparse matrix."""
    n_cells, n_genes = 100, 30

    # Create sparse data (most values zero, some expression)
    np.random.seed(42)
    X = np.zeros((n_cells, n_genes))

    # Add sparse expression (only 10% of values are non-zero)
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.random.normal(5, 1, np.sum(mask))

    # Convert to sparse matrix
    X_sparse = sparse.csr_matrix(X)

    obs = pd.DataFrame(
        {
            "cell_type": ["Type" + str(i % 4) for i in range(n_cells)],
            "cell_id": [f"cell_{i}" for i in range(n_cells)],
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {
            "gene_id": [f"gene_{i}" for i in range(n_genes)],
            "gene_name": [f"Gene_{i}" for i in range(n_genes)],
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    return ad.AnnData(X=X_sparse, obs=obs, var=var, uns={"experiment": "sparse_test"})


@pytest.fixture
def large_adata_with_pca():
    """Create a sample AnnData object for testing."""
    # Create sample data
    n_obs = 100
    n_vars = 20
    n_pcs = 10

    # Create random data
    X = np.random.normal(size=(n_obs, n_vars))

    # Create labels (3 classes)
    labels = np.random.choice(["A", "B", "C"], size=n_obs)

    # Create PCA-like data
    pca_data = np.random.normal(size=(n_obs, n_pcs))

    # Create AnnData object
    adata = ad.AnnData(X)
    adata.obs["cell_type"] = labels
    adata.obsm["X_pca"] = pca_data

    return adata


# AdataDict fixtures
@pytest.fixture
def simple_adata_dict():
    """Basic AdataDict with two samples."""
    return AdataDict(
        {
            ("sample1",): ad.AnnData(
                X=np.array([[1, 2], [3, 4]]),
                obs=pd.DataFrame(
                    {
                        "condition": pd.Categorical(["ctrl", "ctrl"]),
                        "cell_id": ["cell1", "cell2"],
                    },
                    index=["AAG", "AAC"],
                ),
                var=pd.DataFrame(
                    {"gene_id": ["gene1", "gene2"], "gene_name": ["Gene_1", "Gene 2"]},
                    index=["gene1", "gene2"],
                ),
                uns={"sample": "A"},
            ),
            ("sample2",): ad.AnnData(
                X=np.array([[5, 6], [7, 8]]),
                obs=pd.DataFrame(
                    {
                        "condition": pd.Categorical(["treat", "treat"]),
                        "cell_id": ["cell3", "cell4"],
                    },
                    index=["TTG", "TTC"],
                ),
                var=pd.DataFrame(
                    {"gene_id": ["gene1", "gene2"], "gene_name": ["Gene_1", "Gene 2"]},
                    index=["gene1", "gene2"],
                ),
                uns={"sample": "B"},
            ),
        },
        hierarchy=("sample",),
    )

@pytest.fixture
def simple_adata_dict_more_cells():
    """Basic AdataDict with two samples, 4 cells each, split between two conditions."""
    return AdataDict(
        {
            ("sample1",): ad.AnnData(
                X=np.array([
                    [1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8],
                ]),
                obs=pd.DataFrame(
                    {
                        "condition": pd.Categorical(["ctrl", "ctrl", "treat", "treat"]),
                        "cell_id": ["cell1", "cell2", "cell3", "cell4"],
                    },
                    index=["AA1", "AA2", "AA3", "AA4"],
                ),
                var=pd.DataFrame(
                    {"gene_id": ["gene1", "gene2"], "gene_name": ["Gene_1", "Gene_2"]},
                    index=["gene1", "gene2"],
                ),
                uns={"sample": "A"},
            ),
            ("sample2",): ad.AnnData(
                X=np.array([
                    [9, 10],
                    [11, 12],
                    [13, 14],
                    [15, 16],
                ]),
                obs=pd.DataFrame(
                    {
                        "condition": pd.Categorical(["ctrl", "ctrl", "treat", "treat"]),
                        "cell_id": ["cell5", "cell6", "cell7", "cell8"],
                    },
                    index=["TT1", "TT2", "TT3", "TT4"],
                ),
                var=pd.DataFrame(
                    {"gene_id": ["gene1", "gene2"], "gene_name": ["Gene_1", "Gene_2"]},
                    index=["gene1", "gene2"],
                ),
                uns={"sample": "B"},
            ),
        },
        hierarchy=("sample",),
    )


@pytest.fixture
def nested_adata_dict():
    """Nested AdataDict with group/sample hierarchy."""
    return AdataDict(
        {
            ("group1",): AdataDict(
                {
                    ("sample1",): ad.AnnData(
                        X=np.array([[1, 2]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["ctrl"]),
                                "cell_id": ["cell1"],
                            },
                            index=["ATA"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "1", "sample": "A"},
                    ),
                    ("sample2",): ad.AnnData(
                        X=np.array([[3, 4]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["ctrl"]),
                                "cell_id": ["cell2"],
                            },
                            index=["TTT"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "1", "sample": "B"},
                    ),
                }
            ),
            ("group2",): AdataDict(
                {
                    ("sample3",): ad.AnnData(
                        X=np.array([[5, 6]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["treat"]),
                                "cell_id": ["cell3"],
                            },
                            index=["GGG"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "2", "sample": "C"},
                    ),
                    ("sample4",): ad.AnnData(
                        X=np.array([[7, 8]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["treat"]),
                                "cell_id": ["cell4"],
                            },
                            index=["CGT"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "2", "sample": "D"},
                    ),
                }
            ),
        },
        hierarchy=("group", ("sample",)),
    )


@pytest.fixture
def complex_nested_adata_dict():
    """Nested AdataDict with group/sample hierarchy."""
    return AdataDict(
        {
            ("group1",): AdataDict(
                {
                    ("sample1", "celltype1"): ad.AnnData(
                        X=np.array([[1, 2]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["ctrl"]),
                                "cell_id": ["cell1"],
                            },
                            index=["ATA"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "1", "sample": "A"},
                    ),
                    ("sample2", "celltype2"): ad.AnnData(
                        X=np.array([[3, 4]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["ctrl"]),
                                "cell_id": ["cell2"],
                            },
                            index=["TTT"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "1", "sample": "B"},
                    ),
                }
            ),
            ("group2",): AdataDict(
                {
                    ("sample3", "celltype3"): ad.AnnData(
                        X=np.array([[5, 6]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["treat"]),
                                "cell_id": ["cell3"],
                            },
                            index=["GGG"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "2", "sample": "C"},
                    ),
                    ("sample4", "celltype4"): ad.AnnData(
                        X=np.array([[7, 8]]),
                        obs=pd.DataFrame(
                            {
                                "condition": pd.Categorical(["treat"]),
                                "cell_id": ["cell4"],
                            },
                            index=["CGT"],
                        ),
                        var=pd.DataFrame(
                            {
                                "gene_id": ["gene1", "gene2"],
                                "gene_name": ["Gene_1", "Gene 2"],
                            },
                            index=["gene1", "gene2"],
                        ),
                        uns={"group": "2", "sample": "D"},
                    ),
                }
            ),
        },
        hierarchy=("group", ("sample", "celltype")),
    )


@pytest.fixture
def adata_dict_with_different_cell_type_cols():
    """Create an AdataDict where the AnnData have differently named and populated cell type label columns."""
    return AdataDict(
        {
            ("sample1",): ad.AnnData(
                X=np.array([[1, 2], [3, 4]]),
                obs=pd.DataFrame(
                    {
                        "cell_type": ["CD8+ t cell", "cd8 T-cell"],
                        "cell_id": ["cell1", "cell2"],
                    },
                    index=["cell1", "cell2"],
                ),
            ),
            ("sample2",): ad.AnnData(
                X=np.array([[5, 6], [7, 8]]),
                obs=pd.DataFrame(
                    {
                        "labels": ["CD8+ T Cell", "CD8-positive T cell"],
                        "cell_id": ["cell3", "cell4"],
                    },
                    index=["cell3", "cell4"],
                ),
            ),
        }
    )


@pytest.fixture
def error_prone_adata_dict():
    """AdataDict with problematic data for error handling tests."""

    class ProblemAnnData(ad.AnnData):
        """AnnData subclass that raises errors based on the error_type attribute."""

        def __init__(self, error_type: str, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_type = error_type

        def __eq__(self, other):
            pass

        def problematic_method(self):
            """Method that raises errors based on the error_type attribute."""
            if self.error_type == "value":
                raise ValueError("Deliberate value error")
            elif self.error_type == "key":
                raise KeyError("Deliberate key error")
            elif self.error_type == "runtime":
                raise RuntimeError("Deliberate runtime error")
            else:
                pass

    return AdataDict(
        {
            ("good_sample",): ProblemAnnData(
                None, X=np.array([[1, 2]]), uns={"status": "ok"}
            ),
            ("value_error",): ProblemAnnData("value", X=np.array([[1, 2]])),
            ("key_error",): ProblemAnnData("key", X=np.array([[1, 2]])),
            ("runtime_error",): ProblemAnnData("runtime", X=np.array([[1, 2]])),
        }
    )


@pytest.fixture
def adata_dict_with_special_character_keys():
    """AdataDict with keys containing special characters."""
    return AdataDict(
        {
            ("key with spaces", "special#chars"): ad.AnnData(
                X=np.array([[1, 2]]),
                obs=pd.DataFrame(index=["obs1"]),
            ),
            ("number123", "symbols!@"): ad.AnnData(
                X=np.array([[3, 4]]),
                obs=pd.DataFrame(index=["obs2"]),
            ),
        },
        hierarchy=("level1", "level2"),
    )


@pytest.fixture
def adata_dict_with_empty_adata():
    """AdataDict with empty AnnData objects."""
    return AdataDict(
        {
            ("empty_sample1",): ad.AnnData(
                X=np.zeros((0, 0)),
                obs=pd.DataFrame(
                    {"cell_type": pd.Series(dtype="str")}, index=pd.Index([])
                ),
                var=pd.DataFrame(index=pd.Index([])),
            ),
            ("empty_sample2",): ad.AnnData(
                X=np.zeros((0, 0)),
                obs=pd.DataFrame(
                    {"labels": pd.Series(dtype="str")}, index=pd.Index([])
                ),
                var=pd.DataFrame(index=pd.Index([])),
            ),
        },
        hierarchy=("sample",),
    )


# Test functions
def create_test_function(modification: str = "uns"):
    """Create a test function that modifies AnnData in a specific way."""

    def test_func(adata: ad.AnnData):
        if modification == "uns":
            adata.uns["modified"] = True
        elif modification == "obs":
            adata.obs["test"] = [1] * len(adata.obs)
        elif modification == "X":
            adata.X = np.zeros_like(adata.X)
        return True

    return test_func


def create_key_aware_function():
    """Create a test function that uses the adt_key parameter."""

    def key_aware_func(adata: ad.AnnData, adt_key: str = None):
        adata.uns["key"] = adt_key
        return adt_key

    return key_aware_func
