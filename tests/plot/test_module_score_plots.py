"""
unit tests for anndict.plot.module_score_plots
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# disable pylint false positives

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anndata import AnnData
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from anndict.plot.module_score_plots import module_score_barplot, module_score_umap


@pytest.fixture
def basic_adata():
    """Create a basic AnnData object with predictable group means."""
    n_cells = 100

    # Create sample data with known means
    obs = pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50,
            "score1": [1.0] * 50 + [2.0] * 50,  # Mean for A=1.0, B=2.0
            "score2": [3.0] * 50 + [4.0] * 50,  # Mean for A=3.0, B=4.0
            "subgroup": ["X"] * 25 + ["Y"] * 25 + ["X"] * 25 + ["Y"] * 25,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    return AnnData(X=np.random.rand(n_cells, 10), obs=obs)


def test_module_score_barplot_basic(basic_adata):
    """Test basic functionality with single group and score column."""
    fig, ax = module_score_barplot(
        adata=basic_adata, group_cols="group", score_cols="score1"
    )

    # Test return types
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Test plot properties
    assert ax.get_ylabel() == "Mean Module Score"
    assert ax.get_xticklabels()[0].get_rotation() == 90

    # Test data accuracy
    heights = [patch.get_height() for patch in ax.patches]
    np.testing.assert_almost_equal(heights, [1.0, 2.0], decimal=2)


def test_module_score_barplot_multiple_scores(basic_adata):
    """Test with multiple score columns."""
    fig, ax = module_score_barplot(
        adata=basic_adata, group_cols="group", score_cols=["score1", "score2"]
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Test number of bars (2 groups × 2 scores = 4 bars)
    assert len(ax.patches) == 4

    # Test data accuracy
    heights = [patch.get_height() for patch in ax.patches]
    expected_heights = [1.0, 2.0, 3.0, 4.0]
    np.testing.assert_almost_equal(heights, expected_heights, decimal=2)

    # Test legend
    assert ax.get_legend() is not None
    assert len(ax.get_legend().get_texts()) == 2  # Two score columns


def test_module_score_barplot_multiple_groups(basic_adata):
    """Test with multiple grouping columns."""
    fig, ax = module_score_barplot(
        adata=basic_adata, group_cols=["group", "subgroup"], score_cols="score1"
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    # Test number of groups (2 groups × 2 subgroups = 4 tick labels)
    assert len(ax.get_xticklabels()) == 4


def test_module_score_barplot_custom_figsize(basic_adata):
    """Test custom figure size."""
    custom_figsize = (15, 10)
    fig, _ = module_score_barplot(
        adata=basic_adata,
        group_cols="group",
        score_cols="score1",
        figsize=custom_figsize,
    )

    # Test figure size
    assert fig.get_size_inches().tolist() == list(custom_figsize)


def test_module_score_barplot_input_validation():
    """Test input validation."""
    # Create invalid AnnData
    invalid_adata = AnnData(
        X=np.random.rand(20, 10),
        obs=pd.DataFrame(
            {
                "group": ["A"] * 10
                + ["B"] * 10
                # Missing score column
            },
            index=[f"cell_{i}" for i in range(20)],
        ),
    )

    # Test missing score column
    with pytest.raises(KeyError):
        module_score_barplot(
            adata=invalid_adata, group_cols="group", score_cols="nonexistent_score"
        )

    # Test invalid AnnData type
    with pytest.raises(AttributeError):
        module_score_barplot(
            adata="not an AnnData object", group_cols="group", score_cols="score"
        )


def test_module_score_barplot_with_adt_key(basic_adata, capsys):
    """Test function with adt_key parameter and capture print output."""
    adt_key = ("sample1", "condition1")
    _, _ = module_score_barplot(
        adata=basic_adata, group_cols="group", score_cols="score1", adt_key=adt_key
    )

    # Check if adt_key was printed
    captured = capsys.readouterr()
    assert str(adt_key) in captured.out


@pytest.fixture(autouse=True)
def close_figures():
    """Automatically close figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def adata_with_umap():
    """Create an AnnData object with UMAP coordinates and module scores."""
    n_cells = 100

    # Create sample data with known module scores
    obs = pd.DataFrame(
        {
            "score1": np.random.normal(0, 1, n_cells),
            "score2": np.random.normal(2, 1, n_cells),
            "score3": np.random.normal(-1, 1, n_cells),
            "score_with_underscore": np.random.normal(0, 1, n_cells),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    # Create UMAP coordinates
    umap_coords = np.random.normal(0, 1, (n_cells, 2))

    adata = AnnData(X=np.random.rand(n_cells, 10), obs=obs)

    # Add UMAP to obsm
    adata.obsm["X_umap"] = umap_coords

    return adata


def test_module_score_umap_basic(adata_with_umap):
    """Test basic functionality with a single score."""
    fig = module_score_umap(adata=adata_with_umap, score_cols=["score1"])

    assert isinstance(fig, Figure)

    # For a single plot, we expect 2 axes (main plot + colorbar)
    assert len(fig.axes) == 2

    # Main plot should be the first axis
    main_ax = fig.axes[0]
    assert main_ax.get_title() == "Score1"  # Check title formatting
    assert main_ax.get_xlabel() == "UMAP1"
    assert main_ax.get_ylabel() == "UMAP2"

    # Second axis should be the colorbar
    colorbar_ax = fig.axes[1]
    assert colorbar_ax.get_label() == "<colorbar>"


def test_module_score_umap_multiple_scores(adata_with_umap):
    """Test with multiple score columns."""
    score_cols = ["score1", "score2", "score3"]
    fig = module_score_umap(adata=adata_with_umap, score_cols=score_cols)

    assert isinstance(fig, Figure)

    # Calculate expected grid dimensions
    n_cols = int(np.ceil(np.sqrt(len(score_cols))))
    n_rows = int(np.ceil(len(score_cols) / n_cols))
    total_grid_spots = n_rows * n_cols

    # Total axes should be:
    # - One axis per score for the main plots
    # - One axis per score for the colorbars
    # - Empty axes to fill the grid
    expected_axes = (
        len(score_cols) + len(score_cols) + (total_grid_spots - len(score_cols))
    )
    assert len(fig.axes) == expected_axes

    # Check title formatting for each main plot axis
    main_axes = [
        ax for ax in fig.axes if ax.get_title()
    ]  # Get axes with titles (main plots)
    for ax, score in zip(main_axes, score_cols):
        expected_title = " ".join(
            word.capitalize() for word in score.replace("_", " ").split()
        )
        assert ax.get_title() == expected_title
        assert ax.get_xlabel() == "UMAP1"
        assert ax.get_ylabel() == "UMAP2"

    # Check that we have the right number of colorbars
    colorbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(colorbar_axes) == len(score_cols)


def test_module_score_umap_grid_layout(adata_with_umap):
    """Test grid layout with different numbers of plots."""
    # Test with 4 scores (should create 2x2 grid)
    score_cols = ["score1", "score2", "score3", "score_with_underscore"]
    fig = module_score_umap(adata=adata_with_umap, score_cols=score_cols)

    assert isinstance(fig, Figure)
    # For 4 plots in a 2x2 grid, we expect 8 axes (4 main plots + 4 colorbars)
    assert len(fig.axes) == 8

    # Check if it's a properly sized grid
    expected_size = [8, 8]  # 2 cols * 4, 2 rows * 4
    actual_size = fig.get_size_inches().tolist()
    np.testing.assert_array_almost_equal(actual_size, expected_size)


def test_module_score_umap_with_adt_key(adata_with_umap, capsys):
    """Test function with adt_key parameter."""
    adt_key = ("sample1", "condition1")
    _ = module_score_umap(
        adata=adata_with_umap, score_cols=["score1"], adt_key=adt_key
    )

    # Check if adt_key was printed
    captured = capsys.readouterr()
    assert str(adt_key) in captured.out


def test_module_score_umap_title_formatting(adata_with_umap):
    """Test title formatting for different score column names."""
    score_cols = ["score_with_underscore", "score1"]
    fig = module_score_umap(adata=adata_with_umap, score_cols=score_cols)

    # Filter to get only the main plot axes (those with titles)
    main_axes = [ax for ax in fig.axes if ax.get_title()]
    titles = {ax.get_title() for ax in main_axes}

    # Verify all expected titles are present
    expected_titles = {"Score With Underscore", "Score1"}
    assert titles == expected_titles


def test_module_score_umap_extra_kwargs(adata_with_umap):
    """Test passing additional kwargs to sc.pl.umap."""
    fig = module_score_umap(
        adata=adata_with_umap, score_cols=["score1"], frameon=False, vmax=0.5
    )

    assert isinstance(fig, Figure)


def test_module_score_umap_input_validation():
    """Test input validation."""
    # Create invalid AnnData without UMAP coordinates
    invalid_adata = AnnData(
        X=np.random.rand(20, 10),
        obs=pd.DataFrame(
            {"score1": np.random.normal(0, 1, 20)},
            index=[f"cell_{i}" for i in range(20)],
        ),
    )

    # Test missing UMAP coordinates
    with pytest.raises(KeyError):
        module_score_umap(adata=invalid_adata, score_cols=["score1"])

    # Test invalid AnnData type
    with pytest.raises(AttributeError):
        module_score_umap(adata="not an AnnData object", score_cols=["score1"])
