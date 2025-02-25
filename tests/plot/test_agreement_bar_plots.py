"""
unit tests for anndict.plot.agreement_bar_plots
"""
# pylint: disable=redefined-outer-name
# disable pylint false positives

import pytest
import numpy as np

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from seaborn.matrix import ClusterGrid

from anndict.plot.agreement_bar_plots import plot_grouped_average, plot_model_agreement


# plot grouped average tests
def test_plot_grouped_average_basic(large_adata):
    """Test basic functionality with cell type grouping."""
    # Add a numeric column to test averaging
    large_adata.obs["expression_score"] = np.random.uniform(
        0, 1, size=len(large_adata.obs)
    )

    label_value = {"cell_type": "expression_score"}
    plot_grouped_average(large_adata, label_value)


def test_plot_grouped_average_multiple_groups(large_adata):
    """Test with multiple grouping variables."""
    # Add additional grouping and value columns
    large_adata.obs["batch"] = np.random.choice(
        ["Batch1", "Batch2"], size=len(large_adata.obs)
    )
    large_adata.obs["quality_score"] = np.random.uniform(
        0, 1, size=len(large_adata.obs)
    )
    large_adata.obs["expression_level"] = np.random.uniform(
        1, 10, size=len(large_adata.obs)
    )

    label_value = {"cell_type": "quality_score", "batch": "expression_level"}

    plot_grouped_average(large_adata, label_value)


def test_plot_grouped_average_with_adt_key(large_adata):
    """Test with adt_key parameter."""
    large_adata.obs["signal_strength"] = np.random.uniform(
        0, 1, size=len(large_adata.obs)
    )

    label_value = {"cell_type": "signal_strength"}
    adt_key = ("gene_0", "gene_1")  # Using first two genes from var

    plot_grouped_average(large_adata, label_value, adt_key=adt_key)


def test_plot_grouped_average_invalid_input(large_adata):
    """Test error handling for invalid inputs."""
    # Test with non-existent column
    with pytest.raises(ValueError):
        label_value = {"nonexistent_group": "expression_score"}
        plot_grouped_average(large_adata, label_value)

    # Test with non-existent value column
    with pytest.raises(ValueError):
        label_value = {"cell_type": "nonexistent_value"}
        plot_grouped_average(large_adata, label_value)

    # Test with invalid label_value type
    with pytest.raises(AttributeError):
        plot_grouped_average(large_adata, ["invalid", "type"])


def test_plot_grouped_average_single_group(large_adata):
    """Test with a single cell type group."""
    # Create a subset with only one cell type
    single_group_adata = large_adata[large_adata.obs["cell_type"] == "TypeA"].copy()
    single_group_adata.obs["value"] = np.random.uniform(
        0, 1, size=len(single_group_adata.obs)
    )

    label_value = {"cell_type": "value"}
    plot_grouped_average(single_group_adata, label_value)


def test_plot_grouped_average_empty_groups(large_adata):
    """Test handling of empty groups."""
    # Add a group with no members
    large_adata.obs["sparse_group"] = ["Group1"] * (len(large_adata.obs) - 1) + [
        "EmptyGroup"
    ]
    large_adata.obs["test_value"] = np.random.uniform(0, 1, size=len(large_adata.obs))

    label_value = {"sparse_group": "test_value"}
    plot_grouped_average(large_adata, label_value)


# plot model agreement tests
@pytest.fixture
def agreement_adata(large_adata):
    """Extends large_adata with agreement-related columns."""
    # Add tissue information
    large_adata.obs["tissue"] = np.random.choice(
        ["Liver", "Lung"], size=len(large_adata.obs)
    )

    # Add agreement columns
    large_adata.obs["agreement_of_manual_with_model1"] = np.random.uniform(
        0.5, 1.0, size=len(large_adata.obs)
    )
    large_adata.obs["agreement_of_manual_with_model2"] = np.random.uniform(
        0.6, 0.9, size=len(large_adata.obs)
    )
    large_adata.obs["agreement_of_manual_with_model3"] = np.random.uniform(
        0.7, 0.95, size=len(large_adata.obs)
    )

    return large_adata


def test_plot_model_agreement_granularity_0(agreement_adata):
    """Test plotting with granularity=0 (models only)."""
    agreement_cols = [
        "agreement_of_manual_with_model1",
        "agreement_of_manual_with_model2",
    ]

    result = plot_model_agreement(
        agreement_adata,
        group_by="cell_type",
        sub_group_by="tissue",
        agreement_cols=agreement_cols,
        granularity=0,
    )

    assert isinstance(result, (tuple, ClusterGrid))
    if isinstance(result, tuple):
        assert isinstance(result[0], Figure)
        assert isinstance(result[1], Axes)


def test_plot_model_agreement_granularity_1(agreement_adata):
    """Test plotting with granularity=1 (models within cell types)."""
    agreement_cols = [
        "agreement_of_manual_with_model1",
        "agreement_of_manual_with_model2",
    ]

    result = plot_model_agreement(
        agreement_adata,
        group_by="cell_type",
        sub_group_by="tissue",
        agreement_cols=agreement_cols,
        granularity=1,
    )

    assert isinstance(result, (tuple, ClusterGrid))


def test_plot_model_agreement_granularity_2(agreement_adata):
    """Test plotting with granularity=2 (models within cell types and tissues)."""
    agreement_cols = [
        "agreement_of_manual_with_model1",
        "agreement_of_manual_with_model2",
    ]

    result = plot_model_agreement(
        agreement_adata,
        group_by="cell_type",
        sub_group_by="tissue",
        agreement_cols=agreement_cols,
        granularity=2,
    )

    assert isinstance(result, (tuple, ClusterGrid))


def test_plot_model_agreement_all_models(agreement_adata):
    """Test plotting with all available agreement columns."""
    agreement_cols = [
        "agreement_of_manual_with_model1",
        "agreement_of_manual_with_model2",
        "agreement_of_manual_with_model3",
    ]

    result = plot_model_agreement(
        agreement_adata,
        group_by="cell_type",
        sub_group_by="tissue",
        agreement_cols=agreement_cols,
        granularity=2,
    )

    assert isinstance(result, (tuple, ClusterGrid))


def test_plot_model_agreement_invalid_granularity(agreement_adata):
    """Test error handling for invalid granularity values."""
    agreement_cols = ["agreement_of_manual_with_model1"]

    with pytest.raises(ValueError):
        plot_model_agreement(
            agreement_adata,
            group_by="cell_type",
            sub_group_by="tissue",
            agreement_cols=agreement_cols,
            granularity=3,
        )


def test_plot_model_agreement_invalid_columns(agreement_adata):
    """Test error handling for invalid column names."""
    # Test with non-existent agreement column
    with pytest.raises(ValueError):
        plot_model_agreement(
            agreement_adata,
            group_by="cell_type",
            sub_group_by="tissue",
            agreement_cols=["nonexistent_column"],
            granularity=2,
        )

    # Test with non-existent group_by column
    with pytest.raises(ValueError):
        plot_model_agreement(
            agreement_adata,
            group_by="nonexistent_column",
            sub_group_by="tissue",
            agreement_cols=["agreement_of_manual_with_model1"],
            granularity=2,
        )

    # Test with non-existent sub_group_by column
    with pytest.raises(ValueError):
        plot_model_agreement(
            agreement_adata,
            group_by="cell_type",
            sub_group_by="nonexistent_column",
            agreement_cols=["agreement_of_manual_with_model1"],
            granularity=2,
        )


def test_plot_model_agreement_empty_agreement_cols(agreement_adata):
    """Test error handling for empty agreement_cols list."""
    with pytest.raises(IndexError):
        plot_model_agreement(
            agreement_adata,
            group_by="cell_type",
            sub_group_by="tissue",
            agreement_cols=[],
            granularity=2,
        )


def test_plot_model_agreement_single_category(agreement_adata):
    """Test plotting with single category in grouping columns."""
    # Create a subset with single cell type
    single_type_adata = agreement_adata[
        agreement_adata.obs["cell_type"] == "TypeA"
    ].copy()

    agreement_cols = ["agreement_of_manual_with_model1"]

    result = plot_model_agreement(
        single_type_adata,
        group_by="cell_type",
        sub_group_by="tissue",
        agreement_cols=agreement_cols,
        granularity=2,
    )

    assert isinstance(result, (tuple, ClusterGrid))
