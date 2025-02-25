"""
unit tests for anndict.plot.stabilizing_classifier_plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import anndata as ad

from anndict.plot.stabilizing_classifier_plots import (
    plot_training_history,
    plot_label_changes,
)


def test_plot_training_history_adata():
    """Unit test for plot_training_history."""
    # Test data
    history = [0.5, 0.3, 0.2, 0.1]
    adt_key = ("Tissue1", "CellTypeA")

    # Call function
    fig, ax = plot_training_history(history, adt_key)

    # Test assertions
    assert isinstance(fig, plt.Figure)
    assert len(ax.lines) == 1  # Should have one line plot
    assert ax.get_xlabel() == "Iteration"
    assert ax.get_ylabel() == "Percent Label Change"
    assert (
        ax.get_title()
        == "Percent Label Change vs. Iteration - ('Tissue1', 'CellTypeA')"
    )
    assert all(y == h for y, h in zip(ax.lines[0].get_ydata(), history))

    plt.close(fig)  # Clean up


def test_plot_label_changes_count():
    """Unit test for plot_label_changes when plotting count of labels changed."""
    # Create mock AnnData object with test data

    # Create test data with categorical labels
    obs_df = pd.DataFrame(
        {
            "true_labels": pd.Categorical(["A", "A", "B", "B", "C"]),
            "pred_labels": pd.Categorical(["A", "B", "B", "C", "C"]),
        },
        index=["cell1", "cell2", "cell3", "cell4", "cell5"],
    )
    adata = ad.AnnData(X=np.zeros((5, 1)), obs=obs_df)

    # Call function with test parameters
    fig, ax = plot_label_changes(
        adata=adata,
        true_label_key="true_labels",
        predicted_label_key="pred_labels",
        plot_percentage=False,
        stratum="Test Stratum",
    )

    # Test assertions
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "true_labels"
    assert ax.get_ylabel() == "Count of Labels Changed"
    assert ax.get_title() == "Test Stratum"

    # Check the data
    bars = ax.containers[0]
    expected_percentages = {
        "A": 1,  # 1 out of 2 A's changed
        "B": 1,  # 1 out of 2 B's changed
        "C": 0,  # 0 out of 1 C's changed
    }
    actual_heights = {
        label.get_text(): height
        for label, height in zip(ax.get_xticklabels(), bars.datavalues)
    }
    for label, expected_height in expected_percentages.items():
        assert np.isclose(actual_heights[label], expected_height)

    plt.close(fig)  # Clean up


def test_plot_label_changes_percentage():
    """Unit test for plot_label_changes when plotting percentage of labels changed."""
    # Create mock AnnData object with test data

    # Create test data with categorical labels
    obs_df = pd.DataFrame(
        {
            "true_labels": pd.Categorical(["A", "A", "B", "B", "C"]),
            "pred_labels": pd.Categorical(["A", "B", "B", "C", "C"]),
        },
        index=["cell1", "cell2", "cell3", "cell4", "cell5"],
    )
    adata = ad.AnnData(X=np.zeros((5, 1)), obs=obs_df)

    # Call function with test parameters
    fig, ax = plot_label_changes(
        adata=adata,
        true_label_key="true_labels",
        predicted_label_key="pred_labels",
        plot_percentage=True,
        stratum="Test Stratum",
    )

    # Test assertions
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel() == "true_labels"
    assert ax.get_ylabel() == "Percentage of Labels Changed"
    assert ax.get_title() == "Test Stratum"

    # Check the data
    bars = ax.containers[0]
    expected_percentages = {
        "A": 0.5,  # 1 out of 2 A's changed
        "B": 0.5,  # 1 out of 2 B's changed
        "C": 0.0,  # 0 out of 1 C's changed
    }
    actual_heights = {
        label.get_text(): height
        for label, height in zip(ax.get_xticklabels(), bars.datavalues)
    }
    for label, expected_height in expected_percentages.items():
        assert np.isclose(actual_heights[label], expected_height)

    plt.close(fig)  # Clean up
