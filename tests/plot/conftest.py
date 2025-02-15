"""
Common test fixtures for plotting
"""

import pytest
import numpy as np
import pandas as pd

from anndata import AnnData


# Test fixtures for plotting (mainly confusion matrix tests)
@pytest.fixture
def basic_adata():
    """Basic AnnData with true and predicted labels."""
    n_cells = 100
    true_labels = ["A"] * 40 + ["B"] * 30 + ["C"] * 30
    pred_labels = ["A"] * 35 + ["B"] * 35 + ["C"] * 30  # Some misclassifications

    obs = pd.DataFrame(
        {"true_label": true_labels, "predicted_label": pred_labels},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    return AnnData(X=np.random.rand(n_cells, 10), obs=obs)  # Random expression matrix


@pytest.fixture
def colored_adata():
    """AnnData with additional color information."""
    n_cells = 50
    true_labels = ["A"] * 20 + ["B"] * 15 + ["C"] * 15
    pred_labels = ["A"] * 18 + ["B"] * 17 + ["C"] * 15
    colors = ["red"] * 20 + ["blue"] * 15 + ["green"] * 15

    obs = pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": pred_labels,
            "color_group": colors,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    return AnnData(X=np.random.rand(n_cells, 10), obs=obs)


@pytest.fixture
def swapped_adata():
    """AnnData with swapped A/B classifications for testing diagonalization."""
    n_cells = 60
    true_labels = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    pred_labels = ["B"] * 20 + ["A"] * 20 + ["C"] * 20  # Swapped A and B

    obs = pd.DataFrame(
        {"true_label": true_labels, "predicted_label": pred_labels},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    return AnnData(X=np.random.rand(n_cells, 10), obs=obs)


@pytest.fixture
def binary_adata():
    """Simple binary classification AnnData."""
    n_cells = 40
    true_labels = ["A"] * 20 + ["B"] * 20
    pred_labels = ["A"] * 18 + ["B"] * 22

    obs = pd.DataFrame(
        {"true_label": true_labels, "predicted_label": pred_labels},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    return AnnData(X=np.random.rand(n_cells, 10), obs=obs)


@pytest.fixture
def invalid_adata():
    """AnnData with missing label column."""
    return AnnData(
        X=np.random.rand(20, 10),
        obs=pd.DataFrame(
            {
                "true_label": ["A"] * 10
                + ["B"] * 10
                # Missing predicted_label column
            },
            index=[f"cell_{i}" for i in range(20)],
        ),
    )
