"""
unit test for anndict.annotate.cells.label_transfer.stabilizing_classifier
"""

import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData

from anndict.utils.stabilizing_classifier import (
    stable_label,
    stable_label_adata,
)


def test_stable_label_basic_functionality():
    # Create a simple dataset
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    classifier = LogisticRegression(random_state=42)

    # Run stable_label
    trained_clf, history, iterations, final_labels = stable_label(
        x=x, y=y, classifier=classifier, random_state=42
    )

    # Basic checks
    assert isinstance(trained_clf, LogisticRegression)
    assert isinstance(history, list)
    assert isinstance(iterations, int)
    assert isinstance(final_labels, np.ndarray)
    assert final_labels.shape == y.shape
    assert all(label in [0, 1] for label in final_labels)


def test_stable_label_early_convergence():
    # Create a dataset that should converge quickly
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    classifier = LogisticRegression(random_state=42)

    # Run with very loose stability threshold
    _, history, iterations, _ = stable_label(
        x=x, y=y, classifier=classifier, stability_threshold=0.5, random_state=42
    )

    # Should converge before max_iterations
    assert iterations < 100
    assert len(history) == iterations


def test_stable_label_max_iterations():
    # Create highly noisy, overlapping data that won't converge easily
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, (1000, 20))  # More dimensions and samples
    y = rng.integers(0, 2, 1000)
    classifier = LogisticRegression(random_state=42)

    # Run with extremely strict stability threshold
    _, history, iterations, _ = stable_label(
        x=x,
        y=y,
        classifier=classifier,
        stability_threshold=0.0001,  # Much stricter threshold
        max_iterations=50,
        random_state=42,
    )

    # Should not exceed max iterations and history length should match iterations
    assert iterations <= 50
    assert len(history) == iterations


def test_stable_label_reproducibility():
    rng = np.random.default_rng(42)
    x = rng.random((50, 2))
    y = rng.integers(0, 2, 50)

    # Run twice with same random_state
    result1 = stable_label(
        x=x, y=y, classifier=LogisticRegression(random_state=42), random_state=42
    )

    result2 = stable_label(
        x=x, y=y, classifier=LogisticRegression(random_state=42), random_state=42
    )

    # Results should be identical
    assert_array_equal(result1[3], result2[3])  # final_labels
    assert result1[1] == result2[1]  # history
    assert result1[2] == result2[2]  # iterations


def test_stable_label_different_moving_averages():
    rng = np.random.default_rng(42)
    x = rng.random((30, 2))
    y = rng.integers(0, 2, 30)

    # Compare different moving average lengths
    _, history1, _, _ = stable_label(
        x=x,
        y=y,
        classifier=LogisticRegression(random_state=42),
        moving_average_length=3,
        random_state=42,
    )

    _, history2, _, _ = stable_label(
        x=x,
        y=y,
        classifier=LogisticRegression(random_state=42),
        moving_average_length=5,
        random_state=42,
    )

    # Histories should be different
    assert history1 != history2


def test_stable_label_input_immutability():
    # Test that original inputs aren't modified
    x_orig = np.array([[1, 0], [0, 1]])
    y_orig = np.array([0, 1])
    x_copy = x_orig.copy()
    y_copy = y_orig.copy()

    stable_label(
        x=x_orig,
        y=y_orig,
        classifier=LogisticRegression(random_state=42),
        random_state=42,
    )

    # Original inputs should remain unchanged
    assert_array_equal(x_orig, x_copy)
    assert_array_equal(y_orig, y_copy)


# for stable_label_adata
def test_stable_label_adata_basic():
    # Create simple test data
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    obs = pd.DataFrame({"cell_type": ["A", "B", "B", "A"]}, index=["a", "b", "c", "d"])
    adata = AnnData(obs=obs)
    adata.obsm["features"] = x

    classifier = LogisticRegression(random_state=42)

    # Run function
    clf, history, iterations, final_labels, label_encoder = stable_label_adata(
        adata=adata,
        feature_key="features",
        label_key="cell_type",
        classifier=classifier,
        random_state=42,
    )

    # Check outputs
    assert isinstance(clf, LogisticRegression)
    assert isinstance(history, list)
    assert isinstance(iterations, int)
    assert isinstance(final_labels, np.ndarray)
    assert isinstance(label_encoder, LabelEncoder)
    assert len(final_labels) == adata.n_obs
    assert set(final_labels) == set(["A", "B"])


def test_stable_label_adata_kwargs_passing():
    # Create test data
    x = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    obs = pd.DataFrame({"cell_type": ["A", "B", "B", "A"]}, index=["a", "b", "c", "d"])
    adata = AnnData(obs=obs)
    adata.obsm["features"] = x

    # Run with custom max_iterations
    _, _, iterations, _, _ = stable_label_adata(
        adata=adata,
        feature_key="features",
        label_key="cell_type",
        classifier=LogisticRegression(random_state=42),
        max_iterations=5,
    )

    # Check that max_iterations was respected
    assert iterations <= 5
