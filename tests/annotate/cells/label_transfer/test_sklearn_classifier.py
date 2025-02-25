"""
unit tests for anndict.annotate.cells.label_transfer.sklearn_classifier
"""
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# disable pylint false positives

import pytest
import numpy as np
import pandas as pd

from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from anndict.annotate.cells.label_transfer.sklearn_classifier import (
    train_label_classifier,
    transfer_labels_using_classifier,
)
from anndict.annotate.cells.label_transfer.adata_predicoder import AdataPredicoder


def test_train_label_classifier_basic(large_adata_with_pca):
    """Test basic functionality with LogisticRegression."""
    result = train_label_classifier(
        adata=large_adata_with_pca,
        label_key="cell_type",
        feature_key="X_pca",
        classifier_class=LogisticRegression,
        random_state=42,
    )

    # Check return type
    assert isinstance(result, AdataPredicoder)

    # Check if classifier is trained
    assert hasattr(result.classifier, "classes_")

    # Check if label encoder is present
    assert isinstance(result.label_encoder, LabelEncoder)

    # Check if all classes are preserved
    original_classes = set(large_adata_with_pca.obs["cell_type"])
    encoded_classes = set(result.label_encoder.classes_)
    assert original_classes == encoded_classes


def test_train_label_classifier_use_X(large_adata_with_pca):
    """Test using X matrix instead of obsm."""
    result = train_label_classifier(
        adata=large_adata_with_pca,
        label_key="cell_type",
        feature_key="use_X",
        classifier_class=RandomForestClassifier,
        random_state=42,
    )

    assert isinstance(result, AdataPredicoder)
    assert hasattr(result.classifier, "classes_")


def test_train_label_classifier_with_kwargs(large_adata_with_pca):
    """Test passing additional kwargs to classifier."""
    result = train_label_classifier(
        adata=large_adata_with_pca,
        label_key="cell_type",
        feature_key="X_pca",
        classifier_class=LogisticRegression,
        random_state=42,
        C=0.1,
        max_iter=1000,
    )

    assert isinstance(result.classifier, LogisticRegression)
    assert result.classifier.C == 0.1
    assert result.classifier.max_iter == 1000


def test_train_label_classifier_invalid_feature_key(large_adata_with_pca):
    """Test error handling for invalid feature key."""
    with pytest.raises(KeyError):
        train_label_classifier(
            adata=large_adata_with_pca,
            label_key="cell_type",
            feature_key="nonexistent_key",
            classifier_class=LogisticRegression,
            random_state=42,
        )


def test_train_label_classifier_invalid_label_key(large_adata_with_pca):
    """Test error handling for invalid label key."""
    with pytest.raises(KeyError):
        train_label_classifier(
            adata=large_adata_with_pca,
            label_key="nonexistent_label",
            feature_key="X_pca",
            classifier_class=LogisticRegression,
            random_state=42,
        )


def test_train_label_classifier_empty_labels(large_adata_with_pca):
    """Test handling of empty labels."""
    large_adata_with_pca.obs["empty_labels"] = ""

    with pytest.raises(ValueError):
        train_label_classifier(
            adata=large_adata_with_pca,
            label_key="empty_labels",
            feature_key="X_pca",
            classifier_class=LogisticRegression,
            random_state=42,
        )


# transfer labels test functions


@pytest.fixture
def large_adata_with_pcas():
    """Create sample origin and destination AnnData objects for testing."""
    # Create sample data
    n_obs_origin = 100
    n_obs_dest = 80
    n_vars = 20
    n_pcs = 10

    # Create random data for origin
    X_origin = np.random.normal(size=(n_obs_origin, n_vars))
    labels_origin = np.random.choice(["A", "B", "C"], size=n_obs_origin)
    pca_origin = np.random.normal(size=(n_obs_origin, n_pcs))

    # Create random data for destination
    X_dest = np.random.normal(size=(n_obs_dest, n_vars))
    pca_dest = np.random.normal(size=(n_obs_dest, n_pcs))

    # Create AnnData objects
    origin_adata = AnnData(X_origin)
    destination_adata = AnnData(X_dest)

    # Add observations and PCA
    origin_adata.obs["cell_type"] = labels_origin
    origin_adata.obsm["X_pca"] = pca_origin
    destination_adata.obsm["X_pca"] = pca_dest

    return origin_adata, destination_adata


def test_transfer_labels_basic(large_adata_with_pcas):
    """Test basic functionality with LogisticRegression."""
    origin_adata, destination_adata = large_adata_with_pcas

    result = transfer_labels_using_classifier(
        origin_adata=origin_adata,
        destination_adata=destination_adata,
        origin_label_key="cell_type",
        feature_key="X_pca",
        classifier_class=LogisticRegression,
        new_column_name="predicted_label",
        random_state=42,
    )

    # Check return type
    assert isinstance(result, AdataPredicoder)

    # Check if predictions were added to destination_adata
    assert "predicted_label" in destination_adata.obs.columns

    # Check if all predicted labels are valid (from original classes)
    original_classes = set(origin_adata.obs["cell_type"])
    predicted_classes = set(destination_adata.obs["predicted_label"])
    assert predicted_classes.issubset(original_classes)


def test_transfer_labels_use_X(large_adata_with_pcas):
    """Test using X matrix instead of obsm."""
    origin_adata, destination_adata = large_adata_with_pcas

    result = transfer_labels_using_classifier(
        origin_adata=origin_adata,
        destination_adata=destination_adata,
        origin_label_key="cell_type",
        feature_key="use_X",
        classifier_class=RandomForestClassifier,
        random_state=42,
    )

    assert isinstance(result, AdataPredicoder)
    assert "predicted_label" in destination_adata.obs.columns


def test_transfer_labels_custom_column_name(large_adata_with_pcas):
    """Test using custom column name for predictions."""
    origin_adata, destination_adata = large_adata_with_pcas
    custom_column = "custom_predictions"

    _ = transfer_labels_using_classifier(
        origin_adata=origin_adata,
        destination_adata=destination_adata,
        origin_label_key="cell_type",
        feature_key="X_pca",
        classifier_class=LogisticRegression,
        new_column_name=custom_column,
        random_state=42,
    )

    assert custom_column in destination_adata.obs.columns
    assert isinstance(destination_adata.obs[custom_column], pd.Series)


def test_transfer_labels_with_kwargs(large_adata_with_pcas):
    """Test passing additional kwargs to classifier."""
    origin_adata, destination_adata = large_adata_with_pcas

    result = transfer_labels_using_classifier(
        origin_adata=origin_adata,
        destination_adata=destination_adata,
        origin_label_key="cell_type",
        feature_key="X_pca",
        classifier_class=LogisticRegression,
        random_state=42,
        C=0.1,
        max_iter=1000,
    )

    assert isinstance(result.classifier, LogisticRegression)
    assert result.classifier.C == 0.1
    assert result.classifier.max_iter == 1000


def test_transfer_labels_mismatched_features(large_adata_with_pcas):
    """Test error handling for mismatched feature dimensions."""
    origin_adata, destination_adata = large_adata_with_pcas

    # Modify destination PCA to have different dimensions
    destination_adata.obsm["X_pca"] = np.random.normal(
        size=(destination_adata.n_obs, destination_adata.obsm["X_pca"].shape[1] + 1)
    )

    with pytest.raises(ValueError):
        transfer_labels_using_classifier(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_label_key="cell_type",
            feature_key="X_pca",
            classifier_class=LogisticRegression,
            random_state=42,
        )


def test_transfer_labels_invalid_feature_key(large_adata_with_pcas):
    """Test error handling for invalid feature key."""
    origin_adata, destination_adata = large_adata_with_pcas

    with pytest.raises(KeyError):
        transfer_labels_using_classifier(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_label_key="cell_type",
            feature_key="nonexistent_key",
            classifier_class=LogisticRegression,
            random_state=42,
        )


def test_transfer_labels_invalid_label_key(large_adata_with_pcas):
    """Test error handling for invalid label key."""
    origin_adata, destination_adata = large_adata_with_pcas

    with pytest.raises(KeyError):
        transfer_labels_using_classifier(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_label_key="nonexistent_label",
            feature_key="X_pca",
            classifier_class=LogisticRegression,
            random_state=42,
        )
