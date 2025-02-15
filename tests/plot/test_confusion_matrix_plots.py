"""
unit tests for anndict.plot.confusion_matrix_plots
"""

import pytest
import numpy as np


from sklearn.preprocessing import LabelEncoder
from seaborn.matrix import ClusterGrid

from anndict.plot.confusion_matrix_plots import (
    plot_confusion_matrix,
    plot_confusion_matrix_from_adata,
)


# Tests for plot_confusion_matrix
def test_plot_confusion_matrix_basic():
    """Test basic functionality"""
    # Setup test data
    true_labels = ["A", "B", "A", "C", "B", "C", "A", "B"]
    predicted_labels = ["A", "B", "A", "B", "B", "C", "A", "C"]

    # Create and fit label encoder
    le = LabelEncoder()
    le.fit(list(set(true_labels + predicted_labels)))

    # Encode labels
    true_encoded = le.transform(true_labels)
    pred_encoded = le.transform(predicted_labels)

    # Create basic color map
    color_map = {
        "row": {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"},
        "col": {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"},
    }

    # Call function with basic parameters
    result = plot_confusion_matrix(
        true_labels_encoded=true_encoded,
        predicted_labels_encoded=pred_encoded,
        label_encoder=le,
        color_map=color_map,
    )

    # Basic assertions
    assert isinstance(result, ClusterGrid)
    assert result.data.shape == (3, 3)  # 3x3 confusion matrix for 3 classes


def test_plot_confusion_matrix_diagonalization():
    """Test diagonalization functionality"""
    # Setup test data
    true_labels = ["A", "B", "A", "C", "B", "C"]
    predicted_labels = ["A", "B", "A", "B", "B", "C"]

    le = LabelEncoder()
    le.fit(list(set(true_labels + predicted_labels)))

    true_encoded = le.transform(true_labels)
    pred_encoded = le.transform(predicted_labels)

    color_map = {"row": {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}}

    # Test with diagonalization enabled
    result_diag = plot_confusion_matrix(
        true_labels_encoded=true_encoded,
        predicted_labels_encoded=pred_encoded,
        label_encoder=le,
        color_map=color_map,
        diagonalize=True,
    )

    assert isinstance(result_diag, ClusterGrid)


def test_plot_confusion_matrix_custom_labels():
    """Test custom labels functionality"""
    # Setup test data
    true_labels = ["A", "B", "A", "C"]
    predicted_labels = ["A", "B", "A", "B"]

    le = LabelEncoder()
    le.fit(list(set(true_labels + predicted_labels)))

    true_encoded = le.transform(true_labels)
    pred_encoded = le.transform(predicted_labels)

    color_map = {"row": {"A": "#ff0000", "B": "#00ff00", "C": "#0000ff"}}

    custom_ticklabels = ["Label A", "Label B", "Label C"]

    # Test with custom labels
    result = plot_confusion_matrix(
        true_labels_encoded=true_encoded,
        predicted_labels_encoded=pred_encoded,
        label_encoder=le,
        color_map=color_map,
        true_ticklabels=custom_ticklabels,
        predicted_ticklabels=custom_ticklabels,
    )

    assert isinstance(result, ClusterGrid)


def test_plot_confusion_matrix_input_validation():
    """Test input validation"""
    # Setup invalid test data
    true_encoded = np.array([0, 1, 2])
    pred_encoded = np.array([0, 1])  # Different length
    le = LabelEncoder().fit([0, 1, 2])
    color_map = {"row": {"A": "#ff0000"}}

    # Test that it raises ValueError for mismatched lengths
    with pytest.raises(ValueError):
        plot_confusion_matrix(
            true_labels_encoded=true_encoded,
            predicted_labels_encoded=pred_encoded,
            label_encoder=le,
            color_map=color_map,
        )


def test_plot_confusion_matrix_large_labels():
    """Test handling of large number of labels"""
    # Setup test data with more than 40 labels
    n_labels = 45
    true_labels = [f"Label_{i}" for i in range(n_labels)]
    predicted_labels = [f"Label_{i}" for i in range(n_labels)]

    le = LabelEncoder()
    le.fit(true_labels)

    true_encoded = le.transform(true_labels)
    pred_encoded = le.transform(predicted_labels)

    color_map = {"row": {f"Label_{i}": "#ff0000" for i in range(n_labels)}}

    # Test automatic disabling of annotations for large matrices
    result = plot_confusion_matrix(
        true_labels_encoded=true_encoded,
        predicted_labels_encoded=pred_encoded,
        label_encoder=le,
        color_map=color_map,
    )

    assert isinstance(result, ClusterGrid)
    # Check that annotations are disabled for large matrices
    assert len(result.ax_heatmap.texts) == 0


# Tests for plot_confusion_matrix_from_adata
def test_plot_confusion_matrix_from_adata_basic(basic_adata):
    """Test basic functionality with AnnData"""
    result = plot_confusion_matrix_from_adata(
        adata=basic_adata,
        true_label_key="true_label",
        predicted_label_key="predicted_label",
    )

    assert isinstance(result, ClusterGrid)
    assert result.data.shape == (3, 3)  # 3x3 confusion matrix for 3 classes


def test_plot_confusion_matrix_from_adata_with_colors(colored_adata):
    """Test functionality with color information in AnnData"""
    result = plot_confusion_matrix_from_adata(
        adata=colored_adata,
        true_label_key="true_label",
        predicted_label_key="predicted_label",
        row_color_keys="color_group",
        col_color_keys="color_group",
    )

    assert isinstance(result, ClusterGrid)


def test_plot_confusion_matrix_from_adata_diagonalize(swapped_adata):
    """Test diagonalization functionality with AnnData"""
    result = plot_confusion_matrix_from_adata(
        adata=swapped_adata,
        true_label_key="true_label",
        predicted_label_key="predicted_label",
        diagonalize=True,
    )

    assert isinstance(result, ClusterGrid)


def test_plot_confusion_matrix_from_adata_custom_labels(binary_adata):
    """Test custom labels functionality with AnnData"""
    custom_labels = ["Type A", "Type B"]

    result = plot_confusion_matrix_from_adata(
        adata=binary_adata,
        true_label_key="true_label",
        predicted_label_key="predicted_label",
        true_ticklabels=custom_labels,
        predicted_ticklabels=custom_labels,
    )

    assert isinstance(result, ClusterGrid)


def test_plot_confusion_matrix_from_adata_input_validation(invalid_adata):
    """Test input validation with AnnData"""
    # Test that it raises KeyError for missing column
    with pytest.raises(KeyError):
        plot_confusion_matrix_from_adata(
            adata=invalid_adata,
            true_label_key="true_label",
            predicted_label_key="nonexistent_column",
        )

    # Test that it raises TypeError for invalid AnnData
    with pytest.raises(AttributeError):
        plot_confusion_matrix_from_adata(
            adata="not an AnnData object",  # Invalid input
            true_label_key="true_label",
            predicted_label_key="predicted_label",
        )
