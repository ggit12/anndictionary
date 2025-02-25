"""
Plot confusion matrices. Inlcudes the option to rearrange columns and rows to make the matrix appear diagonal, which can facilitate interpretation.
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from seaborn.matrix import ClusterGrid
from anndata import AnnData


from anndict.utils import create_color_map

def plot_confusion_matrix_from_adata(
    adata: AnnData,
    true_label_key: str,
    predicted_label_key: str,
    title: str = 'Confusion Matrix',
    row_color_keys: str | None = None,
    col_color_keys: str | None = None,
    figsize: tuple[int,int] | None = None,
    diagonalize: bool = False,
    true_ticklabels: list[str] | None = None,
    predicted_ticklabels: list[str] | None = None,
    annot: bool | None = None,
    adt_key: tuple[str, ...] | None = None,
) -> ClusterGrid:
    """
    Plots a confusion matrix from an :class:`AnnData` object, with optional row and column colors.

    Set ``diagonalize=True`` to automatically arrange the column and rows to make the confusion matrix appear as diagonal as possible.

    Wraps :func:`plot_confusion_matrix`.

    Parameters
    ------------
    adata 
        An :class:`AnnData`.

    true_label_key
        key to access the true class labels in ``adata.obs``.

    predicted_label_key
        key to access the predicted class labels in ``adata.obs``.

    title
        title of the plot.

    row_color_key
        key for row colors in ``adata.obs``.

    col_color_key
        key for column colors in ``adata.obs``.

    adt_key
        Used when wrapping with :func:`adata_dict_fapply`.

    Returns
    --------
    The confusion matrix plot object.

    See Also
    ---------
    :func:`plot_confusion_matrix` : the main plotting function used.
    """
    print(f"{adt_key}")
    # Check and convert row_color_key and col_color_key to lists if they are not None
    if row_color_keys is not None and not isinstance(row_color_keys, list):
        row_color_keys = [row_color_keys]

    if col_color_keys is not None and not isinstance(col_color_keys, list):
        col_color_keys = [col_color_keys]

    # Get unique labels
    true_labels = adata.obs[true_label_key].astype(str)
    predicted_labels = adata.obs[predicted_label_key].astype(str)

    combined_labels = pd.concat([true_labels, predicted_labels])
    label_encoder = LabelEncoder()
    label_encoder.fit(combined_labels)

    #Encode labels
    true_labels_encoded = label_encoder.transform(true_labels)
    predicted_labels_encoded = label_encoder.transform(predicted_labels)

    # Create label-to-color dictionary for mapping
    true_label_color_dict = None
    if row_color_keys:
        true_label_subset = adata.obs[[true_label_key] + row_color_keys].drop_duplicates().set_index(true_label_key)
        true_label_color_dict = {label: {key: row[key] for key in row_color_keys}
                        for label, row in true_label_subset.iterrows()
                        }

    predicted_label_color_dict = None
    if col_color_keys:
        predicted_label_subset = adata.obs[[predicted_label_key] + col_color_keys].drop_duplicates().set_index(predicted_label_key)
        predicted_label_color_dict = {label: {key: col[key] for key in col_color_keys}
                        for label, col in predicted_label_subset.iterrows()
                        }

    # Compute the row and column colors
    # Get unified color mapping
    keys = list(set(row_color_keys or []).union(col_color_keys or []))
    color_map = create_color_map(adata, keys)

    # Call the main plot function
    return plot_confusion_matrix(true_labels_encoded, predicted_labels_encoded, label_encoder, color_map, title,
                          row_color_keys=row_color_keys, col_color_keys=col_color_keys,
                          true_label_color_dict=true_label_color_dict, predicted_label_color_dict=predicted_label_color_dict,
                          figsize=figsize, diagonalize=diagonalize, true_ticklabels=true_ticklabels,
                          predicted_ticklabels=predicted_ticklabels, annot=annot)


def plot_confusion_matrix(
    true_labels_encoded: np.ndarray,
    predicted_labels_encoded: np.ndarray,
    label_encoder: LabelEncoder,
    color_map: dict[str, dict[str, str]],
    title: str = 'Confusion Matrix',
    row_color_keys: list[str] | None = None,
    col_color_keys: list[str] | None = None,
    true_label_color_dict: dict[str, dict[str, str]] | None = None,
    predicted_label_color_dict: dict[str, dict[str, str]] | None = None,
    figsize: tuple[float, float] | None = None,
    diagonalize: bool = False,
    true_ticklabels: bool | list[str] | None = None,
    predicted_ticklabels: bool | list[str] | None = None,
    annot: bool | None = None
) -> ClusterGrid:
    """
    Plot a confusion matrix using cluster mapping with optional color annotations.

    This function computes a confusion matrix based on encoded true and predicted labels
    and optionally reorders the matrix to maximize diagonal alignment. It supports
    row and column coloring based on custom dictionaries and color maps.

    Parameters
    ----------
    true_labels_encoded
        Encoded true labels.

    predicted_labels_encoded
        Encoded predicted labels.

    label_encoder
        A fitted ``sklearn`` ``LabelEncoder`` that can decode the labels.

    color_map
        A nested dictionary specifying how labels should be colored.
        The first-level keys match possible color keys (e.g., ``'row'``, ``'col'``),
        and each maps to a dict from label attributes to color codes.

    title
        The title of the plot.

    row_color_keys
        Keys to extract color information for the rows from ``color_map``.

    col_color_keys
        Keys to extract color information for the columns from ``color_map``.

    true_label_color_dict
        A nested :class:`dict` specifying row-label-specific color mappings.

    predicted_label_color_dict
        A nested :class:`dict` specifying column-label-specific color mappings.

    true_labels
        Original (not encoded) true labels, if needed.

    predicted_labels
        Original (not encoded) predicted labels, if needed.

    figsize
        Figure size passed to seaborn's :func:`clustermap`.

    diagonalize
        If ``True``, reorders the confusion matrix to maximize its diagonal alignment.

    true_ticklabels
        Labels to display along the rows of the confusion matrix.
        If ``True``, display all; if ``False``, display none.

    predicted_ticklabels
        Labels to display along the columns of the confusion matrix.
        If ``True``, display all; if ``False``, display none.

    annot
        Whether to annotate each cell with its numeric value.

    Returns
    --------
    A seaborn :class:`ClusterGrid` object displaying the plotted confusion matrix.

    Notes
    ------
    - If the number of true or predicted labels exceeds 40, 
        tick labels and annotations are disabled by default for better visibility.
    - The matrix is normalized by the number of samples in each true class.
    - If `diagonalize` is True, reorder the columns and 
        row to diagonalize the matrix to the extent possible.

    Examples
    ---------

    .. code-block:: python

        # Assume true_encoded, pred_encoded, and le are given, along with a color_map dict
        g = plot_confusion_matrix(
            true_labels_encoded=true_encoded,
            predicted_labels_encoded=pred_encoded,
            label_encoder=le,
            color_map={'some_color_key': {'some_label_value': '#ff0000'}}
        )

    """

    labels_true = np.unique(true_labels_encoded)
    labels_pred = np.unique(predicted_labels_encoded)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels_encoded, predicted_labels_encoded, labels=np.arange(len(label_encoder.classes_)))

    # Normalize the confusion matrix by row (i.e., by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_normalized = pd.DataFrame(cm_normalized[np.ix_(labels_true, labels_pred)],
                                 index=label_encoder.inverse_transform(labels_true),
                                 columns=label_encoder.inverse_transform(labels_pred))

    if diagonalize:
        # Sorting the confusion matrix to make it as diagonal as possible
        cost_matrix = -cm_normalized.values  # We need to minimize the cost, hence the negative sign
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Concatenate the optimal indices with the non-optimal ones
        row_ind = np.concatenate((row_ind, np.setdiff1d(np.arange(cm_normalized.shape[0]), row_ind)))
        col_ind = np.concatenate((col_ind, np.setdiff1d(np.arange(cm_normalized.shape[1]), col_ind)))

        cm_normalized = cm_normalized.iloc[row_ind, col_ind]
        labels_true_sorted = label_encoder.inverse_transform(labels_true)[row_ind]
        labels_pred_sorted = label_encoder.inverse_transform(labels_pred)[col_ind]
    else:
        labels_true_sorted = label_encoder.inverse_transform(labels_true)
        labels_pred_sorted = label_encoder.inverse_transform(labels_pred)

    def map_labels_to_colors(labels, label_color_dict, color_map):
        color_list = []
        for label in labels:
            color_dict = label_color_dict.get(label, {})
            colors = [color_map.get(key).get(color_dict.get(key, None), '#FFFFFF') for key in color_map]
            color_list.append(colors)
        return color_list

    row_colors = None
    if row_color_keys:
        row_colors = map_labels_to_colors(labels_true_sorted, true_label_color_dict, color_map)
        row_colors = pd.DataFrame(row_colors, index=labels_true_sorted)

    col_colors = None
    if col_color_keys:
        col_colors = map_labels_to_colors(labels_pred_sorted, predicted_label_color_dict, color_map)
        col_colors = pd.DataFrame(col_colors, index=labels_pred_sorted)

    xticklabels = predicted_ticklabels if predicted_ticklabels is not None else len(labels_pred) <= 40
    yticklabels = true_ticklabels if true_ticklabels is not None else len(labels_true) <= 40
    annot = annot if annot is not None else (len(labels_true) <= 40 and len(labels_pred) <= 40)


    g = sns.clustermap(cm_normalized, annot=annot, fmt=".2f", cmap="Blues",
                       row_colors=row_colors, col_colors=col_colors,
                       xticklabels=xticklabels, yticklabels=yticklabels,
                       row_cluster=False, col_cluster=False, figsize=figsize)

    g.ax_heatmap.set_title(title, y=1.05)
    g.ax_heatmap.set_ylabel('True label')
    g.ax_heatmap.set_xlabel('Predicted label')
    plt.show()

    return g
