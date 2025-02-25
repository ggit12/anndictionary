"""
Functions to plot the training history of stabilizing classifier, 
and a bar graph of the number of label differences between two columns.
"""

import matplotlib.pyplot as plt

from matplotlib.figure import Figure, Axes
from anndata import AnnData

def plot_training_history(
    history: list[float],
    adt_key: tuple[str,...] | None = None
) -> tuple[Figure, Axes]:
    """
    Plot the training history of a model, showing percent label change versus iteration.

    Parameters
    ------------
    history
        The training history. From :func:`anndict.utils.stabilizing_classifier`.

    adt_key
        Used by :func:`adata_dict_fapply` when wrapping this function.

    Returns
    ---------
    The figure.

    See Also
    ---------
    :func:`anndict.utils.stabilizing_classifier` : The function that returns ``history``.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, marker='o')
    ax.set_title(f'Percent Label Change vs. Iteration - {adt_key}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Percent Label Change')
    ax.grid(True)
    return fig, ax


def plot_label_changes(
    adata: AnnData,
    true_label_key: str,
    predicted_label_key: str,
    plot_percentage: bool = True,
    stratum: str = None
) -> tuple[Figure, Axes]:
    """
    Plot the changes between true and predicted labels in an AnnData object.

    Parameters
    ------------
    adata
        An :class:`AnnData`.

    true_label_key
        Key for the true labels in ``adata.obs``.

    predicted_label_key
        Key for the predicted labels in ``adata.obs``.

    plot_percentage
        If ``True``, plot the percentage of labels changed. If ``False``, plot the count of labels changed.

    stratum
         Title for the plot, often used to indicate the stratum. Default is None.

    Returns
    -------
    The figure and axis of the plot.
    """
    # Extract the series from the AnnData object's DataFrame
    data = adata.obs[[predicted_label_key, true_label_key]].copy()

    # Convert to categorical with a common category set
    common_categories = list(set(data[true_label_key].cat.categories).union(set(data[predicted_label_key].cat.categories)))
    data[true_label_key] = data[true_label_key].cat.set_categories(common_categories)
    data[predicted_label_key] = data[predicted_label_key].cat.set_categories(common_categories)

    # Add a mismatch column that checks whether the predicted and true labels are different
    data['Changed'] = data[true_label_key] != data[predicted_label_key]

    # Group by true label key and calculate the sum of mismatches or the mean if plot_percentage
    if plot_percentage:
        change_summary = data.groupby(true_label_key, observed=False)['Changed'].mean()
    else:
        change_summary = data.groupby(true_label_key, observed=False)['Changed'].sum()

    # Sort the summary in descending order
    change_summary = change_summary.sort_values(ascending=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    change_summary.plot(kind='bar', color='red', ax=ax)
    ax.set_xlabel(true_label_key)
    ax.set_ylabel('Percentage of Labels Changed' if plot_percentage else 'Count of Labels Changed')
    ax.set_title(stratum)
    ax.set_xticklabels(change_summary.index, rotation=90)
    plt.xticks(fontsize=8)
    plt.show()

    return fig, ax
