"""
Functions to plot the training history of stabilizing classifier, 
and a bar graph of the number of label differences between two columns.
"""

import matplotlib.pyplot as plt

from anndata import AnnData

def plot_training_history(
    results: dict,
    separate: bool = True
) -> None:
    """
    Plot the training history of a model, showing percent label change versus iteration.

    Parameters
    ------------
    results
        :class:`dict` where keys are strata names and values are dictionaries containing training history. From :func:`anndict.annotate.cells.stabilizing_classifier`.

    separate
        If ``True``, plot each stratum's training history separately. If ``False``, plot all strata together.

    Returns
    ---------
    ``None``

    See Also
    ---------
    :func:`anndict.annotate.cells.stabilizing_classifier`
    """
    if separate:
        for stratum, info in results.items():
            plt.figure(figsize=(10, 6))
            plt.plot(info['history'], marker='o')
            plt.title(f'Percent Label Change vs. Iteration - {stratum}')
            plt.xlabel('Iteration')
            plt.ylabel('Percent Label Change')
            plt.grid(True)
            plt.show()
    else:
        plt.figure(figsize=(10, 6))
        for stratum, info in results.items():
            plt.plot(info['history'], marker='.', label=stratum)
        plt.title('Percent Label Change vs. Iteration - All Strata')
        plt.xlabel('Iteration')
        plt.ylabel('Percent Label Change')
        plt.grid(True)
        plt.legend()
        plt.show()

def plot_changes(
    adata: AnnData,
    true_label_key: str,
    predicted_label_key: str,
    plot_percentage: bool = True,
    stratum: str = None
) -> None:
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
    ---------
    None
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
        change_summary = data.groupby(true_label_key)['Changed'].mean()
    else:
        change_summary = data.groupby(true_label_key)['Changed'].sum()

    # Sort the summary in descending order
    change_summary = change_summary.sort_values(ascending=False)

    # Plotting
    ax = change_summary.plot(kind='bar', color='red', figsize=(10, 6))
    ax.set_xlabel(true_label_key)
    ax.set_ylabel('Percentage of Labels Changed' if plot_percentage else 'Count of Labels Changed')
    ax.set_title(stratum)
    ax.set_xticklabels(change_summary.index, rotation=90)
    plt.xticks(fontsize=8)
    plt.show()
