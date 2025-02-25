"""
Train a classifier iterativley, resampling labels probabilistically until they stabilize. Built-in outlier removal based on density in PCA space (removes the 10% least dense points).
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_random_state
from sklearn.base import ClassifierMixin
from anndata import AnnData

from anndict.utils.pca_density_filter import pca_density_subsets


def stable_label(
    x: np.ndarray,
    y: np.ndarray,
    classifier: ClassifierMixin,
    *,
    max_iterations: int = 100,
    stability_threshold: float = 0.05,
    moving_average_length: int = 3,
    random_state: int | None = None
) -> tuple[ClassifierMixin, list[float], int, np.ndarray]:
    """
    Trains a classifier using a semi-supervised approach where labels are probabilistically reassigned based on classifier predictions.

    Parameters
    -----------
    x
        feature matrix.

    y
        initial labels for all data.

    classifier
        a classifier instance that implements fit and predict_proba methods.

    max_iterations
        maximum number of iterations for updating labels.

    stability_threshold 
        threshold for the fraction of labels changing to consider the labeling stable.

    moving_average_length
        number of past iterations to consider for moving average.

    random_state
        seed for random number generator for reproducibility.

    Returns
    --------
    classifier
        trained classifier.

    history
        percentage of labels that changed at each iteration.

    iterations
        number of iterations run.

    final_labels
        the labels after the last iteration.

    @todo
    - switch pca_density_subsets to use pca_density_filter_main or pca_density_filter_adata
    """
    _ = check_random_state(random_state)
    history = []
    current_labels = y.copy()

    for iteration in range(max_iterations):

        #Call the wrapper function to get the index vector
        dense_on_pca = pca_density_subsets(x, current_labels)

        #Get which labels are non_empty
        has_label = current_labels != -1

        #Train the classifier on cells that are dense in pca space and have labels
        mask = dense_on_pca & has_label
        classifier.fit(x[mask], current_labels[mask])

        # Predict label probabilities
        probabilities = classifier.predict_proba(x)

        #view some predicted probabilities for rows of x
        # print("Sample predicted probabilities for rows of x:", probabilities[:5])

        # Sample new labels from the predicted probabilities
        new_labels = np.array([np.argmax(prob) if max(prob) > 0.8 else current_labels[i] for i, prob in enumerate(probabilities)])
        # new_labels = np.array([np.argmax(prob) for i, prob in enumerate(probabilities)])

        # def transform_row(row, p):
        #     """
        #     Transform an array by raising each element to the power of p and then normalizing these values
        #     so that their sum is 1.

        #     Parameters:
        #     row (np.array): The input array to be transformed.
        #     p (float): The power to which each element of the array is raised.

        #     Returns:
        #     np.array: An array where each element is raised to the power of p and
        #             normalized so that the sum of all elements is 1.
        #     """
        #     row = np.array(row)  # Ensure input is a numpy array
        #     powered_row = np.power(row, p)  # Raise each element to the power p
        #     normalized_row = powered_row / np.sum(powered_row)  # Normalize the powered values
        #     return normalized_row

        # new_labels = np.array([np.random.choice(len(row), p=transform_row(row, 4)) for row in probabilities])

        #randomly flip row label with probability given by confidence in assignment--hopefully prevents "cell type takeover"
        # def random_bool(p):
        #     weights = [p, 1-p]
        #     weights = [w**2 for w in weights]
        #     weights = [w/sum(weights) for w in weights]
        #     return random.choices([False, True], weights=weights, k=1)[0]

        # new_labels = np.array([np.random.choice(len(row)) if random_bool(max(row)) else current_labels[i] for i, row in enumerate(probabilities)])

        # Determine the percentage of labels that changed
        changes = np.mean(new_labels != current_labels)

        # Record the percentage of labels that changed
        history.append(changes)

        # Compute moving average of label changes over the last n iterations
        if len(history) >= moving_average_length:
            moving_average = np.mean(history[-moving_average_length:])
            if moving_average < stability_threshold:
                break

        #update current labels
        current_labels = new_labels

        if len(np.unique(current_labels)) == 1:
            print("converged to single label.")
            break

    return classifier, history, iteration + 1, current_labels

def stable_label_adata(
    adata: AnnData,
    label_key: str,
    feature_key: str,
    classifier: ClassifierMixin,
    **kwargs
) -> tuple[ClassifierMixin, list[float], int, np.ndarray, LabelEncoder]:
    """
    A wrapper for :func:`stable_label` that handles categorical labels.

    Parameters
    -----------
    adata
        AnnData object containing the dataset.

    label_key
        key to access the labels in ``adata.obs``.

    feature_key
        key of data to use in ``adata.obsm``, or ``'use_X'`` to use ``adata.X``.

    classifier
        classifier instance that implements fit and predict_proba methods.

    **kwargs
        keyword args passed directly to :func:`stable_label`.

    Returns
    -----------
    A tuple containing:
    - classifier: trained classifier.
    - history: percentage of labels that changed at each iteration.
    - iterations: number of iterations run.
    - final_labels: text-based final labels after the last iteration.
    - label_encoder: the label encoder used during training (can be used to convert predictions to semantic labels)
    """
    # Initialize Label Encoder
    label_encoder = LabelEncoder()

    # Extract features and labels from adata
    x = adata.X if feature_key == "use_X" else adata.obsm[feature_key]
    y = adata.obs[label_key].values

    # Define a list of values to treat as missing
    missing_values = set(['missing', 'unannotated', '', 'NA'])

    # Replace defined missing values with np.nan
    y = np.array([np.nan if item in missing_values or pd.isna(item) else item for item in y])

    # Encode categorical labels to integers
    encoded_labels = label_encoder.fit_transform(y)

    # Map np.nan's encoding index to -1
    if np.nan in label_encoder.classes_:
        nan_label_index = label_encoder.transform([np.nan])[0]
        encoded_labels[encoded_labels == nan_label_index] = -1

    # Train the classifier using the modified training function that handles probabilistic labels
    trained_classifier, history, iterations, final_numeric_labels = stable_label(x, encoded_labels, classifier, **kwargs)

    # Decode the numeric labels back to original text labels
    final_labels = label_encoder.inverse_transform(final_numeric_labels)

    return trained_classifier, history, iterations, final_labels, label_encoder
