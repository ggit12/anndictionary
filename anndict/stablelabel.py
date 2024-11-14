#stablelabel pipeline--an experimental pipeline to perform error correct on celltype annotations
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import anndata as ad
import os
import re
import pandas as pd
import random
import itertools
from IPython.display import HTML, display

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy.optimize import linear_sum_assignment

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import confusion_matrix


from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff


import harmonypy as hm

from .utils import create_color_map, add_label_to_adata

def get_slurm_cores():
    """
    Returns the total number of CPU cores allocated to a Slurm job based on environment variables.
    """
    # Get the number of CPUs per task (default to 1 if not set)
    cpus_per_task = int(os.getenv('SLURM_CPUS_PER_TASK', 1))
    
    # Get the number of tasks (default to 1 if not set)
    ntasks = int(os.getenv('SLURM_NTASKS', 1))
    
    # Calculate total cores
    total_cores = cpus_per_task * ntasks
    
    return total_cores

def pca_density_filter(data, n_components=3, threshold=0.10):
    """
    Calculate density contours for PCA-reduced data, return the density of all input data,
    and identify the unique variables that were included in the PCA.

    Parameters:
    - data: array-like, shape (n_samples, n_features)
    - n_components: int, number of components for PCA to reduce the data to.

    Returns:
    - pca_data: PCA-reduced data (None if all variables are constant).
    - density: Density values of all the points (None if all variables are constant).
    - unique_variables: List of unique variables that were included in the PCA (empty list if all variables are constant).
    """

    # Check for constant variables (these will not be used by PCA)
    non_constant_columns = np.var(data, axis=0) > 0
    
    # Skip the block if no non-constant variables are found
    if not np.any(non_constant_columns):
        return None, None, []

	# Adjust n_components if necessary
    n_features = np.sum(non_constant_columns)
    n_samples = data.shape[0]
    n_components = min(n_components, n_features, n_samples)
        
    unique_variables = np.arange(data.shape[1])[non_constant_columns]

    # Perform PCA reduction only on non-constant variables
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data[:, non_constant_columns])

    # Calculate the point density for all points
    kde = gaussian_kde(pca_data.T)
    density = kde(pca_data.T)

    # Determine the density threshold
    cutoff = np.percentile(density, threshold * 100)

    return density, cutoff, unique_variables.tolist()

def pca_density_wrapper(X, labels):
    """
    Apply calculate_density_contours_with_unique_variables to subsets of X indicated by labels.
    Returns a vector indicating whether each row in X is above the threshold for its respective label group.
    
    Parameters:
    - X: array-like, shape (n_samples, n_features)
    - labels: array-like, shape (n_samples,), labels indicating the subset to which each row belongs
    
    Returns:
    - index_vector: array-like, boolean vector of length n_samples indicating rows above the threshold
    """
    unique_labels = np.unique(labels)
    index_vector = np.zeros(len(X), dtype=bool)
    
    for label in unique_labels:
        subset = X[labels == label]
        if subset.shape[0] < 10:
            # If fewer than 10 cells, include all cells by assigning density = 1 and cutoff = 0
            density, cutoff = np.ones(subset.shape[0]), 0
        else:
            density, cutoff, _ = pca_density_filter(subset, n_components=3, threshold=0.10)
        
        # Mark rows above the threshold for this label
        high_density_indices = density > cutoff
        global_indices = np.where(labels == label)[0][high_density_indices]
        index_vector[global_indices] = True
    
    return index_vector

# def pca_density_adata_dict(adata_dict, keys):
#     """
#     This function applies PCA-based density filtering to the AnnData objects within adata_dict.
#     If adata_dict contains only one key, the filtering is applied directly. If there are multiple keys,
#     it recursively builds new adata dictionaries for subsets based on the provided keys and applies
#     the filtering to these subsets. Finally, it concatenates the results back into a single AnnData object.
    
#     Parameters:
#     - adata_dict: Dictionary of AnnData objects, with keys indicating different groups.
#     - keys: List of keys to stratify the AnnData objects further if more than one group is present.
    
#     Returns:
#     - AnnData object containing the results of PCA density filtering applied to each subset,
#       with results combined if the initial dictionary had more than one key.
#     """
#     if len(adata_dict) == 1:
#         # Only one group in adata_dict, apply density filter directly
#         label, adata = next(iter(adata_dict.items()))
#         X = adata.X
#         if X.shape[0] < 10:
#             density, cutoff = np.ones(X.shape[0]), 0
#         else:
#             density, cutoff, _ = pca_density_filter(X, n_components=3, threshold=0.10)
#         high_density_indices = density > cutoff
#         index_vector = np.zeros(X.shape[0], dtype=bool)
#         index_vector[high_density_indices] = True
#         add_label_to_adata(adata, np.arange(X.shape[0]), index_vector, 'density_filter')
#         return adata
#     else:
#         # More than one group, handle recursively
#         first_key = keys[0]
#         new_keys = keys[1:]
#         updated_adatas = {}
#         for key, group_adata in adata_dict.items():
#             new_adata_dict = build_adata_dict(group_adata, new_keys, {k: group_adata.obs[k].unique().tolist() for k in new_keys})
#             updated_adatas[key] = pca_density_wrapper(new_adata_dict, new_keys)
#         return concatenate_adata_dict(updated_adatas)
    
def pca_density_adata_dict(adata_dict, keys):
    """
    Applies PCA-based density filtering recursively on subsets of an AnnData dictionary. Each subset
    is determined by the provided keys. The function returns a dictionary where each AnnData object
    has an additional metadata key indicating the result of the density filter. The structure of the
    input dictionary is preserved, and each AnnData object's metadata is updated in-place.

    Parameters:
    - adata_dict: Dictionary of AnnData objects, with keys indicating different groups.
    - keys: List of keys to further stratify the AnnData objects if recursion is needed.

    Returns:
    - Dictionary: Updated adata_dict with the same keys but with each AnnData object having a new metadata key 'density_filter'.
    """
    from .dict import build_adata_dict, concatenate_adata_dict
    if len(keys) == 0:
        # No further keys to split by, apply filtering directly
        for label, adata in adata_dict.items():
            X = adata.X
            if X.shape[0] < 10:
                density, cutoff = np.ones(X.shape[0]), 0
            else:
                density, cutoff, _ = pca_density_filter(X, n_components=3, threshold=0.10)
            high_density_indices = density > cutoff
            index_vector = np.zeros(X.shape[0], dtype=bool)
            index_vector[high_density_indices] = True
            add_label_to_adata(adata, np.arange(X.shape[0]), index_vector, 'density_filter')
    else:
        # Recurse into further keys
        first_key = keys[0]
        new_keys = keys[1:]
        for label, adata in adata_dict.items():
            subgroups = build_adata_dict(adata, [first_key], {first_key: adata.obs[first_key].unique().tolist()})
            pca_density_wrapper(subgroups, new_keys)  # Recursively update each subgroup
            # Combine results back into the original adata entry
            updated_adata = concatenate_adata_dict(subgroups)
            adata_dict[label] = updated_adata

    return adata_dict



def stable_label(X, y, classifier, max_iterations=100, stability_threshold=0.05, moving_average_length=3, random_state=None):
    """
    Trains a classifier using a semi-supervised approach where labels are probabilistically reassigned based on classifier predictions.
    
    Parameters:
    - X: ndarray, feature matrix.
    - y: ndarray, initial labels for all data.
    - classifier: a classifier instance that implements fit and predict_proba methods.
    - max_iterations: int, maximum number of iterations for updating labels.
    - stability_threshold: float, threshold for the fraction of labels changing to consider the labeling stable.
    - moving_average_length: int, number of past iterations to consider for moving average.
    - random_state: int or None, seed for random number generator for reproducibility.
    
    Returns:
    - classifier: trained classifier.
    - history: list, percentage of labels that changed at each iteration.
    - iterations: int, number of iterations run.
    - final_labels: ndarray, the labels after the last iteration.
    """
    rng = check_random_state(random_state)
    history = []
    current_labels = y.copy()
    
    for iteration in range(max_iterations):

        #Call the wrapper function to get the index vector
        dense_on_pca = pca_density_wrapper(X, current_labels)

        #Get which labels are non_empty
        has_label = current_labels != -1

        #Train the classifier on cells that are dense in pca space and have labels
        mask = dense_on_pca & has_label
        classifier.fit(X[mask], current_labels[mask])
        
        # Predict label probabilities
        probabilities = classifier.predict_proba(X)

        #view some predicted probabilities for rows of X
        # print("Sample predicted probabilities for rows of X:", probabilities[:5])
        
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

def stable_label_adata(adata, feature_key, label_key, classifier, max_iterations=100, stability_threshold=0.05, moving_average_length=3, random_state=None):
    """
    A wrapper for train_classifier_with_probabilistic_labels that handles categorical labels.

    Parameters:
    - adata: AnnData object containing the dataset.
    - feature_key: str, key to access the features in adata.obsm.
    - label_key: str, key to access the labels in adata.obs.
    - classifier: classifier instance that implements fit and predict_proba methods.
    - max_iterations, stability_threshold, moving_average_length, random_state: passed directly to train_classifier_with_probabilistic_labels.

    Returns:
    - classifier: trained classifier.
    - history: list, percentage of labels that changed at each iteration.
    - iterations: int, number of iterations run.
    - final_labels: ndarray, text-based final labels after the last iteration.
    - label_encoder: the label encoder used during training (can be used to convert predictions to semantic labels)
    """
    # Initialize Label Encoder
    label_encoder = LabelEncoder()
    
    # Extract features and labels from adata
    X = adata.obsm[feature_key]
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
    trained_classifier, history, iterations, final_numeric_labels = stable_label(
        X, encoded_labels, classifier, max_iterations, stability_threshold, moving_average_length, random_state
    )
    
    # Decode the numeric labels back to original text labels
    final_labels = label_encoder.inverse_transform(final_numeric_labels)
    
    return trained_classifier, history, iterations, final_labels, label_encoder


def update_adata_labels_with_results(adata, results, new_label_key='stable_cell_type'):
    """
    Collects indices and labels from results and adds them to the AnnData object using add_label_to_adata function.

    Parameters:
    - adata: AnnData object to be updated.
    - results: Dictionary containing results, including indices and final_labels.
    - new_label_key: Name of the new column in adata.obs where the labels will be stored.
    """
    # Collect all indices and labels from the results
    all_indices = np.concatenate([info['indices'] for stratum, info in results.items()])
    all_labels = np.concatenate([info['final_labels'] for stratum, info in results.items()])

    # Call the function to add labels to adata
    add_label_to_adata(adata, all_indices, all_labels, new_label_key)


def plot_training_history(results, separate=True):
    """
    Plot the training history of a model, showing percent label change versus iteration.

    Parameters:
    results (dict): Dictionary where keys are strata names and values are dictionaries containing training history.
    separate (bool, optional): If True, plot each stratum's training history separately. If False, plot all strata together. Default is True.

    Returns:
    None
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

# def plot_changes(adata, true_label_key, predicted_label_key, percentage=True, stratum=None):
#     # Extract the series from the AnnData object's DataFrame
#     data = adata.obs[[predicted_label_key, true_label_key]].copy()
    
#     # Add a mismatch column that checks whether the predicted and true labels are different
#     data['Changed'] = data[true_label_key] != data[predicted_label_key]
    
#     # Group by predicted label key and calculate the sum of mismatches or the mean if percentage
#     if percentage:
#         change_summary = data.groupby(true_label_key)['Changed'].mean()
#     else:
#         change_summary = data.groupby(true_label_key)['Changed'].sum()
    
#     # Sort the summary in descending order
#     change_summary = change_summary.sort_values(ascending=False)
    
#     # Plotting
#     ax = change_summary.plot(kind='bar', color='red', figsize=(10, 6))
#     ax.set_xlabel(true_label_key)
#     ax.set_ylabel('Percentage of Labels Changed' if percentage else 'Count of Labels Changed')
#     ax.set_title(stratum)
#     ax.set_xticklabels(change_summary.index, rotation=90)
#     plt.xticks(fontsize=8)
#     plt.show()

def plot_changes(adata, true_label_key, predicted_label_key, percentage=True, stratum=None):
    """
    Plot the changes between true and predicted labels in an AnnData object.

    Parameters:
    adata (AnnData): Annotated data matrix.
    true_label_key (str): Key for the true labels in adata.obs.
    predicted_label_key (str): Key for the predicted labels in adata.obs.
    percentage (bool, optional): If True, plot the percentage of labels changed. If False, plot the count of labels changed. Default is True.
    stratum (str, optional): Title for the plot, often used to indicate the stratum. Default is None.

    Returns:
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
    
    # Group by true label key and calculate the sum of mismatches or the mean if percentage
    if percentage:
        change_summary = data.groupby(true_label_key)['Changed'].mean()
    else:
        change_summary = data.groupby(true_label_key)['Changed'].sum()
    
    # Sort the summary in descending order
    change_summary = change_summary.sort_values(ascending=False)
    
    # Plotting
    ax = change_summary.plot(kind='bar', color='red', figsize=(10, 6))
    ax.set_xlabel(true_label_key)
    ax.set_ylabel('Percentage of Labels Changed' if percentage else 'Count of Labels Changed')
    ax.set_title(stratum)
    ax.set_xticklabels(change_summary.index, rotation=90)
    plt.xticks(fontsize=8)
    plt.show()


def plot_confusion_matrix_from_adata(adata, true_label_key, predicted_label_key, title='Confusion Matrix',
                                     row_color_keys=None, col_color_keys=None, figsize=None, diagonalize=False,
                                     true_ticklabels=None, predicted_ticklabels=None, annot=None):
    """
    Wrapper function to plot a confusion matrix from an AnnData object, with optional row and column colors.
    
    Parameters:
    - adata: AnnData object containing the dataset.
    - true_label_key: str, key to access the true class labels in adata.obs.
    - predicted_label_key: str, key to access the predicted class labels in adata.obs.
    - title: str, title of the plot.
    - row_color_key: str, key for row colors in adata.obs.
    - col_color_key: str, key for column colors in adata.obs.
    """

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
                          true_labels=true_labels, predicted_labels=predicted_labels, figsize=figsize, diagonalize=diagonalize,
                          true_ticklabels=true_ticklabels, predicted_ticklabels=predicted_ticklabels, annot=annot)


def plot_confusion_matrix(true_labels_encoded, predicted_labels_encoded, label_encoder, color_map, title='Confusion Matrix', 
                          row_color_keys=None, col_color_keys=None,
                          true_label_color_dict=None, predicted_label_color_dict=None,
                          true_labels=None, predicted_labels=None, figsize=None,
                          diagonalize=False, true_ticklabels=None, predicted_ticklabels=None, annot=None):
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

    xticklabels = predicted_ticklabels if predicted_ticklabels is not None else (True if len(labels_pred) <= 40 else False)
    yticklabels = true_ticklabels if true_ticklabels is not None else (True if len(labels_true) <= 40 else False)
    annot = annot if annot is not None else (True if len(labels_true) <= 40 and len(labels_pred) <= 40 else False)


    g = sns.clustermap(cm_normalized, annot=annot, fmt=".2f", cmap="Blues",
                       row_colors=row_colors, col_colors=col_colors,
                       xticklabels=xticklabels, yticklabels=yticklabels,
                       row_cluster=False, col_cluster=False, figsize=figsize)

    g.ax_heatmap.set_title(title, y=1.05)
    g.ax_heatmap.set_ylabel('True label')
    g.ax_heatmap.set_xlabel('Predicted label')
    plt.show()

    return g

def plot_sankey(adata, cols, params=None):

    import holoviews as hv
    from collections import defaultdict
    hv.extension('bokeh')

    def f(plot, element):
        plot.handles['plot'].sizing_mode = 'scale_width'
        plot.handles['plot'].x_range.start = -1000
        plot.handles['plot'].x_range.end = 1500


    if params is None:
        params = {}
    
    obs = adata.obs[cols]
    
    # Creating unique labels for each column
    unique_labels = []
    label_dict = defaultdict(dict)
    for col_index, col in enumerate(cols):
        col_data = obs[col].astype(str).tolist()
        for item in col_data:
            if item not in label_dict[col_index]:
                unique_label = f"{item} ({col})"
                label_dict[col_index][item] = unique_label
                unique_labels.append(unique_label)
    
    # Creating source, target and value lists
    source = []
    target = []
    value = []
    for i in range(len(cols) - 1):
        ct_dict = defaultdict(int)
        for a, b in zip(obs[cols[i]].astype(str), obs[cols[i+1]].astype(str)):
            ct_dict[(a, b)] += 1
        for (a, b), v in ct_dict.items():
            source.append(label_dict[i][a])
            target.append(label_dict[i+1][b])
            value.append(v)
    
    # Creating DataFrame for Sankey
    sankey_data = pd.DataFrame({
        'source': source,
        'target': target,
        'value': value
    })
    
    # Appearance parameters
    cmap = params.get('cmap', 'Colorblind')
    label_position = params.get('label_position', 'outer')
    edge_line_width = params.get('edge_line_width', 0)
    edge_color = params.get('edge_color', 'value')  # allows grey edges
    show_values = params.get('show_values', False)
    node_padding = params.get('node_padding', 12)
    node_alpha = params.get('node_alpha', 1.0)
    node_width = params.get('node_width', 30)
    node_sort = params.get('node_sort', True)
    frame_height = params.get('frame_height', 1000)
    frame_width = params.get('frame_width', 2000)
    bgcolor = params.get('bgcolor', 'white')
    apply_ranges = params.get('apply_ranges', True)
    align_thr = params.get('align_thr', -0.1)
    label_font_size = params.get('label_font_size', '12pt')

    colormap_max = max(sankey_data['value'])
    norm = plt.Normalize(vmin=0, vmax=colormap_max)
    colors = plt.cm.plasma(norm(np.linspace(0, colormap_max, 128)))
    
    replace_these = np.where(norm(np.linspace(0, colormap_max, 128)) <= align_thr)[0]
    if replace_these.size > 0:
        colors[replace_these] = [[1, 1, 1, 0] for _ in range(len(replace_these))]
    
    edge_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    if edge_color == "grey":
        # edge_color = "grey"  # Set edge_color to grey
        edge_cmap = None  # No colormap for grey edges
    
    sankey = hv.Sankey(sankey_data, kdims=["source", "target"], vdims=["value"])
    sankey = sankey.opts(
        cmap=cmap, label_position=label_position, edge_color=edge_color, edge_cmap=edge_cmap, colorbar=True if edge_cmap else False,
        edge_line_width=edge_line_width, show_values=show_values, node_padding=node_padding, node_alpha=node_alpha,
        node_width=node_width, node_sort=node_sort, frame_height=frame_height, frame_width=frame_width,
        bgcolor=bgcolor, apply_ranges=apply_ranges, label_text_font_size=label_font_size, hooks=[f]
    )
    sankey = sankey.opts(clim=(0, colormap_max))

    return sankey

def save_sankey(plot, filename, adt_key=None):
    """
    Save a Holoviews Sankey plot as an SVG file.

    Parameters:
    plot : Holoviews plot, The Sankey plot to save.
    filename : str Base filename for the output SVG file.
    adt_key : str, optional Optional identifier to append to the filename.
    """
    import holoviews as hv
    from bokeh.io.webdriver import webdriver_control
    from bokeh.io import export_svgs

    # Reset web driver because sometimes the max connections is hit when writing plots
    webdriver_control.reset()

    # Remove '.svg' if it exists and append '{adt_key}.svg'
    filename = os.path.splitext(filename)[0]
    if adt_key:
        filename += f"_{adt_key}"
    filename += ".svg"

    plot = hv.render(plot)
    plot.output_backend = "svg"

    export_svgs(plot, filename=filename)

def plot_grouped_average(adata, label_value, adt_key=None):
    """
    Plots the average values specified in label_value across each group of label_keys in an AnnData object.

    Parameters:
    - adata: AnnData object containing the data.
    - label_value: dict, keys are the keys in adata.obs for grouping, values are the keys in adata.obs for the values to average.
    - key: to print specified key
    """
    print(adt_key)
    if not all(label in adata.obs for label in label_value.keys()):
        missing_keys = [label for label in label_value.keys() if label not in adata.obs]
        raise ValueError(f"Label key(s) {missing_keys} not found in adata.obs.")
    if not all(value in adata.obs for value in label_value.values()):
        missing_values = [value for value in label_value.values() if value not in adata.obs]
        raise ValueError(f"Value key(s) {missing_values} not found in adata.obs.")
    
    grouped_means = {}
    for label, value in label_value.items():
        grouped_means[label] = adata.obs.groupby(label)[value].mean()

    # Create a DataFrame from the grouped means
    df = pd.DataFrame(grouped_means)
    
    # Plot the results
    df.plot(kind='bar', figsize=(12, 8), color=plt.cm.Paired.colors)
    plt.xlabel('Groups')
    plt.ylabel('Average Scores')
    plt.title('Average Scores across Groups')
    plt.xticks(rotation=90)
    plt.legend(title='Scores')
    plt.show()

def plot_model_agreement(adata, group_by, sub_group_by, model_cols, granularity=2):
    """
    Plots the average values of specified model columns across varying levels of granularity.
    Parameters:
    - adata: AnnData object containing the data.
    - group_by: str, key in adata.obs for the main grouping (e.g., 'cell_type').
    - sub_group_by: str, key in adata.obs for the sub-grouping (e.g., 'tissue').
    - model_cols: list of str, column names for the models (e.g., ['agreement_model_1', 'agreement_model_2']).
    - granularity: int, level of detail in the plot (0 = models only, 1 = models within cell types, 2 = models within cell types and tissues).
    """
    if not all(col in adata.obs for col in model_cols):
        missing_cols = [col for col in model_cols if col not in adata.obs]
        raise ValueError(f"Columns {missing_cols} not found in adata.obs.")
    if group_by not in adata.obs:
        raise ValueError(f"Group key '{group_by}' not found in adata.obs.")
    if sub_group_by not in adata.obs:
        raise ValueError(f"Sub-group key '{sub_group_by}' not found in adata.obs.")

    # Pivot longer to get columns: group_by, sub_group_by, agreement, model_name
    melted = adata.obs.melt(id_vars=[group_by, sub_group_by], value_vars=model_cols,
                            var_name='model_name', value_name='agreement')

    if granularity == 0:
        # Calculate the average scores across all groups within each model
        grouped_means = melted.groupby('model_name')['agreement'].mean().sort_values(ascending=False)
        
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot the bar chart
        grouped_means.plot(kind='bar', ax=ax, colormap='Paired')
        
        # Add value labels on top of each bar
        for i, v in enumerate(grouped_means):
            ax.text(i, v, f'{v * 100:.0f}%', ha='center', va='bottom')

    elif granularity == 1:
        # Calculate the average scores within each model and cell type
        grouped_means = melted.groupby([group_by, 'model_name'])['agreement'].mean().unstack()
        grouped_means = grouped_means  # No model_order sorting in this version
        
        fig, ax = plt.subplots(figsize=(14, 8))
        grouped_means.plot(kind='bar', ax=ax, colormap='Paired')

    elif granularity == 2:
        # Calculate the average scores within each model, cell type, and tissue
        grouped_means = melted.groupby([group_by, sub_group_by, 'model_name'])['agreement'].mean().unstack(level=[1, 2])
        
        # Ensure the data is numeric and allow NaNs (missing values)
        grouped_means = grouped_means.apply(pd.to_numeric, errors='coerce')
        
        # Create a mask for NaN values
        mask = grouped_means.isnull()
        
        # Create a color mapping for tissues using the provided colors
        tissue_colors = [
            "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78",
            "#2ca02c", "#98df8a", "#d62728", "#ff9896",
            "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
            "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
            "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
            "#7f9ec0", "#ffab60", "#5ab4ac",
            # 5 additional concordant colors
            "#8b4513", "#ff6347", "#4682b4", "#dda0dd", "#ffd700"
        ]

        # Ensure that the number of tissues does not exceed the number of available colors
        tissues = grouped_means.columns.get_level_values(0).unique()
        tissue_colors = tissue_colors[:len(tissues)]

        # Create a color map based on the provided colors
        tissue_color_map = dict(zip(tissues, tissue_colors))

        # Create column colors based on tissues
        col_colors = [tissue_color_map[tissue] for tissue in grouped_means.columns.get_level_values(0)]

        
        # Plot heatmap with col_colors
        # fig, ax = plt.subplots(figsize=(16, 10))
        # Create the clustermapimport seaborn as sns

        # Use the 'viridis_r' colormap
        cmap = plt.get_cmap('viridis_r')

        # Set the color for NaN values (e.g., red)
        cmap.set_bad(color='black')

        # Create the clustermap with horizontal lines
        g = sns.clustermap(grouped_means, cmap=cmap, annot=False,
                        mask=mask, cbar_kws={'label': 'Agreement'},
                        linewidths=0, linecolor='black',
                        col_colors=col_colors, col_cluster=False, row_cluster=False,
                        yticklabels=1)

        # Get the axes object
        ax = g.ax_heatmap

        # # Remove all existing lines
        # ax.grid(False)

        # Add back only horizontal lines
        # ax.set_xticks(np.arange(grouped_means.shape[1]+1)-0.5, minor=True)
        # ax.set_yticks(np.arange(grouped_means.shape[0]+1)-0.5, minor=True)
        # ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        # ax.tick_params(which="minor", bottom=False, left=False)

        # Find where col_colors change
        color_changes = []
        for i in range(1, len(col_colors)):
            if col_colors[i] != col_colors[i-1]:
                color_changes.append(i)

        # Add vertical lines at color change positions
        for pos in color_changes:
            ax.axvline(pos, color='black', linewidth=0.5)

        return g
        
        # Create a legend for tissues
        # tissue_handles = [plt.Rectangle((0,0),1,1, color=color) for color in tissue_color_map.values()]
        # ax.legend(tissue_handles, tissue_color_map.keys(), title=sub_group_by, 
        #           loc='center left', bbox_to_anchor=(1, 0.5))
        # return fig, ax
        
    else:
        raise ValueError("Granularity must be 0, 1, or 2.")

    if granularity < 2:
        ax = plt.gca()  # Get current axis for granularity 0 and 1

    ax.set_xlabel(group_by if granularity > 0 else 'Model')
    ax.set_ylabel('Agreement')
    title = 'Average model agreement'
    if granularity == 0:
        title += ''
    elif granularity == 1:
        title += f' by {group_by}'
    elif granularity == 2:
        title += f' by {group_by} and {sub_group_by}'
    ax.set_title(title)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
    
    if granularity < 2:
        ax.legend(title='Models' + ('' if granularity == 0 else ' and Tissues'))

    plt.tight_layout()
    # Return the figure and axis for further editing
    return fig, ax

def plot_model_agreement_categorical(adata, group_by, sub_group_by, model_cols, granularity=2):
    """
    Plots the relative proportions of categories within specified model columns across varying levels of granularity.

    Parameters:
    - adata: AnnData object containing the data.
    - group_by: str, key in adata.obs for the main grouping (e.g., 'cell_type').
    - sub_group_by: str, key in adata.obs for the sub-grouping (e.g., 'tissue').
    - model_cols: list of str, column names for the models (e.g., ['model_1', 'model_2']). These should be categorical.
    - granularity: int, level of detail in the plot (0 = models only, 1 = models within cell types, 2 = models within cell types and tissues).
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Verify that the required columns exist
    if not all(col in adata.obs for col in model_cols):
        missing_cols = [col for col in model_cols if col not in adata.obs]
        raise ValueError(f"Columns {missing_cols} not found in adata.obs.")
    if group_by not in adata.obs:
        raise ValueError(f"Group key '{group_by}' not found in adata.obs.")
    if sub_group_by not in adata.obs:
        raise ValueError(f"Sub-group key '{sub_group_by}' not found in adata.obs.")

    # Ensure that model_cols are categorical or convert numeric types to categories
    for col in model_cols:
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            if pd.api.types.is_numeric_dtype(adata.obs[col]):
                adata.obs[col] = adata.obs[col].astype('category')
            else:
                raise ValueError(f"Column '{col}' must be categorical or convertible to categorical.")

    # Melt the dataframe to get long format
    melted = adata.obs.melt(
        id_vars=[group_by, sub_group_by],
        value_vars=model_cols,
        var_name='model_name',
        value_name='agreement'
    )

    # Ensure 'agreement' is categorical and reverse the order of categories
    if not pd.api.types.is_categorical_dtype(melted['agreement']):
        melted['agreement'] = melted['agreement'].astype('category')

    # Reverse the order of 'agreement' categories
    original_categories = melted['agreement'].cat.categories.tolist()
    reversed_categories = original_categories[::-1]
    melted['agreement'] = melted['agreement'].cat.reorder_categories(reversed_categories, ordered=True)

    if granularity == 0:
        # Calculate counts and proportions
        counts = melted.groupby(['model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby('model_name')['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Sort models based on total proportion of highest agreement category
        highest_agreement = counts.groupby('model_name')['proportion'].max().reset_index()
        sorted_models = highest_agreement.sort_values('proportion', ascending=False)['model_name']
        counts['model_name'] = pd.Categorical(counts['model_name'], categories=sorted_models, ordered=True)

        # Plot grouped bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(
            data=counts,
            x='model_name',
            y='proportion',
            hue='agreement',
            hue_order=reversed_categories,  # Use reversed categories
            ax=ax,
            order=sorted_models
        )

        # Add proportion labels on top of each bar
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height + 0.01,
                    # f'{height:.2f}',
                    f'{height * 100:.0f}%',
                    ha="center"
                )

        # Rotate x-axis tick labels to vertical
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        ax.set_xlabel('Model')
        ax.set_ylabel('Proportion')
        ax.set_title('Proportion of Agreement Categories by Model')
        ax.set_ylim(0, 1.05)
        ax.legend(title='Agreement Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig, ax

    elif granularity == 1:
        # Calculate counts and proportions
        counts = melted.groupby([group_by, 'model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby([group_by, 'model_name'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Sort 'group_by' categories based on total proportion
        total_per_group = counts.groupby(group_by)['proportion'].sum().reset_index()
        sorted_groups = total_per_group.sort_values('proportion', ascending=False)[group_by]
        counts[group_by] = pd.Categorical(counts[group_by], categories=sorted_groups, ordered=True)

        # Plot grouped bar chart with model_name as hue
        g = sns.catplot(
            data=counts,
            x=group_by,
            y='proportion',
            hue='agreement',
            hue_order=reversed_categories,  # Use reversed categories
            col='model_name',
            kind='bar',
            height=6,
            aspect=1,
            sharey=True,
            order=sorted_groups
        )

        g.set_axis_labels(group_by, "Proportion")
        g.set_titles("{col_name}")
        g.set(ylim=(0, 1.05))

        # Rotate x-axis tick labels to vertical
        for ax in g.axes.flat:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Add proportion labels on top of each bar
        for ax in g.axes.flatten():
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    ax.text(
                        p.get_x() + p.get_width() / 2.,
                        height + 0.01,
                        # f'{height:.2f}',
                        f'{height * 100:.0f}%',
                        ha="center"
                    )

        plt.tight_layout()
        return g

    elif granularity == 2:
        # Calculate counts and proportions
        counts = melted.groupby([group_by, sub_group_by, 'model_name', 'agreement']).size().reset_index(name='count')
        total_counts = counts.groupby([group_by, sub_group_by, 'model_name'])['count'].transform('sum')
        counts['proportion'] = counts['count'] / total_counts

        # Prepare data for heatmap
        pivot_table = counts.pivot_table(
            index=[group_by, sub_group_by],
            columns=['model_name', 'agreement'],
            values='proportion',
            fill_value=0
        )

        # Reverse the order of 'agreement' categories in columns
        pivot_table = pivot_table.reindex(columns=reversed_categories, level=2)

        # Sort index based on total proportion
        pivot_table['Total'] = pivot_table.sum(axis=1)
        pivot_table = pivot_table.sort_values('Total', ascending=False)
        pivot_table = pivot_table.drop(columns='Total')

        # Plot heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            pivot_table,
            cmap='viridis',
            annot=True,
            fmt=".2f",
            linewidths=0.5
        )
        plt.title(f'Proportion of Agreement Categories by {group_by} and {sub_group_by}')
        plt.tight_layout()
        return plt.gcf()

    else:
        raise ValueError("Granularity must be 0, 1, or 2.")

def kappa_adata(adata, cols):
    """
    Calculate pairwise Cohen's Kappa, average pairwise Kappa, and Fleiss' Kappa
    for categorical data in adata.obs[cols].
    Parameters:
    - adata: AnnData object.
    - cols: List of columns in adata.obs to use for calculating agreement.
    Returns:
    - Dictionary with keys 'pairwise', 'average_pairwise', and 'fleiss':
    - 'pairwise': A dictionary with pairwise Cohen's Kappa values.
    - 'average_pairwise': A dictionary with the average pairwise Kappa for each rater.
    - 'fleiss': The Fleiss' Kappa value for the overall agreement across all raters.
    """
    # Extract data from adata.obs based on the specified columns
    data = adata.obs[cols].to_numpy()
    num_raters = len(cols)
    kappa_scores = {'pairwise': {}, 'average_pairwise': {}, 'fleiss': None}

    # Calculate pairwise Cohen's Kappa
    for i in range(num_raters):
        rater_kappas = []
        for j in range(num_raters):
            if i != j:
                # Calculate Cohen's Kappa for each pair
                kappa = cohen_kappa_score(data[:, i], data[:, j])
                kappa_scores['pairwise'][(cols[i], cols[j])] = kappa
                rater_kappas.append(kappa)
        
        # Average Kappa for this rater (with every other rater)
        avg_kappa = np.mean(rater_kappas) if rater_kappas else None
        kappa_scores['average_pairwise'][cols[i]] = avg_kappa

    # Fleiss' Kappa calculation
    unique_categories = np.unique(data)
    category_map = {cat: idx for idx, cat in enumerate(unique_categories)}
    fleiss_data = np.zeros((data.shape[0], len(unique_categories)))

    # Count category occurrences per item (per row) using vectorized operations
    for i in range(data.shape[0]):
        row = np.array([category_map[val] for val in data[i]])
        fleiss_data[i] = np.bincount(row, minlength=len(unique_categories))

    # Calculate Fleiss' Kappa
    fleiss_kappa_value = fleiss_kappa(fleiss_data)
    kappa_scores['fleiss'] = fleiss_kappa_value

    return kappa_scores


def krippendorff_alpha_adata(adata, cols, level_of_measurement='nominal'):
    """
    Calculate Krippendorff's Alpha for categorical data in adata.obs[cols].
    Parameters:
    - adata: AnnData object.
    - cols: List of columns in adata.obs to use for calculating agreement.
    - level_of_measurement: The type of data ('nominal', 'ordinal', 'interval', 'ratio'). Default is 'nominal' (for categorical data).
    Returns:
    - Krippendorff's Alpha for the specified columns in adata.obs.
    """
    # Extract data from adata.obs based on the specified columns
    data = adata.obs[cols].to_numpy()

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Flatten the data, fit the encoder, and reshape back
    flat_data = data.ravel()
    encoded_flat = le.fit_transform(flat_data)
    encoded_data = encoded_flat.reshape(data.shape)

    # Transpose the data to match Krippendorff's alpha input format
    # (units as columns, raters as rows)
    encoded_data = encoded_data.T

    # Calculate Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data=encoded_data, level_of_measurement=level_of_measurement)
    
    return alpha


#harmony label functions
def harmony_label_transfer(adata_to_label, master_data, master_subset_column, label_column):
    """
    Perform Harmony integration and transfer labels from master_data to adata_to_label.

    This function subsets master_data based on a provided column to get the cells
    that match in the same column of adata_to_label. It then performs Harmony
    integration on the combined dataset and transfers the specified label column
    from master_data to adata_to_label.

    Parameters:
    adata_to_label : anndata.AnnData The AnnData object to which labels will be transferred.
    master_data : anndata.AnnData The master AnnData object containing the reference data and labels.
    master_subset_column : str The column name in .obs used for subsetting master_data to match adata_to_label.
    label_column : str The column name in .obs of master_data containing the labels to be transferred.

    Returns:
    anndata.AnnData The adata_to_label object with a new column 'harmony_labels' in .obs containing the transferred labels.
    """
    
    # Subset master_data based on the provided column to get matching cells
    matching_cells = master_data.obs[master_data.obs[master_subset_column].isin(adata_to_label.obs[master_subset_column])]
    master_subset = master_data[matching_cells.index]
    
    # Combine adata_to_label and the subset of master_data
    combined_data = ad.concat([adata_to_label, master_subset])
    
    # Perform Harmony integration
    sc.tl.pca(combined_data, svd_solver='arpack')
    harmony_results = hm.run_harmony(combined_data.obsm['X_pca'], combined_data.obs, master_subset_column)
    combined_data.obsm['X_harmony'] = harmony_results.Z_corr.T
    
    # Separate the integrated data back into the original datasets
    adata_to_label_integrated = combined_data[:adata_to_label.n_obs]
    master_integrated = combined_data[adata_to_label.n_obs:]
    
    # Transfer labels from master_data to adata_to_label using the integrated data
    sc.pp.neighbors(master_integrated, use_rep='X_harmony')
    sc.tl.umap(master_integrated)
    sc.tl.leiden(master_integrated, resolution=0.5)
    
    # Transfer the specific label column from master_integrated to adata_to_label_integrated
    master_labels = master_integrated.obs[label_column]
    adata_to_label_integrated.obs[label_column] = master_labels.reindex(adata_to_label_integrated.obs.index).fillna(method='ffill')
    
    # Return adata_to_label with the new labels
    adata_to_label.obs['harmony_labels'] = adata_to_label_integrated.obs[label_column]
    
    # return adata_to_label
