#adata_dict functions
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_random_state
from sklearn.preprocessing import LabelEncoder
import scanpy as sc
import anndata as ad
import os
import pandas as pd
import random
import itertools
from IPython.display import HTML, display

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def check_and_create_strata(adata, strata_keys):
    # Check if any of the strata_keys are not present in adata.obs
    if any(key not in adata.obs.columns for key in strata_keys):
        raise ValueError("one or more of your stratifying variables does not exist in adata.obs")
    
    # Create a new column that combines the values of existing strata_keys, if not already present
    strata_key = '_'.join(strata_keys)
    if strata_key not in adata.obs.columns:
        adata.obs[strata_key] = adata.obs[strata_keys].astype(str).agg('_'.join, axis=1).astype('category')
    else:
        #make sure it's categorical
        adata.obs[strata_key] = adata.obs[strata_key].astype('category')

    return strata_key

def build_adata_dict(adata, strata_keys, desired_strata):
    if isinstance(desired_strata, list):
        # Directly use the list of strata
        return build_adata_dict_main(adata, strata_keys, desired_strata)
    elif isinstance(desired_strata, dict):
        # Generate all combinations of desired strata values across strata_keys
        all_combinations = itertools.product(*(desired_strata[key] for key in strata_keys))
        # Convert tuples of combinations to a format suitable for strata_keys
        combined_strata_list = ['_'.join(map(str, comb)) for comb in all_combinations]
        return build_adata_dict_main(adata, strata_keys, combined_strata_list)
    else:
        raise ValueError("desired_strata must be either a list or a dictionary of lists")

def build_adata_dict_main(adata, strata_keys, desired_strata):
    # Check and create stratifying variable in adata
    strata_key = check_and_create_strata(adata, strata_keys)
    # Initialize the dictionary to store subsets
    subsets_dict = {}
    # Filter adata for each desired stratum and add to the dictionary
    for stratum in desired_strata:
        if stratum in adata.obs[strata_key].cat.categories:
            subset = adata[adata.obs[strata_key] == stratum]
            subsets_dict[stratum] = subset
        else:
            print(f"Warning: '{stratum}' is not a valid category in '{strata_key}'.")
    return subsets_dict

def concatenate_adata_dict(adata_dict, join='outer'):
    """
    Concatenates all AnnData objects in adata_dict into a single AnnData object.

    Parameters:
    - adata_dict: Dictionary of AnnData objects with keys as identifiers.

    Returns:
    - AnnData: A single AnnData object that combines all the subsets in adata_dict.
    """
    # Extract all AnnData objects from the dictionary
    adatas = list(adata_dict.values())

    # Concatenate all AnnData objects into one
    if adatas:
        full_adata = sc.concat(adatas, join=join, index_unique='-')
        return full_adata
    else:
        raise ValueError("adata_dict is empty. No data available to concatenate.")
    
def summarize_metadata_adata_dict(adata_dict, columns):
    """
    Generate summary tables for each AnnData object in the dictionary using the summarize_metadata function.
    
    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - columns (list of str): List of columns from the metadata to summarize. Use '*' to specify joint frequencies of multiple columns.
    
    Returns:
    - dict: A dictionary of summary dictionaries for each AnnData object in the adata_dict.
    """
    summary_dict = {}
    for stratum, adata in adata_dict.items():
        summary_dict[stratum] = summarize_metadata(adata, columns)
    return summary_dict

def display_html_summary_adata_dict(summary_dict_dict):
    """
    Display separate HTML tables for each metadata category in the summary dictionaries,
    arranging up to three tables in a row before starting a new line.
    
    Parameters:
    - summary_dict_dict (dict): A dictionary of summary dictionaries for each AnnData object in the adata_dict.
    """
    for stratum, summary_dict in summary_dict_dict.items():
        print(f"Summary for {stratum}:")
        display_html_summary(summary_dict)

def adata_dict_fapply(adata_dict, func, **kwargs):
    """
    Applies a given function to each AnnData object in the adata_dict.

    Parameters:
    - adata_dict: Dictionary of AnnData objects with keys as identifiers.
    - func: Function to apply to each AnnData object in the dictionary.
    - kwargs: Additional keyword arguments to pass to the function.

    Returns:
    - None: The function modifies the AnnData objects in place.
    """
    for key, adata in adata_dict.items():
        func(adata, **kwargs)


def subsample_adata_dict(adata_dict, n_obs=None, random_state=0):
    """
    Subsamples each AnnData object in the dictionary using Scanpy's subsample function.
    
    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - n_obs (int, optional): Number of cells to subsample. If None, subsampling is not performed. Default is None.
    - random_state (int, optional): Seed for random number generator. Default is 0.
    
    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    for stratum, adata in adata_dict.items():
        if n_obs is not None:
            sc.pp.subsample(adata, n_obs=n_obs, random_state=random_state)
    
def normalize_adata_dict(adata_dict, target_sum=1e4):
    """
    Normalizes each AnnData object in the dictionary using Scanpy's normalize_total.
    """
    for stratum, adata in adata_dict.items():
        sc.pp.normalize_total(adata, target_sum=target_sum)

def log_transform_adata_dict(adata_dict):
    """
    Log-transforms each AnnData object in the dictionary using Scanpy's log1p.
    """
    for stratum, adata in adata_dict.items():
        sc.pp.log1p(adata)

def set_high_variance_genes(adata_dict, n_top_genes=2000, subset=False):
    """
    Identifies high-variance genes in each AnnData object in the dictionary.

    Parameters:
    n_top_genes (int): Number of top high-variance genes to keep.
    subset (bool): Whether to subset the adata to only include high-variance genes.
    """
    for stratum, adata in adata_dict.items():
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=subset)

def scale_adata_dict(adata_dict):
    """
    Scales each AnnData object in the dictionary using Scanpy's scale function.
    """
    for stratum, adata in adata_dict.items():
        sc.pp.scale(adata, max_value=10)

def pca_adata_dict(adata_dict, n_comps=50, mask_var='highly_variable'):
    """
    Performs PCA on each AnnData object in the dictionary using Scanpy's pca function.

    Parameters:
    n_comps (int): Number of principal components to compute.
    mask_var (str): Set to 'highly_variable' to use highly_variable genes (if set). Set to None to use all genes.
    """
    for stratum, adata in adata_dict.items():
        sc.pp.pca(adata, n_comps=n_comps, mask_var=mask_var)

def calculate_umap_adata_dict(adata_dict, use_rep):
    """
    Calculates UMAP embeddings for each subset in the adata_dict.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    use_rep (str): The key in .obsm where the representation matrix is stored.

    Returns:
    dict: A dictionary with the same keys as adata_dict, but values now include UMAP coordinates.
    """
    for stratum, adata_subset in adata_dict.items():
        # Ensure the representation exists
        if use_rep in adata_subset.obsm:
            # Compute neighbors , necessary for UMAP
            sc.pp.neighbors(adata_subset, use_rep=use_rep)
            # Calculate UMAP
            sc.tl.umap(adata_subset)
        else:
            print(f"Representation '{use_rep}' not found in .obsm of '{stratum}' subset.")
    return adata_dict

def plot_umap_adata_dict(adata_dict, color_by):
    """
    Plots UMAP embeddings for each AnnData object in adata_dict, colored by a specified variable.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    color_by (str): A variable by which to color the UMAP plots, typically a column in .obs.
    """
    for stratum, adata_subset in adata_dict.items():
        # Check if UMAP has been computed
        if 'X_umap' in adata_subset.obsm:
            title = [f"{stratum} {color}" for color in color_by]
            sc.pl.umap(adata_subset, color=color_by, title=title)
        else:
            print(f"UMAP not computed for '{stratum}'. Please compute UMAP before plotting.")

def write_h5ad_adata_dict(adata_dict, directory, file_prefix=""):
    """
    Saves each AnnData object from adata_dict into separate .h5ad files.

    Parameters:
    - adata_dict: Dictionary of AnnData objects, with keys as identifiers.
    - directory: String, directory path where .h5ad files will be saved.
    - file_prefix: String, optional prefix for the filenames.

    Example:
    - If ``file_prefix`` is ``experiment1_``, files will be named ``experiment1_group1.h5ad`` for a key ``group1``.
    """
    # Ensure the directory exists, create if it doesn't
    os.makedirs(directory, exist_ok=True)

    # Iterate over the dictionary and save each AnnData object
    for key, adata in adata_dict.items():
        # Construct the file path
        file_path = os.path.join(directory, f"{file_prefix}{key}.h5ad")
        # Save the AnnData object
        sc.write(file_path, adata)


def stable_label_adata_dict(adata_dict, feature_key, label_key, classifier_class, max_iterations=100, stability_threshold=0.05, moving_average_length=3, random_state=None):
    """
    Trains a classifier for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    feature_key (str): Key to access the features in adata.obsm.
    label_key (str): Key to access the labels in adata.obs.
    classifier: Classifier instance that implements fit and predict_proba methods.
    max_iterations, stability_threshold, moving_average_length, random_state: Additional parameters for training.

    Returns:
    results: Dict, keys are the identifiers from adata_dict and values are dictionaries containing the outputs from stable_label_adata.
    """
    stable_label_results = {}
    for stratum, adata in adata_dict.items():
        print(f"Training classifier for {stratum}")

        #create a classifier for this stratum
        classifier = classifier_class(random_state=random_state, max_iter=1000, n_jobs=get_slurm_cores())


        indices = np.array(adata.obs.index)
        trained_classifier, history, iterations, final_labels, label_encoder = stable_label_adata(
            adata, feature_key, label_key, classifier, max_iterations, stability_threshold, moving_average_length, random_state
        )

        stable_label_results[stratum] = {
            'classifier': trained_classifier,
            'history': history,
            'iterations': iterations,
            'final_labels': final_labels,
            'label_encoder': label_encoder,
            'indices': indices
        }

    return stable_label_results

def predict_labels_adata_dict(adata_dict, stable_label_results, feature_key):
    """
    Predicts labels for each AnnData object in adata_dict using the corresponding classifier from stable_label_results,
    and converts numeric predictions back to text labels.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    stable_label_results (dict): Dictionary with keys as identifiers and values as dictionaries containing the trained classifier and other outputs from stable_label_adata.
    feature_key (str): Key to access the features in adata.obsm.

    Returns:
    predictions_dict (dict): Dictionary with keys as identifiers from adata_dict and values as predicted text labels.
    """
    predictions_dict = {}

    for stratum, adata in adata_dict.items():
        if stratum in stable_label_results:
            classifier = stable_label_results[stratum]['classifier']
            label_encoder = stable_label_results[stratum]['label_encoder']
            X = adata.obsm[feature_key]

            # Predict the numeric labels using the trained classifier
            predicted_numeric_labels = classifier.predict(X)

            # Check if predicted labels are within the range of the label encoder's classes
            valid_labels = set(label_encoder.transform(label_encoder.classes_))
            invalid_labels = set(predicted_numeric_labels) - valid_labels

            if invalid_labels:
                print(f"Error: Predicted labels {invalid_labels} are not in the label encoder's classes for {stratum}")
                continue

            # Convert numeric predictions back to text labels
            predicted_text_labels = label_encoder.inverse_transform(predicted_numeric_labels)
            
            # Get the indices of the cells
            indices = np.array(adata.obs.index)

            predictions_dict[stratum] = {
                'indices': indices,
                'predicted_labels': predicted_text_labels
            }
        else:
            print(f"No classifier found for {stratum}. Skipping prediction.")

    return predictions_dict


def update_adata_labels_with_stable_label_results_dict(adata_dict, stable_label_results_dict, new_label_key='stable_cell_type'):
    """
    Updates each AnnData object in adata_dict with new labels from stable_label_results_dict.

    Parameters:
    - adata_dict: Dictionary of AnnData objects to be updated.
    - stable_label_results_dict: Dictionary of dictionaries containing results, including indices and final_labels for each AnnData key.
    - new_label_key: Name of the new column in adata.obs where the labels will be stored.
    """
    update_adata_dict_with_label_dict(adata_dict, stable_label_results_dict, new_label_key=new_label_key, label_key='final_labels')


def update_adata_labels_with_predictions_dict(adata_dict, predictions_dict, new_label_key='predicted_cell_type'):
    """
    Updates each AnnData object in adata_dict with new labels from predictions_dict.

    Parameters:
    - adata_dict: Dictionary of AnnData objects to be updated.
    - predictions_dict: Dictionary of predicted labels for each AnnData key.
    - new_label_key: Name of the new column in adata.obs where the labels will be stored.
    """
    update_adata_dict_with_label_dict(adata_dict, predictions_dict, new_label_key=new_label_key, label_key='predicted_labels')


def update_adata_dict_with_label_dict(adata_dict, results_dict, new_label_key=None, label_key=None):
    """
    Wrapper function to update each AnnData object in adata_dict with new labels from results_dict.
    Accepts either 'final_labels' or 'predicted_labels' as the label key.
    results_dict can be either 1) stable_label_results: the object returned by stable_label_adata_dict()
    or 2) predictions_dict: the object returned by predict_labels_adata_dict

    Parameters:
    - adata_dict: Dictionary of AnnData objects to be updated.
    - results_dict: Dictionary containing results, including indices and labels for each AnnData key.
    - new_label_key: Name of the new column in adata.obs where the labels will be stored.
    - label_key: Key to access the labels in results_dict (either 'final_labels' or 'predicted_labels').
    """
    for key, adata in adata_dict.items():
        if key in results_dict:
            subset_results = results_dict[key]
            indices = subset_results['indices']
            labels = subset_results[label_key]

            add_label_to_adata(adata, indices, labels, new_label_key)




def plot_changes_adata_dict(adata_dict, true_label_key, predicted_label_key, percentage=True):
    """
    Applies the plot_final_mismatches function to each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    predicted_label_key (str): The key in obs for predicted labels.
    true_label_key (str): The key in obs for true labels.
    percentage (bool): If True, plot percentages, otherwise plot counts.
    """
    for stratum, adata in adata_dict.items():
        print(f"Plotting changes for {stratum}")
        plot_changes(adata, true_label_key, predicted_label_key, percentage, stratum)

def plot_confusion_matrix_adata_dict(adata_dict, true_label_key, predicted_label_key,
                                     row_color_keys=None, col_color_keys=None):
    """
    Applies the plot_confusion_matrix_from_adata function to each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    true_label_key (str): The key in obs for true class labels.
    predicted_label_key (str): The key in obs for predicted class labels.
    title (str): Title of the plot, which will be prefixed with the stratum name.
    row_color_keys (list): Optional keys for row colors in adata.obs.
    col_color_keys (list): Optional keys for column colors in adata.obs.
    """
    for stratum, adata in adata_dict.items():
        # Customize title for each subset
        subset_title = f"Confusion Matrix for {stratum}"
        plot_confusion_matrix_from_adata(adata, true_label_key, predicted_label_key, title=subset_title,
                                         row_color_keys=row_color_keys, col_color_keys=col_color_keys)
        
def plot_spatial_adata_dict(adata_dict, color_by, crop_coord=None):
    """
    Plots spatial data for each AnnData object in adata_dict, colored by a specified variable.

    Parameters:
    adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    color_by (str): A variable by which to color the spatial plots, typically a column in .obs.
    """
    for stratum, adata_subset in adata_dict.items():
        # Check if spatial coordinates are available
        if 'spatial' in adata_subset.obsm:
            title = [f"{stratum} {color}" for color in color_by]
            sc.pl.spatial(adata_subset, color=color_by, title=title, crop_coord=crop_coord)
        else:
            print(f"Spatial coordinates not available for '{stratum}'. Please add spatial data before plotting.")
