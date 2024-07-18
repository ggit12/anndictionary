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

import inspect

from .metadata_summary import summarize_metadata, display_html_summary
from .stablelabel import (
    get_slurm_cores,
    pca_density_filter,
    pca_density_wrapper,
    pca_density_adata_dict,
    stable_label,
    stable_label_adata,
    update_adata_labels_with_results,
    plot_training_history,
    plot_changes,
    plot_confusion_matrix_from_adata,
    plot_confusion_matrix
)
from .utils import add_label_to_adata, create_color_map


class AdataDict(dict):
    """
    AdataDict is a dictionary-like container where values are AnnData objects.
    
    This class provides two main functionalities:
    1. It behaves like an AnnData object by passing each method through to each AnnData in the dictionary. For example, adata_dict.obs.group_by("column") will apply the group_by method on the obs attribute of each AnnData object in the dictionary.
    2. It has a method fapply(func, kwargs) that applies a given function func with arguments kwargs to each AnnData object in the dictionary.
    
    Methods:
    __getattr__(attr) Dynamically creates methods that apply the corresponding method of AnnData objects in the dictionary.
        
    fapply(func, kwargs) Applies the provided function func with additional arguments kwargs to each AnnData object in the dictionary.
    
    Attributes:
    Inherits attributes from the built-in dict class.

    """
    def __getattr__(self, attr):
        def method(*args, **kwargs):
            results = {}
            for key, adata in self.items():
                func = getattr(adata, attr)
                results[key] = func(*args, **kwargs)
            return results
        return method
    
    def fapply(self, func, **kwargs):
        results = {}
        for key, adata in self.items():
            results[key] = func(adata, **kwargs)
        return results


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
    sig = inspect.signature(func)
    accepts_key = 'key' in sig.parameters

    for key, adata in adata_dict.items():
        if accepts_key:
            func(adata, key=key, **kwargs)
        else:
            func(adata, **kwargs)


def adata_dict_fapply_return(adata_dict, func, **kwargs):
    """
    Applies a given function to each AnnData object in the adata_dict and returns the results in a dictionary.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - func: Function to apply to each AnnData object in the dictionary.
    - kwargs: Additional keyword arguments to pass to the function.

    Returns:
    - dict: A dictionary with the same keys as adata_dict, containing the results of the function applied to each AnnData object.
    """
    sig = inspect.signature(func)
    accepts_key = 'key' in sig.parameters

    results = {}
    for key, adata in adata_dict.items():
        if accepts_key:
            results[key] = func(adata, key=key, **kwargs)
        else:
            results[key] = func(adata, **kwargs)
    return results

def check_and_create_strata(adata, strata_keys):
    """
    Checks if the specified stratifying variables are present in the AnnData object,
    and creates a new column combining these variables if it does not already exist.

    Parameters:
    adata : (AnnData) An AnnData object.
    strata_keys : (list of str) List of keys (column names) in adata.obs to be used for stratification.

    Returns:
    str: (str) The name of the newly created or verified existing combined strata column.

    Raises:
    ValueError: If one or more of the specified stratifying variables do not exist in adata.obs.
    """
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

def build_adata_dict(adata, strata_keys, desired_strata=None):
    """
    Build a dictionary of AnnData objects split by desired strata values.

    Parameters:
    adata (AnnData): Annotated data matrix.
    strata_keys (list of str): List of column names in `adata.obs` to use for stratification.
    desired_strata (list or dict, optional): List of desired strata values or a dictionary where keys are strata keys and values are lists of desired strata values. If None (Default), all combinations of categories in adata.obs[strata_keys] will be used.
    Returns:
    dict: Dictionary where keys are strata values and values are corresponding AnnData subsets.

    Raises:
    ValueError: If `desired_strata` is neither a list nor a dictionary of lists.
    """
    if desired_strata is None:
        # Generate all combinations of categories in adata.obs[strata_keys]
        all_categories = [adata.obs[key].cat.categories.tolist() for key in strata_keys]
        all_combinations = itertools.product(*all_categories)
        desired_strata = ['_'.join(map(str, comb)) for comb in all_combinations]
        return build_adata_dict_main(adata, strata_keys, desired_strata, print_missing_strata=False)
    
    elif isinstance(desired_strata, list):
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

def build_adata_dict_main(adata, strata_keys, desired_strata, print_missing_strata=True):
    """
    Main function to build a dictionary of AnnData objects based on desired strata values.

    Parameters:
    adata (AnnData): Annotated data matrix.
    strata_keys (list of str): List of column names in `adata.obs` to use for stratification.
    desired_strata (list of str): List of desired strata values.

    Returns:
    dict: Dictionary where keys are strata values and values are corresponding AnnData subsets.
    """
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
            if print_missing_strata:
                print(f"Warning: '{stratum}' is not a valid category in '{strata_key}'.")
    return AdataDict(subsets_dict)

def subsplit_adata_dict(adata_dict, strata_keys, desired_strata):
    """
    Split each value of an AnnData dictionary into further subsets based on additional desired strata.

    Parameters:
    adata_dict (dict): Dictionary where keys are strata values and values are AnnData objects.
    strata_keys (list of str): List of column names in `adata.obs` to use for further stratification.
    desired_strata (list or dict): List of desired strata values or a dictionary where keys are strata keys and values are lists of desired strata values.

    Returns:
    dict: Nested dictionary of AnnData objects split by the additional desired strata.
    """
    #this function takes an adata_dict and splits each value of the dictionary (an anndata) into a dictionary of anndatas
    #Would be correct to call this function: build_adata_dict_from_adata_dict()
    return adata_dict_fapply_return(adata_dict, build_adata_dict, strata_keys=strata_keys, desired_strata=desired_strata)


def concatenate_adata_dict(adata_dict, **kwargs):
    """
    Concatenates all AnnData objects in adata_dict into a single AnnData object. join defualts to "outer" and index_unique defaults to "-".

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments, including 'join' which specifies the join method for concatenation.

    Returns:
    - AnnData: A single AnnData object that combines all the subsets in adata_dict.
    """
    kwargs.setdefault('join', 'outer')
    kwargs.setdefault('index_unique', '-')
    
    adatas = list(adata_dict.values())
    if adatas:
        return sc.concat(adatas, **kwargs)
    else:
        raise ValueError("adata_dict is empty. No data available to concatenate.")


def summarize_metadata_adata_dict(adata_dict, **kwargs):
    """
    Generate summary tables for each AnnData object in the dictionary using the summarize_metadata function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments, including 'columns' which specifies a list of columns from the metadata to summarize. Use '*' to specify joint frequencies of multiple columns.

    Returns:
    - dict: A dictionary of summary dictionaries for each AnnData object in the adata_dict.
    """
    return adata_dict_fapply_return(adata_dict, summarize_metadata, **kwargs)

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


def subsample_adata_dict(adata_dict, **kwargs):
    """
    Subsamples each AnnData object in the dictionary using Scanpy's subsample function.
    
    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the subsample function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    n_obs = kwargs.get('n_obs', None)
    fraction = kwargs.get('fraction', None)

    if n_obs is None and fraction is None:
        fraction = 1
        kwargs['fraction'] = fraction

    def subsample_adata(adata, **kwargs):
        if n_obs is None or adata.n_obs > n_obs:
            sc.pp.subsample(adata, **kwargs)

    adata_dict_fapply(adata_dict, subsample_adata, **kwargs)

def resample_adata(adata, strata_keys, min_num_cells, n_largest_groups=None, **kwargs):
    """
    Resample an AnnData object based on specified strata keys and drop strata with fewer than the minimum number of cells.

    Parameters:
    adata (AnnData): Annotated data matrix.
    strata_keys (list of str): List of column names in adata.obs to use for stratification.
    min_num_cells (int): Minimum number of cells required to retain a stratum.
    kwargs: Additional keyword arguments to pass to the subsample function.

    Returns:
    AnnData: Concatenated AnnData object after resampling and filtering.

    Raises:
    ValueError: If any of the specified strata_keys do not exist in adata.obs.
    """
    # Step 1: Create the strata key
    strata_key = check_and_create_strata(adata, strata_keys)

    # Step 2: Calculate the size of each category
    category_counts = adata.obs[strata_key].value_counts()
    
    # Step 3: Identify the top n largest categories or all categories if n is None
    if n_largest_groups is None:
        selected_categories = category_counts.index.tolist()
    else:
        selected_categories = category_counts.nlargest(n_largest_groups).index.tolist()
    
    # Step 4: Build adata_dict based on the strata key
    strata_dict = build_adata_dict(adata, strata_key, selected_categories)
    
    # Step 5: Subsample each AnnData object in the strata_dict
    subsample_adata_dict(strata_dict, **kwargs)
    
    # Step 6: Drop AnnData objects with fewer than min_num_cells
    filtered_dict = {k: v for k, v in strata_dict.items() if v.n_obs >= min_num_cells}
    
    # Step 7: Concatenate the filtered_dict back to a single AnnData object
    return concatenate_adata_dict(filtered_dict)

def resample_adata_dict(adata_dict, strata_keys, n_largest_groups=None, min_num_cells=0, **kwargs):
    """
    Resample each AnnData object in a dictionary based on specified strata keys and drop strata with fewer than the minimum number of cells.

    Parameters:
    adata_dict (dict): Dictionary where keys are strata values and values are AnnData objects.
    strata_keys (list of str): List of column names in adata.obs to use for stratification.
    min_num_cells (int, optional): Minimum number of cells required to retain a stratum. Default is 0.
    kwargs: Additional keyword arguments to pass to the resample function.

    Returns:
    dict: Dictionary of resampled AnnData objects after filtering.
    """
    return adata_dict_fapply_return(adata_dict, resample_adata, strata_keys=strata_keys, n_largest_groups=n_largest_groups, min_num_cells=min_num_cells, **kwargs)


def normalize_adata_dict(adata_dict, **kwargs):
    """
    Normalizes each AnnData object in the dictionary using Scanpy's normalize_total.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the normalize_total function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.normalize_total, **kwargs)

def log_transform_adata_dict(adata_dict, **kwargs):
    """
    Log-transforms each AnnData object in the dictionary using Scanpy's log1p.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the log1p function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.log1p, **kwargs)

def set_high_variance_genes(adata_dict, **kwargs):
    """
    Identifies high-variance genes in each AnnData object in the dictionary.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the highly_variable_genes function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.highly_variable_genes, **kwargs)

def scale_adata_dict(adata_dict, **kwargs):
    """
    Scales each AnnData object in the dictionary using Scanpy's scale function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the scale function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.scale, **kwargs)

def pca_adata_dict(adata_dict, **kwargs):
    """
    Performs PCA on each AnnData object in the dictionary using Scanpy's pca function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the pca function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.pca, **kwargs)

def calculate_umap_adata_dict(adata_dict, **kwargs):
    """
    Calculates UMAP embeddings for each subset in the adata_dict.

    Parameters:
    - adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    - kwargs: Additional keyword arguments, including 'use_rep' which specifies the key in .obsm where the representation matrix is stored.

    Returns:
    - dict: A dictionary with the same keys as adata_dict, but values now include UMAP coordinates.
    """
    def calculate_umap(adata, **kwargs):
        use_rep = kwargs.get('use_rep')
        if use_rep in adata.obsm:
            sc.pp.neighbors(adata, use_rep=use_rep)
            sc.tl.umap(adata)
        else:
            print(f"Representation '{use_rep}' not found in .obsm of adata.")
    adata_dict_fapply(adata_dict, calculate_umap, **kwargs)
    return adata_dict

def plot_umap_adata_dict(adata_dict, **kwargs):
    """
    Plots UMAP embeddings for each AnnData object in adata_dict, colored by a specified variable.

    Parameters:
    - adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    - kwargs: Additional keyword arguments, including 'color_by' which specifies a variable by which to color the UMAP plots, typically a column in .obs.

    Returns:
    - None: The function creates plots for the AnnData objects.
    """
    def plot_umap(adata, key=None, **kwargs):
        print(f"Plotting UMAP for key: {key}")
        color_by = kwargs.get('color_by')
        if 'X_umap' in adata.obsm:
            title = [f"{color}" for color in color_by]
            sc.pl.umap(adata, color=color_by, title=title)
        else:
            print(f"UMAP not computed for adata with key {key}. Please compute UMAP before plotting.")
    adata_dict_fapply(adata_dict, plot_umap, **kwargs)


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


def stable_label_adata_dict(adata_dict, feature_key, label_key, classifier_class, max_iterations=100, stability_threshold=0.05, moving_average_length=3, random_state=None, **kwargs):
    """
    Trains a classifier for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): Dictionary with keys as identifiers and values as AnnData objects.
    feature_key (str): Key to access the features in adata.obsm.
    label_key (str): Key to access the labels in adata.obs.
    classifier: Classifier instance that implements fit and predict_proba methods.
    max_iterations, stability_threshold, moving_average_length, random_state: Additional parameters for training.
    kwargs: Additional keyword arguments to pass to the classifier constructor.

    Returns:
    results: Dict, keys are the identifiers from adata_dict and values are dictionaries containing the outputs from stable_label_adata.
    """
    stable_label_results = {}
    for stratum, adata in adata_dict.items():
        print(f"Training classifier for {stratum}")

        #create a classifier for this stratum
        classifier = classifier_class(random_state=random_state, **kwargs)


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
