#adata_dict functions
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
from sklearn.metrics import confusion_matrix
import harmonypy as hm

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
    plot_confusion_matrix,
    plot_sankey,
    save_sankey,
    plot_grouped_average,
    harmony_label_transfer
)
from .utils import make_names, add_label_to_adata, create_color_map
from .ai import (
    attempt_ai_integration, 
    generate_file_key, 
    map_cell_type_labels_to_simplified_set, 
    map_gene_labels_to_simplified_set, 
    ai_biological_process, 
    ai_cell_type,
    ai_cell_types_by_comparison,
    ai_compare_cell_types_binary,
    ai_compare_cell_types_categorical,
    ai_resolution_interpretation,
    determine_sign_of_resolution_change
)


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


def adata_dict_fapply(adata_dict, func, **kwargs_dicts):
    """
    Applies a given function to each AnnData object in the adata_dict, with additional
    values from other dictionaries or single values. The other dictionaries should
    be passed as keyword arguments where the keys are the argument names that func takes.

    Parameters:
    - adata_dict: Dictionary of AnnData objects with keys as identifiers.
    - func: Function to apply to each AnnData object in the dictionary.
    - kwargs_dicts: Additional keyword arguments to pass to the function, where each argument can be a dictionary with keys matching the adata_dict or a single value.

    Returns:
    - None: The function modifies the AnnData objects in place.
    """
    import inspect
    sig = inspect.signature(func)
    accepts_key = 'key' in sig.parameters

    for key, adata in adata_dict.items():
        func_args = {}
        for arg_name, arg_value in kwargs_dicts.items():
            if isinstance(arg_value, dict):
                if key in arg_value:
                    func_args[arg_name] = arg_value[key]
            else:
                func_args[arg_name] = arg_value
        
        if accepts_key:
            func(adata, key=key, **func_args)
        else:
            func(adata, **func_args)


def adata_dict_fapply_return(adata_dict, func, **kwargs_dicts):
    """
    Applies a given function to each AnnData object in the adata_dict, with additional
    values from other dictionaries or single values. The other dictionaries should
    be passed as keyword arguments where the keys are the argument names that func takes.

    Parameters:
    - adata_dict: Dictionary of AnnData objects with keys as identifiers.
    - func: Function to apply to each AnnData object in the dictionary.
    - kwargs_dicts: Additional keyword arguments to pass to the function, where each argument can be a dictionary with keys matching the adata_dict or a single value.

    Returns:
    - dict: A dictionary with the same keys as adata_dict, containing the results of the function applied to each AnnData object.
    """
    import inspect
    sig = inspect.signature(func)
    accepts_key = 'key' in sig.parameters

    results = {}
    for key, adata in adata_dict.items():
        func_args = {}
        for arg_name, arg_value in kwargs_dicts.items():
            if isinstance(arg_value, dict):
                if key in arg_value:
                    func_args[arg_name] = arg_value[key]
            else:
                func_args[arg_name] = arg_value
        
        if accepts_key:
            results[key] = func(adata, key=key, **func_args)
        else:
            results[key] = func(adata, **func_args)
    return results
        

# def adata_dict_fapply(adata_dict, func, **kwargs):
#     """
#     Applies a given function to each AnnData object in the adata_dict.

#     Parameters:
#     - adata_dict: Dictionary of AnnData objects with keys as identifiers.
#     - func: Function to apply to each AnnData object in the dictionary.
#     - kwargs: Additional keyword arguments to pass to the function.

#     Returns:
#     - None: The function modifies the AnnData objects in place.
#     """
#     sig = inspect.signature(func)
#     accepts_key = 'key' in sig.parameters

#     for key, adata in adata_dict.items():
#         if accepts_key:
#             func(adata, key=key, **kwargs)
#         else:
#             func(adata, **kwargs)


# def adata_dict_fapply_return(adata_dict, func, **kwargs):
#     """
#     Applies a given function to each AnnData object in the adata_dict and returns the results in a dictionary.

#     Parameters:
#     - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
#     - func: Function to apply to each AnnData object in the dictionary.
#     - kwargs: Additional keyword arguments to pass to the function.

#     Returns:
#     - dict: A dictionary with the same keys as adata_dict, containing the results of the function applied to each AnnData object.
#     """
#     sig = inspect.signature(func)
#     accepts_key = 'key' in sig.parameters

#     results = {}
#     for key, adata in adata_dict.items():
#         if accepts_key:
#             results[key] = func(adata, key=key, **kwargs)
#         else:
#             results[key] = func(adata, **kwargs)
#     return results


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


def read_adata_dict(paths):
    """
    Reads .h5ad files from a list of paths and returns them in a dictionary.

    For each element in the provided list of paths, if the element is a directory,
    it reads all .h5ad files in that directory. If the element is an .h5ad file,
    it reads the file directly. The results are returned as a dictionary with
    keys generated by generate_file_key() or fallback to "ad1", "ad2", ... if 
    generate_file_key() fails.

    Parameters:
    paths (list): A list of paths to directories or .h5ad files.

    Returns:
    dict: A dictionary with keys generated by generate_file_key() or fallback keys as "ad1", "ad2", ... and values as AnnData objects.
    """
    adata_dict = {}
    count = 1
    
    for path in paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith(".h5ad"):
                    file_path = os.path.join(path, file)
                    key = attempt_ai_integration(generate_file_key, lambda: f"ad{count}", file_path)
                    adata_dict[key] = ad.read_h5ad(file_path)
                    count += 1
        elif path.endswith(".h5ad"):
            key = attempt_ai_integration(generate_file_key, lambda: f"ad{count}", path)
            adata_dict[key] = ad.read_h5ad(path)
            count += 1
    
    return adata_dict


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

def set_var_index(adata_dict, column):
    """
    Set the index of adata.var to the specified column for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary where keys are identifiers and values are AnnData objects.
    column (str): The column name to set as the index of adata.var.

    Returns:
    dict: A dictionary with the same structure as adata_dict, where the var attribute of each AnnData object has its index set to the specified column.
    """
    def set_var_index_main(adata, column):
        adata.var = adata.var.set_index(column)
        return adata
    return adata_dict_fapply_return(adata_dict, set_var_index_main, column=column)


def set_obs_index(adata_dict, column):
    """
    Set the index of adata.obs to the specified column for each AnnData object in adata_dict.

    Parameters:
    adata_dict (dict): A dictionary where keys are identifiers and values are AnnData objects.
    column (str): The column name to set as the index of adata.obs.

    Returns:
    dict: A dictionary with the same structure as adata_dict, where the obs attribute of each AnnData object has its index set to the specified column.
    """
    def set_obs_index_main(adata, column):
        adata.obs = adata.obs.set_index(column)
        return adata

    return adata_dict_fapply_return(adata_dict, set_obs_index_main, column=column)


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
    strata_dict = build_adata_dict(adata, [strata_key], selected_categories)
    
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


def neighbors_adata_dict(adata_dict, **kwargs):
    """
    Calculates neighborhood graph for each AnnData object in the dictionary using Scanpy's neighbors function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the sc.pp.neighbors function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.pp.neighbors, **kwargs)


def leiden_adata_dict(adata_dict, **kwargs):
    """
    Performs Leiden clustering for each AnnData object in the dictionary using Scanpy's leiden function.

    Parameters:
    - adata_dict (dict): Dictionary of AnnData objects with keys as identifiers.
    - kwargs: Additional keyword arguments to pass to the sc.tl.leiden function.

    Returns:
    - None: The function modifies the input AnnData objects in place.
    """
    adata_dict_fapply(adata_dict, sc.tl.leiden, **kwargs)


def leiden_subcluster(adata, groupby, **kwargs):
    """
    Perform Leiden clustering on subgroups of cells.
    This function applies Leiden clustering to subgroups of cells defined by the groupby parameter.

    Parameters:
    adata : AnnData Annotated data matrix.
    groupby : str Column name in adata.obs for grouping cells before subclustering.
    kwargs : dict Additional keyword arguments to pass to the leiden_adata_dict function.

    Returns:
    None, The function modifies the input AnnData object in-place.
    """
    adata_dict = build_adata_dict(adata, strata_keys=[groupby])
    leiden_adata_dict(adata_dict, **kwargs)
    
    

def leiden_subcluster_adata_dict(adata_dict, groupby, **kwargs):
    """
    This function applies the leiden_subcluster function to each AnnData object
    in the provided dictionary.
    
    Parameters:
    adata_dict : dict Dictionary of AnnData objects.
    groupby : str Column name in adata.obs for grouping cells before subclustering.
    kwargs : dict Additional keyword arguments to pass to the leiden_subcluster function.

    Returns:
    None The function modifies the input AnnData objects in-place.
    """
    adata_dict_fapply(adata_dict, leiden_subcluster, groupby=groupby, **kwargs)


def calculate_umap_adata_dict(adata_dict, **kwargs):
    """
    Calculates UMAP embeddings for each subset in the adata_dict.

    Parameters:
    - adata_dict (dict): A dictionary with keys as strata and values as AnnData objects.
    - kwargs: Additional keyword arguments, including 'use_rep' which specifies the key in .obsm where the representation matrix is stored.

    Returns:
    - dict: A dictionary with the same keys as adata_dict, but values now include UMAP coordinates.
    """
    # def calculate_umap(adata, **kwargs):
    #     use_rep = kwargs.get('use_rep')
    #     if use_rep in adata.obsm:
    #         sc.pp.neighbors(adata, use_rep=use_rep)
    #         sc.tl.umap(adata)
    #     else:
    #         print(f"Representation '{use_rep}' not found in .obsm of adata.")
    # adata_dict_fapply(adata_dict, calculate_umap, **kwargs)
    adata_dict_fapply(adata_dict, sc.tl.umap, **kwargs)
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
                                     row_color_keys=None, col_color_keys=None, figsize=(10,10), diagonalize=False):
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
                                         row_color_keys=row_color_keys, col_color_keys=col_color_keys, figsize=figsize, diagonalize=diagonalize)


def harmony_label_transfer_adata_dict(adata_dict, master_data, master_subset_column='tissue', label_column='cell_type'):
    adata_dict_fapply(adata_dict, harmony_label_transfer, master_data=master_data, master_subset_column=master_subset_column, label_column=label_column)


#AI integrations
def ai_determine_leiden_resolution(adata, initial_resolution):
    """
    Adjusts the Leiden clustering resolution of an AnnData object based on AI feedback.

    Args:
        adata (AnnData): The annotated data matrix.
        initial_resolution (float): The initial resolution for Leiden clustering.

    Returns:
        float: The final resolution value after adjustments based on AI interpretation.

    This function iteratively performs Leiden clustering on the AnnData object,
    generates a UMAP plot, and uses an AI model to interpret the plot and suggest
    whether to increase, decrease, or maintain the current resolution. The resolution
    is adjusted by ±0.1 based on the AI's suggestion until no further adjustment is needed.
    """
    resolution = initial_resolution
    previous_sign_change = None
    
    k = 0
    while k <= 10:
        sc.tl.leiden(adata, resolution=resolution)
        
        # Plot UMAP colored by Leiden clusters
        def plot_umap():
            sc.pl.umap(adata, color='leiden', show=False)

        # Get AI interpretation
        annotation = ai_resolution_interpretation(plot_umap)
        
        # Determine the sign of resolution change
        sign_change = determine_sign_of_resolution_change(annotation)
        
        # Check if the resolution needs to be adjusted
        if sign_change == 0:
            return resolution
        elif previous_sign_change is not None and sign_change != previous_sign_change:
            resolution = round(resolution + 0.15 * sign_change, 2)
            return resolution
        else:
            resolution = round(resolution + 0.15 * sign_change, 2)
        
        previous_sign_change = sign_change
        k = k + 1

    return resolution


def ai_determine_leiden_resolution_adata_dict(adata_dict, initial_resolution=1):
    """
    Adjusts Leiden clustering resolution for each AnnData object in a dictionary based on AI feedback.

    Args:
        adata_dict (dict): Dictionary of AnnData objects.
        initial_resolution (float): Initial resolution for Leiden clustering (default is 1).

    Returns: dict: Dictionary with final resolution values after AI-based adjustments.
    """
    return adata_dict_fapply_return(adata_dict, ai_determine_leiden_resolution, initial_resolution=initial_resolution)


def simplify_obs_column(adata, column, new_column_name, simplification_level=''):
    """
    Simplifies labels in the specified column of the AnnData object and stores the result
    in a new column using the map_cell_type_labels_to_simplified_set().

    Args:
    adata (AnnData): The AnnData object containing the data.
    column (str): The column in adata.obs containing the cell type labels to simplify.
    new_column_name (str): The name of the new column to store the simplified labels.
    simplification_level (str, optional): Defaults to ''. A qualitative description of how much you want the labels to be simplified. Could be anything, like  'extremely', 'barely', or 'compartment-level'.

    """
    # Get the unique labels from the specified column
    unique_labels = adata.obs[column].unique()

    # Get the mapping of original labels to simplified labels using the provided function
    label_mapping = map_cell_type_labels_to_simplified_set(unique_labels, simplification_level=simplification_level)

    # Apply the mapping to create the new column in the AnnData object
    adata.obs[new_column_name] = adata.obs[column].map(label_mapping)

    return label_mapping


def simplify_obs_column_adata_dict(adata_dict, column, new_column_name, simplification_level=''):
    """
    Applies simplify_obs_column to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, simplify_obs_column, column=column, new_column_name=new_column_name, simplification_level=simplification_level)


def create_label_hierarchy(adata, col, simplification_levels):
    """
    Create a hierarchy of simplified labels based on a given column in AnnData.

    This function generates multiple levels of simplified labels from an original
    column in the AnnData object. Each level of simplification is created using
    the specified simplification levels.

    Parameters:
    adata : AnnData Annotated data matrix containing the column to be simplified.
    col : str Name of the column in adata.obs to be simplified.
    simplification_levels : list List of simplification levels to apply. Each level should be a value that can be used by the simplify_obs_column function.

    Returns:
    --------
    dict A dictionary mapping new column names to their corresponding simplified label mappings. The keys are the names of the new columns created for each simplification level, and the values are the mappings returned by simplify_obs_column for each level.
    """
    base_col_name = col
    simplified_mapping = {}
    for level in simplification_levels:
        new_col_name = f"{base_col_name}_{make_names([level])[0]}"
        simplified_mapping[new_col_name] = simplify_obs_column(adata, col, new_col_name, simplification_level=level)
        col = new_col_name
    return simplified_mapping


def create_label_hierarchy_adata_dict(adata_dict, col, simplification_levels):
    """
    Applies create_label_hierarchy to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, create_label_hierarchy, col = col, simplification_levels = simplification_levels)


def simplify_var_index(adata, column, new_column_name, simplification_level=''):
    """
    Simplifies gene names in the index of the AnnData object's var attribute based on a boolean column,
    and stores the result in a new column using the map_gene_labels_to_simplified_set().

    Args:
    adata (AnnData): The AnnData object containing the data.
    column (str): The boolean column in adata.var used to select genes for simplification.
    new_column_name (str): The name of the new column to store the simplified labels.
    simplification_level (str, optional): Defaults to ''. A qualitative description of how much you want the labels to be simplified. Could be anything, like 'extremely', 'barely', or 'compartment-level'.

    Raises:
    ValueError: If more than 1000 genes are selected for simplification or if the column is not boolean.
    """
    if not pd.api.types.is_bool_dtype(adata.var[column]):
        raise ValueError(f"The column '{column}' must be a boolean index column.")

    # Get the index of the true indices in the boolean column
    selected_genes = adata.var.index[adata.var[column]]

    if len(selected_genes) > 1000:
        raise ValueError("Cannot simplify more than 1000 genes at a time.")

    # Get the mapping of original labels to simplified labels using the provided function
    label_mapping = map_gene_labels_to_simplified_set(selected_genes, simplification_level=simplification_level)

    # Apply the mapping to create the new column in the AnnData object
    adata.var[new_column_name] = adata.var.index.to_series().map(label_mapping).fillna(adata.var.index.to_series())

    return label_mapping


def simplify_var_index_adata_dict(adata_dict, column, new_column_name, simplification_level=''):
    """
    Applies simplify_var_index to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, simplify_var_index, column=column, new_column_name=new_column_name, simplification_level=simplification_level)


def ai_annotate(func, adata, groupby, n_top_genes, label_column):
    """
    Annotate clusters based on the top marker genes for each cluster.

    This function performs differential expression analysis to identify marker genes for each cluster
    and applies func to determine the label for each cluster based on the top n
    marker genes. The results are added to the AnnData object and returned as a DataFrame.

    Parameters:
    adata : AnnData
    groupby : str Column in adata.obs to group by for differential expression analysis.
    n_top_genes : int The number of top marker genes to consider for each cluster.
    label_column : str The name of the new column in adata.obs where the annotations will be stored.

    Returns:
    pd.DataFrame A DataFrame with a column for the top marker genes for each cluster.
    """
    #run the differential expression analysis
    sc.tl.rank_genes_groups(adata, groupby, method='t-test')

    # Initialize a dictionary to store cell type annotations
    cell_type_annotations = {}

    # Initialize a list to store the results
    results = []

    # Get the rank genes groups result
    rank_genes_groups = adata.uns['rank_genes_groups']
    clusters = rank_genes_groups['names'].dtype.names  # List of clusters

    # Loop through each cluster and get the top n marker genes
    for cluster in clusters:
        top_genes = rank_genes_groups['names'][cluster][:n_top_genes]
        cell_type = func(top_genes)
        cell_type_annotations[cluster] = cell_type

        results.append({
            groupby: cluster,
            label_column: cell_type,
            f"top_{n_top_genes}_genes": list(top_genes)
        })

    # Create a new column in .obs for cell type annotations
    adata.obs[label_column] = adata.obs[groupby].map(cell_type_annotations)

    return pd.DataFrame(results)


def ai_annotate_cell_type(adata, groupby, n_top_genes, label_column='ai_cell_type'):
    """
    Annotate cell types based on the top marker genes for each cluster.

    This function performs differential expression analysis to identify marker genes for each cluster
    and applies ai_annotate_cell_type to determine the cell type for each cluster based on the top 
    marker genes. The results are added to the AnnData object and returned as a DataFrame.

    Parameters:
    adata : AnnData
    groupby : str Column in adata.obs to group by for differential expression analysis.
    n_top_genes : int The number of top marker genes to consider for each cluster.
    label_column : str, optional (default: 'ai_cell_type') The name of the new column in adata.obs where the cell type annotations will be stored.

    Returns:
    pd.DataFrame A DataFrame with a column for the top marker genes for each cluster.
    """
    return ai_annotate(func=ai_cell_type, adata=adata, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def ai_annotate_cell_type_adata_dict(adata_dict, groupby, n_top_genes=10, label_column='ai_cell_type'):
    """
    Applies ai_annotate_cell_type to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_type, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def ai_annotate_cell_type_by_comparison(adata, groupby, n_top_genes, label_column='ai_cell_sub_type'):
    """
    Annotate cell types by comparison using AI.

    This function wraps the ai_annotate function to perform cell type annotation
    based on comparison of gene expression profiles.

    Parameters:
    adata : AnnData Annotated data matrix.
    groupby : str Column name in adata.obs for grouping cells.
    n_top_genes : int Number of top genes to consider for annotation.
    label_column : str, optional Name of the column to store the AI-generated cell type labels (default: 'ai_cell_sub_type').

    Returns:
    AnnData Annotated data with AI-generated cell type labels.
    """
    return ai_annotate(func=ai_annotate_cell_type_by_comparison, adata=adata, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def ai_annotate_cell_type_by_comparison_adata_dict(adata_dict, groupby, n_top_genes=10, label_column='ai_cell_sub_type'):
    """
    Applies ai_annotate_cell_type_by_comparison to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_type_by_comparison, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def ai_annotate_cell_sub_type(adata, cell_type_col, sub_cluster_col, new_label_col):
    """
    Annotate cell subtypes using AI.

    This function performs AI-based annotation of cell subtypes by first grouping cells
    by their main cell type, then annotating subtypes within each group.

    Parameters:
    adata : AnnData Annotated data matrix.
    cell_type_col : str Column name in adata.obs containing main cell type labels.
    sub_cluster_col : str Column name in adata.obs containing sub-cluster information.
    new_label_col : str Name of the column to store the AI-generated subtype labels.

    Returns:
    --------
    tuple A tuple containing:
    AnnData: Concatenated annotated data with AI-generated subtype labels.
    dict: Mapping of original labels to AI-generated labels.
    """
    #build adata_dict based on cell_type_col
    adata_dict = build_adata_dict(adata, strata_keys=cell_type_col)

    label_mappings = ai_annotate_cell_type_by_comparison_adata_dict(adata_dict, groupby=sub_cluster_col, n_top_genes=10, label_column=new_label_col)

    adata = concatenate_adata_dict(adata_dict)

    return adata, label_mappings

def ai_annotate_cell_subtype_adata_dict(adata_dict, cell_type_col, new_label_col):
    """
    Annotate cell subtypes for a dictionary of AnnData objects.

    This function applies the ai_annotate_cell_sub_type function to each AnnData object
    in the provided dictionary.

    Parameters:
    adata_dict : dict Dictionary of AnnData objects.
    cell_type_col : str Column name in adata.obs containing main cell type labels.
    new_label_col : str Name of the column to store the AI-generated subtype labels.

    Returns:
    dict Dictionary of annotated AnnData objects with AI-generated subtype labels.
    """

    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_sub_type, cell_type_col=cell_type_col, new_label_col=new_label_col)

def ai_annotate_biological_process(adata, groupby, n_top_genes, label_column='ai_biological_process'):
    """
    Annotate biological processes based on the top n marker genes for each cluster.

    This function performs differential expression analysis to identify marker genes for each cluster
    and applies a user-defined function to determine the biological processes for each cluster based on the top 
    marker genes. The results are added to the AnnData object and returned as a DataFrame.

    Parameters:
    adata : AnnData
    groupby : str Column in adata.obs to group by for differential expression analysis.
    n_top_genes : int The number of top marker genes to consider for each cluster.
    label_column : str, optional (default: 'ai_cell_type') The name of the new column in adata.obs where the cell type annotations will be stored.

    Returns:
    pd.DataFrame A DataFrame with a column for the top marker genes for each cluster.
    """
    return ai_annotate(func=ai_biological_process, adata=adata, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def ai_annotate_biological_process_adata_dict(adata_dict, groupby, n_top_genes=10, label_column='ai_biological_process'):
    """
    Applies ai_annotate_biological_process to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_biological_process, groupby=groupby, n_top_genes=n_top_genes, label_column=label_column)


def ai_unify_labels(adata_dict, label_columns, new_label_column, simplification_level='unified, typo-fixed'):
    """
    Unifies cell type labels across multiple AnnData objects by mapping them to a simplified, unified set of labels.

    Parameters:
    adata_dict (dict): Dictionary where keys are identifiers and values are AnnData objects.
    label_columns (dict): Dictionary where keys should be the same as the keys of adata_dict and values are the column names in .obs containing the original labels.
    new_label_column (str): Name of the new column to be created in .obs for storing the harmonized labels.

    Returns:
    dict: A mapping dictionary where the keys are the original labels and the values are the unified labels.
    """
    #todo: use adata_dict_fapply instead of loops
    def get_unique_labels_from_obs_column(adata, label_column):
        return adata.obs[label_column].unique().tolist()

    def apply_mapping_to_adata(adata, mapping_dict, original_column, new_column, key=None):
        adata.obs[new_column] = adata.obs[original_column].map(mapping_dict)

    # Step 1: Aggregate all labels
    # aggregated_labels = adata_dict_fapply_return(adata_dict, get_unique_labels_from_obs_column, )
    aggregated_labels = []
    for key in adata_dict:
        labels = get_unique_labels_from_obs_column(adata_dict[key], label_columns[key])
        aggregated_labels.extend(labels)
    unique_labels_list = list(set(aggregated_labels))

    # Step 2: Get the mapping dictionary
    mapping_dict = map_cell_type_labels_to_simplified_set(unique_labels_list, simplification_level=simplification_level)

    # Step 3: Apply the mapping to each anndata in adata_dict
    for key in adata_dict:
        apply_mapping_to_adata(adata_dict[key], mapping_dict, label_columns[key], new_label_column)

    return mapping_dict


def create_label_df(adata, cols):
    """
    Create a DataFrame of unique rows from specified columns in adata.obs.
    
    Parameters:
    adata: AnnData object containing the data.
    cols: List of column names to extract and deduplicate.
    
    Returns:
    pd.DataFrame: DataFrame with unique rows based on the specified columns.
    """
    data = adata.obs[cols]
    unique_df = data.drop_duplicates().reset_index(drop=True)
    return unique_df


def ai_label_agreement(adata, cols, new_col_name, func):
    """
    Apply a function row-wise to the unique rows of specified columns in adata.obs.
    
    Parameters:
    adata: AnnData object The AnnData object containing the data.
    cols: list List of column names to extract and deduplicate.
    new_col_name: str Name of the new column to store the function results.
    func: function Function to apply to each row, which takes two string arguments.
    
    Returns:
    pd.DataFrame DataFrame with an additional column for the function results.
    """
    unique_df = create_label_df(adata, cols)
    unique_df[new_col_name] = unique_df.apply(lambda row: func(row[0], row[1]), axis=1)
    return unique_df

def ai_compare_cell_type_labels(adata, cols, new_col_name='agreement', comparison_level='binary'):
    """
    Compare cell type labels using a specified comparison function.
    
    Parameters:
    adata: AnnData object containing the data.
    cols: List of two column names to extract and compare.
    comparison_level: 'binary' or 'categorical', determines which comparison function to use.
    
    Returns:
    pd.DataFrame: DataFrame with an additional column for the comparison results.
    """
     # Check if cols is of length 2
    if len(cols) != 2:
        raise ValueError("The 'cols' parameter must contain exactly two elements. Simultaneous comparison between more than 2 columns may be added in the future.")
    
    # Check if comparison_level is valid
    if comparison_level not in ['binary', 'categorical']:
        raise ValueError("comparison_level must be either 'binary' or 'categorical'.")
    
    if comparison_level == 'binary':
        label_agreement = ai_label_agreement(adata, cols, new_col_name, ai_compare_cell_types_binary)
    elif comparison_level == 'categorical':
        label_agreement = ai_label_agreement(adata, cols, new_col_name, ai_compare_cell_types_categorical)

    # Merge 'agreement' column of label_agreement back to adata.obs
    adata.obs = adata.obs.merge(label_agreement, on=cols, how='left')

    return label_agreement

def ai_compare_cell_type_labels_adata_dict(adata_dict, cols, new_col_name='agreement', comparison_level='binary'):
    """
    Applies ai_compare_cell_type_labels to each anndata in an anndict.
    """
    return adata_dict_fapply_return(adata_dict, ai_compare_cell_type_labels, cols=cols, new_col_name=new_col_name, comparison_level=comparison_level)


def plot_sankey_adata_dict(adata_dict, cols, params=None):
    """
    Applies plot_sankey to each anndata in an anndict
    """
    return adata_dict_fapply_return(adata_dict, plot_sankey, cols=cols, params=params)


def save_sankey_adata_dict(plot_dict, filename):
    """
    Saves each sankey plot in a dictionary (i.e. the return value of plot_sankey_adata_dict)
    """
    adata_dict_fapply(plot_dict, save_sankey, filename=filename)

def plot_grouped_average_adata_dict(adata_dict, label_value):
    """
    plots the grouped average of a value for each group of a label. label_value must be a dictionary of dictionaries. For example, if adata_dict has two anndata with keys 'ad1' and 'ad2', then setting label_value = {'ad1':{'cell_type':'pct_counts_mt'}, 'ad2':{'cell_type':'pct_counts_mt'}} would plot the average of pct_counts_mt for each cell type in the anndata on separate plots for each anndata in adata_dict.
    """
    adata_dict_fapply(adata_dict, plot_grouped_average, label_value=label_value)