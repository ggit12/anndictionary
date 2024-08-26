#functions that aren't part of the stablelabel pipeline and operate on a single anndata (unstratified)
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

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import subprocess


def make_names(names):
    """
    Convert a list of names into valid and unique Python identifiers.

    Args:
        names (list of str): List of names to be transformed.

    Returns:
        list of str: Valid and unique Python identifiers.
    """
    # Equivalent of R's make.names() function in Python
    valid_names = []
    seen = {}
    for name in names:
        # Replace invalid characters with underscores
        clean_name = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        if clean_name in seen:
            seen[clean_name] += 1
            clean_name = f"{clean_name}.{seen[clean_name]}"
        else:
            seen[clean_name] = 0
        valid_names.append(clean_name)
    return valid_names


def add_label_to_adata(adata, indices, labels, new_label_key):
    """
    Adds a label to the AnnData object in a specified column for given indices.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - labels: Array of labels corresponding to the indices.
    - new_label_key: Name of the column in adata.obs where the labels will be stored.
    """
    add_col_to_adata_obs(adata, indices, labels, new_label_key)
    

def add_col_to_adata_obs(adata, indices, values, new_col_name):
    """
    Adds a label to the AnnData object in a specified column for given indices.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - values: Array of labels corresponding to the indices.
    - new_col_name: Name of the column in adata.obs where the labels will be stored.
    """
    if isinstance(values[0], (int, np.integer)):
        dtype = int
    elif isinstance(values[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.obs[new_col_name] = np.full(adata.obs.shape[0], np.nan, dtype=dtype)
    adata.obs.loc[indices, new_col_name] = values


def add_col_to_adata_var(adata, indices, values, new_col_name):
    """
    Adds a label to the AnnData object in a specified column for given indices in adata.var.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - values: Array of labels corresponding to the indices.
    - new_label_key: Name of the column in adata.var where the labels will be stored.
    """
    if isinstance(values[0], (int, np.integer)):
        dtype = int
    elif isinstance(values[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.var[new_col_name] = np.full(adata.var.shape[0], np.nan, dtype=dtype)
    adata.var.loc[indices, new_col_name] = values


def convert_obs_col_to_category(adata, col_name):
    """
    Convert a column in AnnData.obs to category dtype.
    
    Parameters:
    adata (AnnData): The AnnData object.
    col_name (str): The name of the column in adata.obs to convert.
    
    Returns:
    None: The function modifies the adata object in-place.
    """
    if col_name not in adata.obs.columns:
        raise ValueError(f"Column '{col_name}' not found in adata.obs")
    
    adata.obs[col_name] = adata.obs[col_name].astype('category')


def convert_obs_col_to_string(adata, col_name):
    """
    Convert a column in AnnData.obs to string dtype.
    
    Parameters:
    adata (AnnData): The AnnData object.
    col_name (str): The name of the column in adata.obs to convert.
    
    Returns:
    None: The function modifies the adata object in-place.
    """
    if col_name not in adata.obs.columns:
        raise ValueError(f"Column '{col_name}' not found in adata.obs")
    
    adata.obs[col_name] = adata.obs[col_name].astype(str)


def convert_obs_index_to_str(adata):
    """
    Converts the index of .obs to a string
    """
    adata.obs.index = adata.obs.index.astype(str)


def get_adata_columns(adata, col_startswith=None, col_endswith=None, col_contains=None, 
                      not_col_startswith=None, not_col_endswith=None, not_col_contains=None):
    columns = adata.obs.columns
    matched_columns = []

    if col_startswith:
        for start in col_startswith:
            matched_columns.extend([col for col in columns if col.startswith(start)])

    if col_endswith:
        for end in col_endswith:
            matched_columns.extend([col for col in columns if col.endswith(end)])

    if col_contains:
        for contain in col_contains:
            matched_columns.extend([col for col in columns if contain in col])

    if not_col_startswith:
        for start in not_col_startswith:
            matched_columns = [col for col in matched_columns if not col.startswith(start)]

    if not_col_endswith:
        for end in not_col_endswith:
            matched_columns = [col for col in matched_columns if not col.endswith(end)]

    if not_col_contains:
        for contain in not_col_contains:
            matched_columns = [col for col in matched_columns if contain not in col]

    return list(set(matched_columns))

def create_color_map(adata, keys):
    """
    Creates a unified color map for given keys from an AnnData object, differentiating
    between continuous and categorical data.

    Parameters:
    - adata: AnnData object.
    - keys: list of str, keys for which the color map is required.

    Returns:
    - dict: A color map linking unique values or ranges from the specified keys to colors.
    """
    color_map = {}
    for key in keys:
        if pd.api.types.is_numeric_dtype(adata.obs[key]):
            # Create a continuous colormap
            min_val, max_val = adata.obs[key].min(), adata.obs[key].max()
            norm = plt.Normalize(min_val, max_val)
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
            # Store the scalar map directly
            color_map[key] = scalar_map
        else:
            # Handle as categorical
            unique_values = pd.unique(adata.obs[key])
            color_palette = sns.color_palette("husl", n_colors=len(unique_values))
            color_palette_hex = [matplotlib.colors.rgb2hex(color) for color in color_palette]
            color_map[key] = dict(zip(unique_values, color_palette_hex))
    
    return color_map

def UCE_adata(adata_paths):
    """
    Runs the eval_single_anndata.py script for each specified anndata path. This function is included largely for illustrative purposes and will be restrictivley slow with a gpu.

    Parameters:
    adata_paths (list of str): A list of paths to the .h5ad files to be processed.

    This function constructs the command to run the eval_single_anndata.py script 
    with specified arguments and then executes the command using subprocess.run.
    
    The function assumes that a uce-compatible conda environment is already activated 
    and the working directory is correctly set to UCE (i.e. as in https://github.com/snap-stanford/UCE)


    Example usage:
    UCE_adata(["../dat/liver.h5ad", "../dat/kidney.h5ad"])
    """
    for adata_path in adata_paths:
        # Command to run the python script with the specified arguments
        command = [
            'accelerate', 'launch', 'eval_single_anndata.py',
            '--adata_path', adata_path,
            '--dir', './uce_wd/',
            '--species', 'human',
            '--model_loc', './model_files/33l_8ep_1024t_1280.torch',
            '--filter', 'True',
            '--batch_size', '25',
            '--nlayers', '33'
        ]

        # Run the command
        subprocess.run(command)

