#functions that aren't part of the stablelabel pipeline and operate on a single anndata (unstratified)
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
    

def add_col_to_adata_obs(adata, indices, labels, new_label_key):
    """
    Adds a label to the AnnData object in a specified column for given indices.

    Parameters:
    - adata: AnnData object to be updated.
    - indices: Array of indices where labels will be assigned.
    - labels: Array of labels corresponding to the indices.
    - new_label_key: Name of the column in adata.obs where the labels will be stored.
    """
    if isinstance(labels[0], (int, np.integer)):
        dtype = int
    elif isinstance(labels[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.obs[new_label_key] = np.full(adata.obs.shape[0], np.nan, dtype=dtype)
    adata.obs.loc[indices, new_label_key] = labels


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

