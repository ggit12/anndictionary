#metadata summary functions
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

def summarize_metadata(adata, columns):
    """
    Generate a summary for specified metadata columns in an anndata object as a dictionary.
    
    Parameters:
        adata (anndata.AnnData): The anndata object containing the data.
        columns (list of str): List of columns from the metadata. 
                               Use '*' to specify joint frequencies of multiple columns.
    
    Returns:
        dict: A dictionary with keys as column descriptions and values as a DataFrame of counts.
    """
    results = {}
    
    for col in columns:
        if '*' in col:
            # Handle joint frequencies
            sub_cols = col.split('*')
            combined_data = adata.obs[sub_cols]
            joint_freq = combined_data.groupby(sub_cols).size().unstack(fill_value=0)
            joint_freq = combined_data.groupby(sub_cols, observed=True).size().unstack(fill_value=0)
            results[col.replace('*', ' x ')] = joint_freq
        else:
            # Calculate frequency for a single column
            freq = adata.obs[col].value_counts().to_frame('count')
            results[col] = freq
    
    return results

def display_html_summary(summary_dict):
    """
    Display separate HTML tables for each metadata category in the summary dictionary,
    arranging up to three tables in a row before starting a new line.

    Parameters:
        summary_dict (dict): The dictionary containing frequency data for metadata columns.
    """
    html = '<div style="display: flex; flex-wrap: wrap;">'
    table_count = 0
    
    for category, data in summary_dict.items():
        if table_count % 3 == 0 and table_count != 0:
            html += '<div style="flex-basis: 100%; height: 20px;"></div>'
        
        table_html = f'<div style="flex: 1; padding: 10px;"><h3>{category}</h3>'
        # Start the table and add a header row
        table_html += '<table border="1"><tr><th></th>'  # Empty header for the row labels
        table_html += ''.join(f'<th>{col}</th>' for col in data.columns) + '</tr>'  # Column headers
        
        for index, row in data.iterrows():
            # Include row labels as the first column and the rest of the data in subsequent columns
            table_html += f'<tr><td>{index}</td>' + ''.join(f'<td>{val}</td>' for val in row) + '</tr>'
        
        table_html += '</table></div>'
        html += table_html
        table_count += 1
    
    html += '</div>'
    display(HTML(html))
