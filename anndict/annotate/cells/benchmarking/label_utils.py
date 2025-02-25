"""
This module contains utility functions for label comparison.
"""

import pandas as pd

from anndata import AnnData

def create_label_df(
    adata: AnnData,
    cols1: list[str],
    cols2: list[str]
) -> pd.DataFrame:
    """
    Creates a :class:`DataFrame` of unique label combinations from 
    the specified columns in ``cols1`` and ``cols2``, only including 
    combinations that exist in ``adata.obs[[*cols1, *cols2]]``.

    Parameters
    ----------
    adata
        An :class:`AnnData`.

    cols1
        List of columns to compare with cols2.

    cols2
        List of columns to compare with cols1.

    Returns
    -------
    :class:`DataFrame` containing unique combinations of the specified columns.
    """
    # Combine all columns
    all_cols = cols1 + cols2

    # Get unique combinations that exist in adata.obs
    unique_combinations = adata.obs[all_cols].drop_duplicates()

    # Melt the DataFrame to get all combinations in two columns
    melted_df = pd.melt(unique_combinations,
                        id_vars=cols1,
                        value_vars=cols2,
                        var_name='col2_name',
                        value_name='col2')

    # Melt again to get col1 in a single column
    result_df = pd.melt(melted_df,
                        id_vars=['col2_name', 'col2'],
                        value_vars=cols1,
                        var_name='col1_name',
                        value_name='col1')

    # Keep only the relevant columns and drop duplicates
    result_df = result_df[['col1', 'col2']].drop_duplicates()

    return result_df
