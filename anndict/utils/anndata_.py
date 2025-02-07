"""
This module contains and adata_dict wrappers for :mod:`anndata`.
Some of these ease manipulation of and information retrieval from :class:`AnnData`.
"""

import re

import numpy as np

from anndata import AnnData

from anndict.utils.anndictionary_ import enforce_semantic_list


def remove_genes(
    adata: AnnData,
    genes_to_remove: list[str],
    adt_key: tuple[str,...] | None = None
) -> None:
    """
    Remove specified genes from an :class:`AnnData` object in-place.

    Parameters
    -----------
    adata
        The :class:`AnnData` object to modify.

    genes_to_remove
        A list of gene names to remove.

    adt_key
        Used by :func:`adata_dict_fapply` and :func:`adata_dict_fapply_return` when passing this function.

    Returns
    --------
    ``None``

    Notes
    ------
    Modifies ``adata`` in-place.
    """
    # Get the list of genes to remove that are actually in the dataset
    genes_to_remove = adata.var_names.intersection(genes_to_remove)

    # Remove the specified genes
    # (the only way to do this in-place for now is to use the protected member of the class)
    adata._inplace_subset_var(~adata.var_names.isin(genes_to_remove)) # pylint: disable=protected-access

    print(f"Removed {len(genes_to_remove)} genes from {adt_key}. {adata.n_vars} genes remaining.")


def add_col_to_adata_obs(
    adata: AnnData,
    indices: list[int] | tuple[int, ...],
    values: list[int | float | str] | tuple[int | float | str, ...],
    new_col_name: str
) -> None:
    """
    Adds a column to ``adata.obs`` for given indices.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    indices
        Array of indices where labels will be assigned.

    values
        Array of labels corresponding to the indices.

    new_col_name
        Name of the new column to be created in ``adata.obs``.


    Returns
    --------
    ``None``

    Notes
    ------
    Modifies ``adata`` in-place.
    """
    if isinstance(values[0], (int, np.integer)):
        dtype = int
    elif isinstance(values[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.obs[new_col_name] = np.full(adata.obs.shape[0], np.nan, dtype=dtype)
    adata.obs.loc[indices, new_col_name] = values


def add_col_to_adata_var(
    adata: AnnData,
    indices: list[int] | tuple[int, ...],
    values: list[int | float | str] | tuple[int | float | str, ...],
    new_col_name: str
) -> None:
    """
    Adds a column to ``adata.var`` for given indices.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    indices
        Array of indices where labels will be assigned.

    values
        Array of labels corresponding to the indices.

    new_label_key
        Name of the column in adata.var where the labels will be stored.

    Returns
    --------
    ``None``

    Notes
    ------
    Modifies ``adata`` in-place.
    """
    if isinstance(values[0], (int, np.integer)):
        dtype = int
    elif isinstance(values[0], (float, np.floating)):
        dtype = float
    else:
        dtype = str

    adata.var[new_col_name] = np.full(adata.var.shape[0], np.nan, dtype=dtype)
    adata.var.loc[indices, new_col_name] = values


def convert_obs_col_to_category(
    adata: AnnData,
    cols: str | list[str]
) -> None:
    """
    Convert column(s) in ``adata.obs`` to ``category`` dtype.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    cols
        The column name(s) in ``adata.obs`` to convert.

    Returns
    --------
    ``None``

    Notes
    ------
    Modifies ``adata`` in-place.
    """
    if isinstance(cols, str):  # Allow single string input as well
        cols = [cols]

    missing_cols = [col for col in cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in adata.obs")

    for col in cols:
        adata.obs[col] = adata.obs[col].astype('category')


def convert_obs_col_to_string(
    adata: AnnData,
    cols: str | list[str] 
) -> None:
    """
    Convert column(s) in ``adata.obs`` to ``str`` dtype.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    cols
        The column name(s) in ``adata.obs`` to convert.

    Returns
    --------
    ``None``

    Notes
    ------
    Modifies ``adata`` in-place.
    """
    if isinstance(cols, str):  # Allow single string input as well
        cols = [cols]

    missing_cols = [col for col in cols if col not in adata.obs.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in adata.obs")

    for col in cols:
        adata.obs[col] = adata.obs[col].astype(str)


def convert_obs_index_to_str(
    adata: AnnData
) -> None:
    """
    Converts the index of ``adata.obs`` to ``str``.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    Returns
    --------
    ``None``

    Notes
    ------
    Modifies ``adata`` in-place.
    """
    adata.obs.index = adata.obs.index.astype(str)


def get_adata_columns(
    adata: AnnData,
    starts_with: str | list[str] | None = None,
    ends_with: str | list[str] | None = None,
    contains: str | list[str] | None = None, 
    not_starts_with: str | list[str] | None = None,
    not_ends_with: str | list[str] | None = None,
    not_contains: str | list[str] | None = None
) -> list[str]:
    """
    A simple string matching and filtering function to get column names from ``adata.obs``.
    Get the column names 

    Parameters
    ----------
    adata
        An AnnData object.

    starts_with
        Include columns that start with this.

    ends_with
        Include columns that end with this.

    contains
        Include columns that contain this.

    not_starts_with
        Exclude columns that start with this.

    not_ends_with
        Exclude columns that end with this.

    not_contains
        Exclude columns that contain this.

    Returns
    -------
    A list of column names from ``adata.obs`` that match the specified criteria.

    Examples
    --------
    .. code-block:: python

        import anndict as adt

        adata.obs.columns
        > Index(['cell_type', 'total_counts', 'total_counts_mt', 'total_counts_protein_coding',  
             'pct_counts_mt', 'pct_counts_protein_coding'], dtype='object')

        #get all total counts columns
        get_adata_columns(adata, starts_with='total_counts')
        > ['total_counts', 'total_counts_mt', 'total_counts_protein_coding']

        #get all mitochondrial columns
        get_adata_columns(adata, contains='mt')
        > ['total_counts_mt', 'pct_counts_mt']

        #get all columns that are not total counts
        get_adata_columns(adata, not_starts_with='total_counts')
        > ['cell_type', 'pct_counts_mt', 'pct_counts_protein_coding']
    """
    columns = adata.obs.columns
    matched_columns = []

    #Get local variables and enforce that the filtering parameters are lists
    filter_params = ['starts_with', 'ends_with', 'contains', 
                     'not_starts_with', 'not_ends_with', 'not_contains']

    for param in filter_params:
        val = locals()[param]
        if isinstance(val, str):
            locals()[param] = [val]

    if starts_with:
        for start in starts_with:
            matched_columns.extend([col for col in columns if col.startswith(start)])

    if ends_with:
        for end in ends_with:
            matched_columns.extend([col for col in columns if col.endswith(end)])

    if contains:
        for contain in contains:
            matched_columns.extend([col for col in columns if contain in col])

    if not_starts_with:
        for start in not_starts_with:
            matched_columns = [col for col in matched_columns if not col.startswith(start)]

    if not_ends_with:
        for end in not_ends_with:
            matched_columns = [col for col in matched_columns if not col.endswith(end)]

    if not_contains:
        for contain in not_contains:
            matched_columns = [col for col in matched_columns if contain not in col]

    return list(set(matched_columns))


def filter_gene_list(
    adata: AnnData,
    gene_list: list[str]
) -> list[str]:
    """
    Filter a list of genes based on their presence in ``adata.var.index``. Will also update list with suffixes if needed to match gene names in ``adata.var.index``

    Parameters
    -----------
        adata
            An :class:`AnnData`.

        gene_list
            List of gene names to be filtered and updated with possible unique suffixes.

    Returns
    --------
    Updated list of genes found in ``adata.var.index``, including suffix variations.
    """
    enforce_semantic_list(adata.var.index)
    updated_gene_list = []
    for gene in gene_list:
        # Create a regex pattern to match the gene name and its possible unique suffixes, case-insensitive
        pattern = re.compile(r'^' + re.escape(gene) + r'(-\d+)?$', re.IGNORECASE)

        # Find all matching genes in adata.var.index
        matching_genes = [g for g in adata.var.index if pattern.match(g)]

        if matching_genes:
            updated_gene_list.extend(matching_genes)
        # else:
            # print(f"Gene '{gene}' not found in adata.var.index after making unique.")

    # Remove any duplicates in the updated marker list
    updated_gene_list = list(set(updated_gene_list))
    return updated_gene_list