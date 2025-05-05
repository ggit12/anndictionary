"""
This module contains functions that operate on a single 
``adata`` to make multiple columns in the ``.obs`` 
have a shared set of categories.
"""

#the following functions also unify labels but serve a different purpose than ai_unify_labels.
#ai_unify_labels is meant to unify labels across multiple adata
#the following set of ensure_label functions are meant to operate within a single adata
#and do not communicate across multiple adata in a dict

from pandas import DataFrame
from anndata import AnnData

from anndict.utils import normalize_label
from anndict.automated_label_management.clean_single_column.in_adata_obs import (
    map_cell_type_labels_to_simplified_set
)

def ensure_label_consistency_adata(
    adata: AnnData,
    cols: str | list[str],
    simplification_level: str = 'unified, typo-fixed',
    new_col_prefix: str = 'consistent'
    ) -> dict:
    """
    Wrapper function to ensure label consistency 
    across specified columns in an :class:`AnnData` object.

    Parameters
    -----------
    adata
        An :class:`AnnData` object.

    cols
        :class:`List` of column names in ``adata.obs`` to ensure label consistency.

    simplification_level
        Qualitative direction about how to process the labels.

    new_col_prefix
        Prefix to use when creating new columns in ``adata.obs``. 
        Setting ``new_col_prefix = ""`` would overwrite the original columns.

    Returns
    --------
    :class:`Dict` mapping original labels to the unified set of labels.

    Notes
    -------
    Updates ``adata`` in-place with a shared set of labels 
    labels in ``adata.obs[new_col_prefix + cols]``.

    Useful when calculating inter-rater reliability. 
    To calculate measures of inter-rater reliability, see below:

    See Also
    ----------

    :func:`kappa_adata` : To calculate both Cohen's and 
        Fleiss's Kappa, a measure of inter-rater reliability.
    :func:`krippendorff_alpha_adata`: To calculate Krippendorff's 
        Alpha, a measure of inter-rater reliability.
    """
    # Step 0: make sure cols is a list
    if isinstance(cols, str):
        cols = [cols]

    # Step 1: Extract the relevant columns from adata.obs into a DataFrame
    df = adata.obs[cols].copy()

    # Step 2: Ensure label consistency using the helper function
    consistent_df, label_map = ensure_label_consistency_main(df, simplification_level)

    # Step 3: Create new columns in adata.obs with the prefix
    for col in cols:
        new_col_name = f"{new_col_prefix}_{col}"
        adata.obs[new_col_name] = consistent_df[col]

    return label_map


def ensure_label_consistency_main(df: DataFrame,
    simplification_level: str = 'unified, typo-fixed'
) -> tuple[DataFrame, dict]:
    """
    Function to ensure label consistency across multiple columns in a DataFrame
    by mapping labels to a unified and simplified set.

    Parameters
    -----------
    df
        a :class:`DataFrame` containing categorical columns across 
        which to unify category labels (so that all columns share the same set of labels).

    simplification_level
        Qualitative direction about how to process the labels.

    Returns
    --------
    class:`DataFrame` containing the columns in ``df``, now with labels shared across all columns
    class:`dict` containing the full mapping of original labels to the new, shared set of labels

    """
    # Step 1: Create a unified set of unique labels across all columns (before normalization)
    unique_labels = set()
    for column in df.columns:
        unique_labels.update(df[column].unique())

    # Normalize the unique labels with normalize_label(unique_labels)
    normalized_labels = {normalize_label(label) for label in unique_labels}

    # Create a dict called normalization_mapping from original to normalized labels
    normalization_mapping = {label: normalize_label(label) for label in unique_labels}

    # Step 2: Normalize all labels in the DataFrame using the normalization_mapping
    for column in df.columns:
        df[column] = df[column].map(normalization_mapping)

    # Step 3: Use the external function to map normalized labels to a simplified set
    normalized_labels_list = list(normalized_labels)
    mapping_dict = map_cell_type_labels_to_simplified_set(
        normalized_labels_list, simplification_level=simplification_level)

    # Step 4: Apply the mapping dictionary to all columns
    for column in df.columns:
        df[column] = df[column].map(mapping_dict)

    # Use normalization_mapping and mapping_dict to create one dict that goes from
    # keys of normalization_mapping (original labels) to values of mapping_dict (simplified labels)
    label_map = {original_label: mapping_dict[normalized_label]
                     for original_label, normalized_label in normalization_mapping.items()}

    return df, label_map
