"""
This module contains functions that operate across multiple adata, to make a sure that each adata has a .obs column with a set of categories that is shared between all the adata.
"""

from anndict.adata_dict import AdataDict
from anndict.automated_label_management.clean_single_column.in_adata_obs import map_cell_type_labels_to_simplified_set


def ai_unify_labels(
    adata_dict: AdataDict,
    label_columns: dict[tuple[str, ...], str],
    new_label_column: str,
    simplification_level: str = "unified, typo-fixed",
) -> dict:
    """
    Unifies cell type labels across multiple AnnData objects 
    by mapping them to a simplified, unified set of labels.

    Parameters
    ------------
    adata_dict
        An :class:`AdataDict`.

    label_columns
        :class:`dict` where keys should be the same as the keys of ``adata_dict`` and 
        values are the column names in the corresponding ``adata.obs`` containing the 
        original labels.

    new_label_column
        Name of the new column to be created in each ``adata.obs`` for storing the unified labels.
    
    simplification_level
        Instructions on how to unify the labels.

    Returns
    ---------
    A mapping :class:`dict` where the keys are the original labels and 
    the values are the unified labels.

    Notes
    -------
    Modifies each ``adata`` in ``adata_dict`` in-place by 
    adding ``adata.obs[new_label_column]`` with the unified label mapping.

    Example
    ---------

    .. code-block:: python

        #import package
        import anndict as adt

        #configure LLM backend
        configure_llm_backend('your-provider-name','your-provider-model-name',api_key='your-provider-api-key')
        
        #load the data as an AdataDict
        adata_paths = ['path/to/adata1.h5ad', 'path/to/adata2.h5ad']
        adata_dict = adt.read_adata_dict_from_h5ad(adata_paths)

        #define the label columns for each adata
        label_columns = {
            ('adata1',): 'cell_type',
            ('adata2',): 'cell_type_label', # this could be different in each adata
        }
        new_label_column = 'unified_cell_type'

        #unify the labels across the adata
        mapping_dict = adt.ai_unify_labels(
            adata_dict,
            label_columns=label_columns,
            new_label_column=new_label_column,
            simplification_level="unified, typo-fixed"
        )

        # Now each adata in adata_dict has a new column 'unified_cell_type'

        # Write the adata_dict to disk (an adata_dict on disk is just a directory containing .h5ad files)
        adt.write_adata_dict(adata_dict, 'path/to/unified_adata_dict/')

    """
    # TODO: port string normalization from ai_unify_labels to this function
    # TODO: use adata_dict_fapply instead of loops
    def get_unique_labels_from_obs_column(adata, label_column):
        return adata.obs[label_column].unique().tolist()

    def apply_mapping_to_adata(adata, mapping_dict, original_column, new_column, adt_key=None): # pylint: disable=unused-argument
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
