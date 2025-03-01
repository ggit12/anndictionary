"""
This module contains adata_dict wrappers for functions in `anndict`.
"""

from typing import Type, Literal
from functools import wraps

from pandas import DataFrame
from sklearn.base import ClassifierMixin

from anndict.adata_dict import (
    AdataDict,
    adata_dict_fapply,
    adata_dict_fapply_return,
)

from anndict.annotate import (
    ai_annotate_biological_process,
    ai_annotate_cell_type,
    ai_compare_cell_type_labels_pairwise,
    ai_annotate_cell_sub_type,
    ai_determine_leiden_resolution,
    harmony_label_transfer,
    transfer_labels_using_classifier,
)
from anndict.plot import (
    plot_sankey,
    save_sankey,
    plot_grouped_average,
    plot_label_changes,
    plot_confusion_matrix_from_adata
)
from anndict.utils.anndictionary_ import (
    summarize_metadata,
    display_html_summary,

)
from anndict.utils.pca_density_filter import (
    pca_density_filter_adata,

)
from anndict.utils.anndata_ import (
    remove_genes,

)
from anndict.automated_label_management import (
    simplify_var_index,
    simplify_obs_column,
    create_label_hierarchy,
    ensure_label_consistency_adata,
)

@wraps(remove_genes)
def remove_genes_adata_dict(
    adata_dict: AdataDict,
    genes_to_remove: list[str]
) -> None:
    """ 
    Wrapper for remove_genes.
    """
    adata_dict_fapply(adata_dict, remove_genes, genes_to_remove=genes_to_remove)

@wraps(ai_annotate_biological_process)
def ai_annotate_biological_process_adata_dict(
    adata_dict: AdataDict,
    groupby: str,
    n_top_genes: int = 10,
    new_label_column: str = 'ai_biological_process'
) -> AdataDict:
    """
    Wrapper for ai_annotate_biological_process.
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_biological_process, max_retries=3, groupby=groupby, n_top_genes=n_top_genes, new_label_column=new_label_column)

@wraps(simplify_var_index)
def simplify_var_index_adata_dict(
    adata_dict: AdataDict,
    column: str,
    new_column_name: str,
    simplification_level: str = ''
) -> AdataDict:
    """
    Wrapper for simplify_var_index.
    """
    return adata_dict_fapply_return(adata_dict, simplify_var_index, max_retries=3, column=column, new_column_name=new_column_name, simplification_level=simplification_level)

@wraps(ensure_label_consistency_adata)
def ensure_label_consistency_adata_dict(
    adata_dict: AdataDict,
    cols: list[str],
    simplification_level: str = 'unified, typo-fixed',
    new_col_prefix: str = 'consistent'
) -> AdataDict:
    """
    Wrapper for ensure_label_consistency_adata.
    """
    return adata_dict_fapply_return(adata_dict, ensure_label_consistency_adata, cols=cols, simplification_level=simplification_level, new_col_prefix=new_col_prefix)

@wraps(simplify_obs_column)
def simplify_obs_column_adata_dict(
    adata_dict: AdataDict,
    column: str,
    new_column_name: str,
    simplification_level: str = ''
) -> dict[tuple[str,...], dict[str,str]]:
    """
    Wrapper for simplify_obs_column.
    """
    return adata_dict_fapply_return(adata_dict, simplify_obs_column, max_retries=3, column=column, new_column_name=new_column_name, simplification_level=simplification_level)

@wraps(create_label_hierarchy)
def create_label_hierarchy_adata_dict(
    adata_dict: AdataDict,
    col: str,
    simplification_levels: list[str]
) -> dict:
    """
    Wrapper for create_label_hierarchy.
    """
    return adata_dict_fapply_return(adata_dict, create_label_hierarchy, max_retries=3, col=col, simplification_levels=simplification_levels)

@wraps(ai_annotate_cell_type)
def ai_annotate_cell_type_adata_dict(
    adata_dict: AdataDict,
    groupby: str,
    n_top_genes: int = 10,
    new_label_column: str = 'ai_cell_type',
    tissue_of_origin_col: str | None = None
) -> dict[tuple[str,...], DataFrame]:
    """
    Wrapper for ai_annotate_cell_type.
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_type, max_retries=3, groupby=groupby, n_top_genes=n_top_genes, new_label_column=new_label_column, tissue_of_origin_col=tissue_of_origin_col)

@wraps(ai_compare_cell_type_labels_pairwise)
def ai_compare_cell_type_labels_pairwise_adata_dict(
    adata_dict: AdataDict,
    cols1: list[str],
    cols2: list[str],
    new_col_prefix: str = 'agreement',
    comparison_level: str = 'binary'
) -> dict[tuple[str, ...], DataFrame]:
    """
    Wrapper for ai_compare_cell_type_labels_pairwise.
    """
    return adata_dict_fapply_return(adata_dict, ai_compare_cell_type_labels_pairwise, max_retries=3, cols1=cols1, cols2=cols2, new_col_prefix=new_col_prefix, comparison_level=comparison_level)

@wraps(ai_annotate_cell_sub_type)
def ai_annotate_cell_sub_type_adata_dict(
    adata_dict,
    cell_type_col,
    sub_cluster_col,
    new_label_col,
    tissue_of_origin_col=None,
    n_top_genes=10
) -> tuple[AdataDict, dict[str,str]]:
    """
    Wrapper for ai_annotate_cell_sub_type.
    """
    results = adata_dict_fapply_return(adata_dict, ai_annotate_cell_sub_type, max_retries=3, cell_type_col=cell_type_col, sub_cluster_col=sub_cluster_col, new_label_col=new_label_col, tissue_of_origin_col=tissue_of_origin_col, n_top_genes=n_top_genes)
    annotated_adata_dict = {key: result[0] for key, result in results.items()}
    label_mappings_dict = {key: result[1] for key, result in results.items()}

    return annotated_adata_dict, label_mappings_dict

@wraps(ai_determine_leiden_resolution)
def ai_determine_leiden_resolution_adata_dict(
    adata_dict: AdataDict,
    initial_resolution: float = 1
) -> AdataDict:
    """
    Wrapper for ai_determine_leiden_resolution.
    """
    return adata_dict_fapply_return(adata_dict, ai_determine_leiden_resolution, max_retries=3, initial_resolution=initial_resolution)

@wraps(transfer_labels_using_classifier)
def transfer_labels_using_classifier_adata_dict(
    origin_adata_dict: AdataDict,
    destination_adata_dict: AdataDict,
    origin_label_key: str,
    feature_key: str | Literal["use_X"],
    classifier_class: Type[ClassifierMixin],
    new_column_name: str = 'predicted_label',
    random_state: int | None = None,
    **kwargs
) -> AdataDict:
    """
    Wrapper for transfer_labels_using_classifier.
    """
    return adata_dict_fapply_return(
        origin_adata_dict,
        transfer_labels_using_classifier,
        destination_adata=destination_adata_dict,
        origin_label_key=origin_label_key,
        feature_key=feature_key,
        classifier_class=classifier_class,
        new_column_name=new_column_name,
        random_state=random_state,
        **kwargs
    )

@wraps(harmony_label_transfer)
def harmony_label_transfer_adata_dict(adata_dict,
    master_data,
    master_subset_column='tissue',
    label_column='cell_type'):
    """
    Wrapper for harmony_label_transfer.
    """
    adata_dict_fapply(adata_dict, harmony_label_transfer, master_data=master_data, master_subset_column=master_subset_column, label_column=label_column)

@wraps(plot_sankey)
def plot_sankey_adata_dict(adata_dict, cols, params=None):
    """
    Wrapper for plot_sankey.
    """
    return adata_dict_fapply_return(adata_dict, plot_sankey, cols=cols, params=params)

@wraps(save_sankey)
def save_sankey_adata_dict(plot_dict, filename):
    """
    Wrapper for save_sankey.
    """
    adata_dict_fapply(plot_dict, save_sankey, filename=filename)

@wraps(plot_grouped_average)
def plot_grouped_average_adata_dict(adata_dict, label_value):
    """
    Wrapper for plot_grouped_average.
    """
    adata_dict_fapply(adata_dict, plot_grouped_average, label_value=label_value)

@wraps(plot_label_changes)
def plot_label_changes_adata_dict(adata_dict, true_label_key, predicted_label_key, percentage=True):
    """
    Wrapper for plot_label_changes.
    """
    adata_dict_fapply(adata_dict, plot_label_changes, true_label_key=true_label_key, predicted_label_key=predicted_label_key, percentage=percentage, use_multithreading=False)

@wraps(plot_confusion_matrix_from_adata)
def plot_confusion_matrix_adata_dict(adata_dict, true_label_key, predicted_label_key,
                                     row_color_keys=None, col_color_keys=None, figsize=(10,10), diagonalize=False):
    """
    Wrapper for plot_confusion_matrix_from_adata.
    """
    adata_dict_fapply(adata_dict, plot_confusion_matrix_from_adata, true_label_key=true_label_key, predicted_label_key=predicted_label_key, row_color_keys=row_color_keys, col_color_keys=col_color_keys, figsize=figsize, diagonalize=diagonalize, use_multithreading=False)

@wraps(summarize_metadata)
def summarize_metadata_adata_dict(adata_dict, **kwargs):
    """
    Wrapper for summarize_metadata.
    """
    return adata_dict_fapply_return(adata_dict, summarize_metadata, **kwargs)

@wraps(display_html_summary)
def display_html_summary_adata_dict(summary_dict, parent_key=None):
    """Wrapper for display_html_summary."""
    if parent_key is None:
        parent_key = ()

    # If we've reached a dictionary whose keys are all strings,
    # treat it as terminal and pass to display_html_summary.
    if all(isinstance(k, str) for k in summary_dict.keys()):
        if parent_key:
            print(f"Summary for: {'->'.join(map(str, parent_key))}")
        display_html_summary(summary_dict)
        return

    # Otherwise, recurse deeper for each sub-dict
    for key, value in summary_dict.items():
        new_parent_key = parent_key + (key,)
        if isinstance(value, dict):
            display_html_summary_adata_dict(value, parent_key=new_parent_key)
        else:
            # If this branch is reached, 'value' is terminal data
            # rather than a dict, so pass to display_html_summary.
            print(f"Summary for: {'->'.join(map(str, new_parent_key))}")
            display_html_summary(value)


@wraps(pca_density_filter_adata)
def pca_density_adata_dict(adata_dict, **kwargs):
    """
    Wrapper for pca_density_filter_adata.
    """
    return adata_dict_fapply_return(adata_dict, pca_density_filter_adata, **kwargs)
