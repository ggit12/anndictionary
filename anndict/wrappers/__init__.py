"""
This module contains :func:`adata_dict_fapply` and :func:`adata_dict_fapply_return` wrappers for common packages and functions.
"""

from .scanpy_ import (
    sample_and_drop_adata_dict,
    normalize_adata_dict,
    log_transform_adata_dict,
    set_high_variance_genes_adata_dict,
    rank_genes_groups_adata_dict,
    scale_adata_dict,
    pca_adata_dict,
    neighbors_adata_dict,
    leiden_adata_dict,
    leiden_sub_cluster,
    leiden_sub_cluster_adata_dict,
    calculate_umap_adata_dict,
    plot_umap_adata_dict,
    plot_spatial_adata_dict

)

# from .squidpy_ import(
#     compute_spatial_neighbors_adata_dict,
#     perform_colocalization_adata_dict,
#     plot_colocalization_adata_dict,
#     compute_interaction_matrix_adata_dict,
#     plot_interaction_matrix_adata_dict,

# )



from .anndictionary_ import (
    remove_genes_adata_dict,
    ai_annotate_biological_process_adata_dict,
    simplify_var_index_adata_dict,
    ensure_label_consistency_adata_dict,
    simplify_obs_column_adata_dict,
    create_label_hierarchy_adata_dict,
    ai_annotate_cell_type_adata_dict,
    ai_compare_cell_type_labels_pairwise_adata_dict,
    ai_annotate_cell_sub_type_adata_dict,
    ai_determine_leiden_resolution_adata_dict,
    harmony_label_transfer_adata_dict,
    plot_sankey_adata_dict,
    save_sankey_adata_dict,
    plot_grouped_average_adata_dict,
    plot_label_changes_adata_dict,
    plot_confusion_matrix_adata_dict,
    summarize_metadata_adata_dict,
    display_html_summary_adata_dict,
    pca_density_adata_dict,

)


__all__ = [

    # scanpy_
    "sample_and_drop_adata_dict",
    "normalize_adata_dict",
    "log_transform_adata_dict",
    "set_high_variance_genes_adata_dict",
    "rank_genes_groups_adata_dict",
    "scale_adata_dict",
    "pca_adata_dict",
    "neighbors_adata_dict",
    "leiden_adata_dict",
    "leiden_sub_cluster",
    "leiden_sub_cluster_adata_dict",
    "calculate_umap_adata_dict",
    "plot_umap_adata_dict",
    "plot_spatial_adata_dict",

    # squidpy_
    # "compute_spatial_neighbors_adata_dict",
    # "perform_colocalization_adata_dict",
    # "plot_colocalization_adata_dict",
    # "compute_interaction_matrix_adata_dict",
    # "plot_interaction_matrix_adata_dict",

    # anndictionary_
    "remove_genes_adata_dict",
    "ai_annotate_biological_process_adata_dict",
    "simplify_var_index_adata_dict",
    "ensure_label_consistency_adata_dict",
    "simplify_obs_column_adata_dict",
    "create_label_hierarchy_adata_dict",
    "ai_annotate_cell_type_adata_dict",
    "ai_compare_cell_type_labels_pairwise_adata_dict",
    "ai_annotate_cell_sub_type_adata_dict",
    "ai_determine_leiden_resolution_adata_dict",
    "harmony_label_transfer_adata_dict",
    "plot_sankey_adata_dict",
    "save_sankey_adata_dict",
    "plot_grouped_average_adata_dict",
    "plot_label_changes_adata_dict",
    "plot_confusion_matrix_adata_dict",
    "summarize_metadata_adata_dict",
    "display_html_summary_adata_dict",
    "pca_density_adata_dict",
]
