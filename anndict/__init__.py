from . import metadata_summary
from . import utils
from . import ai
from . import stablelabel
from . import dict
from . import strata
from . import spatial_dict

from .metadata_summary import (
    summarize_metadata,
    display_html_summary
)

from .utils import (
    make_names, 
    add_col_to_adata_obs,
    add_label_to_adata,
    create_color_map,
    UCE_adata
)

from .ai import (
    configure_llm_backend, 
    get_llm_config, 
    get_client, 
    get_anthropic_client, 
    call_llm, 
    call_openai_llm, 
    call_anthropic_llm, 
    convert_messages_to_anthropic_prompt, 
    ai_cell_types_by_comparison, 
    get_openai_client, 
    enforce_semantic_list, 
    extract_dictionary_from_ai_string, 
    attempt_ai_integration, 
    generate_file_key, 
    map_cell_type_labels_to_simplified_set, 
    map_gene_labels_to_simplified_set, 
    ai_biological_process, 
    ai_cell_type,
    ai_compare_cell_types_binary,
    ai_compare_cell_types_categorical,
    encode_plot_for_openai, 
    ai_resolution_interpretation,
    determine_sign_of_resolution_change
)

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

from .dict import (
    AdataDict,
    adata_dict_fapply, 
    adata_dict_fapply_return,
    check_and_create_strata,
    read_adata_dict,
    build_adata_dict, 
    subsplit_adata_dict,  
    concatenate_adata_dict, 
    set_var_index,
    set_obs_index,
    remove_genes,
    remove_genes_adata_dict,
    resample_adata, 
    resample_adata_dict,
    normalize_adata_dict, 
    log_transform_adata_dict, 
    set_high_variance_genes, 
    scale_adata_dict, 
    pca_adata_dict, 
    neighbors_adata_dict,
    leiden_adata_dict,
    leiden_sub_cluster,
    leiden_sub_cluster_adata_dict,
    calculate_umap_adata_dict, 
    plot_umap_adata_dict, 
    write_h5ad_adata_dict, 
    stable_label_adata_dict, 
    predict_labels_adata_dict, 
    update_adata_labels_with_stable_label_results_dict, 
    update_adata_labels_with_predictions_dict, 
    update_adata_dict_with_label_dict, 
    plot_changes_adata_dict, 
    plot_confusion_matrix_adata_dict, 
    summarize_metadata_adata_dict,
    display_html_summary_adata_dict,
    subsample_adata_dict,
    harmony_label_transfer_adata_dict,
    simplify_obs_column, 
    simplify_obs_column_adata_dict, 
    simplify_var_index, 
    simplify_var_index_adata_dict, 
    ai_determine_leiden_resolution, 
    ai_determine_leiden_resolution_adata_dict, 
    ai_annotate, 
    ai_annotate_by_comparison, 
    ai_annotate_cell_type, 
    ai_annotate_cell_type_adata_dict, 
    ai_annotate_cell_type_by_comparison,
    ai_annotate_cell_type_by_comparison_adata_dict,
    ai_annotate_cell_sub_type,
    ai_annotate_cell_sub_type_adata_dict,
    ai_annotate_biological_process, 
    ai_annotate_biological_process_adata_dict, 
    ai_unify_labels, 
    create_label_df, 
    ai_label_agreement, 
    ai_compare_cell_type_labels, 
    ai_compare_cell_type_labels_adata_dict, 
    create_label_hierarchy, 
    create_label_hierarchy_adata_dict, 
    plot_sankey_adata_dict, 
    save_sankey_adata_dict, 
    plot_grouped_average_adata_dict
)

from .strata import (
    preprocess_adata_strata,
    stable_label_adata_strata,
    plot_confusion_matrix_across_strata
)

from .spatial_dict import (
    read_data,
    get_steps_and_coords,
    populate_sparse_array,
    process_gene_counts,
    create_anndata,
    add_blank_image_to_adata,
    build_adata_from_transcript_positions,
    build_adata_from_visium,
    plot_spatial_adata_dict,
    compute_spatial_neighbors_adata_dict,
    perform_colocalization_adata_dict,
    plot_colocalization_adata_dict,
    compute_interaction_matrix_adata_dict,
    plot_interaction_matrix_adata_dict
)

__all__ = [
    'get_slurm_cores', 
    'pca_density_filter', 
    'pca_density_wrapper', 
    'pca_density_adata_dict', 
    'stable_label', 
    'stable_label_adata', 
    'update_adata_labels_with_results', 
    'plot_training_history', 
    'plot_changes', 
    'plot_confusion_matrix_from_adata', 
    'plot_confusion_matrix', 
    'check_and_create_strata', 
    'read_adata_dict', 
    'build_adata_dict', 
    'subsplit_adata_dict', 
    'concatenate_adata_dict',
    'AdataDict', 
    'adata_dict_fapply',
    'adata_dict_fapply_return', 
    'resample_adata',
    'resample_adata_dict',
    'normalize_adata_dict', 
    'log_transform_adata_dict', 
    'set_high_variance_genes', 
    'scale_adata_dict', 
    'pca_adata_dict', 
    'neighbors_adata_dict', 
    'leiden_adata_dict', 
    'calculate_umap_adata_dict', 
    'plot_umap_adata_dict', 
    'write_h5ad_adata_dict', 
    'stable_label_adata_dict', 
    'predict_labels_adata_dict', 
    'update_adata_labels_with_stable_label_results_dict', 
    'update_adata_labels_with_predictions_dict', 
    'update_adata_dict_with_label_dict', 
    'plot_changes_adata_dict', 
    'plot_confusion_matrix_adata_dict', 
    'plot_spatial_adata_dict',
    'summarize_metadata_adata_dict',
    'display_html_summary_adata_dict',
    'subsample_adata_dict',
    'summarize_metadata',
    'display_html_summary',
    'add_col_to_adata_obs',
    'add_label_to_adata',
    'create_color_map',
    'UCE_adata',
    'preprocess_adata_strata',
    'stable_label_adata_strata',
    'plot_confusion_matrix_across_strata',
    'read_data',
    'get_steps_and_coords',
    'populate_sparse_array',
    'process_gene_counts',
    'create_anndata',
    'add_blank_image_to_adata',
    'build_adata_from_transcript_positions',
    'build_adata_from_visium',
    'compute_spatial_neighbors_adata_dict',
    'perform_colocalization_adata_dict',
    'plot_colocalization_adata_dict',
    'compute_interaction_matrix_adata_dict',
    'plot_interaction_matrix_adata_dict',
    'harmony_label_transfer', 
    'harmony_label_transfer_adata_dict',
    'get_openai_client', 
    'attempt_ai_integration', 
    'generate_file_key', 
    'map_cell_type_labels_to_simplified_set', 
    'map_gene_labels_to_simplified_set', 
    'ai_biological_process', 
    'ai_cell_type',
    'simplify_obs_column', 
    'simplify_obs_column_adata_dict', 
    'simplify_var_index', 
    'simplify_var_index_adata_dict', 
    'ai_determine_leiden_resolution', 
    'ai_determine_leiden_resolution_adata_dict', 
    'ai_annotate', 
    'ai_annotate_by_comparison',
    'ai_annotate_cell_type', 
    'ai_annotate_cell_type_adata_dict', 
    'ai_annotate_biological_process', 
    'ai_annotate_biological_process_adata_dict', 
    'ai_unify_labels',
    'set_var_index',
    'set_obs_index', 
    'encode_plot_for_openai', 
    'ai_resolution_interpretation', 
    'determine_sign_of_resolution_change', 
    'plot_sankey', 
    'plot_sankey_adata_dict',
    'create_label_hierarchy',
    'create_label_hierarchy_adata_dict',
    'make_names',
    'save_sankey',
    'save_sankey_adata_dict',
    'ai_compare_cell_types_binary',
    'ai_compare_cell_types_categorical'
    'create_label_df', 
    'ai_label_agreement', 
    'ai_compare_cell_type_labels', 
    'ai_compare_cell_type_labels_adata_dict', 
    'plot_grouped_average', 
    'plot_grouped_average_adata_dict',
    'enforce_semantic_list', 
    'extract_dictionary_from_ai_string',
    'leiden_sub_cluster',
    'leiden_sub_cluster_adata_dict',
    'configure_llm_backend', 
    'get_llm_config', 
    'get_client', 
    'get_anthropic_client', 
    'call_llm', 
    'call_openai_llm', 
    'call_anthropic_llm', 
    'convert_messages_to_anthropic_prompt', 
    'ai_cell_types_by_comparison', 
    'ai_annotate_cell_type_by_comparison',
    'ai_annotate_cell_type_by_comparison_adata_dict',
    'ai_annotate_cell_sub_type',
    'ai_annotate_cell_sub_type_adata_dict',
    'remove_genes',
    'remove_genes_adata_dict'
]