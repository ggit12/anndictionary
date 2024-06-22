from . import metadata_summary
from . import utils
from . import stablelabel
from . import dict
from . import strata

from .metadata_summary import (
    summarize_metadata,
    display_html_summary
)

from .utils import (
    add_label_to_adata,
    create_color_map
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
    plot_confusion_matrix
)

from .dict import (
    adata_dict_fapply, 
    adata_dict_fapply_return,
    check_and_create_strata,
    build_adata_dict, 
    subsplit_adata_dict,  
    concatenate_adata_dict, 
    resample_adata, 
    resample_adata_dict,
    normalize_adata_dict, 
    log_transform_adata_dict, 
    set_high_variance_genes, 
    scale_adata_dict, 
    pca_adata_dict, 
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
    plot_spatial_adata_dict,
    summarize_metadata_adata_dict,
    display_html_summary_adata_dict,
    subsample_adata_dict
)


from .strata import (
    preprocess_adata_strata,
    stable_label_adata_strata,
    plot_confusion_matrix_across_strata
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
    'build_adata_dict', 
    'subsplit_adata_dict', 
    'concatenate_adata_dict', 
    'adata_dict_fapply',
    'adata_dict_fapply_return', 
    'resample_adata',
    'resample_adata_dict',
    'normalize_adata_dict', 
    'log_transform_adata_dict', 
    'set_high_variance_genes', 
    'scale_adata_dict', 
    'pca_adata_dict', 
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
    'add_label_to_adata',
    'create_color_map',
    'preprocess_adata_strata',
    'stable_label_adata_strata',
    'plot_confusion_matrix_across_strata'
]
