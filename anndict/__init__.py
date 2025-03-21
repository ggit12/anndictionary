"""
Main init for anndictionary.
If on Mac, runs a multithreading configuration check before allowing import.
"""
__version__ = "0.3.1.4.2"
__author__ = "ggit12"

import os
import platform
import sys
import numba  # Import numba to interact with its threading layer

#import AnnDictionary namespace
from . import utils
from . import adata_dict
from . import llm
from . import wrappers
from . import automated_label_management

from .adata_dict import (
    #adata_dict.py
    AdataDict,
    to_nested_tuple,
    to_nested_list,

    #adata_dict_fapply.py
    adata_dict_fapply,
    adata_dict_fapply_return,

    # build.py
    build_adata_dict,

    # add_stratification.py
    add_stratification,

    # write.py
    write_adata_dict,

    #read.py
    read_adata_dict,
    read_adata_dict_from_h5ad,

    # concatenate.py
    concatenate_adata_dict,

    # utils.py
    check_and_create_stratifier,
)

from .llm import (
    #llm_call.py
    configure_llm_backend,
    get_llm_config,
    call_llm,
    retry_call_llm,

    #parse_llm_response.py
    extract_dictionary_from_ai_string,
    extract_list_from_ai_string,
    process_llm_category_mapping,

    #provider_initializer_mapping.py
    LLMProviders,

)

from .annotate import (

    # Genes
    #--------
    ai_make_cell_type_gene_list,

    # Cells
    #--------
    # De Novo
    ai_cell_type,
    ai_cell_types_by_comparison,
    ai_from_expected_cell_types,
    ai_biological_process,
    ai_annotate,
    ai_annotate_by_comparison,
    ai_annotate_cell_type,
    ai_annotate_from_expected_cell_types,
    ai_annotate_cell_type_by_comparison,
    ai_annotate_cell_sub_type,
    ai_annotate_biological_process,
    cell_type_marker_gene_score,
    ai_determine_leiden_resolution,

    # Label Transfer
    harmony_label_transfer,
    train_label_classifier,
    transfer_labels_using_classifier,

    # Benchmarking
    create_label_df,
    kappa_adata, krippendorff_alpha_adata,
    ai_compare_cell_type_labels_pairwise,

)

from .plot import (
    #module_score_plots.py
    module_score_barplot,
    module_score_umap,

    #annotate_genes_on_heatmap.py
    annotate_gene_groups_with_ai_biological_process,

    #agreement_bar_plots
    plot_grouped_average,
    plot_model_agreement,
    plot_model_agreement_categorical,

    #sankey_plots.py
    plot_sankey,
    save_sankey,

    #confusion_matrix_plots.py
    plot_confusion_matrix_from_adata,
    plot_confusion_matrix,

    #stabilizing_classifier_plots.py
    plot_training_history,
    plot_label_changes,

)


from .automated_label_management import (

    #clean single column
    #--------------------
    # in_adata_obs.py
    simplify_obs_column,
    create_label_hierarchy,
    map_cell_type_labels_to_simplified_set,

    # in_adata_var.py
    simplify_var_index,
    map_gene_labels_to_simplified_set,

    #unify_columns_within_adata.py
    ensure_label_consistency_main,
    ensure_label_consistency_adata,

    #unify_columns_between_adata.py
    ai_unify_labels,

)

from .utils import (

    # anndictionary_.py
    enforce_semantic_list,
    make_names,
    normalize_string,
    normalize_label,
    create_color_map,
    get_slurm_cores,
    summarize_metadata,
    display_html_summary,

    # anndata_.py
    remove_genes,
    add_col_to_adata_obs,
    add_col_to_adata_var,
    add_col_to_pd_df,
    convert_obs_col_to_category,
    convert_obs_col_to_string,
    convert_obs_index_to_str,
    get_adata_columns,
    filter_gene_list,

    # scanpy_.py
    sample_adata_dict,
    sample_and_drop,

    # pca_density_filter.py
    pca_density_filter_main,
    pca_density_filter_adata,

    # stabilizing_classifier.py
    stable_label,
    stable_label_adata,

    # read_spatial_data.py
    add_blank_image_to_adata,
    build_adata_from_transcript_positions,
    build_adata_from_visium,


)


__all__ = [
    # Core
    # -----------
    #adata_dict.py
    "AdataDict",
    "to_nested_tuple",
    "to_nested_list",

    #adata_dict_fapply.py
    "adata_dict_fapply",
    "adata_dict_fapply_return",

    # build.py
    "build_adata_dict",

    # add_stratification.py
    "add_stratification",

    # write.py
    "write_adata_dict",

    #read.py
    "read_adata_dict",
    "read_adata_dict_from_h5ad",

    # concatenate.py
    "concatenate_adata_dict",

    # utils.py
    "check_and_create_stratifier",

    # LLM
    # --------
    #llm_call.py
    "configure_llm_backend",
    "get_llm_config",
    "call_llm",
    "retry_call_llm",

    #parse_llm_response.py
    "extract_dictionary_from_ai_string",
    "extract_list_from_ai_string",
    "process_llm_category_mapping",

    #provider_initializer_mapping.py
    "LLMProviders",


    # Annotate
    # -----------
    # Cells
    # De Novo
    "ai_cell_type",
    "ai_cell_types_by_comparison",
    "ai_from_expected_cell_types",
    "ai_biological_process",
    "ai_annotate",
    "ai_annotate_by_comparison",
    "ai_annotate_from_expected_cell_types",
    "ai_annotate_cell_type",
    "ai_annotate_cell_type_by_comparison",
    "ai_annotate_cell_sub_type",
    "ai_annotate_biological_process",
    "cell_type_marker_gene_score",
    "ai_determine_leiden_resolution",

    # Label Transfer
    "harmony_label_transfer",
    "train_label_classifier",
    "transfer_labels_using_classifier",

    # Benchmarking
    "create_label_df",
    "kappa_adata",
    "krippendorff_alpha_adata",
    "ai_compare_cell_type_labels_pairwise",

    # Genes
    # Make Marker Gene Lists
    "ai_make_cell_type_gene_list",

    # Plot
    # -------
    # from module_score_plots.py
    "module_score_barplot",
    "module_score_umap",

    # from annotate_genes_on_heatmap.py
    "annotate_gene_groups_with_ai_biological_process",

    # from agreement_bar_plots.py
    "plot_grouped_average",
    "plot_model_agreement",
    "plot_model_agreement_categorical",

    # from sankey_plots.py
    "plot_sankey",
    "save_sankey",

    # from confusion_matrix_plots.py
    "plot_confusion_matrix_from_adata",
    "plot_confusion_matrix",

    # from stabilizing_classifier_plots.py
    "plot_training_history",
    "plot_label_changes",

    # Automated Label Management
    # ----------------------------

    #clean single column
    # from in_adata_obs.py
    "simplify_obs_column",
    "create_label_hierarchy",
    "map_cell_type_labels_to_simplified_set",

    # from in_adata_var.py
    "simplify_var_index",
    "map_gene_labels_to_simplified_set",

    # from unify_columns_within_adata.py
    "ensure_label_consistency_main",
    "ensure_label_consistency_adata",

    # from unify_columns_between_adata.py
    "ai_unify_labels",

    # Utils
    # -------
    # anndictionary_.py
    "enforce_semantic_list",
    "make_names",
    "normalize_string",
    "normalize_label",
    "create_color_map",
    "get_slurm_cores",
    "summarize_metadata",
    "display_html_summary",

    # scanpy_.py
    "sample_adata_dict",
    "sample_and_drop",

    # anndata_.py
    "remove_genes",
    "add_col_to_adata_obs",
    "add_col_to_adata_var",
    "add_col_to_pd_df",
    "convert_obs_col_to_category",
    "convert_obs_col_to_string",
    "convert_obs_index_to_str",
    "get_adata_columns",
    "filter_gene_list",

    # pca_density_filter.py
    "pca_density_filter_main",
    "pca_density_filter_adata",

    # read_spatial_data.py
    "add_blank_image_to_adata",
    "build_adata_from_transcript_positions",
    "build_adata_from_visium",

]


# Run mac system check for multithreading compatibility
if platform.system() == "Darwin":
    try:
        numba.config.THREADING_LAYER = 'tbb'
        # numba.set_num_threads(2)

        @numba.jit(nopython=True, parallel=True)
        def _test_func():
            acc = 0
            for i in numba.prange(4): # pylint: disable=not-an-iterable
                acc += i
            return acc

        _test_func()
        if numba.config.THREADING_LAYER != 'tbb':
            raise RuntimeError("Expected TBB threading layer, got something else.")

    except Exception:
        # Print only our custom error and exit; no traceback will be shown.
        sys.tracebacklimit = 0  # Suppress traceback
        raise RuntimeError(
            "Failed to initialize TBB threading layer on macOS!\n"
            "Try re-installing numba + TBB via conda (run exactly these 3 lines of code):\n"
            "  pip uninstall tbb numba\n"
            "  conda remove tbb numba\n"
            "  conda install -c conda-forge tbb numba\n"
            "Then restart python and re-attempt import\n"
        ) from None


# Run mac system configuration for multithreading
if platform.system() == 'Darwin':
    # Set Numba threading layer to 'tbb'
    if os.getenv("NUMBA_THREADING_LAYER") is None:
        os.environ["NUMBA_THREADING_LAYER"] = "tbb"
        numba.config.THREADING_LAYER = 'tbb'  # Explicitly set the threading layer using config
