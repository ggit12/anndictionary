"""
This module contains utility functions for ``AnnDictionary``.
"""

from .anndictionary_ import (
    enforce_semantic_list,
    make_names,
    normalize_string,
    normalize_label,
    create_color_map,
    get_slurm_cores,
    summarize_metadata,
    display_html_summary,

)

from .anndata_ import (
    remove_genes,
    add_col_to_adata_obs,
    add_col_to_adata_var,
    add_col_to_pd_df,
    convert_obs_col_to_category,
    convert_obs_col_to_string,
    convert_obs_index_to_str,
    get_adata_columns,
    filter_gene_list,

)

from .scanpy_ import (
    sample_adata_dict,
    sample_and_drop,

)

from .pca_density_filter import (
    pca_density_filter_main,
    pca_density_filter_adata,
    pca_density_subsets,

)

from .stabilizing_classifier import (
    stable_label,
    stable_label_adata,
)

from .read_spatial_data import (
    read_transcript_coords,
    get_steps_and_coords,
    populate_sparse_array,
    process_gene_counts,
    create_anndata,
    add_blank_image_to_adata,
    build_adata_from_transcript_positions,
    build_adata_from_visium,
)

from .uce_ import (
    uce_adata,

)

__all__ = [
    #anndictionary_.py
    "enforce_semantic_list",
    "make_names",
    "normalize_string",
    "normalize_label",
    "create_color_map",
    "get_slurm_cores",
    "summarize_metadata",
    "display_html_summary",

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

    # scanpy_.py
    "sample_adata_dict",
    "sample_and_drop",

    # pca_density_filter.py
    "pca_density_filter_main",
    "pca_density_filter_adata",
    "pca_density_subsets",

    # stabilizing_classifier.py
    "stable_label",
    "stable_label_adata",

    # read_spatial_data.py
    "read_transcript_coords",
    "get_steps_and_coords",
    "populate_sparse_array",
    "process_gene_counts",
    "create_anndata",
    "add_blank_image_to_adata",
    "build_adata_from_transcript_positions",
    "build_adata_from_visium",

    # uce_.py
    "uce_adata",
]
