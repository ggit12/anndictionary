"""
This module contains functions that use LLMs to automate label management within and between dataframes.
All of these functions are for processing category labels in pd dfs. We separate these functions
into 3 categories based on their input, see the left navigation
sidebar for the docs, and a short description below.

The functions are all broadly based around easing friction when working with categorical labels.
In the context of single cell RNA sequencing data, some common examples include:

- "My cell type labels have typos."
    - i.e. ``T cells`` and ``T cels``
- "I downloaded data from 3 different groups, and they each used a different label for the same cell type"
    - i.e. ``macrophage``, ``Macrophage.``, ``MΦ``
- "I want to coarsen the cell type labels."
    - i.e. Map ``CD8+ T cells`` and ``CD4+ T cells`` to a single category called ``T cells``


Single Column
---------------
**Input:** a single column.
    - These functions may generate more than one output column/category mapping, but they only look at a single set of input categories/labels.

Within Adata
--------------
**Input:** multiple columns within a single :class:`DataFrame` (i.e. ``.obs`` from a single ``adata``)
    - These functions process multiple columns within a single ``DataFrame``, usually outputting a single column that resolves differences among the input columns.

Between Adata
---------------
**Input** multiple DataFrames (i.e. ``.obs`` from several ``adata``)
    - These functions process columns across :class:`DataFrames`, usually inserting a new column into each that shares labels/categories across the :class:`DataFrames`.
"""

#post-process single categorical columns
from .clean_single_column import (
    simplify_obs_column,
    simplify_var_index,
    create_label_hierarchy,
    map_cell_type_labels_to_simplified_set,
    map_gene_labels_to_simplified_set,

)

#make categorical columns within a df share a common set of labels
from .unify_columns_within_adata import (
    ensure_label_consistency_main,
    ensure_label_consistency_adata,

)

#make categorical columns between dfs share a common set of labels
from .unify_columns_between_adata import (
    ai_unify_labels
)

__all__ = [
    # from clean single column
    # --------------------------
    # in_adata_obs.py
    "simplify_obs_column",
    "create_label_hierarchy",
    "map_cell_type_labels_to_simplified_set",

    # in_adata_var.py
    "simplify_var_index",
    "map_gene_labels_to_simplified_set",

    # from unify_columns_within_adata.py
    "ensure_label_consistency_main",
    "ensure_label_consistency_adata",

    # from unify_columns_between_adata.py
    "ai_unify_labels",
]
