"""
This module contains functions that process labels within a single column.
"""

from .in_adata_obs import (
    simplify_obs_column,
    create_label_hierarchy,
    map_cell_type_labels_to_simplified_set,

)

from .in_adata_var import (
    simplify_var_index,
    map_gene_labels_to_simplified_set,

)

__all__ = [
    #in_adata_obs.py
    "simplify_obs_column",
    "create_label_hierarchy",
    "map_cell_type_labels_to_simplified_set",

    #in_adata_var.py
    "simplify_var_index",
    "map_gene_labels_to_simplified_set",

]
