"""
This subpackage contains functions to annotate cells de novo (from scratch), based on marker genes.

Generally in this subpackage, you will find a function named ``ai_annotate_blank`` that 
takes an :class:`AnnData` object, and internally calls ``ai_blank``, a function that takes a gene list.

For convenience, we provide documentation for both forms of the function in case you want to use an :class:`AnnData` object as input, 
or directly pass lists of genes for flexibility/customization of inputs.
"""

#the base ai_annotate functions
from .base import (
    ai_annotate,
    ai_annotate_by_comparison
)

#cell annotation methods--by gene list
from .annotate_cell_type import (
    ai_cell_type,
    ai_annotate_cell_type
)

#cell type annotation methods--by gene list (in the context of other gene lists)
from .annotate_cell_type_by_comparison import (
    ai_cell_types_by_comparison,
    ai_annotate_cell_type_by_comparison,

)

#cell type annotation methods--by gene list (in the context of other gene lists, drawing from a list of expected cell types)
from .annotate_from_expected_cell_types import (
    ai_from_expected_cell_types,
    ai_annotate_from_expected_cell_types,

)

from .annotate_cell_subtype import (
    ai_annotate_cell_type_by_comparison_adata_dict,
    ai_annotate_cell_sub_type,
)

#annotate groups of cells with biological process (and not cell type)
from .annotate_biological_process import (
    ai_biological_process,
    ai_annotate_biological_process,
)

#automatically calculate gene module scores for each cell type
from .cell_type_score import (
    cell_type_marker_gene_score,
)

#automatically determine cluster resolution
from .automated_clustering import (
    ai_determine_leiden_resolution,
)

__all__ = [
    # base.py
    "ai_annotate",
    "ai_annotate_by_comparison",

    # annotate_cell_type.py
    "ai_cell_type",
    "ai_annotate_cell_type",

    # annotate_cell_type_by_comparison.py
    "ai_cell_types_by_comparison",
    "ai_annotate_cell_type_by_comparison",

    # annotate_from_expected_cell_types.py
    "ai_from_expected_cell_types",
    "ai_annotate_from_expected_cell_types",

    # annotate_cell_subtype.py
    "ai_annotate_cell_type_by_comparison_adata_dict",
    "ai_annotate_cell_sub_type",

    # annotate_biological_process.py
    "ai_biological_process",
    "ai_annotate_biological_process",

    # cell_type_score.py
    "cell_type_marker_gene_score",

    # automated_clustering.py
    "ai_determine_leiden_resolution"
]
