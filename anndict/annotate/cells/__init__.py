"""
This subpackage contains functions to annotate and assess cell annotation.
"""

#annotate cells de novo (from scratch)
from .de_novo import (
    ai_annotate, ai_annotate_by_comparison,
    ai_cell_type, ai_annotate_cell_type,
    ai_cell_types_by_comparison,
    ai_from_expected_cell_types,
    ai_annotate_cell_type_by_comparison,
    ai_annotate_from_expected_cell_types,
    ai_annotate_cell_type_by_comparison_adata_dict,
    ai_annotate_cell_sub_type,
    ai_biological_process, ai_annotate_biological_process,
    cell_type_marker_gene_score,
    ai_determine_leiden_resolution
)

#transfer labels from an already labeled adata to an unlabaled adata
from .label_transfer import (
    harmony_label_transfer,
    train_label_classifier,
    transfer_labels_using_classifier,
)

#analyze annotation results
from .benchmarking import (
    create_label_df,
    kappa_adata, krippendorff_alpha_adata,
    ai_compare_cell_type_labels_pairwise
)


__all__ = [
    # De Novo
    #--------
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
    "ai_determine_leiden_resolution",

    # Label Transfer
    #--------------
    # harmony.py
    "harmony_label_transfer",

    # sklearn_classifier.py
    "train_label_classifier",
    "transfer_labels_using_classifier",

    # Benchmarking
    #------------
    # label_utils.py
    "create_label_df",

    # agreement_metrics.py
    "kappa_adata",
    "krippendorff_alpha_adata",

    # ai_compare_cell_type_labels.py
    "ai_compare_cell_type_labels_pairwise",
]
