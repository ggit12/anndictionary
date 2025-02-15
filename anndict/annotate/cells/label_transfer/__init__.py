"""
This subpackage contains functions to transfer labels (from an already labeled adata to an unlabeled adata).
"""

#label transfer based on harmony integration
from .harmony import (
    harmony_label_transfer,
)

#label transfer using an sklearn classifier
from .sklearn_classifier import (
    train_label_classifier,
    transfer_labels_using_classifier,
)


__all__ = [
    # harmony.py
    "harmony_label_transfer",

    # sklearn_classifier.py
    "train_label_classifier",
    "transfer_labels_using_classifier",
]