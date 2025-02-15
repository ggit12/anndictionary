"""
Defines the Predicoder class, a container for a classifier and its label encoder that automatically decodes predictions.
"""

from typing import Any

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder


class Predicoder:
    """Predicter+Decoder. Container for a classifier and its label encoder that automatically decodes predictions."""

    def __init__(
        self,
        classifier: ClassifierMixin,
        label_encoder: LabelEncoder
    ):
        self.classifier = classifier
        self.label_encoder = label_encoder

    def predict(self, x: Any) -> np.ndarray:
        """Get predictions in the original label space."""
        y_encoded = self.classifier.predict(x)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba(self, x: Any) -> np.ndarray:
        """Get probability predictions if the classifier supports them."""
        if hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(x)
        raise AttributeError(f"{self.classifier.__class__.__name__} does not implement predict_proba")
