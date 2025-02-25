"""
Anndata-specific ``Predicoder`` that remembers which ``adata.obsm`` key it was trained on.
"""
# pylint: disable=arguments-renamed
#disable pylint false positives

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData

from .predicoder import Predicoder

class AdataPredicoder(Predicoder):
    """
    A layer-aware Predicoder for AnnData objects that stores predictions directly in adata.obs.
    
    Parameters
    ----------
    classifier
        The classifier to wrap
    label_encoder
        The label encoder used to transform labels
    feature_key
        Key of data to use in ``adata.obsm``, or ``'use_X'`` to use ``adata.X``
    
    See Also
    --------
    :class:`Predicoder` : the base class.
    """

    def __init__(
        self,
        classifier: ClassifierMixin,
        label_encoder: LabelEncoder,
        feature_key: str,
    ):
        super().__init__(classifier, label_encoder)
        self.feature_key = feature_key

    def _get_x(self, adata: AnnData) -> np.ndarray:
        """Extract features from the AnnData object."""
        if self.feature_key == "use_X":
            return adata.X
        if self.feature_key not in adata.obsm:
            raise KeyError(
                f"Feature key '{self.feature_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        return adata.obsm[self.feature_key]

    def predict(
        self,
        adata: AnnData,
        new_column_name: str = 'predicted_label'
    ) -> None:
        """
        Predict labels and store them in ``adata.obs[new_column_name]``.
        
        Parameters
        ----------
        adata
            An :class:`AnnData`.

        new_column_name
            Key to store predictions in ``adata.obs``

        Returns
        -------
        None

        Notes
        -----
        Updates ``adata`` in-place with predictions, stored in ``adata.obs[new_column_name]``.
        """
        x = self._get_x(adata)
        predictions = super().predict(x)
        adata.obs[new_column_name] = predictions

    def predict_proba(
        self,
        adata: AnnData,
        new_column_name: str = 'predicted_prob'
    ) -> None:
        """
        Get predicted probabilities of membership for each class and store them in ``adata.obs``.
        
        Parameters
        ----------
        adata
            An :class:`AnnData`.

        new_column_name
            Prefix for probability columns in ``adata.obs`` (default: 'predicted_prob')
            Each class probability will be stored as '{new_column_name}_{class_name}'
            
        Returns
        -------
        None

        Notes
        -----
        Updates ``adata`` in-place with predicted probabilities, stored in ``adata.obs``.
        """
        x = self._get_x(adata)
        probabilities = super().predict_proba(x)

        # Store probability for each class
        for idx, class_name in enumerate(self.label_encoder.classes_):
            col_name = f"{new_column_name}_{class_name}"
            adata.obs[col_name] = probabilities[:, idx]
