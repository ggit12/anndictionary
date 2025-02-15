"""
This module handles label transfer via sklearn models.
"""

from typing import Type, Literal

from sklearn.base import ClassifierMixin
from anndata import AnnData

from anndict.utils.stabilizing_classifier import stable_label_adata

from .adata_predicoder import AdataPredicoder


def train_label_classifier(
    adata: AnnData,
    label_key: str,
    feature_key: str | Literal["use_X"],
    classifier_class: Type[ClassifierMixin],
    *,
    random_state: int | None = None,
    **kwargs
) -> AdataPredicoder:
    """
    Trains a classifier on the given ``adata``; used internally by :func:`transfer_labels_using_classifier`.

    Parameters
    ----------
    adata
        An :class:`AnnData` containing the original labels.
        A classifier will be trained on this ``adata``.

    label_key
        Key in ``adata.obs`` containing the original labels.

    feature_key
        Key of data to use in ``adata.obsm``, or ``'use_X'`` to use ``adata.X``.

    classifier_class
        Any classifier inheriting from :class:`sklearn.base.ClassifierMixin`. 
        Pass as a class, e.g. ``LogisticRegression``, and not an already-instantiated object.

    random_state
        random state seed passed to :func:`stable_label_adata`.

    **kwargs
        Additional keyword arguments passed to the classifier constructor.

    Returns
    -------
    A :class:`AdataPredicoder`, containing the trained classifier and label encoder/decoder.

    See Also
    --------
    :class:`AdataPredicoder` : The container class for classifier+label encoder/decoder.
    :func:`stable_label_adata` : The function that trains the classifier on ``adata``.

    Examples
    --------
    **Case 1: Using a logistic regression classifier**

    .. code-block:: python

        import anndict as adt
        from sklearn.linear_model import LogisticRegression

        train_label_classifier(
            adata=adata,
            label_key='cell_type',
            feature_key='X_pca',
            classifier_class=LogisticRegression,
            penalty='l2', #one kwarg for LogisticRegression
            fit_intercept=True, #another kwarg for LogisticRegression
        )

    **Case 2: Using a random forest classifier**

    .. code-block:: python

        import anndict as adt
        from sklearn.ensemble import RandomForestClassifier

        train_label_classifier(
            adata=adata,
            label_key='cell_type',
            feature_key='X_pca',
            classifier_class=RandomForestClassifier,
            n_estimators=1000, #one kwarg for RandomForestClassifier
            max_features='sqrt', #another kwarg for RandomForestClassifier
        )

    """
    # create a classifier for this stratum
    classifier = classifier_class(random_state=random_state, **kwargs)

    classifier, _, _, _, label_encoder = stable_label_adata(
        adata, label_key, feature_key, classifier, max_iterations=1, random_state=random_state
    ) # max_iterations=1 causes classifier to only train one round, without label resampling

    predicoder = AdataPredicoder(classifier, label_encoder, feature_key)

    return predicoder

def transfer_labels_using_classifier(
    origin_adata: AnnData,
    destination_adata: AnnData,
    origin_label_key: str,
    feature_key: str | Literal["use_X"],
    classifier_class: Type[ClassifierMixin],
    new_column_name: str = 'predicted_label',
    random_state: int | None = None,
    **kwargs
) -> AdataPredicoder:
    """
    Transfers labels from ``origin_adata`` to ``destination_adata`` 
    using a classifier of type ``classifier_class``.

    Supported classifiers include any ``sklearn`` classifier inheriting 
    from :class:`sklearn.base.ClassifierMixin`.

    Parameters
    ----------
    origin_adata
        An :class:`AnnData` containing the original labels.
        A classifier will be trained on this adata.

    destination_adata
        An :class:`AnnData` containing the new cells to be 
        labeled. Must contain the same ``.obsm[feature_key]`` 
        as ``origin_adata`` if ``feature_key`` is not ``'use_X'``.

    origin_label_key
        Key in ``origin_adata.obs`` containing the original labels.

    feature_key
        Key of data to use in ``origin_adata.obsm``, or ``'use_X'`` to use ``origin_adata.X``.

    classifier_class
        Any classifier inheriting from :class:`sklearn.base.ClassifierMixin`. 
        Pass as a class, e.g. ``LogisticRegression``, and not an already-instantiated object.

    new_column_name
        The name of the new column in ``destination_adata.obs`` where 
        the predicted labels will be stored.
    
    random_state
        random state seed passed to :func:`stable_label_adata`.
    
    **kwargs
        Additional keyword arguments passed to the classifier constructor.

    Returns
    -------
    A :class:`AdataPredicoder` that contains the 
    trained classifier, and automatically decodes 
    predicted labels into text labels. Can be used 
    to calculate class membership probabilities or 
    predict on other :class:`AnnData`.

    Notes
    -----
    Modifies ``destination_adata`` in-place.

    See Also
    --------
    :class:`AdataPredicoder` : The container class for classifier+label 
        encoder/decoder.
    :func:`train_label_classifier` : The function that trains the 
        classifier on ``origin_adata``.

    Examples
    --------
    **Case 1: Using a logistic regression classifier**

    .. code-block:: python

        import anndict as adt
        from sklearn.linear_model import LogisticRegression

        transfer_labels(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_label_key='cell_type',
            feature_key='X_pca',
            classifier_class=LogisticRegression,
            new_column_name='predicted_label',
            penalty='l2', #one kwarg for LogisticRegression
            fit_intercept=True, #another kwarg for LogisticRegression
        )

    **Case 2: Using a random forest classifier**

    .. code-block:: python

        import anndict as adt
        from sklearn.ensemble import RandomForestClassifier

        transfer_labels(
            origin_adata=origin_adata,
            destination_adata=destination_adata,
            origin_label_key='cell_type',
            feature_key='X_pca',
            classifier_class=RandomForestClassifier,
            new_column_name='predicted_label',
            n_estimators=1000, #one kwarg for RandomForestClassifier
            max_features='sqrt', #another kwarg for RandomForestClassifier
        )

    """
    adata_predicoder = train_label_classifier(
        adata=origin_adata, feature_key=feature_key, label_key=origin_label_key, classifier_class=classifier_class, random_state=random_state, **kwargs)

    adata_predicoder.predict(destination_adata, new_column_name=new_column_name)
    return adata_predicoder
