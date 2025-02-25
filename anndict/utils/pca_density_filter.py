"""
Filters cells based on their density in PCA space.
Can be useful to remove outliers, missannotated cells, or to get a more homogenous population.
"""

from warnings import warn

import numpy as np

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from scipy import sparse
from scipy.sparse import spmatrix
from anndata import AnnData

from anndict.utils.anndata_ import add_col_to_adata_obs, add_col_to_adata_var

# This return behavior is intentional to allow simplified execution of the filter
# pylint: disable=inconsistent-return-statements
def pca_density_filter_adata(
    adata: AnnData,
    use_layer: str | None = None,
    new_filter_column_name: str = "density_filter",
    drop_cells: bool = False,
    **kwargs
):
    """
    Filters  :func:`pca_density_filter` that accepts :class:`AnnData` as input.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    use_layer
        name of the layer to use for PCA (default: None, use adata.X).

    new_filter_column_name
        Name of the new boolean column to be created in ``adata.obs``.

    **kwargs
        Additional keyword arguments to pass to :func:`pca_density_filter_main`.

    Returns
    --------
    If ``drop_cells`` is ``True``, returns a new :class:`AnnData` with the filtered cells.
    Otherwise, modifies ``adata`` in-place by adding a boolean masking column.
    """
    # Get the data to use for PCA
    data = adata.layers[use_layer] if use_layer in adata.layers else adata.X

    # Calculate density (skip if fewer than 10 cells)
    if data.shape[0] < 10:
        density, cutoff, variables_used_in_pca = (
            np.ones(data.shape[0]),
            0,
            np.zeros(data.shape[1], dtype=bool),
        )
    else:
        density, cutoff, variables_used_in_pca = pca_density_filter_main(data, **kwargs)

    # Mark cells above the threshold
    high_density_indices = density > cutoff
    cell_passed_filter_bool = np.zeros(data.shape[0], dtype=bool)
    cell_passed_filter_bool[high_density_indices] = True

    # record which cells were passed
    # Note: this function converts index_vector to
    # ``pd.BooleanDtype`` (from ``bool``) when adding col to adata
    add_col_to_adata_obs(
        adata, np.arange(data.shape[0]), cell_passed_filter_bool, new_filter_column_name
    )

    # record which variables were used in PCA
    # Note: this function converts variables_used_in_pca to
    # ``pd.BooleanDtype`` (from ``bool``) when adding col to adata
    variable_used_in_pca_bool = np.zeros(adata.var.shape[0], dtype=bool)
    variable_used_in_pca_bool[variables_used_in_pca] = True
    add_col_to_adata_var(
        adata,
        np.arange(adata.var.shape[0]),
        variable_used_in_pca_bool,
        new_filter_column_name,
    )

    # Drop cells if requested
    if drop_cells:
        adata = adata[cell_passed_filter_bool].copy()
        return adata


def pca_density_filter_main(
    data: np.ndarray | spmatrix, n_components: int = 3, threshold: float = 0.10
) -> tuple[np.ndarray, float, list[int]]:
    """
    Calculate density contours for PCA-reduced data, return the density of all input data,
    and identify the indices of the variables that were included in the PCA.

    Parameters
    -----------
    data
        array-like, shape (n_samples, n_features)

    n_components
        number of components for PCA to reduce the data to.

    threshold
        percentile threshold for density values. Points with
        density above this threshold are considered high-density and will be retained.

    Returns
    --------
    A tuple containing:
        - pca_data: PCA-reduced data (None if all variables are constant).
        - density: Density values of all the points (None if all variables are constant).
        - variables_used_in_pca: Indices of unique variables that were included 
        in the PCA (empty list if all variables are constant).
    """

    # Convert to dense if sparse
    if sparse.issparse(data):
        data = data.toarray()

    # Check for constant variables (these will not be used by PCA)
    non_constant_columns = np.var(data, axis=0) > 0
    variables_used_in_pca = np.arange(data.shape[1])[non_constant_columns]

    # Skip the block if no non-constant variables are found
    if not np.any(non_constant_columns):
        return None, None, []

    # Adjust n_components if necessary
    n_features = np.sum(non_constant_columns)
    n_samples = data.shape[0]
    n_components = min(n_components, n_features, n_samples)

    # Perform PCA reduction only on non-constant variables
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data[:, non_constant_columns])

    # Calculate the point density for all points
    kde = gaussian_kde(pca_data.T)
    density = kde(pca_data.T)

    # Determine the density threshold
    cutoff = np.percentile(density, threshold * 100)

    return density, cutoff, variables_used_in_pca.tolist()


def pca_density_subsets(data: np.ndarray | spmatrix, labels: np.ndarray) -> np.ndarray:
    """
    Apply calculate_density_contours_with_unique_variables to subsets of X indicated by labels.
    Returns a vector indicating whether each row in X is above the threshold for its respective
    label group.

    Parameters
    -----------
    data
        array-like, shape (n_samples, n_features)
    labels
        array-like, shape (n_samples,), labels indicating the subset to which each row belongs

    Returns
    --------
    index_vector
        array-like, boolean vector of length n_samples indicating rows above the threshold
    """
    warn(
        "pca_density_subsets() is deprecated and will be removed \
        in a future version. Use pca_density_filter_adata() or pca_density_filter_main() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    unique_labels = np.unique(labels)
    index_vector = np.zeros(len(data), dtype=bool)

    for label in unique_labels:
        subset = data[labels == label]
        if subset.shape[0] < 10:
            # If fewer than 10 cells, include all cells by assigning density = 1 and cutoff = 0
            density, cutoff = np.ones(subset.shape[0]), 0
        else:
            density, cutoff, _ = pca_density_filter_main(
                subset, n_components=3, threshold=0.10
            )

        # Mark rows above the threshold for this label
        high_density_indices = density > cutoff
        global_indices = np.where(labels == label)[0][high_density_indices]
        index_vector[global_indices] = True

    return index_vector
