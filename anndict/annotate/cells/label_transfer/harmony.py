"""
Label transfer via harmony integration.
"""

import anndata as ad
import scanpy as sc
import harmonypy as hm

from anndata import AnnData

def harmony_label_transfer(
    origin_adata: AnnData,
    destination_adata: AnnData,
    origin_subset_column: str,
    label_column: str
) -> None:
    """
    Perform Harmony integration and transfer labels from ``origin_adata`` to ``destination_adata``.

    This function subsets ``origin_adata`` based on a provided column to get the cells
    that match in the same column of ``destination_adata``. It then performs Harmony
    integration on the combined dataset and transfers the specified label column
    from origin_adata to ``destination_adata``.

    Parameters
    ----------
    origin_adata
        The origin AnnData object containing the reference data and labels.

    destination_adata
        The AnnData object to which labels will be transferred.

    origin_subset_column
        The column name in ``origin_adata.obs`` used for subsetting ``origin_adata`` to match ``destination_adata``.

    label_column
        The column name in ``origin_adata.obs`` containing the labels to be transferred.

    Returns
    -------
    None

    Notes
    -----
    Modifies ``destination_adata`` in-place, adding a new column 'harmony_labels' in ``adata_to_label.obs`` containing the transferred labels.
    """

    # Subset origin_adata based on the provided column to get matching cells
    matching_cells = origin_adata.obs[origin_adata.obs[origin_subset_column].isin(destination_adata.obs[origin_subset_column])]
    origin_subset = origin_adata[matching_cells.index]

    # Combine destination_adata and the subset of origin_adata
    combined_data = ad.concat([destination_adata, origin_subset])

    # Perform Harmony integration
    sc.tl.pca(combined_data, svd_solver='arpack')
    harmony_results = hm.run_harmony(combined_data.obsm['X_pca'], combined_data.obs, origin_subset_column)
    combined_data.obsm['X_harmony'] = harmony_results.Z_corr.T

    # Separate the integrated data back into the original datasets
    destination_adata_integrated = combined_data[:destination_adata.n_obs]
    origin_integrated = combined_data[destination_adata.n_obs:]

    # Transfer labels from origin_adata to destination_adata using the integrated data
    sc.pp.neighbors(origin_integrated, use_rep='X_harmony')
    sc.tl.umap(origin_integrated)
    sc.tl.leiden(origin_integrated, resolution=0.5)

    # Transfer the specific label column from origin_integrated to destination_adata_integrated
    origin_labels = origin_integrated.obs[label_column]
    destination_adata_integrated = destination_adata_integrated.copy() # Make it not a view
    destination_adata_integrated.obs[label_column] = origin_labels.reindex(destination_adata_integrated.obs.index).ffill()

    # Update destination_adata with the new labels
    destination_adata.obs['harmony_labels'] = destination_adata_integrated.obs[label_column]
