"""
This module contains adata_dict wrappers for ``squidpy``.
"""
from functools import wraps

import squidpy as sq

from anndict.adata_dict import AdataDict, adata_dict_fapply, adata_dict_fapply_return

@wraps(sq.gr.spatial_neighbors)
def compute_spatial_neighbors_adata_dict(
    adata_dict: AdataDict,
    **kwargs
    ) -> None:
    """
    Wrapper for :func:`sq.gr.spatial_neighbors`.
    """
    adata_dict_fapply(adata_dict, sq.gr.spatial_neighbors, **kwargs)

@wraps(sq.gr.co_occurrence)
def perform_colocalization_adata_dict(
    adata_dict: AdataDict,
    cluster_key: str = "cell_type",
    **kwargs
) -> None:
    """
    Wrapper for :func:`sq.gr.co_occurrence`.
    """
    adata_dict_fapply(adata_dict, sq.gr.co_occurrence, cluster_key=cluster_key, **kwargs)

@wraps(sq.pl.co_occurrence)
def plot_colocalization_adata_dict(
    adata_dict: AdataDict,
    cluster_key: str = "cell_type",
    source_cell_type: str | None = None,
    figsize: tuple[float, float] = (10, 5),
    **kwargs
) -> None:
    """
    Wrapper for :func:`sq.pl.co_occurrence`.
    """
    def plot_coloc(adata, **kwargs):
        if source_cell_type:
            # Get matches for the source cell type in the cluster key
            matches = [ct for ct in adata.obs[cluster_key].unique() if source_cell_type in ct]
            sq.pl.co_occurrence(adata, cluster_key=cluster_key, clusters=matches, figsize=figsize, **kwargs)
        else:
            sq.pl.co_occurrence(adata, cluster_key=cluster_key, figsize=figsize, **kwargs)
    adata_dict_fapply(adata_dict, plot_coloc, use_multithreading=False, **kwargs)

@wraps(sq.gr.interaction_matrix)
def compute_interaction_matrix_adata_dict(
    adata_dict,
    cluster_key="cell_type",
    **kwargs
) -> dict:
    """
    Wrapper for :func:`sq.gr.interaction_matrix`.
    """
    return adata_dict_fapply_return(adata_dict, sq.gr.interaction_matrix, cluster_key=cluster_key, **kwargs)

@wraps(sq.pl.interaction_matrix)
def plot_interaction_matrix_adata_dict(
    adata_dict,
    cluster_key="cell_type",
    **kwargs
) -> None:
    """
    Wrapper for :func:`sq.pl.interaction_matrix`.
    """
    adata_dict_fapply(adata_dict, sq.pl.interaction_matrix, cluster_key=cluster_key, use_multithreading=False, **kwargs)
