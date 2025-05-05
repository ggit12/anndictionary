"""
This module contains visualizations functions for cell typing.
"""
import numpy as np
import scanpy as sc


import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
from anndata import AnnData


def module_score_barplot(adata: AnnData,
    group_cols: str | list[str],
    score_cols: str | list[str],
    figsize: tuple[int,int] = (10,8),
    adt_key: tuple[str,...] | None = None,
    ) -> tuple[Figure, Axes]:
    """
    Create a bar plot of mean module scores grouped by specified columns.

    Parameters
    ------------
    adata
        The :class:`AnnData` object containing the data.

    group_cols
        The column(s) in adata.obs to group by.

    score_cols
        The column(s) in adata.obs that contain the module scores.
    
    figsize
        The figure size (width, height)

    adt_key
        Used by :func:`adata_dict_fapply` and :func:`adata_dict_fapply_return` 
        when passing this function.

    Returns
    --------
    The :class:`Figure` and :class:`Axes` objects of the plot.

    Examples
    ---------

    .. code-block:: python

        import anndict as adt

        # Calculate Scores
        adt.cell_type_marker_gene_score(adata, cell_type_col='cell_type', species='Human', list_length="longer")

        # Calculate a umap
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

        # Plot the results
        plots = adt.module_score_barplot(adata, group_cols='cell_type', score_cols=score_cols)

    See Also
    ---------
    :func:`cell_type_marker_gene_score` : To calculate cell type marker gene scores.

    """
    #print adt_key if provided
    if adt_key:
        print(adt_key)

    # Ensure group_cols and score_cols are lists
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    if isinstance(score_cols, str):
        score_cols = [score_cols]

    # Select module score columns and the group columns
    module_scores = adata.obs[score_cols + group_cols]

    # Group by the group_cols and compute mean module scores
    mean_scores = module_scores.groupby(group_cols, observed=False).mean()

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the mean module scores as a grouped bar plot
    mean_scores.plot(kind='bar', ax=ax)

    # Set labels, title, and legend location
    ax.set_ylabel('Mean Module Score')
    ax.legend(title=None, loc=6, bbox_to_anchor=(1,0.5))

    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig, ax


def module_score_umap(adata: AnnData,
    score_cols: list[str],
    adt_key: tuple[str,...] | None = None,
    **kwargs
) -> tuple[Figure, np.ndarray[Axes]]:
    """
    Generates UMAP plots for specified module scores in a single figure.

    Parameters
    -----------
    adata
        Annotated data matrix containing UMAP coordinates and module scores.

    score_cols
        List of column names in `adata` containing module scores to plot.
    
    adt_key
        Used by :func:`adata_dict_fapply` and :func:`adata_dict_fapply_return` 
        when passing this function.

    **kwargs
        Additional keyword arguments passed to `sc.pl.umap`, including ``'vmax'`` 
        for color scaling (default is ``'p99'``).

    Returns
    --------
    :class:`Figure` containing the UMAP plots.

    Examples
    ---------

    .. code-block:: python

        import anndict as adt

        # Calculate Scores
        adt.cell_type_marker_gene_score(adata, cell_type_col='cell_type', species='Human', list_length="longer")

        # Calculate a umap
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

        # Plot the results
        plots = adt.module_score_umap(adata, score_cols=score_cols + ['cell_type'])

    """
    # Print adt_key if provided
    if adt_key:
        print(adt_key)

    # Extract vmax from kwargs or default to 'p99'
    vmax = kwargs.pop('vmax', 'p99')

    n_plots = len(score_cols)
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Ensure axes is a flat array of axes
    axes = np.atleast_1d(axes).ravel()

    for _, (score_col, ax) in enumerate(zip(score_cols, axes)):
        # Process title
        title = ' '.join(word.capitalize() for word in score_col.replace('_', ' ').split())
        # Plot the UMAP
        sc.pl.umap(adata, color=score_col, title=title, vmax=vmax, ax=ax, show=False, **kwargs)
        ax.set_title(title)

    # Turn off any unused axes
    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    return fig, axes
