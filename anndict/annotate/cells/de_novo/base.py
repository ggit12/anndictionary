"""
This module contains core functions for de novo annotation of cells based on marker genes and LLMs.
The functions in this module are called by other annotation functions.
We include these functions in the docs for reference, but you should not generally use them directly.
"""
import warnings
import pandas as pd
import scanpy as sc

from pandas import DataFrame
from anndata import AnnData

from anndict.utils.anndata_ import convert_obs_col_to_category

def _validate_and_prepare_rank_genes(
    adata: AnnData,
    groupby: str
) -> tuple:
    """
    Validates the ``groupby`` column and ensures ``rank_genes_groups`` analysis is present.

    Parameters
    ----------
    adata
        An :class:`AnnData`.

    groupby
        The column name to group by
        
    Returns
    -------
    A :class:`tuple` of the form ``(rank_genes_groups, clusters)``
    """
    # Ensure the groupby column is categorical
    if not isinstance(adata.obs[groupby].dtype, pd.CategoricalDtype):
        adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Get the number of categories in the groupby column
    n_categories = len(adata.obs[groupby].cat.categories)

    # Warn if there are more than 50 categories
    if n_categories > 50:
        warnings.warn(
            f"The '{groupby}' column has {n_categories} groups, which may result in "
            "slow runtimes. Ensure that {groupby} is not continuous data.", 
            UserWarning
        )

    # Check if rank_genes_groups has already been run
    if 'rank_genes_groups' not in adata.uns or adata.uns['rank_genes_groups']['params']['groupby'] != groupby:
        print(
            f"rerunning diffexp analysis because not found in adata.uns for "
            f"adata.obs['{groupby}']. (run before annotating to avoid this)"
        )
        sc.tl.rank_genes_groups(adata, groupby, method='t-test')

    rank_genes_groups = adata.uns['rank_genes_groups']
    clusters = rank_genes_groups['names'].dtype.names

    return rank_genes_groups, clusters

def _validate_metadata_columns(
    adata: AnnData,
    tissue_col: str | None = None,
    cell_type_col: str | None = None
) -> tuple | None:
    """
    Validates the existence of metadata columns and returns which ones are valid.
    
    Parameters
    ----------
    adata
        An :class:`AnnData`.

    tissue_col
        Name of tissue column

    cell_type_col
        Name of cell type column
        
    Returns
    -------
    A :class:`tuple` of the form ``(tissue_col, cell_type_col)``
    or ``None`` for invalid columns
    """
    if tissue_col and tissue_col not in adata.obs.columns:
        warnings.warn(
            f"Tissue of origin column '{tissue_col}' not found in adata.obs, "
            "will not consider tissue of origin for cell type annotation.",
            UserWarning
        )
        tissue_col = None

    if cell_type_col and cell_type_col not in adata.obs.columns:
        warnings.warn(
            f"Cell type of origin column '{cell_type_col}' not found in adata.obs, "
            "will not consider cell type of origin for annotation.",
            UserWarning
        )
        cell_type_col = None

    return tissue_col, cell_type_col

def _get_cluster_metadata(
    adata: AnnData,
    clusters: list[str | int] | tuple[str | int],
    groupby: str,
    tissue_col: str | None = None,
    cell_type_col: str | None = None
) -> tuple:
    """
    Gets tissue and cell type metadata for each cluster if available.

    Parameters
    ----------
    adata
        An :class:`AnnData`.

    clusters
        Sequence of cluster names

    groupby
        The column name used for grouping

    tissue_col
        Name of tissue column

    cell_type_col
        Name of cell type column

    Returns
    -------
    A :class:`tuple` of the form ``(cluster_to_tissue, cluster_to_cell_type)``
    """
    cluster_to_tissue = {}
    cluster_to_cell_type = {}

    if tissue_col or cell_type_col:
        for cluster in clusters:
            mask = adata.obs[groupby] == cluster

            if tissue_col:
                tissues = adata.obs.loc[mask, tissue_col].unique()
                cluster_to_tissue[cluster] = tissues[0] if len(tissues) == 1 else ", ".join(tissues)

            if cell_type_col:
                cluster_to_cell_type[cluster] = adata.obs.loc[mask, cell_type_col].unique().tolist()

    return cluster_to_tissue, cluster_to_cell_type

def _update_adata_and_create_results(
    adata: AnnData,
    cell_type_annotations: dict,
    groupby: str,
    new_label_column: str,
    n_top_genes: int,
    top_genes_lists: dict | list | None = None
) -> DataFrame:
    """
    Updates the AnnData object with annotations and creates the results DataFrame.

    Parameters
    ----------
    adata
        An :class:`AnnData`.

    cell_type_annotations
        Mapping of clusters to their annotations

    groupby
        The column name used for grouping

    new_label_column
        Name of the new annotation column

    n_top_genes
        Number of top genes used

    top_genes_lists
        Top genes for each cluster

    Returns
    -------
    Results DataFrame
    """
    # Create results list
    results = []
    for cluster, annotation in cell_type_annotations.items():
        result = {
            groupby: cluster,
            new_label_column: annotation,
        }

        if isinstance(top_genes_lists, dict):
            result[f"top_{n_top_genes}_genes"] = top_genes_lists[cluster]
        elif isinstance(top_genes_lists, list):
            cluster_idx = list(cell_type_annotations.keys()).index(cluster)
            result[f"top_{n_top_genes}_genes"] = top_genes_lists[cluster_idx]

        results.append(result)

    # Update adata
    adata.obs[new_label_column] = adata.obs[groupby].map(cell_type_annotations)
    convert_obs_col_to_category(adata, new_label_column)

    return pd.DataFrame(results)

def ai_annotate(
    func: callable,
    adata: AnnData,
    groupby: str,
    n_top_genes: int,
    new_label_column: str,
    tissue_of_origin_col: str = None,
    **kwargs,
) -> DataFrame:
    """
    Annotate clusters based on the top marker genes for each cluster.

    This uses marker genes for each cluster and applies func to 
    determine the label for each cluster based on the top n
    marker genes. The results are added to the AnnData 
    object and returned as a DataFrame.

    If rank_genes_groups hasn't been run on the adata, 
    this function will automatically run ``sc.tl.rank_genes_groups``

    Parameters
    ------------
    func
        A function that takes ``gene_list`` 
        **:** :class:`list` **[** :class:`str` **]** and returns ``annotation`` **:** :class:`str`.

    adata
        An :class:`AnnData` object.

    groupby
        Column in ``adata.obs`` to group by for differential expression analysis.

    n_top_genes
        The number of top marker genes to consider for each cluster.

    new_label_column
        The name of the new column in ``adata.obs`` where the annotations will be stored.

    tissue_of_origin_col
        Name of a column in ``adata.obs`` that contains 
        the tissue of orgin. Used to provide context to the LLM.

    **kwargs
        additional kwargs passed to ``func``

    Returns
    ---------
    A ``pd.DataFrame`` with a column for the top marker genes for each cluster.

    Notes
    -------
    This function also modifies the input ``adata`` in place, 
    adding annotations to ``adata.obs[new_label_col]``
    """
    # Validate and prepare rank genes data
    rank_genes_groups, clusters = _validate_and_prepare_rank_genes(adata, groupby)

    # Validate metadata columns
    tissue_of_origin_col, _ = _validate_metadata_columns(adata, tissue_col=tissue_of_origin_col)

    # Get cluster metadata
    cluster_to_tissue, _ = _get_cluster_metadata(
        adata, clusters, groupby, tissue_col=tissue_of_origin_col
    )

    # Process each cluster
    cell_type_annotations = {}
    top_genes_dict = {}

    for cluster in clusters:
        if tissue_of_origin_col:
            kwargs['tissue'] = cluster_to_tissue[cluster]

        top_genes = list(rank_genes_groups['names'][cluster][:n_top_genes])
        top_genes_dict[cluster] = top_genes

        annotation = func(top_genes, **kwargs)
        cell_type_annotations[cluster] = annotation

    return _update_adata_and_create_results(
        adata, cell_type_annotations, groupby, new_label_column,
        n_top_genes, top_genes_dict
    )

def ai_annotate_by_comparison(
    func :callable,
    adata: AnnData,
    groupby: str,
    n_top_genes: int,
    new_label_column: str,
    cell_type_of_origin_col: str = None,
    tissue_of_origin_col: str = None,
    **kwargs,
) -> DataFrame:
    """
    Annotate clusters based on the top marker genes for each cluster, 
    in the context of the other clusters' marker genes.

    This uses marker genes for each cluster and applies func to 
    determine the label for each cluster based on the top n
    marker genes. The results are added to the AnnData object 
    and returned as a DataFrame.

    If rank_genes_groups hasn't been run on the adata, this 
    function will automatically run ``sc.tl.rank_genes_groups``

    Parameters
    -------------
    func
        A function that takes ``gene_lists`` **:** :class:`list` 
        **[** :class:`list` **[** :class:`str` **] ]** and

        returns ``annotations`` **:** :class:`list` **[** :class:`str` **]**, 
        one for each :class:`list` of genes in ``gene_lists``.

    adata
        An :class:`AnnData` object.

    groupby
        Column in ``adata.obs`` to group by for differential expression analysis.

    n_top_genes
        The number of top marker genes to consider for each cluster.

    new_label_column
        The name of the new column in ``adata.obs`` where the annotations will be stored.

    cell_type_of_origin_col
        Name of a column in ``adata.obs`` that contains the 
        cell type of orgin. Used for context to the LLM.

    tissue_of_origin_col
        Name of a column in ``adata.obs`` that contains the 
        tissue of orgin. Used to provide context to the LLM.

    **kwargs
        additional kwargs passed to ``func``

    Returns
    --------
    A ``pd.DataFrame`` with a column for the top marker genes for each cluster.

    Notes
    -------
    This function also modifies the input ``adata`` in place, 
    adding annotations to ``adata.obs[new_label_col]``
    """
    # Validate and prepare rank genes data
    rank_genes_groups, clusters = _validate_and_prepare_rank_genes(adata, groupby)

    # Validate metadata columns
    tissue_of_origin_col, cell_type_of_origin_col = _validate_metadata_columns(
        adata, tissue_of_origin_col, cell_type_of_origin_col
    )

    # Get cluster metadata
    cluster_to_tissue, cluster_to_cell_type = _get_cluster_metadata(
        adata, clusters, groupby, tissue_of_origin_col, cell_type_of_origin_col
    )

    # Get top genes for all clusters
    top_genes = [list(rank_genes_groups['names'][cluster][:n_top_genes]) for cluster in clusters]

    # Prepare kwargs with metadata
    if tissue_of_origin_col:
        kwargs['tissues'] = [cluster_to_tissue[cluster] for cluster in clusters]
    if cell_type_of_origin_col:
        kwargs['cell_types'] = [cluster_to_cell_type[cluster] for cluster in clusters]

    # Get annotations for all clusters at once
    annotations = func(top_genes, **kwargs)

    # Create cell type annotations dictionary
    cell_type_annotations = dict(zip(clusters, annotations))

    return _update_adata_and_create_results(
        adata, cell_type_annotations, groupby, new_label_column,
        n_top_genes, top_genes
    )
