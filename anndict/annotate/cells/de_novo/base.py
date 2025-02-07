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

    This uses marker genes for each cluster and applies func to determine the label for each cluster based on the top n
    marker genes. The results are added to the AnnData object and returned as a DataFrame.

    If rank_genes_groups hasn't been run on the adata, this function will automatically run ``sc.tl.rank_genes_groups``

    Parameters
    ------------
    func
        A function that takes ``gene_list`` **:** :class:`list` **[** :class:`str` **]** and returns ``annotation`` **:** :class:`str`.

    adata
        An :class:`AnnData` object.

    groupby
        Column in ``adata.obs`` to group by for differential expression analysis.

    n_top_genes
        The number of top marker genes to consider for each cluster.

    new_label_column
        The name of the new column in ``adata.obs`` where the annotations will be stored.

    tissue_of_origin_col
        Name of a column in ``adata.obs`` that contains the tissue of orgin. Used to provide context to the LLM.

    **kwargs
        additional kwargs passed to ``func``

    Returns
    ---------
    A ``pd.DataFrame`` with a column for the top marker genes for each cluster.

    Notes
    -------
    This function also modifies the input ``adata`` in place, adding annotations to ``adata.obs[new_label_col]``
    """
    # Ensure the groupby column is categorical
    if not pd.api.types.is_categorical_dtype(adata.obs[groupby]):
        adata.obs[groupby] = adata.obs[groupby].astype('category')

    # Get the number of categories in the groupby column
    n_categories = len(adata.obs[groupby].cat.categories)

    # Warn if there are more than 50 categories
    if n_categories > 50:
        warnings.warn(f"The '{groupby}' column has {n_categories} groups, which may result in slow runtimes. Ensure that {groupby} is not continuous data.", UserWarning)

    # Check if rank_genes_groups has already been run
    if 'rank_genes_groups' not in adata.uns or adata.uns['rank_genes_groups']['params']['groupby'] != groupby:
        # Run the differential expression analysis
        print(f"rerunning diffexp analysis because not found in adata.uns for adata.obs['{groupby}']. (run before annotating to avoid this)")
        sc.tl.rank_genes_groups(adata, groupby, method='t-test')

    # Get the rank genes groups result
    rank_genes_groups = adata.uns['rank_genes_groups']
    clusters = rank_genes_groups['names'].dtype.names

    # Check if tissue_of_origin_col exists in adata.obs
    if tissue_of_origin_col and tissue_of_origin_col not in adata.obs.columns:
        warnings.warn(f"Tissue of origin column '{tissue_of_origin_col}' not found in adata.obs, will not consider tissue of origin for cell type annotation.", UserWarning)
        tissue_of_origin_col = None

    # Get mapping of cluster to tissue if tissue_of_origin_col is provided
    cluster_to_tissue = {}
    if tissue_of_origin_col:
        for cluster in clusters:
            tissue = adata.obs.loc[adata.obs[groupby] == cluster, tissue_of_origin_col].unique()
            if len(tissue) > 1:
                tissue = ", ".join(tissue)
            else:
                tissue = tissue[0]
            cluster_to_tissue[cluster] = tissue

    # Initialize a dictionary to store cell type annotations
    cell_type_annotations = {}

    # Initialize a list to store the results
    results = []

    # Loop through each cluster and get the top n marker genes, then get cell type based on these marker genes
    for cluster in clusters:
        # Add tissue to kwargs if tissue_of_origin_col is provided
        if tissue_of_origin_col:
            kwargs['tissue'] = cluster_to_tissue[cluster]

        # Get top n genes
        top_genes = rank_genes_groups['names'][cluster][:n_top_genes]

        # Get annotation via func
        annotation = func(top_genes, **kwargs)
        cell_type_annotations[cluster] = annotation

        results.append({
            groupby: cluster,
            new_label_column: annotation,
            f"top_{n_top_genes}_genes": list(top_genes)
        })

    # Create a new column in .obs for cell type annotations
    adata.obs[new_label_column] = adata.obs[groupby].map(cell_type_annotations)

    # Convert annotation to categorical dtype
    convert_obs_col_to_category(adata, new_label_column)

    return pd.DataFrame(results)


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
    Annotate clusters based on the top marker genes for each cluster, in the context of the other clusters' marker genes.

    This uses marker genes for each cluster and applies func to determine the label for each cluster based on the top n
    marker genes. The results are added to the AnnData object and returned as a DataFrame.

    If rank_genes_groups hasn't been run on the adata, this function will automatically run ``sc.tl.rank_genes_groups``

    Parameters
    -------------
    func
        A function that takes ``gene_lists`` **:** :class:`list` **[** :class:`list` **[** :class:`str` **] ]** and

        returns ``annotations`` **:** :class:`list` **[** :class:`str` **]**, one for each :class:`list` of genes in ``gene_lists``.

    adata
        An :class:`AnnData` object.

    groupby
        Column in ``adata.obs`` to group by for differential expression analysis.

    n_top_genes
        The number of top marker genes to consider for each cluster.

    new_label_column
        The name of the new column in ``adata.obs`` where the annotations will be stored.

    cell_type_of_origin_col
        Name of a column in ``adata.obs`` that contains the cell type of orgin. Used for context to the LLM.

    tissue_of_origin_col
        Name of a column in ``adata.obs`` that contains the tissue of orgin. Used to provide context to the LLM.

    **kwargs
        additional kwargs passed to ``func``

    Returns
    --------
    A ``pd.DataFrame`` with a column for the top marker genes for each cluster.

    Notes
    -------
    This function also modifies the input ``adata`` in place, adding annotations to ``adata.obs[new_label_col]``
    """
    # Check if rank_genes_groups has already been run
    if 'rank_genes_groups' not in adata.uns or adata.uns['rank_genes_groups']['params']['groupby'] != groupby:
        # Run the differential expression analysis
        sc.tl.rank_genes_groups(adata, groupby, method='t-test')

    # Initialize a dictionary to store cell type annotations
    cell_type_annotations = {}

    # Initialize a list to store the results
    results = []

    # Get the rank genes groups result
    rank_genes_groups = adata.uns['rank_genes_groups']
    clusters = rank_genes_groups['names'].dtype.names  # List of clusters

    # Check if tissue_of_origin_col exists in adata.obs
    if tissue_of_origin_col and tissue_of_origin_col not in adata.obs.columns:
        warnings.warn(f"Tissue of origin column '{tissue_of_origin_col}' not found in adata.obs, will not consider tissue of origin for cell type annotation.", UserWarning)
        tissue_of_origin_col = None

    # Check if cell_type_of_origin_col exists in adata.obs
    if cell_type_of_origin_col and cell_type_of_origin_col not in adata.obs.columns:
        warnings.warn(f"Cell type of origin column '{cell_type_of_origin_col}' not found in adata.obs, will not consider cell type of origin for annotation.", UserWarning)
        cell_type_of_origin_col = None

    # Get mappings of clusters to tissues and cell types
    cluster_to_tissue = {}
    cluster_to_cell_type = {}

    if tissue_of_origin_col or cell_type_of_origin_col:
        for cluster in clusters:
            mask = adata.obs[groupby] == cluster

            # Map the cluster to tissues if tissue_of_origin_col is provided
            if tissue_of_origin_col:
                cluster_to_tissue[cluster] = adata.obs.loc[mask, tissue_of_origin_col].unique().tolist()

            # Map the cluster to cell types if cell_type_of_origin_col is provided
            if cell_type_of_origin_col:
                cluster_to_cell_type[cluster] = adata.obs.loc[mask, cell_type_of_origin_col].unique().tolist()

    # Create a list of lists for top genes
    top_genes = [list(rank_genes_groups['names'][cluster][:n_top_genes]) for cluster in clusters]

    # Create a list of tissues for each cluster and add to kwargs if tissue_of_origin_col is provided
    if tissue_of_origin_col:
        tissues_per_cluster = [cluster_to_tissue[cluster] for cluster in clusters]
        kwargs['tissues'] = tissues_per_cluster

    # Create a list of cell types for each cluster and add to kwargs if cell_type_of_origin_col is provided
    if cell_type_of_origin_col:
        cell_types_per_cluster = [cluster_to_cell_type[cluster] for cluster in clusters]
        kwargs['cell_types'] = cell_types_per_cluster

    # Call func with the list of lists
    annotations = func(top_genes, **kwargs)

    # Loop through each cluster and annotation
    for cluster, annotation in zip(clusters, annotations):
        cell_type_annotations[cluster] = annotation
        results.append({
            groupby: cluster,
            new_label_column: annotation,
            f"top_{n_top_genes}_genes": top_genes[clusters.index(cluster)]
        })

    # Create a new column in .obs for cell type annotations
    adata.obs[new_label_column] = adata.obs[groupby].map(cell_type_annotations)

    # Convert annotation to categorical dtype
    convert_obs_col_to_category(adata, new_label_column)

    return pd.DataFrame(results)
