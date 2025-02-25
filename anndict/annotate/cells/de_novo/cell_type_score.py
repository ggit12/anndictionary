"""
This module contains functions that calculate cell type marker gene scores automaticallly (i.e. you supply only the cell type, not the marker genes).
"""
import numpy as np
import scanpy as sc

from anndata import AnnData

from anndict.utils.anndata_ import filter_gene_list
from anndict.utils.anndictionary_ import enforce_semantic_list
from anndict.annotate.genes import ai_make_cell_type_gene_list


def cell_type_marker_gene_score(
    adata: AnnData,
    cell_type_col: str = None,
    cell_types: list[str] = None,
    species: str = "Human",
    list_length: str = None,
    score_name: str = "_score",
    adt_key: tuple = None,
    **kwargs,
) -> None:
    """
    Compute marker gene scores for specified cell types. Must provide 
    either a list of cell types, or a column that contains cell_type labels.

    Parameters
    ------------
    adata
        An :class:`AnnData` object.

    cell_type_col
        Column name in ``adata.obs`` containing cell type annotations.

    cell_types
        List of cell types for which to compute the marker gene scores.

    species
        Species for gene list generation. Defaults to ``'Human'``.

    list_length
        Qualitative length of the marker gene list. Can be anything like 
        ``'short'`` or ``'long'``. Try ``'long'`` if you are having trouble 
        getting valid genes that are present in your dataset.

    score_name
        Suffix for the computed score names. Defaults to '_score'.

    adt_key
        Used by :func:`adata_dict_fapply` or :func:`adata_dict_fapply_return` 
        when passing this function to them.

    **kwargs
        Optional keyword args passed to :func:`sc.tl.score_genes`.

    Returns
    --------
    This function modifies the input ``adata`` object in-place. See **Notes** for details.

    Notes
    ------
        - ``adata.var`` is updated with boolean columns indicating genes used in the scores.
        - ``adata.obs`` is updated with new columns containing the computed scores for each 
            observation.

    Examples
    ---------

    .. code-block:: python

        import anndict as adt

        # Calculate Scores
        adt.cell_type_marker_gene_score(adata, cell_type_col='cell_type', species='Human', list_length="longer")

    See Also
    ---------
    :func:`module_score_umap` : To conveniently visualize these module scores on UMAPs.
    :func:`module_score_barplot` : To conveniently visualize these module scores as a bar plot.
    """

    score_name_suffix = score_name

    # Check for conflicting parameters
    if cell_types is not None and cell_type_col is not None:
        raise ValueError("Provide either 'cell_type_col' or 'cell_types', not both.")

    if cell_types is None:
        if cell_type_col is not None:
            cell_types = adata.obs[cell_type_col].unique().tolist()
        else:
            raise ValueError("Either 'cell_type_col' or 'cell_types' must be provided.")
    else:
        # Ensure cell_types is a list
        if isinstance(cell_types, str):
            cell_types = [cell_types]

    # Ensure cell_types contains only valid labels
    enforce_semantic_list(cell_types)

    for cell_type in cell_types:
        cell_type = str(cell_type)  # Ensure cell_type is a string
        # Set the score_name per cell type
        score_name = f"{cell_type}{score_name_suffix}"

        # Generate gene list using ai_gene_list function
        gene_list = ai_make_cell_type_gene_list(cell_type, species, list_length=list_length)

        # Filter the gene list based on genes present in adata
        gene_list = filter_gene_list(adata, gene_list)

        # Mark genes included in this score in adata.var
        adata.var[score_name] = adata.var.index.isin(gene_list)

        # calculate score if any valid genes, otherwise print warning and assign score value as NaN.
        if gene_list:
            # Compute the gene score and store it in adata.obs[score_name]
            sc.tl.score_genes(adata, gene_list=gene_list, score_name=score_name, **kwargs)
        else:
            # Assign NaN to adata.obs[score_name] for all observations
            adata.obs[score_name] = np.nan
            print(f"No valid genes for {cell_type} in {adt_key if adt_key else ''}. Assigning score value as NaN")
