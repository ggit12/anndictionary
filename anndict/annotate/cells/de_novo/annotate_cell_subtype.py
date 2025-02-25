"""
This module contains functions to annotate cell subtype.
"""
from functools import wraps

from anndata import AnnData
from anndict.adata_dict import AdataDict, adata_dict_fapply_return, build_adata_dict, concatenate_adata_dict
from anndict.annotate.cells.de_novo.annotate_cell_type_by_comparison import ai_annotate_cell_type_by_comparison

@wraps(ai_annotate_cell_type_by_comparison)
def ai_annotate_cell_type_by_comparison_adata_dict(
    adata_dict: AdataDict,
    groupby: str,
    n_top_genes: int = 10,
    new_label_column: str = 'ai_cell_type_by_comparison',
    cell_type_of_origin_col: str | None = None,
    tissue_of_origin_col: str | None = None,
    **kwargs
) -> AdataDict:
    """
    Wrapper for ai_annotate_cell_type_by_comparison.
    """
    return adata_dict_fapply_return(adata_dict, ai_annotate_cell_type_by_comparison, max_retries=3, groupby=groupby, n_top_genes=n_top_genes, new_label_column=new_label_column, cell_type_of_origin_col=cell_type_of_origin_col, tissue_of_origin_col=tissue_of_origin_col, **kwargs)

def ai_annotate_cell_sub_type(adata: AnnData,
    cell_type_col: str,
    sub_cluster_col: str,
    new_label_column: str,
    tissue_of_origin_col: str = None,
    n_top_genes: int = 10
) -> tuple[AnnData, dict]:
    """
    Annotate cell subtypes using an LLM.
    This function performs LLM-based annotation of cell subtypes by first grouping cells
    by their main cell type, then annotating subtypes within each group.

    Parameters
    ----------
    adata
        An :class:`AnnData`.

    cell_type_col
        Column name in adata.obs containing main cell type labels.

    sub_cluster_col
        Column name in adata.obs containing sub-cluster information.

    new_label_col
        Name of the column to store the LLM-generated subtype labels.

    tissue_of_origin_col
        Missing description for this parameter.

    n_top_genes
        Missing description for this parameter.

    Returns
    -------
    A :class:`tuple` containing:

    :class:`AnnData`
        Concatenated annotated data with LLM-generated subtype labels.

    :class:`dict`
        Mapping of original labels to LLM-generated labels.

    Examples
    --------

    .. code-block:: python
    
        import anndict as adt

        # This will annotate the cell subtypes based on the top 10 differentially expressed genes in each group

        adt.ai_annotate_cell_sub_type(
            adata,
            cell_type_col='cell_type',           # Each cell will be assumed to be of the cell type indicated by this column
            sub_cluster_col='sub_cluster',       # This is the column that indicates the sub-cluster of each cell
            new_label_column='ai_cell_subtype',
            tissue_of_origin_col='tissue',
            n_top_genes=10)

    """
    #build adata_dict based on cell_type_col
    adata_dict = build_adata_dict(adata, strata_keys=[cell_type_col])

    label_mappings = ai_annotate_cell_type_by_comparison_adata_dict(adata_dict, groupby=sub_cluster_col, n_top_genes=n_top_genes, new_label_column=new_label_column, tissue_of_origin_col=tissue_of_origin_col, subtype=True)

    adata = concatenate_adata_dict(adata_dict, index_unique=None) #setting index_unique=None avoids index modification

    return adata, label_mappings
