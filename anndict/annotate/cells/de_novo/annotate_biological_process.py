"""
This module annotates groups of cells with a biological process, based on the group's enriched genes
"""

from pandas import DataFrame
from anndata import AnnData

from anndict.utils import enforce_semantic_list
from anndict.llm import call_llm
from anndict.annotate.cells.de_novo.base import ai_annotate


def ai_biological_process(gene_list: list[str]
) -> str:
    """
    Describes the most prominent biological process represented by a list of genes using an LLM.

    Parameters
    -------------
    gene_list
        The list of genes to be described.

    Returns
    ---------
    The biological process annotation of ``gene_list``.
    """
    if not gene_list:
        raise ValueError("Empty gene list passed to ai_biological_process")

    #enforce that labels are semantic
    enforce_semantic_list(gene_list)

    # Prepare the prompt
    if len(gene_list) == 1:
        base_prompt = f"In a few words and without restating any part of \
            the question, describe the single most prominent biological \
            process represented by the gene: {gene_list[0]}"
    else:
        genes_str = "    ".join(gene_list[:-1])
        base_prompt = f"In a few words and without restating any part of \
            the question, describe the single most prominent biological \
            process represented by the genes: {genes_str}, and {gene_list[-1]}"

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system", "content": "You are a terse molecular biologist."},
        {"role": "user", "content": base_prompt}
    ]

    # Call the LLM using the call_llm function
    annotation = call_llm(
        messages=messages,
        max_tokens=200,
        temperature=0
    )

    return annotation


def ai_annotate_biological_process(
    adata: AnnData,
    groupby: str,
    n_top_genes: int,
    new_label_column: str = 'ai_biological_process'
) -> DataFrame:
    """
    Annotate biological processes based on the top n marker genes for each cluster.

    This function performs differential expression analysis to identify marker genes 
    for each cluster and applies a user-defined function to determine the biological 
    processes for each cluster based on the top marker genes. The results are added 
    to the AnnData object and returned as a DataFrame.

    Parameters
    ------------
    adata
        An :class:`AnnData` object.

    groupby
        Column in ``adata.obs`` to group by for differential expression analysis.

    n_top_genes
        The number of top marker genes to consider.

    label_column
        The name of the new column in ``adata.obs`` where the cell type annotations will be stored.

    Returns
    --------
    A ``pd.DataFrame`` with a column for the top marker genes for each cluster.

    Notes
    -------
    This function also modifies the input ``adata`` 
    in place, adding annotations to ``adata.obs[new_label_col]``

    Examples
    ---------

    .. code-block:: python

        import anndict as adt

        # This will annotate the treatment group with biological processes based on the top 10 differentially expressed genes in each group
        ai_annotate_biological_process(adata, groupby='treatment_vs_control',
            n_top_genes=10, new_label_column='ai_biological_process')

    """
    return ai_annotate(func=ai_biological_process, adata=adata, groupby=groupby, n_top_genes=n_top_genes, new_label_column=new_label_column)
