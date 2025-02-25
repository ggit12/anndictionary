"""
This module annotates gene lists with cell types based on a list of expected cell types. 
Can pass tissue information. 
Can pass cell type information if looking for subtypes with ``subtype=True``.

Based on :func:`~anndict.annotate.cells.de_novo.ai_cell_types_by_comparison`.
"""

from anndata import AnnData

from anndict.utils import enforce_semantic_list
from anndict.llm import retry_call_llm, process_llm_category_mapping
from .base import ai_annotate_by_comparison


def ai_from_expected_cell_types(
    gene_lists: list[list[str]],
    expected_cell_types: list[str] = None,
    cell_types: list[str] | None = None,
    tissues: list[str] = None,
    subtype: bool = False
) -> list[str]:
    """
    Returns cell type labels for multiple lists of marker genes as determined by an LLM.

    Parameters
    -------------
    gene_lists
        A list containing multiple lists of genes to be described.

    expected_cell_types
        A list of expected cell types to provide context to the LLM. 
        The output labels will draw from this list.

    cell_types
        A list of cell types to provide context to the LLM. 
        Each entry corresponds to a gene list in ``gene_lists``.

    tissues
        A list of tissues of origin to provide context to the LLM. 
        Each entry corresponds to a gene list in ``gene_lists``.

    subtype
        Whether to try to annotate cell type (if ``False``) or subtype (if ``True``).

    Returns
    --------
    The cell type labels generated by an LLM for each gene list.
    """
    if not gene_lists:
        raise ValueError("Empty list of gene lists passed to ai_cell_type_by_comparison")

    if cell_types and not subtype:
        raise ValueError("Cannot provide cell types without subtype=True")

    if not expected_cell_types:
        raise ValueError("Expected cell types must be provided. \
            To generate cell type annotations by comparison without providing \
            expected cell types, use :func:`ai_cell_types_by_comparison`.")

    if not tissues:
        raise ValueError("Tissues must be provided. \
            To generate cell type annotations by comparison without providing \
            expected cell types, use :func:`ai_cell_types_by_comparison`.")

    if len(tissues) != len(gene_lists):
        raise ValueError("len(tissues) must match len(gene_lists).")

    # Enforce semantic_list for each gene list
    for gene_list in gene_lists:
        enforce_semantic_list(gene_list)

    # Prepare the system prompt
    system_prompt = (
        # Description of desired response style
        "You are a terse molecular biologist. You respond in a few words \
        and without restating any part of the question. "

        # Prime with expected cell types
        f"Note that this is {set(tissues)} tissue and we expect to find \
        the following cell types: {'    '.join(expected_cell_types)}."

        # Initial instructions
        f"Compare and contrast gene sets to identify the most likely \
        cell type based on marker genes."
    )

    # Prepare the initial user prompt for contrasting all gene lists
    # initial_prompt = f"Tissue: {tissues}, " if tissue else ""
    # initial_prompt += f"Cell Type: {cell_type}, " if cell_type else ""
    initial_prompt = "Briefly compare and contrast the following gene sets:\n"
    for i, gene_list in enumerate(gene_lists):
        tissue_str = " " + ', '.join(tissues[i]) if tissues and tissues[i] else ""
        cell_type_str = " " + ', '.join(cell_types[i]) if cell_types and cell_types[i] else ""

        initial_prompt += f"{i + 1}){tissue_str}{cell_type_str} {('    '.join(gene_list))}\n"

    # Initialize the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt}
    ]

    # Get the initial contrast response
    contrast_response = retry_call_llm(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to contrast gene sets",
        call_llm_kwargs={'max_tokens': 300, 'temperature': 0},
        max_attempts=1
    )

    # Append the contrast response to the conversation
    messages.append({"role": "assistant", "content": contrast_response})

    messages.append({"role": "user", "content": "Provide only the new label. "})

    # Process each gene list
    cell_subtype_labels = []
    for i, gene_list in enumerate(gene_lists):
        tissue_str = " " + ', '.join(tissues[i]) if tissues and tissues[i] else ""
        cell_type_str = " " + ', '.join(cell_types[i]) if cell_types and cell_types[i] else ""

        gene_set_prompt = f"What is the cell{tissue_str}{cell_type_str} {'sub' if subtype else ''}type \
            label for the gene set: {('    '.join(gene_list))}?"
        messages.append({"role": "user", "content": gene_set_prompt})

        # Get the subtype label
        subtype_label = retry_call_llm(
            messages=messages,
            process_response=lambda x: x.strip(),
            failure_handler=lambda cell_type_str=cell_type_str, i=i: cell_type_str if cell_types and cell_types[i] else "Unknown",
            call_llm_kwargs={'max_tokens': 50, 'temperature': 0},
            max_attempts=1
        )

        cell_subtype_labels.append(subtype_label)
        messages.append({"role": "assistant", "content": subtype_label})

    # print(f"{messages}")

    # Map the returned labels to the original expected cell types
    ## Construct LLM mapping dictionary from expected_cell_types
    llm_dict = {cell_type: cell_type for cell_type in expected_cell_types}

    ## Use the function to get the mapping of cell_subtype_labels to expected_cell_types
    mapped_labels = process_llm_category_mapping(cell_subtype_labels, llm_dict)

    ## Replace cell_subtype_labels with their mapped values
    updated_cell_subtype_labels = [mapped_labels[label] for label in cell_subtype_labels]

    return updated_cell_subtype_labels

def ai_annotate_from_expected_cell_types(
    adata: AnnData,
    groupby: str,
    n_top_genes: int,
    *,
    expected_cell_types: list[str] = None,
    new_label_column: str ='ai_cell_type_by_comparison',
    cell_type_of_origin_col: str = None,
    tissue_of_origin_col: str = None,
    **kwargs
) -> AnnData:
    """
    Annotate cell types by comparison using LLMs.

    This function wraps the ai_annotate function to perform cell type annotation
    based on comparison of gene expression profiles.

    Parameters
    -------------
    adata
        AnnData Annotated data matrix.

    groupby
        Column name in adata.obs for grouping cells.

    n_top_genes
        Number of top genes to consider for annotation.

    expected_cell_types
        A list of expected cell types to provide context to the LLM. 
        The output labels will draw from this list.

    new_label_column
        Name of the column to store the AI-generated cell type labels.

    cell_type_of_origin_col
        Name of a column in ``adata.obs`` that contains the cell type 
        of orgin. Used for context to the LLM.

    tissue_of_origin_col
        Name of a column in ``adata.obs`` that contains the tissue of orgin. 
        Used to provide context to the LLM.

    adt_key
        Used by :func:`adata_dict_fapply` or :func:`adata_dict_fapply_return` 
        when passing this function to them.

    Returns
    -------
    :class:`AnnData` with LLM-generated cell type labels stored in ``adata.obs[label_column]``.

    Examples
    --------

    .. code-block:: python

        import anndict as adt

        # This will annotate each cluster based on its top differentially expressed genes after considering all gene lists in the context of each other.
        # The expected cell types are provided as context, and output will be attempted to be mapped back to those cell types to ensure consistency.
        ai_annotate_from_expected_cell_types(
            adata,
            groupby='leiden',
            n_top_genes=10,
            expected_cell_types=['T cell', 'B cell', 'Macrophage', 'Fibroblast'],  # The list of expected cell types
            new_label_column='ai_cell_type_by_comparison',
            cell_type_of_origin_col='cell_type',
            tissue_of_origin_col='tissue                                           # Tissue information for context. Required.
        )
    """
    if not expected_cell_types:
        raise ValueError("Expected cell types must be provided. \
            To generate cell type annotations by comparison without providing \
            expected cell types, use :func:`ai_annotate_cell_type_by_comparison` instead.")

    if not tissue_of_origin_col:
        raise ValueError("tissue_of_origin_col must be provided. \
        To generate cell type annotations by comparison without providing \
        expected cell types, use :func:`ai_annotate_cell_type_by_comparison` instead.")

    return ai_annotate_by_comparison(func=ai_from_expected_cell_types, adata=adata, groupby=groupby, n_top_genes=n_top_genes, expected_cell_types=expected_cell_types,new_label_column=new_label_column, cell_type_of_origin_col=cell_type_of_origin_col, tissue_of_origin_col=tissue_of_origin_col, **kwargs)
