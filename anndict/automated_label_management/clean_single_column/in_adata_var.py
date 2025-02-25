"""
Clean single columns in ``adata.var``
"""

import ast

import pandas as pd

from anndata import AnnData

from anndict.utils import enforce_semantic_list
from anndict.llm import (
    retry_call_llm,
    extract_dictionary_from_ai_string,
    process_llm_category_mapping
)

def simplify_var_index(
    adata: AnnData,
    column: str,
    new_column_name: str,
    simplification_level: str = ''
    ) -> dict:
    """

    Simplifies gene names in the index of the :class:`AnnData` object's var 
    attribute based on a boolean column, and stores the result in a new 
    column using the :func:`map_gene_labels_to_simplified_set`. This function 
    assumes that ``adata.var`` contains gene symbols (i.e. PER1, IL1A) 
    and not numeric indices or accession numbers.

    Parameters
    ------------
    adata
        The :class:`AnnData` object containing the data.

    column
        The boolean column in ``adata.var`` used to select genes for simplification.

    new_column_name
        The name of the new column to store the simplified labels.

    simplification_level
        A qualitative description of how much you want the labels to be simplified. 
        Could be anything, like ``'extremely'``, ``'barely'``, or ``'pathway-level'``.

    Returns
    --------
    A :class:`dict` containing the map from the current labels to the simplified labels

    Raises
    --------
    ValueError
        If more than 1000 genes are selected for simplification or if the 
        masking column (used to select genes) is not boolean.

    Notes
    -------
    Modifies ``adata`` by adding ``adata.var[new_column_name]`` (i.e. the new labels) in-place.

    Example
    ---------
    .. code-block:: python

        import anndict as adt

        print(adata.var)
        >  index        simplify
        > 'HSP90AA1'    1
        > 'HSPA1A'      1
        > 'HSPA1B'      1
        > 'CLOCK'       1
        > 'ARNTL'       1
        > 'PER1'        1
        > 'IL1A'        1
        > 'IL6'         1
        > 'APOD'        0
        > 'CFD'         0

        label_mapping = adt.simplify_var_index(adata,
                                '',
                                new_column_name = 'functional_category',
                                simplification_level='functional category level'
                                )

        print(adata.var) # New column added
        >  index        simplify        functional_category
        > 'HSP90AA1'    1               'Heat Shock Proteins'
        > 'HSPA1A'      1               'Heat Shock Proteins'
        > 'HSPA1B'      1               'Heat Shock Proteins'
        > 'CLOCK'       1               'Circadian Rythm'
        > 'ARNTL'       1               'Circadian Rythm'
        > 'PER1'        1               'Circadian Rythm'
        > 'IL1A'        1               'Interleukin'
        > 'IL6'         1               'Interleukin'
        > 'APOD'        0                Nan
        > 'CFD'         0                Nan

    """
    if not pd.api.types.is_bool_dtype(adata.var[column]):
        raise ValueError(f"The column '{column}' must be a boolean column.")

    # Get the indices of True in the boolean column
    selected_genes = adata.var.index[adata.var[column]]

    if len(selected_genes) > 1000:
        raise ValueError("Cannot simplify more than 1000 genes at a time.")

    # Get the mapping of original labels to simplified labels using the provided function
    label_mapping = map_gene_labels_to_simplified_set(
        selected_genes, simplification_level=simplification_level)

    # Apply the mapping to create the new column in the AnnData object
    adata.var[new_column_name] = adata.var.index.to_series()\
        .map(label_mapping).fillna(adata.var.index.to_series())

    return label_mapping


def map_gene_labels_to_simplified_set(
    labels: list[str],
    simplification_level: str = '',
    batch_size: int = 50
    ) -> dict:
    """
    Maps a list of genes to a simplified set of labels using an LLM, processing in batches.

    Parameters
    -----------
    labels
        The list of labels to be mapped.

    simplification_level
        A qualitative description of how much you want the labels to be simplified.

    batch_size
        The number of labels to process in each batch.

    Returns
    ----------
    A :class:`dict` mapping the original labels to the simplified set of labels.

    Example
    ---------
    .. code-block:: python

        import anndict as adt

        gene_labels = ['HSP90AA1',
                       'HSPA1A',
                       'HSPA1B',
                       'CLOCK',
                       'ARNTL',
                       'PER1',
                       'IL1A',
                       'IL6'
                       ]

        label_mapping = adt.map_gene_labels_to_simplified_set(gene_labels,
                                simplification_level='functional category level'
                                )

        print(label_mapping)

        > {
        >     'HSP90AA1': 'Heat Shock Protein',
        >     'HSPA1A': 'Heat Shock Protein',
        >     'HSPA1B': 'Heat Shock Protein',
        >     'CLOCK': 'Circadian Rhythm',
        >     'ARNTL': 'Circadian Rhythm',
        >     'PER1': 'Circadian Rhythm',
        >     'IL1A': 'Interleukin',
        >     'IL6': 'Interleukin'
        > }
    """
    # Enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the initial prompt
    initial_labels_str = "    ".join(labels)

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system",
        "content": f"You are a python dictionary mapping generator that takes \
                a list of genes and provides a mapping to a {simplification_level} \
                simplified set as a dictionary. \
                Example: HSP90AA1    HSPA1A    HSPA1B    CLOCK    ARNTL    PER1    IL1A    IL6 -> \
                {{'HSP90AA1':'Heat Shock Proteins','HSPA1A':'Heat Shock Proteins',\
                'HSPA1B':'Heat Shock Proteins','CLOCK':'Circadian Rhythm',\
                'ARNTL':'Circadian Rhythm','PER1':'Circadian Rhythm',\
                'IL1A':'Interleukins','IL6':'Interleukins'}}"},
        {"role": "user",
        "content": f"Here is the full list of gene labels to be simplified: \
                {initial_labels_str}. Acknowledge that you've seen all labels. \
                Do not provide the mapping yet."}
    ]

    # Get initial acknowledgment
    initial_response = retry_call_llm(
        messages=messages,
        process_response=lambda x: x,
        failure_handler=lambda: "Failed to process initial prompt",
        call_llm_kwargs={'max_tokens': 30, 'temperature': 0},
        max_attempts=1
    )
    messages.append({"role": "assistant", "content": initial_response})

    def process_batch(batch_labels):
        batch_str = "    ".join(batch_labels)
        messages.append({"role": "user", \
                            "content": f"Provide a mapping for this batch of gene labels. \
                            Generate only a dictionary: {batch_str} -> "})

        def process_response(response):
            cleaned_mapping = extract_dictionary_from_ai_string(response)
            return ast.literal_eval(cleaned_mapping)

        def failure_handler(labels):
            print(f"Simplification failed for gene labels: {labels}")
            return {label: label for label in labels}

        call_llm_kwargs = {
            'max_tokens': min(300 + 25*len(batch_labels), 4000),
            'temperature': 0
        }
        failure_handler_kwargs = {'labels': batch_labels}

        batch_mapping = retry_call_llm(
            messages=messages,
            process_response=process_response,
            failure_handler=failure_handler,
            call_llm_kwargs=call_llm_kwargs,
            failure_handler_kwargs=failure_handler_kwargs
        )
        messages.append({"role": "assistant", "content": str(batch_mapping)})
        return batch_mapping

    # Process all labels in batches
    full_mapping = {}
    for i in range(0, len(labels), batch_size):
        batch = labels[i:i+batch_size]
        batch_mapping = process_batch(batch)
        full_mapping.update(batch_mapping)

    # Final pass to ensure consistency
    final_mapping = process_llm_category_mapping(labels, full_mapping)

    return final_mapping
