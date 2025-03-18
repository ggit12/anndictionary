"""
Clean single columns in ``adata.obs``
"""

import ast

from anndata import AnnData

from anndict.utils import make_names, enforce_semantic_list, convert_obs_col_to_category
from anndict.llm import retry_call_llm, extract_dictionary_from_ai_string, process_llm_category_mapping

def simplify_obs_column(
    adata: AnnData,
    column: str,
    new_column_name: str,
    simplification_level: str = ''
    ) -> dict:
    """
    Simplifies labels in the specified column of the AnnData object and stores the result
    in a new column using :func:`map_cell_type_labels_to_simplified_set`.

    Parameters
    -------------
    adata 
        The :class:`AnnData` object containing the data.

    column
        The column in ``adata.obs`` containing the cell type labels to simplify.

    new_column_name
        The name of the new column to store the simplified labels.

    simplification_level
        A qualitative description of how much you want the labels to be 
        simplified. Could be anything, like  ``'extremely'``, ``'barely'``, 
        or ``'compartment-level'``.

    Returns
    --------
    A :class:`dict` mapping the original labels to the simplified set of labels.

    Notes
    -------
    Modifies ``adata`` by adding ``adata.obs[new_column_name]`` (i.e. the new labels) in-place.

    Example
    ---------
    .. code-block:: python

        import anndict as adt

        print(adata.obs)
        >     cell_subtype
        > 0   cd8+ t cell
        > 1   cd4+ t cell
        > 2   venous endothelial cell
        > 3   arterial endothelial cell

        label_mapping = adt.simplify_obs_column(adata,
                                'cell_subtype',
                                new_column_name = 'cell_type',
                                simplification_level = 'cell type level'
                                )

        print(adata.obs) # New column added
        >     cell_subtype               cell_type_level
        > 0   cd8+ t cell                t cell
        > 1   cd4+ t cell                t cell
        > 2   venous endothelial cell    endothelial cell
        > 3   arterial endothelial cell  endothelial cell
    """
    # Get the unique labels from the specified column
    unique_labels = adata.obs[column].unique()

    # Get the mapping of original labels to simplified labels using the provided function
    label_mapping = map_cell_type_labels_to_simplified_set(
        unique_labels, simplification_level=simplification_level)

    # Apply the mapping to create the new column in the AnnData object
    adata.obs[new_column_name] = adata.obs[column].map(label_mapping)

    # Convert annotation to categorical dtype
    convert_obs_col_to_category(adata, new_column_name)

    return label_mapping

def create_label_hierarchy(
    adata: AnnData,
    col: str,
    simplification_levels: list[str]
    ) -> dict:
    """
    Create a hierarchy of simplified labels based on a given column in AnnData.

    This function generates multiple levels of simplified labels from a single
    column in ``adata.obs``. Each successive level of simplification is created using
    the specified simplification level.

    Parameters
    -----------
    adata
        Annotated data matrix containing the column to be simplified.

    col
        Name of the column in ``adata.obs`` to be simplified.

    simplification_levels
        List of simplification levels to apply. Each level should be a 
        value that can be used by the simplify_obs_column function.

    Returns
    --------
    A :class:`dict` mapping new column names to their corresponding simplified label mappings. 
    The keys are the names of the new columns created for each simplification level, and 
    the values are the mappings returned by :func:`simplify_obs_column` for each level.

    Example
    ---------
    .. code-block:: python

        import anndict as adt

        print(adata.obs)
        >     subtype
        > 0   cd8+ t cell
        > 1   cd4+ t cell
        > 2   venous endothelial cell
        > 3   arterial endothelial cell

        label_mapping = adt.create_label_hierarchy(adata,
                                'subtype',
                                ['cell type level', 'cell class level', 'compartment level'])

        print(adata.obs) # New columns added
        >     subtype                    subtype_cell_type_level     subtype_cell_class_level    subtype_compartment_level
        > 0   cd8+ t cell                t cell                      lymphocyte                  immune
        > 1   cd4+ t cell                t cell                      lymphocyte                  immune
        > 2   venous endothelial cell    endothelial cell            endothelial cell            stromal
        > 3   arterial endothelial cell  endothelial cell            endothelial cell            stromal

    """
    base_col_name = col
    simplified_mapping = {}
    for level in simplification_levels:
        new_col_name = f"{base_col_name}_{make_names([level])[0]}"
        simplified_mapping[new_col_name] = simplify_obs_column(
            adata, col, new_col_name, simplification_level=level)
        col = new_col_name
    return simplified_mapping


# Label simplification functions
def map_cell_type_labels_to_simplified_set(
    labels: list[str],
    simplification_level: str = '',
    batch_size: int = 50
    ) -> dict:
    """
    Maps a list of labels to a simplified set of labels using an LLM, processing in batches.

    Parameters
    -----------
    labels
        The list of labels to be mapped.

    simplification_level
        A qualitative description of how much you want the labels to be simplified. 
        Or a direction about how to simplify the labels. Could be anything, like 
        ``'extremely'``, ``'barely'``, ``'compartment-level'``, ``'remove-typos'``

    batch_size
        The number of labels to process in each batch.
    
    Returns
    ---------
    A :class:`dict` mapping the original labels to the simplified set of labels.

    Example
    ---------
    .. code-block:: python

        import anndict as adt

        cell_sub_types = ['cd8+ t cell',
                          'cd4+ t cell',
                          'venous endothelial cell',
                          'arterial endothelial cell',
                          ]

        label_mapping = adt.map_cell_type_labels_to_simplified_set(cell_sub_types,
                                simplification_level = 'cell type level'
                                )

        print(label_mapping)

        > {
        >     'cd8+ t cell': 't cell',
        >     'cd4+ t cell': 't cell',
        >     'venous endothelial cell': 'endothelial cell',
        >     'arterial endothelial cell': 'endothelial cell'
        > }
    """
    #TODO could allow passing custom examples
    #enforce that labels are semantic
    enforce_semantic_list(labels)

    # Prepare the initial prompt
    initial_labels_str = "    ".join(labels)

    # Prepare the messages for the Chat Completions API
    messages = [
        {"role": "system",
         "content": f"You are a python dictionary mapping generator \
                that takes a list of categories and provides a mapping to a \
                {simplification_level} simplified set as a dictionary. Generate only a dictionary. \
                Example: Fibroblast.    Fibroblasts.    CD8-positive T Cells.    CD4-positive T Cells. -> \
                {{'Fibroblast.':'Fibroblast','Fibroblasts.':'Fibroblast',\
                'CD8-positive T Cells.':'T Cell','CD4-positive T Cells.':'T Cell'}}"},
        {"role": "user",
         "content": f"Here is the full list of labels to be simplified: {initial_labels_str}. \
                Succinctly acknowledge that you've seen all labels. Do not provide the mapping yet."}
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
        messages.append({"role": "user",
                         "content": f"Provide a mapping for this batch \
                                of labels. Generate only a dictionary: {batch_str} -> "})

        def process_response(response):
            cleaned_mapping = extract_dictionary_from_ai_string(response)
            return ast.literal_eval(cleaned_mapping)

        def failure_handler(labels):
            print(f"Simplification failed for labels: {labels}")
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
