"""
AnnDictionary utility functions.
"""

import os
import re

import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

from IPython.display import HTML, display
from anndata import AnnData


def enforce_semantic_list(
    lst: list[str]
) -> bool:
    """
    This function runs a number of checks to make sure that 
    the input is a semantic list, and not i.e. integers cast as strings.

    Parameters
    -----------
    lst
        List of strings to be checked.

    Returns
    --------
    ``True`` if the list passes all checks.

    Raises
    ------
    ValueError
        If the input list contains any of: NaN, numeric, or numeric cast as string.

    Examples
    --------
    .. code-block:: python

        enforce_semantic_list(['gene1', 'gene2', 'gene3'])

    Notes
    -----
    This function is useful for ensuring that the input list 
    contains semantic labels (i.e. gene symbols or cell types) and 
    not integer labels for LLM interpretation. It checks if all items are 
    strings and if any item can be converted to float.
    """
    error_message = "input list appears to contain any of: NaN, numeric, or \
        numeric cast as string. Please ensure you are passing semantic labels \
        (i.e. gene symbols or cell types) and not integer labels for AI \
        interpretation. Make sure adata.var.index and adata.obs.index are \
        not integers or integers cast as strings."

    def get_context(lst, index):
        before = lst[index - 1] if index > 0 else None
        after = lst[index + 1] if index < len(lst) - 1 else None
        return before, after

    # Check if all items are strings
    for index, item in enumerate(lst):
        if not isinstance(item, str):
            before, after = get_context(lst, index)
            raise ValueError(
                f"{error_message} Item at index {index} is not a \
                string: {item}. Context: Before: {before}, After: {after}"
            )

    # Check if any item can be converted to float
    for index, item in enumerate(lst):
        try:
            float(item)
        except ValueError:
            pass
        else:
            before, after = get_context(lst, index)
            raise ValueError(
                f"{error_message} Item at index {index} can be cast to \
                a number: {item}. Context: Before: {before}, After: {after}"
            )

    return True


def make_names(
    names: list[str]
) -> list[str]:
    """
    Convert a list of names into valid and unique Python identifiers.

    Parameters
    ----------
    names
        List of names to be transformed.

    Returns
    -------
        Valid and unique Python identifiers.
    """
    # Equivalent of R's make.names() function in Python
    valid_names = []
    seen = {}
    for name in names:
        # Replace invalid characters with underscores
        clean_name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
        if clean_name in seen:
            seen[clean_name] += 1
            clean_name = f"{clean_name}.{seen[clean_name]}"
        else:
            seen[clean_name] = 0
        valid_names.append(clean_name)
    return valid_names


def normalize_string(
    s: str
) -> str:
    """
    Removes non-alphanumeric characters and converts to lowercase.

    Parameters
    ----------
    s
        String to be normalized.

    Returns
    -------
        Normalized string.

    Examples
    --------
    .. code-block:: python

        normalize_string("Hello, World!")
        > "hello world"
    """
    return re.sub(r"[^\w\s]", "", s.lower())


def normalize_label(
    label: str
) -> str:
    """
    Calls normalize-string and handles NaN values.

    Parameters
    ----------
    label
        Label to be normalized.

    Returns
    -------
        Normalized string.

    Examples
    --------
    .. code-block:: python

        normalize_label("Hello, World!")
        > "hello world"

    .. code-block:: python

        normalize_label(np.nan)
        > "missing"
    """
    if pd.isna(label):  # Handle NaN values
        return "missing"
    return normalize_string(label.strip())


def create_color_map(adata: AnnData,
    keys: list[str]
) -> dict:
    """
    Creates a unified color map for given keys from an :class:`AnnData` object, differentiating
    between continuous and categorical data.

    Parameters
    ----------
    adata
        An :class`AnnData`.
    keys
        keys for which the color map is required.

    Returns
    -------
    A color map linking unique values or ranges from the specified keys to colors.
    """
    color_map = {}
    for key in keys:
        if pd.api.types.is_numeric_dtype(adata.obs[key]):
            # Create a continuous colormap
            min_val, max_val = adata.obs[key].min(), adata.obs[key].max()
            norm = plt.Normalize(min_val, max_val)
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
            # Store the scalar map directly
            color_map[key] = scalar_map
        else:
            # Handle as categorical
            unique_values = pd.unique(adata.obs[key])
            color_palette = sns.color_palette("husl", n_colors=len(unique_values))
            color_palette_hex = [
                matplotlib.colors.rgb2hex(color) for color in color_palette
            ]
            color_map[key] = dict(zip(unique_values, color_palette_hex))

    return color_map


def get_slurm_cores():
    """
    Returns the total number of CPU cores allocated to a Slurm job based on environment variables.

    Returns
    -------
    The total number of CPU cores allocated to the Slurm job.

    Examples
    --------
    .. code-block:: python

        get_slurm_cores()
        > 40

    Notes
    -----
    Checks the environment variables ``SLURM_CPUS_PER_TASK`` and ``SLURM_NTASKS`` to
    calculate the total number of CPU cores allocated to the job by multiplying their values.

    For environment variables that are not set, the function defaults to 1 CPU core.
    """
    # Get the number of CPUs per task (default to 1 if not set)
    cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))

    # Get the number of tasks (default to 1 if not set)
    ntasks = int(os.getenv("SLURM_NTASKS", "1"))

    # Calculate total cores
    total_cores = cpus_per_task * ntasks

    return total_cores


def summarize_metadata(
    adata: AnnData,
    cols: str | list[str]
) -> dict:
    """
    Generate a summary for specified metadata column(s) in ``adata``.

    Parameters
    -----------
    adata
        The anndata object containing the data.

    columns
        List of columns in ``adata.obs``. Use '*' to specify joint frequencies of multiple columns.

    Returns
    --------
    A :class:`dict` with keys as column descriptions and values as a :class:`DataFrame` of counts.

    Notes
    -----
    Use '*' to specify joint frequencies of multiple columns.

    Examples
    ---------

    **Case 1: calculate frequency of donor and tissue columns**

    .. code-block:: python

        import anndict as adt
        adt.wrappers.summarize_metadata(adata, ['donor', 'tissue'])

    **Case 2: calculate joint frequency of donor x tissue**

    .. code-block:: python

        import anndict as adt
        adt.wrappers.summarize_metadata(adata, ['donor*tissue'])

    See Also
    ---------
    :func:`get_adata_columns` : To use string matching to 
        retrieve specific column names from ``adata``.
    :func:`display_html_summary` : To print the results of 
        :func:`summarize_metadata` as an html table.
    :func:`anndict.wrappers.anndictionary_.summarize_metadata_adata_dict` :  
    :func:`anndict.wrappers.anndictionary_.display_html_summary_adata_dict` : 
    """
    results = {}

    for col in cols:
        if "*" in col:
            # Handle joint frequencies
            sub_cols = col.split("*")
            combined_data = adata.obs[sub_cols]

            # Convert categorical columns to string and replace NaN with a placeholder
            for sub_col in sub_cols:
                if isinstance(combined_data[sub_col].dtype, pd.CategoricalDtype):
                    combined_data[sub_col] = combined_data[sub_col].astype(str)

            # Replace NaN with a placeholder to include them in groupby
            combined_data = combined_data.fillna("NaN")

            joint_freq = combined_data.groupby(sub_cols).size().unstack(fill_value=0)
            joint_freq = (
                combined_data.groupby(sub_cols, observed=True)
                .size()
                .unstack(fill_value=0)
            )
            results[col.replace("*", " x ")] = joint_freq
        else:
            # Calculate frequency for a single column
            freq = adata.obs[col].value_counts(dropna=False).to_frame("count")
            results[col] = freq

    return results

# This behaviour is intentional for ease of use in Jupyter notebooks.
# pylint: disable=inconsistent-return-statements
def display_html_summary(
    summary_dict,
    return_html=False
) -> str:
    """
    Display separate HTML tables for each metadata category in the summary dictionary,
    arranging up to three tables in a row before starting a new line.

    Parameters
    ----------
    summary_dict
        The :class:`dict` containing frequency data for metadata columns.

    Returns
    -------
    If ``return_html`` is ``True``, returns the HTML string. Otherwise, returns ``None``.

    Notes
    -----
    Displays the results of :func:`summarize_metadata` as an HTML table.

    Examples
    --------

    .. code-block:: python

        import anndict as adt
        summary_dict = adt.wrappers.summarize_metadata(adata, ['donor', 'tissue', 'donor*tissue'])
        adt.wrappers.display_html_summary(summary_dict)

    See Also
    --------
    :func:`summarize_metadata` : To generate the summary data.
    """
    html = '<div style="display: flex; flex-wrap: wrap;">'
    table_count = 0

    for category, data in summary_dict.items():
        if table_count % 3 == 0 and table_count != 0:
            html += '<div style="flex-basis: 100%; height: 20px;"></div>'

        table_html = f'<div style="flex: 1; padding: 10px;"><h3>{category}</h3>'
        # Start the table and add a header row
        table_html += (
            '<table border="1"><tr><th></th>'  # Empty header for the row labels
        )
        table_html += (
            "".join(f"<th>{col}</th>" for col in data.columns) + "</tr>"
        )  # Column headers

        for index, row in data.iterrows():
            # Include row labels as the first column and the rest of the data in subsequent columns
            table_html += (
                f"<tr><td>{index}</td>"
                + "".join(f"<td>{val}</td>" for val in row)
                + "</tr>"
            )

        table_html += "</table></div>"
        html += table_html
        table_count += 1

    html += "</div>"
    display(HTML(html))

    if return_html:
        return html
