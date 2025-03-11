"""
Concatenate :class:`AdataDict` back to a single :class:`AnnData`
"""

import gc
import warnings

import anndata as ad

from anndata import AnnData

from .adata_dict import AdataDict
from .adata_dict_fapply import adata_dict_fapply
from .adata_dict_utils import to_nested_list


def concatenate_adata_dict(
    adata_dict: AdataDict,
    *,
    new_col_name: str | None = None,
    lower_peak_memory: bool = False,
    **kwargs,
) -> AnnData:
    """
    Concatenate all AnnData objects in `adata_dict` into a single AnnData object. 
    If only a single AnnData object is present, return it as is.

    Parameters
    ------------
    adata_dict
        :class:`AdataDict`

    new_col_name
        If provided, the name of the new column that will store the ``adata_dict`` 
        key in ``.obs`` of the concatenated AnnData. Defaults to ``None``.

    lower_peak_memory
        If True, the the function concatenates the AnnData objects in `adata_dict` sequentially, 
        and deletes the data from ``adata_dict`` as it does so. This reduces peak memory usage.

    kwargs
        Additional keyword arguments for concatenation.

    Returns
    -------
    A single :class:`AnnData` object. 
    The ``.obs`` will contain a new column specifying the key of the :class:`AnnData` of origin.

    Raises
    ------
    ValueError
        If `adata_dict` is empty.

    Warnings
    --------
    ``lower_peak_memory`` defaults to ``False``. If `lower_peak_memory` is set to ``True``, 
    ``adata_dict`` will be deleted during the concatenation process.

    Notes
    -----
    Memory is only freed if no other references to the :class:`AnnData` objects in ``adata_dict`` exist.

    """
    kwargs.setdefault("join", "outer")
    kwargs.setdefault("index_unique", None)  # Ensure original indices are kept

    # Cache the hierarchy
    cached_hierarchy = adata_dict.hierarchy

    # Flatten the adata_dict
    adata_dict.flatten()

    #Get the keys
    all_keys = list(adata_dict.keys())

    if not all_keys:
        raise ValueError("adata_dict is empty. No data available to concatenate.")

    # add the key to the obs to keep track after merging
    def add_key_to_obs_adata_dict(adata_dict, new_col_name=new_col_name):
        def add_adt_key_to_obs(adata, new_col_name=new_col_name, adt_key=None):
            adata.obs[new_col_name] = [adt_key] * adata.n_obs

        adata_dict_fapply(adata_dict, add_adt_key_to_obs)

    if new_col_name:
        add_key_to_obs_adata_dict(adata_dict)

    if len(all_keys) == 1:
        return adata_dict[all_keys[0]]  # Return the single AnnData object

    if lower_peak_memory:
        warnings.warn("`lower_peak_memory=True`. `adata_dict` will be deleted during merge.")
        # Initialize the merged AnnData with the first one
        first_key = all_keys[0]
        adata = adata_dict[first_key]
        del adata_dict[first_key]
        gc.collect()

        for key in all_keys[1:]:
            adata = ad.concat([adata, adata_dict[key]], **kwargs)
            del adata_dict[key]
            gc.collect()

        return adata

    # If lower_peak_memory=False, just concat at once
    adatas = list(adata_dict.values())
    adata = ad.concat(adatas, **kwargs)

    # Restore the hierarchy
    adata_dict.set_hierarchy(to_nested_list(cached_hierarchy))

    return adata
