"""
This module contains the function to add a stratification to an existing :class:`AdataDict`.
"""

from __future__ import annotations #allows type hinting without circular dependency

from typing import TYPE_CHECKING

from .adata_dict_utils import to_nested_list, to_nested_tuple
from .adata_dict_fapply import adata_dict_fapply_return
from .build import build_adata_dict

if TYPE_CHECKING:
    from .adata_dict import AdataDict


def add_stratification(
    adata_dict: AdataDict,
    strata_keys: list[str],
    *,
    desired_strata: list | dict | None = None,
) -> AdataDict:
    """
    Split each value of an AnnData dictionary into further subsets based on additional desired strata.

    Parameters
    ------------
    adata_dict
        An :class:`AdataDict`

    strata_keys
        List of column names in `adata.obs` to use for further stratification.

    desired_strata
        List of desired strata values or a dictionary where keys are strata keys and values 
        are lists of desired strata values.

    Returns
    -------
    Nested :class:`AdataDict`, where the top-level is now stratified by ``strata_keys``.

    Raises
    ------
    ValueError
        If any of the `strata_keys` are already in the hierarchy.

    Examples
    --------
    Case 1: Build by Donor first, then add Tissue stratification after

    .. code-block:: python

        import pandas as pd
        import anndict as adt
        from anndata import AnnData
        # Create an example AnnData object
        adata = AnnData(obs=pd.DataFrame({
        >     "Donor": ["Donor1", "Donor1", "Donor2"],
        >     "Tissue": ["Tissue1", "Tissue2", "Tissue1"]
        > }))
        # First, build an AdataDict grouped/stratified by Donor
        strata_keys = ["Donor"]
        adata_dict = adt.build_adata_dict(adata, strata_keys)
        print(adata_dict)
        > {
        >     ("Donor1",): adata_d1,
        >     ("Donor2",): adata_d2,
        > }
        # Then, add a stratification by ``Tissue``
        strata_keys = ["Tissue"]
        adata_dict.add_stratification(strata_keys)
        print(adata_dict)
        > {
        >     ("Tissue1",) : {
        >         ("Donor1",) : adata_d1_t1,
        >         ("Donor2",) : adata_d2_t1
        >         },
        >     ("Tissue2",) : {
        >         ("Donor1",) : adata_d1_t2
        >         }
        > }
        # Note 1 If you wanted a new object instead of modifying the original ``adata_dict``, you can instead do:
        new_adata_dict = adt.add_stratification(adata_dict, strata_keys)

        # Note 2: we can always flatten or rearrange the nesting structure
        adata_dict.set_hierarchy(["Donor","Tissue"])
        print(adata_dict)
        > {
        >     ("Donor1", "Tissue1"): adata_d1_t1,
        >     ("Donor1", "Tissue2"): adata_d1_t2,
        >     ("Donor2", "Tissue1"): adata_d2_t1,
        > }
        # For example, if we want Donor as the top-level index
        adata_dict.set_hierarchy(["Donor",["Tissue"]])
        >             {
        >     ("Donor1",) : {
        >         ("Tissue1",) : adata_d1_t1,
        >         ("Tissue2",) : adata_d1_t2
        >         },
        >     ("Donor2",) : {
        >         ("Tissue1",) : adata_d2_t1
        >         }
        > }
    """

    # Get the hierarchy and check for redundant stratification
    cached_hierarchy = adata_dict.hierarchy

    for strata_key in strata_keys:
        if strata_key in adata_dict.flatten_nesting_list(to_nested_list(cached_hierarchy)):
            raise ValueError(f"adata_dict is already stratified by {strata_key}")

    # Flatten the adata_dict
    adata_dict.flatten()
    flat_hierarchy = to_nested_list(adata_dict.hierarchy)


    # Split the adata_dict
    adata_dict = adata_dict_fapply_return(adata_dict, build_adata_dict,
                                         strata_keys=strata_keys,
                                         desired_strata=desired_strata,
                                         return_as_adata_dict=True)

    # directly set the hierarchy of the updated adata_dict
    adata_dict._hierarchy = to_nested_tuple([*flat_hierarchy, [strata_keys]]) # pylint: disable=protected-access

    # Create new hierarchy with strata_keys at top level
    new_hierarchy = [*strata_keys, to_nested_list(cached_hierarchy)]

    # Set the new hierarchy
    adata_dict.set_hierarchy(new_hierarchy)

    return adata_dict
