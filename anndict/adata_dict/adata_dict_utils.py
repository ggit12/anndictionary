"""
utils for :class:`AdataDict`
"""

from __future__ import annotations  # allows type hinting without circular dependency
from typing import TYPE_CHECKING

from .adata_dict_fapply import adata_dict_fapply_return

if TYPE_CHECKING:
    from .adata_dict import AdataDict


def to_nested_tuple(nested_list):
    """
    Recursively convert a nested list into a nested tuple.
    """
    if isinstance(nested_list, list):
        return tuple(to_nested_tuple(item) for item in nested_list)
    return nested_list


def to_nested_list(nested_tuple):
    """
    Recursively convert a nested tuple into a nested list.
    """
    if isinstance(nested_tuple, tuple):
        return list(to_nested_list(item) for item in nested_tuple)
    return nested_tuple


def set_var_index_func(adata_dict: AdataDict, column: str) -> AdataDict:
    """
    Set the index of ``adata.var`` to the specified
    column for each :class:`AnnData` in ``adata_dict``.

    Parameters
    ------------
    adata_dict
        An :class:`AdataDict`.

    column
        The column name to set as the index of ``adata.var``.

    Returns
    --------
    An :class:`AdataDict` with the same structure as
    ``adata_dict``, where the ``.var`` attribute of each :class:`AnnData`
    object has its index set to the specified column.
    """

    def set_var_index_main(adata, column):
        adata.var = adata.var.set_index(column)
        return adata

    return adata_dict_fapply_return(
        adata_dict, set_var_index_main, column=column, return_as_adata_dict=True
    )


def set_obs_index_func(adata_dict: AdataDict, column: str) -> AdataDict:
    """
    Set the index of adata.obs to the specified column for each AnnData object in adata_dict.

    Parameters
    -----------
    adata_dict
        An :class:`AdataDict`.

    column
        The column name to set as the index of ``adata.obs``.

    Returns
    --------
    An :class:`AdataDict` with the same structure as 
    ``adata_dict``, where the ``.obs`` attribute of 
    each :class:`AnnData` object has its index set to the specified column.
    """

    def set_obs_index_main(adata, column):
        adata.obs = adata.obs.set_index(column)
        return adata

    return adata_dict_fapply_return(
        adata_dict, set_obs_index_main, column=column, return_as_adata_dict=True
    )
