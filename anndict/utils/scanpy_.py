"""
Utility functions for :mod:`scanpy`.
"""

import scanpy as sc

from anndata import AnnData

from anndict.adata_dict import (
    AdataDict,
    adata_dict_fapply,
    build_adata_dict,
    concatenate_adata_dict,
)

def sample_adata_dict(
    adata_dict: AdataDict,
    **kwargs
) -> None:
    """
    Samples each :class:`AnnData` in ``adata_dict`` using :func:`sc.pp.subsample`.
    
    Parameters
    -----------
    adata_dict
        An :class:`AdataDict`.

    kwargs
        Additional keyword arguments to pass to :func:`sc.pp.sample`

    Returns
    --------
    ``None`` and modifies ``adata_dict`` in-place if ``copy`` is ``False`` (default). 
    If ``copy`` is ``True``, returns a new :class:`AdataDict`.

    @todo switch to sc.pp.sample when scanpy required version is upgraded past 1.11.0
    """
    n_obs = kwargs.get('n_obs', None)
    fraction = kwargs.get('fraction', None)

    if n_obs is None and fraction is None:
        fraction = 1
        kwargs['fraction'] = fraction

    def sample_adata(adata, **kwargs):
        if n_obs is None or adata.n_obs > n_obs:
            sc.pp.subsample(adata, **kwargs)

    return adata_dict_fapply(adata_dict, sample_adata, return_as_adata_dict=True, **kwargs)


def sample_and_drop(
    adata: AnnData,
    strata_keys: list[str] | str,
    min_num_cells: int = 0,
    n_largest_groups: int | None = None,
    **kwargs
) -> AnnData:
    """
    Sample ``adata`` based on specified strata keys and 
    drop strata with fewer than the ``min_num_cells``. Can
    optionally retain only the ``n_largest_groups``.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    strata_keys
        List of column names in adata.obs to use for stratification.

    min_num_cells
        Minimum number of cells required to retain a stratum.

    n_largest_groups
        If specified, keep only the ``n_largest_groups``.

    kwargs
        Additional keyword arguments passed to :func:`sample_adata` and :func:`sc.pp.subsample`.

    Returns
    --------
    Concatenated :class:`AnnData` object after resampling and filtering.

    Raises
    --------
    ValueError
        If any of the specified ``strata_keys`` do not exist in ``adata.obs``.

    Notes
    -----
    In the case of ties when selecting the largest groups, all tied groups 
    are kept. So you may end up with more than ``n_largest_groups``.
    """
    # Input handling
    if isinstance(strata_keys, str):
        strata_keys = [strata_keys]

    # Step 1: build adata_dict based on the strata keys
    # Note: this will be a flat AdataDict
    adata_dict = build_adata_dict(adata, strata_keys=strata_keys)

    # Step 2: Sample each AnnData object in the adata_dict
    sample_adata_dict(adata_dict, **kwargs)

    # Step 3: get number of cells in each group
    def get_n_obs(adata):
        return adata.n_obs

    n_obs_dict = adata_dict.fapply(get_n_obs)

    # Step 4: get keys that are above min_num_cells, sorted by number of cells:
    keys_sorted = sorted({k: v for k, v in n_obs_dict.items() if v >= min_num_cells},
                    key=n_obs_dict.get)

    # Step 5: if n_largest_groups is specified, keep only the largest groups
    # In the case of ties, all tied groups are kept
    if n_largest_groups is not None:
        idx = max(0, len(keys_sorted) - n_largest_groups)
        threshold = n_obs_dict[keys_sorted[idx]]
        keys_sorted = [k for k in keys_sorted if n_obs_dict[k] >= threshold]

    # Make the final mask
    def check_membership(adata, keys, adt_key=None): # pylint: disable=unused-argument
        return adt_key in keys
    mask_dict = adata_dict.fapply(check_membership, keys=keys_sorted)

    # Filter based on the mask
    adata_dict.index_bool(mask_dict, inplace=True)

    # Step 7: Concatenate back to a single AnnData object
    #setting index_unique=None avoids index modification
    return concatenate_adata_dict(adata_dict, index_unique=None)
