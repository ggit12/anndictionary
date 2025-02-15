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
    check_and_create_stratifier,
)

def subsample_adata_dict(
    adata_dict: AdataDict,
    **kwargs
) -> None:
    """
    Subsamples each :class:`AnnData` in ``adata_dict`` using :func:`sc.pp.subsample`.
    
    Parameters
    -----------
    adata_dict
        An :class:`AdataDict`.

    kwargs
        Additional keyword arguments to pass to :func:`sc.pp.subsample`

    Returns
    --------
    None

    Notes
    -----
    The function modifies ``adata_dict`` in-place.
    """
    n_obs = kwargs.get('n_obs', None)
    fraction = kwargs.get('fraction', None)

    if n_obs is None and fraction is None:
        fraction = 1
        kwargs['fraction'] = fraction

    def subsample_adata(adata, **kwargs):
        if n_obs is None or adata.n_obs > n_obs:
            sc.pp.subsample(adata, **kwargs)

    adata_dict_fapply(adata_dict, subsample_adata, **kwargs)


def resample_adata(
    adata: AnnData,
    strata_keys: list[str],
    min_num_cells: int,
    n_largest_groups: int | None = None,
    **kwargs
) -> AnnData:
    """
    Resample ``adata`` based on specified strata keys and 
    drop strata with fewer than the minimum number of cells.

    Parameters
    -----------
    adata
        An :class:`AnnData`.

    strata_keys
        List of column names in adata.obs to use for stratification.

    min_num_cells
        Minimum number of cells required to retain a stratum.

    kwargs
        Additional keyword arguments to pass to the subsample function.

    Returns
    --------
    Concatenated :class:`AnnData` object after resampling and filtering.

    Raises
    --------
    ValueError
        If any of the specified ``strata_keys`` do not exist in ``adata.obs``.
    """
    # Step 1: Create the strata key
    strata_key = check_and_create_stratifier(adata, strata_keys)

    # Step 2: Calculate the size of each category
    category_counts = adata.obs[strata_key].value_counts()

    # Step 3: Identify the top n largest categories or all categories if n is None
    if n_largest_groups is None:
        selected_categories = category_counts.index.tolist()
    else:
        selected_categories = category_counts.nlargest(n_largest_groups).index.tolist()

    # Step 4: Build adata_dict based on the strata key
    strata_dict = build_adata_dict(adata, [strata_key], desired_strata=selected_categories)

    # Step 5: Subsample each AnnData object in the strata_dict
    subsample_adata_dict(strata_dict, **kwargs)

    # Step 6: Drop AnnData objects with fewer than min_num_cells
    filtered_dict = {k: v for k, v in strata_dict.items() if v.n_obs >= min_num_cells}

    # Step 7: Concatenate the filtered_dict back to a single AnnData object
    #setting index_unique=None avoids index modification
    return concatenate_adata_dict(filtered_dict, index_unique=None)
