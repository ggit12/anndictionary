"""
This module contains the adata_dict_fapply family of functions, the core functions of Anndictionary.
"""
from __future__ import annotations #allows type hinting without circular dependency

import inspect

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import TYPE_CHECKING
from typing import Any

from .dict_utils import check_dict_structure

if TYPE_CHECKING:
    from anndata import AnnData
    from .adata_dict import AdataDict


def get_depth(
    max_depth: int | str | list[str] | tuple[str, ...] | None,
    adata_dict: "AdataDict" | dict,
) -> int | None:
    """
    Resolve ``max_depth`` into an integer depth.

    Parameters
    ------------
    max_depth
        The depth at which to stop recursing and apply ``func``.
        - ``None`` (default): recurse to leaves.
        - ``0``: apply to the full input ``adata_dict``.
        - ``1``: apply to each top-level value in ``adata_dict``.
        - ``int``: stop at that integer depth (0-indexed from root).
        - ``str`` / ``list[str]`` / ``tuple[str, ...]``: stop at the depth matching the name(s) in ``adata_dict.hierarchy``.    

    adata_dict
        An :class:`AdataDict`.

    Returns
    -------
    The integer depth, or ``None`` if ``max_depth`` is ``None``.
    """
    if max_depth is None:
        return None

    if isinstance(max_depth, int):
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        return max_depth

    if not hasattr(adata_dict, "hierarchy") or not adata_dict.hierarchy:
        raise ValueError(
            "Hierarchy not set so can't use semantic depth; use integer max_depth or recreate the adata_dict with build_adata_dict() and try again."
        )

    target_levels = {max_depth} if isinstance(max_depth, str) else set(max_depth)

    def _get_levels_local(nesting, levels=None, depth=0):
        if levels is None:
            levels = []
        if len(levels) <= depth:
            levels.append([])
        for item in nesting:
            if isinstance(item, (list, tuple)):
                _get_levels_local(item, levels, depth + 1)
            else:
                levels[depth].append(item)
        return levels

    levels = _get_levels_local(adata_dict.hierarchy)
    for depth, level_items in enumerate(levels):
        if not target_levels.isdisjoint(level_items):
            # hierarchy depth 0 corresponds to top-level keys (max_depth=1). Root is max_depth=0.
            return depth + 1

    raise ValueError(f"Level(s) {max_depth} not found in hierarchy.")

# This is intentional and necessary for error handling
# pylint: disable=inconsistent-return-statements
def apply_func(
    adt_key: tuple[str, ...],
    adata: AnnData,
    func: callable,
    accepts_key: bool,
    max_retries: int,
    catch_errors: bool,
    **func_args
) -> Any | None:
    """
    Applies a function to adata with retries (max_retries many times), optionally passing adt_key.
    Returns whatever the applied function returns.
    
    Parameters
    ------------
    adt_key
        Key to pass to the function if accepts_key is True

    adata
        Data to pass to the function

    func
        Function to apply

    accepts_key
        Whether the function accepts an adt_key parameter

    max_retries
        Maximum number of retry attempts after the first failure.
        (i.e. 0 means no retries, 1 means one retry after the first failure, for a total of 2 attempts)

    catch_errors
        If ``False``, raise exceptions instead of catching errors.

    **func_args
        Additional arguments to pass to func
        
    Returns
    -------
    The return value of func, or
    the last exception object on failure (after ``max_retries``, if ``catch_errors`` is ``True``)
    
    Raises
    ------
    Exception: If ``catch_errors`` is ``False`` and ``func`` raises an exception

    Notes
    -----
    Be careful when setting ``max_retries != 0``. If retries are used, 
    the same function will be called again on `whatever the current data 
    is`, which may not be the original data. For example, if ``func`` 
    log-transform the data, then does something that fails, the second 
    time ``func`` is called, the already log-transformed data will have 
    another log transform applied.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(adata, adt_key=adt_key, **func_args) if accepts_key else func(adata, **func_args)

        except Exception as e: #pylint: disable=broad-except
            if not catch_errors:
                raise
            # print(f"Error processing {adt_key} on attempt {attempt}: {e}")
            if attempt >= max_retries:
                print(f"Failed to process {adt_key} after {max_retries} attempts: {e}")
                return e

def adata_dict_fapply(
    adata_dict: AdataDict,
    func: callable,
    *,
    use_multithreading: bool = True,
    num_workers: int | None = None,
    max_retries: int = 0,
    catch_errors: bool = True,
    return_as_adata_dict: bool = False,
    max_depth: int | str | list[str] | tuple[str, ...] | None = None,
    **kwargs_dicts: Any,
) -> dict | AdataDict | None:
    """
    Applies ``func`` to each :class:`AnnData` in ``adata_dict``, with error handling,
    retry mechanism, and the option to use either threading or sequential execution. The 
    return behaviour is based on the return behaviour of ``func``.

    ``kwargs`` can be :class:`Any`. if a ``kwarg`` is a :class:`dict` 
    with keys that match ``adata_dict``, then values are broadcast to 
    the ``func`` call on the corresponding key-value of ``adata_dict``. Otherwise, 
    the kwarg is directly broadcast to all calls of ``func``.

    Parameters
    ------------
    adata_dict
        An :class:`AdataDict`.  

    func
        Function to apply to each :class:`AnnData` object in ``adata_dict``.

    use_multithreading
        If True, use ``ThreadPoolExecutor``; if False, execute sequentially. Default is True.

    num_workers
        Number of worker threads to use. If :obj:`None`, defaults to the number of CPUs available.

    max_retries
        Maximum number of retry attempts after the first failure.
        (i.e. 0 means no retries, 1 means one retry after the first failure, for a total of 2 attempts)

    catch_errors
        If ``False``, raise exceptions instead of catching errors.

    return_as_adata_dict
        Whether to return the results as a :class:`dict` (if ``False``) 
        or :class:`AdataDict` (if ``True``).

    max_depth
        The depth at which to stop recursing and apply ``func``.
        - ``None`` (default): recurse to leaves.
        - ``0``: apply to the full input ``adata_dict``.
        - ``1``: apply to each top-level value in ``adata_dict``.
        - ``int``: stop at that integer depth (0-indexed from root).
        - ``str`` / ``list[str]`` / ``tuple[str, ...]``: stop at the depth matching the name(s) in ``adata_dict.hierarchy``.

    kwargs_dicts
        Additional keyword arguments to pass to the function.

    Returns
    -------
    - If all results are None: None
    - Otherwise: A :class:`dict` or :class:`AdataDict` (based on return_as_adata_dict)
      containing all results of ``func`` applied to each AnnData

    Notes
    -----
    Be careful when setting ``max_retries != 0``. If retries are used, 
    the same function will be called again on `whatever the current data 
    is`, which may not be the original data. For example, if ``func`` 
    log-transform the data, then does something that fails, the second 
    time ``func`` is called, the already log-transformed data will have 
    another log transform applied.
    """
    from .adata_dict import AdataDict  # pylint: disable=import-outside-toplevel

    sig = inspect.signature(func)
    accepts_key = "adt_key" in sig.parameters
    results = {}
    has_non_none_result = False

    max_depth = get_depth(max_depth, adata_dict)
    if max_depth == 0:
        return apply_func(
            (),
            adata_dict,
            func,
            accepts_key,
            max_retries,
            catch_errors,
            **kwargs_dicts,
        )

    if return_as_adata_dict:
        hierarchy = adata_dict.hierarchy if hasattr(adata_dict, "hierarchy") else ()

    # Separate kwargs into those that match the structure of adata_dict
    # vs. those that do not. For matching ones, pass kwarg_value[adt_key];
    # otherwise, pass kwarg_value as-is for every key.
    matching_kwargs = {}
    non_matching_kwargs = {}
    for arg_name, arg_value in kwargs_dicts.items():
        if isinstance(arg_value, (dict, AdataDict)) and check_dict_structure(adata_dict, arg_value, max_depth=max_depth):
            matching_kwargs[arg_name] = arg_value
        else:
            non_matching_kwargs[arg_name] = arg_value

    def process_item(adt_key, adata, current_depth: int, matching_kwarg_subtrees: dict[str, Any]):
        nonlocal has_non_none_result
        if max_depth is not None and current_depth >= max_depth:
            pass
        elif isinstance(adata, (dict, AdataDict)):
            nested_results = {}
            for nested_key, nested_adata in adata.items():
                child_matching = {}
                for arg_name, subtree in matching_kwarg_subtrees.items():
                    if isinstance(subtree, (dict, AdataDict)):
                        child_matching[arg_name] = subtree[nested_key]
                    else:
                        child_matching[arg_name] = subtree
                nested_results[nested_key] = process_item(
                    nested_key,
                    nested_adata,
                    current_depth + 1,
                    child_matching,
                )
            return AdataDict(nested_results) if return_as_adata_dict else nested_results

        # is an AnnData leaf OR we hit max_depth and 'adata' is a dict/AdataDict
        # Build the kwargs for this specific adt_key
        current_kwargs = {}
        current_kwargs.update(non_matching_kwargs)
        # For matching kwargs, we track the aligned subtree as we recurse (so nested dict keys
        # don't need to be flattened into a global key path).
        current_kwargs.update(matching_kwarg_subtrees)

        result = apply_func(
            adt_key,
            adata,
            func,
            accepts_key,
            max_retries,
            catch_errors,
            **current_kwargs
        )
        if result is not None:
            has_non_none_result = True
        return result

    if use_multithreading:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for adt_key, adata in adata_dict.items():
                top_matching = {k: v[adt_key] for k, v in matching_kwargs.items()}
                if isinstance(adata, (dict, AdataDict)):
                    # Handle nested structures sequentially
                    results[adt_key] = process_item(adt_key, adata, 1, top_matching)
                else:
                    # Only use threading for leaf nodes (actual AnnData objects)
                    futures[executor.submit(
                        process_item,
                        adt_key,
                        adata,
                        1,
                        top_matching,
                    )] = adt_key

            for future in as_completed(futures):
                adt_key = futures[future]
                results[adt_key] = future.result()

    else:
        for adt_key, adata in adata_dict.items():
            top_matching = {k: v[adt_key] for k, v in matching_kwargs.items()}
            results[adt_key] = process_item(adt_key, adata, 1, top_matching)

    # If requested, return as an AdataDict
    if return_as_adata_dict:
        return AdataDict(results, hierarchy)

    # If all results were None, return None
    if not has_non_none_result:
        return None

    # Else, return as a regular dict
    return results


def adata_dict_fapply_return(
    adata_dict: AdataDict,
    func: callable,
    *,
    use_multithreading: bool = True,
    num_workers: int | None = None,
    max_retries: int = 0,
    catch_errors: bool = True,
    return_as_adata_dict: bool = False,
    max_depth: int | str | list[str] | tuple[str, ...] | None = None,
    **kwargs_dicts: Any,
) -> dict | AdataDict | None:
    """Completely deprecated. Use adata_dict_fapply instead. Stub left for compatibility."""
    raise RuntimeError(
    "fapply_return behaviour has been replaced by adata_dict_fapply(). "
    "Replace this call with adata_dict_fapply(...) directly; "
    "no other code changes are needed.")
