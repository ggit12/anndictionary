"""
Utility functions for dictionaries.
"""

def check_dict_structure(
    dict_1: dict,
    dict_2: dict,
    *,
    full_depth: bool = False
) -> bool:
    """
    If ``full_depth`` is ``False`` (default) check that ``dict_2`` contains at least
    the same nested-key structure as ``dict_1`` (extra depth in ``dict_2``
    is ignored).  
    If ``full_depth`` is ``True``, require both dictionaries to match at every
    level.

    Parameters
    ----------
    dict_1
        Template dictionary (sets maximum depth when ``full_depth`` is False).

    dict_2
        Candidate dictionary to compare.

    full_depth
        If ``True``, check that both dictionaries have the same keys and nesting structure at every level.
        If ``False``, check that ``dict_2`` has at least the same keys and nesting structure as ``dict_1``, ignoring extra depth in ``dict_2``.

    Returns
    -------
    ``True`` if both dictionaries have the same keys and nesting structure (ignoring extra depth in ``dict_2`` if ``full_depth`` is ``False``), ``False`` otherwise.
    """

    if not (isinstance(dict_1, dict) and isinstance(dict_2, dict)):
        return False

    def _match(template: dict, candidate: dict) -> bool:
        # Reject extra or missing sibling keys at this depth
        if set(candidate.keys()) != set(template.keys()):
            return False

        for k, v in template.items():
            cand_v = candidate[k]
            if isinstance(v, dict):
                if not isinstance(cand_v, dict):
                    return False
                if not _match(v, cand_v):
                    return False
            else:
                # Template stops here. Extra depth in candidate is OK
                # unless strict checking is requested.
                if full_depth and isinstance(cand_v, dict):
                    return False
        return True

    if full_depth:
        # Two-way comparison
        return _match(dict_1, dict_2) and _match(dict_2, dict_1)

    # One-way comparison
    return _match(dict_1, dict_2)


def all_leaves_are_of_type(
    data,
    target_type
) -> bool:
    """
    Recursively check if all leaves in a nested dictionary or list are of a specific type.

    Parameters
    ----------
    data
        The nested dictionary or list to check.

    target_type
        The type to check for at the leaves.

    Returns
    -------
    ``True`` if all leaves are of the target type, ``False`` otherwise.
    """
    # function only recurses through dicts
    if isinstance(data, dict):
        if not data:  # empty dict fails
            return False
        for value in data.values():
            if not all_leaves_are_of_type(value, target_type):
                return False
        return True

    # once we hit a non-dict, we check the type directly
    return isinstance(data, target_type)
