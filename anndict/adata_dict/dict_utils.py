"""
Utility functions for dictionaries.
"""

def check_dict_structure(
    dict_1: dict,
    dict_2: dict,
    *,
    exact: bool = False,
    max_depth: int | None = None,
) -> bool:
    """
    Check whether two dicts share the same nested-key structure.

    When ``exact`` is ``True`` (default), keys and nesting depth must match
    in both directions.  When ``exact`` is ``False``, the check is one-way:
    wherever ``dict_1`` has a dict value, ``dict_2`` must also have a dict
    with the same keys; wherever ``dict_1`` has a leaf, any value in
    ``dict_2`` is accepted (extra depth in ``dict_2`` is OK).

    ``max_depth`` independently limits how many nesting levels are compared.
    Below that limit, dict values in ``dict_1`` that are dicts are treated
    as leaves (deeper structure is ignored).

    Parameters
    ----------
    dict_1
        Template dictionary.

    dict_2
        Candidate dictionary to compare.

    exact
        If ``True``, require keys and nesting to match in both directions.
        If ``False``, only require ``dict_2`` to match ``dict_1``'s
        structure; extra depth in ``dict_2`` is ignored.

    max_depth
        If not ``None``, limit the comparison to this many levels of nesting.
        At the depth limit, deeper structure in the template is ignored.

    Returns
    -------
    ``True`` if the structures match under the rules described above,
    ``False`` otherwise.
    """

    if not (isinstance(dict_1, dict) and isinstance(dict_2, dict)):
        return False

    def _match(template: dict, candidate: dict, depth: int = 1) -> bool:
        if set(candidate.keys()) != set(template.keys()):
            return False

        at_limit = max_depth is not None and depth >= max_depth

        for k, v in template.items():
            cand_v = candidate[k]
            if isinstance(v, dict) and not at_limit:
                if not isinstance(cand_v, dict):
                    return False
                if not _match(v, cand_v, depth + 1):
                    return False
        return True

    if exact:
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
