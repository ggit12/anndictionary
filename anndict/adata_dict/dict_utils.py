"""
Utility functions for dictionaries.
"""

def check_dict_structure(
    dict_1: dict,
    dict_2: dict
) -> bool:
    """
    Compare two nested dictionaries to see if their keys and nesting structure match.

    Parameters
    ----------
    dict_1 : dict
        First dictionary to compare.

    dict_2 : dict
        Second dictionary to compare.

    Returns
    -------
    True if both dictionaries have the same keys and nesting structure, False otherwise.
    """
    def compare_nested_keys(d1, d2):
        """Recursively compare keys in nested dictionaries, respecting nesting structure."""
        for key in d1:
            if key not in d2:
                return False
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                if not compare_nested_keys(d1[key], d2[key]):
                    return False
            elif isinstance(d1[key], dict) != isinstance(d2[key], dict):
                return False
        return True

    return compare_nested_keys(dict_1, dict_2) and compare_nested_keys(dict_2, dict_1)


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
