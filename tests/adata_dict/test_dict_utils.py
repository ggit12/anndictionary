"""
unit tests for anndict.adata_dict.dict_utils 
"""
# pylint: disable=redefined-outer-name
#disable pylint false positives

from copy import deepcopy

import pytest

from anndict.adata_dict.dict_utils import check_dict_structure, all_leaves_are_of_type

# Base test dictionaries
@pytest.fixture
def dict_1():
    """Base dictionary for testing"""
    return {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        }
    }

@pytest.fixture
def matching_dict():
    """Matching dictionary for testing"""
    return {
        "a": 10,
        "b": {
            "c": 20,
            "d": {
                "e": 30
            }
        }
    }

def test_matching_structure(dict_1, matching_dict):
    """Test when both dictionaries have exactly matching structure"""
    assert check_dict_structure(dict_1, matching_dict, full_depth=True)

def test_extra_key_in_dict1(dict_1):
    """Test when dict_1 has an extra key that dict_2 doesn't have"""
    # create a deep copy (so that modifications to the original don't affect the copy)
    dict_2 = deepcopy(dict_1)

    #add an extra key to original
    dict_1["b"]["extra_key"] = 4

    # shouldn't pass check
    assert not check_dict_structure(dict_1, dict_2, full_depth=True)

def test_extra_key_in_dict2(dict_1, matching_dict):
    """Test when dict_2 has an extra key that dict_1 doesn't have"""

    # copy the original
    dict_2 = matching_dict.copy()

    #add an extra key to the copy
    dict_2["b"]["extra_key"] = 40

    # shouldn't pass check
    assert not check_dict_structure(dict_1, dict_2, full_depth=True)

def test_different_nesting(dict_1):
    """Test when dictionaries have same keys but different nesting structure"""
    # define dict_2 with different nesting, same keys
    dict_2 = {
        "a": 10,
        "b": {
            "c": 20,
            "d": 30,
            "e": 40
        }
    }

    # shouldn't pass check
    assert not check_dict_structure(dict_1, dict_2, full_depth=True)

# def test_dict2_as_nested_dict():
#     """Test when dictionaries have same keys but different nesting structure"""
#     # define dict_2 with different nesting, same keys
#     dict_1  = {
#         "a": 1,
#         "b": {
#             "c": 2,
#             "d": 3
#         }
#     }

#     dict_2 = {
#         "a": {"a1": 10, "a2": 20},
#         "b": {
#             "c": {"c1": 30, "c2": 40},
#             "d": {"d1": 50}
#         }
#     }

#     # shouldn't pass check
#     assert check_dict_structure(dict_1, dict_2, full_depth=False)


#tests for all_leaves_are_of_type
def test_all_leaves_are_integers(dict_1):
    """Test when all leaves are integers"""
    assert all_leaves_are_of_type(dict_1, int)

def test_mixed_types():
    """Test when leaves have mixed types"""
    data = {
        "a": 1,
        "b": "string",
        "c": 3
    }
    assert not all_leaves_are_of_type(data, int)

    data = {
        "a": 1,
        "b": 2,
        "c": [3, 4]
    }
    assert not all_leaves_are_of_type(data, int)

def test_empty_structures():
    """Test with empty structures"""
    assert not all_leaves_are_of_type({}, int)
    assert not all_leaves_are_of_type([], int)
    assert not all_leaves_are_of_type((), int)

def test_single_value():
    """Test with non-nested single values"""
    assert all_leaves_are_of_type(42, int)
    assert not all_leaves_are_of_type("string", int)

def test_nested_lists():
    """Test with a list"""
    data = [1, [2, 3], [4, [5, 6]]]
    assert not all_leaves_are_of_type(data, int)
