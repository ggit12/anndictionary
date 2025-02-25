"""
Unit tests for utils for :class:`AdataDict`
"""

import pytest

from anndict.adata_dict import AdataDict
from anndict.adata_dict import (
    to_nested_tuple,
    to_nested_list,
    set_var_index_func,
    set_obs_index_func,
)

# Fixtures for nested list/tuple tests remain the same
@pytest.fixture
def simple_nested_list():
    return [1, [2, 3], [4, [5, 6]]]


@pytest.fixture
def simple_nested_tuple():
    return (1, (2, 3), (4, (5, 6)))


# Tests for nested conversion functions remain the same
def test_to_nested_tuple_simple(simple_nested_list):
    result = to_nested_tuple(simple_nested_list)
    assert isinstance(result, tuple)
    assert result == (1, (2, 3), (4, (5, 6)))


def test_to_nested_list_simple(simple_nested_tuple):
    result = to_nested_list(simple_nested_tuple)
    assert isinstance(result, list)
    assert result == [1, [2, 3], [4, [5, 6]]]


def test_set_var_index_single_column(simple_adata_dict):

    # Call the function to set the .var index with a single column ("gene_id")
    result = set_var_index_func(simple_adata_dict, "gene_id")

    # Verify that the var index name and values are updated for sample1
    assert result[("sample1",)].var.index.name == "gene_id"
    assert list(result[("sample1",)].var.index) == ["gene1", "gene2"]

    # Verify that the var index name and values are updated for sample2
    assert result[("sample2",)].var.index.name == "gene_id"
    assert list(result[("sample2",)].var.index) == ["gene1", "gene2"]


def test_set_var_index_invalid_column(simple_adata_dict):
    # Call the function
    result = set_var_index_func(simple_adata_dict, "nonexistent_column")

    # Verify that result is actually an Exception object
    assert isinstance(result[('sample1',)], KeyError), "Expected an exception object to be returned."


def test_set_obs_index_single_column(simple_adata_dict):

    # Call the function to set the .obs index with a single column ("cell_id")
    result = set_obs_index_func(simple_adata_dict, "cell_id")

    # Verify that the obs index name and values are updated for sample1
    assert result[("sample1",)].obs.index.name == "cell_id"
    assert list(result[("sample1",)].obs.index) == ["cell1", "cell2"]

    # Verify that the obs index name and values are updated for sample2
    assert result[("sample2",)].obs.index.name == "cell_id"
    assert list(result[("sample2",)].obs.index) == ["cell3", "cell4"]


def test_set_obs_index_invalid_column(simple_adata_dict):
    # Call the function
    result = set_obs_index_func(simple_adata_dict, "nonexistent_column")

    # Verify that result is actually an Exception object
    assert isinstance(result[('sample1',)], KeyError), "Expected an exception object to be returned."

def test_empty_adata_dict():

    # Create an empty AdataDict and attempt to set the var index
    empty_dict = AdataDict()
    result = set_var_index_func(empty_dict, "gene_id")

    # Verify the function returns an AdataDict and it remains empty
    assert isinstance(result, AdataDict)
    assert len(result) == 0
