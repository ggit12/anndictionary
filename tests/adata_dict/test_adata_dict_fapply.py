"""
Unit tests for the adata_dict_fapply module.
"""
# pylint: disable=unused-argument
#disable pylint false positives

import threading

import pytest
import numpy as np

from anndict.adata_dict import (
    AdataDict,
    adata_dict_fapply,
)

from anndict.adata_dict.adata_dict_fapply import (
    apply_func,
    get_depth,
)

from tests.conftest import create_test_function, create_key_aware_function

# Tests for apply_func
def test_apply_func_basic(simple_adata):
    """Test basic function application."""
    apply_func('test', simple_adata, create_test_function(), False, 0, True)
    assert simple_adata.uns['modified'] is True

def test_apply_func_with_key(simple_adata):
    """Test function application with key awareness."""
    apply_func('test_key', simple_adata, create_key_aware_function(), True, 0, True)
    assert simple_adata.uns['key'] == 'test_key'

def test_apply_func_retry(simple_adata):
    """Test retry mechanism."""
    attempts = []
    def retry_func(adata, **kwargs):
        attempts.append(1)
        if len(attempts) < 2:
            raise ValueError("First attempt fails")
        adata.uns['modified'] = True

    apply_func('test', simple_adata, retry_func, False, 2, True)
    assert simple_adata.uns['modified'] is True
    assert len(attempts) == 2

# Tests for apply_func_return
def test_apply_func_return_basic(simple_adata):
    """Test basic function application with return value."""
    def return_true(adata, **kwargs):
        adata.uns['modified'] = True
        return True

    result = apply_func('test', simple_adata, return_true, False, 0, True)
    assert simple_adata.uns['modified'] is True
    assert result is True

def test_apply_func_return_with_key(simple_adata):
    """Test function application with key awareness and return value."""
    result = apply_func('test_key', simple_adata, create_key_aware_function(), True, 0, True)
    assert simple_adata.uns['key'] == 'test_key'
    assert result == 'test_key'

def test_apply_func_return_retry(simple_adata):
    """Test retry mechanism with return value."""
    attempts = []
    def retry_func(adata, **kwargs):
        attempts.append(1)
        if len(attempts) < 2:
            raise ValueError("First attempt fails")
        adata.uns['modified'] = True
        return 'success'

    result = apply_func('test', simple_adata, retry_func, False, 2, True)
    assert simple_adata.uns['modified'] is True
    assert len(attempts) == 2
    assert result == 'success'

def test_apply_func_return_error(simple_adata):
    """Test error handling with return value."""
    def error_func(adata, **kwargs):
        raise ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        apply_func('test', simple_adata, error_func, False, 0, False)

def test_apply_func_return_max_retries(simple_adata):
    """Test retry mechanism reaches max attempts and fails."""
    attempts = []
    def always_fail_func(adata, **kwargs):
        attempts.append(1)
        raise ValueError("This function always fails")

    apply_func('test', simple_adata, always_fail_func, False, 3, True)

    assert len(attempts) == 4  # Verify it tried exactly 4 times (one initial try, 3 retries (the max_retries value))

def test_retry_accumulates_changes(simple_adata):
    """Confirm that retries operate on modified data from previous attempts."""
    def add_then_fail(adata, **kwargs):
        # Add 1 to all values in the data
        adata.X = adata.X + 1
        # Then fail
        raise ValueError("Failing after adding 1")

    # Create a copy of the original data to compare later
    original_values = simple_adata.X.copy()

    # Try to run with 2 retries (3 total attempts)
    apply_func('test', simple_adata, add_then_fail, False, 2, True)

    # After 3 attempts (original + 2 retries), each value should be 3 higher
    # than the original because each attempt added 1
    assert np.allclose(simple_adata.X, original_values + 3)

# Tests for adata_dict_fapply
def test_fapply_sequential(simple_adata_dict):
    """Test sequential processing."""
    adata_dict_fapply(simple_adata_dict, create_test_function(), use_multithreading=False)
    assert all(adata.uns.get('modified', False) for adata in simple_adata_dict.values())

def test_fapply_threaded(simple_adata_dict):
    """Test threaded processing."""
    adata_dict_fapply(simple_adata_dict, create_test_function(), use_multithreading=True, num_workers=2)
    assert all(adata.uns.get('modified', False) for adata in simple_adata_dict.values())

def test_fapply_nested(nested_adata_dict):
    """Test processing of nested AdataDict."""
    adata_dict_fapply(nested_adata_dict, create_test_function())
    for group in nested_adata_dict.values():
        assert all(adata.uns.get('modified', False) for adata in group.values())

def test_fapply_with_kwargs_simple(simple_adata_dict):
    """Test kwargs broadcasting."""
    kwarg1 = 'value1'

    def func_with_kwargs(adata, kwarg1=None):
        adata.uns['kwarg1'] = kwarg1

    adata_dict_fapply(simple_adata_dict, func_with_kwargs, kwarg1=kwarg1)
    assert simple_adata_dict[('sample1',)].uns['kwarg1'] == 'value1'
    assert simple_adata_dict[('sample2',)].uns['kwarg1'] == 'value1'

def test_fapply_with_kwargs_broadcasting(simple_adata_dict):
    """Test kwargs broadcasting."""
    kwarg1 = {('sample1',): 'value1', ('sample2',): 'value2'}

    def func_with_kwargs(adata, kwarg1=None):
        adata.uns['kwarg1'] = kwarg1

    adata_dict_fapply(simple_adata_dict, func_with_kwargs, kwarg1=kwarg1)
    assert simple_adata_dict[('sample1',)].uns['kwarg1'] == 'value1'
    assert simple_adata_dict[('sample2',)].uns['kwarg1'] == 'value2'

def test_fapply_with_kwargs_broadcasting_deeper_subtree(simple_adata_dict):
    """
    Test kwargs broadcasting when a kwarg dict is deeper than adata_dict.

    If the kwarg dict matches adata_dict's structure up to some depth, then the entire
    kwarg subtree at that depth should be passed to ``func`` as the kwarg value.
    """
    kwarg1 = {
        ('sample1',): {'subtree': {'x': 1, 'y': [1, 2, 3]}},
        ('sample2',): {'subtree': {'x': 2}},
    }

    def func_with_kwargs(adata, kwarg1=None):
        adata.uns['kwarg1'] = kwarg1

    adata_dict_fapply(simple_adata_dict, func_with_kwargs, kwarg1=kwarg1)
    assert simple_adata_dict[('sample1',)].uns['kwarg1'] == {'subtree': {'x': 1, 'y': [1, 2, 3]}}
    assert simple_adata_dict[('sample2',)].uns['kwarg1'] == {'subtree': {'x': 2}}

def test_fapply_with_kwargs_broadcasting_nested_2lvl_kwarg_3lvl(nested_adata_dict):
    """
    Test kwargs broadcasting where ``adata_dict`` has 2 levels and a kwarg dict has 3.

    Here we apply ``func`` at depth=1 (group-level), so the kwarg dict matches up to the
    group key, and the entire subtree under that group (sample -> extra level) should be
    passed through as the kwarg value.
    """
    kwarg1 = {
        ('group1',): {
            ('sample1',): {'payload': {'x': 1}},
            ('sample2',): {'payload': {'x': 2, 'y': [1, 2]}},
        },
        ('group2',): {
            ('sample3',): {'payload': {'x': 3}},
            ('sample4',): {'payload': {'x': 4}},
        },
    }

    def return_kwarg_subtree(obj, kwarg1=None):
        return kwarg1

    out = adata_dict_fapply(
        nested_adata_dict,
        return_kwarg_subtree,
        use_multithreading=False,
        max_depth=1,
        kwarg1=kwarg1,
    )
    assert out[('group1',)] == kwarg1[('group1',)]
    assert out[('group2',)] == kwarg1[('group2',)]

def test_fapply_with_kwargs_broadcasting_2lvl_template_3lvl_kwarg_apply_at_leaf():
    adata_dict = {
        ('group1',): {('sub1',): object()},
        ('group2',): {('sub1',): object()},
    }

    kwarg1 = {
        ('group1',): {('sub1',): {'payload': 1}},
        ('group2',): {('sub1',): {'payload': 2}},
    }

    def f(obj, kwarg1=None):
        return kwarg1

    out = adata_dict_fapply(adata_dict, f, use_multithreading=False, kwarg1=kwarg1)
    assert out[('group1',)][('sub1',)] == {'payload': 1}
    assert out[('group2',)][('sub1',)] == {'payload': 2}

def test_fapply_obs_modification(simple_adata_dict):
    """Test modification of obs dataframe."""
    adata_dict_fapply(simple_adata_dict, create_test_function('obs'))
    assert all('test' in adata.obs.columns for adata in simple_adata_dict.values())

def test_fapply_X_modification(simple_adata_dict):
    """Test modification of X matrix."""
    adata_dict_fapply(simple_adata_dict, create_test_function('X'))
    assert all(adata.X[0, 0] == 0 for adata in simple_adata_dict.values())

def test_fapply_raises_error(simple_adata_dict):
    """Test error handling."""
    def error_func(adata):
        raise ValueError("Intentional error")

    with pytest.raises(ValueError, match="Intentional error"):
        adata_dict_fapply(simple_adata_dict, error_func, catch_errors=False)

# Tests for adata_dict_fapply_return
def test_fapply_return_sequential(simple_adata_dict):
    """Test sequential processing with return values."""
    results = adata_dict_fapply(
        simple_adata_dict,
        create_key_aware_function(),
        use_multithreading=False
    )
    assert results[('sample1',)] == ('sample1',)
    assert results[('sample2',)] == ('sample2',)

def test_fapply_return_threaded(simple_adata_dict):
    """Test threaded processing with return values."""
    results = adata_dict_fapply(
        simple_adata_dict,
        create_key_aware_function(),
        use_multithreading=True,
        num_workers=2
    )
    assert results[('sample1',)] == ('sample1',)
    assert results[('sample2',)] == ('sample2',)

def test_fapply_return_as_adata_dict(simple_adata_dict):
    """Test return as AdataDict."""
    results = adata_dict_fapply(
        simple_adata_dict,
        create_key_aware_function(),
        return_as_adata_dict=True
    )
    assert isinstance(results, AdataDict)
    assert results.hierarchy == simple_adata_dict.hierarchy
    assert results[('sample1',)] == ('sample1',)

def test_fapply_return_nested(nested_adata_dict):
    """Test return values with nested structure."""
    results = adata_dict_fapply(nested_adata_dict, create_key_aware_function())
    assert isinstance(results[('group1',)], dict)
    assert results[('group1',)][('sample1',)] == ('sample1',)
    assert results[('group2',)][('sample3',)] == ('sample3',)

# Error handling tests
def test_error_handling(error_prone_adata_dict):
    results = adata_dict_fapply(
        error_prone_adata_dict,
        lambda x: x.problematic_method(),
        max_retries=1
    )

    # This sample is good, so the result should be None (no error).
    assert results[('good_sample',)] is None

    # Each error key now returns the actual exception object, so check types:
    assert isinstance(results[('value_error',)], ValueError)
    assert isinstance(results[('key_error',)], KeyError)
    assert isinstance(results[('runtime_error',)], RuntimeError)

# Threading-specific tests
def test_thread_safety(simple_adata_dict):
    """Test thread safety with shared counter."""
    shared_counter = {'count': 0}
    lock = threading.Lock()

    def thread_safe_func(adata):
        with lock:
            shared_counter['count'] += 1
        return True

    adata_dict_fapply(
        simple_adata_dict,
        thread_safe_func,
        use_multithreading=True,
        num_workers=2
    )

    assert shared_counter['count'] == len(simple_adata_dict)

def test_thread_pool_size(nested_adata_dict):
    """Test thread pool size limitation."""
    thread_ids = set()
    lock = threading.Lock()

    def collect_thread_ids(adata):
        with lock:
            thread_ids.add(threading.get_ident())
        return True

    adata_dict_fapply(
        nested_adata_dict,
        collect_thread_ids,
        use_multithreading=True,
        num_workers=2
    )

    # Should only have used 2 threads plus main thread
    assert len(thread_ids) <= 3

# Hierarchy preservation test
def test_hierarchy_preservation(nested_adata_dict):
    """Test preservation of hierarchy in nested AdataDict."""
    results = adata_dict_fapply(
        nested_adata_dict,
        create_test_function(),
        return_as_adata_dict=True
    )
    assert results.hierarchy == nested_adata_dict.hierarchy


# ---------------------------
# get_depth + max_depth tests
# ---------------------------

def test_get_depth_none(simple_adata_dict):
    assert get_depth(None, simple_adata_dict) is None


def test_get_depth_int(simple_adata_dict):
    assert get_depth(0, simple_adata_dict) == 0
    assert get_depth(1, simple_adata_dict) == 1
    assert get_depth(2, simple_adata_dict) == 2


def test_get_depth_negative_raises(simple_adata_dict):
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        get_depth(-1, simple_adata_dict)


def test_get_depth_semantic_requires_hierarchy():
    with pytest.raises(
        ValueError,
        match="Hierarchy not set so can't use semantic depth; use integer max_depth or recreate the adata_dict with build_adata_dict\\(\\) and try again\\.",
    ):
        get_depth("sample", {})


def test_get_depth_semantic_resolves_to_integer_depth(nested_adata_dict):
    # hierarchy=("group", ("sample",))
    assert get_depth("group", nested_adata_dict) == 1
    assert get_depth("sample", nested_adata_dict) == 2
    assert get_depth(["group", "sample"], nested_adata_dict) == 1
    assert get_depth(("sample",), nested_adata_dict) == 2


def test_get_depth_semantic_not_found_raises(nested_adata_dict):
    with pytest.raises(ValueError, match=r"Level\(s\) .* not found in hierarchy"):
        get_depth("not_a_level", nested_adata_dict)


def test_max_depth_zero_applies_to_root(nested_adata_dict):
    seen = []

    def record(obj, adt_key=None):
        seen.append((type(obj).__name__, adt_key))
        return True

    out = adata_dict_fapply(nested_adata_dict, record, use_multithreading=False, max_depth=0)
    assert out is True
    assert seen == [("AdataDict", ())]


def test_max_depth_one_applies_to_top_level_children_only(nested_adata_dict):
    seen = []

    def record(obj, adt_key=None):
        seen.append((type(obj).__name__, adt_key))
        return True

    out = adata_dict_fapply(nested_adata_dict, record, use_multithreading=False, max_depth=1)
    assert set(out.keys()) == set(nested_adata_dict.keys())
    assert len(seen) == len(nested_adata_dict)
    assert all(t == "AdataDict" for t, _ in seen)


def test_max_depth_none_recurses_to_anndata_leaves(nested_adata_dict):
    from anndata import AnnData

    seen_types = []

    def record(obj, **kwargs):
        seen_types.append(type(obj))
        return True

    out = adata_dict_fapply(nested_adata_dict, record, use_multithreading=False, max_depth=None)
    assert out is not None
    n_leaves = sum(len(group) for group in nested_adata_dict.values())
    assert len(seen_types) == n_leaves
    assert all(t is AnnData for t in seen_types)


def test_max_depth_semantic_group_equivalent_to_one(nested_adata_dict):
    from anndata import AnnData

    def fail_on_anndata(obj, **kwargs):
        assert not isinstance(obj, AnnData)
        return True

    out = adata_dict_fapply(nested_adata_dict, fail_on_anndata, use_multithreading=False, max_depth="group")
    assert set(out.keys()) == set(nested_adata_dict.keys())


def test_max_depth_semantic_sample_equivalent_to_leaf_application(nested_adata_dict):
    from anndata import AnnData

    seen = []

    def record(obj, **kwargs):
        seen.append(type(obj))
        return True

    adata_dict_fapply(nested_adata_dict, record, use_multithreading=False, max_depth="sample")
    assert all(t is AnnData for t in seen)
