"""
Unit tests for per-task RNG isolation in threaded adata_dict_fapply.
"""
# pylint: disable=protected-access,unused-argument

import random
import threading
import time

from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np
import pandas as pd
import anndata as ad
import scipy.stats

from scipy import sparse
from sklearn.utils import check_random_state, shuffle

from anndict.adata_dict import adata_dict_fapply
from anndict.adata_dict import _task_rng
from anndict.adata_dict._task_rng import run_in_task, task_rng_scope


STRATA = {f"s{i}": i for i in range(8)}


def patched_objects():
    """Every global object that a task RNG scope replaces."""
    objects = {
        "np.random.mtrand._rand": np.random.mtrand._rand,
        "ThreadPoolExecutor.submit": ThreadPoolExecutor.submit,
        "threading.Thread.start": threading.Thread.start,
        "scipy.stats.norm._random_state": scipy.stats.norm._random_state,
    }
    objects.update({f"np.random.{name}": getattr(np.random, name) for name in _task_rng._NUMPY_FUNCTIONS})
    objects.update({f"random.{name}": getattr(random, name) for name in _task_rng._PYTHON_FUNCTIONS})
    try:
        import igraph  # pylint: disable=import-outside-toplevel
        objects["igraph.set_random_number_generator"] = igraph.set_random_number_generator
    except ImportError:
        pass
    return objects


def assert_restored(before):
    """Assert that every patched global is the original object again."""
    after = patched_objects()
    assert after.keys() == before.keys()
    assert [name for name in before if after[name] is not before[name]] == []


def call_in_new_thread(fn, *args):
    """Run ``fn(*args)`` in a new thread and return its result."""
    out = {}
    thread = threading.Thread(target=lambda: out.update(value=fn(*args)))
    thread.start()
    thread.join()
    return out["value"]


def seed_wait_draw(seed, draw):
    """Build a function that seeds a global generator, lets other threads run, then draws from it."""
    def func(adata):
        seed()
        time.sleep(0.01)
        return draw()
    return func


def run_sequential_and_threaded(func, make_strata=lambda: dict(STRATA)):
    """Apply ``func`` with and without threads."""
    sequential = adata_dict_fapply(make_strata(), func, use_multithreading=False, catch_errors=False)
    threaded = adata_dict_fapply(make_strata(), func, use_multithreading=True, num_workers=4, catch_errors=False)
    return sequential, threaded


def assert_same_results(expected, actual):
    """Assert that two fapply results are equal key by key."""
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)


def make_adata_strata(n_obs=200, n_vars=100, n_strata=8):
    """Flat dict of AnnData objects with random counts."""
    rng = np.random.RandomState(0)
    strata = {}
    for i in range(n_strata):
        adata = ad.AnnData(rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32))
        adata.var_names = [f"gene{j}" for j in range(n_vars)]
        strata[f"s{i}"] = adata
    return strata


class GeneratorWithoutGetrandbits:
    """Like scanpy's RNGIgraph: wraps a NumPy RandomState and has no getrandbits."""

    def __init__(self, seed):
        self._rng = np.random.RandomState(seed)

    def __getattr__(self, attr):
        return getattr(self._rng, "normal" if attr == "gauss" else attr)


# Scope lifecycle
def test_scope_restores_globals():
    before = patched_objects()
    with task_rng_scope():
        assert np.random.mtrand._rand is _task_rng._NUMPY_PROXY
    assert_restored(before)


def test_scope_restores_globals_after_exception():
    before = patched_objects()
    with pytest.raises(ValueError):
        with task_rng_scope():
            raise ValueError("boom")
    assert_restored(before)


def test_nested_scopes_restore_after_outermost_exits():
    before = patched_objects()
    with task_rng_scope():
        with task_rng_scope():
            pass
        assert np.random.mtrand._rand is _task_rng._NUMPY_PROXY
    assert_restored(before)


# Routing
def test_outside_tasks_uses_real_global_generators():
    with task_rng_scope():
        np.random.seed(7)
        random.seed(7)
    assert np.random.random() == np.random.RandomState(7).random_sample()
    assert random.random() == random.Random(7).random()


def test_task_seed_gives_global_seed_stream():
    def draws():
        np.random.seed(0)
        random.seed(0)
        return np.concatenate([np.random.random(3), check_random_state(None).random_sample(3)]), random.random()

    with task_rng_scope() as spawn:
        numpy_draws, python_draw = call_in_new_thread(run_in_task, spawn(), draws)
    np.testing.assert_array_equal(numpy_draws, np.random.RandomState(0).random_sample(6))
    assert python_draw == random.Random(0).random()


def test_task_leaves_real_global_generators_untouched():
    np.random.seed(1)
    random.seed(1)
    expected_numpy = np.random.RandomState(1)
    expected_numpy.randint(0, 2**32, size=4, dtype=np.uint32)  # the scope draws its seed entropy
    expected_python = random.Random(1)
    expected_python.getrandbits(128)

    def draws():
        np.random.seed(0)
        random.seed(0)
        return np.random.random(10), random.random()

    with task_rng_scope() as spawn:
        call_in_new_thread(run_in_task, spawn(), draws)
    assert np.random.random() == expected_numpy.random_sample()
    assert random.random() == expected_python.random()


def test_spawned_generators_are_reproducible_and_independent():
    def spawn_draws():
        np.random.seed(3)
        random.seed(3)
        with task_rng_scope() as spawn:
            task_rngs = [spawn() for _ in range(3)]
        return [(task_rng.numpy.random_sample(), task_rng.python.random()) for task_rng in task_rngs]

    first = spawn_draws()
    assert first == spawn_draws()
    assert len(set(first)) == len(first)


def test_threads_started_by_task_share_its_generators():
    existing_pool = ThreadPoolExecutor(max_workers=1)
    existing_pool.submit(lambda: None).result()  # start its worker before the scope

    def draw_later():
        time.sleep(0.01)
        return np.random.random(2)

    def task():
        np.random.seed(0)
        with ThreadPoolExecutor(max_workers=1) as new_pool:
            from_new_pool = new_pool.submit(draw_later).result()
        from_existing_pool = existing_pool.submit(draw_later).result()
        return np.concatenate([from_new_pool, from_existing_pool, call_in_new_thread(draw_later)])

    try:
        with task_rng_scope() as spawn:
            draws = call_in_new_thread(run_in_task, spawn(), task)
            # the existing pool's worker is not left using the task's generators
            np.random.seed(5)
            outside_task = existing_pool.submit(np.random.random).result()
    finally:
        existing_pool.shutdown()
    np.testing.assert_array_equal(draws, np.random.RandomState(0).random_sample(6))
    assert outside_task == np.random.RandomState(5).random_sample()


@pytest.mark.parametrize(
    "make_generator",
    [None, lambda: random.Random(0), lambda: GeneratorWithoutGetrandbits(0)],
    ids=["random module", "random.Random", "without getrandbits"],
)
def test_igraph_in_task_matches_calling_thread(make_generator):
    igraph = pytest.importorskip("igraph")

    def edges():
        random.seed(0)
        if make_generator is not None:
            igraph.set_random_number_generator(make_generator())
        try:
            return np.array(igraph.Graph.Erdos_Renyi(n=50, m=150).get_edgelist())
        finally:
            igraph.set_random_number_generator(random)

    expected = edges()
    with task_rng_scope() as spawn:
        in_task = call_in_new_thread(run_in_task, spawn(), edges)
    np.testing.assert_array_equal(in_task, expected)


# adata_dict_fapply: functions that seed a global generator, then draw after other threads have run
GLOBAL_RNG_USES = {
    "np.random": (lambda: np.random.seed(0), lambda: np.random.random(5)),
    "pandas sample": (lambda: np.random.seed(0), lambda: pd.Series(range(1000)).sample(10).to_numpy()),
    "sklearn": (lambda: np.random.seed(0), lambda: shuffle(np.arange(1000), random_state=None)[:10]),
    "scipy.sparse": (lambda: np.random.seed(0), lambda: sparse.random(20, 20, density=0.2, random_state=None).toarray()),
    "scipy.stats": (lambda: np.random.seed(0), lambda: scipy.stats.norm.rvs(size=5)),
    "random": (lambda: random.seed(0), lambda: [random.random() for _ in range(5)]),
}


@pytest.mark.parametrize("seed, draw", list(GLOBAL_RNG_USES.values()), ids=list(GLOBAL_RNG_USES))
def test_fapply_threaded_global_rng_matches_sequential(seed, draw):
    assert_same_results(*run_sequential_and_threaded(seed_wait_draw(seed, draw)))


def test_fapply_threaded_igraph_matches_sequential():
    igraph = pytest.importorskip("igraph")
    func = seed_wait_draw(
        lambda: random.seed(0),
        lambda: np.array(igraph.Graph.Erdos_Renyi(n=50, m=150).get_edgelist()),
    )
    assert_same_results(*run_sequential_and_threaded(func))


def test_fapply_threaded_child_threads_match_sequential():
    def draw_later():
        time.sleep(0.01)
        return np.random.random(2)

    def func(adata):
        np.random.seed(0)
        with ThreadPoolExecutor(max_workers=1) as pool:
            from_pool = pool.submit(draw_later).result()
        return np.concatenate([from_pool, call_in_new_thread(draw_later)])

    assert_same_results(*run_sequential_and_threaded(func))


def test_fapply_nested_threaded_matches_sequential():
    inner = seed_wait_draw(lambda: np.random.seed(0), lambda: np.random.random(2))

    def outer(adata):
        np.random.seed(adata)
        inner_results = adata_dict_fapply(
            {f"i{i}": i for i in range(4)}, inner, use_multithreading=True, num_workers=2, catch_errors=False
        )
        return np.concatenate([inner_results[key] for key in sorted(inner_results)] + [np.random.random(2)])

    assert_same_results(*run_sequential_and_threaded(outer))


def test_fapply_threaded_unseeded_draws_follow_global_seeds():
    def draw(adata):
        time.sleep(0.01)
        return np.array([np.random.random(), random.random()])

    def run():
        np.random.seed(123)
        random.seed(123)
        return adata_dict_fapply(dict(STRATA), draw, use_multithreading=True, num_workers=4, catch_errors=False)

    first = run()
    assert_same_results(first, run())
    assert len({tuple(values) for values in first.values()}) == len(first)


def test_fapply_threaded_restores_globals():
    before = patched_objects()
    adata_dict_fapply(dict(STRATA), lambda adata: np.random.random(), use_multithreading=True)
    assert_restored(before)


def test_fapply_threaded_restores_globals_after_error():
    before = patched_objects()

    def fail(adata):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        adata_dict_fapply(dict(STRATA), fail, use_multithreading=True, catch_errors=False)
    assert_restored(before)


def test_fapply_isolate_rng_false_does_not_patch():
    original = np.random.mtrand._rand
    seen = []

    def record(adata):
        seen.append(np.random.mtrand._rand is original)

    adata_dict_fapply(dict(STRATA), record, use_multithreading=True, isolate_rng=False)
    assert seen and all(seen)


# adata_dict_fapply with scanpy functions that use global generators
def test_fapply_threaded_scanpy_score_genes_matches_sequential():
    sc = pytest.importorskip("scanpy")

    def score(adata):
        sc.tl.score_genes(adata, gene_list=[f"gene{j}" for j in range(10)], ctrl_size=10, n_bins=5, random_state=0)
        return adata.obs["score"].to_numpy()

    assert_same_results(*run_sequential_and_threaded(score, make_adata_strata))


def test_fapply_threaded_scanpy_subsample_matches_sequential():
    sc = pytest.importorskip("scanpy")

    def subsample(adata):
        return sc.pp.subsample(adata, n_obs=50, copy=True).obs_names.to_numpy()

    assert_same_results(*run_sequential_and_threaded(subsample, make_adata_strata))


def test_fapply_threaded_scanpy_leiden_igraph_matches_sequential():
    sc = pytest.importorskip("scanpy")
    pytest.importorskip("igraph")
    strata = make_adata_strata(n_obs=300, n_vars=10)
    for adata in strata.values():
        sc.pp.neighbors(adata, use_rep="X", n_neighbors=10)

    def leiden(adata):
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False, random_state=0)
        return adata.obs["leiden"].to_numpy().astype(str)

    def make_strata():
        return {key: adata.copy() for key, adata in strata.items()}

    assert_same_results(*run_sequential_and_threaded(leiden, make_strata))
