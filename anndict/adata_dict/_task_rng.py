"""
Per-task copies of process-global random number generators, used by threaded :func:`adata_dict_fapply`.

Many functions draw from process-global RNGs, often right after seeding them: NumPy's legacy global
``RandomState`` (e.g. :func:`scanpy.tl.score_genes`, :func:`scanpy.pp.subsample`), :mod:`random`, and
igraph's generator. Threads running such functions reseed and consume each other's streams, so results
depend on thread scheduling.

While :func:`task_rng_scope` is active, the global RNG entry points are routed to generators owned by the
task running in the current thread, so a function that seeds a global RNG gets the same results as when it
runs alone. Threads that are not running a task use the real global generators.
"""
# Patching these globals is the purpose of this module
# pylint: disable=protected-access,global-statement,invalid-name,too-few-public-methods
from __future__ import annotations

import functools
import importlib.util
import random
import sys
import threading

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _futures_thread
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import numpy as np

# The real global generators, captured before anything is patched
_NUMPY_GLOBAL = np.random.mtrand._rand
_PYTHON_GLOBAL = random._inst

_local = threading.local()
_lock = threading.Lock()
_active_scopes = 0
_originals: dict[tuple[Any, str], Any] = {}
_patched: list[tuple[Any, str]] = []
_igraph_module = None
_igraph_generator_outside_tasks: Any = random


class TaskRng:
    """
    The generators owned by one task. Threads started by the task share them.
    """
    __slots__ = ("numpy", "python", "igraph")

    def __init__(self, numpy_rng: np.random.RandomState, python_rng: random.Random):
        self.numpy = numpy_rng
        self.python = python_rng
        # generator the task passed to igraph.set_random_number_generator; None means :mod:`random`
        self.igraph = None


# ---- NumPy -------------------------------------------------------------------------------------

def _numpy_rng() -> np.random.RandomState:
    task_rng = getattr(_local, "task_rng", None)
    return _NUMPY_GLOBAL if task_rng is None else task_rng.numpy


_RANDOM_STATE_METHODS = frozenset(
    name for name in dir(np.random.RandomState)
    if not name.startswith("_") and callable(getattr(np.random.RandomState, name))
)


def _route_numpy(name: str, target: str) -> Callable:
    def routed(*args, **kwargs):
        return getattr(_numpy_rng(), target)(*args, **kwargs)
    routed.__name__ = name
    routed.__doc__ = getattr(np.random.RandomState, target).__doc__
    return routed


def _numpy_proxy_method(name: str) -> Callable:
    def method(self, *args, **kwargs):  # pylint: disable=unused-argument
        return getattr(_numpy_rng(), name)(*args, **kwargs)
    method.__name__ = name
    return method


class _NumpyGlobalProxy(np.random.RandomState):
    """
    Replacement for ``np.random.mtrand._rand``, which sklearn and scipy use when ``random_state=None``.
    """


for _name in _RANDOM_STATE_METHODS:
    setattr(_NumpyGlobalProxy, _name, _numpy_proxy_method(_name))

_NUMPY_PROXY = _NumpyGlobalProxy()

# np.random.<name> functions are bound to the original global at import, so replacing _rand does not reach them
_NUMPY_FUNCTIONS = {
    name: _route_numpy(name, name) for name in np.random.mtrand.__all__ if name in _RANDOM_STATE_METHODS
}
_NUMPY_FUNCTIONS.update(
    {alias: _route_numpy(alias, "random_sample") for alias in ("ranf", "sample") if hasattr(np.random, alias)}
)
if hasattr(np.random, "get_bit_generator"):
    _NUMPY_FUNCTIONS["get_bit_generator"] = lambda: _numpy_rng()._bit_generator


def _swap_scipy_stats_generators(old: Any, new: Any) -> None:
    """
    Module-level scipy.stats distributions captured NumPy's global generator when scipy.stats was imported.
    """
    stats = sys.modules.get("scipy.stats")
    if stats is None:
        return
    from scipy.stats._distn_infrastructure import rv_generic  # pylint: disable=import-outside-toplevel
    for distribution in vars(stats).values():
        if isinstance(distribution, rv_generic) and distribution._random_state is old:
            distribution._random_state = new


# ---- Python random -----------------------------------------------------------------------------

def _python_rng() -> random.Random:
    task_rng = getattr(_local, "task_rng", None)
    return _PYTHON_GLOBAL if task_rng is None else task_rng.python


def _route_python(name: str) -> Callable:
    def routed(*args, **kwargs):
        return getattr(_python_rng(), name)(*args, **kwargs)
    routed.__name__ = name
    routed.__doc__ = getattr(random.Random, name).__doc__
    return routed


# random.<name> functions are bound to random._inst at import
_PYTHON_FUNCTIONS = {
    name: _route_python(name) for name in random.__all__
    if callable(getattr(random.Random, name, None)) and callable(getattr(random, name, None))
}


# ---- igraph ------------------------------------------------------------------------------------
# igraph's C core keeps the default generator per OS thread, while python-igraph keeps one
# process-wide table of the Python callbacks passed to set_random_number_generator.

def _igraph_rng() -> Any:
    task_rng = getattr(_local, "task_rng", None)
    if task_rng is None:
        return _igraph_generator_outside_tasks
    return random if task_rng.igraph is None else task_rng.igraph


class _IgraphGenerator:
    """
    The generator igraph holds while a scope is active. Forwards to the generator selected in the current thread.
    """

    def getrandbits(self, k: int) -> int:
        """Forward to the selected generator, emulating python-igraph when it has no ``getrandbits``."""
        rng = _igraph_rng()
        getrandbits = getattr(rng, "getrandbits", None)
        if getrandbits is None:
            # python-igraph calls randint(0, 2**32 - 1) for a generator without getrandbits
            return rng.randint(0, (1 << k) - 1)
        return getrandbits(k)

    def randint(self, a: int, b: int) -> int:
        """Forward to the selected generator."""
        return _igraph_rng().randint(a, b)

    def random(self) -> float:
        """Forward to the selected generator."""
        return _igraph_rng().random()

    def gauss(self, mu: float, sigma: float) -> float:
        """Forward to the selected generator."""
        return _igraph_rng().gauss(mu, sigma)


_IGRAPH_GENERATOR = _IgraphGenerator()


def _set_igraph_generator(generator: Any) -> None:
    """
    Replacement for ``igraph.set_random_number_generator`` that records the generator per task.
    """
    global _igraph_generator_outside_tasks
    set_generator = _originals[(_igraph_module, "set_random_number_generator")]
    task_rng = getattr(_local, "task_rng", None)
    if generator is None:
        # igraph reverts this thread to its built-in C generator
        if task_rng is not None:
            task_rng.igraph = None
        return set_generator(None)
    if task_rng is None:
        _igraph_generator_outside_tasks = generator
    else:
        task_rng.igraph = generator
    # setting it from this thread is what points this thread's igraph default at the Python callbacks
    return set_generator(_IGRAPH_GENERATOR)


def _import_igraph() -> Any:
    # import up front so a task cannot import igraph after patching and bypass it
    if "igraph" not in sys.modules and importlib.util.find_spec("igraph") is None:
        return None
    import igraph  # pylint: disable=import-outside-toplevel
    return igraph


# ---- Threads started by a task -----------------------------------------------------------------

def _run_with(task_rng: TaskRng, fn: Callable, /, *args, **kwargs) -> Any:
    previous = getattr(_local, "task_rng", None)
    _local.task_rng = task_rng
    try:
        return fn(*args, **kwargs)
    finally:
        _local.task_rng = previous


def _submit(self, fn, /, *args, **kwargs):
    """
    Replacement for ``ThreadPoolExecutor.submit``: work submitted by a task runs with the task's generators.
    """
    task_rng = getattr(_local, "task_rng", None)
    if task_rng is not None:
        fn = functools.partial(_run_with, task_rng, fn)
    return _originals[(ThreadPoolExecutor, "submit")](self, fn, *args, **kwargs)


def _start(self):
    """
    Replacement for ``threading.Thread.start``: threads started by a task run with the task's generators.
    """
    task_rng = getattr(_local, "task_rng", None)
    # executor workers are handled per work item by _submit
    if task_rng is not None and getattr(self, "_target", None) is not getattr(_futures_thread, "_worker", None):
        self.run = functools.partial(_run_with, task_rng, self.run)
    return _originals[(threading.Thread, "start")](self)


# ---- Scope -------------------------------------------------------------------------------------

def _install() -> None:
    global _igraph_module
    patches: list[tuple[Any, str, Any]] = [(np.random.mtrand, "_rand", _NUMPY_PROXY)]
    patches += [(np.random, name, routed) for name, routed in _NUMPY_FUNCTIONS.items()]
    patches += [(random, name, routed) for name, routed in _PYTHON_FUNCTIONS.items()]
    patches += [(ThreadPoolExecutor, "submit", _submit), (threading.Thread, "start", _start)]
    igraph = _import_igraph()
    if igraph is not None:
        patches.append((igraph, "set_random_number_generator", _set_igraph_generator))

    for owner, name, replacement in patches:
        _originals[(owner, name)] = getattr(owner, name)
        setattr(owner, name, replacement)
        _patched.append((owner, name))
    _swap_scipy_stats_generators(_NUMPY_GLOBAL, _NUMPY_PROXY)

    if igraph is not None:
        _igraph_module = igraph
        _originals[(igraph, "set_random_number_generator")](_IGRAPH_GENERATOR)


def _uninstall() -> None:
    global _igraph_module
    _swap_scipy_stats_generators(_NUMPY_PROXY, _NUMPY_GLOBAL)
    while _patched:
        owner, name = _patched.pop()
        setattr(owner, name, _originals[(owner, name)])
    if _igraph_module is not None:
        _igraph_module.set_random_number_generator(_igraph_generator_outside_tasks)
        _igraph_module = None


@contextmanager
def task_rng_scope() -> Iterator[Callable[[], TaskRng]]:
    """
    Route the process-global RNGs to per-task generators until the scope exits.

    Scopes can be nested and can run concurrently; the globals are restored when the last one exits.

    Yields
    ------
    A function returning the :class:`TaskRng` for the next task. Run each task with :func:`run_in_task`.
    Task generators are seeded, in the order they are requested, from the calling thread's NumPy and
    :mod:`random` generators, each of which advances by one draw when the scope is entered.
    """
    global _active_scopes
    with _lock:
        if _active_scopes == 0:
            try:
                _install()
            except BaseException:
                _uninstall()
                raise
        _active_scopes += 1
    try:
        numpy_seeds = np.random.SeedSequence(np.random.randint(0, 2**32, size=4, dtype=np.uint32))
        python_seeds = np.random.SeedSequence(random.getrandbits(128))

        def spawn() -> TaskRng:
            numpy_seed, = numpy_seeds.spawn(1)
            python_seed, = python_seeds.spawn(1)
            return TaskRng(
                np.random.RandomState(np.random.MT19937(numpy_seed)),
                random.Random(int.from_bytes(python_seed.generate_state(2, np.uint64).tobytes(), "little")),
            )

        yield spawn
    finally:
        with _lock:
            _active_scopes -= 1
            if _active_scopes == 0:
                _uninstall()


def run_in_task(task_rng: TaskRng, fn: Callable, /, *args, **kwargs) -> Any:
    """
    Call ``fn(*args, **kwargs)`` as a task that owns ``task_rng``. Must run inside :func:`task_rng_scope`.
    """
    if _igraph_module is not None:
        # a new thread's igraph default is igraph's C generator; match the calling thread, whose default is Python's
        _originals[(_igraph_module, "set_random_number_generator")](_IGRAPH_GENERATOR)
    return _run_with(task_rng, fn, *args, **kwargs)
