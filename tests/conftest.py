"""Test-wide setup.

Two things every test process needs, set before anything imports pygame or
matplotlib: a headless display, and a single compute thread per process.

The thread cap matters because the suite runs under ``pytest-xdist``. Each
worker is its own process with its own JAX/XLA runtime, and by default each
would size its thread pool to the whole machine -- so N workers oversubscribe
the box by a factor of N and every one of them slows down. One thread per
worker leaves the parallelism to xdist, which is where it belongs.
"""

import os

import pytest

for _var, _value in (
    # pygame and matplotlib both fail on a runner with no display unless told
    # to render offscreen. CI has no display; neither does a container.
    ("SDL_VIDEODRIVER", "dummy"),
    ("SDL_AUDIODRIVER", "dummy"),
    ("MPLBACKEND", "Agg"),
    # These are read once, when the respective library first initialises its
    # thread pool, so they have to be in place before the first import.
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("JAX_PLATFORMS", "cpu"),
):
    os.environ.setdefault(_var, _value)

# XLA keeps its own CPU thread pool, sized to the whole machine and not
# governed by the variables above. Both flags are needed together: XLA reads
# XLA_FLAGS as a flag string only when it starts with "--" and otherwise tries
# to open it as a file and aborts the process.
_XLA_SINGLE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
os.environ["XLA_FLAGS"] = " ".join(
    filter(None, (os.environ.get("XLA_FLAGS", ""), _XLA_SINGLE_THREAD))
)


# XLA compilation, cached across processes and across runs.
#
# Profiling the fast job found its cost is not stepping the plants but
# *compiling* them: the four aircraft tuner smoke tests take two gradient steps
# each and still cost 149 s, because reverse-mode through an RK4 aircraft builds
# a large graph, and shrinking the problem to 2 targets and 100 steps barely
# moved them. The graph is the cost, not the rollout.
#
# JAX can persist compiled executables to disk, keyed on the computation itself,
# so the same graph is only ever compiled once. Measured on one of those tests:
# 44.9 s cold, 3.4 s warm. Under xdist every worker is a separate process, so a
# shared directory also stops N workers each compiling the same thing; and in CI
# the directory is restored from the actions cache, which is what makes it a
# once-per-code-change cost rather than a once-per-push one.
#
# Entries are keyed by the jaxpr, the backend and the JAX version, so a stale
# entry is never reused -- a changed computation simply misses.
_CACHE_DIR = os.environ.get(
    "TARGETGYM_JAX_CACHE",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".jax_cache"
    ),
)


# A worker past this resident size drops its compiled programs before the next
# test. Two workers at this size, with the compile in flight, stay inside a CI
# runner's 16 GB: on 2026-11-14 the per-module drop alone let both workers
# grow until the runner shut down, 16 % into the small tier.
_FREE_ABOVE_BYTES = 4 * 1024**3


def _resident_bytes():
    """This process's resident size, or 0 where ``/proc`` does not say."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _drop_compiled_programs():
    import ctypes
    import gc

    import jax

    jax.clear_caches()
    gc.collect()
    try:  # glibc keeps freed pages unless asked to hand them back
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@pytest.fixture(autouse=True, scope="module")
def _free_compiled_programs():
    """Drop compiled programs after each module, where ``FREE_COMPILED_PROGRAMS``
    is set. A cold run on the older JAX fills a CI runner with them, whatever
    the worker count, and the on-disk cache brings back what is needed."""
    yield
    if os.environ.get("FREE_COMPILED_PROGRAMS"):
        _drop_compiled_programs()


@pytest.fixture(autouse=True)
def _free_compiled_programs_when_large(request):
    """Inside a module too: one parametrised over every task compiles them all.

    Each worker also leaves the name of its test in a file, which the CI watch
    line prints: a test that takes the runner down reports nothing itself."""
    if os.environ.get("FREE_COMPILED_PROGRAMS"):
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        with open(os.path.join(_CACHE_DIR, f"current-test-{worker}"), "w") as f:
            f.write(request.node.nodeid)
    yield
    if (
        os.environ.get("FREE_COMPILED_PROGRAMS")
        and _resident_bytes() > _FREE_ABOVE_BYTES
    ):
        _drop_compiled_programs()


def pytest_configure(config):  # noqa: ARG001
    """Enable the persistent compilation cache once the process has a JAX."""
    try:
        import jax
    except ImportError:  # pragma: no cover - jax is a hard dependency
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", _CACHE_DIR)
    # The defaults only cache compilations already costing over a second, which
    # skips the many medium-sized graphs that make up most of this suite's bill.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
