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
