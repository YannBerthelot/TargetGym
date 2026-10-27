"""``scripts/record_baselines.py``: how it starts its worker processes."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "record_baselines_under_test", ROOT / "scripts" / "record_baselines.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_workers_are_spawned_not_forked():
    """JAX has threads running in the recorder's own process before the pool
    starts, and a process forked with threads running can deadlock. Linux
    forks by default: on GitHub's runners every recording job hung with no
    seed done until its five-hour timeout."""
    rb = _load()
    assert rb._spawn.get_start_method() == "spawn"
    with rb._pool(1) as pool:
        assert pool._mp_context is rb._spawn


GB = 1024**3


def test_heavy_planners_get_a_pool_sized_for_their_compile():
    """A per-seed worker compiles its own planner, and the formation oracle's
    compile alone peaks at about 7.4 GB: four of them at once take down a
    16 GB runner. Each task's jobs run in a pool sized for that task, the
    heaviest first, and never fewer than one worker."""
    rb = _load()
    jobs = [
        ("mpc", "first_order", 0),
        ("mpc", "formation", 0),
        ("pid", "formation", 0),
        ("mpc", "patrol", 0),
        ("pid", "first_order", 0),
    ]
    pools = rb._pools(jobs, available=15 * GB, cores=4)
    assert [workers for _, workers in pools] == [1, 3, 4]
    assert pools[0][0] == [("mpc", "formation", 0), ("pid", "formation", 0)]
    assert pools[1][0] == [("mpc", "patrol", 0)]
    assert pools[2][0] == [("mpc", "first_order", 0), ("pid", "first_order", 0)]
    # A laptop with room to spare keeps the default width, capped by cores.
    assert rb._pools(jobs[:1], available=24 * GB, cores=14) == [(jobs[:1], 12)]
    # Never an empty pool, however tight memory is.
    assert rb._pools(jobs[1:2], available=1 * GB, cores=4)[0][1] == 1
