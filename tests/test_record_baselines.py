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
