"""``scripts/record_baselines.py``: how it starts its worker processes, and
what its two PID modes write. ``--pid-only`` is the local check: it re-runs
the PIDs, reports returns that moved, and writes nothing. ``--part expert``
records the PIDs' part of each row and keeps the MPCs' part as recorded."""

from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import pathlib
import shutil

import pytest

from target_gym import provenance, records
from target_gym.registry import REGISTRY

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


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A copy of the shipped records in place of the package's, and the
    recorder's seeds run on threads, so a run stays in this process."""
    monkeypatch.setattr(records, "DATA", tmp_path / "data")
    shutil.copytree(ROOT / "src" / "target_gym" / "data", records.DATA)
    rb = _load()
    monkeypatch.setattr(rb, "_pool", lambda workers: cf.ThreadPoolExecutor(workers))
    return rb


def _snapshot() -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in records.DATA.rglob("*.json")}


def test_the_pid_check_writes_nothing(store):
    before = _snapshot()
    assert store.main(["--envs", "first_order", "--pid-only"]) == 0
    assert _snapshot() == before


def test_recording_the_expert_part_keeps_the_oracle_part(store):
    """The row was stale in its expert part; the run refreshes that part,
    reproduces the PID returns, and leaves the oracle's keys and every other
    file as they were."""
    old = records.load_row("baseline_returns", "first_order")
    records.write("baseline_returns", "first_order", {**old, "expert_fingerprint": "x"})
    before = _snapshot()
    assert store.main(["--envs", "first_order", "--part", "expert"]) == 0
    row = records.load_row("baseline_returns", "first_order")
    assert row["expert_fingerprint"] == provenance.expert_fingerprint(
        REGISTRY["first_order"]
    )
    assert records.same(row["pid_returns"], old["pid_returns"])
    assert {k: row.get(k) for k in records.PARTS["oracle"]} == {
        k: old.get(k) for k in records.PARTS["oracle"]
    }
    changed = {p for p, b in _snapshot().items() if before.get(p) != b}
    assert changed == {str(records.path("baseline_returns", "first_order"))}


def test_the_seeds_certificates_merge_into_the_rows_worst_and_counts():
    """The oracle part of a row: the solves and failures summed, the mean
    iterations weighted by solves, each certificate's worst value, and the
    guard flags summed. By hand: (2 x 3 + 4 x 1) / 6 = 1.7 iterations."""
    a = {"solver_calls": 2, "solver_failures": 1, "solver_capped": 0}
    a |= {"solver_mean_iters": 3.0, "solver_max_error": 1e-9, "solver_guard_flags": 1}
    b = {"solver_calls": 4, "solver_failures": 0, "solver_capped": 1}
    b |= {"solver_mean_iters": 1.0, "solver_max_error": 4e-9, "solver_guard_flags": 2}
    assert _load()._merge_reports([a, b]) == {
        "solver_calls": 6, "solver_failures": 1, "solver_capped": 1,
        "solver_mean_iters": 1.7, "solver_max_error": 4e-9, "solver_guard_flags": 3,
    }  # fmt: skip
