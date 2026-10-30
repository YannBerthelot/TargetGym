"""``target_gym.records``: the recorded data, one JSON file per task.

Contract (from the 2026-10-28 cleanup decision and the task brief): each kind
of record (``baseline_returns``, ``protocol_results``, ``hold_measurements``,
``env_versions``) keeps one file per task, so a recording PR that re-records
one task rewrites that task's files and nothing else, and two such PRs never
conflict. Loading a kind gives back ``{task: row}`` exactly as written.
Two recordings of a row agree when their numbers are within ``REL_TOL`` and
everything else is equal, ignoring what only describes the recording (its
timing, date and CPU), because GitHub's runners do not all have the same CPU
and the last bits of a long rollout differ between them. Every recording
notes the CPU it was taken on.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from target_gym import records
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An empty store in place of the package's data directory."""
    monkeypatch.setattr(records, "DATA", tmp_path / "data")
    return tmp_path / "data"


def _snapshot(root: pathlib.Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*.json")}


def test_a_written_row_loads_back_as_written(store):
    row = {"fingerprint": "f1", "pid_returns": [-1.5, -0.25], "steps": 360}
    records.write("baseline_returns", "battery", row)
    assert records.load("baseline_returns") == {"battery": row}
    assert records.load_row("baseline_returns", "battery") == row
    assert records.load_row("baseline_returns", "hvac") == {}


def test_writing_one_task_leaves_every_other_file_untouched(store):
    for task in ("battery", "hvac", "plane"):
        for kind in records.KINDS:
            records.write(kind, task, {"fingerprint": "f1", "gain": 1.0})
    before = _snapshot(store)
    records.write("protocol_results", "hvac", {"fingerprint": "f2", "gain": 2.0})
    after = _snapshot(store)
    changed = sorted(k for k in after if after[k] != before.get(k))
    assert changed == [str(pathlib.Path("protocol_results", "hvac.json"))]


def test_an_unknown_kind_is_refused(store):
    with pytest.raises(ValueError):
        records.write("baseline_return", "battery", {})


@pytest.mark.parametrize(
    "old, new, agree",
    [
        # 12.661 -> 12.676 is #97's runner-to-runner move: 1.1e-3 relative.
        ({"gain": 12.661}, {"gain": 12.676}, True),
        ({"gain": 1.0}, {"gain": 1.02}, False),  # 2 % is beyond 1 %
        ({"pid": {"hold": 100.0}}, {"pid": {"hold": 100.9}}, True),
        ({"pid": {"hold": 100.0}}, {"pid": {"hold": 101.1}}, False),
        ({"returns": [1.0, 2.0]}, {"returns": [1.0, 2.0, 3.0]}, False),
        ({"fingerprint": "f1"}, {"fingerprint": "f2"}, False),
        ({"mpc_trips": 0}, {"mpc_trips": 1}, False),
        ({"gain": None}, {"gain": None}, True),
        ({"gain": 1.0}, {"gain": 1.0, "hold": 2.0}, False),
        # Costs this small are float noise around an exact zero.
        ({"hold": 3.0e-12}, {"hold": 3.1e-12}, True),
        ({"hold": 0.0}, {"hold": 0.0}, True),
    ],
)
def test_two_recordings_agree_within_the_tolerance(old, new, agree):
    assert records.same(old, new) is agree
    assert records.same(new, old) is agree


def test_what_describes_the_recording_is_not_compared():
    old = {"gain": 1.0, "seconds": 3.0, "generated": "then", "cpu": "Apple M2"}
    new = {"gain": 1.0, "seconds": 9.0, "generated": "now", "cpu": "AMD EPYC 7763"}
    assert records.same(old, new)


def test_a_stamped_row_notes_its_cpu_and_keeps_its_numbers():
    row = {"fingerprint": "f1", "gain": 1.0}
    stamped = records.stamped(row)
    assert isinstance(stamped["cpu"], str) and stamped["cpu"]
    assert isinstance(stamped["generated"], str) and stamped["generated"]
    assert records.same(row, stamped)
    assert "cpu" not in row, "the input is not mutated"


def test_the_package_ships_a_row_for_every_registered_task():
    """The data loads from the package itself (a checkout or an installed
    wheel alike), with a protocol and hold row for every task and a version
    stamp for every task's current version."""
    assert set(records.load("protocol_results")) == set(REGISTRY)
    assert set(records.load("hold_measurements")) == set(REGISTRY)
    stamps = records.load("env_versions")
    assert {s.versioned_name for s in REGISTRY.values()} <= set(stamps)
    with_mpc = {n for n, s in REGISTRY.items() if s.make_mpc is not None}
    assert with_mpc <= set(records.load("baseline_returns"))


def _stale_records():
    spec = importlib.util.spec_from_file_location(
        "stale_records_fold_under_test", ROOT / "scripts" / "stale_records.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fragment(root: pathlib.Path, kind: str, task: str, row: dict) -> None:
    path = root / kind / f"{task}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))


def test_folding_a_re_recorded_task_rewrites_only_that_tasks_files(
    store, tmp_path, capsys
):
    """What the CI recorder commits: a re-record of ``hvac`` changes hvac's
    files and leaves every other task's byte for byte."""
    for task in ("battery", "hvac", "plane"):
        records.write("baseline_returns", task, {"fingerprint": "f1", "r": [1.0]})
        records.write("protocol_results", task, {"fingerprint": "f1", "gain": 1.0})
    before = _snapshot(store)
    rows = tmp_path / "rows-hvac"
    _fragment(rows, "baseline_returns", "hvac", {"fingerprint": "f2", "r": [2.0]})
    _fragment(rows, "protocol_results", "hvac", {"fingerprint": "f2", "gain": 2.0})

    assert _stale_records().main(["merge", str(rows)]) == 0

    after = _snapshot(store)
    changed = sorted(k for k in after if after[k] != before[k])
    assert changed == [
        str(pathlib.Path("baseline_returns", "hvac.json")),
        str(pathlib.Path("protocol_results", "hvac.json")),
    ]
    assert capsys.readouterr().out.split() == ["hvac"]


def test_a_re_record_within_the_tolerance_commits_nothing(store, tmp_path, capsys):
    """A weekly re-record on a runner with another CPU reproduces the row to
    within the tolerance: nothing is written, so nothing is committed."""
    records.write("protocol_results", "plane", {"fingerprint": "f1", "gain": 12.661})
    before = _snapshot(store)
    rows = tmp_path / "rows-plane"
    _fragment(
        rows,
        "protocol_results",
        "plane",
        {"fingerprint": "f1", "gain": 12.676, "cpu": "AMD EPYC 7763"},
    )

    assert _stale_records().main(["merge", str(rows)]) == 0

    assert _snapshot(store) == before
    assert capsys.readouterr().out.split() == []
