"""``scripts/evaluate_baselines.py``'s summary table, printed after every run.

Contract: the table renders every recorded row. A row scored on the job's
cost terms has no tracking split (no ``hold_tracking``: cstr's lost
production and off-spec, the compressor's energy and zone), so its room and
MPC score read as not available ("—"), and the rest of its line is printed.
The table runs after the rows are written, so a crash there gives a run that
recorded everything a failing exit code (2026-11-26, cstr's re-record).
"""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "evaluate_baselines_under_test", ROOT / "scripts" / "evaluate_baselines.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _part(**extra):
    return {"gain": 0.5, "hold": 0.2, "reach_cost": 0.1, "failure_rate": 0.0, **extra}


def _cells(table: str, name: str) -> list[str]:
    (line,) = [x for x in table.splitlines() if x.startswith(f"| `{name}` |")]
    return [c.strip() for c in line.strip("|").split("|")]


def test_a_row_without_a_tracking_split_prints_with_room_and_score_unavailable():
    ev = _load()
    rows = {"job_task": {"pid": _part(), "mpc": _part(hold=0.1), "ceiling": 0.05}}
    cells = _cells(ev.table(rows), "job_task")
    # columns: plant, ceiling, PID gain, MPC gain, room, MPC score, PID hold, MPC hold, ...
    assert cells[4] == "—" and cells[5] == "—"
    assert cells[6] == "0.2" and cells[7] == "0.1"


def test_a_row_with_a_tracking_split_still_gets_its_room():
    ev = _load()
    rows = {"track_task": {"pid": _part(hold_tracking=0.2), "ceiling": 0.1}}
    cells = _cells(ev.table(rows), "track_task")
    assert cells[4] != "—"
