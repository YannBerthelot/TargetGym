"""The horizon files (H13): a compact summary per task in the package, and
the full evidence outside it, in git but not in the wheel.

Contract (docs/proposals/oracle-horizon-protocol.md, H13): the summary in
``data/oracle_horizons/<task>.json`` holds T_c, N, M, P and the pair and seed
that set each when the task is compliant, none of them when it is not, the
class per CV and the label counts, in under a kilobyte; it names the evidence
file, which lives under ``evidence/`` at the repository root. A task that falls
non-compliant keeps its current oracle (N3): where that oracle plans on a
horizon, the summary keeps the last compliant one and names the measurement it
came from (``horizon_kept_from``), as cstr-v6 does (owner, 2026-11-09).
"""

import json
import pathlib

import pytest

from target_gym import records

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUMMARIES = sorted((records.DATA / "oracle_horizons").glob("*.json"))
HORIZON = ("T_c_steps", "N", "M", "P", "T_c_set_by", "P_set_by")


@pytest.mark.parametrize("path", SUMMARIES, ids=lambda p: p.stem)
def test_a_summary_is_compact_and_its_evidence_is_outside_the_package(path):
    row = json.loads(path.read_text())
    assert path.stat().st_size < 1024
    assert {"compliant", "classes", "counts", "protocol", "cpu", "generated"} <= set(
        row
    )
    planned = row["compliant"] or "horizon_kept_from" in row
    assert all(k in row for k in HORIZON) == planned
    assert not any(k in row for k in HORIZON) or planned
    assert not (row["compliant"] and "horizon_kept_from" in row)
    evidence = ROOT / row["evidence"]
    assert evidence.is_file() and records.DATA.parent not in evidence.parents
    assert "pairs" in json.loads(evidence.read_text())


@pytest.mark.parametrize("task", ["plane_energy", "plane_sine"])
def test_tasks_sharing_a_plant_take_its_test(task):
    """H24: a plant's test runs once, on the shortest episode's window, so these
    tasks read `plane`'s T_c and P, and only their planned stages differ."""
    own, plane = (records.load_row("oracle_horizons", n) for n in (task, "plane"))
    assert (own["T_c_steps"], own["P"]) == (plane["T_c_steps"], plane["P"])


AIRCRAFT_TESTS = ("plane", "plane3d_heading", "patrol")


@pytest.mark.parametrize("task", AIRCRAFT_TESTS)
def test_the_record_names_the_cv_that_tripped(task):
    """H9, with every trip limit a CV (H6): a trip names what tripped."""
    evidence = ROOT / records.load_row("oracle_horizons", task)["evidence"]
    pairs = json.loads(evidence.read_text())["pairs"]
    trips = [t for p in pairs for test in p["tests"] for t in test["trips"].values()]
    assert trips, "these tests trip, so this has something to read"
    assert all(t["cv"] is not None for t in trips)


def test_a_non_compliant_cstr_keeps_the_horizon_its_oracle_plans_on():
    """cstr-v6's water-on-T bump tests are all invalid, so it is non-compliant;
    its oracle keeps the horizon measured on v5 (T_c 1, P 32) and still builds."""
    from target_gym.registry import REGISTRY

    row = records.load_row("oracle_horizons", "cstr")
    assert not row["compliant"] and row["counts"]["valid"] == 0
    assert (row["T_c_steps"], row["P"]) == (1, 32)
    assert row["horizon_kept_from"]["generated"] == "2026-11-03T20:59:53+00:00"
    spec = REGISTRY["cstr"]
    spec.make_mpc(spec.make_env(), spec.make_test_params())
