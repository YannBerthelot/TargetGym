"""The family sweep's draws (charter B7, as amended 2026-11-10).

Contract: every ``ours`` number with a range is drawn inside it; numbers the table
names as coupled move together, each group on one shared coordinate, the ``lo`` end
of one with the ``lo`` end of the others.
"""

import importlib
import pathlib
import sys

import numpy as np
import pytest

from target_gym import charter
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sweep = importlib.import_module("family_sweep")

TABLE = {
    "ranges": {
        "a": {"lo": -20.0, "hi": -1.0},
        "b": {"lo": -10.9, "hi": -6.7},
        "c": {"lo": 900.0, "hi": 1800.0},
    },
    "coupled": [["a", "b"]],
}


def where(d, name):
    r = TABLE["ranges"][name]
    return (d[name] - r["lo"]) / (r["hi"] - r["lo"])


def test_every_draw_is_inside_its_ranges():
    for d in sweep.draws(TABLE):
        for name, r in TABLE["ranges"].items():
            assert min(r["lo"], r["hi"]) <= d[name] <= max(r["lo"], r["hi"])


def test_coupled_numbers_sit_at_the_same_point_of_their_ranges():
    for d in sweep.draws(TABLE):
        assert abs(where(d, "a") - where(d, "b")) < 1e-9


def test_an_uncoupled_number_is_drawn_on_its_own():
    draws = sweep.draws(TABLE)
    spread = [where(d, "c") - where(d, "a") for d in draws]
    assert np.std(spread) > 0.1


# --- the shared runner (every charter task) -----------------------------------
# Contract (charter B7, owner 2026-11-12): a task whose ours numbers have no
# sourced range is swept on its shipped member alone, one draw that changes
# nothing; a number the table derives from drawn ones follows them on each draw;
# a draw overrides the scenario's and the row's own changes, and an integer field
# stays an integer. On a draw the property holds when the ladder keeps a rung
# (one that trips no seed) and that rung's median beats every null's in every
# scenario.


def test_a_table_without_ranges_is_swept_on_the_shipped_member_alone():
    assert sweep.draws({"ranges": {}, "no_range": ["x"]}) == [{}]


def test_a_derived_number_follows_its_drawn_numbers():
    table = {
        "ranges": {"rate": {"lo": 2.0, "hi": 4.0}, "price": {"lo": 30.0, "hi": 150.0}},
        "derived": {"cost": "186.7 / rate / 1000.0 * price"},
    }
    for d in sweep.draws(table):
        assert d["cost"] == pytest.approx(186.7 / d["rate"] / 1000.0 * d["price"])


class _Params(dict):
    def replace(self, **kw):
        return _Params({**self, **kw})


class _Spec:
    """A spec's one seam here: ``make_test_params(**changes)``."""

    name = "toy"
    default = _Params(alpha=1.0, steps=10, load=0.5)

    def make_test_params(self, **kw):
        return self.default.replace(**kw)


def test_a_draw_overrides_the_scenario_and_keeps_integers_whole():
    drawn = sweep.Drawn(_Spec(), {"alpha": 3.0, "steps": 12.6})
    p = drawn.make_test_params(alpha=9.0, load=0.8)  # a scenario's own changes
    assert p == {"alpha": 3.0, "steps": 13, "load": 0.8}
    assert isinstance(p["steps"], int)


def _run(medians: dict, trips: int = 0) -> dict:
    return {
        "trips": trips,
        "scenarios": {s: {"costs": [m - 1, m, m + 1]} for s, m in medians.items()},
    }


def test_the_property_holds_when_the_kept_rung_beats_every_null_everywhere():
    ref = _run({"normal": 1.0, "storm": 2.0})
    nulls = {"hold": _run({"normal": 3.0, "storm": 2.5})}
    assert sweep.holds(ref, nulls) is True


def test_a_null_that_wins_one_scenario_breaks_the_property():
    ref = _run({"normal": 1.0, "storm": 2.0})
    nulls = {"hold": _run({"normal": 3.0, "storm": 1.5})}
    assert sweep.holds(ref, nulls) is False


def test_no_rung_kept_or_no_null_run_breaks_the_property():
    assert sweep.holds(None, {"hold": _run({"normal": 3.0})}) is False
    assert sweep.holds(_run({"normal": 1.0}), {}) is False


# --- every charter task has a table (B7) --------------------------------------


@pytest.mark.parametrize("task", charter.charter_tasks())
def test_every_charter_task_has_a_sweep_table_on_its_own_params(task):
    table = sweep.table(task)
    assert table is not None, f"{task}: no ours.json"
    default = REGISTRY[task].make_test_params()
    names = [*table["ranges"], *table.get("derived", {})]
    for name in names:
        assert hasattr(default, name), f"{task}: {name} is not a parameter"
    for name, r in table["ranges"].items():
        assert r["lo"] != r["hi"] and r["why"], f"{task}: {name}"
