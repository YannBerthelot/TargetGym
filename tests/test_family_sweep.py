"""The family sweep's draws (charter B7, as amended 2026-11-10).

Contract: every ``ours`` number with a range is drawn inside it; numbers the table
names as coupled move together, each group on one shared coordinate, the ``lo`` end
of one with the ``lo`` end of the others.
"""

import importlib
import pathlib
import sys

import numpy as np

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
