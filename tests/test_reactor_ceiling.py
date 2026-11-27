"""The reactor's ceiling file, as the charter record reads it.

Contract: ``charter.ceiling_line`` is the reader (the record's ``ceiling`` part),
so the file the reactor's script assembles must read there as a J3 line: every
scenario's per-seed costs, ``J`` their means weighted by the scenarios' weights,
converged only if every seed's solves were. A day measured twice, or a seed not
measured, writes no file: a partial ceiling would read as a cheaper one.

By hand, two scenarios of weights 0.75 and 0.25 on seeds 100 and 101: costs 2 and
4 (mean 3) and 10 and 30 (mean 20) give J = 0.75 x 3 + 0.25 x 20 = 7.25.
"""

import importlib
import pathlib
import sys

import pytest

from target_gym import charter

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ceiling = importlib.import_module("reactor_ceiling")

WEIGHTS = {"calm": 0.75, "busy": 0.25}
SEEDS = (100, 101)


def _piece(scenario, seeds, costs, trips=None, converged=None):
    n = len(seeds)
    return {
        scenario: {
            "seeds": list(seeds),
            "costs": list(costs),
            "trips": list(trips or [0] * n),
            "converged": list(converged or [True] * n),
        }
    }


def test_the_pieces_read_as_the_records_ceiling_line():
    pieces = [
        _piece("calm", [100], [2.0]),
        _piece("busy", [100, 101], [10.0, 30.0], trips=[0, 1]),
        _piece("calm", [101], [4.0]),
    ]
    stored = ceiling.assemble(pieces, WEIGHTS, SEEDS)
    line = charter.ceiling_line(stored, WEIGHTS)
    assert line["scenarios"]["calm"]["costs"] == [2.0, 4.0]
    assert line["scenarios"]["busy"]["costs"] == [10.0, 30.0]
    assert line["J"] == pytest.approx(7.25)
    assert line["trips"] == 1
    # every seed's solves converged; with one start and no recorded stop rule its
    # agreement is not measured, so as a ceiling it is not converged (shared
    # rules, 2026-11-27), while as an MPC line (no starts to agree) it is
    assert line["converged"] is False
    assert charter.ceiling_line(stored, WEIGHTS, agreement=False)["converged"] is True


def test_one_seed_whose_solves_did_not_converge_marks_the_line():
    pieces = [
        _piece("calm", SEEDS, [2.0, 4.0]),
        _piece("busy", SEEDS, [10.0, 30.0], converged=[True, False]),
    ]
    line = charter.ceiling_line(ceiling.assemble(pieces, WEIGHTS, SEEDS), WEIGHTS)
    assert line["converged"] is False


@pytest.mark.parametrize(
    "pieces",
    [
        [_piece("calm", SEEDS, [2.0, 4.0])],  # a scenario not measured
        [_piece("calm", SEEDS, [2.0, 4.0]), _piece("busy", [100], [10.0])],
        [_piece("calm", SEEDS, [2.0, 4.0]), _piece("busy", [100, 100], [1.0, 2.0])],
        [_piece("calm", SEEDS, [2.0, 4.0]), _piece("idle", SEEDS, [1.0, 2.0])],
    ],
    ids=["no_scenario", "no_seed", "seed_twice", "unknown_scenario"],
)
def test_a_partial_or_doubled_measurement_writes_no_file(pieces):
    with pytest.raises(SystemExit):
        ceiling.assemble(pieces, WEIGHTS, SEEDS)


def test_the_script_measures_the_reactors_own_scenarios_on_the_ladders_seeds():
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    weights = scenarios.weights(REGISTRY["reactor"])
    pieces = [_piece(s, ceiling.SEEDS, [1.0] * 10) for s in weights]
    stored = ceiling.assemble(pieces, weights)
    assert ceiling.SEEDS == tuple(range(100, 110))
    assert charter.ceiling_line(stored, weights)["J"] == pytest.approx(1.0)
