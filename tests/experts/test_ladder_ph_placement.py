"""The pH task's reference, placed by the exit rule on the real plant.

Contract (charter J1, owner, 2026-11-23): the permit is pH 6 to 9, cheap side low, and
the score charges every step outside it as an excursion, the price of a violation.
Each rung holds the first tenth of a pH above 6 at which it leaves the permit on none
of the ladder's seeds in any scenario, each candidate run from episodes started on it,
and the ladder keeps the cheapest rung that trips none and holds every seed it can.
The plant starts where the kept rung holds (rule 17): the scored configuration's own
set point is the kept one, so an agent and the reference run the same episodes.

One full ladder build on the scored plant: bump tests at seven operating points, then
ten seeds of five 2 h scenarios per candidate and per rung. About seven minutes.
"""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.large


@pytest.fixture(scope="module")
def ladder():
    spec = importlib.util.spec_from_file_location(
        "reference_ladder", ROOT / "scripts" / "reference_ladder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_ladder"] = module
    spec.loader.exec_module(module)
    return module


def test_the_ph_reference_holds_the_lowest_setpoint_that_never_leaves_the_permit(
    ladder,
):
    from target_gym.experts.ladder import zone_grid
    from target_gym.registry import REGISTRY

    name, loop = "ph_neutralization", "pH on trim"
    row = ladder.ROWS[name]()
    rungs = ladder.build(name, row)
    grid = zone_grid(6.0, 8.0, "low")
    # Each rung on its own: its candidates run from the permit's low edge, in order,
    # and stop at the one it keeps; every one before it left the permit on some
    # seed; none of its seeds is left out as unheld.
    assert set(row.placements) == set(rungs)
    for placed in (row.placements[k][loop] for k in rungs):
        scan = placed["exits"]
        assert list(scan) == pytest.approx(grid[: len(scan)])
        assert placed["setpoint"] == pytest.approx(grid[len(scan) - 1])
        assert all(list(scan.values())[:-1])
        assert placed["unheld"] == {} and placed["holds"]
    # Kept: a rung that is charged no excursion on any seed, in any scenario, from
    # episodes started on its own set point.
    results = {k: ladder.run_rung(name, row, k, c) for k, c in rungs.items()}
    kept = ladder.keep_rung(row, results)
    reference = ladder.Reference(kept, rungs[kept])
    assert ladder.zone_exits(name, row, reference, ("excursion",)) == ()
    # Every later run starts on it, and so does the task as scored.
    setpoint = row.placements[kept][loop]["setpoint"]
    assert row.placed[loop][1] == pytest.approx(setpoint)
    assert row.score["setpoint_pH"] == pytest.approx(setpoint)
    scored = REGISTRY[name].make_test_params()
    assert float(scored.setpoint_pH) == pytest.approx(setpoint)
