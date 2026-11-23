"""The pH task's reference, placed by the exit rule on the real plant.

Contract (charter J1, owner, 2026-11-23): the permit is pH 6 to 8, cheap side low, and
the score charges every step outside it as an excursion, the price of a violation. The
reference holds the first tenth of a pH above 6 at which it leaves the permit on none
of the ladder's seeds in any scenario, each candidate run from episodes started on it.
The plant starts there (rule 17): the scored configuration's own set point is the kept
one, so an agent and the reference run the same episodes.

One full ladder build on the scored plant: bump tests at seven operating points, then
ten seeds of five 2 h scenarios per candidate. About four minutes.
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

    row = ladder.ROWS["ph_neutralization"]()
    rungs = ladder.build("ph_neutralization", row)
    _, setpoint = row.placed["pH on trim"]
    scan = row.scans["pH on trim"]
    grid = zone_grid(6.0, 8.0, "low")
    # The candidates run from the permit's low edge, in order, and stop at the kept
    # one; every one before it left the permit on some seed.
    assert list(scan) == pytest.approx(grid[: len(scan)])
    assert setpoint == pytest.approx(grid[len(scan) - 1])
    assert all(n > 0 for n in list(scan.values())[:-1])
    # Kept: no seed is charged an excursion, in any scenario.
    kept = ladder.Reference("pi", rungs["pi"])
    assert ladder.zone_exits("ph_neutralization", row, kept, ("excursion",)) == 0
    # Every later run starts on it, and so does the task as scored.
    assert row.score["setpoint_pH"] == pytest.approx(setpoint)
    scored = REGISTRY["ph_neutralization"].make_test_params()
    assert float(scored.setpoint_pH) == pytest.approx(setpoint)
