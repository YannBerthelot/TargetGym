"""Where the slab building's reference holds the room (charter J1, zones).

Contract (owner, 2026-11-23, "lowest with no exit", each rung on its own):
leaving the comfort band, 21 to 26 C, in office hours is charged as lost work
(``PHYSICS.md`` sec. 4 and 5). So each rung of the ladder holds the lowest
candidate of the zone's grid at which it leaves the band on none of the
ladder's seeds, in any scenario, on the days an evaluation scores (4 to 10,
owner, 2026-11-24: the burn-in and the ceiling's tail are read by no line). The grid is set from the zone alone: the cheap
edge, 21 C, and ten equal steps to the centre, 23.5 C, so 0.25 K apart.

Checked here without the ladder's own count of exits: each rung is run over
the suite at its own set point and the score's ``comfort`` term read directly.

One ladder build on the scored plant (bump tests, then ten seeds of three
twelve-day runs per candidate and per rung) and one suite per rung.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from target_gym import evaluation, scenarios
from target_gym.benchmark import run_policy, task_info
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.large

BAND_LOW, BAND_CENTRE, STEP = 21.0, 23.5, 0.25
LOOP = "T_air on supply"
#: The rungs the row has: one room loop, the balance's heating curve, the core's
#: inner loop (``scripts/ladder_rows/hvac.py``).
RUNGS = ("pi", "feedforward", "cascade")


@pytest.fixture(scope="module")
def built():
    spec = importlib.util.spec_from_file_location(
        "reference_ladder", ROOT / "scripts" / "reference_ladder.py"
    )
    ladder = importlib.util.module_from_spec(spec)
    sys.modules["reference_ladder"] = ladder
    spec.loader.exec_module(ladder)
    row = ladder.ROWS["hvac"]()
    return ladder, row, ladder.build("hvac", row)


@pytest.mark.parametrize("rung", RUNGS)
def test_each_rungs_set_point_is_a_candidate_of_the_bands_grid(built, rung):
    _, row, _ = built
    setpoint = row.placements[rung][LOOP]["setpoint"]
    steps = (setpoint - BAND_LOW) / STEP
    assert steps == pytest.approx(round(steps), abs=1e-6)
    assert BAND_LOW <= setpoint <= BAND_CENTRE


@pytest.mark.parametrize("rung", RUNGS)
def test_each_rung_is_charged_no_lost_work_on_any_seed(built, rung):
    ladder, row, rungs = built
    spec = REGISTRY["hvac"]
    env = spec.make_env()
    policy = ladder.Reference(rung, rungs[rung])
    for scenario in scenarios.suite(spec):
        params = scenarios.params_for(
            spec, scenario, sensor_noise_scale=ladder.NOISE, **row.score
        )
        task = task_info("hvac", env, params, len(ladder.SEEDS))
        rollout = run_policy(policy, env, params, task, seeds=ladder.SEEDS)
        charged = np.asarray(rollout["terms"]["comfort"]) > 0.0  # (seeds, steps)
        charged = charged[:, evaluation.scored_window(spec, charged.shape[1])]
        assert not charged.any(), (scenario.name, int(charged.any(axis=1).sum()))


@pytest.mark.parametrize("rung", RUNGS)
def test_every_lower_candidate_leaves_the_band_on_some_seed(built, rung):
    # The scan the build kept for this rung: it runs from the edge and stops at
    # the first candidate with no exit, so each one before its set point left.
    _, row, _ = built
    own = row.placements[rung][LOOP]
    scan = own["exits"]
    assert not scan[own["setpoint"]]
    assert all(left for point, left in scan.items() if point < own["setpoint"])
