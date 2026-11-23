"""Where the slab building's reference holds the room (charter J1, zones).

Contract (owner, 2026-11-23, "lowest with no exit"): leaving the comfort band,
21 to 26 C, in office hours is charged as lost work (``PHYSICS.md`` sec. 4 and
5), so the room loop's set point is the lowest candidate of the zone's grid at
which the reference leaves the band on none of the ladder's seeds, in any
scenario. The grid is set from the zone alone: the cheap edge, 21 C, and ten
equal steps to the centre, 23.5 C, so 0.25 K apart.

Checked here without the ladder's own count of exits: the pi rung, which the
rule places, is run over the suite and the score's ``comfort`` term read
directly.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from target_gym import scenarios
from target_gym.benchmark import run_policy, task_info
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.large

BAND_LOW, BAND_CENTRE, STEP = 21.0, 23.5, 0.25
LOOP = "T_air on supply"


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


def test_the_set_point_is_a_candidate_of_the_bands_grid(built):
    _, row, _ = built
    _, setpoint = row.placed[LOOP]
    steps = (setpoint - BAND_LOW) / STEP
    assert steps == pytest.approx(round(steps), abs=1e-6)
    assert BAND_LOW <= setpoint <= BAND_CENTRE


def test_the_reference_is_charged_no_lost_work_on_any_seed(built):
    ladder, row, rungs = built
    spec = REGISTRY["hvac"]
    env = spec.make_env()
    policy = ladder.Reference("pi", rungs["pi"])
    for scenario in scenarios.suite(spec):
        params = scenarios.params_for(
            spec, scenario, sensor_noise_scale=ladder.NOISE, **row.score
        )
        task = task_info("hvac", env, params, len(ladder.SEEDS))
        rollout = run_policy(policy, env, params, task, seeds=ladder.SEEDS)
        charged = np.asarray(rollout["terms"]["comfort"]) > 0.0
        assert not charged.any(), (scenario.name, int(charged.any(axis=1).sum()))


def test_every_lower_candidate_leaves_the_band_on_some_seed(built):
    # The scan the build kept: it runs from the edge and stops at the first
    # candidate with no exit, so each one before the kept set point left.
    _, row, _ = built
    _, setpoint = row.placed[LOOP]
    scan = row.scans[LOOP]
    assert scan[setpoint] == 0
    assert all(left > 0 for point, left in scan.items() if point < setpoint)
