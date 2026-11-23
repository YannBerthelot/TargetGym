"""Where the slab building's reference holds the room (charter J1, zones).

Contract (owner, 2026-11-23): the comfort band, 21 to 26 C, is priced in office
hours only (08:00 to 18:00, ``PHYSICS.md`` sec. 4 and 5), and the room swings
through the day. What can fall under the band is each day's occupied minimum, so
on every rung of the ladder the room loop's setpoint is where those minima
average two of their own standard deviations above 21 C. They are read as the
ladder reads them: the room sensor, on the bump-test seeds 100 to 109, over the
second week of the normal fortnight.

The minima are taken here straight from the schedule: 15 min steps, so a day is
96 readings and its office hours are readings 32 to 71.
"""

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.large

BAND_LOW, BAND_CENTRE = 21.0, 23.5
DAY, OPENS, CLOSES = 96, 32, 72
SECOND_WEEK = range(7, 14)


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


@pytest.mark.parametrize("rung", ["pi", "feedforward", "cascade"])
def test_each_rung_holds_the_days_occupied_minimum_two_sigma_above_the_band(
    built, rung
):
    ladder, row, rungs = built
    sigma, setpoint = row.placed[f"T_air on supply [{rung}]"]
    assert sigma > 0
    assert BAND_LOW < setpoint < BAND_CENTRE
    rooms = ladder.cv_traces("hvac", row, rungs[rung], 0)
    minima = np.array(
        [
            np.min(room[d * DAY + OPENS : d * DAY + CLOSES])
            for room in rooms
            for d in SECOND_WEEK
        ]
    )
    # 70 minima: their mean has a standard error of sigma / sqrt(70); four of them.
    tolerance = 4.0 * sigma / len(minima) ** 0.5
    assert np.mean(minima) == pytest.approx(BAND_LOW + 2.0 * sigma, abs=tolerance)
    assert np.std(minima, ddof=1) == pytest.approx(sigma, rel=0.2)
