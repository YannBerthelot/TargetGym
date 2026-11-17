"""Glass room study: the fair line, the planner without foresight on a measured spread.

The planner is the library's oracle without a seed (``glass_furnace/oracle.py``: a plan
every knot from the true state on the walks' mean path, the chance of a lost step
priced). There it reads its forecast's spread from the model's own walks. Here it is
handed the spread measured on the forecasts of the oracle as it shipped before, a plan
a day (``mpc_line.py`` at 7143497 on the five scenarios and seeds 100 to 104, the 154
days whose plan started inside its limits, measured 2026-11-16): nothing before 4 h,
then growing to 0.0070 of share at 16 h, and held there. The model's own spread is a
fifth to a third wider (0.0083 to 0.0096 at 16 h).

The report's fair line ran this script at 5108011, which carried its own copy of the
charge: there a lost step's worth was the pull at the plan's start, and the tail took
the 16 h spread whatever its lead.

Run: PYTHONPATH=src python scripts/glass_room/fair_line.py SCENARIO SEED OUT_DIR [MAX_ITER]
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import mpc_line  # noqa: E402

from target_gym.glass_furnace import oracle as O  # noqa: E402

#: Spread of the fining's forecast error at each hour of lead, 0 to 16 h, share.
SPREAD = np.array([0.0, 0.0, 1e-5, 3e-5, 0.00012, 0.00034, 0.00069, 0.00114, 0.00164,
                   0.00216, 0.0027, 0.00327, 0.00388, 0.00456, 0.00531, 0.00614, 0.00701])  # fmt: skip


def spread(dt):
    """The measured spread on each step of ``dt`` s up to 16 h of lead."""
    hours = (np.arange(int(round(16 * 3600.0 / dt))) + 1) * dt / 3600.0
    return np.interp(hours, np.arange(len(SPREAD)), SPREAD)


def make(env, params):
    return O.GlassOracle(env, params, spread=spread(float(params.delta_t)))


if __name__ == "__main__":
    mpc_line.main(make=make, tag="_fair")
