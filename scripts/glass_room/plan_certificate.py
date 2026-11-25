"""Glass room study: the kept plans' KKT certificate read at wider active bands. The
oracle counts a limit active within ``oracle.ACTIVE`` (1e-6, scaled) of its edge, about
two float32 steps of a fining share; a plan resting a few steps inside its limits then
shows no active limit and a certificate of 1 whatever its stationarity. Each seed's best
plan of ``scripts/glass_slow_oracle.py`` is read again, not iterated, at each band.

Run: PYTHONPATH=src python scripts/glass_room/plan_certificate.py <ceil_dir> [scenario]
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402
from plan_traces import SEEDS, best_knots  # noqa: E402

from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.glass_furnace import oracle as O  # noqa: E402

BANDS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
_kkt, seen = O.kkt, {}


def at_each_band(grad, rows, jac, lo_gap, hi_gap):
    for band in BANDS:
        O.ACTIVE = band
        seen[band] = _kkt(grad, rows, jac, lo_gap, hi_gap)
    O.ACTIVE = BANDS[0]
    return seen[BANDS[0]]


O.kkt = at_each_band
name = sys.argv[2] if len(sys.argv) > 2 else "normal"
p = L.scenario_params(name)
n = int(p.max_steps_in_episode)
for seed in SEEDS:
    reset, keys = O.episode_keys(seed, n)
    _, state = L.ENV.reset_env(reset, p)
    plan = O.solve(
        L.ENV, plan_params(L.SPEC, p), state, keys, 120,
        start=best_knots(sys.argv[1], name, seed), max_iter=0,
    )  # fmt: skip
    read = {band: (round(k, 4), active) for band, (k, active) in seen.items()}
    print(name, seed, plan.how, round(plan.cost, 3), read, flush=True)
