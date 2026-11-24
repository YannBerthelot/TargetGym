"""Glass room study: the ladder's own suite and seeds (five scenarios, seeds 100 to
109, sensor noise 1, a week from the reset), under the kept cascade as the ladder tuned
it on 2026-11-13 and under two variants of it:

- the crown setpoint following the pull charged, which is what the row's balance
  feedforward would do if the crown loop did not integrate it away;
- that table alone, with no quality loop.

Run: PYTHONPATH=src python scripts/glass_room/ladder_suite.py <out.json>
"""

import json
import pathlib
import sys

import jax
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym import scenarios  # noqa: E402
from target_gym.experts.ladder import Cascade, Pi, Rung, ladder_controller  # noqa: E402

H, DT = L.HOUR, L.DT
SEEDS = tuple(range(100, 110))
ROW = L.ladder_row()
G_TABLE, CROWN_AT_SETPOINT = (
    185.0,
    1820.75,
)  # K per unit pull; K where fining holds 0.5523
PULL0 = float(L.scenario_params("normal").pull)


def cascade(table, outer=True):
    kc = L.OUTER["kc"] if outer else 0.0
    loops = (Pi(0, kc, L.OUTER["ti"], L._alpha(L.OUTER["tf"])),)
    inner = Pi(0, L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"]))
    base = L.CROWN_REST if outer else CROWN_AT_SETPOINT
    if table:
        rest = lambda o, xp: base + G_TABLE * (o[8] / PULL0 - 1.0)  # noqa: E731
    else:
        rest = lambda o, xp: base  # noqa: E731
    rung = Rung(
        "cascade", loops, np.ones((1, 1)), (Cascade(lambda o, xp: o[0], rest, inner),)
    )
    read = lambda o, xp: xp.stack([o[4]])  # noqa: E731
    target = lambda o, xp: xp.stack(
        [xp.asarray(L.PLACED, dtype=xp.float32)]
    )  # noqa: E731
    return ladder_controller(rung, read, target, ROW.balance, DT)


POLICIES = {
    "the kept cascade, as tuned": cascade(False),
    "its crown setpoint following the pull": cascade(True),
    "the crown table alone, no quality loop": cascade(True, outer=False),
}
weights = scenarios.weights(L.SPEC)
res = {}
for label, c in POLICIES.items():
    per = {}
    for name in weights:
        p = L.scenario_params(name)
        f = jax.jit(jax.vmap(lambda seed, p=p, c=c: L.run_controller(c, p, seed)))
        tr = {k: np.asarray(v) for k, v in f(np.asarray(SEEDS)).items()}
        worth = tr["pull"] * DT / 1000.0 * float(p.lost_glass_price)
        mark = (tr["canal"] < float(p.canal_target)) | (tr["X"] < float(p.fining_pass))
        per[name] = dict(
            cost=tr["cost"].mean(1),
            fuel=tr["fuel_cost"].mean(1),
            lost=((tr["cost"] - tr["fuel_cost"]) / worth).mean(1),
            mark_h=mark.sum(1) / H,
        )
    J = sum(weights[n] * per[n]["cost"] for n in per)
    res[label] = dict(
        J=[float(x) for x in J],
        J_median=float(np.median(J)),
        J_mean=float(J.mean()),
        scenarios={
            n: {k: float(v.mean()) for k, v in d.items()} for n, d in per.items()
        },
        fuel=float(sum(weights[n] * per[n]["fuel"].mean() for n in per)),
        lost=float(sum(weights[n] * per[n]["lost"].mean() for n in per)),
        mark_h={n: float(per[n]["mark_h"].mean()) for n in per},
    )
    r = res[label]
    print(
        f"{label}: J median {r['J_median']:.2f}, mean {r['J_mean']:.2f} | fuel {r['fuel']:.2f} | glass lost "
        f"{100*r['lost']:.2f} % | hours under a mark: "
        + ", ".join(f"{n} {h:.1f}" for n, h in r["mark_h"].items()),
        flush=True,
    )
    print(
        "   per scenario: "
        + ", ".join(f"{n} {d['cost']:.2f}" for n, d in r["scenarios"].items()),
        flush=True,
    )
json.dump(res, open(sys.argv[1], "w"), indent=1)
