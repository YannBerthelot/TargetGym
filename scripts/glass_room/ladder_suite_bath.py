"""Glass room study: the ladder's suite and seeds under a loop on the bath's true mean
temperature, which no instrument reads: a lower bound on the room over the kept rung,
on the ladder's own protocol (five scenarios, seeds 100 to 109, noise 1, a week from
the reset).

The crown setpoint follows the pull charged; on top of it a PI by SIMC on the crown
setpoint's step (proto_bath.py) holds the bath's mass-mean temperature where the reset
put it.

Run: PYTHONPATH=src python scripts/glass_room/ladder_suite_bath.py <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym import scenarios  # noqa: E402

P, E, H, DT = L.P, L.E, L.HOUR, L.DT
SEEDS = tuple(range(100, 110))
ROW = L.ladder_row()
G_TABLE, CROWN_AT_SETPOINT = 185.0, 1820.75
PULL0 = float(L.scenario_params("normal").pull)
KC, TI = 28.54, 1.9 * 3600.0
ki, tii, ai = L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"])


def bath_mean(plant, grid):
    a = P.layout(grid)
    w = a.dz[:, None] * a.dx * a.wc
    return jnp.sum(plant.T * w) / jnp.sum(w)


def episode(p, seed):
    k_reset, keys = L.episode_keys(seed, int(p.max_steps_in_episode))
    obs, st = L.ENV.reset_env(k_reset, p)
    target = bath_mean(st.plant, p.grid)
    crown0 = obs[0]  # the crown the reset state holds

    def f(c, k):
        s, (fi, Ii, Io), o = c
        e = target - bath_mean(s.plant, p.grid)
        Io2 = Io + e * DT
        trim = KC * (e + Io2 / TI)
        fi = fi + ai * (o[0] - fi)
        ei = crown0 + G_TABLE * (o[8] / PULL0 - 1.0) + trim - fi
        Ii2 = Ii + ei * DT
        asked = ROW.balance(o, jnp)[0] + ki * (ei + Ii2 / tii)
        u = jnp.clip(asked, -1.0, 1.0)
        moved = u != asked
        o1, s1, r, _, _ = L.ENV.step_env(k, s, jnp.reshape(u, (1,)), p)
        tr = L.trace_of(s1, r, p)
        return (
            s1,
            (fi, jnp.where(moved, Ii, Ii2), jnp.where(moved, Io, Io2)),
            o1,
        ), dict(
            cost=tr["cost"],
            fuel=tr["fuel_cost"],
            X=tr["X"],
            canal=tr["canal"],
            pull=tr["pull"],
        )

    return jax.lax.scan(f, (st, (obs[0], jnp.zeros(()), jnp.zeros(())), obs), keys)[1]


weights = scenarios.weights(L.SPEC)
per = {}
for name in weights:
    p = L.scenario_params(name)
    tr = {
        k: np.asarray(v)
        for k, v in jax.jit(jax.vmap(lambda seed, p=p: episode(p, seed)))(
            np.asarray(SEEDS)
        ).items()
    }
    worth = tr["pull"] * DT / 1000.0 * float(p.lost_glass_price)
    mark = (tr["canal"] < float(p.canal_target)) | (tr["X"] < float(p.fining_pass))
    per[name] = dict(
        cost=tr["cost"].mean(1),
        fuel=tr["fuel"].mean(1),
        lost=((tr["cost"] - tr["fuel"]) / worth).mean(1),
        mark_h=mark.sum(1) / H,
        X_min=tr["X"].min(1),
    )
J = sum(weights[n] * per[n]["cost"] for n in per)
out = dict(
    J=[float(x) for x in J],
    J_median=float(np.median(J)),
    J_mean=float(J.mean()),
    scenarios={n: {k: float(v.mean()) for k, v in d.items()} for n, d in per.items()},
)
print(
    f"loop on the bath's true mean: J median {out['J_median']:.2f}, mean {out['J_mean']:.2f}"
)
for n, d in out["scenarios"].items():
    print(
        f"   {n}: cost {d['cost']:.2f}, fuel {d['fuel']:.2f}, glass lost {100*d['lost']:.2f} %, under a mark {d['mark_h']:.1f} h, X min {d['X_min']:.3f}"
    )
json.dump(out, open(sys.argv[1], "w"), indent=1)
