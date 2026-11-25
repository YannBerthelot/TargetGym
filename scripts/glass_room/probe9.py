"""Glass room study, probe 9: inside one transient. A planned pull change at hour 48,
walks off, under the cascade with its crown setpoint following the pull charged: what
the crown, the fined glass past the last port, the canal and its cooling loop do in the
hours after, and what the score charges for it.

Run: PYTHONPATH=src python scripts/glass_room/probe9.py <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

from target_gym.experts.ladder import Cascade, Pi, Rung, ladder_controller  # noqa: E402

P, E = L.P, L.E
H = L.HOUR
row = L.ladder_row()
base = L.scenario_params("normal", walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
PULL0 = float(base.pull)
a = P.layout(base.grid)
dz = np.asarray(a.dz)


def controller(sp, table=True):
    loops = (Pi(0, L.OUTER["kc"], L.OUTER["ti"], L._alpha(L.OUTER["tf"])),)
    inner = Pi(0, L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"]))
    rest = (
        (lambda o, xp: 1820.75 + 185.0 * (o[8] / PULL0 - 1.0))
        if table
        else (lambda o, xp: 1820.75)
    )
    cas = Cascade(lambda o, xp: o[0], rest, inner)
    rung = Rung("cascade", loops, np.ones((1, 1)), (cas,))
    return ladder_controller(
        rung,
        lambda o, xp: xp.stack([o[4]]),
        lambda o, xp: xp.stack([xp.asarray(sp, dtype=xp.float32)]),
        row.balance,
        L.DT,
    )


def run(c, p0, change, n):
    quiet = p0
    keys = jax.random.split(jax.random.PRNGKey(31), 25 * 24 * H)
    (s, m, _), _ = L.roll(c, quiet, L.start(c, quiet, 0), keys)
    p = p0.replace(pull_change=change, pull_change_window_h=(48.0, 48.0))
    st, _, _ = L.start(c, p, 0)
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )
    ks = jax.random.split(jax.random.PRNGKey(32), n)

    def f(carry, k):
        s0, mem, o = carry
        act, mem = c.step(c.params, mem, o, jnp)
        o1, s1, r, _, _ = L.ENV.step_env(k, s0, act, p)
        read = P.sensors(s1.plant, p.grid, p.plant)
        T = s1.plant.T
        return (s1, mem, o1), dict(
            cost=-r,
            X=read["fining"],
            hot=read["hot"],
            canal=read["canal"],
            valve=s1.fuel,
            cooling=s1.cooling / p.cooling_max,
            pull=s1.pull,
            refiner=jnp.sum(T[:5, 13] * a.dz[:5])
            / jnp.sum(a.dz[:5]),  # top 16 cm, unfired zone
            waist=jnp.sum(T[:5, 16] * a.dz[:5]) / jnp.sum(a.dz[:5]),
            end=jnp.sum(T[:5, 18] * a.dz[:5]) / jnp.sum(a.dz[:5]),
            lost=E.compute_reward_terms(s1, p)["tracking"] > 0,
        )

    return jax.lax.scan(f, (st, m, L.ENV.get_obs(st, p)), ks)[1]


N = 12 * 24 * H
res = {}
cases = [
    ("rise 30 %", 0.80, 0.30, 0.53),
    ("rise 42 %", 0.75, 0.42, 0.53),
    ("cut 30 %", 1.065, -0.296, 0.5523),
]
for label, fac, change, sp in cases:
    p0 = base.replace(pull=PULL0 * fac)
    for name, c in (
        ("table", controller(sp)),
        ("as recorded", controller(0.5523, table=False)),
    ):
        t = {
            k: np.asarray(v)
            for k, v in jax.jit(lambda c=c: run(c, p0, change, N))().items()
        }
        e = 48 * H
        b = slice(e - 6 * H, e)  # the six hours before
        out = {"lost_h": float(t["lost"].sum() / H)}
        for k in ("hot", "refiner", "waist", "end", "canal"):
            d = t[k][e:] - t[k][b].mean()
            i = int(np.argmax(np.abs(d)))
            rate = np.abs(
                t[k][e + H :] - t[k][e:-H]
            ).max()  # largest change in one hour
            out[k] = dict(
                before_C=float(t[k][b].mean() - 273.15),
                peak_dK=float(d[i]),
                peak_at_h=float(i / H),
                settled_dK=float(d[-1]),
                max_K_per_h=float(rate),
            )
        out["X"] = dict(
            before=float(t["X"][b].mean()),
            low=float(t["X"][e:].min()),
            low_at_h=float(np.argmin(t["X"][e:]) / H),
            high=float(t["X"][e:].max()),
            settled=float(t["X"][-1]),
        )
        out["cooling"] = dict(
            before=float(t["cooling"][b].mean()),
            max=float(t["cooling"][e:].max()),
            min=float(t["cooling"][e:].min()),
            hours_at_limit=float(
                ((t["cooling"][e:] > 0.999) | (t["cooling"][e:] < 0.001)).sum() / H
            ),
        )
        out["canal_over_h"] = float(
            (t["canal"][e:] > base.canal_target + 3.0).sum() / H
        )
        out["cost"] = dict(
            before=float(t["cost"][b].mean()),
            first_48h=float(t["cost"][e : e + 48 * H].mean()),
            settled=float(t["cost"][-24 * H :].mean()),
        )
        res[f"{label}, {name}"] = out
        print(f"{label}, {name}:", json.dumps(out), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
