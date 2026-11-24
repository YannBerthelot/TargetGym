"""Glass room study, probe 8: where the delay sits. Steps at a held crown, read on the
fining share of the glass leaving as the task scores it (every depth layer of the last
column alike) and on the forward stream alone (the top layers, flow weighted).

Run: PYTHONPATH=src python scripts/glass_room/probe8.py <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

P, E = L.P, L.E
H = L.HOUR
row = L.ladder_row()
quiet = L.scenario_params("normal", walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
a = P.layout(quiet.grid)
held = L.cascade(outer=False, bias=row.balance, crown_rest=1820.75)
keys = jax.random.split(jax.random.PRNGKey(7), 30 * 24 * H)
c0, _ = jax.jit(lambda: L.roll(held, quiet, L.start(held, quiet, 0), keys))()


def roll_layers(params, carry, n):
    ks = jax.random.split(jax.random.PRNGKey(9), n)

    def f(c, k):
        s, m, o = c
        act, m = held.step(held.params, m, o, jnp)
        o1, s1, r, _, _ = L.ENV.step_env(k, s, act, params)
        X, T = s1.plant.fining, s1.plant.T
        return (s1, m, o1), (
            X[:, -1],
            X[:, 15],
            T[:, -1],
            X[0, 11],
            s1.plant.m,
            s1.fuel,
        )

    return jax.lax.scan(f, carry, ks)[1]


pp = quiet.plant
steps = {
    "none": {},
    "band +10 K": dict(
        plant=pp.replace(T_fine_low=pp.T_fine_low + 10, T_fine_high=pp.T_fine_high + 10)
    ),
    "pull +10%": dict(pull=quiet.pull * 1.10),
    "sulfate +50%": dict(sulfate=quiet.sulfate * 1.5),
    "cullet -10 pts": dict(cullet=quiet.cullet - 0.10),
    "blanket +20%": dict(thickness=quiet.thickness * 1.2),
}
N = 5 * 24 * H
dz = np.asarray(a.dz)
res = {}
base = None
for label, ch in steps.items():
    Xe, Xw, Te, Xs, m, fuel = (
        np.asarray(v)
        for v in jax.jit(lambda p2=quiet.replace(**ch): roll_layers(p2, c0, N))()
    )
    scored = (Xe * dz).sum(1) / dz.sum()
    top = (Xe[:, :4] * dz[:4]).sum(1) / dz[:4].sum()  # top 8 cm of the last column
    upper = (Xe[:, :6] * dz[:6]).sum(1) / dz[:6].sum()  # top 32 cm
    if base is None:
        base = dict(scored=scored, top=top, upper=upper, waist=Xw, hot=Xs, line=m)
        print(
            "at rest: scored",
            scored[-1],
            "top 8 cm",
            top[-1],
            "top 32 cm",
            upper[-1],
            "exit layers",
            np.round(Xe[-1], 3),
            "T exit C",
            np.round(Te[-1] - 273.15, 0),
            "blanket kg/m2",
            np.round(m[-1], 1),
        )
        continue
    out = {}
    for name, series in (
        ("scored", scored),
        ("top 8 cm", top),
        ("top 32 cm", upper),
        ("hot zone surface", Xs),
    ):
        ref = base[
            {
                "scored": "scored",
                "top 8 cm": "top",
                "top 32 cm": "upper",
                "hot zone surface": "hot",
            }[name]
        ]
        d = series - ref
        final = d[-1]
        t10 = float(np.argmax(np.abs(d) > 0.1 * abs(final)) / H)
        t50 = float(np.argmax(np.abs(d) > 0.5 * abs(final)) / H)
        out[name] = dict(
            d6=float(d[6 * H]),
            d12=float(d[12 * H]),
            d24=float(d[24 * H]),
            d120=float(final),
            t10_h=t10,
            t50_h=t50,
        )
        print(
            f"{label:16s} {name:18s} 6h {d[6*H]:+.4f} 12h {d[12*H]:+.4f} 24h {d[24*H]:+.4f} "
            f"120h {final:+.4f}  10% at {t10:.1f} h, half at {t50:.1f} h",
            flush=True,
        )
    res[label] = out
json.dump(res, open(sys.argv[1], "w"), indent=1)
