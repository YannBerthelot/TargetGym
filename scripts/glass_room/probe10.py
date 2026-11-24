"""Glass room study, probe 10: what the sourced transient mechanisms would see. Under
the cascade with its crown setpoint following the pull, on the normal day, in a gas
swing and in a pull change: how far and how fast the crown, the bath where it touches
the tank blocks, and the fined glass past the last port move.

Sourced marks (REALITY.md section 10): crown swings of 25 to 30 K (corrosion about twice
as fast), a 50 K dip and return at the glass-contact blocks (bubbles, knots, stones),
1400 C where glass holding 0.3 wt % SO3 starts to release it when heated.

Run: PYTHONPATH=src python scripts/glass_room/probe10.py <out.json>
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
base = L.scenario_params("normal")
PULL0 = float(base.pull)
a = P.layout(base.grid)
dz = a.dz


def controller(sp=0.5523):
    loops = (Pi(0, L.OUTER["kc"], L.OUTER["ti"], L._alpha(L.OUTER["tf"])),)
    inner = Pi(0, L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"]))
    rest = lambda o, xp: 1820.75 + 185.0 * (o[8] / PULL0 - 1.0)  # noqa: E731
    rung = Rung(
        "cascade", loops, np.ones((1, 1)), (Cascade(lambda o, xp: o[0], rest, inner),)
    )
    return ladder_controller(
        rung,
        lambda o, xp: xp.stack([o[4]]),
        lambda o, xp: xp.stack([xp.asarray(sp, dtype=xp.float32)]),
        row.balance,
        L.DT,
    )


C = controller()


def run(p0, p, seed, n):
    quiet = p0.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
    keys = jax.random.split(jax.random.PRNGKey(31), 25 * 24 * H)
    (s, m, _), _ = L.roll(C, quiet, L.start(C, quiet, 0), keys)
    st, _, _ = L.start(C, p, seed)
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )

    def f(carry, k):
        s0, mem, o = carry
        act, mem = C.step(C.params, mem, o, jnp)
        o1, s1, r, _, _ = L.ENV.step_env(k, s0, act, p)
        T = s1.plant.T
        return (s1, mem, o1), dict(
            crown=jnp.max(s1.plant.T_roof[0]),
            melter_floor=T[-1, 7],  # bottom glass under the hot spot
            melter_mean=jnp.sum(T[:, 8] * dz)
            / jnp.sum(dz),  # depth mean past the spring zone
            hot_surface=jnp.sum(T[:4, 10] * dz[:4])
            / jnp.sum(dz[:4]),  # top 8 cm, fining zone
            first_unfired=jnp.sum(T[:5, 12] * dz[:5]) / jnp.sum(dz[:5]),  # top 16 cm
            refiner=jnp.sum(T[:5, 13] * dz[:5]) / jnp.sum(dz[:5]),
            waist_mean=jnp.sum(T[:, 16] * dz) / jnp.sum(dz),
            lost=E.compute_reward_terms(s1, p)["tracking"] > 0,
        )

    return jax.lax.scan(f, (st, m, L.ENV.get_obs(st, p)), L.episode_keys(seed, n)[1])[1]


def marks(x):
    x = np.asarray(x) - 273.15
    w = 12 * H
    rise12 = max(float((x[w:] - x[:-w]).max()), 0.0)
    fall12 = max(float((x[:-w] - x[w:]).max()), 0.0)
    rise1 = float((x[H:] - x[:-H]).max())
    return dict(
        mean=float(x.mean()),
        lo=float(x.min()),
        hi=float(x.max()),
        span=float(x.max() - x.min()),
        rise_12h=rise12,
        fall_12h=fall12,
        rise_1h=rise1,
    )


def swings(x, size):
    """Reversals larger than ``size`` K in the trace (a rainflow-like count of half cycles)."""
    x = np.asarray(x)
    n, direction = 0, 0
    hi = lo = x[0]
    for v in x[1:]:
        hi, lo = max(hi, v), min(lo, v)
        if direction >= 0 and hi - v > size:
            n, direction, lo = n + 1, -1, v
            hi = v
        elif direction <= 0 and v - lo > size:
            n, direction, hi = n + 1, 1, v
            lo = v
    return n


N = 5040
cases = {
    "normal day": (base, base, (100, 101, 102, 103, 104)),
    "gas lean 6 %": (base, L.scenario_params("gas_lean"), (100, 101, 102)),
    "gas rich 6 %": (base, L.scenario_params("gas_rich"), (100, 101, 102)),
    "pull rise 10 %": (base, L.scenario_params("pull_up"), (100, 101, 102)),
    "pull rise 30 %": (
        base.replace(pull=PULL0 * 0.8),
        base.replace(pull=PULL0 * 0.8, pull_change=0.30),
        (100, 101, 102),
    ),
    "pull cut 30 %": (
        base.replace(pull=PULL0 * 1.065),
        base.replace(pull=PULL0 * 1.065, pull_change=-0.296),
        (100, 101, 102),
    ),
}
res = {}
for label, (p0, p, seeds) in cases.items():
    f = jax.jit(lambda seed, p0=p0, p=p: run(p0, p, seed, N))
    rows = []
    for seed in seeds:
        t = {k: np.asarray(v) for k, v in f(seed).items()}
        r = {k: marks(v) for k, v in t.items() if k != "lost"}
        r["crown_swings_25K"] = swings(t["crown"], 25.0)
        r["crown_swings_10K"] = swings(t["crown"], 10.0)
        r["h_first_unfired_over_1400"] = float((t["first_unfired"] > 1673.15).sum() / H)
        r["lost_h"] = float(t["lost"].sum() / H)
        rows.append(r)
    agg = {}
    for k in rows[0]:
        if isinstance(rows[0][k], dict):
            agg[k] = {q: float(np.mean([r[k][q] for r in rows])) for q in rows[0][k]}
        else:
            agg[k] = float(np.mean([r[k] for r in rows]))
    res[label] = agg
    print(
        label,
        "| crown span %.0f K, rise in 1 h %.0f K, swings over 25 K: %.1f, over 10 K: %.1f"
        % (
            agg["crown"]["span"],
            agg["crown"]["rise_1h"],
            agg["crown_swings_25K"],
            agg["crown_swings_10K"],
        ),
        "| melter floor span %.1f, 12 h rise %.1f"
        % (agg["melter_floor"]["span"], agg["melter_floor"]["rise_12h"]),
        "| bath mean span %.1f, 12 h rise %.1f"
        % (agg["melter_mean"]["span"], agg["melter_mean"]["rise_12h"]),
        "| fining-zone surface span %.1f, 12 h rise %.1f"
        % (agg["hot_surface"]["span"], agg["hot_surface"]["rise_12h"]),
        "| first unfired: mean %.0f C, span %.1f, 12 h rise %.1f, h over 1400 C %.0f"
        % (
            agg["first_unfired"]["mean"],
            agg["first_unfired"]["span"],
            agg["first_unfired"]["rise_12h"],
            agg["h_first_unfired_over_1400"],
        ),
        "| refiner span %.1f, 12 h rise %.1f"
        % (agg["refiner"]["span"], agg["refiner"]["rise_12h"]),
        "| waist mean span %.1f" % agg["waist_mean"]["span"],
        "| lost %.1f h" % agg["lost_h"],
        flush=True,
    )
json.dump(res, open(sys.argv[1], "w"), indent=1)
