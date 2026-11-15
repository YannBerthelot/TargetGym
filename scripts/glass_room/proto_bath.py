"""Glass room study, prototype: what moves the bath, and whether a loop on the bath
itself can hold it still.

1. The blocks' upward travel over a week at a held crown, with every walk, each alone
   and none.
2. A step of the crown setpoint at a held crown: how the bath's mean, the floor couple
   under bay 3 and the floor couple at the melter's end answer.
3. A second loop on top of the held crown, by SIMC on that step, on (a) the bath's true
   mean temperature and (b) the floor couple under bay 3 as the agent reads it.

Run: PYTHONPATH=src python scripts/glass_room/proto_bath.py <out.json>
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import proto as M  # noqa: E402

L, P, E, H, DT = M.L, M.P, M.E, M.H, M.DT
N, BURN = 10 * 24 * H, 3 * 24 * H
p = M.BASE
ki, tii, ai = L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"])
res = {}


def bath_mean(plant):
    return jnp.sum(plant.T * M.CELL_MASS) / jnp.sum(M.CELL_MASS)


def run(params, seed, n, loop=None, step_at=None, step_K=0.0, settled=None):
    """A week at a held crown (the table on the pull charged), optionally with a second
    PI loop ``loop = (which, kc, ti, target)`` trimming the crown setpoint, or a step
    of the crown setpoint at ``step_at``."""
    par = tuple(jnp.asarray(v) for v in (0.5523, 0.0, 1.0e9, 0.0))
    s, m = M.settle(8, par, params) if settled is None else settled
    k_reset, keys = L.episode_keys(seed, n)
    _, st = L.ENV.reset_env(k_reset, params)
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )
    aux = M.Aux(st.plant.T, jnp.zeros((2, M.ND, M.NC)))
    obs = L.ENV.get_obs(st, params)
    target = (
        jnp.where(True, bath_mean(st.plant), 0.0)
        if loop is None or loop[0] == "bath"
        else obs[1]
    )

    def f(c, x):
        k, i = x
        s0, a0, (fi, Ii, Io), o = c
        if loop is None:
            trim, Io2 = (
                jnp.where(
                    i >= (step_at if step_at is not None else n + 1), step_K, 0.0
                ),
                Io,
            )
        else:
            which, kc, ti, _ = loop
            y = bath_mean(s0.plant) if which == "bath" else o[1]
            e = target - y
            Io2 = Io + e * DT
            trim = kc * (e + Io2 / ti)
        fi = fi + ai * (o[0] - fi)
        ei = M.CROWN0 + M.G_TABLE * (o[8] / M.PULL0 - 1.0) + trim - fi
        Ii2 = Ii + ei * DT
        asked = M.ROW.balance(o, jnp)[0] + ki * (ei + Ii2 / tii)
        u = jnp.clip(asked, -1.0, 1.0)
        moved = u != asked
        o1, s1, a1, info = M.step(k, s0, a0, jnp.reshape(u, (1,)), params)
        info["bath"] = bath_mean(s1.plant)
        info["floor3"] = P.sensors(s1.plant, params.grid, params.plant)["floor_bay"]
        info["floor_end"] = P.sensors(s1.plant, params.grid, params.plant)["floor"]
        return (
            s1,
            a1,
            (fi, jnp.where(moved, Ii, Ii2), jnp.where(moved, Io, Io2)),
            o1,
        ), info

    mem = (obs[0], m[3], jnp.zeros(()))
    return jax.lax.scan(f, (st, aux, mem, obs), (keys, jnp.arange(n)))[1]


def week(info):
    t = {k: np.asarray(v) for k, v in info.items()}
    c = M.costs(t, 0.01, 0.0, BURN)
    return dict(
        travel=float(t["rise"][BURN:].sum()),
        end=float(t["rise_end"][BURN:].sum()),
        fuel=c["fuel"],
        marks_h=c["cliff_h"],
        lost=c["contact_share"],
        total=c["total"],
        X_min=float(t["X"][BURN:].min()),
        bath_sd=float(t["bath"][BURN:].std()),
        crown_span=float(t["hot"][BURN:].max() - t["hot"][BURN:].min()),
    )


def mean_of(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


SEEDS = (100, 101, 102, 103, 104)


def main():
    only3 = len(sys.argv) > 2  # reuse the step fits in argv[2], run part 3 alone
    # ---- 1. which walk moves the bath
    sd = np.asarray(p.walk_sd, float)
    cases = {"every walk": sd, "none (sensor noise only)": sd * 0}
    for i, name in enumerate(E.WALKS):
        one = np.zeros(5)
        one[i] = sd[i]
        cases[f"{name} alone"] = one
    for label, w in ({} if only3 else cases).items():
        q = p.replace(walk_sd=tuple(float(x) for x in w))
        f = jax.jit(lambda seed, q=q: run(q, seed, N))
        r = mean_of([week(f(seed)) for seed in SEEDS[:3]])
        res[f"walks: {label}"] = r
        print(
            f"1. {label:26s} blocks rise {r['travel']:.1f} K a week (end zones {r['end']:.1f}), bath sd {r['bath_sd']:.2f} K, "
            f"crown span {r['crown_span']:.0f} K, X min {r['X_min']:.3f}",
            flush=True,
        )

    # ---- 2. the crown setpoint's step
    quiet = p.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
    n2 = 8 * 24 * H
    fits = json.load(open(sys.argv[2]))["step"] if only3 else {}
    t = (
        {}
        if only3
        else {
            k: np.asarray(v)
            for k, v in jax.jit(
                lambda: run(quiet, 0, n2, step_at=24 * H, step_K=5.0)
            )().items()
        }
    )
    for name in (() if only3 else ("bath", "floor3", "floor_end", "X")):
        y = t[name][24 * H :] - t[name][24 * H - 1]
        final = y[-1]
        theta = float(np.argmax(np.abs(y) > 0.05 * abs(final)) * DT)
        t63 = float(np.argmax(np.abs(y) > 0.632 * abs(final)) * DT)
        fits[name] = dict(
            k=float(final / 5.0),
            theta_h=theta / 3600.0,
            tau_h=(t63 - theta) / 3600.0,
            peak=float(y[np.argmax(np.abs(y))] / 5.0),
        )
        print(
            f"2. crown setpoint +5 K: {name:9s} gain {final/5.0:+.4f} per K, first 5 % at {theta/3600:.1f} h, "
            f"lag {(t63-theta)/3600:.1f} h, largest {fits[name]['peak']:+.4f} per K",
            flush=True,
        )
    res["step"] = fits

    # ---- 3. a loop on the bath
    for which in ("bath", "floor3"):
        k, theta, tau = (
            fits[which]["k"],
            fits[which]["theta_h"] * 3600.0,
            fits[which]["tau_h"] * 3600.0,
        )
        theta = max(theta, DT)
        kc = tau / (k * 2.0 * theta)  # SIMC, tc = theta
        ti = min(tau, 8.0 * theta)
        for scale in (1.0, 0.3):
            loop = ("bath" if which == "bath" else "floor", kc * scale, ti, 0.0)
            f = jax.jit(lambda seed, loop=loop: run(p, seed, N, loop=loop))
            r = mean_of([week(f(seed)) for seed in SEEDS])
            label = f"loop on {'the true bath mean' if which == 'bath' else 'the floor couple under bay 3'}, gain x{scale:g}"
            res[label] = dict(r, kc=kc * scale, ti_h=ti / 3600.0)
            print(
                f"3. {label:48s} Kc {kc*scale:.2f} K/K, Ti {ti/3600:.1f} h | blocks rise {r['travel']:.1f} K (end zones {r['end']:.1f}), "
                f"bath sd {r['bath_sd']:.2f} K, crown span {r['crown_span']:.0f} K, fuel {r['fuel']:.2f}, marks lost {r['marks_h']:.1f} h, "
                f"X min {r['X_min']:.3f}, glass lost at q10 1 % {100*r['lost']:.2f} %, total {r['total']:.2f}",
                flush=True,
            )
    f = jax.jit(lambda seed: run(p, seed, N))
    r = mean_of([week(f(seed)) for seed in SEEDS])
    res["crown held"] = r
    print(
        f"3. crown held, no second loop: blocks rise {r['travel']:.1f} K, bath sd {r['bath_sd']:.2f} K, fuel {r['fuel']:.2f}, "
        f"glass lost at q10 1 % {100*r['lost']:.2f} %, total {r['total']:.2f}",
        flush=True,
    )
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
