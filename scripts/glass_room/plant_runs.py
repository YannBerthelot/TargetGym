"""Glass room study: the registered plant with its defects, under the controllers the
prototype compared. The env's own step and reward, no wrapper.

    PYTHONPATH=src python scripts/glass_room/plant_runs.py normal <out.json> [block_release]
    PYTHONPATH=src python scripts/glass_room/plant_runs.py rise <out.json> [block_release]
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import lib as L  # noqa: E402

P, E, H, DT = L.P, L.E, L.HOUR, L.DT
N, BURN = 10 * 24 * H, 3 * 24 * H
ROW = L.ladder_row()
G_TABLE, CROWN0 = 185.0, 1820.75
ko, tio, ao = L.OUTER["kc"], L.OUTER["ti"], L._alpha(L.OUTER["tf"])
ki, tii, ai = L.INNER["kc"], L.INNER["ti"], L._alpha(L.INNER["tf"])
#: SIMC on the crown setpoint's step (proto_bath.py, part 2): K of crown per K, s.
LOOPS = {"floor": (2.11, 3.8 * 3600.0), "bath": (28.54, 1.9 * 3600.0)}


def bath_mean(plant, grid):
    a = P.layout(grid)
    w = a.dz[:, None] * a.dx * a.wc
    return jnp.sum(plant.T * w) / jnp.sum(w)


def run(mode, p0, p, seed, n=N, sp=0.5523):
    """``mode``: "cascade" (the ladder's, crown setpoint following the pull), "held"
    (no second loop), "floor" (a loop on the floor couple under bay 3), "bath" (a loop
    on the bath's true mean)."""
    base = BASE_PULL

    def law(mem, o, s, target):
        y, Io, f, Ii = mem
        if mode == "cascade":
            y = y + ao * (o[4] - y)
            e = sp - y
            kc, ti = ko, tio
        elif mode == "held":
            e, kc, ti = jnp.zeros(()), 0.0, 1.0
        else:
            kc, ti = LOOPS[mode]
            e = target - (o[1] if mode == "floor" else bath_mean(s.plant, p.grid))
        Io2 = Io + e * DT
        trim = kc * (e + Io2 / ti)
        f = f + ai * (o[0] - f)
        ei = CROWN0 + G_TABLE * (o[8] / base - 1.0) + trim - f
        Ii2 = Ii + ei * DT
        asked = ROW.balance(o, jnp)[0] + ki * (ei + Ii2 / tii)
        u = jnp.clip(asked, -1.0, 1.0)
        moved = u != asked
        return jnp.reshape(u, (1,)), (
            y,
            jnp.where(moved, Io, Io2),
            f,
            jnp.where(moved, Ii, Ii2),
        )

    quiet = p0.replace(walk_sd=(0.0,) * 5, sensor_noise_scale=0.0)
    obs, st = L.ENV.reset_env(jax.random.PRNGKey(0), quiet)
    mem = (obs[4], jnp.zeros(()), obs[0], jnp.zeros(()))
    t0 = obs[1] if mode == "floor" else bath_mean(st.plant, p.grid)

    def settle(c, k):
        s, m, o, tgt = c
        a_, m = law(m, o, s, tgt)
        o1, s1, _, _, _ = L.ENV.step_env(k, s, a_, quiet)
        # while settling, the second loop's target follows where the plant goes
        tgt2 = o1[1] if mode == "floor" else bath_mean(s1.plant, p.grid)
        return (s1, m, o1, tgt2), None

    held = "held" if mode in ("floor", "bath") else mode
    (s, m, _, _), _ = jax.lax.scan(
        settle if mode in ("cascade", "held") else settle,
        (st, mem, obs, t0),
        jax.random.split(jax.random.PRNGKey(31), 25 * 24 * H),
    )
    del held
    k_reset, keys = L.episode_keys(seed, n)
    _, st = L.ENV.reset_env(k_reset, p)
    st = st.replace(
        plant=s.plant,
        cooling=s.cooling,
        integral=s.integral,
        fuel=s.fuel,
        limit=s.limit,
    )
    obs = L.ENV.get_obs(st, p)
    target = obs[1] if mode == "floor" else bath_mean(st.plant, p.grid)
    m = (m[0], jnp.zeros(()) if mode in ("floor", "bath") else m[1], m[2], m[3])

    def f(c, k):
        s0, mem, o = c
        a_, mem = law(mem, o, s0, target)
        o1, s1, _, _, _ = L.ENV.step_env(k, s0, a_, p)
        terms = E.compute_reward_terms(s1, p)
        read = P.sensors(s1.plant, p.grid, p.plant)
        return (s1, mem, o1), dict(
            fuel=terms["running"],
            lost=terms["tracking"],
            pull=s1.pull,
            X=read["fining"],
            hot=read["hot"],
            shed=read["defects"],
            reboil=jnp.zeros(()),
            mark=(read["canal"] < p.canal_target) | (read["fining"] < p.fining_pass),
        )

    return jax.lax.scan(f, (st, m, obs), keys)[1]


def week(info, p):
    t = {k: np.asarray(v)[BURN:] for k, v in info.items()}
    worth = t["pull"] * DT / 1000.0 * float(p.lost_glass_price)
    return dict(
        fuel=float(t["fuel"].mean()),
        total=float((t["fuel"] + t["lost"]).mean()),
        lost_share=float((t["lost"] / worth).mean()),
        mark_h=float(t["mark"].sum() / H),
        shed_share=float((1.0 - np.exp(-t["shed"])).mean()),
        reboil_share=float((1.0 - np.exp(-t["reboil"])).mean()),
        X_min=float(t["X"].min()),
        crown_span=float(t["hot"].max() - t["hot"].min()),
    )


def mean_of(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


BASE_PULL = float(L.scenario_params("normal").pull)

if __name__ == "__main__":
    which, out = sys.argv[1], sys.argv[2]
    base = L.scenario_params("normal")
    if len(sys.argv) > 3:
        base = base.replace(plant=base.plant.replace(block_release=float(sys.argv[3])))
    print(
        "block_release",
        float(base.plant.block_release),
        "release from",
        float(base.plant.block_release_T) - 273.15,
        "C",
        flush=True,
    )
    res = {}
    if which == "normal":
        p0, p, seeds = base, base, (100, 101, 102, 103, 104)
    else:
        p0 = base.replace(pull=base.pull * 0.8)
        p, seeds = p0.replace(pull_change=0.30, pull_change_window_h=(96.0, 96.0)), (
            100,
            101,
            102,
        )
    for mode in ("held", "cascade", "floor", "bath"):
        f = jax.jit(lambda seed, mode=mode: run(mode, p0, p, seed))
        r = mean_of([week(f(seed), p) for seed in seeds])
        res[mode] = r
        print(
            f"{mode:8s} fuel {r['fuel']:.2f} | glass lost {100*r['lost_share']:.3f} % "
            f"(marks {r['mark_h']:.1f} h) | total {r['total']:.2f} | "
            f"crown span {r['crown_span']:.0f} K | X min {r['X_min']:.3f}",
            flush=True,
        )
    json.dump(res, open(out, "w"), indent=1)
