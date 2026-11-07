"""Attempt 3 against the twelve tests, read on what decides lost glass: the fining share
reaching the canal, against its pass mark. The margin (the refiner cooling's work, in kelvin
of the glass leaving) is reported beside it.

    python glass_slow_props.py SETTLED.npz step|settle|lookalike|haste|walks|gain

Paths are differences from a base run with the same walks. Controllers come from step
tests with no search. Nothing here feeds back into a parameter of the plant.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace import physical_env as E

EP = E.EnvParams()
QUIET = EP.replace(walk_sd=(0.0,) * len(E.WALKS))
H = int(3600 / EP.plant.delta_t)  # steps per hour
DAY = 24 * H
d = np.load(sys.argv[1])
F0, Q0 = float(d["fuel"]), float(d["cooling"])
S0 = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k not in ("fuel", "cooling")})
KEY = jax.random.PRNGKey(0)
NAMES = ("margin", "fining", "floor", "couple", "canal", "hot", "cooling share")


def margin(state, ep):
    """Cooling in kelvin of the leaving glass, and below zero the canal's shortfall."""
    canal = P.sensors(state.fields, ep.grid, ep.plant)["canal"]
    short = jnp.minimum(canal - ep.canal_target - ep.cooling_above, 0.0)
    return state.cooling / (ep.pull * ep.plant.c_glass) + short


def trace(state, ep):
    r = P.sensors(state.fields, ep.grid, ep.plant)
    return jnp.stack(
        [margin(state, ep), r["fining"], r["floor"], r["crown"], r["canal"], r["hot"],
         state.cooling / ep.cooling_max]
    )  # fmt: skip


def run(fuel, ep, n, key=KEY, s0=S0, policy=None):
    """``(traces, costs, obs)`` over ``n`` steps; ``fuel`` a number, a path, or None with
    ``policy(carry, obs, walk) -> (carry, fuel)``."""
    _, st = E.reset(key, s0, ep, fuel=F0, cooling=Q0)
    us = jnp.broadcast_to(F0 if fuel is None else fuel, (n,))

    def f(c, x):
        st, pc, obs = c
        k, u = x
        if policy is not None:
            pc, u = policy(pc, obs, st.walk)
        obs, st, _, costs = E.step(k, st, u, ep)
        return (st, pc, obs), (trace(st, ep), jnp.stack(list(costs.values())), obs)

    obs0 = E.observe(st, ep)
    pc0 = jnp.zeros(2)
    return jax.lax.scan(f, (st, pc0, obs0), (jax.random.split(key, n), us))[1]


def fopdt(y):
    f = y[-H:].mean()
    t = lambda q: np.argmax(np.abs(y) > q * abs(f)) / H
    tau = 1.5 * (t(0.632) - t(0.283))
    return f, t(0.01), t(0.632), tau, t(0.632) - tau


def run_step():
    n = DAY * 7
    base = np.asarray(jax.jit(lambda: run(F0, QUIET, n)[0])())
    up = np.asarray(jax.jit(lambda: run(F0 * 1.01, QUIET, n)[0])())
    pulse = jnp.where(jnp.arange(n) < 4 * H, 1.04, 1.0) * F0
    pl = np.asarray(jax.jit(lambda: run(pulse, QUIET, n)[0])())
    print(f"design: fuel {F0:.4f}, margin {base[0, 0]:.1f} K, fining {base[0, 1]:.3f}")
    print("fuel +1 % | settled, t1 %, t63, lag, fitted dead time (h)")
    for i, name in enumerate(NAMES[:6]):
        print(f"  {name:8s}", np.round(fopdt(up[:, i] - base[:, i]), 3))
    print("fuel +4 % for 4 h: peak, hours after the end, value at the end")
    for i, name in enumerate(NAMES[:5]):
        y = pl[:, i] - base[:, i]
        k = np.argmax(np.abs(y))
        print(f"  {name:8s} {y[k]:8.3f} {k / H - 4:6.1f} {y[4 * H - 1]:8.3f}")


def run_settle():
    n = DAY * 25
    for rise in (-20.0, 20.0):
        s0 = S0.replace(T=S0.T + rise)
        y = np.asarray(jax.jit(lambda: run(F0, QUIET, n, s0=s0)[0])())
        day = y[-DAY:].mean(0)
        drift = (day - y[-6 * DAY : -5 * DAY].mean(0)) / 5
        print(
            f"{rise:+.0f} K start: last day",
            np.round(day[:5], 3),
            "drift/day",
            np.round(drift[:5], 4),
        )


def run_lookalike():
    n = DAY

    def with_walk(i, size):
        ep = QUIET

        def policy(c, obs, walk):
            return c, F0

        _, st = E.reset(KEY, S0, ep, fuel=F0, cooling=Q0)

        def f(st, k):
            st = st.replace(walk=st.walk.at[i].set(size))  # held, not walking
            obs, st, _, _ = E.step(k, st, F0, ep)
            return st, (trace(st, ep), obs)

        return jax.lax.scan(f, st, jax.random.split(KEY, n))[1]

    base = [np.asarray(x) for x in jax.jit(lambda: with_walk(0, 0.0))()]
    foam = [np.asarray(x) for x in jax.jit(lambda: with_walk(4, 0.5))()]
    gas1 = [np.asarray(x) for x in jax.jit(lambda: with_walk(0, 0.01))()]
    at = lambda y, h: y[int(h * H) - H // 2 : int(h * H)].mean(0)
    x = (
        0.01
        * (at(foam[1], 2) - at(base[1], 2))[0]
        / (at(gas1[1], 2) - at(base[1], 2))[0]
    )
    gas = [np.asarray(v) for v in jax.jit(lambda: with_walk(0, x))()]
    print(
        f"gas +{100 * x:.2f} % matches sulfate +50 % on the couple at 2 h (noise-free obs)"
    )
    print("  h | sulfate: couple, air, stack, cooling share, fining | gas: same")
    for h in (1, 2, 3, 4, 6, 8, 12, 24):
        o = lambda r: np.round((at(r[1], h) - at(base[1], h))[[0, 6, 7, 5]], 3)
        fn = lambda r: round(float(at(r[0], h)[1] - at(base[0], h)[1]), 4)
        print(f"  {h:3d}", o(foam), fn(foam), "|", o(gas), fn(gas))
    for lab, r in (("sulfate", foam), ("gas", gas)):
        e = np.abs(r[0][:, 1] - base[0][:, 1])
        print(f"  {lab}: fining moves past 0.002 at {np.argmax(e > 0.002) / H:.1f} h")


def simc_fining():
    """PI from the fuel step on the fining share. The two-point fit finds no dead time,
    so the first visible move stands for it; closed-loop time equal to it (SIMC)."""
    n = DAY * 7
    base = np.asarray(run(F0, QUIET, n)[0][:, 1])
    up = np.asarray(run(F0 * 1.01, QUIET, n)[0][:, 1])
    f, first, _, tau, theta = fopdt(up - base)
    k = f / (0.01 * F0)  # share per kg/s
    theta = max(theta, first)
    return k, tau / (k * 2 * theta), min(tau, 8 * theta) * 3600


TARGET = EP.fining_pass + 0.01  # ours: about half a fuel percent of safety


def pi_policy(kc, ti, target, ff=None):
    """Valve from the observed fining share; ``ff`` also sees the hidden walks."""

    def policy(c, obs, walk):
        e = target - obs[4]
        integ = c[0] + kc * e * EP.plant.delta_t / ti
        u = F0 + kc * e + integ
        if ff is not None:
            u = (u - ff @ walk[1:]) / (1.0 + walk[0])
        return c.at[0].set(integ), jnp.clip(u, 0.25, 0.75)

    return policy


def run_haste():
    k, kc, ti = simc_fining()
    print(
        f"fining gain {k:.3f} per kg/s; PI {kc:.4f} kg/s per unit, Ti {ti / 3600:.1f} h"
    )
    n = DAY * 14
    f0 = float(run(F0, QUIET, 2)[0][0, 1])
    step = 0.02  # about one fuel percent of fining
    for mult in (1.0, 3.0, 6.0):
        pol = pi_policy(kc * mult, ti, f0 + step)
        y = np.asarray(jax.jit(lambda: run(None, QUIET, n, policy=pol)[0])())
        e = (y[:, 1] - f0) / step
        out = np.nonzero(np.abs(e - 1.0) > 0.2)[0]
        settle = (out[-1] + 1) / H if out.size else 0.0
        print(
            f"  gain x{mult:3.0f}: overshoot {100 * (e.max() - 1):5.0f} %, within 20 % for"
            f" good {settle:6.1f} h, crown {y[:, 3].min() - 273.15:.0f} to"
            f" {y[:, 3].max() - 273.15:.0f} C"
        )


def run_walks():
    n = DAY * 10
    keys = jax.random.split(jax.random.PRNGKey(7), 6)
    k, kc, ti = simc_fining()
    f0 = float(run(F0, QUIET, 2)[0][0, 1])
    # Settled sensitivities of the fining share to each hidden walk, in fuel.
    sens = []
    for i, size in ((1, 0.02), (2, 0.05), (3, -0.2), (4, 0.5)):
        _, st = E.reset(KEY, S0, QUIET, fuel=F0, cooling=Q0)

        def f(st, kk, i=i, size=size):
            st = st.replace(walk=st.walk.at[i].set(size))
            _, st, _, _ = E.step(kk, st, F0, QUIET)
            return st, P.sensors(st.fields, QUIET.grid, QUIET.plant)["fining"]

        y = jax.lax.scan(f, st, jax.random.split(KEY, DAY * 7))[1]
        sens.append((float(y[-1]) - f0) / size / k)
    ff = jnp.asarray(sens)
    print("fining per walk unit, in fuel kg/s:", np.round(sens, 4))
    for label, fuel, pol in (
        ("valve held", F0, None),
        ("PI on fining", None, pi_policy(kc, ti, TARGET)),
        ("PI, told the walks", None, pi_policy(kc, ti, TARGET, ff)),
    ):
        go = jax.jit(jax.vmap(lambda kk: run(fuel, EP, n, key=kk, policy=pol)))
        tr, costs, _ = [np.asarray(x) for x in go(keys)]
        day = costs[:, 2 * DAY :].sum(1) / 8.0  # EUR per day after two days
        lost_h = (costs[:, 2 * DAY :, 1] > 0).sum(1) / H
        trip_h = (costs[:, 2 * DAY :, 2] > 0).sum(1) / H
        print(
            f"{label:20s} EUR/day {np.round(day.sum(1))}, fuel {np.round(day[:, 0])},"
            f" lost glass h {lost_h}, trip h {trip_h},"
            f" fining sd {tr[:, 2 * DAY :, 1].std():.4f}"
        )


def run_gain():
    states = {
        "design": {},
        "pull -2 %": {"pull": EP.pull * 0.98},
        "pull +2 %": {"pull": EP.pull * 1.02},
        "cullet 15": {"cullet": 0.15},
        "cullet 35": {"cullet": 0.35},
        "blanket -20 %": {"thickness": 0.16},
        "blanket +20 %": {"thickness": 0.24},
    }
    n0, n1 = DAY * 20, DAY * 27
    steps = jnp.where(jnp.arange(n1) < n0, F0, F0 * 1.01)
    for name, kw in states.items():
        ep = QUIET.replace(**kw)
        base = np.asarray(jax.jit(lambda: run(F0, ep, n1)[0][:, 1])())
        up = np.asarray(jax.jit(lambda: run(steps, ep, n1)[0][:, 1])())
        y = (up - base)[n0:]
        r = fopdt(y)
        print(
            f"{name:14s} fining gain at 24 h {y[DAY - 1] / (0.01 * F0):6.3f}, 7 d"
            f" {r[0] / (0.01 * F0):6.3f} per kg/s, first move {r[1]:5.1f} h,"
            f" t63 {r[2]:5.1f} h"
        )


{
    "step": run_step,
    "settle": run_settle,
    "lookalike": run_lookalike,
    "haste": run_haste,
    "walks": run_walks,
    "gain": run_gain,
}[sys.argv[2]]()
