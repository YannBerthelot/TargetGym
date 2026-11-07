"""Attempt 2's plant against attempt 3's definitions, reading the floor at the melter's end.

    python glass_floor_props.py ORBIT.npz step|settle|haste|lookalike|walks|gain

Every path is a difference from a base run on the same clock, so the reversal cancels. Walk
sizes are ours, inside the owner's +-10 points and +-20 %. Controllers come from step tests,
with no search. Nothing here feeds back into a parameter of the plant.
"""

import sys

import jax

if sys.argv[2] == "settle":  # float32 stalls here: see the proposal
    jax.config.update("jax_enable_x64", True)

import glass_physical_settle as Z
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace.env import firing_fraction

g = P.Grid()
p = P.PhysicalParams()
a = P.layout(g)
nm = a.nm
d = np.load(sys.argv[1])
f0 = float(d["fuel"])
S0 = P.Fields(**{k: jnp.asarray(d[k], dtype=float) for k in d if k != "fuel"})
PULL, CUL, TH, SUL = 5.79, 0.25, 0.2, 0.01
H, DAY, DT = 120, 2880, p.delta_t
NAMES = ("floor", "bottom bay 3", "couple", "canal", "air", "stack")
KC_INNER, TI_INNER = 0.004, 1800.0  # the crown loop, kg/s per K


def step(s, u, cullet, thick, sulfate, t, pull=PULL):
    q = p.replace(sulfate_batch=sulfate)
    n, b = P.plant_step(s, u * firing_fraction(t, q), pull, cullet, thick, t, g, q)
    _, tc, bottom, canal = P.sensors(n, Z.DEPTH, Z.BAY, g)
    return n, jnp.stack([n.T[-1, nm - 1], bottom, tc, canal, b["T_air"], b["T_stack"]])


def hold(u, cullet=CUL, thick=TH, sulfate=SUL, n=DAY * 8, s0=S0):
    """Readings with ``u`` (a number or a path of ``n``) held from ``s0``."""
    us = jnp.broadcast_to(u, (n,))
    f = lambda s, x: step(s, x[1], cullet, thick, sulfate, x[0])
    return jax.lax.scan(f, s0, (jnp.arange(n), us))


def fopdt(y):
    """Dead time and lag by the two-point fit, from one step's path."""
    f = y[-200:].mean()
    t = lambda q: np.argmax(np.abs(y) > q * abs(f)) / H
    tau = 1.5 * (t(0.632) - t(0.283))
    return f, t(0.01), t(0.632), tau, t(0.632) - tau


def crown_held(ref, rise, n):
    """Fuel set by a slow PI so the couple tracks its base path plus ``rise``."""

    def f(c, x):
        s, integ, tc = c
        t, r = x
        e = r + rise - tc
        integ = integ + KC_INNER * e * DT / TI_INNER
        u = jnp.clip(f0 + KC_INNER * e + integ, 0.2, 0.8)
        s, y = step(s, u, CUL, TH, SUL, t)
        return (s, integ, y[2]), y

    return jax.lax.scan(f, (S0, 0.0, ref[0]), (jnp.arange(n), ref))[1]


def run_step():
    n = DAY * 8
    base = jax.jit(lambda: hold(f0, n=n)[1])()
    fuel = jax.jit(lambda: hold(f0 * 1.01, n=n)[1])()
    held = jax.jit(lambda: crown_held(base[:, 2], 5.0, n))()
    pulse = jax.jit(
        lambda: hold(jnp.where(jnp.arange(n) < 4 * H, 1.04, 1.0) * f0, n=n)[1]
    )()
    base, fuel, held, pulse = [np.asarray(x) for x in (base, fuel, held, pulse)]
    for label, y in (("fuel +1 %", fuel), ("crown +5 K, held", held)):
        print(label, "| settled K, t1 %, t63, lag, fitted dead time, h")
        for i, name in enumerate(NAMES):
            print(f"  {name:13s}", np.round(fopdt(y[:, i] - base[:, i]), 1))
    print("fuel +4 % for 4 h, then back: peak K, hours after the end, K at the end")
    for i, name in enumerate(NAMES[:4]):
        y = pulse[:, i] - base[:, i]
        k = np.argmax(np.abs(y))
        print(f"  {name:13s} {y[k]:6.2f} {k / H - 4:6.1f} {y[4 * H - 1]:6.2f}")


def run_settle():
    n = DAY * 25
    for rise in (-20.0, 20.0):
        y = np.asarray(
            jax.jit(lambda: hold(f0, n=n, s0=S0.replace(T=S0.T + rise))[1])()
        )
        print(
            f"{rise:+.0f} K start, last cycle mean",
            np.round(y[-100:, :3].mean(0), 3),
            y.dtype,
        )


def simc_fuel():
    """PI from the fuel step on the floor: closed-loop time equal to the dead time."""
    base, fuel = [np.asarray(hold(u)[1][:, 0]) for u in (f0, f0 * 1.01)]
    f, _, _, tau, theta = fopdt(fuel - base)
    k = f / (0.01 * f0)
    return k, tau / (k * 2 * theta), min(tau, 8 * theta) * 3600


def run_haste():
    _, kc, ti = simc_fuel()
    n = DAY * 14
    ref = jax.jit(lambda: hold(f0, n=n)[1][:, 0])()

    def closed(kc):
        def f(c, x):
            s, integ, y = c
            t, r = x
            e = r + 5.0 - y
            integ = integ + kc * e * DT / ti
            u = jnp.clip(f0 + kc * e + integ, 0.25, 0.7)
            s, o = step(s, u, CUL, TH, SUL, t)
            return (s, integ, o[0]), jnp.stack([o[0], u])

        return jax.lax.scan(f, (S0, 0.0, ref[0]), (jnp.arange(n), ref))[1]

    print(f"PI: {kc:.5f} kg/s per K, integral {ti / 3600:.1f} h, target +5 K")
    run = jax.jit(closed)
    for m in (1.0, 3.0, 6.0, 10.0):
        y = np.asarray(run(kc * m))
        e = y[:, 0] - np.asarray(ref)
        out = np.nonzero(np.abs(e - 5.0) > 1.0)[0]
        print(f"  gain x{m:4.1f}: peak {e.max():6.2f} K, overshoot {20 * (e.max() - 5):5.0f} %, "
              f"into +-1 K for good {(out[-1] + 1) / H:6.1f} h, valve {y[:, 1].min():.3f} to {y[:, 1].max():.3f}")  # fmt: skip


def run_lookalike():
    n = DAY
    b = lambda **k: np.asarray(jax.jit(lambda: hold(n=n, **k)[1])())
    y0, foam, gas1 = b(u=f0), b(u=f0, sulfate=SUL * 1.5), b(u=f0 * 1.01)
    at = lambda y, h: y[int(h * H) - 100 : int(h * H)].mean(0)
    x = 0.01 * (at(foam, 2) - at(y0, 2))[2] / (at(gas1, 2) - at(y0, 2))[2]
    gas = b(u=f0 * (1 + x))
    print(f"gas +{100 * x:.2f} % matches foam on the couple at 2 h")
    for h in (1, 2, 3, 4, 6, 8, 12, 24):
        print(f"  {h:3d} h | foam, then gas: floor, couple, air, stack |",
              np.round((at(foam, h) - at(y0, h))[[0, 2, 4, 5]], 2),
              np.round((at(gas, h) - at(y0, h))[[0, 2, 4, 5]], 2))  # fmt: skip
    for lab, y in (("foam", foam), ("gas", gas)):
        e = np.abs(y[:, 0] - y0[:, 0])
        print(f"  {lab}: floor past 0.1 K at {np.argmax(e > 0.1) / H:.1f} h, past 0.36 K at {np.argmax(e > 0.36) / H:.1f} h")  # fmt: skip


def run_walks():
    n, skip = DAY * 10, DAY * 3
    settled = {}
    b = np.asarray(jax.jit(lambda: hold(f0, n=DAY * 14)[1])())
    for lab, arg in (
        ("fuel", (f0 * 1.01, CUL, TH, SUL)),
        ("cullet", (f0, CUL + 0.05, TH, SUL)),
        ("thick", (f0, CUL, TH * 0.8, SUL)),
        ("sulfate", (f0, CUL, TH, SUL * 1.5)),
    ):
        y = np.asarray(jax.jit(lambda: hold(*arg, n=DAY * 14)[1])())
        settled[lab] = float((y - b)[-1, 0])
    print("floor settled, K:", {k: round(v, 2) for k, v in settled.items()})
    k = settled["fuel"] / (0.01 * f0)
    ff = jnp.array(
        [
            settled["cullet"] / 0.05 / k,
            settled["thick"] / (-0.2 * TH) / k,
            settled["sulfate"] / (0.5 * SUL) / k,
        ]
    )
    kc, ti = 41.5 / (k * 9.8), 39.2 * 3600  # SIMC, from the fuel step
    ko, to = 1.52, 7.0 * 3600  # SIMC, from the crown-held step
    ref, tgt = jnp.asarray(b[:n, 2]), float(b[0, 0])

    def ou(key, sd, hours):
        phi = np.exp(-1 / H / hours)
        e = jax.random.normal(key, (n,)) * sd * jnp.sqrt(1 - phi**2)
        return jax.lax.scan(lambda v, w: (phi * v + w, v), 0.0, e)[1]

    def walks(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return (
            1 + ou(k1, 0.01, 24.0),
            CUL + ou(k2, 0.05, 12.0),
            TH * (1 + ou(k3, 0.10, 12.0)),
            SUL * (1 + ou(k4, 0.17, 12.0)),
        )

    def loop(key, mode):
        def f(c, x):
            s, i1, i2, fl, tc = c
            t, gas, cu, th, su, r = x
            e = tgt - fl
            if mode == "held":
                u = f0
            elif mode == "cascade":
                i1 = i1 + ko * e * DT / to
                ec = r + ko * e + i1 - tc
                i2 = i2 + KC_INNER * ec * DT / TI_INNER
                u = f0 + KC_INNER * ec + i2
            else:
                i1 = i1 + kc * e * DT / ti
                u = f0 + kc * e + i1
            if mode != "held" and mode != "PI" and mode != "cascade":
                # Sees some hidden upsets, through settled gains only.
                seen = jnp.array([c in mode for c in "cts"], dtype=float)
                u = u - ff @ (seen * jnp.array([cu - CUL, th - TH, su - SUL]))
                u = u / gas if "g" in mode else u
            u = jnp.clip(u, 0.25, 0.7) if mode != "held" else u
            s, o = step(s, u * gas, cu, th, su, t)
            return (s, i1, i2, o[0], o[2]), o[0]

        c0 = (S0, 0.0, 0.0, tgt, ref[0])
        return jax.lax.scan(f, c0, (jnp.arange(n), *walks(key), ref))[1]

    keys = jax.random.split(jax.random.PRNGKey(7), 6)
    # Told upsets are named by letter: gas, cullet, thickness, sulfate.
    for mode in ("held", "PI", "cascade", "gcts", "c", "g", "s", "gts"):
        y = np.asarray(jax.jit(jax.vmap(lambda k: loop(k, mode)))(keys))
        e = y[:, skip:] - tgt
        rms = np.sqrt((e**2).mean(1))
        out = np.abs(y - tgt) > 3.0
        first = [out[i].argmax() / H if out[i].any() else np.inf for i in range(6)]
        print(f"{mode:8s} floor rms {np.round(rms, 2)} all {np.sqrt((e**2).mean()):.2f} "
              f"worst {np.abs(e).max():.1f}, first out of +-3 K, h {np.round(first, 0)}")  # fmt: skip


def run_gain():
    cases = {
        "design": (PULL, CUL, TH),
        "pull -5 %": (PULL * 0.95, CUL, TH),
        "pull +5 %": (PULL * 1.05, CUL, TH),
        "cullet 20": (PULL, 0.20, TH),
        "cullet 30": (PULL, 0.30, TH),
        "blanket -20 %": (PULL, CUL, 0.16),
        "blanket +20 %": (PULL, CUL, 0.24),
    }

    def after(s, u, pull, cu, th):
        f = lambda s, t: step(s, u, cu, th, SUL, t, pull)
        return jax.lax.scan(f, s, jnp.arange(DAY * 8))[1][:, 0]

    def pair(pull, cu, th):
        f = lambda s, t: step(s, f0, cu, th, SUL, t, pull)
        s = jax.lax.scan(f, S0, jnp.arange(DAY * 20))[0]
        return after(s, f0, pull, cu, th), after(s, f0 * 1.005, pull, cu, th)

    args = [jnp.array([c[i] for c in cases.values()]) for i in range(3)]
    base, up = [np.asarray(v) for v in jax.jit(jax.vmap(pair))(*args)]
    du = 0.005 * f0
    for i, name in enumerate(cases):
        y = up[i] - base[i]
        print(f"{name:14s} floor gain K per kg/s at 24 h {y[DAY - 1] / du:6.0f}, 8 d {y[-1] / du:6.0f}, "
              f"dead time {fopdt(y)[4]:5.1f} h, t63 {fopdt(y)[2]:5.1f} h")  # fmt: skip


{
    "step": run_step,
    "settle": run_settle,
    "haste": run_haste,
    "lookalike": run_lookalike,
    "walks": run_walks,
    "gain": run_gain,
}[sys.argv[2]]()
