"""Settle attempt 2's plant: the firing that holds a crown, then the orbit.

    python glass_physical_settle.py OUT.npz [n_depth skin cols n_roof roof_skin dt]

The crown is the hot face's mean over a cycle in the bay of the plant data's peak.
Nothing here feeds back into a parameter of the plant.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace.env import firing_fraction

DEPTH, BAY = 0.036, 3  # the couple, ours; the bay of W2000b's peak at 14 m
CYCLE = 100  # steps of a double reversal


def advance(s, fuel, g, p, steps, charge=5.79, cullet=0.25, thick=0.2, t0=0):
    """(fields, hot-face path) of ``steps`` held steps; ``fuel`` before the gaps."""

    def f(s, t):
        n, _ = P.plant_step(
            s, fuel * firing_fraction(t, p), charge, cullet, thick, t, g, p
        )
        return n, n.T_roof[0, BAY]

    return jax.lax.scan(f, s, t0 + jnp.arange(steps))


def crown_at(fuel, s, g, p, days):
    run = jax.jit(lambda s, f: advance(s, f, g, p, 28800))
    for _ in range(days // 10):
        s, y = jax.block_until_ready(run(s, fuel))
    return s, float(y[-CYCLE:].mean()) - 273.15


def settle(target, g, p, guess=(0.44, 0.46), days=30):
    lo, hi = guess
    s = P.initial_fields(g, p)
    s, c_lo = crown_at(lo, s, g, p, days)
    s, c_hi = crown_at(hi, s, g, p, days)
    for _ in range(6):
        f = hi + (target - c_hi) * (hi - lo) / (c_hi - c_lo)
        f = float(np.clip(f, 0.38, 0.55))
        lo, c_lo = hi, c_hi
        hi = f
        s, c_hi = crown_at(hi, s, g, p, days)
        if abs(c_hi - target) < 0.3:
            break
    # Hand back the orbit's own start: a whole number of cycles.
    run = jax.jit(lambda s: advance(s, hi, g, p, 2880 * 10 - (2880 * 10 % CYCLE)))
    s, _ = jax.block_until_ready(run(s))
    return hi, s


if __name__ == "__main__":
    a = sys.argv
    kw = dict(zip(("n_depth", "skin", "cols_per_bay", "n_roof", "roof_skin"), a[2:7]))
    g = P.Grid(
        n_depth=int(kw.get("n_depth", 10)), skin=float(kw.get("skin", 0.006)),
        cols_per_bay=int(kw.get("cols_per_bay", 2)), n_roof=int(kw.get("n_roof", 8)),
        roof_skin=float(kw.get("roof_skin", 0.003)),
    )  # fmt: skip
    p = P.PhysicalParams()
    if len(a) > 7:
        p = p.replace(delta_t=float(a[7]))
    fuel, s = settle(1590.0, g, p)
    np.savez(
        a[1],
        fuel=fuel,
        **{k: np.asarray(getattr(s, k)) for k in s.__dataclass_fields__},
    )
    print(f"fuel {fuel:.4f}  SEC {fuel * p.LHV / 5.79 / 1e6:.2f} GJ/t", flush=True)
