"""Fuel-to-reading gain across states an episode reaches, each settled at held fuel.

python glass_physical_gain.py ORBIT.npz
"""

import sys

import glass_physical_settle as Z
import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P

g = P.Grid()
p = P.PhysicalParams()
d = np.load(sys.argv[1])
f0 = float(d["fuel"])
S0 = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k != "fuel"})
CASES = {
    "design": (f0, 5.79, 0.25, 0.2),
    "fuel -2 %": (f0 * 0.98, 5.79, 0.25, 0.2),
    "fuel +2 %": (f0 * 1.02, 5.79, 0.25, 0.2),
    "pull -5 %": (f0, 5.79 * 0.95, 0.25, 0.2),
    "pull +5 %": (f0, 5.79 * 1.05, 0.25, 0.2),
    "cullet 20": (f0, 5.79, 0.20, 0.2),
    "cullet 30": (f0, 5.79, 0.30, 0.2),
    "blanket -20 %": (f0, 5.79, 0.25, 0.16),
    "blanket +20 %": (f0, 5.79, 0.25, 0.24),
}


def pair(fuel, charge, cullet, thick):
    s, _ = Z.advance(S0, fuel, g, p, 2880 * 20, charge, cullet, thick)

    def after(u):
        n, y = Z.advance(s, u, g, p, 2880 * 8, charge, cullet, thick)
        return y[jnp.array([119, 719, 2879, 23039])]

    return s.T_roof[0, Z.BAY], after(fuel), after(fuel * 1.005), P.cover(s.m).sum()


args = [jnp.array([c[i] for c in CASES.values()]) for i in range(4)]
c, a, b, cov = [np.asarray(v) for v in jax.jit(jax.vmap(pair))(*args)]
for i, n in enumerate(CASES):
    du = 0.005 * args[0][i]
    print(f"{n:15s} crown {c[i] - 273.15:7.1f} cover cells {cov[i]:.2f} gain K per kg/s at 1, 6, 24, 192 h",
          np.round((b[i] - a[i]) / float(du), 0))  # fmt: skip
