"""Residence times of attempt 3's melt, by washout: every cell starts marked, batch glass
arrives unmarked, and the canal's mark falls as new glass reaches it. The first 0.1 % to
arrive is the minimum residence time, as a CFD's fastest 0.1 % of particles; the area under
the mark is the mean. The fining release is switched off so the mark is a pure tracer.

    python glass_slow_residence.py SETTLED.npz [days]
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P

P.fining_done = lambda T, p: jnp.zeros_like(T)  # the mark is carried, never released
g, p = P.Grid(), P.PhysicalParams()
a = P.layout(g)
d = np.load(sys.argv[1])
fuel, cool = float(d["fuel"]), float(d["cooling"])
s0 = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k not in ("fuel", "cooling")})
s0 = s0.replace(fining=jnp.ones_like(s0.T))
days = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
n = int(days * 86400 / p.delta_t)


def run(s):
    def f(s, _):
        s, b = P.plant_step(s, fuel, 5.79, 0.25, 0.2, cool, g, p)
        return s, P.sensors(s, g, p)["fining"]

    return jax.lax.scan(f, s, None, length=n)[1]


mark = np.asarray(jax.jit(run)(s0))
hours = np.arange(1, n + 1) * p.delta_t / 3600.0
volume = float(jnp.sum(a.dz[:, None] * a.dx * a.wc))
held = volume * p.rho_glass / 5.79 / 3600.0
first = hours[np.argmax(mark < 0.999)]
mean = float(np.sum(mark) * p.delta_t / 3600.0)
# From the hot spot: only the top 10 cm of bay 3's columns marked, as fining sets them.
cols = slice(P.COUPLE_BAY * g.cols_per_bay, (P.COUPLE_BAY + 1) * g.cols_per_bay)
spot = jnp.zeros_like(s0.T).at[: P.N_FINE, cols].set(1.0)
pulse = np.asarray(jax.jit(run)(s0.replace(fining=spot)))
print(f"from the hot spot's top 10 cm: canal mark past 1e-4 at"
      f" {hours[np.argmax(pulse > 1e-4)]:.1f} h, past 1e-3 at {hours[np.argmax(pulse > 1e-3)]:.1f} h,"
      f" peak at {hours[np.argmax(pulse)]:.1f} h")  # fmt: skip
print(f"minimum residence (first 0.1 %) {first:.1f} h, first 1 % {hours[np.argmax(mark < 0.99)]:.1f} h,"
      f" mean {mean:.1f} h (mark left at the end {mark[-1]:.3f}), volume over pull {held:.1f} h,"
      f" minimum over mean {first / mean:.2f}")  # fmt: skip
