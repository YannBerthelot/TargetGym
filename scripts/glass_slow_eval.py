"""Attempt 3's box nulls on its scenario suite: cost per day, lost glass and crown.

    python glass_slow_eval.py SETTLED.npz [seeds]

Box nulls are constant valves at the middle and at each bound of the action box
(``target_gym.null_policies``), plus the settled design valve for reference. Seeds are
the PRNG keys 0 .. seeds - 1 of each scenario; the first day is kept, since rule 17
starts the episode in established operation.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.glass_furnace import physical as P
from target_gym.glass_furnace import physical_env as E
from target_gym.glass_furnace.physical_scenarios import suite

d = np.load(sys.argv[1])
F0, Q0 = float(d["fuel"]), float(d["cooling"])
S0 = P.Fields(**{k: jnp.asarray(d[k]) for k in d if k not in ("fuel", "cooling")})
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
BASE = E.EnvParams()
N = BASE.plant.max_steps_in_episode
H = 3600.0 / BASE.plant.delta_t


def episode(key, fuel, ep):
    k0, k1 = jax.random.split(key)
    _, st = E.reset(k0, S0, ep, fuel=F0, cooling=Q0)

    def f(st, k):
        _, st, _, c = E.step(k, st, fuel, ep)
        hot = P.sensors(st.fields, ep.grid, ep.plant)["hot"]
        return st, jnp.stack([c["fuel"], c["lost_glass"], hot])

    return jax.lax.scan(f, st, jax.random.split(k1, N))[1]


def main():
    keys = jax.random.split(jax.random.PRNGKey(0), 64)[:SEEDS]  # as the oracle's
    policies = {
        "box min": BASE.fuel_min,
        "box mid": 0.5 * (BASE.fuel_min + BASE.fuel_max),
        "box max": BASE.fuel_max,
        "design": F0,
    }
    days = N / H / 24.0
    for sc in suite():
        ep = BASE.replace(**sc.overrides)
        print(f"== {sc.name}: {sc.description}", flush=True)
        for name, fuel in policies.items():
            run = jax.jit(jax.vmap(lambda k, u=fuel, e=ep: episode(k, u, e)))
            y = np.asarray(run(keys))
            eur = y[:, :, :2].sum(1) / days
            lost_h = (y[:, :, 1] > 0).sum(1) / H
            print(
                f"  {name:8s} {fuel:.3f} kg/s: EUR/day {np.round(eur.sum(1))},"
                f" fuel {np.round(eur[:, 0].mean())}, lost glass h {np.round(lost_h, 1)},"
                f" hottest face max {y[:, :, 2].max() - 273.15:.1f} C",
                flush=True,
            )


if __name__ == "__main__":
    main()
