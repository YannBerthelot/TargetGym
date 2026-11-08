"""Attempt 3's box nulls on its scenario suite: cost per day, lost glass and crown.

    python glass_slow_eval.py [seeds]

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

ENV = E.FloatFurnace()
F0 = E.design_state()[0]
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
BASE = ENV.default_params
N = BASE.max_steps_in_episode
H = 3600.0 / BASE.delta_t


def episode(key, fuel, ep):
    k0, k1 = jax.random.split(key)
    _, st = ENV.reset_env(k0, ep)
    a = E.to_action(jnp.asarray(fuel), ep)

    def f(st, k):
        _, st, _, _, _ = ENV.step_env(k, st, a, ep)
        c = E.compute_reward_terms(st, ep)
        hot = P.sensors(st.plant, ep.grid, ep.plant)["hot"]
        return st, jnp.stack([c["running"], c["tracking"], hot])

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
            # Crown wear has no family price, so it is measured as is (charter rule 19).
            hot = y[:, :, 2] - 273.15
            above = [float((hot > c).sum(1).mean() / H) for c in (1600.0, 1610.0)]
            print(
                f"  {name:8s} {fuel:.3f} kg/s: EUR/day {np.round(eur.sum(1))},"
                f" fuel {np.round(eur[:, 0].mean())}, lost glass h {np.round(lost_h, 1)},"
                f" hottest face max {hot.max():.1f} C, h over 1600 and 1610 C"
                f" {above[0]:.1f}, {above[1]:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
