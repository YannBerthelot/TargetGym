"""The null policies on the rebuilt reactor, seed by seed against the reference.

The shared table in ``data/null_policies`` predates the two-input plant, so this
fits its own: a quadratic in the load through the reference's actions on training
seeds 0 to 9. This is the ten-seed check of the rebuild, not rule 4, which runs once
from the tag on hashed seeds.

    uv run python scripts/reactor_nulls.py ladder.json
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import reactor_foresight as foresight  # noqa: E402
import reactor_ladder as ladder  # noqa: E402

from target_gym.reactor import Reactor, ReactorParams  # noqa: E402


def fit_table(env, p, ref):
    """The reference's actions as a quadratic in the load, on training seeds."""
    seeds = jnp.arange(10)
    actions = jax.vmap(lambda s: foresight.reference_actions(env, p, ref, s)[0])(seeds)
    loads = jax.vmap(lambda s: _loads(env, p, s, actions[s]))(seeds)
    x, y = np.asarray(loads).ravel(), np.asarray(actions).reshape(-1, 2)
    return jnp.asarray(np.polyfit(x, y, 2).T, jnp.float32)


def _loads(env, p, seed, plan):
    _, s = env.reset_env(jax.random.PRNGKey(seed), p)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), p.max_steps_in_episode)

    def step(s, x):
        a, key = x
        obs, s, _, _, _ = env.step_env(key, s, a, p)
        return s, obs[3]

    return jax.lax.scan(step, s, (plan, keys))[1]


def main():
    env, p = Reactor(), ReactorParams()
    saved = json.load(open(sys.argv[1]))
    ref = ladder.reference(saved, p)
    reference = np.asarray(saved["table"][saved["best_rung"]]["cost_by_seed"])
    coef = fit_table(env, p, ref)
    zero = jnp.asarray(0.0)

    def table(obs):
        return jax.vmap(lambda c: jnp.polyval(c, obs[3]))(coef)

    def constant(u):
        return lambda m, o: (jnp.asarray(u, jnp.float32), m)

    def table_trim(m, o):
        m = m + 0.0132 * (o[5] - o[4])
        return jnp.clip(table(o) + jnp.stack([m, zero]), -1.0, 1.0), m

    def rod_integrator(m, o):
        rod = o[2] + 2.0 / 0.015 * 0.0163 * (o[3] - o[0])
        return jnp.stack([rod, zero]), m

    def tavg_integrator(m, o):
        return jnp.stack([o[2] + 0.0132 * (o[5] - o[4]), zero]), m

    nulls = {
        "constant low": constant([-1.0, -1.0]),
        "constant middle": constant([0.0, 0.0]),
        "constant high": constant([1.0, 1.0]),
        "rods parked, no boron": constant([ladder.PARK, 0.0]),
        "table": lambda m, o: (table(o), m),
        "table plus Tavg integrator": table_trim,
        "witness: rod integrator": rod_integrator,
        "witness: Tavg integrator": tavg_integrator,
    }
    out = {}
    for name, act in nulls.items():
        row = ladder.rows(env, p, act, zero)
        costs = np.asarray(row["cost_by_seed"])
        row["wins_over_reference"] = int(np.sum(costs < reference))
        out[name] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
