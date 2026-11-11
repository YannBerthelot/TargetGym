"""The reactor's table of starting axial tilts (the physics review's finding 3).

Run with ``uv run python scripts/reactor_tilt_table.py``. The expert (the
reference) runs a whole day of each scenario on training seeds 0 to 99, scored
configuration, from no tilt, and the day's last ``(I_tilt, Xe_tilt)`` is kept.
Rows are taken in proportion to the scenarios' weights, ``ROWS`` in all, so a draw
from the table is yesterday's end of day for a day picked by its weight. Prints
the table for ``ReactorParams.initial_tilts``. Held-out and nominal seeds (100 and
up) are not used.
"""

import jax
import numpy as np

from target_gym import scenarios
from target_gym.reactor import Reactor
from target_gym.reactor.experts import make_reactor_pid_controller
from target_gym.registry import REGISTRY

ROWS = 100
SEEDS = 100


def end_of_day(params, seed):
    env, expert = Reactor(), make_reactor_pid_controller()
    obs, s = env.reset_env(jax.random.PRNGKey(seed), params)
    keys = jax.random.split(jax.random.PRNGKey(seed + 7), params.max_steps_in_episode)

    def step(c, k):
        s, obs, m = c
        a, m = expert.step(expert.params, m, obs)
        obs, s, _, _, _ = env.step_env(k, s, a, params)
        return (s, obs, m), None

    (s, _, _), _ = jax.lax.scan(step, (s, obs, expert.init_state), keys)
    return float(s.I_tilt), float(s.Xe_tilt)


def main():
    spec = REGISTRY["reactor"]
    suite = scenarios.suite(spec)
    counts = [round(sc.weight * ROWS) for sc in suite]
    counts[int(np.argmax(counts))] += ROWS - sum(counts)
    table = []
    for sc, count in zip(suite, counts):
        params = scenarios.params_for(spec, sc)
        ends = [end_of_day(params, seed) for seed in range(min(count, SEEDS))]
        table += [(round(i, 6), round(x, 6)) for i, x in ends]
        tilts = np.array(ends) if ends else np.zeros((0, 2))
        print(sc.name, count, "Xe_tilt range", tilts[:, 1].min() if count else None,
              tilts[:, 1].max() if count else None, flush=True)  # fmt: skip
    print("TABLE", tuple(table))


if __name__ == "__main__":
    main()
