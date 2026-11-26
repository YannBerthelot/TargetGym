"""A disturbance left alone decays at its stated rate, whatever the step.

Contract (owner, 2026-11-24, "exact decay"): each mean-reverting disturbance is
the process ``dx = -theta x dt + noise``. With its innovations off, a deviation
``d`` from its rest is therefore ``d exp(-theta t)`` after a time ``t``, the
solution of that equation, and not the ``d (1 - theta dt)^n`` of n Euler steps,
which decays ``theta dt / 2`` faster than stated and depends on the step.

Expected values are that closed form, from the rate each plant states. Over the
20 steps read here an Euler step is off by 1e-3 to 6e-3 of the deviation (pH the
least, at theta dt = 0.01), and float32 by about 2e-6, so 5e-5 tells them apart.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.cement_kiln.env import FEED_OU_THETA
from target_gym.hvac.env import WEATHER_OU_THETA
from target_gym.pc_gym.ph_neutralization.env import BUFFER_OU_THETA

STEPS = 20
TOLERANCE = 5e-5

#: task: (state field, its rest, its rate in 1/s, the deviation set).
CASES = {
    "hvac": ("weather_dev", lambda p: 0.0, lambda p: WEATHER_OU_THETA, 3.0),
    "ph_neutralization": (
        "q2",
        lambda p: p.q2_nominal,
        lambda p: BUFFER_OU_THETA,
        0.4,
    ),
    "cement_kiln": (
        "raw_meal",
        lambda p: p.raw_meal_nominal,
        lambda p: FEED_OU_THETA,
        5.0,
    ),
    "reactor": ("load_wobble", lambda p: 0.0, lambda p: p.demand_theta, 0.004),
}


def _seconds(before, after, params):
    """Time between two states: the reactor counts its own sub-steps."""
    if hasattr(before, "physics_time"):
        return float(after.physics_time - before.physics_time) * params.delta_t
    return float(after.time - before.time) * params.delta_t


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_deviation_left_alone_decays_at_the_stated_rate(name):
    field, rest, rate, deviation = CASES[name]
    spec = registry.REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params(
        sensor_noise_scale=0.0, **{f: 0.0 for f in spec.noise_fields}
    )
    key = jax.random.PRNGKey(0)
    _, start = env.reset_env(key, params)
    start = start.replace(**{field: rest(params) + deviation})
    action = np.zeros(env.action_space(params).shape, np.float32)

    state = start
    for t in range(STEPS):
        _, state, _, _, _ = env.step_env(
            jax.random.fold_in(key, t), state, action, params
        )

    left = float(getattr(state, field)) - rest(params)
    expected = deviation * np.exp(-rate(params) * _seconds(start, state, params))
    assert left == pytest.approx(expected, rel=TOLERANCE)
