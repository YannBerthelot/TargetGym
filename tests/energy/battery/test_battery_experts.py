"""The battery's oracle: the signal's next value, inside the cycle-depth bound.

Where the charge window does not bind, its claim is that it leaves only the
regulation innovation. With the noise switched off it tracks the target to
float32 rounding, which pins the block it reads (the one the step is scored
against, ``dispatch_block(t + 1)``). With the noise on, its mean error is the
closed-form floor ``e_floor``. Levels of 0.1 MW keep the window out of the way.
"""

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.energy.battery.oracle import ScheduleFeedforward
from target_gym.experts.mpc import plan_params


def _errors(params, seed):
    """|target - power| on every step of one episode under the oracle."""
    spec = registry.get("battery")
    env = spec.make_env()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    oracle.reset()
    key = jax.random.PRNGKey(seed)
    _, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    errors = []
    for _ in range(int(params.max_steps_in_episode)):
        action = np.atleast_1d(oracle.step(None, state))
        _, state, _, _, info = step(key, state, action, params)
        assert not bool(info["tripped"])
        errors.append(abs(float(state.target_power) - float(state.power)))
    return np.array(errors)


def test_the_registry_builds_the_feedforward():
    spec = registry.get("battery")
    oracle = spec.make_mpc(spec.make_env(), spec.make_test_params())
    assert isinstance(oracle, ScheduleFeedforward)
    assert oracle.solver_report() == {}


def _unbound(**fields):
    return (
        registry.get("battery")
        .make_test_params()
        .replace(dispatch_range=1e5, max_steps_in_episode=720, **fields)
    )


def test_it_never_trips_where_following_exactly_would():
    """The default window, on seeds where the pack runs out under exact following."""
    params = registry.get("battery").make_test_params()
    for seed in range(100, 104):
        _errors(params, seed)  # asserts no trip


def test_without_noise_it_tracks_every_step_to_rounding():
    params = _unbound(dispatch_noise_std=0.0)
    # float32 rounding leaves a few W; reading the wrong block misses by the jump
    # between levels, typically tens of kW here.
    tolerance = 1e-4 * float(params.power_max)  # 100 W
    for seed in range(3):
        assert _errors(params, seed).max() < tolerance


def test_with_noise_it_holds_the_closed_form_floor():
    params = _unbound()
    mean_error = np.mean([_errors(params, seed).mean() for seed in range(3)])
    # 2160 steps of |N(0, 10 kW)| have a standard error of about 130 W.
    assert mean_error == pytest.approx(float(params.e_floor), rel=0.05)
