"""The battery's oracle: feedforward of the scheduled dispatch level.

Its claim is that it leaves only the dispatch noise. With the noise switched
off it tracks the target to float32 rounding, which pins the block it reads
(the one the step is scored against, ``dispatch_block(t + 1)``). With the
noise on, its mean error is the closed-form floor ``e_floor``.
"""

import jax
import numpy as np
import pytest

from target_gym import registry
from target_gym.energy.battery.experts import ScheduleFeedforward
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


def test_without_noise_it_tracks_every_step_to_rounding():
    params = registry.get("battery").make_test_params().replace(dispatch_noise_std=0.0)
    # float32 rounding leaves about 1-2 W; reading the wrong block misses by
    # the jump between levels, typically hundreds of kW.
    tolerance = 1e-4 * float(params.power_max)  # 100 W
    for seed in range(3):
        assert _errors(params, seed).max() < tolerance


def test_with_noise_it_holds_the_closed_form_floor():
    params = registry.get("battery").make_test_params()
    mean_error = np.mean([_errors(params, seed).mean() for seed in range(3)])
    # 1080 steps of |N(0, 2 kW)| have a standard error of about 37 W.
    assert mean_error == pytest.approx(float(params.e_floor), rel=0.05)
