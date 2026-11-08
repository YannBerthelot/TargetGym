"""Unmeasured load steps on ``cstr``, and the ``cstr`` analyser
(docs/proposals/industrial-fixes.md, PR 4).

Contract, taken from the proposal and not from the code:

- A load arrives only after the task's fixed settle, so the hold is scored
  against it, then moves every ``load_period`` steps.
- No observation channel carries it.
- It belongs to the episode: every policy on one episode key meets the same
  loads, whatever its actions and step keys, and other episodes meet others.
- It stays inside its limits, and ``load_scale`` 0 switches it off.
- Every ``cstr`` analyser reading is 1 min old, 4 steps of 0.25 min, and the
  temperature is current. The reward reads the plant, not the analyser.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import eval as E
from target_gym.registry import REGISTRY

#: A raw action that keeps each plant inside its trip band for a whole episode.
# four_tank left in its version 4: the rig's disturbances are an extra outlet
# and a cup (pc_gym/four_tank/PHYSICS.md), tested in test_four_tank_reality.py.
RAW = {"cstr": 0.0}
LIMIT = {"cstr": ("Ti_load", "Caf_load")}
TRACKED = {"cstr": "C_a"}
NAMES = tuple(RAW)


def _rollout(name, seed=0, raw=None, step_seed=0, **overrides):
    """``(load, obs, states, reward, params)`` per step under a constant action."""
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params(**overrides)
    state = env.reset_env(jax.random.PRNGKey(seed), params)[1]
    n = int(params.max_steps_in_episode)
    action = jnp.full(env.action_space(params).shape, RAW[name] if raw is None else raw)

    def step(s, k):
        obs, s, reward, *_ = env.step_env(k, s, action, params)
        return s, (s.load, obs, s, reward)

    keys = jax.random.split(jax.random.PRNGKey(step_seed), n)
    load, obs, states, reward = jax.lax.scan(step, state, keys)[1]
    return np.asarray(load), np.asarray(obs), states, np.asarray(reward), params


@pytest.mark.parametrize("name", NAMES)
def test_a_load_arrives_after_the_settle_and_moves_on_its_period(name):
    load, *_, params = _rollout(name)
    start, period = int(params.load_start), int(params.load_period)
    assert start >= E.fixed_settle(name)
    moves = np.flatnonzero(np.any(np.diff(load, axis=0) != 0.0, axis=1)) + 1
    np.testing.assert_array_equal(moves, np.arange(start, len(load), period))
    assert np.all(load[:start] == 0.0)


@pytest.mark.parametrize("name", NAMES)
def test_a_load_stays_inside_its_limits(name):
    params = REGISTRY[name].make_test_params()
    limits = np.array([float(getattr(params, f)) for f in LIMIT[name]])
    loads = np.concatenate([_rollout(name, seed=s)[0] for s in range(20)])
    assert np.all(np.abs(loads) <= limits * (1.0 + 1e-6))
    # The draws use the range: without this a load of 0 would pass.
    assert np.all(np.abs(loads).max(axis=0) >= 0.8 * limits)


@pytest.mark.parametrize("name", NAMES)
def test_every_policy_on_an_episode_key_meets_the_same_loads(name):
    base = _rollout(name)[0]
    other = _rollout(name, raw=RAW[name] + 0.1, step_seed=1)[0]
    np.testing.assert_array_equal(base, other)
    assert not np.array_equal(base, _rollout(name, seed=1)[0])


@pytest.mark.parametrize("name", NAMES)
def test_the_scale_multiplies_a_load_and_0_removes_it(name):
    assert np.all(_rollout(name, load_scale=0.0)[0] == 0.0)
    half = _rollout(name, load_scale=0.5)[0]
    np.testing.assert_allclose(half, 0.5 * _rollout(name)[0], rtol=1e-6)


@pytest.mark.parametrize("name", NAMES)
def test_a_load_moves_the_plant_and_no_observation_carries_it(name):
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    _, obs, states, *_ = _rollout(name)
    _, quiet_obs, quiet, *_ = _rollout(name, load_scale=0.0)
    start = int(params.load_start)
    field = TRACKED[name]
    # Identical up to the arrival, whose step is the first to feel it.
    same = np.asarray(getattr(states, field)[:start])
    np.testing.assert_array_equal(same, np.asarray(getattr(quiet, field)[:start]))
    assert getattr(states, field)[-1] != getattr(quiet, field)[-1]
    state = jax.tree_util.tree_map(lambda x: x[start + 1], states)
    np.testing.assert_array_equal(
        env.get_obs(state, params),
        env.get_obs(state.replace(load=jnp.zeros(2)), params),
    )


def test_the_analyser_reports_the_concentration_of_one_minute_ago():
    _, obs, states, reward, params = _rollout("cstr", sensor_noise_scale=0.0)
    C_a = np.asarray(states.C_a)
    late = 4  # 1 min / 0.25 min
    np.testing.assert_array_equal(obs[late:, 0], C_a[:-late])
    np.testing.assert_array_equal(obs[:, 1], np.asarray(states.T))
    # The reward reads the plant now: (target - C_a)^2 / e_floor^2.
    target, floor = np.asarray(states.target_CA), float(params.e_floor)
    np.testing.assert_allclose(reward, -(((target - C_a) / floor) ** 2), rtol=1e-5)
