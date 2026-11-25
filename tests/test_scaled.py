"""The optional scaling wrapper (``target_gym.scaled``).

Contract, from the owner's decision of 2026-11-25 (picker, "Documented
wrapper, score unchanged"), not from the code:

- an optional gymnax wrapper scales observations and rewards by fixed per-task
  constants taken from each plant's documented ranges;
- it never touches the scored return: a policy's return read through the
  wrapper (``info["raw_reward"]``) is the one the env gives without it, to the
  bit, and ``score_policy`` and the charter records read the raw reward;
- stated range: a plant inside its documented ranges gives scaled observations
  in [-1, 1], so the shipped expert's short rollout on each of the five suite
  tasks stays there;
- the scaled reward is the raw one over the task's constant, so its sign and
  ordering are the score's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym as tg
from target_gym import scaled
from target_gym.charter import FLAGSHIP

STEPS = 100


def _rollouts(name: str):
    """The expert on the raw env, then its actions replayed through the wrapper
    on the same keys: raw rewards, wrapped rewards, wrapped info, scaled obs
    and raw obs, one row per step."""
    env, params = tg.make(name)
    wrapped = scaled.ScaledEnv(env, name)
    expert = tg.make_expert(name)
    keys = jax.random.split(jax.random.PRNGKey(0), STEPS + 1)

    def raw(carry, key):
        obs, state, memory = carry
        action, memory = expert.step(memory, obs)
        obs2, state, reward, *_ = env.step(key, state, action, params)
        return (obs2, state, memory), (action, reward, obs)

    obs0, state0 = env.reset(keys[0], params)
    _, (actions, rewards, raw_obs) = jax.lax.scan(
        raw, (obs0, state0, expert.reset()), keys[1:]
    )

    def replay(carry, xs):
        obs, state = carry
        key, action = xs
        obs2, state, reward, _, _, info = wrapped.step(key, state, action, params)
        return (obs2, state), (reward, info["raw_reward"], obs)

    z0, w0 = wrapped.reset(keys[0], params)
    _, (w_rewards, w_raw, z) = jax.lax.scan(replay, (z0, w0), (keys[1:], actions))
    return rewards, w_rewards, w_raw, z, raw_obs


@pytest.fixture(scope="module", params=FLAGSHIP)
def rollout(request):
    return request.param, _rollouts(request.param)


def test_every_suite_task_has_documented_constants():
    assert set(FLAGSHIP) <= set(scaled.SCALES)
    for name in FLAGSHIP:
        env, params = tg.make(name)
        obs, _ = env.reset(jax.random.PRNGKey(0), params)
        s = scaled.SCALES[name]
        assert len(s.low) == len(s.high) == obs.shape[-1], name
        assert all(lo < hi for lo, hi in zip(s.low, s.high)), name
        assert s.reward > 0, name


@pytest.mark.medium
def test_the_return_read_through_the_wrapper_is_the_scored_one(rollout):
    name, (rewards, _, w_raw, _, _) = rollout
    np.testing.assert_array_equal(np.asarray(w_raw), np.asarray(rewards))


@pytest.mark.medium
def test_the_scaled_reward_is_the_raw_one_over_the_tasks_constant(rollout):
    name, (rewards, w_rewards, _, _, _) = rollout
    # float32 division: one rounding, under 1e-6 relative.
    np.testing.assert_allclose(
        np.asarray(w_rewards) * scaled.SCALES[name].reward,
        np.asarray(rewards),
        rtol=1e-6,
    )


@pytest.mark.medium
def test_the_experts_scaled_observations_stay_in_the_stated_range(rollout):
    name, (_, _, _, z, _) = rollout
    z = np.asarray(z)
    assert np.all(np.isfinite(z)), name
    worst = np.abs(z).max(axis=0)
    assert np.all(worst <= 1.0), (name, worst.round(3).tolist())


@pytest.mark.medium
def test_unscaling_gives_back_the_plants_observation(rollout):
    name, (_, _, _, z, raw_obs) = rollout
    back = scaled.unscale_obs(name, jnp.asarray(z))
    s = scaled.SCALES[name]
    # float32: the affine map loses about 1e-7 of the documented span.
    atol = 1e-5 * (np.asarray(s.high) - np.asarray(s.low))
    assert np.all(np.abs(np.asarray(back) - np.asarray(raw_obs)) <= atol), name


def test_a_policy_on_scaled_observations_reads_the_raw_ones_through_raw_policy():
    """``raw_policy`` hands ``score_policy`` a policy on the plant's own
    observation: what it passes on is the scaled observation."""
    seen = []
    policy = scaled.raw_policy("cstr", lambda z: seen.append(z) or jnp.zeros(2))
    s = scaled.SCALES["cstr"]
    low, high = jnp.asarray(s.low), jnp.asarray(s.high)
    policy(low)
    policy(high)
    np.testing.assert_allclose(np.asarray(seen[0]), -1.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(seen[1]), 1.0, atol=1e-6)
