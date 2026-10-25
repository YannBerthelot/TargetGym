"""``reset_where_done``: a controller's memory at an episode boundary inside a
jitted loop, where nothing calls ``pid.reset()``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.experts import reset_where_done
from target_gym.registry import REGISTRY

WITH_PID = sorted(name for name, spec in REGISTRY.items() if spec.has_pid)


def _leaves_equal(a, b):
    return all(
        np.array_equal(np.asarray(x), np.asarray(y))
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


def _moved(tree):
    """Every leaf changed: booleans flipped, numbers plus one."""
    return jax.tree.map(
        lambda x: (
            jnp.logical_not(x)
            if jnp.asarray(x).dtype == jnp.bool_
            else jnp.asarray(x) + 1
        ),
        tree,
    )


@pytest.mark.parametrize("name", WITH_PID)
def test_done_restores_the_initial_state_and_not_done_keeps_it(name):
    """On every PID's state structure: ``done`` gives back ``init_state``, and
    not-done gives back the state unchanged. The state is moved by hand, since
    some PIDs (the battery's, the wind turbine's below rated) start with
    nothing to integrate."""
    ctrl = REGISTRY[name].make_pid_controller()
    state = _moved(ctrl.init_state)
    assert not _leaves_equal(state, ctrl.init_state)

    assert _leaves_equal(
        reset_where_done(True, state, ctrl.init_state), ctrl.init_state
    )
    assert _leaves_equal(reset_where_done(False, state, ctrl.init_state), state)


def test_a_batch_resets_only_the_rows_that_are_done():
    """On ``env.expert_policy``'s batched state, with a nested state and a
    boolean leaf (the unstable CSTR's cascade), from a batched or an unbatched
    ``fresh``."""
    env = REGISTRY["unstable_cstr"].make_env()
    params = REGISTRY["unstable_cstr"].make_test_params()
    policy = env.expert_policy
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, _ = jax.vmap(env.reset_env, in_axes=(0, None))(keys, params)
    state = policy.init_state(3)
    for _ in range(3):
        _, state = policy(state, obs)
    done = jnp.array([True, False, True])

    fresh = REGISTRY["unstable_cstr"].make_pid_controller().init_state
    for init in (policy.init_state(3), fresh):
        reset = reset_where_done(done, state, init)
        for new, old, zero in zip(
            jax.tree.leaves(reset),
            jax.tree.leaves(state),
            jax.tree.leaves(policy.init_state(3)),
        ):
            np.testing.assert_array_equal(new[1], old[1])
            np.testing.assert_array_equal(new[::2], zero[::2])


def test_back_to_back_episodes_in_one_scan_match_fresh_ones():
    """The use it exists for: one ``lax.scan`` running two CSTR episodes back
    to back, the second starting at step ``K``. Resetting the PID there makes
    the second episode's rewards those of a fresh run; without it the first
    episode's integral leaks in and they differ."""
    spec = REGISTRY["cstr"]
    env, params = spec.make_env(), spec.make_test_params()
    ctrl = spec.make_pid_controller()
    K = 50
    key0, key1 = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

    def rewards(first_key, n, switch_at=None, reset_pid=True):
        obs, env_state = env.reset_env(first_key, params)
        obs1, env_state1 = env.reset_env(key1, params)

        def body(carry, t):
            obs, env_state, pid_state, key = carry
            action, pid_state = ctrl.act(pid_state, obs)
            obs, env_state, reward, _, _ = env.step_env(key, env_state, action, params)
            done = t == (-1 if switch_at is None else switch_at - 1)
            obs = reset_where_done(done, obs, obs1)
            env_state = reset_where_done(done, env_state, env_state1)
            key = reset_where_done(done, key, key1)
            if reset_pid:
                pid_state = reset_where_done(done, pid_state, ctrl.init_state)
            return (obs, env_state, pid_state, key), reward

        carry = (obs, env_state, ctrl.init_state, first_key)
        return jax.lax.scan(body, carry, jnp.arange(n))[1]

    fresh = jax.jit(lambda: rewards(key1, K))()
    chained = jax.jit(lambda: rewards(key0, 2 * K, switch_at=K))()
    leaked = jax.jit(lambda: rewards(key0, 2 * K, switch_at=K, reset_pid=False))()

    np.testing.assert_allclose(chained[K:], fresh, rtol=1e-6, atol=0.0)
    assert not np.allclose(leaked[K:], fresh, rtol=1e-3)
