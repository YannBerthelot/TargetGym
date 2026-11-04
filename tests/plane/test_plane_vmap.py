"""The 2D aircraft under ``jax.vmap`` with a tuple action.

``tests/test_env_conformance.py`` covers vmap for every registered environment.
This file stays for what the generic version cannot reach: the
2D aircraft accepts its action as a *tuple* of ``(power, stick)`` rather than a
stacked array, which is a distinct code path through the batching rules.
"""

import jax
import jax.numpy as jnp
import numpy as np

from target_gym.plane.env_jax import Airplane2D

N = 3


def test_step_batches_over_seeds_with_a_tuple_action():
    env = Airplane2D()
    keys = jax.random.split(jax.random.PRNGKey(42), num=N)
    obs, state = jax.vmap(env.reset, in_axes=0)(key=keys)

    action = (jnp.ones(N), jnp.zeros(N))
    next_obs, next_state, reward, terminated, truncated, _ = jax.vmap(
        env.step, in_axes=0
    )(key=keys, state=state, action=action)

    assert next_obs.shape == obs.shape
    assert reward.shape == (N,)
    assert terminated.shape == (N,) and truncated.shape == (N,)
    assert np.isfinite(np.asarray(next_obs)).all()
    assert np.isfinite(np.asarray(reward)).all()

    # The step has to have done something, and something different per seed.
    assert not np.allclose(np.asarray(next_obs), np.asarray(obs))
    assert not np.allclose(np.asarray(next_obs)[0], np.asarray(next_obs)[1])
    for leaf in jax.tree_util.tree_leaves(next_state):
        assert np.asarray(leaf).shape[0] == N
