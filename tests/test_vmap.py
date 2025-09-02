"""Make sure that functions are jax jitable/vmapable"""

import jax
import jax.numpy as jnp

from plane.env_jax import Airplane2D


def test_reset():
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    jax.vmap(env.reset, in_axes=0)(jax.random.split(key, num=3))


def test_step():
    N = 3
    env = Airplane2D()
    key = jax.random.PRNGKey(seed=42)
    keys = jax.random.split(key, num=N)
    obs, state = jax.vmap(env.reset, in_axes=0)(key=keys)
    action = (jnp.ones(N), jnp.zeros(N))
    n_obs, state, reward, terminated, truncated, _ = jax.vmap(env.step, in_axes=0)(
        key=keys, state=state, action=action
    )
