import jax
import jax.numpy as jnp

from target_gym.pc_gym.cstr.env_jax import CSTR


def test_reset_draws_its_feed_inside_s1s_days():
    """[S1, Tables 3, 4]: the feed at 23.1 and 24.9 C on two days."""
    env = CSTR()
    p = env.default_params
    for k in range(8):
        _, s = env.reset_env(jax.random.PRNGKey(k))
        assert p.feed_T_range[0] <= float(s.T_feed) <= p.feed_T_range[1]


def test_only_the_interlock_trips():
    """[S3]: the interlock acts on high temperature; below it nothing trips,
    however cold the reactor."""
    env = CSTR()
    p = env.default_params
    _, s = env.reset_env(jax.random.PRNGKey(0))
    assert not bool(env.is_terminated(s.replace(T=jnp.asarray(300.0)), p))
    assert bool(env.is_terminated(s.replace(T=jnp.asarray(p.T_trip)), p))
