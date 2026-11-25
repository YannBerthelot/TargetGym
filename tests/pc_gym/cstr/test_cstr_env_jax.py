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


def test_a_zero_feed_period_holds_the_feed_where_it_starts():
    """CSTRParams: ``feed_T_period`` 0 holds the feed where it starts. Run
    eagerly too, where the params are plain floats rather than tracers."""
    env = CSTR()
    p = env.default_params.replace(feed_T_period=0.0)
    with jax.disable_jit():
        _, s = env.reset_env(jax.random.PRNGKey(0), p)
        start = float(s.T_feed)
        for k in range(3):
            _, s, *_ = env.step_env(
                jax.random.PRNGKey(k), s, jnp.zeros((2,), jnp.float32), p
            )
            assert float(s.T_feed) == start


def test_a_window_starts_in_established_production():
    """Charter B5 and [S13, pp. 436, 438]: the window starts with the water
    valve held short of its limit, at 95 % of its span, the feed that takes and
    the reactor on its setpoint; where the feed's top binds, the feed at its top
    and the water under 95 %. The observation shows both valves in percent."""
    env = CSTR()
    p = env.default_params
    for k in range(8):
        obs, s = env.reset_env(jax.random.PRNGKey(k), p.replace(sensor_noise_scale=0.0))
        water, feed = float(obs[2]), float(obs[3])
        assert obs.shape == (5,)
        assert abs(float(obs[4]) - p.T_setpoint) < 1e-3
        assert abs(float(s.T) - p.T_setpoint) < 1e-3
        at_rest = abs(water - 95.0) < 0.01 and feed < 100.0
        feed_bound = abs(feed - 100.0) < 0.01 and water < 95.0
        assert at_rest or feed_bound, (k, water, feed)
