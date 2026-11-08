import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.four_tank.env_jax import FourTank


@pytest.fixture
def env():
    return FourTank()


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_reset_starts_with_the_setpoints_on_the_levels(env, key):
    """Established operation (charter rule 17): the episode opens at rest on
    its setpoints."""
    _, state = env.reset_env(key)
    assert jnp.allclose(state.h1, state.target_h1)
    assert jnp.allclose(state.h2, state.target_h2)


def test_obs_contains_all_levels_and_targets(env, key):
    _, state = env.reset_env(key)
    obs = env.get_obs(state, env.default_params.replace(sensor_noise_scale=0.0))
    assert jnp.allclose(obs[0], state.h1)
    assert jnp.allclose(obs[1], state.h2)
    assert jnp.allclose(obs[2], state.h3)
    assert jnp.allclose(obs[3], state.h4)
    assert jnp.allclose(obs[4], state.target_h1)
    assert jnp.allclose(obs[5], state.target_h2)


def test_an_overflowing_tank_holds_at_its_rim_and_does_not_trip(env, key):
    """A full tank overflows back to the reservoir [S4, p. 15]; no trip."""
    p = env.default_params
    _, state = env.reset_env(key)
    obs, state, _, term, info = env.step_env(
        key, state.replace(h1=p.h_max), jnp.array([1.0, 1.0]), p
    )
    assert float(state.h1) <= p.h_max
    assert not bool(info["tripped"])
