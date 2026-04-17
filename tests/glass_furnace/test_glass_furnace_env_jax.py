import jax
import jax.numpy as jnp
import pytest

from target_gym.glass_furnace.env import (
    N_SETPOINTS,
    GlassFurnaceParams,
    GlassFurnaceState,
)
from target_gym.glass_furnace.env_jax import GlassFurnace


def _make_state(**overrides) -> GlassFurnaceState:
    """Build a GlassFurnaceState with sensible defaults for the new schema."""
    defaults = dict(
        time=0,
        T_crown=1577.0,
        T_melt=1497.0,
        T_work=1227.0,
        target_T_crown=1587.0,
        target_schedule=jnp.full((N_SETPOINTS,), 1587.0),
        m_pull_disturbance=jnp.zeros(()),
        fuel_flow=1.0,
    )
    defaults.update(overrides)
    return GlassFurnaceState(**defaults)


def test_reset_env_returns_obs_and_state():
    env = GlassFurnace()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    assert obs.shape == env.obs_shape
    assert isinstance(state, GlassFurnaceState)

    assert state.time == 0

    # Initial crown temperature within the configured initial range
    assert (
        env.default_params.initial_T_crown_range[0]
        <= float(state.T_crown)
        <= env.default_params.initial_T_crown_range[1]
    )
    # Target within target range
    assert (
        env.default_params.target_T_crown_range[0]
        <= float(state.target_T_crown)
        <= env.default_params.target_T_crown_range[1]
    )
    # Glass zones initialised to the configured defaults
    assert float(state.T_melt) == pytest.approx(env.default_params.initial_T_melt)
    assert float(state.T_work) == pytest.approx(env.default_params.initial_T_work)


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_step_env_updates_state_and_obs(method):
    env = GlassFurnace(integration_method=method)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    action = jnp.array([0.0])

    obs2, state2, reward, done, info = env.step_env(key, state, action)

    assert obs2.shape == env.obs_shape
    assert isinstance(state2, GlassFurnaceState)
    assert state2.time == state.time + 1
    assert jnp.isfinite(reward)
    assert done.dtype == jnp.bool_

    assert "last_state" in info
    assert isinstance(info["last_state"], GlassFurnaceState)


def test_action_and_observation_space():
    env = GlassFurnace()

    a_space = env.action_space()
    assert a_space.shape == (1,)
    assert jnp.all(a_space.low == -1.0)
    assert jnp.all(a_space.high == 1.0)

    o_space = env.observation_space(env.default_params)
    assert o_space.shape == env.obs_shape

    s_space = env.state_space(env.default_params)
    assert s_space.shape == len(GlassFurnaceState.__dataclass_fields__)


def test_is_terminal_propagates_logic():
    env = GlassFurnace()
    # A "nice" state shouldn't terminate
    state = _make_state()
    terminated, truncated = env.is_terminal(state, env.default_params)
    assert bool(terminated) is False
    assert bool(truncated) is False


def test_get_obs_matches_manual_call():
    env = GlassFurnace()
    state = _make_state()
    obs1 = env.get_obs(state)
    obs2 = env.get_obs(state, env.default_params)
    assert obs1.shape == env.obs_shape
    assert jnp.allclose(obs1, obs2)


def test_observation_hides_glass_temperatures():
    """Glass-zone temperatures should not appear in the observation vector."""
    env = GlassFurnace()
    state = _make_state(T_melt=1234.5, T_work=5678.9)
    obs = env.get_obs(state)
    # Only 3 dims — T_crown, fuel_flow, target
    assert obs.shape == (3,)
    # Glass temperatures (1234.5, 5678.9) must not leak into the obs
    assert not bool(jnp.any(jnp.isclose(obs, 1234.5)))
    assert not bool(jnp.any(jnp.isclose(obs, 5678.9)))


def test_full_episode_is_finite():
    """Running a short episode with constant zero raw fuel should stay finite."""
    env = GlassFurnace()
    params = GlassFurnaceParams(max_steps_in_episode=50)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)

    for _ in range(50):
        _, state, reward, done, _ = env.step_env(key, state, jnp.array([0.0]), params)
        assert jnp.isfinite(state.T_crown)
        assert jnp.isfinite(state.T_melt)
        assert jnp.isfinite(state.T_work)
        assert jnp.isfinite(reward)
