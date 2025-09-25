import jax
import jax.numpy as jnp
import pytest

from target_gym.pc_gym.cstr.env import CSTRParams, CSTRState
from target_gym.pc_gym.cstr.env_jax import CSTR


def test_reset_env_returns_obs_and_state():
    env = CSTR()
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    # Obs shape matches env.obs_shape
    assert obs.shape == env.obs_shape
    assert isinstance(state, CSTRState)

    # Reset initializes time to 0
    assert state.time == 0

    # Initial C_a is within allowed range
    assert (
        env.default_params.initial_CA_range[0]
        <= state.C_a
        <= env.default_params.initial_CA_range[1]
    )

    # Target_Ca is within allowed range
    assert (
        env.default_params.target_CA_range[0]
        <= state.target_CA
        <= env.default_params.target_CA_range[1]
    )


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_step_env_updates_state_and_obs(method):
    env = CSTR(integration_method=method)
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key)

    action = 0.0  # neutral action

    obs2, state2, reward, done, info = env.step_env(
        key,
        state,
        action,
    )

    # Obs shape is correct
    assert obs2.shape == env.obs_shape

    # State updated and time advanced
    assert isinstance(state2, CSTRState)
    assert state2.time == state.time + 1

    # Reward is finite
    assert jnp.isfinite(reward)

    # Done is boolean
    assert done.dtype == jnp.bool_

    # Info dict contains last_state
    assert "last_state" in info
    assert isinstance(info["last_state"], CSTRState)


def test_action_and_observation_space():
    env = CSTR()

    a_space = env.action_space()
    assert a_space.shape == (1,)
    assert jnp.all(a_space.low == -1.0)
    assert jnp.all(a_space.high == 1.0)

    o_space = env.observation_space(env.default_params)
    assert o_space.shape == env.obs_shape

    s_space = env.state_space(env.default_params)
    assert s_space.shape == len(CSTRState.__dataclass_fields__)


def test_is_terminal_propagates_logic():
    env = CSTR()
    state = CSTRState(time=0, C_a=0.5, T=350.0, target_CA=0.6, T_c=298.0)
    term, trunc = env.is_terminal(state, env.default_params)
    # assert isinstance(result, jnp.ndarray)


def test_get_obs_matches_manual_call():
    env = CSTR()
    state = CSTRState(time=0, C_a=0.5, T=350.0, target_CA=0.6, T_c=298.0)

    obs1 = env.get_obs(state)
    obs2 = env.get_obs(state, env.default_params)
    assert obs1.shape == env.obs_shape
    assert jnp.allclose(obs1, obs2)
