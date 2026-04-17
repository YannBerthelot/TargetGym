import jax.numpy as jnp
import pytest

from target_gym.pc_gym.first_order.env import (
    FirstOrderParams,
    FirstOrderState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
)


@pytest.fixture
def params():
    return FirstOrderParams()


@pytest.fixture
def state(params):
    return FirstOrderState(time=0, x=0.0, target_x=1.0, u=0.0)


# -----------------------------------------------------------------------
# Velocity / ODE structure
# -----------------------------------------------------------------------


def test_velocity_shape(params):
    v, _ = compute_velocity(jnp.array([0.5]), action=1.0, params=params)
    assert v.shape == (1,)


def test_velocity_positive_when_driven_from_zero(params):
    """With u > 0 and x = 0, the state should be driven upward (dxdt > 0)."""
    v, _ = compute_velocity(jnp.array([0.0]), action=1.0, params=params)
    assert v[0] > 0.0


def test_velocity_negative_when_above_equilibrium(params):
    """With u = 0 and x > 0, the system decays back to zero (dxdt < 0)."""
    v, _ = compute_velocity(jnp.array([1.0]), action=0.0, params=params)
    assert v[0] < 0.0


def test_equilibrium_at_steady_state(params):
    """dxdt = 0 when x = K * u (steady-state condition: (K*u - x)/tau = 0)."""
    u = 0.5
    x_eq = params.K * u
    v, _ = compute_velocity(jnp.array([x_eq]), action=u, params=params)
    assert jnp.isclose(v[0], 0.0, atol=1e-6)


def test_larger_tau_slows_dynamics(params):
    """Doubling tau should halve the rate of change for the same deviation."""
    x = 0.0
    u = 1.0
    v_fast, _ = compute_velocity(
        jnp.array([x]), action=u, params=FirstOrderParams(tau=0.5)
    )
    v_slow, _ = compute_velocity(
        jnp.array([x]), action=u, params=FirstOrderParams(tau=1.0)
    )
    assert abs(v_fast[0]) == pytest.approx(2 * abs(v_slow[0]), rel=1e-5)


def test_larger_gain_increases_velocity(params):
    """Higher K means a larger driving term for the same u."""
    x = 0.0
    u = 1.0
    v_low, _ = compute_velocity(
        jnp.array([x]), action=u, params=FirstOrderParams(K=1.0)
    )
    v_high, _ = compute_velocity(
        jnp.array([x]), action=u, params=FirstOrderParams(K=2.0)
    )
    assert v_high[0] > v_low[0]


# -----------------------------------------------------------------------
# Integration
# -----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_next_state_advances_time(method, params, state):
    new_state, _ = compute_next_state(0.5, state, params, integration_method=method)
    assert new_state.time == state.time + 1
    assert not jnp.allclose(new_state.x, state.x)


def test_integration_methods_agree_for_small_dt(state):
    params = FirstOrderParams(delta_t=1e-4)
    action = 0.3
    s_euler, _ = compute_next_state(action, state, params, integration_method="euler_1")
    s_rk4, _ = compute_next_state(action, state, params, integration_method="rk4_1")
    assert jnp.allclose(s_euler.x, s_rk4.x, atol=1e-5)


# -----------------------------------------------------------------------
# Action scaling and clipping
# -----------------------------------------------------------------------


def test_action_scaled_to_range(params, state):
    """Raw action -1 maps to u_min, raw action +1 maps to u_max."""
    s_min, _ = compute_next_state(-1.0, state, params, integration_method="euler_1")
    s_max, _ = compute_next_state(1.0, state, params, integration_method="euler_1")
    assert s_min.u == pytest.approx(params.u_min)
    assert s_max.u == pytest.approx(params.u_max)


def test_action_clipped_beyond_range(params, state):
    s_over, _ = compute_next_state(100.0, state, params, integration_method="euler_1")
    s_under, _ = compute_next_state(-100.0, state, params, integration_method="euler_1")
    assert s_over.u == pytest.approx(params.u_max)
    assert s_under.u == pytest.approx(params.u_min)


# -----------------------------------------------------------------------
# Terminal / Reward
# -----------------------------------------------------------------------


def test_not_terminal_in_bounds(params, state):
    terminated, truncated = check_is_terminal(state, params)
    assert not terminated
    assert not truncated


def test_terminal_when_out_of_bounds(params, state):
    term, _ = check_is_terminal(state.replace(x=params.x_max + 0.1), params)
    assert term
    term, _ = check_is_terminal(state.replace(x=params.x_min - 0.1), params)
    assert term


def test_truncated_at_max_steps(params, state):
    _, truncated = check_is_terminal(
        state.replace(time=params.max_steps_in_episode), params
    )
    assert truncated


def test_reward_is_1_at_target(params):
    state = FirstOrderState(time=0, x=1.0, target_x=1.0, u=0.0)
    reward = compute_reward(state, params)
    assert reward == pytest.approx(1.0)


def test_reward_decreases_with_error(params):
    r_small = compute_reward(
        FirstOrderState(time=0, x=0.9, target_x=1.0, u=0.0), params
    )
    r_large = compute_reward(
        FirstOrderState(time=0, x=0.5, target_x=1.0, u=0.0), params
    )
    assert r_small > r_large


def test_reward_is_finite(params, state):
    assert jnp.isfinite(compute_reward(state, params))
