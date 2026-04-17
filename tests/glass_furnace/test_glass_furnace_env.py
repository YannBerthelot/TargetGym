import jax
import jax.numpy as jnp
import pytest

from target_gym.glass_furnace.env import (
    N_SETPOINTS,
    GlassFurnaceParams,
    GlassFurnaceState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
)


def _default_state(
    params: GlassFurnaceParams, target: float = 1587.0
) -> GlassFurnaceState:
    return GlassFurnaceState(
        time=0,
        T_crown=1577.0,
        T_melt=params.initial_T_melt,
        T_work=params.initial_T_work,
        target_T_crown=target,
        target_schedule=jnp.full((N_SETPOINTS,), target),
        m_pull_disturbance=jnp.zeros(()),
        fuel_flow=1.0,
    )


def test_compute_velocity_shape_and_trends():
    """Velocity vector has correct shape and expected qualitative trends."""
    params = GlassFurnaceParams()
    position = jnp.array([1577.0, 1497.0, 1427.0])  # [T_crown, T_melt, T_work] in °C

    v_high, _ = compute_velocity(
        position, action=1.8, m_pull=params.m_pull, params=params
    )  # high fuel
    v_low, _ = compute_velocity(
        position, action=0.4, m_pull=params.m_pull, params=params
    )  # low fuel

    # Shape: 3 state derivatives
    assert v_high.shape == (3,)

    # More fuel should raise dT_crown/dt (strictly)
    assert v_high[0] > v_low[0]

    # Glass derivatives don't depend on the action directly — only through T_crown.
    # So at the same position, dT_melt and dT_work are independent of fuel flow.
    assert jnp.allclose(v_high[1], v_low[1])
    assert jnp.allclose(v_high[2], v_low[2])

    # With zero fuel, crown should cool (heat loss dominates)
    v_zero, _ = compute_velocity(
        position, action=0.0, m_pull=params.m_pull, params=params
    )
    assert v_zero[0] < 0.0


def test_no_fuel_cools_crown():
    """With zero fuel flow, the crown must eventually cool."""
    params = GlassFurnaceParams()
    position = jnp.array([1577.0, 1497.0, 1227.0])
    v, _ = compute_velocity(position, action=0.0, m_pull=params.m_pull, params=params)
    # All exhaust + wall losses, no heat input -> T_crown drops
    assert v[0] < 0.0


def test_glass_cooler_than_crown_heats_from_crown():
    """If crown is much hotter than glass, glass zones should receive positive net heat."""
    params = GlassFurnaceParams()
    # Crown way above glass -> radiation dominates wall loss
    position = jnp.array([1627.0, 1227.0, 1127.0])
    v, _ = compute_velocity(position, action=1.0, m_pull=params.m_pull, params=params)
    assert v[1] > 0.0  # melt heats
    assert v[2] > 0.0  # work heats


def test_pull_rate_advects_heat_forward():
    """Advection at pull rate moves heat from melt to work zone."""
    params = GlassFurnaceParams()
    # Same crown temp, but T_melt > T_work -> advection term positive for work
    pos_1 = jnp.array([1577.0, 1507.0, 1127.0])
    pos_2 = jnp.array([1577.0, 1507.0, 1427.0])  # T_work closer to T_melt

    v1, _ = compute_velocity(pos_1, action=1.0, m_pull=params.m_pull, params=params)
    v2, _ = compute_velocity(pos_2, action=1.0, m_pull=params.m_pull, params=params)

    # Larger T_melt - T_work gap -> larger advection contribution -> dT_work/dt larger in v1
    assert v1[2] > v2[2]


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_compute_next_state_progression(method):
    """State evolves and time increments."""
    params = GlassFurnaceParams()
    state = _default_state(params)
    key = jax.random.PRNGKey(0)

    new_state, metrics = compute_next_state(
        fuel_raw=0.5,
        state=state,
        params=params,
        key=key,
        integration_method=method,
    )

    assert isinstance(new_state, GlassFurnaceState)
    assert new_state.time == state.time + 1
    assert not jnp.allclose(new_state.T_crown, state.T_crown)

    # fuel_flow is the actual converted fuel, clamped to [fuel_min, fuel_max]
    assert params.fuel_min <= new_state.fuel_flow <= params.fuel_max
    assert metrics is None


def test_integration_methods_agree_for_small_dt():
    """RK4 and Euler should give similar results with a small dt."""
    # Disable pull-rate noise so the two methods share the same dynamics.
    params = GlassFurnaceParams(delta_t=0.01, m_pull_noise_std=0.0)
    state = _default_state(params)
    key = jax.random.PRNGKey(0)

    action = 0.3
    new_state_euler, _ = compute_next_state(
        fuel_raw=action,
        state=state,
        params=params,
        key=key,
        integration_method="euler_1",
    )
    new_state_rk4, _ = compute_next_state(
        fuel_raw=action,
        state=state,
        params=params,
        key=key,
        integration_method="rk4_1",
    )

    assert jnp.allclose(new_state_euler.T_crown, new_state_rk4.T_crown, atol=1e-2)
    assert jnp.allclose(new_state_euler.T_melt, new_state_rk4.T_melt, atol=1e-2)
    assert jnp.allclose(new_state_euler.T_work, new_state_rk4.T_work, atol=1e-2)


def test_action_clipping():
    """Raw action is scaled and clipped to [fuel_min, fuel_max]."""
    params = GlassFurnaceParams()
    state = _default_state(params)
    key = jax.random.PRNGKey(0)

    new_state, _ = compute_next_state(
        fuel_raw=100.0,
        state=state,
        params=params,
        key=key,
        integration_method="euler_1",
    )
    assert float(new_state.fuel_flow) == pytest.approx(params.fuel_max)

    new_state, _ = compute_next_state(
        fuel_raw=-100.0,
        state=state,
        params=params,
        key=key,
        integration_method="euler_1",
    )
    assert float(new_state.fuel_flow) == pytest.approx(params.fuel_min)


def test_reward_peaks_on_target():
    """Reward is maximal when T_crown matches the target."""
    params = GlassFurnaceParams()
    state_on = _default_state(params, target=1587.0).replace(
        T_crown=1587.0,
        fuel_flow=params.fuel_min,
    )
    state_off = state_on.replace(T_crown=1507.0)

    r_on = float(compute_reward(state_on, params))
    r_off = float(compute_reward(state_off, params))

    # At perfect tracking, reward = 1.0 minus fuel penalty at fuel_min (which is 0).
    assert r_on == pytest.approx(1.0)
    assert r_on > r_off


def test_terminal_on_crown_out_of_bounds():
    """Out-of-range crown temperatures terminate the episode."""
    params = GlassFurnaceParams()

    state_cold = _default_state(params).replace(
        time=10,
        T_crown=params.T_crown_min - 1.0,
    )
    terminated, _ = check_is_terminal(state_cold, params)
    assert bool(terminated) is True

    state_hot = state_cold.replace(T_crown=params.T_crown_max + 1.0)
    terminated, _ = check_is_terminal(state_hot, params)
    assert bool(terminated) is True

    state_ok = state_cold.replace(T_crown=1577.0)
    terminated, _ = check_is_terminal(state_ok, params)
    assert bool(terminated) is False


def test_truncation_on_max_steps():
    """Reaching max_steps_in_episode sets truncated."""
    params = GlassFurnaceParams(max_steps_in_episode=50)
    state = _default_state(params).replace(time=50)
    _, truncated = check_is_terminal(state, params)
    assert bool(truncated) is True


def test_steady_state_energy_balance_sign():
    """
    Sanity check: at operating T_crown, there should exist a fuel flow in range that
    produces dT_crown/dt ~ 0.  At fuel_min it must be decreasing, at fuel_max increasing.
    """
    params = GlassFurnaceParams()
    position = jnp.array([1577.0, 1497.0, 1227.0])

    v_min, _ = compute_velocity(
        position, action=params.fuel_min, m_pull=params.m_pull, params=params
    )
    v_max, _ = compute_velocity(
        position, action=params.fuel_max, m_pull=params.m_pull, params=params
    )

    assert v_min[0] < 0.0  # too little fuel: crown cools
    assert v_max[0] > 0.0  # too much fuel: crown heats
