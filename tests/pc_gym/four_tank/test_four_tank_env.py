import jax.numpy as jnp
import numpy as _np
import pytest

from target_gym.pc_gym.four_tank.env import (
    FourTankParams,
    FourTankState,
    check_is_terminal,
    compute_next_state,
    compute_velocity,
)


@pytest.fixture
def params():
    # Version-1 reward: the reward tests in this file describe the capped
    # log shape; version 2 is covered by tests/test_reward_contract.py.
    return FourTankParams(reward_version=1)


@pytest.fixture
def state():
    return FourTankState(
        time=0,
        h1=0.5,
        h2=0.5,
        h3=0.3,
        h4=0.3,
        target_h1=0.7,
        target_h2=0.7,
        v1=5.0,
        v2=5.0,
    )


# -----------------------------------------------------------------------
# Velocity / ODE structure
# -----------------------------------------------------------------------


def test_gravity_draining_with_no_pumps(params, state):
    """With zero pump input, all non-empty tanks drain under gravity."""
    v, _ = compute_velocity(
        jnp.array([state.h1, state.h2, state.h3, state.h4]),
        action=jnp.array([0.0, 0.0]),
        params=params,
    )
    # Upper tanks (h3, h4) have no pump inflow when v=0, only outflow
    assert v[2] < 0.0  # dh3/dt < 0
    assert v[3] < 0.0  # dh4/dt < 0


def test_pump_v1_fills_lower_tank_1(params):
    """Pump v1 drives inflow to tank 1 (gamma1 fraction) and tank 4 (1-gamma1 fraction)."""
    # With h3=0 there's no cascade inflow from tank 3 to tank 1
    h = jnp.array([0.01, 0.5, 0.0, 0.5])
    v_low, _ = compute_velocity(h, action=jnp.array([1.0, 0.0]), params=params)
    v_high, _ = compute_velocity(h, action=jnp.array([9.0, 0.0]), params=params)
    # Higher v1 -> more inflow to tank 1
    assert v_high[0] > v_low[0]


def test_pump_v2_fills_lower_tank_2(params):
    h = jnp.array([0.5, 0.01, 0.5, 0.0])
    v_low, _ = compute_velocity(h, action=jnp.array([0.0, 1.0]), params=params)
    v_high, _ = compute_velocity(h, action=jnp.array([0.0, 9.0]), params=params)
    assert v_high[1] > v_low[1]


def test_split_ratio_v1_distributes_to_h1_and_h4(params):
    """Pump v1 splits: gamma1 fraction goes to h1, (1-gamma1) to h4.
    With h=0 for all tanks, sqrt(max(h,0))=0 so only pump inflow terms contribute.
    """
    h = jnp.array([0.0, 0.0, 0.0, 0.0])
    v, _ = compute_velocity(h, action=jnp.array([5.0, 0.0]), params=params)

    inflow_h1 = params.gamma1 * params.k1 * 5.0 / params.A1
    inflow_h4 = (1 - params.gamma1) * params.k1 * 5.0 / params.A4

    assert v[0] == pytest.approx(inflow_h1, rel=1e-5)
    assert v[3] == pytest.approx(inflow_h4, rel=1e-5)


def test_split_ratio_v2_distributes_to_h2_and_h3(params):
    """Pump v2 splits: gamma2 fraction goes to h2, (1-gamma2) to h3."""
    h = jnp.array([0.0, 0.0, 0.0, 0.0])
    v, _ = compute_velocity(h, action=jnp.array([0.0, 5.0]), params=params)

    inflow_h2 = params.gamma2 * params.k2 * 5.0 / params.A2
    inflow_h3 = (1 - params.gamma2) * params.k2 * 5.0 / params.A3

    assert v[1] == pytest.approx(inflow_h2, rel=1e-5)
    assert v[2] == pytest.approx(inflow_h3, rel=1e-5)


def test_outflow_proportional_to_sqrt_level(params):
    """Tank outflow rate scales with sqrt(h) (Torricelli's law)."""
    # 4x level -> 2x outflow (sqrt(4h) = 2*sqrt(h))
    h_base = 0.25
    h_quad = 1.0  # 4x

    v_base, _ = compute_velocity(
        jnp.array([h_base, h_base, 0.0, 0.0]),
        action=jnp.array([0.0, 0.0]),
        params=params,
    )
    v_quad, _ = compute_velocity(
        jnp.array([h_quad, h_quad, 0.0, 0.0]),
        action=jnp.array([0.0, 0.0]),
        params=params,
    )
    # Outflow term: -(a/A)*sqrt(2g)*sqrt(h); so ratio should be sqrt(4) = 2
    assert float(abs(v_quad[0])) == pytest.approx(2 * float(abs(v_base[0])), rel=1e-4)


def test_no_outflow_at_zero_level(params):
    """sqrt(max(h, 0)) ensures zero outflow when tank is empty."""
    h = jnp.array([0.0, 0.0, 0.0, 0.0])
    v, _ = compute_velocity(h, action=jnp.array([0.0, 0.0]), params=params)
    # With h=0 and v=0, all velocities should be 0
    assert jnp.allclose(v, 0.0, atol=1e-10)


# -----------------------------------------------------------------------
# Integration
# -----------------------------------------------------------------------


def test_integration_methods_agree_for_small_dt(state):
    params = FourTankParams(delta_t=0.01)
    action = jnp.array([0.3, 0.3])
    s_euler, _ = compute_next_state(action, state, params, integration_method="euler_1")
    s_rk4, _ = compute_next_state(action, state, params, integration_method="rk4_1")
    assert jnp.allclose(s_euler.h1, s_rk4.h1, atol=1e-4)
    assert jnp.allclose(s_euler.h2, s_rk4.h2, atol=1e-4)


# -----------------------------------------------------------------------
# Action scaling
# -----------------------------------------------------------------------


def test_action_scaled_to_pump_range(params, state):
    action_min = jnp.array([-1.0, -1.0])
    action_max = jnp.array([1.0, 1.0])
    s_min, _ = compute_next_state(
        action_min, state, params, integration_method="euler_1"
    )
    s_max, _ = compute_next_state(
        action_max, state, params, integration_method="euler_1"
    )
    assert s_min.v1 == pytest.approx(params.v_min)
    assert s_min.v2 == pytest.approx(params.v_min)
    assert s_max.v1 == pytest.approx(params.v_max)
    assert s_max.v2 == pytest.approx(params.v_max)


# -----------------------------------------------------------------------
# Terminal / Reward
# -----------------------------------------------------------------------


def test_not_terminal_in_bounds(params, state):
    term, trunc = check_is_terminal(state, params)
    assert not term
    assert not trunc


def test_a_full_or_empty_tank_does_not_trip(params, state):
    """No overflow trip or interlock was found on the rig, and an empty tank
    is its physical floor (REALITY.md, Limits [S2, p. 121; S4, p. 15])."""
    for h in (0.0, params.h_max):
        term, _ = check_is_terminal(state.replace(h1=h, h3=h), params)
        assert not term


# ---------------------------------------------------------------------------
# Reachability and loop pairing
#
# Added after the target range was found to sit entirely above what the plant
# can produce: with both pumps saturated the steady state tops out at
# h1 = 0.360, h2 = 0.429, while targets were sampled from (0.5, 1.0). Every
# episode was unwinnable, and the shared effectiveness contract could not see
# it because the PID still beat every constant action -- both simply sat far
# from setpoint. These tests make that class of defect impossible to reintroduce.
# ---------------------------------------------------------------------------


def _steady_levels(v1, v2, p):
    """Steady state of the four tanks at constant pump voltages."""
    g = p.g
    h3 = ((1 - p.gamma2) * p.k2 * v2 / p.a3) ** 2 / (2 * g)
    h4 = ((1 - p.gamma1) * p.k1 * v1 / p.a4) ** 2 / (2 * g)
    h1 = ((p.gamma1 * p.k1 * v1 + p.a3 * _np.sqrt(2 * g * h3)) / p.a1) ** 2 / (2 * g)
    h2 = ((p.gamma2 * p.k2 * v2 + p.a4 * _np.sqrt(2 * g * h4)) / p.a2) ** 2 / (2 * g)
    return h1, h2


def test_relative_gain_array_demands_the_cross_pairing():
    """gamma1 + gamma2 < 1 puts the plant in the non-minimum-phase regime.

    Johansson (2000) gives lambda11 = g1 g2 / (g1 + g2 - 1). A negative value
    means closing one diagonal loop reverses the sign of the other, so integral
    action on the diagonal pairing is unstable -- which is why the shipped PID
    drives v1 from h2 and v2 from h1.
    """
    p = FourTankParams(reward_version=1)
    assert p.gamma1 + p.gamma2 < 1.0, "expected the non-minimum-phase configuration"
    lam = p.gamma1 * p.gamma2 / (p.gamma1 + p.gamma2 - 1.0)
    assert lam < 0.0

    # The size of the cross gains against the rigs' is in
    # test_four_tank_reality.py.


def _voltages_for(t1, t2, p, load=(0.0, 0.0)):
    """Pump voltages that hold a target pair at steady state under a load."""
    A = _np.array(
        [
            [p.gamma1 * p.k1, (1 - p.gamma2) * p.k2],
            [(1 - p.gamma1) * p.k1, p.gamma2 * p.k2],
        ]
    )
    b = _np.array(
        [
            p.a1 * _np.sqrt(2 * p.g * t1) - load[0],
            p.a2 * _np.sqrt(2 * p.g * t2) - load[1],
        ]
    )
    return _np.linalg.solve(A, b)
