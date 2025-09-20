import math

import jax.numpy as jnp
import pytest

from target_gym.pc_gym.cstr.env import (
    CSTRParams,
    CSTRState,
    compute_next_state,
    compute_velocity,
)
from target_gym.utils import convert_raw_action_to_range


def test_compute_velocity_shape_and_trends():
    """Check velocity vector has correct shape and expected qualitative trends."""
    params = CSTRParams(delta_t=0.1)
    C_a = 1.0
    T = 350.0
    position = jnp.array([C_a, T])
    # Compare cooling effect: lowering T_c should reduce dT/dt
    v_high, _ = compute_velocity(
        position, action=302.0, params=params
    )  # warmer coolant
    v_low, _ = compute_velocity(position, action=295.0, params=params)  # colder coolant

    # Shape
    assert v_high.shape == (2,)

    # C_a should decrease due to reaction + inflow
    assert v_high[0] < 0.0

    # Colder coolant should lead to lower dT/dt
    assert v_low[1] < v_high[1]

    # In the trivial case with no reactant, heat release vanishes -> cooling dominates
    v_no_reaction, _ = compute_velocity((0.0, T), action=295.0, params=params)
    assert v_no_reaction[1] < 0.0


@pytest.mark.parametrize("method", ["euler_1", "rk2_1", "rk4_1"])
def test_compute_next_state_progression(method):
    """State should evolve and time should increment."""
    params = CSTRParams(delta_t=0.1)
    state = CSTRState(C_a=1.0, T=350.0, T_c=298.0, t=0, target_CA=0.85)

    new_state, metrics = compute_next_state(
        T_c_raw=0.5,
        state=state,
        params=params,
        integration_method=method,
    )

    # State updates
    assert isinstance(new_state, CSTRState)
    assert new_state.t == state.t + 1
    assert not jnp.allclose(new_state.C_a, state.C_a)
    assert not jnp.allclose(new_state.T, state.T)

    # T_c is clamped to [min, max]
    assert (
        params.T_c_min
        <= convert_raw_action_to_range(new_state.T_c, params.T_c_min, params.T_c_max)
        <= params.T_c_max
    )

    # metrics is None (first-order dynamics mode)
    assert metrics is None


def test_integration_methods_agree_for_small_dt():
    """RK4 and Euler should give similar results if dt is small."""
    params = CSTRParams(delta_t=1e-4)
    state = CSTRState(C_a=1.0, T=350.0, T_c=298.0, t=0, target_CA=0.85)

    action = 0.3
    new_state_euler, _ = compute_next_state(
        T_c_raw=action, state=state, params=params, integration_method="euler_1"
    )
    new_state_rk4, _ = compute_next_state(
        T_c_raw=action, state=state, params=params, integration_method="rk4_1"
    )

    assert jnp.allclose(new_state_euler.C_a, new_state_rk4.C_a, atol=1e-5)
    assert jnp.allclose(new_state_euler.T, new_state_rk4.T, atol=1e-5)


def test_action_clipping():
    """Raw action should be scaled and then clamped between [T_c_min, T_c_max]."""
    params = CSTRParams(delta_t=0.1)
    state = CSTRState(C_a=1.0, T=350.0, T_c=298.0, t=0, target_CA=0.85)

    # Very large raw input -> clipped to max
    action = 100.0
    new_state, _ = compute_next_state(
        T_c_raw=action,
        state=state,
        params=params,
        integration_method="euler_1",
    )
    assert (
        convert_raw_action_to_range(new_state.T_c, params.T_c_min, params.T_c_max)
        == params.T_c_max
    )

    # Very negative raw input -> clipped to min
    new_state, _ = compute_next_state(
        T_c_raw=-action, state=state, params=params, integration_method="euler_1"
    )
    assert (
        convert_raw_action_to_range(new_state.T_c, params.T_c_min, params.T_c_max)
        == params.T_c_min
    )
