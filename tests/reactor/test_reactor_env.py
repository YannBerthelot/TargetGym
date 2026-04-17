import jax.numpy as jnp
import pytest

import jax

from target_gym.reactor.env import (
    BETA_I,
    BETA_TOT,
    LAMBDA_I,
    N_GROUPS,
    N_SETPOINTS,
    ReactorParams,
    ReactorState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
    steady_state_precursors,
    steady_state_xenon,
)


def _default_state(
    params: ReactorParams, target: float = 1.0, n: float = 1.0
) -> ReactorState:
    I_hat, Xe_hat = steady_state_xenon(n, params)
    return ReactorState(
        time=0,
        n=n,
        C=steady_state_precursors(n, params),
        T_fuel=params.initial_T_fuel,
        T_coolant=params.initial_T_coolant,
        I_hat=I_hat,
        Xe_hat=Xe_hat,
        target_n=target,
        target_schedule=jnp.full((N_SETPOINTS,), target),
        demand_key=jax.random.PRNGKey(42),
        rho_ext=jnp.zeros(()),
    )


def _position_from_state(state: ReactorState) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.array([state.n]),
            state.C,
            jnp.array([state.T_fuel, state.T_coolant, state.I_hat, state.Xe_hat]),
        ]
    )


def test_beta_i_matches_beta_tot():
    """Sanity: sum of partial delayed neutron fractions = beta_tot."""
    assert float(BETA_I.sum()) == pytest.approx(BETA_TOT, rel=1e-6)
    assert BETA_I.shape == (N_GROUPS,)
    assert LAMBDA_I.shape == (N_GROUPS,)


def test_compute_velocity_shape():
    params = ReactorParams()
    state = _default_state(params)
    position = _position_from_state(state)
    v, _ = compute_velocity(position, action=0.0, params=params)
    # 1 (n) + 6 (precursors) + 2 (T_fuel, T_coolant) + 2 (I_hat, Xe_hat)
    assert v.shape == (11,)


def test_steady_state_neutron_balance_at_zero_reactivity():
    """
    At rho_ext=0 and zero thermal feedback (T at reference), dn/dt should be
    (approximately) zero when precursors are at their steady-state values.
    """
    params = ReactorParams(
        initial_T_fuel=900.0,
        initial_T_coolant=580.0,  # match T_*_ref exactly
        T_fuel_ref=900.0,
        T_coolant_ref=580.0,
    )
    state = _default_state(params, n=1.0)
    position = _position_from_state(state)
    v, _ = compute_velocity(position, action=0.0, params=params)
    # dn/dt should be effectively zero: precursor source exactly balances prompt loss.
    assert abs(float(v[0])) < 1e-4
    # Precursor derivatives are also zero at steady state.
    assert jnp.allclose(v[1:7], 0.0, atol=1e-6)


def test_positive_reactivity_increases_power():
    """Withdrawing rods (positive rho_ext) must make dn/dt > 0."""
    params = ReactorParams(
        initial_T_fuel=900.0,
        T_fuel_ref=900.0,
        initial_T_coolant=580.0,
        T_coolant_ref=580.0,
    )
    state = _default_state(params, n=1.0)
    position = _position_from_state(state)

    # Positive rod reactivity (within prompt-safe range)
    v_pos, _ = compute_velocity(
        position, action=0.5 * params.rho_ext_max, params=params
    )
    v_neg, _ = compute_velocity(
        position, action=0.5 * params.rho_ext_min, params=params
    )

    assert float(v_pos[0]) > 0.0
    assert float(v_neg[0]) < 0.0


def test_doppler_feedback_is_stabilising():
    """
    Raising fuel temperature above T_fuel_ref must *decrease* dn/dt
    (alpha_fuel is negative — this is what makes the reactor passively safe).
    """
    params = ReactorParams(
        initial_T_fuel=900.0,
        T_fuel_ref=900.0,
        initial_T_coolant=580.0,
        T_coolant_ref=580.0,
    )
    state_cold = _default_state(params, n=1.0)
    state_hot = state_cold.replace(T_fuel=1100.0)  # +200 K over reference

    v_cold, _ = compute_velocity(
        _position_from_state(state_cold), action=0.0, params=params
    )
    v_hot, _ = compute_velocity(
        _position_from_state(state_hot), action=0.0, params=params
    )

    assert float(v_hot[0]) < float(v_cold[0])


def test_power_increases_fuel_temperature():
    """At nominal power with initial cold fuel, fuel should heat up."""
    params = ReactorParams()
    # Start with fuel/coolant at the same temperature (no exchange) — at n=1,
    # P_ref heats the fuel and dT_fuel/dt must be positive.
    state = _default_state(params, n=1.0).replace(T_fuel=600.0, T_coolant=600.0)
    v, _ = compute_velocity(_position_from_state(state), action=0.0, params=params)
    # Index 7 = T_fuel derivative.
    assert float(v[7]) > 0.0


@pytest.mark.parametrize("method", ["euler_50", "rk2_50", "rk4_50"])
def test_compute_next_state_progression(method):
    """State evolves and time increments."""
    params = ReactorParams()
    state = _default_state(params)

    new_state, metrics = compute_next_state(
        rho_raw=0.3,
        state=state,
        params=params,
        integration_method=method,
    )
    assert isinstance(new_state, ReactorState)
    assert new_state.time == state.time + 1
    # rho_ext is the converted action, clamped to [rho_ext_min, rho_ext_max]
    assert params.rho_ext_min <= float(new_state.rho_ext) <= params.rho_ext_max
    # Precursors have the right shape
    assert new_state.C.shape == (N_GROUPS,)
    assert metrics is None


def test_action_rate_limiting():
    """Rod movement is rate-limited: one step cannot reach full travel."""
    params = ReactorParams()
    state = _default_state(params)  # starts with rho_ext=0

    # One step toward max withdrawal: should move by rod_speed_withdraw * delta_t
    new_state, _ = compute_next_state(rho_raw=1.0, state=state, params=params)
    expected_step = params.rod_speed_withdraw * params.delta_t
    assert float(new_state.rho_ext) == pytest.approx(expected_step, rel=1e-4)
    assert float(new_state.rho_ext) < params.rho_ext_max  # not at max yet

    # One step toward max insertion: should move by rod_speed_insert * delta_t
    new_state, _ = compute_next_state(rho_raw=-1.0, state=state, params=params)
    expected_step = -params.rod_speed_insert * params.delta_t
    assert float(new_state.rho_ext) == pytest.approx(expected_step, rel=1e-4)

    # After many steps, should reach the limit
    st = state
    for _ in range(200):
        st, _ = compute_next_state(rho_raw=1.0, state=st, params=params)
    assert float(st.rho_ext) == pytest.approx(params.rho_ext_max, rel=1e-4)


def test_reward_peaks_on_target():
    """Reward is maximal when n matches target_n (and zero rod reactivity)."""
    params = ReactorParams()
    state_on = _default_state(params, target=1.0, n=1.0)
    state_off = state_on.replace(n=0.3)

    r_on = float(compute_reward(state_on, params))
    r_off = float(compute_reward(state_off, params))

    # At perfect tracking with rho_ext=0, reward = 1.0 - 0 = 1.0
    assert r_on == pytest.approx(1.0)
    assert r_on > r_off


def test_rod_motion_penalty_applied():
    """Non-zero rod reactivity reduces the reward even at perfect tracking."""
    params = ReactorParams()
    state = _default_state(params, target=1.0, n=1.0)
    state_with_rods = state.replace(rho_ext=jnp.asarray(params.rho_ext_max))
    r_neutral = float(compute_reward(state, params))
    r_rodded = float(compute_reward(state_with_rods, params))
    assert r_neutral > r_rodded
    # Bound: the penalty cannot exceed rod_motion_weight.
    assert r_neutral - r_rodded <= params.rod_motion_weight + 1e-6


def test_terminal_on_overpower():
    """n > n_max terminates the episode (SCRAM)."""
    params = ReactorParams()
    state = _default_state(params).replace(time=10, n=params.n_max + 0.1)
    terminated, _ = check_is_terminal(state, params)
    assert bool(terminated) is True


def test_terminal_on_fuel_melting():
    """T_fuel > T_fuel_max terminates the episode."""
    params = ReactorParams()
    state = _default_state(params).replace(time=10, T_fuel=params.T_fuel_max + 50.0)
    terminated, _ = check_is_terminal(state, params)
    assert bool(terminated) is True


def test_truncation_on_max_steps():
    """Reaching max_steps_in_episode sets truncated."""
    params = ReactorParams(max_steps_in_episode=50)
    state = _default_state(params).replace(time=50)
    _, truncated = check_is_terminal(state, params)
    assert bool(truncated) is True


def test_steady_state_precursors_formula():
    """Steady-state precursor concentrations satisfy the textbook formula."""
    params = ReactorParams()
    C_ss = steady_state_precursors(1.0, params)
    # At steady state and zero net reactivity:
    #   dC_i/dt = 0  =>  C_i = (beta_i / (lambda_i * Lambda)) * n
    expected = BETA_I / (LAMBDA_I * params.Lambda_gen)
    assert jnp.allclose(C_ss, expected)


def test_settles_toward_lower_power_when_rods_inserted():
    """
    Running many steps with deep rod insertion should drive power downward.
    This is an end-to-end sanity check on the coupled dynamics.
    """
    # Zero rod-motion weight so we only see the physics, not the shaping.
    params = ReactorParams(rod_motion_weight=0.0, max_steps_in_episode=400)
    state = _default_state(params, target=1.0, n=1.0)

    # Insert rods hard (-1 raw → rho_ext_min).
    for _ in range(20):
        state, _ = compute_next_state(rho_raw=-1.0, state=state, params=params)

    assert float(state.n) < 1.0
