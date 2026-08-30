"""Physics validation for the regenerative glass furnace.

Enforces ``src/target_gym/glass_furnace/PHYSICS.md``. Assertions are emergent
figures of merit checked against published float-furnace operating data --
specific energy consumption, regenerator effectiveness, air preheat, stack
temperature, glass residence time -- plus structural invariants that need no
source (energy balance signs, monotonicity, temperature ordering).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.glass_furnace.env import (
    N_REGEN_NODES,
    N_SETPOINTS,
    GlassFurnaceParams,
    GlassFurnaceState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_velocity,
    glass_c_p,
    reversal_phase,
    solve_T_gas,
    specific_energy_consumption,
)
from target_gym.glass_furnace.env_jax import GlassFurnace

# Published float-furnace validation targets (PHYSICS.md §2).
SEC_RANGE_GJ_PER_T = (4.0, 6.5)
CROWN_RANGE_C = (1540.0, 1630.0)
AIR_PREHEAT_RANGE_C = (1000.0, 1400.0)
STACK_RANGE_C = (400.0, 660.0)
REGEN_EFFECTIVENESS_RANGE = (0.65, 0.85)
RESIDENCE_RANGE_H = (24.0, 32.0)


@pytest.fixture(scope="module")
def params():
    return GlassFurnaceParams()


def _state(params, **overrides):
    env = GlassFurnace()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    return state.replace(**overrides) if overrides else state


def _run_to_steady(fuel_raw, params=None, hours=140.0):
    """Hold a constant fuel rate until the furnace settles."""
    env = GlassFurnace()
    p = params or GlassFurnaceParams(
        max_steps_in_episode=int(hours * 3600 / 30), m_pull_noise_std=0.0
    )
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    action = jnp.array([fuel_raw])
    terminated = False
    for _ in range(p.max_steps_in_episode):
        _, state, _, terminated, _ = env.step_env(key, state, action, p)
        if bool(terminated):
            break
    return state, bool(terminated), p


# ---------------------------------------------------------------------------
# 1. Structural invariants (first principles)
# ---------------------------------------------------------------------------


def test_velocity_vector_has_one_entry_per_state(params):
    state = _state(params)
    position = jnp.concatenate(
        [
            jnp.array([state.T_crown, state.T_melt, state.T_work, state.m_batch]),
            state.T_rA,
            state.T_rB,
        ]
    )
    v, _ = compute_velocity(
        position,
        action=0.6,
        m_pull=params.m_pull,
        phase=0.0,
        T_gas=1650.0,
        params=params,
    )
    assert v.shape == (4 + 2 * N_REGEN_NODES,)
    assert np.all(np.isfinite(np.asarray(v)))


def test_more_fuel_gives_a_hotter_flame(params):
    """The quasi-steady flame temperature must rise with firing rate."""
    kwargs = dict(
        T_crown=1580.0, T_melt=1520.0, T_work=1300.0, T_air=1200.0, melt_open=0.8
    )
    temps = [
        float(solve_T_gas(1650.0, m_fuel=f, params=params, **kwargs))
        for f in (0.45, 0.60, 0.75)
    ]
    assert temps[0] < temps[1] < temps[2]


def test_flame_is_hotter_than_every_surface_it_heats(params):
    """Heat flows flame -> crown -> glass, so the flame must be the hottest."""
    state, terminated, _ = _run_to_steady(-0.2)
    assert not terminated
    assert state.T_gas > state.T_crown > state.T_melt > state.T_work


def test_working_end_is_cooler_than_the_melt(params):
    """Glass flows melt -> working end and is conditioned down for forming.

    A working end hotter than the zone feeding it is thermodynamically
    backwards; an earlier revision had exactly that, because the conditioning
    heat extraction was missing.
    """
    for fuel_raw in (-0.5, -0.2):
        state, terminated, _ = _run_to_steady(fuel_raw)
        if terminated:
            continue
        assert state.T_work < state.T_melt


def test_preheated_air_is_cooler_than_the_exhaust_that_heats_it(params):
    """Second law: the regenerator cannot preheat air above the flue gas."""
    state, terminated, _ = _run_to_steady(-0.2)
    assert not terminated
    assert state.T_air_preheat < state.T_gas
    assert state.T_stack > params.T_ambient


def test_glass_specific_heat_increases_with_temperature(params):
    temps = [float(glass_c_p(T, params)) for T in (400.0, 900.0, 1400.0)]
    assert temps[0] < temps[1] < temps[2]
    assert 950.0 < temps[0] < 1150.0
    assert 1300.0 < temps[2] < 1500.0


def test_batch_blanket_shields_the_melt(params):
    """More unmelted batch must reduce the heat reaching the glass."""
    base = _state(params)

    def melt_rate(m_batch):
        position = jnp.concatenate(
            [
                jnp.array([base.T_crown, base.T_melt, base.T_work, m_batch]),
                base.T_rA,
                base.T_rB,
            ]
        )
        v, _ = compute_velocity(
            position,
            action=0.6,
            m_pull=params.m_pull,
            phase=0.0,
            T_gas=1650.0,
            params=params,
        )
        return float(v[1])  # dT_melt/dt

    assert melt_rate(0.0) > melt_rate(params.m_batch_full)


def test_reversal_alternates_between_chambers(params):
    """The two regenerators must swap roles every reversal_period."""
    steps_per_reversal = params.reversal_period / params.delta_t
    phases = [
        float(reversal_phase(int(t), params))
        for t in (0, steps_per_reversal * 0.5, steps_per_reversal * 1.5)
    ]
    assert phases[0] == 0.0
    assert phases[1] == 0.0
    assert phases[2] == 1.0


# ---------------------------------------------------------------------------
# 2. Validation against published float-furnace data (PHYSICS.md §2)
# ---------------------------------------------------------------------------


def test_glass_residence_time_matches_a_float_line():
    """C_melt / (m_pull * c_p) should give the documented 24-32 h."""
    p = GlassFurnaceParams()
    residence_h = p.C_melt / (p.m_pull * glass_c_p(1500.0, p)) / 3600.0
    assert RESIDENCE_RANGE_H[0] < residence_h < RESIDENCE_RANGE_H[1]


def test_sustained_operation_uses_a_realistic_amount_of_fuel():
    """Holding a realistic crown temperature must cost 4-6.5 GJ/tonne.

    The headline efficiency figure for a float furnace, and the single
    strongest check on the regenerator: without heat recovery the same crown
    temperature costs roughly twice as much fuel.
    """
    state, terminated, p = _run_to_steady(-0.2)
    assert not terminated, "furnace could not hold a steady state at mid throttle"
    assert CROWN_RANGE_C[0] < float(state.T_crown) < CROWN_RANGE_C[1]
    sec = float(specific_energy_consumption(state, p))
    assert SEC_RANGE_GJ_PER_T[0] < sec < SEC_RANGE_GJ_PER_T[1], f"{sec:.2f} GJ/t"


def test_regenerator_reaches_realistic_effectiveness():
    """Temperature effectiveness (T_air - T_amb)/(T_gas - T_amb) ~ 0.65-0.85."""
    state, terminated, p = _run_to_steady(-0.2)
    assert not terminated
    eff = (float(state.T_air_preheat) - p.T_ambient) / (
        float(state.T_gas) - p.T_ambient
    )
    assert (
        REGEN_EFFECTIVENESS_RANGE[0] < eff < REGEN_EFFECTIVENESS_RANGE[1]
    ), f"{eff:.2f}"


def test_air_preheat_and_stack_temperatures_are_realistic():
    state, terminated, _ = _run_to_steady(-0.2)
    assert not terminated
    assert AIR_PREHEAT_RANGE_C[0] < float(state.T_air_preheat) < AIR_PREHEAT_RANGE_C[1]
    assert STACK_RANGE_C[0] < float(state.T_stack) < STACK_RANGE_C[1]


def test_regenerator_holds_a_hot_to_cold_gradient():
    """A checker stack must be hottest at the flue end and coolest at the stack.

    A single lumped node cannot deliver hot preheated air *and* a cool stack;
    the gradient is what makes both possible, which is why the checker is
    modelled as several nodes in series.
    """
    state, terminated, _ = _run_to_steady(-0.2)
    assert not terminated
    profile = np.asarray(state.T_rA)
    assert profile[0] > profile[-1], f"no gradient: {profile}"


# ---------------------------------------------------------------------------
# 3. Operating envelope and control interface
# ---------------------------------------------------------------------------


def test_underfiring_and_overfiring_both_terminate():
    """Both extremes of the action range must be genuinely unsurvivable.

    If either bound were sustainable, half the action range would be free and
    the task would lose its failure modes.
    """
    _, cold_terminated, _ = _run_to_steady(-1.0)
    _, hot_terminated, _ = _run_to_steady(1.0)
    assert cold_terminated, "minimum fuel did not cool the furnace past its limit"
    assert hot_terminated, "maximum fuel did not overheat the furnace"


@pytest.mark.parametrize("method", ["rk2_2", "rk4_2", "rk4_4"])
def test_state_advances_under_each_integrator(method, params):
    state = _state(params)
    new_state, _ = compute_next_state(
        fuel_raw=0.0,
        state=state,
        params=params,
        key=jax.random.PRNGKey(0),
        integration_method=method,
    )
    assert new_state.time == state.time + 1
    assert params.fuel_min <= float(new_state.fuel_flow) <= params.fuel_max
    assert new_state.T_rA.shape == (N_REGEN_NODES,)
    for leaf in jax.tree_util.tree_leaves(new_state):
        assert np.all(np.isfinite(np.asarray(leaf)))


def test_action_is_clipped_to_the_fuel_range(params):
    state = _state(params)
    for raw, expected in ((-5.0, params.fuel_min), (5.0, params.fuel_max)):
        new_state, _ = compute_next_state(
            fuel_raw=raw, state=state, params=params, key=jax.random.PRNGKey(0)
        )
        assert float(new_state.fuel_flow) == pytest.approx(expected, rel=1e-5)


def test_reward_peaks_on_target(params):
    state = _state(
        params, target_T_crown=1580.0, T_crown=1580.0, fuel_flow=params.fuel_min
    )
    on_target = float(compute_reward(state, params))
    off_target = float(compute_reward(state.replace(T_crown=1500.0), params))
    assert on_target > off_target
    assert on_target == pytest.approx(1.0, abs=1e-5)


def test_fuel_costs_reward(params):
    """Burning more fuel must reduce reward at equal tracking."""
    lean = _state(
        params, target_T_crown=1580.0, T_crown=1580.0, fuel_flow=params.fuel_min
    )
    rich = lean.replace(fuel_flow=params.fuel_max)
    assert float(compute_reward(lean, params)) > float(compute_reward(rich, params))


def test_terminal_on_crown_out_of_bounds(params):
    state = _state(params)
    assert bool(
        check_is_terminal(state.replace(T_crown=params.T_crown_max + 1.0), params)[0]
    )
    assert bool(
        check_is_terminal(state.replace(T_crown=params.T_crown_min - 1.0), params)[0]
    )


def test_truncation_on_max_steps():
    p = GlassFurnaceParams(max_steps_in_episode=50)
    env = GlassFurnace()
    _, state = env.reset_env(jax.random.PRNGKey(0), p)
    assert bool(check_is_terminal(state.replace(time=50), p)[1])


def test_setpoint_schedule_has_the_declared_length(params):
    state = _state(params)
    assert state.target_schedule.shape == (N_SETPOINTS,)
    assert isinstance(state, GlassFurnaceState)
