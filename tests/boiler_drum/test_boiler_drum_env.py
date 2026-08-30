"""Physics validation for the boiler drum.

Enforces ``src/target_gym/boiler_drum/PHYSICS.md``.

The behavioural half of this file is the important half. Shrink and swell are
the reason this environment exists, and they are *emergent* -- nothing in the
model states "the level goes up when steam demand rises". It falls out of
tracking steam as mass and letting a falling pressure expand it. A refactor
that quietly reverted to a quasi-steady quality formulation would leave every
coefficient correct and produce no inverse response at all, which is what
these tests are here to catch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.boiler_drum.env import (
    BoilerDrumParams,
    BoilerDrumState,
    check_is_terminal,
    circulation_flow,
    circulation_ratio,
    compute_reward,
    compute_velocity,
    drum_level,
    latent_heat,
    saturation_temperature,
    steady_state,
    steam_density,
    steam_enthalpy,
    void_fraction,
    water_density,
    water_enthalpy,
)
from target_gym.boiler_drum.env_jax import BoilerDrum

# IAPWS saturation anchors: p [bar], t_s [C], rho_w, rho_s, h_w [kJ/kg], h_s [kJ/kg]
IAPWS = [
    (60.0, 275.6, 758.0, 30.8, 1213.7, 2784.3),
    (70.0, 285.8, 739.7, 36.5, 1267.4, 2772.6),
    (80.0, 295.0, 720.9, 42.5, 1317.1, 2758.6),
    (85.0, 299.3, 711.5, 45.3, 1341.0, 2751.0),
    (90.0, 303.3, 705.2, 48.8, 1363.7, 2742.9),
    (100.0, 311.0, 688.4, 55.5, 1408.0, 2724.7),
    (110.0, 318.1, 671.6, 62.6, 1450.6, 2705.4),
]


@pytest.fixture(scope="module")
def params():
    return BoilerDrumParams()


# ---------------------------------------------------------------------------
# 1. Steam properties (PHYSICS.md section 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p_bar,t_s,rho_w,rho_s,h_w,h_s", IAPWS)
def test_property_fits_match_steam_tables(params, p_bar, t_s, rho_w, rho_s, h_w, h_s):
    """Quadratic fits must stay within 1 % of IAPWS across 60-110 bar."""
    assert float(saturation_temperature(p_bar, params)) == pytest.approx(t_s, rel=0.01)
    assert float(water_density(p_bar, params)) == pytest.approx(rho_w, rel=0.01)
    assert float(steam_density(p_bar, params)) == pytest.approx(rho_s, rel=0.01)
    assert float(water_enthalpy(p_bar, params)) == pytest.approx(h_w * 1e3, rel=0.01)
    assert float(steam_enthalpy(p_bar, params)) == pytest.approx(h_s * 1e3, rel=0.01)


def test_latent_heat_at_nominal_pressure(params):
    assert float(latent_heat(85.0, params)) == pytest.approx(1410e3, rel=0.01)


def test_latent_heat_falls_as_pressure_rises(params):
    """Water and steam approach each other toward the critical point."""
    heats = [float(latent_heat(p, params)) for p in (60.0, 85.0, 110.0)]
    assert heats[0] > heats[1] > heats[2]


def test_property_monotonicity(params):
    """Saturation temperature and steam density rise with pressure; water
    density falls."""
    ps = np.linspace(60.0, 110.0, 100)
    ts = np.array([float(saturation_temperature(p, params)) for p in ps])
    rw = np.array([float(water_density(p, params)) for p in ps])
    rs = np.array([float(steam_density(p, params)) for p in ps])
    assert np.all(np.diff(ts) > 0)
    assert np.all(np.diff(rs) > 0)
    assert np.all(np.diff(rw) < 0)


def test_water_is_far_denser_than_steam(params):
    """The density ratio is what drives natural circulation at all."""
    assert float(water_density(85.0, params)) / float(steam_density(85.0, params)) > 10


# ---------------------------------------------------------------------------
# 2. Operating point and circulation
# ---------------------------------------------------------------------------


def test_steady_state_actually_balances(params):
    """Every derivative must vanish at the solved operating point."""
    V_wt, m_sr, m_sd, _, Q = steady_state(params)
    v, _ = compute_velocity(
        jnp.array([params.p_nominal, V_wt, m_sr, m_sd]),
        (Q, params.q_steam_nominal),
        params.q_steam_nominal,
        params,
    )
    assert np.abs(np.asarray(v)).max() < 1e-4


def test_nominal_firing_is_self_consistent(params):
    """Q must be exactly the heat needed to raise feedwater to saturated steam."""
    _, _, _, _, Q = steady_state(params)
    expected = params.q_steam_nominal * (
        float(steam_enthalpy(params.p_nominal, params)) - params.h_feedwater
    )
    assert float(Q) == pytest.approx(expected, rel=1e-6)
    assert float(Q) == pytest.approx(params.Q_nominal, rel=0.02)


def test_circulation_ratio_is_in_the_natural_circulation_band(params):
    """5-15 for a natural-circulation drum boiler. This is what sets k_friction."""
    assert 5.0 < float(circulation_ratio(params)) < 15.0


def test_circulation_ratio_falls_with_load(params):
    """More steam per unit of circulation as the boiler is pushed harder."""
    ratios = [float(circulation_ratio(params, q_steam=q)) for q in (70.0, 93.35, 110.0)]
    assert ratios[0] > ratios[1] > ratios[2]
    assert all(5.0 < r < 15.0 for r in ratios)


def test_riser_void_fraction_is_physical(params):
    _, m_sr, _, _, _ = steady_state(params)
    alpha_v = float(void_fraction(m_sr, params.p_nominal, params))
    assert 0.3 < alpha_v < 0.6


def test_drum_inventory_closes_to_the_drum_volume(params):
    """Water + bubbles + steam space must add to the drum volume.

    Nothing in the model is fitted to this: the drum volume is specified
    independently of the inventory the steady state solves for.
    """
    V_wt, m_sr, m_sd, _, _ = steady_state(params)
    rho_s = float(steam_density(params.p_nominal, params))
    alpha_v = float(void_fraction(m_sr, params.p_nominal, params))
    V_water = float(V_wt) - params.V_dc - (1.0 - alpha_v) * params.V_r
    V_bubbles = float(m_sd) / rho_s
    V_space = (rho_s * (params.V_t - float(V_wt)) - float(m_sr) - float(m_sd)) / rho_s
    assert V_water + V_bubbles + V_space == pytest.approx(params.V_d, rel=0.02)


def test_geometry_closes(params):
    assert params.V_d + params.V_r + params.V_dc == pytest.approx(params.V_t)


def test_more_voidage_drives_more_circulation(params):
    """A lighter riser column means a bigger density head."""
    flows = [
        float(circulation_flow(a, params.p_nominal, params)) for a in (0.2, 0.4, 0.6)
    ]
    assert flows[0] < flows[1] < flows[2]


def test_metal_dominates_the_pressure_inertia(params):
    """Over half the boiler's energy storage is in the steel, not the water.

    This is the standard result for drum boilers and it is why pressure is so
    tightly coupled to firing rate.
    """
    p_bar = params.p_nominal
    eps = 1e-3
    d = lambda f: (float(f(p_bar + eps, params)) - float(f(p_bar - eps, params))) / (
        2 * eps * 1e5
    )
    V_wt = params.V_wt_nominal
    V_st = params.V_t - V_wt
    rho_w, rho_s = float(water_density(p_bar, params)), float(
        steam_density(p_bar, params)
    )
    h_w, h_s = float(water_enthalpy(p_bar, params)), float(
        steam_enthalpy(p_bar, params)
    )
    metal = params.m_metal * params.C_metal * d(saturation_temperature)
    total = (
        V_st * (rho_s * d(steam_enthalpy) + h_s * d(steam_density))
        + V_wt * (rho_w * d(water_enthalpy) + h_w * d(water_density))
        + metal
        - params.V_t
    )
    assert 0.4 < metal / total < 0.8


# ---------------------------------------------------------------------------
# 3. Shrink and swell -- the behaviour this environment exists for
# ---------------------------------------------------------------------------


def _open_loop(params, Q, q_feed, q_steam, seconds=300.0, dt=0.5):
    """Integrate the bare dynamics at a fixed operating condition.

    Deliberately bypasses ``step_env``: the environment's Ornstein-Uhlenbeck
    steam demand would pull the load back toward nominal during the step and
    corrupt the response being measured.
    """
    V_wt, m_sr, m_sd, level_ref, _ = steady_state(params)
    y = jnp.array([params.p_nominal, V_wt, m_sr, m_sd])
    lvl, press = [], []
    for _ in range(int(seconds / dt)):
        k1, _ = compute_velocity(y, (Q, q_feed), q_steam, params)
        k2, _ = compute_velocity(y + 0.5 * dt * k1, (Q, q_feed), q_steam, params)
        k3, _ = compute_velocity(y + 0.5 * dt * k2, (Q, q_feed), q_steam, params)
        k4, _ = compute_velocity(y + dt * k3, (Q, q_feed), q_steam, params)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        lvl.append(float(drum_level(y[1], y[2], y[3], level_ref, y[0], params)))
        press.append(float(y[0]))
    t = np.arange(1, len(lvl) + 1) * dt
    return t, np.array(lvl), np.array(press)


def _nominal(params):
    _, _, _, _, Q = steady_state(params)
    return float(Q), params.q_steam_nominal


def test_increasing_steam_demand_makes_the_level_rise_first(params):
    """SWELL: the defining non-minimum-phase behaviour of a drum boiler.

    Pressure falls, so every bubble expands and the water inventory flashes.
    The level goes *up* even though mass is leaving -- and only later falls.
    """
    Q, q_s = _nominal(params)
    t, lvl, press = _open_loop(params, Q, q_s, q_s + 5.0)
    assert press[-1] < press[0], "steam demand should drop pressure"
    assert lvl.max() > 0.005, f"no swell: peak level {lvl.max()*100:.2f} cm"
    assert lvl[-1] < 0.0, "level should end below normal once mass has left"
    # It must actually reverse, not merely overshoot.
    assert np.argmax(lvl) < np.argmin(lvl)


def test_swell_magnitude_matches_reported_load_steps(params):
    """25-100 mm is the documented range for a load step on a drum boiler."""
    Q, q_s = _nominal(params)
    _, lvl, _ = _open_loop(params, Q, q_s, q_s + 5.0)
    assert 0.015 < lvl.max() < 0.10


def test_swell_grows_with_the_size_of_the_load_step(params):
    Q, q_s = _nominal(params)
    peaks = [_open_loop(params, Q, q_s, q_s + d)[1].max() for d in (5.0, 10.0, 20.0)]
    assert peaks[0] < peaks[1] < peaks[2]


def test_adding_feedwater_makes_the_level_dip_first(params):
    """SHRINK: subcooled feedwater collapses bubbles, so the level falls before
    the added mass shows up.

    The dip is small (see D1 in PHYSICS.md) -- bounded by the feedwater
    subcooling -- but it must be present and correctly signed.
    """
    Q, q_s = _nominal(params)
    t, lvl, _ = _open_loop(params, Q, q_s + 10.0, q_s)
    early = lvl[t <= 30.0]
    assert early.min() < -1e-4, f"no shrink: minimum {early.min()*100:.3f} cm"
    assert lvl[-1] > 0.05, "level must rise substantially once the mass arrives"
    assert np.argmin(lvl) < np.argmax(lvl)


def test_more_firing_swells_the_level(params):
    """Harder firing boils more water, so voidage and level rise."""
    Q, q_s = _nominal(params)
    _, lvl, press = _open_loop(params, 1.05 * Q, q_s, q_s)
    assert press[-1] > press[0], "more firing should raise pressure"
    assert lvl.max() > 0.005


def test_level_does_not_self_regulate(params):
    """Drum level is an integrator: a small feedwater bias walks it away.

    This is why the controller cannot settle on a constant bias, and it is
    what every constant-action baseline fails on.
    """
    Q, q_s = _nominal(params)
    _, lvl, _ = _open_loop(params, Q, q_s + 2.0, q_s, seconds=600.0)
    assert lvl[-1] > 0.05
    # Still climbing at the end -- it has not found a steady state.
    assert lvl[-1] > lvl[len(lvl) // 2]


def test_balanced_operation_holds_the_level(params):
    """The complement: at the balanced point nothing should drift."""
    Q, q_s = _nominal(params)
    _, lvl, press = _open_loop(params, Q, q_s, q_s, seconds=600.0)
    assert np.abs(lvl).max() < 0.01
    assert np.abs(press - params.p_nominal).max() < 0.5


def test_falling_pressure_expands_the_steam(params):
    """The mechanism behind the swell, isolated: at constant steam mass a lower
    pressure means a higher void fraction."""
    _, m_sr, _, _, _ = steady_state(params)
    voids = [float(void_fraction(m_sr, p, params)) for p in (80.0, 85.0, 90.0)]
    assert voids[0] > voids[1] > voids[2]


# ---------------------------------------------------------------------------
# 4. Task interface
# ---------------------------------------------------------------------------


def test_observation_hides_the_void_distribution(params):
    """No plant instrument reads riser voidage, and it drives the whole
    inverse response."""
    env = BoilerDrum()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(m_sr=1234.5, m_sd=567.8, V_wt=43.21)
    obs = env.get_obs(state, params)
    assert obs.shape == (7,)
    for hidden in (1234.5, 567.8, 43.21):
        assert not bool(jnp.any(jnp.isclose(obs, hidden)))


def test_both_level_limits_terminate(params):
    """High level carries water to the turbine; low level uncovers the tubes."""
    env = BoilerDrum()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    for level in (params.level_trip + 0.01, -params.level_trip - 0.01):
        assert bool(check_is_terminal(state.replace(level=level), params)[0])
    assert not bool(check_is_terminal(state.replace(level=0.0), params)[0])


def test_pressure_limits_terminate(params):
    env = BoilerDrum()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert bool(
        check_is_terminal(state.replace(pressure=params.pressure_min - 1.0), params)[0]
    )
    assert bool(
        check_is_terminal(state.replace(pressure=params.pressure_max + 1.0), params)[0]
    )


def test_reward_peaks_on_both_targets(params):
    env = BoilerDrum()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    on = state.replace(level=0.0, pressure=85.0, target_pressure=85.0, Q_fuel=0.0)
    assert float(compute_reward(on, params)) > float(
        compute_reward(on.replace(level=0.2), params)
    )
    assert float(compute_reward(on, params)) > float(
        compute_reward(on.replace(pressure=90.0), params)
    )


def test_reset_starts_balanced(params):
    """A boiler entering a dispatch window is already lined out."""
    env = BoilerDrum()
    for seed in range(5):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, BoilerDrumState)
        assert abs(float(state.level)) <= max(map(abs, params.initial_level_range))
        assert float(state.pressure) == pytest.approx(params.p_nominal)
        assert params.target_pressure_range[0] <= float(state.target_pressure)
        assert float(state.target_pressure) <= params.target_pressure_range[1]
        assert float(state.m_sr) > 0.0 and float(state.m_sd) > 0.0


def test_state_stays_finite_over_an_episode(params):
    env = BoilerDrum()
    p = params.replace(max_steps_in_episode=200)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    _jstep = jax.jit(env.step_env)
    for _ in range(200):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = _jstep(sub, state, jnp.zeros(2), p)
        for leaf in jax.tree_util.tree_leaves(state):
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert np.isfinite(float(reward))
        if bool(terminated):
            break
