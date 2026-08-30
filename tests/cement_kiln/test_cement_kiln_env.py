"""Physics validation for the cement rotary kiln.

Enforces ``src/target_gym/cement_kiln/PHYSICS.md``.

The behavioural half matters most here. The transport delay is the reason this
environment exists and it is *emergent* -- there is no delay parameter in the
model, only upwind advection of the solid states. A refactor that quietly made
the bed well-mixed would leave every coefficient correct and destroy the whole
control problem, which is what these tests exist to catch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.cement_kiln.env import (
    CementKilnParams,
    CementKilnState,
    burning_zone_temperature,
    check_is_terminal,
    compute_reward,
    discharge_lime,
    flame_profile,
    gas_sweep,
    kiln_feed_rate,
    residence_time,
    specific_heat_consumption,
    steady_profile,
)
from target_gym.cement_kiln.env_jax import CementKiln


@pytest.fixture(scope="module")
def params():
    return CementKilnParams()


@pytest.fixture(scope="module")
def profile(params):
    """The lined-out axial profile at the nominal operating point."""
    T_solid, T_wall, alpha, lime = steady_profile(params)
    T_gas, _, _, T_exhaust = gas_sweep(T_solid, T_wall, params.fuel_nominal, params)
    return dict(
        T_solid=np.asarray(T_solid),
        T_wall=np.asarray(T_wall),
        alpha=np.asarray(alpha),
        lime=np.asarray(lime),
        T_gas=np.asarray(T_gas),
        T_exhaust=float(T_exhaust),
    )


# ---------------------------------------------------------------------------
# 1. Geometry and material flow
# ---------------------------------------------------------------------------


def test_length_to_diameter_ratio_is_typical(params):
    """13-17 for a preheater kiln."""
    assert 13.0 < params.length / params.diameter < 17.0


def test_residence_time_matches_the_sullivan_correlation(params):
    """25-40 min is the documented range for a rotary kiln."""
    assert 25.0 < float(residence_time(2.5, params)) / 60.0 < 40.0
    assert 20.0 < float(residence_time(3.0, params)) / 60.0 < 30.0


def test_residence_time_falls_with_kiln_speed(params):
    taus = [float(residence_time(r, params)) for r in (2.0, 3.0, 4.5)]
    assert taus[0] > taus[1] > taus[2]


def test_cross_section_widths_sum_to_the_circumference(params):
    """The covered arc plus the exposed wall arc is the whole circumference."""
    total = params.w_wall_bed + params.w_wall_gas
    assert total == pytest.approx(np.pi * params.diameter, rel=0.01)


def test_kiln_is_fed_calcined_hot_meal_not_raw_meal(params):
    """Calcination sheds CO2 upstream, so the kiln sees far less mass.

    Feeding the kiln the raw meal rate overstates its thermal load by nearly
    50 %, which is exactly the error the duty cross-check caught.
    """
    feed = float(kiln_feed_rate(params.raw_meal_nominal, params))
    assert feed == pytest.approx(36.6, rel=0.02)
    assert feed < 0.72 * params.raw_meal_nominal


def test_flame_is_distributed_and_normalised(params):
    """A point source at the burner leaves the rest of the kiln cold."""
    frac = np.asarray(flame_profile(params))
    assert frac.sum() == pytest.approx(1.0, rel=1e-5)
    assert np.all(np.diff(frac) > 0), "heat release must grow toward the burner"
    assert frac[-1] < 0.35, "flame too concentrated in the last zone"


# ---------------------------------------------------------------------------
# 2. Energy balance (PHYSICS.md section 2)
# ---------------------------------------------------------------------------


def test_kiln_duty_agrees_with_the_published_split(params):
    """Two independent routes to the burner duty must agree.

    Top-down: 40 % of a published 3.2 MJ/kg specific consumption.
    Bottom-up: over the *calcined* feed, sensible heat plus residual
    calcination plus clinker formation. Nothing is fitted to make these match,
    and getting the bottom-up wrong is how the raw-meal error was caught.
    """
    clinker = params.raw_meal_nominal * (1.0 - params.caco3_fraction * 0.44)
    top_down = 0.40 * 3.2e6 * clinker

    feed = float(kiln_feed_rate(params.raw_meal_nominal, params))
    residual = (1.0 - params.calcination_upstream) * params.caco3_fraction
    bottom_up = feed * (
        params.cp_solid * (1450.0 - 800.0)
        + residual * params.h_calcination
        + params.h_clinkerisation
    )
    assert bottom_up == pytest.approx(top_down, rel=0.08)


def test_energy_closes_on_the_converged_profile(params, profile):
    """Conservation: everything in must come out.

    In: fuel plus the secondary air from the clinker cooler.
    Out: exhaust gas to the precalciner, shell losses, and heat into the
    charge. This is the load-bearing check on the whole model -- it ties the
    gas sweep, the reactions and the shell losses to one another.
    """
    p = params
    feed = float(kiln_feed_rate(p.raw_meal_nominal, p))
    gas = p.fuel_nominal * p.air_fuel_ratio * p.excess_air
    dz = p.length / p.n_zones
    A_shell = np.pi * p.diameter * dz

    q_in = p.fuel_nominal * p.fuel_lhv + gas * p.cp_gas * (
        p.T_secondary_air - p.T_ambient
    )
    exhaust = gas * p.cp_gas * (profile["T_exhaust"] - p.T_ambient)
    shell = float((p.U_shell * A_shell * (profile["T_wall"] - p.T_ambient)).sum())
    residual = (1.0 - p.calcination_upstream) * p.caco3_fraction
    solid = feed * (
        p.cp_solid * (profile["T_solid"][-1] - p.T_feed)
        + residual * p.h_calcination * profile["alpha"][-1]
        + p.h_clinkerisation * (1.0 - profile["lime"][-1])
    )
    assert exhaust + shell + solid == pytest.approx(q_in, rel=0.02)
    # Shell loss should be the documented 8-12 % of fuel.
    assert 0.05 < shell / (p.fuel_nominal * p.fuel_lhv) < 0.15


def test_specific_heat_consumption_is_in_the_published_band(params):
    """3.0-3.5 MJ/kg clinker for a modern precalciner kiln."""
    env = CementKiln()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(fuel=params.fuel_nominal, raw_meal=params.raw_meal_nominal)
    assert 3.0 < float(specific_heat_consumption(state, params)) < 3.5


def test_gas_is_far_faster_than_the_solid(params):
    """Justifies solving the gas quasi-steadily rather than dynamically."""
    gas_flow = params.fuel_nominal * params.air_fuel_ratio * params.excess_air
    rho_gas = 0.25
    area = np.pi * params.diameter**2 / 4 * 0.9
    gas_residence = params.length / (gas_flow / (rho_gas * area))
    solid_residence = float(residence_time(params.rpm_nominal, params))
    assert gas_residence < 20.0
    assert solid_residence / gas_residence > 50.0


# ---------------------------------------------------------------------------
# 3. Axial profile
# ---------------------------------------------------------------------------


def test_burning_zone_reaches_clinkering_temperature(profile):
    """~1450 C is where alite forms."""
    assert 1420.0 < profile["T_solid"].max() - 273.0 < 1540.0


def test_back_end_gas_temperature_is_realistic(profile):
    """1000-1200 C leaving the kiln toward the precalciner."""
    assert 1000.0 < profile["T_exhaust"] - 273.0 < 1200.0


def test_free_lime_at_discharge_is_saleable(profile):
    """0.5-2 % is the commercial band."""
    assert 0.005 < profile["lime"][-1] < 0.021


def test_solid_heats_monotonically_along_the_kiln(profile):
    """The charge must warm on its way to the burner, not cool."""
    T = profile["T_solid"]
    assert np.all(np.diff(T[:-1]) > 0), "charge cools somewhere along the kiln"
    assert T[0] < T.max()


def test_gas_is_hotter_than_solid_everywhere(profile):
    """Counter-current heat transfer only runs one way."""
    assert np.all(profile["T_gas"] > profile["T_solid"])


def test_wall_sits_between_gas_and_solid(profile):
    """The refractory is heated by gas and gives to the bed."""
    assert np.all(profile["T_wall"] < profile["T_gas"])
    mid = slice(2, -2)
    assert np.all(profile["T_wall"][mid] > profile["T_solid"][mid])


def test_calcination_completes_before_discharge(profile):
    """Residual carbonate must be gone well before the burning zone."""
    assert profile["alpha"][-1] > 0.99


def test_free_lime_falls_monotonically(profile):
    """Lime is consumed as the charge burns; it never re-forms."""
    assert np.all(np.diff(profile["lime"]) <= 1e-6)
    assert profile["lime"][0] > 0.9, "meal enters essentially unburnt"


# ---------------------------------------------------------------------------
# 4. Transport delay -- the behaviour this environment exists for
# ---------------------------------------------------------------------------


def _open_loop(params, fuel_raw, rpm_raw, steps, seed=0):
    """Constant action from the lined-out state, with the feed disturbance off."""
    p = params.replace(feed_noise_std=0.0, max_steps_in_episode=steps + 1)
    env = CementKiln()
    key = jax.random.PRNGKey(seed)
    _, state = env.reset_env(key, p)
    action = jnp.array([fuel_raw, rpm_raw])
    lime, T_bz = [], []
    step = jax.jit(env.step_env)
    for _ in range(steps):
        _, state, _, terminated, _ = step(key, state, action, p)
        lime.append(float(discharge_lime(state)))
        T_bz.append(float(burning_zone_temperature(state)))
        if bool(terminated):
            break
    return np.array(lime), np.array(T_bz)


def _nominal_action(params):
    f = (
        2.0
        * (params.fuel_nominal - params.fuel_min)
        / (params.fuel_max - params.fuel_min)
        - 1.0
    )
    r = (
        2.0 * (params.rpm_nominal - params.rpm_min) / (params.rpm_max - params.rpm_min)
        - 1.0
    )
    return f, r


def test_nominal_action_holds_the_operating_point(params):
    """The complement to every step test: nothing drifts on its own."""
    f, r = _nominal_action(params)
    lime, T_bz = _open_loop(params, f, r, 120)
    assert np.abs(lime - lime[0]).max() < 0.004
    assert np.abs(T_bz - T_bz[0]).max() < 25.0


def test_free_lime_lags_a_fuel_change_by_a_residence_time(params):
    """The transport delay, measured -- and it is emergent.

    Nothing in the model contains a delay parameter. Fuel heats the burning
    zone locally, so free lime starts moving within minutes, but the *bulk* of
    the response has to wait for material to traverse the kiln: half the
    eventual change takes about a full residence time and ninety per cent takes
    several. A controller acting on the discharge assay is acting on
    substantially stale information, and that is the whole difficulty.
    """
    f, r = _nominal_action(params)
    lime, _ = _open_loop(params, f + 0.30, r, 300)
    dt_min = params.delta_t / 60.0
    tau_min = float(residence_time(params.rpm_nominal, params)) / 60.0

    move = lime - lime[0]
    total = move[-1]
    assert abs(total) > 1e-3, "fuel step produced no response to measure"

    def time_to(frac):
        return float(np.argmax(np.abs(move) > frac * abs(total))) * dt_min

    t50, t90 = time_to(0.5), time_to(0.9)
    assert (
        t50 > 0.6 * tau_min
    ), f"half-response in {t50:.1f} min vs {tau_min:.1f} min residence"
    assert (
        t90 > 1.5 * tau_min
    ), f"90 % response in {t90:.1f} min -- too fast to be transport-limited"


def test_a_feed_change_takes_a_residence_time_to_show(params):
    """Changing what goes in only shows at the discharge after it arrives.

    Starts from the profile lined out at the *nominal* feed and then steps the
    feed, so the transient is the transport of the change down the kiln.
    Resetting with the new feed rate would line out at the new value and show
    nothing.
    """
    f, r = _nominal_action(params)
    lower = 0.88 * params.raw_meal_nominal
    p_step = params.replace(
        feed_noise_std=0.0, raw_meal_nominal=lower, max_steps_in_episode=400
    )
    p_ref = params.replace(feed_noise_std=0.0, max_steps_in_episode=400)

    env = CementKiln()
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p_ref)  # lined out at nominal feed
    state = state.replace(raw_meal=jnp.asarray(lower))

    step = jax.jit(env.step_env)
    action = jnp.array([f, r])
    lime = []
    for _ in range(300):
        _, state, _, terminated, _ = step(key, state, action, p_step)
        lime.append(float(discharge_lime(state)))
        if bool(terminated):
            break
    lime = np.array(lime)

    move = lime - lime[0]
    assert abs(move[-1]) > 1e-3, "feed step produced no response to measure"
    t90 = float(np.argmax(np.abs(move) > 0.9 * abs(move[-1]))) * params.delta_t / 60.0
    assert t90 > 15.0, f"feed change fully visible in {t90:.1f} min"


def test_more_fuel_eventually_lowers_free_lime(params):
    """Direction check: a hotter kiln burns the charge out further."""
    f, r = _nominal_action(params)
    base, _ = _open_loop(params, f, r, 120)
    hot, _ = _open_loop(params, f + 0.30, r, 120)
    assert hot[-1] < base[-1]


def test_kiln_speed_changes_the_delay_itself(params):
    """Speed is a qualitatively different input from fuel.

    It sets residence time and holdup together, so it moves the plant's
    dynamics rather than adding heat somewhere else -- and it does so at nearly
    unchanged burning-zone temperature.
    """
    f, _ = _nominal_action(params)
    slow_raw = 2.0 * (2.4 - params.rpm_min) / (params.rpm_max - params.rpm_min) - 1.0
    fast_raw = 2.0 * (4.2 - params.rpm_min) / (params.rpm_max - params.rpm_min) - 1.0
    slow_lime, slow_T = _open_loop(params, f, slow_raw, 200)
    fast_lime, fast_T = _open_loop(params, f, fast_raw, 200)

    # Longer residence burns the charge out further.
    assert slow_lime[-1] < fast_lime[-1]
    # ...while barely moving the burning-zone temperature.
    assert abs(slow_T[-1] - fast_T[-1]) < 60.0
    assert float(residence_time(2.4, params)) > float(residence_time(4.2, params))


def test_free_lime_is_very_sensitive_to_temperature(params):
    """A 280 kJ/mol activation energy makes small thermal errors expensive."""
    p = params
    k = lambda T: p.A_clinker * np.exp(-p.E_clinker / (8.314 * T))
    tau_low = 1.0 / k(1500.0) / 60.0  # ~1230 C
    tau_high = 1.0 / k(1750.0) / 60.0  # ~1480 C
    assert tau_low > 10.0 * tau_high
    assert tau_high < 3.0


# ---------------------------------------------------------------------------
# 5. Task interface
# ---------------------------------------------------------------------------


def test_observation_hides_the_axial_profile(params):
    """64 states are dynamic and eight numbers come out."""
    env = CementKiln()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    obs = env.get_obs(state, params)
    assert obs.shape == (8,)
    n_dynamic = 4 * params.n_zones
    assert n_dynamic > 8 * 4, "the profile should dwarf the observation"
    # Interior zone states must not appear in the observation.
    for value in (state.T_solid[5], state.T_wall[5], state.lime[5], state.alpha[5]):
        assert not bool(jnp.any(jnp.isclose(obs, value, rtol=1e-6)))


def test_overheating_terminates(params):
    env = CementKiln()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    hot = state.replace(T_solid=state.T_solid.at[-1].set(params.T_bz_max + 10.0))
    assert bool(check_is_terminal(hot, params)[0])


def test_going_cold_terminates(params):
    env = CementKiln()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    cold = state.replace(T_solid=jnp.full_like(state.T_solid, params.T_bz_min - 10.0))
    assert bool(check_is_terminal(cold, params)[0])


def test_nominal_state_does_not_terminate(params):
    env = CementKiln()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert not bool(check_is_terminal(state, params)[0])


def test_both_failure_modes_are_reachable(params):
    """A boundary nothing can reach is not a constraint."""
    f, r = _nominal_action(params)
    hot_lime, _ = _open_loop(params, 1.0, r, 200)
    cold_lime, _ = _open_loop(params, -1.0, r, 400)
    assert len(hot_lime) < 200, "maximum firing never overheats the kiln"
    assert len(cold_lime) < 400, "minimum firing never lets the kiln go cold"


def test_reward_peaks_on_target(params):
    env = CementKiln()
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    on = state.replace(
        lime=state.lime.at[-1].set(0.012),
        target_lime=jnp.asarray(0.012),
        fuel=jnp.asarray(params.fuel_min),
    )
    off = on.replace(lime=on.lime.at[-1].set(0.030))
    assert float(compute_reward(on, params)) > float(compute_reward(off, params))


def test_reset_starts_lined_out(params):
    """A kiln entering a shift is already at its operating point."""
    env = CementKiln()
    for seed in range(4):
        _, state = env.reset_env(jax.random.PRNGKey(seed), params)
        assert isinstance(state, CementKilnState)
        assert state.T_solid.shape == (params.n_zones,)
        assert 1420.0 < float(burning_zone_temperature(state)) - 273.0 < 1540.0
        assert 0.005 < float(discharge_lime(state)) < 0.021
        assert params.target_lime_range[0] <= float(state.target_lime)
        assert float(state.target_lime) <= params.target_lime_range[1]


def test_state_stays_finite_over_an_episode(params):
    env = CementKiln()
    p = params.replace(max_steps_in_episode=150)
    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, p)
    for _ in range(150):
        key, sub = jax.random.split(key)
        _, state, reward, terminated, _ = env.step_env(sub, state, jnp.zeros(2), p)
        for leaf in jax.tree_util.tree_leaves(state):
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert np.isfinite(float(reward))
        if bool(terminated):
            break
