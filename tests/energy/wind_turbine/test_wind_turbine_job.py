"""The wind turbine's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Limits (rule 14): overspeed is a trip, handled by the safety system [S4] at
  about 20 % above rated speed [S17]: past it the plant is lost to a shutdown.
  Low rotor speed is not a trip: no source calls it one (REALITY.md sec. 3).
  The converter holds electrical power at or under the current set point, the
  rating or a curtailment's level (owner, 2026-11-23): S1's controller holds
  constant power in Region 3 [S1 Sec. 7.2] and S1's only margin is 10 % on
  torque [Table 7-2], so whatever torque is asked, no step ends above the set
  point, the step a curtailment starts included, and asked for more the turbine
  delivers the set point itself.
- Score (rules 15 and 23, Amendment 3), in dollars per step:
  * production short of what the wind makes available, min(rating, the wind's
    power at peak Cp), at 60 $/MWh; over-production is free;
  * the tower's fatigue: its cost, 1750 $/kW x 5000 kW x 0.75 x 0.263 =
    $1,726,031 [S20], spread so that 20 years [S20] at the design damage rate
    cost exactly that; damage is the rainflow count of the base moment (owner,
    2026-11-23; ASTM E1049, test_wind_turbine_fatigue.py), (range / 1 MN m)^4 a
    cycle and half that a half cycle;
  * the pitch system's wear: 1750 x 5000 x 0.75 x 0.0266 = $174,563 [S20], the
    same way over its design travel;
  * a trip: 2 h of rated production at 60 $/MWh, $600, plus a $1,000 manual
    reset callout [S21]: $1,600.
- Observation (rule 16): the controller's readings plus the nacelle
  accelerometer [S3, S5] and the nacelle anemometer [S6]; the rating, a
  schedule known ahead, is read. The anemometer stands at the hub, so it reads
  the hub's point wind, the mean plus the hub's own turbulence and any gust,
  not the wind the rotor averages over its disc (owner, 2026-11-23).
- Episode (rule 17): established above-rated operation, 10 min [S7, S2], at a
  mean wind between 13 and 25 m/s [S2 Table 2], starting at rated speed and
  power with the tower at its static deflection under the rotor's thrust.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.wind_turbine import env as W
from target_gym.energy.wind_turbine.env_jax import WindTurbine

PARAMS = W.WindTurbineParams(sensor_noise_scale=0.0)
RATED = 12.1 * 2.0 * np.pi / 60.0  # rad/s, S1 Table 1-1


@pytest.fixture(scope="module")
def env():
    return WindTurbine()


def _reset(env, seed=0, params=PARAMS):
    return env.reset_env(jax.random.PRNGKey(seed), params)[1]


def _step(state, params=PARAMS):
    held = jnp.stack(
        [
            2.0 * state.pitch / params.pitch_max - 1.0,
            2.0 * state.torque / params.torque_max - 1.0,
        ]
    )
    return W.compute_next_state(held, state, params, jax.random.PRNGKey(0))[0]


RATED_TORQUE = 5.0e6 / (0.944 * 97.0 * RATED)  # N m: 5 MW at rated speed, S1 p. 19
FULL_TORQUE = 47_402.91  # N m, S1 Table 7-2


def _ask(state, torque, params=PARAMS):
    """One step with the pitch held and ``torque`` (N m) asked of the generator."""
    action = jnp.stack(
        [
            2.0 * state.pitch / params.pitch_max - 1.0,
            2.0 * torque / params.torque_max - 1.0,
        ]
    )
    return W.compute_next_state(action, state, params, jax.random.PRNGKey(0))[0]


# Full torque would deliver 1.10 x the speed's share of 5 MW: 5.5 MW at rated
# speed, 6.3 MW at 1.15. The torque starts 10 % under the limit and needs two
# 3,750 N m steps to reach it, so the fourth step is on it. 1e-5 and 1e-4 are
# float32 on 5e6 W.
@pytest.mark.parametrize("speed", [1.0, 1.05, 1.15])
def test_the_converter_holds_power_at_the_rating_whatever_torque_is_asked(env, speed):
    state = _reset(env).replace(
        omega=jnp.asarray(speed * RATED),
        torque=jnp.asarray(0.9 * RATED_TORQUE / speed),
    )
    for _ in range(4):
        state = _ask(state, FULL_TORQUE)
        power = float(W.electrical_power(state.omega, state.torque, PARAMS))
        assert power <= 5.0e6 * (1.0 + 1e-5)
    assert power == pytest.approx(5.0e6, rel=1e-4)


def test_under_the_set_point_the_converter_leaves_the_torque_alone(env):
    """At 0.9 of rated speed 31 kN m is 3.2 MW, under the 5 MW set point: asked
    from 30 kN m it is met in one step (1 kN m is inside a step's 3.75)."""
    state = _reset(env).replace(
        omega=jnp.asarray(0.9 * RATED), torque=jnp.asarray(30_000.0)
    )
    assert float(_ask(state, 31_000.0).torque) == pytest.approx(31_000.0, rel=1e-5)


def test_a_curtailment_holds_from_the_step_it_starts(env):
    """A 10 % curtailment at a second: with full torque asked throughout, power
    is at the 5 MW rating for the three steps before it and at 4.5 MW from the
    step the set point falls, never above either."""
    p = PARAMS.replace(
        derate_to=0.9,
        derate_window=(1.0, 1.0),
        turbulence_iref=0.0,
        v_mean_range=(15.0, 15.0),
    )
    state = _reset(env, params=p)
    powers = []
    for _ in range(8):
        state = _ask(state, FULL_TORQUE, p)
        powers.append(float(W.electrical_power(state.omega, state.torque, p)))
    np.testing.assert_allclose(powers[:3], 5.0e6, rtol=1e-4)
    np.testing.assert_allclose(powers[3:], 4.5e6, rtol=1e-4)
    assert max(powers[:3]) <= 5.0e6 * (1.0 + 1e-5)
    assert max(powers[3:]) <= 4.5e6 * (1.0 + 1e-5)


@pytest.mark.parametrize("wind", [13.0, 25.0])
def test_no_step_of_a_turbulent_run_ends_above_the_set_point(env, wind):
    """Full torque asked through 2 min of turbulence, the pitch held and the
    overspeed level lifted so the plant runs on whatever the rotor does."""
    p = PARAMS.replace(v_mean_range=(wind, wind), overspeed_factor=1e9)
    state = _reset(env, seed=4, params=p)
    held = 2.0 * state.pitch / p.pitch_max - 1.0

    def body(s, k):
        s1 = W.compute_next_state(jnp.stack([held, 1.0]), s, p, k)[0]
        return s1, (W.electrical_power(s1.omega, s1.torque, p), s1.target_power)

    power, target = jax.lax.scan(
        body, state, jax.random.split(jax.random.PRNGKey(4), 480)
    )[1]
    assert float(jnp.max(power / target)) <= 1.0 + 1e-5


@pytest.mark.parametrize("ratio, trips", [(1.15, False), (1.19, False), (1.21, True)])
def test_overspeed_past_twenty_percent_of_rated_trips(env, ratio, trips):
    state = _reset(env).replace(omega=jnp.asarray(ratio * RATED))
    tripped, _ = W.check_is_terminal(_step(state), PARAMS)
    assert bool(tripped) is trips


def test_a_slow_rotor_is_not_a_trip(env):
    state = _reset(env).replace(omega=jnp.asarray(0.3 * RATED))
    tripped, _ = W.check_is_terminal(_step(state), PARAMS)
    assert not bool(tripped)


def test_a_trip_costs_two_hours_of_production_and_a_reset(env):
    state = _reset(env).replace(omega=jnp.asarray(1.3 * RATED))
    reward = float(W.compute_reward(_step(state), PARAMS))
    assert reward == pytest.approx(-1600.0, rel=1e-3)


def _producing(env, MW, wind=18.0):
    state = _reset(env).replace(v_wind=jnp.asarray(wind), damage=0.0, pitch_travel=0.0)
    torque = MW * 1e6 / (PARAMS.eta_gen * PARAMS.N_gear * RATED)
    return state.replace(omega=jnp.asarray(RATED), torque=jnp.asarray(torque))


@pytest.mark.parametrize("MW, short_MW", [(5.0, 0.0), (4.0, 1.0), (5.5, 0.0)])
def test_production_short_of_the_rating_is_priced_and_over_production_is_free(
    env, MW, short_MW
):
    """In 18 m/s the wind offers more than the rating. 1 MW short for 0.25 s is
    1 x 0.25 / 3600 MWh, at 60 $/MWh $0.004167."""
    terms = env.reward_terms(_producing(env, MW), PARAMS)
    assert float(terms["lost"]) == pytest.approx(short_MW * 0.25 / 3600.0 * 60.0)


def test_in_a_lull_only_what_the_wind_offers_is_owed(env):
    """At 10 m/s the wind carries 0.944 x 0.5 x 1.225 x pi 63^2 x 1000 x
    0.465861 = 3.3587 MW at peak Cp, so 3 MW is 0.3587 MW short: $0.001494."""
    terms = env.reward_terms(_producing(env, 3.0, wind=10.0), PARAMS)
    assert float(terms["lost"]) == pytest.approx(0.3587 * 0.25 / 3600 * 60, rel=2e-3)


def test_twenty_years_at_the_design_rates_cost_the_tower_and_the_pitch_system(env):
    life = 20 * 365.25 * 24 * 3600 / 0.25
    state = _producing(env, 5.0).replace(
        damage=jnp.asarray(PARAMS.tower_damage_design),
        pitch_travel=jnp.asarray(PARAMS.pitch_travel_design),
    )
    terms = env.reward_terms(state, PARAMS)
    assert float(terms["tower"]) * life == pytest.approx(1_726_031.25, rel=1e-4)
    assert float(terms["pitch"]) * life == pytest.approx(174_562.5, rel=1e-4)


def test_a_sine_moment_closes_two_half_cycles_of_its_range_per_period(env):
    """A base moment swinging +/- 10 MN m from 0. By ASTM's rule every range of a
    constant swing holds the record's first point, so each closes as a half
    cycle: 19 of 20 MN m between the 20 extremes of 10 periods, 0.5 x 20^4 =
    80,000 each, and the two end ranges of 10 MN m, 0.5 x 10^4 each."""
    stack = jnp.zeros(W.RAINFLOW_DEPTH)
    depth, moment, direction, total = jnp.int32(1), jnp.zeros(()), jnp.zeros(()), 0.0
    for k in range(1, 12 * 10 + 1):  # 10 periods of 12 samples
        new = jnp.asarray(10e6 * np.sin(2 * np.pi * k / 12 + 1e-3), jnp.float32)
        stack, depth, direction, damage = W.rainflow_step(
            stack, depth, moment, direction, new, PARAMS.fatigue_exponent
        )
        moment = new
        total += float(damage)
    assert total == pytest.approx(19 * 80_000 + 10**4, rel=1e-3)


def test_the_observation_carries_the_rating_accelerometer_and_anemometer(env):
    state = _reset(env)
    obs = np.asarray(env.get_obs(state, PARAMS))
    assert obs.shape == (7,)
    assert obs[4] == pytest.approx(5.0)
    # The hub's point wind: the mean plus the hub's turbulence (no gust on a
    # normal day), from the plant's definition of it.
    assert obs[6] == pytest.approx(float(state.v_mean + state.v_turb), rel=1e-6)


def test_the_anemometer_reads_the_hub_not_the_rotor(env):
    """Through a minute of turbulence the channel follows the hub's point wind
    step by step, which the rotor's averaged wind does not: at 18 m/s the two
    differ by about 1 m/s (standard deviation), 0.2 m/s at the very least."""
    p = PARAMS.replace(v_mean_range=(18.0, 18.0), overspeed_factor=1e9)
    state = _reset(env, seed=2, params=p)
    held = env.initial_action(state, p)

    def body(s, k):
        obs, s1, *_ = env.step_env(k, s, held, p)
        return s1, (obs[6], s1.v_mean + s1.v_turb, s1.v_wind)

    read, hub, rotor = (
        np.asarray(x, float)
        for x in jax.lax.scan(
            body, state, jax.random.split(jax.random.PRNGKey(2), 240)
        )[1]
    )
    np.testing.assert_allclose(read, hub, rtol=1e-6)
    assert np.std(read - rotor) > 0.2


def test_the_anemometer_sees_the_gust(env):
    """An extreme operating gust is one coherent rise over the hub and the disc
    alike [S13]. Its size follows the turbulence (3.3 sigma1 / 1.3), so the gust
    is the run with it less the same seed's run without: the same wind on the
    anemometer and on the rotor, 4.4 m/s at its peak at 15 m/s."""

    def run(gust):
        p = PARAMS.replace(
            gust=gust,
            gust_window=(2.0, 2.0),
            v_mean_range=(15.0, 15.0),
            overspeed_factor=1e9,
        )
        state = _reset(env, params=p)
        held = env.initial_action(state, p)

        def body(s, k):
            obs, s1, *_ = env.step_env(k, s, held, p)
            return s1, (obs[6], s1.v_wind)

        keys = jax.random.split(jax.random.PRNGKey(0), 80)
        return [np.asarray(x, float) for x in jax.lax.scan(body, state, keys)[1]]

    (read, rotor), (read0, rotor0) = run(1.0), run(0.0)
    assert np.abs(rotor - rotor0).max() == pytest.approx(4.43, rel=0.02)
    np.testing.assert_allclose(read - read0, rotor - rotor0, atol=1e-4)


def test_the_accelerometer_reads_the_tower_tops_acceleration(env):
    """Over one step the tower-top velocity changes by about the mean of the
    acceleration read at its two ends, once the tower is ringing."""
    p = PARAMS.replace(turbulence_iref=0.0)
    state = _step(_reset(env, params=p).replace(v_tower=jnp.asarray(0.05)), p)
    a0 = float(env.get_obs(state, p)[5])
    nxt = _step(state, p)
    a1 = float(env.get_obs(nxt, p)[5])
    dv = float(nxt.v_tower - state.v_tower)
    assert dv == pytest.approx(0.5 * (a0 + a1) * p.delta_t, rel=0.05)


def test_the_episode_is_ten_minutes_of_operation():
    assert PARAMS.max_steps_in_episode * PARAMS.delta_t == pytest.approx(600.0)


@pytest.mark.parametrize("seed", range(6))
def test_each_episode_starts_in_established_operation(env, seed):
    state = _reset(env, seed)
    assert 13.0 <= float(state.v_mean) <= 25.0
    assert float(state.omega) == pytest.approx(RATED, rel=1e-6)
    assert float(W.electrical_power(state.omega, state.torque, PARAMS)) == (
        pytest.approx(5.0e6, rel=1e-6)
    )
    thrust = W.rotor_thrust(state.omega, state.v_wind, state.pitch, PARAMS)
    assert float(state.v_tower) == 0.0
    assert float(state.x_tower) * float(W.tower_stiffness(PARAMS)) == (
        pytest.approx(float(thrust), rel=1e-4)
    )


def test_the_rating_does_not_move_on_a_normal_day(env):
    state = _reset(env)
    for _ in range(50):
        state = _step(state)
    assert float(state.target_power) == pytest.approx(5.0e6)


def test_a_curtailment_lowers_the_rating_once_at_its_drawn_time(env):
    p = PARAMS.replace(derate_to=0.9, derate_window=(1.0, 1.0))
    state = _reset(env, params=p)
    targets = []
    for _ in range(8):
        state = _step(state, p)
        targets.append(float(state.target_power))
    assert targets[:3] == [5.0e6] * 3 and targets[3:] == [4.5e6] * 5


def test_the_scenario_suite_builds_and_every_scenario_runs():
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    spec = REGISTRY["wind_turbine"]
    suite = scenarios.suite(spec)
    assert 4 <= len(suite) <= 10 and suite[0].name == "normal"
    assert all(s.source for s in suite)
    for s in suite:
        p = scenarios.params_for(spec, s, max_steps_in_episode=8)
        e = spec.make_env()
        obs, state = e.reset_env(jax.random.PRNGKey(0), p)
        for _ in range(8):
            obs, state, reward, _, _ = e.step_env(
                jax.random.PRNGKey(0), state, e.initial_action(state, p), p
            )
        assert np.isfinite(float(reward)) and np.all(np.isfinite(np.asarray(obs)))
