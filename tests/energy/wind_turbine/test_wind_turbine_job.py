"""The wind turbine's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Limits (rule 14): overspeed is a trip, handled by the safety system [S4] at
  about 20 % above rated speed [S17]: past it the plant is lost to a shutdown.
  Low rotor speed is not a trip: no source calls it one (REALITY.md sec. 3).
- Score (rules 15 and 23, Amendment 3), in dollars per step:
  * production short of what the wind makes available, min(rating, the wind's
    power at peak Cp), at 60 $/MWh; over-production is free;
  * the tower's fatigue: its cost, 1750 $/kW x 5000 kW x 0.75 x 0.263 =
    $1,726,031 [S20], spread so that 20 years [S20] at the design damage rate
    cost exactly that; damage closes per half-cycle of the base moment between
    turning points, 0.5 (range / 1 MN m)^4;
  * the pitch system's wear: 1750 x 5000 x 0.75 x 0.0266 = $174,563 [S20], the
    same way over its design travel;
  * a trip: 2 h of rated production at 60 $/MWh, $600, plus a $1,000 manual
    reset callout [S21]: $1,600.
- Observation (rule 16): the controller's readings plus the nacelle
  accelerometer [S3, S5] and the nacelle anemometer [S6]; the rating, a
  schedule known ahead, is read.
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
    """A base moment swinging +/- 10 MN m: each half-cycle spans 20 MN m, damage
    0.5 x 20^4 = 80,000, two per period."""
    state = _reset(env).replace(moment=0.0, moment_turn=0.0, moment_dir=0.0)
    total = 0.0
    for k in range(1, 12 * 10 + 1):  # 10 periods of 12 samples
        moment = 10e6 * np.sin(2 * np.pi * k / 12 + 1e-3)
        m, turn, d, damage = W.count_half_cycles(state, jnp.asarray(moment), PARAMS)
        state = state.replace(moment=m, moment_turn=turn, moment_dir=d)
        total += float(damage)
    # The first half-cycle starts at 0, not at a peak: 19 full ranges, one half.
    assert total == pytest.approx(19 * 80_000 + 0.5 * 10**4, rel=1e-3)


def test_the_observation_carries_the_rating_accelerometer_and_anemometer(env):
    state = _reset(env)
    obs = np.asarray(env.get_obs(state, PARAMS))
    assert obs.shape == (7,)
    assert obs[4] == pytest.approx(5.0)
    assert obs[6] == pytest.approx(float(state.v_wind), rel=1e-6)


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
