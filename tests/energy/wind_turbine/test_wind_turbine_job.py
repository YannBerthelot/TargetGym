"""The wind turbine's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Limits (rule 14): overspeed is a trip, handled by the safety system [S4] at
  about 20 % above rated speed [S17]: past it the plant is lost to a shutdown.
  Low rotor speed is not a trip: no source calls it one (REALITY.md sec. 3).
- Score (rule 15): the energy delivered off the rating, in MWh per step, and a
  trip costs rated power over a restart, ROSCO's 400 s start-up [S9 Ex. 32]:
  5 MW x 400 s = 0.5556 MWh.
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


def test_a_trip_costs_rated_power_over_a_400_s_restart(env):
    state = _reset(env).replace(omega=jnp.asarray(1.3 * RATED))
    reward = float(W.compute_reward(_step(state), PARAMS))
    assert reward == pytest.approx(-5.0 * 400.0 / 3600.0, rel=1e-3)


@pytest.mark.parametrize("off_MW", [0.0, 1.0, -0.5])
def test_the_score_is_the_energy_off_the_rating(env, off_MW):
    """One quarter-second step 1 MW off the rating is 1 x 0.25 / 3600 MWh,
    above or below it alike."""
    state = _reset(env)
    torque = (5.0e6 + off_MW * 1e6) / (PARAMS.eta_gen * PARAMS.N_gear * RATED)
    state = state.replace(omega=jnp.asarray(RATED), torque=jnp.asarray(torque))
    reward = float(W.compute_reward(state, PARAMS))
    assert reward == pytest.approx(-abs(off_MW) * 0.25 / 3600.0, abs=1e-9)


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


def test_the_rating_does_not_move_during_an_episode(env):
    state = _reset(env)
    for _ in range(50):
        state = _step(state)
    assert float(state.target_power) == pytest.approx(5.0e6)
