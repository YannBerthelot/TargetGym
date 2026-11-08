"""The slab building's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Limits (rule 14): nothing trips. Below 4.4 C the plant's frost protection
  takes the supply to its maximum until the room exceeds 7.2 C [1]; the supply
  never exceeds the TABS family's 49 C [14].
- Score (rule 15, Amendment 3), euros: the heat pump's electricity, slab heat
  over COP = 0.52 T_sup / (T_sup - T_brine) in kelvin with brine at 0 C, at a
  three-period tariff (valle 0.139 00-08 h, llano 0.184, punta 0.263 10-14 and
  18-22 h, [19]); and lost office work while occupied (08:00 to 18:00 [6]),
  0.02 x 15 people x EUR 31.8 = EUR 9.54 per kelvin-hour of room outside 21 to
  26 C [9, p. 5] or of floor above 29 C [5] [20, 21].
- Observation (rule 16): the room, outdoor and its 24 h mean [9, 13], the
  slab core [7], the return water [12], the supply, the clock, the band, the
  next 24 h's forecast mean outdoor temperature, and the tariff now with the
  hours to its next change.
- Episode (rule 17): two weeks [11, p. 1], from established operation at the
  band's heating edge, no setback [7, p. 7], [15].
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.hvac.env import HVACParams, check_is_terminal, compute_reward
from target_gym.hvac.env_jax import BuildingHVAC

PARAMS = HVACParams(sensor_noise_scale=0.0)
HOURS = PARAMS.delta_t / 3600.0


@pytest.fixture(scope="module")
def env():
    return BuildingHVAC()


def _reset(env, seed=0):
    return env.reset_env(jax.random.PRNGKey(seed), PARAMS)


@pytest.mark.parametrize("T_air", [-5.0, 3.0, 40.0])
def test_nothing_trips(env, T_air):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(state.replace(T_air=jnp.asarray(T_air)), PARAMS)
    assert not bool(tripped)


def test_frost_protection_takes_the_supply_to_its_maximum(env):
    _, state = _reset(env)
    cold = state.replace(frost=jnp.asarray(1.0))
    off = jnp.array([-1.0])
    _, after, _, _, _ = env.step_env(jax.random.PRNGKey(0), cold, off, PARAMS)
    assert float(after.T_supply) == pytest.approx(49.0)


def test_frost_protection_starts_below_4_4_and_holds_until_7_2(env):
    _, state = _reset(env)
    off = jnp.array([-1.0])
    params = PARAMS.replace(T_out_mean=-40.0, T_out_amplitude=0.0, T_out_noise_std=0.0)
    cold = state.replace(
        T_mass=jnp.asarray(3.0),
        T_core=jnp.asarray(3.0),
        T_out_hist=jnp.full_like(state.T_out_hist, -40.0),
    )
    _, after, _, _, _ = env.step_env(jax.random.PRNGKey(0), cold, off, params)
    assert float(after.T_air) < 4.4 and float(after.frost) == 1.0
    warm = after.replace(T_air=jnp.asarray(6.0))
    _, kept, _, _, _ = env.step_env(jax.random.PRNGKey(0), warm, off, params)
    assert float(kept.frost) == 1.0 or float(kept.T_air) > 7.2


def test_supply_never_exceeds_the_tabs_maximum(env):
    _, state = _reset(env)
    _, after, _, _, _ = env.step_env(
        jax.random.PRNGKey(0), state, jnp.array([1.0]), PARAMS
    )
    assert float(after.T_supply) <= 49.0


@pytest.mark.parametrize(
    "T_supply, cop", [(35.0, 0.52 * 308.15 / 35.0), (55.0, 0.52 * 328.15 / 55.0)]
)
def test_the_heat_pumps_cop_is_in_the_datasheets_range(T_supply, cop):
    """4.58 at W35 against datasheets' 4.4 to 5.0; 3.10 at W55 against 2.7 to
    3.2 [18]."""
    from target_gym.hvac.env import heat_pump_cop

    assert float(heat_pump_cop(T_supply, PARAMS)) == pytest.approx(cop, rel=1e-5)


@pytest.mark.parametrize(
    "hour, price",
    [(3, 0.139), (9, 0.184), (12, 0.263), (16, 0.184), (20, 0.263), (23, 0.184)],
)
def test_the_tariff_has_three_periods(hour, price):
    from target_gym.hvac.env import electricity_price

    assert float(electricity_price(hour * 4, PARAMS)) == pytest.approx(price)


def test_inside_the_band_only_the_electricity_is_paid(env):
    """2 kW of slab heat at 30 C supply in the quarter hour from 10:00 (punta):
    COP 0.52 x 303.15 / 30 = 5.2546, 0.095154 kWh, EUR 0.025026."""
    _, state = _reset(env)
    inside = state.replace(
        time=41,
        T_air=jnp.asarray(23.0),
        T_floor=jnp.asarray(24.0),
        Q_slab=jnp.asarray(2000.0),
        T_supply=jnp.asarray(30.0),
    )
    assert float(-compute_reward(inside, PARAMS)) == pytest.approx(0.025026, rel=1e-4)


@pytest.mark.parametrize(
    "T_air, T_floor, kelvin",
    [(20.0, 22.0, 1.0), (27.5, 28.0, 1.5), (23.0, 30.0, 1.0), (19.0, 30.0, 3.0)],
)
def test_occupied_discomfort_is_priced_as_lost_work(env, T_air, T_floor, kelvin):
    """K outside 21 to 26 C plus K of floor above 29 C, for a quarter hour at
    12:00, at EUR 9.54 per Kh, with no heat."""
    _, state = _reset(env)
    out = state.replace(
        time=48,
        T_air=jnp.asarray(T_air),
        T_floor=jnp.asarray(T_floor),
        Q_slab=jnp.asarray(0.0),
    )
    assert float(-compute_reward(out, PARAMS)) == pytest.approx(
        9.54 * kelvin * 0.25, rel=1e-5
    )


def test_out_of_hours_discomfort_costs_nothing(env):
    _, state = _reset(env)
    night = state.replace(time=8, T_air=jnp.asarray(18.0), Q_slab=jnp.asarray(0.0))
    assert float(compute_reward(night, PARAMS)) == 0.0


def test_the_observation_reads_the_forecast_and_the_tariff(env):
    """At midnight: valle, eight hours to its end. The forecast is the next
    24 h's mean of the daily cycle with today's deviation decaying at the
    weather's rate, by hand here as a plain sum."""
    from target_gym.hvac.env import WEATHER_OU_THETA, outdoor_temperature

    obs, state = _reset(env, 2)
    decay = 1.0 - WEATHER_OU_THETA * PARAMS.delta_t
    path = [
        float(outdoor_temperature(k, float(state.weather_dev) * decay**k, PARAMS))
        for k in range(1, 97)
    ]
    assert float(obs[9]) == pytest.approx(sum(path) / 96.0, rel=1e-4)
    assert float(obs[10]) == pytest.approx(0.139)
    assert float(obs[11]) == pytest.approx(8.0)


def test_the_scenarios_are_the_familys_and_each_runs():
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    spec = REGISTRY["hvac"]
    suite = scenarios.suite(spec)
    assert [s.name for s in suite] == [
        "normal",
        "cold_snap",
        "shoulder_season",
        "sunny_spell",
        "low_gains",
    ]
    env = spec.make_env()
    for sc in suite:
        assert "[" in sc.source and "REALITY.md" in sc.source, sc.name
        p = scenarios.params_for(spec, sc, sensor_noise_scale=0.0)
        obs, state = env.reset_env(jax.random.PRNGKey(0), p)
        for k in range(5):
            obs, state, r, _, _ = env.step_env(
                jax.random.PRNGKey(k), state, env.initial_action(state, p), p
            )
        assert np.all(np.isfinite(np.asarray(obs))) and np.isfinite(float(r))


def test_the_observation_reads_the_slab_and_the_days_mean_outdoor(env):
    obs, state = _reset(env, 4)
    np.testing.assert_allclose(
        np.asarray(obs[:6]),
        [
            float(state.T_air),
            float(state.T_out),
            float(np.mean(np.asarray(state.T_out_hist))),
            float(state.T_core),
            float(state.T_return),
            float(state.T_supply),
        ],
        rtol=1e-5,
    )
    assert float(obs[8]) == pytest.approx(23.5)


def test_the_days_mean_follows_the_last_96_outdoor_readings(env):
    _, state = _reset(env, 5)
    action = env.initial_action(state, PARAMS)
    step = jax.jit(env.step_env)
    outs = [float(t) for t in state.T_out_hist]
    for k in range(10):
        _, state, _, _, _ = step(jax.random.PRNGKey(k), state, action, PARAMS)
        outs.append(float(state.T_out))
    assert float(np.mean(np.asarray(state.T_out_hist))) == pytest.approx(
        float(np.mean(outs[-96:])), rel=1e-5
    )


@pytest.mark.parametrize("seed", range(5))
def test_episode_starts_established_at_the_bands_heating_edge(env, seed):
    """Settled for the last day's mean weather at 21 C: the room starts within
    a kelvin of it, and the slab is warmer than the room and cooler than the
    supply, as a heating slab is."""
    _, state = _reset(env, seed)
    assert abs(float(state.T_air) - 21.0) < 1.0
    assert float(state.T_air) < float(state.T_core) <= float(state.T_supply)
    assert float(state.frost) == 0.0


def test_the_episode_is_two_weeks(env):
    assert PARAMS.max_steps_in_episode * PARAMS.delta_t == 14 * 86400.0
