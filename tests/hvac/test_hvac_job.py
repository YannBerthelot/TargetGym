"""The slab building's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Limits (rule 14): nothing trips. Below 4.4 C the plant's frost protection
  takes the supply to its maximum until the room exceeds 7.2 C [1]; the supply
  never exceeds 55 C [17].
- Score (rule 15), euros: the slab's heat at EUR 0.10/kWh, and EUR 0.03 per
  kelvin-hour of room outside 21 to 26 C [9, p. 5] or of floor above 29 C [5].
- Observation (rule 16): the room, outdoor and its 24 h mean [9, 13], the
  slab core [7], the return water [12], the supply, the clock and the band.
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
    assert float(after.T_supply) == pytest.approx(55.0)


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


def test_supply_never_exceeds_the_screed_limit(env):
    _, state = _reset(env)
    _, after, _, _, _ = env.step_env(
        jax.random.PRNGKey(0), state, jnp.array([1.0]), PARAMS
    )
    assert float(after.T_supply) <= 55.0


def test_inside_the_band_only_the_heat_is_paid(env):
    """2 kW for 15 min is 0.5 kWh, EUR 0.05."""
    _, state = _reset(env)
    inside = state.replace(
        T_air=jnp.asarray(23.0), T_floor=jnp.asarray(24.0), Q_slab=jnp.asarray(2000.0)
    )
    assert float(-compute_reward(inside, PARAMS)) == pytest.approx(0.05, rel=1e-5)


@pytest.mark.parametrize(
    "T_air, T_floor, kelvin",
    [(20.0, 22.0, 1.0), (27.5, 28.0, 1.5), (23.0, 30.0, 1.0), (19.0, 30.0, 3.0)],
)
def test_discomfort_is_priced_per_kelvin_hour(env, T_air, T_floor, kelvin):
    """K outside 21 to 26 C plus K of floor above 29 C, for a quarter hour, at
    EUR 0.03 per Kh, with no heat."""
    _, state = _reset(env)
    out = state.replace(
        T_air=jnp.asarray(T_air), T_floor=jnp.asarray(T_floor), Q_slab=jnp.asarray(0.0)
    )
    assert float(-compute_reward(out, PARAMS)) == pytest.approx(
        0.03 * kelvin * 0.25, rel=1e-5
    )


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
