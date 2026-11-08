"""The grid battery's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Lever (REALITY.md sec. 2 and 6, NESO-ST para. 6.11(v), 6.13): the action is
  the Operational Baseline. The plant delivers the baseline plus the contracted
  response; under-delivery is not a lever. A baseline can be changed only for
  a settlement period more than 60 min ahead (gate closure, NESO-GUIDE p. 36).
- Limits (rule 14): the energy rule and the unavailability rule are priced,
  never trips (NESO-ST para. 6.12, Schedule 3). Cell voltage and temperature
  are protected: the BMS refuses power, nothing trips (EVE-LF280K).
- Score (rule 15): availability payment lost, in MW h of contracted
  availability: a whole 30 min settlement period once the unit is unavailable
  for 0.1 % of it or fails the energy check at its start (NESO-ST Schedule 3,
  para. 6.12).
- Observation (rule 16): the BMS's reported SoC, the locked baselines, the
  period's energy requirements and the time to the next period.
- Episode (rule 17): one 4 h EFA block in established operation, starting
  with the reported energy meeting the block's requirement both ways
  (NESO-GUIDE p. 37, item 2).
"""

import jax
import jax.numpy as jnp
import pytest

from target_gym.energy.battery import env as battery
from target_gym.energy.battery.env import BatteryParams, check_is_terminal
from target_gym.energy.battery.env_jax import GridBattery

PARAMS = BatteryParams(sensor_noise_scale=0.0)
ENV = GridBattery()
KEY = jax.random.PRNGKey(3)
PER_PERIOD = int(round(PARAMS.period_seconds / PARAMS.delta_t))


def _reset(seed=0, params=PARAMS):
    return ENV.reset_env(jax.random.PRNGKey(seed), params)


def _step(state, raw=0.0, params=PARAMS):
    return ENV.step_env(KEY, state, jnp.full(1, raw), params)


def test_the_episode_is_one_efa_block_at_one_second():
    assert PARAMS.delta_t == 1.0
    assert PARAMS.max_steps_in_episode * PARAMS.delta_t == 4 * 3600


@pytest.mark.parametrize("seed", range(8))
def test_the_block_starts_with_its_energy_requirement_met(seed):
    """Reported energy at least the contracted volume each way, at 25 degC."""
    obs, state = _reset(seed)
    volume_MWh = float(battery.response_volume(PARAMS)) / 3.6e9
    capacity_MWh = PARAMS.energy_capacity / 3.6e9
    assert float(obs[0]) * capacity_MWh >= volume_MWh - 1e-6
    assert (1.0 - float(obs[0])) * capacity_MWh >= volume_MWh - 1e-6
    assert float(obs[9]) == pytest.approx(volume_MWh, rel=1e-6)
    assert float(obs[10]) == pytest.approx(volume_MWh, rel=1e-6)
    assert float(state.T_cell) == 25.0


def test_the_contract_keeps_the_reserve_for_soc_recovery():
    """NESO-SOE sec. 3: DR reserves 40 % of the offered quantity, so contract
    plus reserve is the unit's rating, and the baseline spans the reserve."""
    contracted = float(battery.contracted_power(PARAMS))
    assert contracted * 1.4 == pytest.approx(PARAMS.power_max, rel=1e-6)
    assert float(battery.baseline_limit(PARAMS)) == pytest.approx(0.4 * contracted)


def test_a_baseline_is_submitted_past_gate_closure_only():
    """At t = 0 the periods starting at 0, 30 and 60 min are closed; the action
    lands on the period starting at 90 min."""
    _, state = _reset()
    _, state, _, _, _ = _step(state, 1.0)
    limit = float(battery.baseline_limit(PARAMS))
    assert [float(b) for b in state.baselines[:4]] == pytest.approx(
        [0.0, 0.0, 0.0, limit]
    )


def test_the_observation_shows_the_locked_baselines():
    _, state = _reset()
    baselines = state.baselines.at[0].set(1.0e5).at[1].set(2.0e5).at[2].set(3.0e5)
    obs = ENV.get_obs(state.replace(baselines=baselines), PARAMS)
    assert [float(o) for o in obs[6:9]] == pytest.approx([0.1, 0.2, 0.3])
    assert float(obs[11]) == pytest.approx(PARAMS.period_seconds)


def test_the_plant_delivers_the_baseline_plus_the_response():
    """No under-delivery inside the protected limits (NESO-ST para. 6.13)."""
    _, state = _reset()
    state = state.replace(baselines=state.baselines.at[0].set(3.0e5))
    target = 3.0e5 + float(state.regulation)
    state = state.replace(target_power=jnp.float32(target))
    _, after, reward, _, _ = _step(state, -1.0)
    assert float(after.power) == pytest.approx(target, rel=1e-6)
    assert float(reward) == 0.0


@pytest.mark.parametrize("soc", [0.0, 1.0])
def test_an_empty_or_full_pack_never_trips(soc):
    _, state = _reset()
    state = state.replace(soc=jnp.float32(soc))
    assert not bool(check_is_terminal(state, PARAMS)[0])
    _, after, _, terminated, info = _step(state)
    assert not bool(terminated) and not bool(info["tripped"])


@pytest.mark.parametrize("T", [-30.0, 70.0])
def test_cell_temperature_out_of_range_is_protected_not_a_trip(T):
    _, state = _reset()
    state = state.replace(T_cell=jnp.float32(T))
    _, after, _, terminated, info = _step(state)
    assert not bool(terminated) and not bool(info["tripped"])
    assert float(after.power) == 0.0


def _short(state, seconds):
    """``seconds`` of steps from an empty pack asked to discharge."""
    state = state.replace(
        soc=jnp.float32(0.0),
        soc_error=jnp.float32(0.0),
        baselines=state.baselines.at[0].set(5.0e5),
        target_power=jnp.float32(5.0e5),
    )
    costs = []
    for _ in range(seconds):
        _, state, reward, _, _ = _step(state, 0.0, PARAMS.replace(signal_std=0.0))
        state = state.replace(soc=jnp.float32(0.0), target_power=jnp.float32(5.0e5))
        costs.append(-float(reward))
    return costs


def test_one_second_short_of_a_period_is_not_unavailability():
    """0.1 % of 1800 s is 1.8 s: one 1 s step short costs nothing."""
    _, state = _reset()
    assert sum(_short(state.replace(signal_x=jnp.float32(0.0)), 1)) == 0.0


def test_two_seconds_short_lose_the_period_once():
    """The second short second reaches 0.1 %: the period's whole availability,
    contracted MW times 0.5 h, charged once."""
    _, state = _reset()
    costs = _short(state.replace(signal_x=jnp.float32(0.0)), 5)
    lost = float(battery.contracted_power(PARAMS)) / 1e6 * 0.5
    assert costs == pytest.approx([0.0, lost, 0.0, 0.0, 0.0])


def test_failing_the_energy_check_at_a_period_start_loses_that_period():
    """NESO-ST para. 6.12: below the requirement at a period's start, the unit
    is deemed unavailable for it."""
    params = PARAMS.replace(signal_std=0.0)
    _, state = _reset(0, params)
    state = state.replace(
        time=PER_PERIOD - 1,
        soc=jnp.float32(0.05),
        soc_error=jnp.float32(0.0),
        signal_x=jnp.float32(0.0),
        regulation=jnp.float32(0.0),
        target_power=jnp.float32(0.0),
    )
    obs, state, reward, terminated, _ = _step(state, 0.0, params)
    lost = float(battery.contracted_power(PARAMS)) / 1e6 * 0.5
    assert -float(reward) == pytest.approx(lost)
    assert float(obs[12]) == 1.0 and not bool(terminated)
    _, _, reward, _, _ = _step(state, 0.0, params)
    assert float(reward) == 0.0  # charged once per period


def test_the_score_has_no_tracking_term():
    _, state = _reset()
    terms = ENV.reward_terms(state, PARAMS)
    assert "tracking" not in terms and "availability" in terms
