"""Behaviour tests for the grid battery against real plants (charter step 3).

Each test is one of the 13 proposed tests in ``COLD_REVIEW.md``, numbered as
there. Its source, the quantity and the tolerance are the audit's, written
before any run. The sources are in ``REALITY.md``.

Plant tests drive the module's public physics (``plant_step``, ``cell_ocv``)
at the task's step. Episode tests drive the environment with the baseline held
at zero, so they read the plant and the signal, not a controller.
"""

import importlib.util
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.energy.battery import env as battery
from target_gym.energy.battery.env import BatteryParams
from target_gym.energy.battery.env_jax import GridBattery

PARAMS = BatteryParams(sensor_noise_scale=0.0)
ENV = GridBattery()
_PJM = pathlib.Path(__file__).parents[3] / "scripts" / "pjm_regulation_score.py"


def _drive(soc, T, power_W, seconds, params=PARAMS):
    """Constant ``power_W`` at the meter through ``plant_step`` for ``seconds``:
    per step, the state of charge after it, the delivered power and the cell
    terminal voltage."""
    steps = int(round(seconds / params.delta_t))

    def body(s, _):
        s1, delivered, _, v = battery.plant_step(s, T, power_W, params)
        return s1, (s1, delivered, v)

    _, out = jax.lax.scan(body, jnp.float32(soc), None, length=steps)
    return [np.asarray(o, np.float64) for o in out]


def _cycle_energy(params=PARAMS):
    """AC energy in and out, J, over test 1's cycle: full charge from empty to
    the stop, 10 min rest, full discharge to the stop, at 25 degC."""
    hours = 4.0 * 3600.0
    socs, p_in, _ = _drive(0.0, 25.0, -params.power_max, hours, params)
    rest, _, _ = _drive(socs[-1], 25.0, 0.0, 600.0, params)
    _, p_out, _ = _drive(rest[-1], 25.0, params.power_max, hours, params)
    dt = params.delta_t
    return -p_in.sum() * dt, p_out.sum() * dt


def _episode(seeds, steps, params=PARAMS):
    """Roll the env with the baseline action held at 0 (raw 0 is a zero
    baseline). Returns per seed and step: true SoC, observation, delivered and
    required power."""
    params = params.replace(max_steps_in_episode=steps + 1)

    def one(seed):
        key = jax.random.PRNGKey(seed)
        obs0, state = ENV.reset_env(key, params)

        def body(s, _):
            obs, s1, _, _, _ = ENV.step_env(key, s, jnp.zeros(1), params)
            return s1, (s1.soc, obs, s1.power, s.target_power)

        _, out = jax.lax.scan(body, state, None, length=steps)
        return state.soc, obs0, out

    return jax.jit(jax.vmap(one))(jnp.asarray(seeds))


def test_01_round_trip_efficiency_is_in_the_sourced_span():
    """NREL-ATB 85 %, Megapack 92.0 to 93.7 % (REALITY.md sec. 3): AC out over
    AC in in [84 %, 94.7 %]."""
    e_in, e_out = _cycle_energy()
    assert 0.84 <= e_out / e_in <= 0.947


def test_02_lfp_plateau_is_flat():
    """ARX-LFP eq. 5: 1 mV is nearly 5 % SoC. Mean dOCV/dSoC per cell over SoC
    0.2 to 0.8 in [0.1, 0.4] mV per % SoC."""
    ocv = lambda s: battery.cell_ocv(s, PARAMS)  # noqa: E731
    slope_V_per_unit = (float(ocv(0.8)) - float(ocv(0.2))) / 0.6
    mV_per_percent = slope_V_per_unit * 1e3 / 100.0
    assert 0.1 <= mV_per_percent <= 0.4


def test_03_the_pack_stops_at_the_cells_voltage_window():
    """EVE-LF280K: 2.5 to 3.65 V. 0.5C charge to the stop, then 0.5C discharge
    to the stop; the terminal voltage at each stop within 0.05 V."""
    half_c = 0.5 * PARAMS.energy_capacity / 3600.0  # W, half the capacity per hour
    socs, p, v = _drive(0.5, 25.0, -half_c, 3.0 * 3600.0)
    assert abs(p[-1]) < 1e-3 * half_c  # it has stopped
    assert v.max() == pytest.approx(3.65, abs=0.05)
    socs, p, v = _drive(socs[-1], 25.0, half_c, 3.0 * 3600.0)
    assert abs(p[-1]) < 1e-3 * half_c
    assert v.min() == pytest.approx(2.5, abs=0.05)


def test_04_cold_cells_refuse_charge_and_accept_discharge():
    """EVE-LF280K item 8: charge 0 to 55 degC, discharge -20 to 55 degC. At
    -5 degC a full charge is refused and a full discharge accepted, exactly."""
    _, charged, _, _ = battery.plant_step(0.5, -5.0, -PARAMS.power_max, PARAMS)
    _, discharged, _, _ = battery.plant_step(0.5, -5.0, PARAMS.power_max, PARAMS)
    assert float(charged) == 0.0
    assert float(discharged) == pytest.approx(PARAMS.power_max, rel=1e-6)


def test_05a_the_reported_soc_is_not_the_true_soc():
    """MPS-ACCURE: BMS errors up to +-15 %, read as 2 sigma, a factor 3 each
    way: the sd of observed minus true SoC over 400 resets in [2.5 %, 22.5 %]."""

    def gap(seed):
        obs, state = ENV.reset_env(jax.random.PRNGKey(seed), PARAMS)
        return obs[0] - state.soc

    gaps = np.asarray(jax.vmap(gap)(jnp.arange(400)))
    assert 0.025 <= gaps.std() <= 0.225


def test_05b_heating_the_cells_moves_the_reported_soc():
    """ARX-LFP Fig. 5: about +0.51 % per degC on the 25 degC curve. At rest on
    the plateau, 10 K more moves the gap by 5.1 % +- 50 %. One step at rest
    each way, so the BMS reports from the cells' temperature."""
    params = PARAMS.replace(signal_std=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    rest = state.replace(
        soc=jnp.float32(0.5),
        signal_x=jnp.float32(0.0),
        regulation=jnp.float32(0.0),
        target_power=jnp.float32(0.0),
    )

    def gap(s):
        obs, after, _, _, _ = ENV.step_env(
            jax.random.PRNGKey(1), s, jnp.zeros(1), params
        )
        return obs[0] - after.soc

    cool = gap(rest)
    warm = gap(rest.replace(T_cell=rest.T_cell + 10.0))
    assert float(warm - cool) == pytest.approx(0.051, rel=0.5)


@pytest.mark.medium
def test_06_ah_counting_drift_does_not_shrink():
    """ARX-LFP sec. I, MPS-ACCURE: coulomb counting cannot correct an initial
    error. Over 100 seeds of 4 h on the task's signal, the mean |gap| at the
    end is no smaller than at the start, and there is a gap to keep."""
    steps = int(4 * 3600 / PARAMS.delta_t)
    soc0, obs0, (soc, obs, _, _) = _episode(np.arange(100), steps)
    start = np.abs(np.asarray(obs0[:, 0]) - np.asarray(soc0))
    end = np.abs(np.asarray(obs[:, -1, 0]) - np.asarray(soc[:, -1]))
    assert start.mean() > 0.0
    assert end.mean() >= start.mean()


@pytest.mark.medium
def test_07_the_regulation_signal_is_energy_neutral_over_a_day():
    """PNNL-ESS App. B: 24 h signals energy neutral. |integral of the target|
    over the integral of |target| at most 5 % per seed over 24 h, baseline 0."""
    steps = int(24 * 3600 / PARAMS.delta_t)
    _, _, (_, _, _, target) = _episode(np.arange(10), steps)
    target = np.asarray(target, np.float64)
    ratio = np.abs(target.sum(axis=1)) / np.abs(target).sum(axis=1)
    assert np.all(ratio <= 0.05), ratio


def test_08_an_exact_follower_returns_to_its_start_less_the_losses():
    """PNNL-ESS sec. 5.3.2: the test ends at the initial SoC. Following an
    energy-neutral +-1 p.u. cycle exactly, the drop equals (1 - RTE) / 2 times
    throughput over capacity, RTE from test 1, +-20 %. The cycle is four
    15 min legs at -1, +1, +1, -1 p.u. of rated power."""
    e_in, e_out = _cycle_energy()
    rte = e_out / e_in
    soc = 0.5
    throughput = 0.0
    for sign in (-1.0, 1.0, 1.0, -1.0):
        socs, p, _ = _drive(soc, 25.0, sign * PARAMS.power_max, 900.0)
        soc = socs[-1]
        throughput += np.abs(p).sum() * PARAMS.delta_t
    drop = 0.5 - soc
    expected = (1.0 - rte) / 2.0 * throughput / PARAMS.energy_capacity
    assert drop == pytest.approx(expected, rel=0.2)


def test_09_leaving_the_energy_rule_is_priced_and_timed_not_a_trip():
    """NESO-SOE Table 2, NESO-ST para. 6.11(iii), 6.12, Schedule 3. A unit
    exports 2/50 of its contracted volume by response in period 1 and holds:
    the next period requires 48/50 of it, the one after the whole volume
    (recovery up to 20 % per period), and nothing resets. Exact."""
    params = PARAMS.replace(signal_std=0.0)
    _, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    per_period = int(round(params.period_seconds / params.delta_t))
    volume = battery.response_volume(params)  # J
    quiet = dict(signal_x=jnp.float32(0.0), regulation=jnp.float32(0.0))
    state = state.replace(
        time=per_period - 1,
        period_response=jnp.float32(2.0 / 50.0 * volume),
        target_power=jnp.float32(0.0),
        **quiet,
    )
    key = jax.random.PRNGKey(1)
    obs, state, _, terminated, _ = ENV.step_env(key, state, jnp.zeros(1), params)
    assert not bool(terminated)
    assert float(obs[9]) * 3.6e9 == pytest.approx(48.0 / 50.0 * volume, rel=1e-5)
    state = state.replace(time=2 * per_period - 1, **quiet)
    obs, state, _, _, _ = ENV.step_env(key, state, jnp.zeros(1), params)
    assert float(obs[9]) * 3.6e9 == pytest.approx(volume, rel=1e-5)

    # A deficit larger than one period's recovery comes back by 20 % a period.
    _, state = ENV.reset_env(jax.random.PRNGKey(0), params)
    state = state.replace(
        time=per_period - 1,
        period_response=jnp.float32(0.3 * volume),
        target_power=jnp.float32(0.0),
        **quiet,
    )
    obs, state, _, _, _ = ENV.step_env(key, state, jnp.zeros(1), params)
    assert float(obs[9]) * 3.6e9 == pytest.approx(0.7 * volume, rel=1e-5)
    state = state.replace(time=2 * per_period - 1, **quiet)
    obs, state, _, _, _ = ENV.step_env(key, state, jnp.zeros(1), params)
    assert float(obs[9]) * 3.6e9 == pytest.approx(0.9 * volume, rel=1e-5)


def test_10_the_step_resolves_the_service():
    """PNNL-ESS Foreword: data every second for regulation. delta_t <= 1 s."""
    assert PARAMS.delta_t <= 1.0


def test_11_contingency_droop_answers_a_frequency_step():
    """AUR-Y1, AEMO-HPR sec. 2.2: deadband 50 +- 0.15 Hz, 1.72 % droop, about
    100 ms. A step from 50.00 to 49.77 Hz with regulation held gives
    (49.85 - 49.77) / 50 / 0.0172 = 0.093 p.u. more power, +-10 %, by the first
    sample. The task must expose a grid frequency to run this."""
    _, state = ENV.reset_env(jax.random.PRNGKey(0), PARAMS)
    assert hasattr(state, "grid_frequency"), "the task has no grid frequency"
    key = jax.random.PRNGKey(1)
    _, held, _, _, _ = ENV.step_env(key, state, jnp.zeros(1), PARAMS)
    _, dipped, _, _, _ = ENV.step_env(
        key, state.replace(grid_frequency=jnp.float32(49.77)), jnp.zeros(1), PARAMS
    )
    extra = float(dipped.power - held.power) / PARAMS.power_max
    assert extra == pytest.approx(0.093, rel=0.1)


def test_12_cycle_life_is_at_least_the_datasheets():
    """EVE-LF280K item 10: 6000 cycles or more at 0.5C/0.5C to 80 % capacity.
    Over 50 full 0.5C cycles the energy a full discharge delivers falls by at
    most 0.2 / 6000 = 3.33e-5 per cycle (one-sided)."""
    half_c = 0.5 * PARAMS.energy_capacity / 3600.0
    soc, delivered = 0.0, []
    for _ in range(50):
        socs, _, _ = _drive(soc, 25.0, -half_c, 2.5 * 3600.0)
        socs, p, _ = _drive(socs[-1], 25.0, half_c, 2.5 * 3600.0)
        soc = socs[-1]
        delivered.append(p.sum() * PARAMS.delta_t)
    fade_per_cycle = (delivered[0] - delivered[-1]) / delivered[0] / 49.0
    assert fade_per_cycle <= 0.2 / 6000.0


def test_13_a_normal_day_scores_above_pjms_qualification_line():
    """AEMO-HPR Fig. 2, PJM-M12 4.4.1: the PJM performance score with 10 s
    latency, over 40 min from mid SoC on the energy-neutral signal, at least
    0.75. AReg is the contracted regulation power."""
    spec = importlib.util.spec_from_file_location("pjm_regulation_score", _PJM)
    pjm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pjm)
    steps = int(40 * 60 / PARAMS.delta_t)
    _, _, (_, _, power, target) = _episode([0], steps)
    areg = float(battery.contracted_power(PARAMS))
    score = pjm.score(
        np.asarray(power[0]), np.asarray(target[0]), areg, PARAMS.delta_t, skip_s=0.0
    )
    assert score >= 0.75
