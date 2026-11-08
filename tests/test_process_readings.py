"""The process family's readings, tested against the Process table of
docs/proposals/measurement-noise.md.

Contract, taken from the proposal and not from the code:

- Each channel below scatters by the table's sigma, in the observation's units,
  with rounding adding ``step^2 / 12`` to the variance where the table gives a
  step.
- Setpoints and the controller's own commands stay exact.
- Derived channels inherit the noise of what they are computed from:
  ``T_15`` from ``x[14]`` and the delivered flow from ``dp`` and ``demand_dev``.
- The analysers hold their noise for a cycle while their value stays current.
- ``first_order``'s reading is a convention on an abstract lag, so only its
  figure is tested, not an instrument.
- A cost on the measured copy has the proposal's floor of
  ``(sigma / e_floor)^2`` per step.

``tests/test_measurement_noise.py`` holds these tasks to the shared contract
(same noise for every policy, scale 0 is the noise-free task, targets exact).
This file holds what is particular to each task.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.base import measured_state
from target_gym.compressor_surge.env import KPA, consumer_flow, demand_opening
from target_gym.pc_gym.distillation.env import tray_temperature
from target_gym.registry import REGISTRY

#: Independent draws per estimate. The standard deviation of n normal draws has
#: a relative standard error of 1 / sqrt(2 n), 0.8% here.
DRAWS = 4000
#: Allowed standard errors, in every statistical check below.
SE = 4.0

#: Distillation's boiling points differ by 13.5 K (Skogestad's column A), so
#: 3e-3 mole fraction on x[14] is the RTD's 0.04 K.
TRAY_SLOPE = 13.5
#: The analyser cycle, in the column's 1-minute steps (PHYSICS.md, Readings).
CYCLE_STEPS = 5


def _with_rounding(sigma, step):
    """Uniform quantisation adds ``step^2 / 12`` to the variance (Widrow), and
    both rounded readings here have sigma above step / 2, where that holds."""
    return np.sqrt(sigma**2 + step**2 / 12.0)


#: Channel -> its scatter, in the observation's units.
SCATTER = {
    "cstr": {0: 5e-4, 1: 0.04},
    "first_order": {0: 1.1e-3},
    "unstable_cstr": {0: 5e-4, 1: 0.04, 2: 0.04},
    "distillation": {0: 4e-4, 1: 4e-4, 6: 3e-3 * TRAY_SLOPE},
    "ph_neutralization": {
        0: _with_rounding(0.01, 0.01),
        4: _with_rounding(0.01, 0.01),
    },
    "compressor_surge": {
        0: 9.17 / KPA,
        1: 0.021,
        # In percent. 1 rpm of rated, at the source's assumed 10,000 rpm.
        3: 100.0 * _with_rounding(5.8e-5, 1e-4),
        5: 100.0 * 1.5e-3,
    },
}
#: Setpoints and the controller's own commands.
EXACT = {
    "cstr": (2,),
    "first_order": (1,),
    "unstable_cstr": (3,),
    # Flows, specs, and the feed meter (FC 1), read exactly: ours.
    "distillation": (2, 3, 4, 5, 7),
    # Valves, set point, the monitor's excursion clock, the flow meter and the
    # basin level, read exactly (ours: no figures found).
    "ph_neutralization": (1, 2, 3, 5, 6, 7),
    # Drive setpoint, header setpoint, and the surge counter (ours: exact).
    "compressor_surge": (4, 6, 7),
}
#: Channels with a scatter of their own arithmetic, tested on their own below.
DERIVED = {"compressor_surge": (2,)}
#: Steps between draws of a channel, where that is not every step.
PERIOD = {"distillation": {0: CYCLE_STEPS, 1: CYCLE_STEPS}}

#: A copy of the state with the target exactly on the tracked value, so a
#: tracking cost on the measured copy is noise alone.
ON_TARGET = {
    "cstr": lambda s: s.replace(target_CA=s.C_a),
    "first_order": lambda s: s.replace(target_x=s.x),
    "unstable_cstr": lambda s: s.replace(
        target_levels=jnp.full_like(s.target_levels, s.C_a)
    ),
}
#: The proposal's floor per step, with rounding's ``step^2 / 12`` on pH. The
#: table's 0.11 and 0.034 are (9.17 Pa / 27.5 Pa)^2 and (1.1e-3 / 6e-3)^2.
FLOOR = {
    "cstr": (5e-4 / 5e-4) ** 2,
    "first_order": (1.1e-3 / 6e-3) ** 2,
    "unstable_cstr": (5e-4 / 5e-4) ** 2,
}
#: Independent tracked readings behind each floor: a sum of k equal chi-squared
#: terms has standard deviation floor * sqrt(2 / k).
TRACKED = {"distillation": 2}

tasks = pytest.mark.parametrize("name", sorted(SCATTER))


def _reset(name):
    spec = REGISTRY[name]
    env, params = spec.make_env(), spec.make_test_params()
    state = env.reset_env(jax.random.PRNGKey(0), params)[1]
    return env, params, state


def _residuals(env, state, params, times):
    """``reading - true value`` of each channel, one row per ``time``."""
    quiet = env.get_obs(state, params.replace(sensor_noise_scale=0.0))

    def one(t):
        return env.get_obs(state.replace(time=t), params) - quiet

    return np.asarray(jax.vmap(one)(jnp.asarray(times, jnp.int32)))


@pytest.fixture(scope="module")
def scatter():
    """Each task's residuals over ``DRAWS`` updates of its slowest reading."""
    out = {}

    def get(name):
        if name not in out:
            env, params, state = _reset(name)
            stride = max(PERIOD.get(name, {}).values(), default=1)
            out[name] = _residuals(env, state, params, stride * np.arange(DRAWS))
        return out[name]

    return get


@tasks
def test_each_reading_scatters_by_the_proposals_sigma(name, scatter):
    residuals = scatter(name)
    for channel, sigma in SCATTER[name].items():
        assert residuals[:, channel].std() == pytest.approx(
            sigma, rel=SE / np.sqrt(2 * DRAWS)
        ), f"channel {channel}"


@tasks
def test_noise_has_no_bias(name, scatter):
    residuals = scatter(name)
    for channel, sigma in SCATTER[name].items():
        assert abs(residuals[:, channel].mean()) < SE * sigma / np.sqrt(DRAWS)


@tasks
def test_setpoints_and_the_controllers_own_commands_stay_exact(name, scatter):
    residuals = scatter(name)
    for channel in EXACT[name]:
        np.testing.assert_array_equal(residuals[:, channel], 0.0)


@tasks
def test_every_channel_is_either_noisy_or_exact(name):
    """A channel added later must take a side."""
    env = REGISTRY[name].make_env()
    named = {*SCATTER[name], *EXACT[name], *DERIVED.get(name, ())}
    assert sorted(named) == list(range(env.obs_shape[0]))


# Tasks scored on the operator's job (charter rule 15) have no tracking term.
@pytest.mark.parametrize("name", sorted(FLOOR))
def test_measured_cost_carries_the_proposals_floor(name):
    env, params, state = _reset(name)
    state = ON_TARGET[name](state)
    stride = max(PERIOD.get(name, {}).values(), default=1)

    def tracking(t):
        measured = measured_state(env, state.replace(time=t), params)
        return env.reward_terms(measured, params, tripped=jnp.array(False))["tracking"]

    costs = jax.vmap(tracking)(stride * jnp.arange(DRAWS, dtype=jnp.int32))
    spread = FLOOR[name] * np.sqrt(2.0 / TRACKED.get(name, 1))
    assert float(costs.mean()) == pytest.approx(
        FLOOR[name], abs=SE * spread / np.sqrt(DRAWS)
    )


# ---------------------------------------------------------------------------
# distillation
# ---------------------------------------------------------------------------


def test_analysers_hold_their_noise_through_a_cycle_and_redraw_at_the_next():
    env, params, state = _reset("distillation")
    residuals = _residuals(env, state, params, np.arange(2 * CYCLE_STEPS))
    for channel in (0, 1):  # yD, xB
        first, second = np.split(residuals[:, channel], 2)
        np.testing.assert_array_equal(first, first[0])
        np.testing.assert_array_equal(second, second[0])
        assert first[0] != second[0]


def test_a_held_analyser_reports_no_change_within_a_cycle():
    """A chromatograph reports once a cycle, so a change in the product within
    it does not reach the reading. Reversed from the proposal's first form, which
    held only the noise (docs/proposals/industrial-fixes.md)."""
    env, params, state = _reset("distillation")
    moved = state.replace(time=state.time + 1, x=state.x.at[-1].add(1e-3))
    assert float(env.get_obs(moved, params)[0]) == float(env.get_obs(state, params)[0])


def test_the_tray_temperature_updates_every_step():
    env, params, state = _reset("distillation")
    residuals = _residuals(env, state, params, np.arange(CYCLE_STEPS))
    assert len(set(residuals[:, 6].tolist())) == CYCLE_STEPS


def test_the_two_analysers_scatter_independently():
    env, params, state = _reset("distillation")
    residuals = _residuals(env, state, params, CYCLE_STEPS * np.arange(DRAWS))
    assert abs(np.corrcoef(residuals[:, 0], residuals[:, 1])[0, 1]) < SE / np.sqrt(
        DRAWS
    )


def test_the_tray_temperature_is_computed_from_the_measured_x14():
    env, params, state = _reset("distillation")
    state = state.replace(time=7)
    measured = measured_state(env, state, params)
    assert float(env.get_obs(state, params)[6]) == pytest.approx(
        float(tray_temperature(measured)), abs=1e-4
    )
    assert float(measured.x[14]) != float(state.x[14])


def test_stages_with_no_instrument_stay_exact():
    env, params, state = _reset("distillation")
    measured = measured_state(env, state.replace(time=7), params)
    interior = np.setdiff1d(np.arange(state.x.size), [0, 14, state.x.size - 1])
    np.testing.assert_array_equal(measured.x[interior], state.x[interior])


# ---------------------------------------------------------------------------
# compressor_surge
# ---------------------------------------------------------------------------


def _at_nominal_opening(state):
    return state.replace(
        demand_levels=jnp.full_like(state.demand_levels, 0.6),
        demand_dev=jnp.zeros_like(state.demand_dev),
        block_clock=jnp.zeros_like(state.block_clock),
    )


def test_delivered_flow_is_computed_from_the_measured_dp_and_opening():
    env, params, state = _reset("compressor_surge")
    state = state.replace(time=7)
    measured = measured_state(env, state, params)
    opening = demand_opening(
        measured.demand_levels,
        measured.block_clock * params.delta_t,
        measured.demand_dev,
        params,
    )
    assert float(env.get_obs(state, params)[2]) == pytest.approx(
        float(consumer_flow(measured.dp, opening, params)), rel=1e-5
    )
    assert float(measured.demand_dev) != float(state.demand_dev)


def test_delivered_flow_scatters_like_its_meter():
    """The source's figures: 0.14% of flow through the opening at the nominal
    0.6, from sigma 8.4e-4, and 0.03% from dp, which add in quadrature."""
    env, params, state = _reset("compressor_surge")
    state = _at_nominal_opening(state)
    quiet = env.get_obs(state, params.replace(sensor_noise_scale=0.0))
    flow = _residuals(env, state, params, np.arange(DRAWS))[:, 2]
    dp_part = 0.5 * (9.17 / (float(state.dp) - params.dp_out))
    expected = float(quiet[2]) * np.hypot(8.4e-4 / 0.6, dp_part)
    assert flow.std() == pytest.approx(expected, rel=SE / np.sqrt(2 * DRAWS))
    assert float(quiet[2]) > 0.0


def test_field_noise_does_not_depend_on_the_plant():
    """One episode key, one noise path: two plants at one update carry the same
    draw on each field. The delivered flow's own residual scales with the flow,
    so this is checked on the fields behind it."""
    env, params, state = _reset("compressor_surge")
    state = state.replace(time=7)
    other = state.replace(dp=state.dp + 1000.0, m_c=state.m_c + 1.0)
    a, b = measured_state(env, state, params), measured_state(env, other, params)
    # Four float32 steps of a value near 3e4 Pa, the subtraction's rounding.
    assert float(a.dp - state.dp) == pytest.approx(float(b.dp - other.dp), abs=0.016)
    assert float(a.m_c - state.m_c) == pytest.approx(float(b.m_c - other.m_c), abs=4e-6)
    assert float(a.demand_dev) == float(b.demand_dev)


def test_action_noise_reaches_the_actuators_inside_their_rate_limit():
    """A command at the actuators, so neither rate limit binds, and the
    positioner's scatter moves them. Far from them both move at their limit
    whatever the command, so a step shows nothing."""
    env, params, state = _reset("compressor_surge")
    span = params.N_max - params.N_min
    action = jnp.array(
        [2.0 * (float(state.N_ramp) - params.N_min) / span - 1.0, 2.0 * state.x - 1.0],
        jnp.float32,
    )
    key = jax.random.PRNGKey(0)
    noisy = env.step_env(key, state, action, params.replace(action_noise=0.05))[1]
    clean = env.step_env(key, state, action, params)[1]
    assert float(noisy.N_ramp) != float(clean.N_ramp)
    assert float(noisy.x) != float(clean.x)


def test_the_surge_trip_stays_true_under_noise():
    """The hidden trip variable has no instrument, so ``phi_min`` keeps its
    true value on the measured copy."""
    env, params, state = _reset("compressor_surge")
    measured = measured_state(env, state.replace(time=7), params)
    assert float(measured.phi_min) == float(state.phi_min)
    assert float(measured.m_c) != float(state.m_c)
