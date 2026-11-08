"""The industrial and energy readings, against the proposal's table.

Contract, from docs/proposals/measurement-noise.md and not from the code:

- Each instrument in the "Industrial and energy" table reaches its observation
  channel at its sigma, unbiased. Setpoints, schedules, the clock and the
  controller's own commands stay exact.
- Derived channels inherit their field's noise: ``fuel_pct``, ``heat_pct``,
  ``omega_rpm`` and ``torque_pct`` are unit changes of what is measured.
- Wind electrical power, battery cell voltage and solar gain are computed, and
  each is its own instrument: none inherits the noise of what it is computed
  from. ``cement_kiln`` reads ``max(T_solid)`` once, after the max.
- Free lime is a lab sample: its noise holds between samples, while the value
  stays current.
- ``reward``, trips and the dynamics read the true state, at any noise scale.
- ``info["measured_reward"]`` has a floor of ``(sigma / e_floor)^2`` per step
  and carries the draw the observation carried.

Assumptions the proposal leaves open, stated here and in each ``PHYSICS.md``:

- ``reactor`` flux is the table's 1.9e-4 at every power. The source says at
  most, proportional to ``n``, and a reading's sigma is a constant, so this is
  an upper bound over the operating range.
- A free-lime sample is hourly. No standard or handbook gives this.
- The measured reward of ``reactor``, summed over its six sub-steps, carries
  the one draw of the step's observation: a controller gets one reading a step.

``tests/test_measurement_noise.py`` holds these tasks to the shared contract:
a reading is pure in the state, targets stay exact, scale 0 is the noise-free
task, noise runs under jit, vmap and scan.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import reward as R
from target_gym.base import measured_reward, measured_state
from target_gym.registry import REGISTRY

#: Channel to sigma in the observation's units, by hand from the table. Channels
#: not listed stay exact, apart from the held ones in ``HELD``.
SCATTER = {
    "glass_furnace": {0: 0.36, 1: 0.32, 2: 100.0 * 3e-4 / 0.698},  # fuel_max
    # Tavg is the mean of two RTDs, so 0.045 over sqrt(2).
    "reactor": {0: 1.9e-4, 1: 0.045, 4: 0.045 / math.sqrt(2.0)},
    "hvac": {
        0: 0.023,
        1: 0.023,
        2: 100.0 * 27.0 / 7200.0,  # Q_heat_max
        3: 1.0 / 175.0,  # 1 W/m2 of solar_peak
    },
    "boiler_drum": {0: 5e-4, 1: 0.022, 2: 0.047},
    "cement_kiln": {1: 1.6, 2: 0.33, 3: 1.2, 4: 0.13},
    "wind_turbine": {
        0: 5.2e-4 * 60.0 / (2.0 * math.pi),  # rad/s to rpm
        1: 0.2,
        2: 100.0 * 90.0 / 47_400.0,  # torque_max
        3: 1e-3,  # 1 kW in MW
    },
    "battery": {0: 4.7e-5, 1: 5.5e-5, 2: 0.03, 3: 3.3e-4},
}
TASKS = list(SCATTER)
#: Channels computed from one instrument share its draw: Tavg is half the hot-leg
#: RTD, so it correlates with that channel at 1 / sqrt(2).
SHARED_INSTRUMENT = {("reactor", 1, 4): 1.0 / math.sqrt(2.0)}
#: Free lime in percent: sigma, and steps of 30 s in one hourly sample.
HELD = {"cement_kiln": {0: (100.0 * 1.8e-4, 120)}}

#: (task, channel) to digital step in the observation's units.
STEPS = {
    ("hvac", 0): 0.1,
    ("hvac", 1): 0.1,
    ("cement_kiln", 0): 0.01,  # 1e-4 mass fraction, in percent
    ("cement_kiln", 1): 0.1,
    ("cement_kiln", 3): 0.1,
    ("battery", 2): 152.59e-6 / 0.05,  # GPIO step over the 50 mV/K divider
}

#: A state outside each task's envelope.
OUTSIDE = {
    "glass_furnace": lambda s: s.replace(T_crown=jnp.float32(2000.0)),
    "reactor": lambda s: s.replace(n=jnp.float32(2.0)),
    "hvac": lambda s: s.replace(T_air=jnp.float32(50.0)),
    # Past the high trip for 12 s, more than its 10 s delay (boiler_drum/env.py).
    "boiler_drum": lambda s: s.replace(
        level=jnp.float32(1.0), time_high=jnp.float32(12.0)
    ),
    "cement_kiln": lambda s: s.replace(T_solid=s.T_solid + 1000.0),
    "wind_turbine": lambda s: s.replace(omega=3.0 * s.omega),
    "battery": lambda s: s.replace(soc=jnp.float32(1.5)),
}

DRAWS = 2000


def _setup(name, unrounded=True):
    """Rounding hides ``epsilon``, so the scatter is read with no digital step."""
    spec = REGISTRY[name]
    env, params = spec.env_factory(), spec.make_test_params()
    if unrounded:
        env.readings = tuple(dataclasses.replace(r, step=0.0) for r in env.readings)
    return env, params, env.reset_env(jax.random.PRNGKey(0), params)[1]


def _quiet(params):
    return params.replace(sensor_noise_scale=0.0)


def _noise(env, params, state, times):
    """Each channel's reading less its true value, at each of ``times``."""

    def at(t):
        s = state.replace(time=t)
        return env.get_obs(s, params) - env.get_obs(s, _quiet(params))

    return np.asarray(jax.jit(jax.vmap(at))(times), np.float64)


def _action(env, params):
    space = env.action_space(params)
    return jnp.broadcast_to(0.5 * (space.low + space.high), space.shape).astype(
        jnp.float32
    )


# ---------------------------------------------------------------------------
# What reaches the observation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TASKS)
def test_each_instrument_reaches_its_channel_at_its_figure_and_nowhere_else(name):
    """2000 draws put a sample standard deviation within ``4 / sqrt(2 * 2000)``,
    9%, of sigma, and the mean within 4 standard errors of 0."""
    env, params, state = _setup(name)
    noise = _noise(env, params, state, jnp.arange(DRAWS))
    for channel, sigma in SCATTER[name].items():
        assert noise[:, channel].std() == pytest.approx(sigma, rel=0.09), channel
        assert abs(noise[:, channel].mean()) < 4 * sigma / math.sqrt(DRAWS), channel
    held = set(HELD.get(name, ()))
    for channel in set(range(noise.shape[1])) - set(SCATTER[name]) - held:
        assert np.all(noise[:, channel] == 0.0), channel


@pytest.mark.parametrize("name", TASKS)
def test_each_reading_is_its_own_draw(name):
    """Battery voltage must not inherit the state of charge's noise, nor wind
    power the torque's: either would correlate two channels near 0.7. Over 2000
    draws an independent pair has a sample correlation of sd 0.022."""
    env, params, state = _setup(name)
    noise = _noise(env, params, state, jnp.arange(DRAWS))
    channels = sorted(SCATTER[name])
    for i, a in enumerate(channels):
        for b in channels[i + 1 :]:
            correlation = np.corrcoef(noise[:, a], noise[:, b])[0, 1]
            shared = SHARED_INSTRUMENT.get((name, a, b), 0.0)
            assert abs(correlation - shared) < 0.1, (a, b)


@pytest.mark.parametrize("task, channel", sorted(STEPS), ids=str)
def test_readings_round_to_the_instruments_step(task, channel):
    env, params, state = _setup(task, unrounded=False)
    read = jax.jit(jax.vmap(lambda t: env.get_obs(state.replace(time=t), params)))
    # A sample apart: the free-lime channel draws once in 120 steps.
    values = np.asarray(read(120 * jnp.arange(50))[:, channel])
    step = STEPS[(task, channel)]
    units = values.astype(np.float64) / step
    tolerance = 3 * np.spacing(np.abs(values)).astype(np.float64) / step
    assert np.all(np.abs(units - np.round(units)) <= tolerance)


def test_the_burning_zone_reads_once_after_the_max():
    """The hot zones level. One draw on the whole profile leaves the max
    unbiased, where 15 separate draws lift it by about 1.74 sigma. The settled
    profile cannot show this: its peak stands 6 K clear of the next zone, and
    separate draws lift that by 0.005 sigma. Zone 0 sits out, having its own
    pyrometer."""
    env, params, state = _setup("cement_kiln")
    level = state.replace(T_solid=jnp.full(16, 1750.0).at[0].set(1100.0))
    noise = _noise(env, params, level, jnp.arange(DRAWS))
    assert abs(noise[:, 1].mean()) < 4 * 1.6 / math.sqrt(DRAWS)
    assert noise[:, 1].std() == pytest.approx(1.6, rel=0.09)


def test_free_lime_noise_holds_between_lab_samples():
    """An hourly sample at 30 s a step: 120 steps of one draw, then a new one,
    while the pyrometer on the same state draws every step."""
    env, params, state = _setup("cement_kiln")
    _, per_sample = HELD["cement_kiln"][0]
    noise = _noise(env, params, state, jnp.arange(3 * per_sample))
    blocks = noise[:, 0].reshape(3, per_sample)
    for block in blocks:
        np.testing.assert_allclose(block, block[0], atol=5e-7)  # 4 float32 steps
    assert np.all(np.abs(np.diff(blocks[:, 0])) > 1e-6)
    assert noise[:, 1].reshape(3, per_sample).std(axis=1).min() > 1.0


def test_free_lime_scatters_at_its_figure_across_samples():
    env, params, state = _setup("cement_kiln")
    sigma, per_sample = HELD["cement_kiln"][0]
    noise = _noise(env, params, state, per_sample * jnp.arange(DRAWS))
    assert noise[:, 0].std() == pytest.approx(sigma, rel=0.09)
    assert abs(noise[:, 0].mean()) < 4 * sigma / math.sqrt(DRAWS)


# ---------------------------------------------------------------------------
# What stays true
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TASKS)
def test_noise_never_reaches_the_plant_or_the_reward(name):
    """The whole state stays equal, whatever the noise scale. That covers
    ``glass_furnace``: a reading written back would enter ``T_crown_meas``, the
    lag's own state, and every later reading with it."""
    env, params, start = _setup(name, unrounded=False)
    key = jax.random.PRNGKey(1)
    action = _action(env, params)
    step = jax.jit(env.step_env)

    def run(p):
        state, out = start, []
        for t in range(6):
            _, state, reward, done, info = step(
                jax.random.fold_in(key, t), state, action, p
            )
            out.append((state, reward, done, info["tripped"]))
        return out

    loud = params.replace(sensor_noise_scale=1000.0)
    for a, b in zip(run(loud), run(_quiet(params))):
        jax.tree_util.tree_map(np.testing.assert_array_equal, a, b)


@pytest.mark.parametrize("name", TASKS)
def test_the_measured_trip_is_the_one_it_is_given(name):
    """A measured copy can sit outside the envelope when the plant is not, and
    the other way round, so the check must not run on it."""
    env, params, state = _setup(name)
    outside = OUTSIDE[name](state)
    assert bool(env.is_terminated(outside, params))
    assert not bool(env.is_terminated(state, params))

    missed = env.reward_terms(outside, params, tripped=jnp.bool_(False))
    assert float(missed["failure"]) == 0.0
    charged = env.reward_terms(state, params, tripped=jnp.bool_(True))
    assert float(charged["failure"]) == pytest.approx(float(R.trip_cost(params)))
    assert all(float(v) == 0.0 for k, v in charged.items() if k != "failure")


# ---------------------------------------------------------------------------
# The measured reward
# ---------------------------------------------------------------------------


def _floor(env, params, state, times):
    """Mean measured cost of a plant resting exactly on target."""

    def at(t):
        return -measured_reward(
            env, state.replace(time=t), jnp.bool_(False), jnp.float32(0.0), params
        )

    return float(jnp.mean(jax.jit(jax.vmap(at))(times)))


def test_glass_furnace_measured_reward_has_the_proposals_floor():
    """``(0.36 / 1)^2`` per step, on a chi-squared of one degree of freedom:
    sd ``sqrt(2) * 0.1296`` per draw, on top of the fuel's ``w * c / c_hold``,
    whose meter scatters by 0.05 % of it."""
    env, params, state = _setup("glass_furnace")
    resting = state.replace(
        T_crown=state.target_T_crown,
        T_crown_meas=state.target_T_crown,
        fuel_flow=jnp.float32(params.fuel_min),
    )
    floor = _floor(env, params, resting, jnp.arange(DRAWS))
    running = params.running_weight * params.fuel_min / params.c_hold
    assert floor == pytest.approx(
        0.1296 + running, abs=4 * 1.414 * 0.1296 / math.sqrt(DRAWS)
    )


def test_cement_kiln_measured_reward_has_the_proposals_floor():
    """``(1.8e-4 / 5e-4)^2`` per step. One draw per sample, so the steps are an
    hour apart. Zero fuel costs nothing."""
    env, params, state = _setup("cement_kiln")
    resting = state.replace(
        lime=state.lime.at[-1].set(state.target_lime), fuel=jnp.float32(0.0)
    )
    times = HELD["cement_kiln"][0][1] * jnp.arange(DRAWS)
    floor = _floor(env, params, resting, times)
    assert floor == pytest.approx(0.1296, abs=4 * 1.414 * 0.1296 / math.sqrt(DRAWS))


def test_boiler_drum_measured_reward_has_the_steam_meters_floor():
    """On plan, the measured cost is the steam meter's scatter priced as steam
    off the plan: ``E|N(0, 0.047)| = 0.047 sqrt(2 / pi) = 0.0375`` kg/s, times
    2 s, is 7.50e-5 t per step, with sd ``0.047 sqrt(1 - 2 / pi) * 2e-3 =
    5.67e-5`` t per draw."""
    env, params, state = _setup("boiler_drum")
    resting = state.replace(level=jnp.float32(0.0), q_steam=state.q_demand)
    floor = _floor(env, params, resting, jnp.arange(DRAWS))
    assert floor == pytest.approx(7.50e-5, abs=4 * 5.67e-5 / math.sqrt(DRAWS))


def test_glass_furnace_measured_reward_is_the_cost_of_the_reading_received():
    """A plant knows its crown only through the thermocouple, so the measured
    cost is the observation's, lag and noise together, while ``reward`` scores
    the true crown. In K, at ``e_floor`` 1, plus fuel as a share of the hold."""
    env, params, state = _setup("glass_furnace", unrounded=False)
    key = jax.random.PRNGKey(2)
    for t in range(5):
        obs, state, reward, _, info = env.step_env(
            jax.random.fold_in(key, t), state, jnp.zeros(1), params
        )
        crown, _, fuel_pct, _, target, _ = np.asarray(obs, np.float64)
        running = fuel_pct * params.fuel_max / 100.0
        running *= params.running_weight / params.c_hold
        expected = (target - crown) ** 2 + running
        assert float(-info["measured_reward"]) == pytest.approx(expected, abs=2e-2)
        true_cost = float(state.target_T_crown - state.T_crown) ** 2 + (
            float(state.fuel_flow) * params.running_weight / params.c_hold
        )
        assert float(-reward) == pytest.approx(true_cost, abs=2e-2)


@pytest.mark.parametrize("name, sigma_W", [("wind_turbine", 1e3), ("battery", 330.0)])
def test_measured_power_is_the_meters(name, sigma_W):
    """The proposal's reason for wind's own reading: speed times torque, each
    through its own sensor, scatters about 10.6 kW against a 1 kW meter, and a
    learner must see the meter. Tracking is linear at ``per_W`` per W, so with
    the target 100 kW above the plant, far past the noise, the measured error
    gives the measured power, and that must be the power the policy received.
    """
    env, params, state = _setup(name)
    true_MW = float(env.get_obs(state, _quiet(params))[3])
    state = state.replace(target_power=jnp.float32(true_MW * 1e6 + 1e5))
    per_W = params.imbalance_price / 1e6 * params.delta_t / 3600.0

    def read(t):
        s = state.replace(time=t)
        terms = env.reward_terms(
            measured_state(env, s, params), params, tripped=jnp.bool_(False)
        )
        return terms["tracking"] / per_W, env.get_obs(s, params)[3]

    error, obs_MW = jax.jit(jax.vmap(read))(jnp.arange(DRAWS))
    power = true_MW * 1e6 + 1e5 - np.asarray(error, np.float64)
    assert (power - true_MW * 1e6).std() == pytest.approx(sigma_W, rel=0.09)
    np.testing.assert_allclose(power / 1e6, obs_MW, atol=3e-6)  # 3 W in float32


def test_reactor_measured_reward_carries_the_steps_one_reading():
    """Tracking is linear at $230/h per K, so 30 s of it is $1.917 per K. With Tavg
    well above program, a reading ``d`` above the plant adds ``1.917 * d`` to the
    step's cost, and ``d`` is what the observation of that step shows."""
    env, params, state = _setup("reactor")
    hot = state.replace(T_coolant=state.T_coolant + 8.0, T_cold=state.T_cold + 8.0)
    key = jax.random.PRNGKey(3)
    for t in range(5):
        obs, new, reward, _, info = env.step_env(
            jax.random.fold_in(key, t), hot.replace(time=t), jnp.array([0.2]), params
        )
        d = float(obs[4]) - 0.5 * float(new.T_coolant + new.T_cold)
        assert abs(d) > 0.0
        cost = float(reward) - float(info["measured_reward"])
        assert cost == pytest.approx(230.0 * 30.0 / 3600.0 * d, abs=5e-4)
