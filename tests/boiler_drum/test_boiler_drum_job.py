"""The drum boiler's limits, score, observation and episode (charter rules
14 to 17), each from ``REALITY.md``.

Contract:
- Trips (rule 14): the level past -330 mm [S9] or +254 mm [S20] for more than
  10 s [S9] trips the plant; a shorter excursion does not. Pressure has no
  sourced limit, so it never trips.
- Score (rule 15, Amendment 3), GBP: steam off the load plan beyond the steam
  meter's 0.047 kg/s, as generation at 3.5 kg/kWh priced at GBP 80/MWh; a trip
  loses a 2.5 h hot restart's generation [S17]: 93.35 kg/s * 9000 s / 3.5
  kg/kWh = 240.04 MWh, GBP 19,203.4.
- Observation (rule 16): the plan is read: load now, load planned, and the
  seconds to the ramp.
- Episode (rule 17): established operation at a planned load between 40 % and
  100 % of nominal, balanced at normal water level, one ramp at 1 to 6 %/min
  ([S10], [S17]).
- Upsets (rule 20): an ID fan runback caps firing at what the runback load
  needs and runs the plan down to it [S10]; a feed pump trip leaves half the
  nominal feed until the standby pump delivers 30 to 60 s later [S10, S11],
  and runs the unit back too. Each is read as it strikes, not before.
- Scenarios: normal, industrial_swing, low_load, id_fan_runback and
  feed_pump_trip, each citing REALITY.md.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.boiler_drum.env import (
    BoilerDrumParams,
    check_is_terminal,
    compute_reward,
    compute_velocity,
)
from target_gym.boiler_drum.env_jax import BoilerDrum

PARAMS = BoilerDrumParams(sensor_noise_scale=0.0)


@pytest.fixture(scope="module")
def env():
    return BoilerDrum()


def _reset(env, seed=0, params=PARAMS):
    return env.reset_env(jax.random.PRNGKey(seed), params)


def _held(state, level, seconds, params=PARAMS):
    """``state`` after ``seconds`` with the level held at ``level``: the trip
    timers count what the plant reports, step by step."""
    high = jnp.asarray(0.0)
    low = jnp.asarray(0.0)
    for _ in range(int(round(seconds / params.delta_t))):
        high = jnp.where(level >= params.level_trip_high, high + params.delta_t, 0.0)
        low = jnp.where(level <= params.level_trip_low, low + params.delta_t, 0.0)
    return state.replace(level=jnp.asarray(level), time_high=high, time_low=low)


@pytest.mark.parametrize("level", [0.26, -0.34])
def test_level_past_a_trip_for_ten_seconds_or_less_does_not_trip(env, level):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(_held(state, level, 10.0), PARAMS)
    assert not bool(tripped)


@pytest.mark.parametrize("level", [0.26, -0.34])
def test_level_past_a_trip_for_more_than_ten_seconds_trips(env, level):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(_held(state, level, 12.0), PARAMS)
    assert bool(tripped)


def test_trips_are_asymmetric(env):
    """-0.30 m is inside the low trip at -330 mm; +0.30 m is past the high
    trip at +254 mm."""
    _, state = _reset(env)
    low, _ = check_is_terminal(_held(state, -0.30, 20.0), PARAMS)
    high, _ = check_is_terminal(_held(state, 0.30, 20.0), PARAMS)
    assert not bool(low) and bool(high)


@pytest.mark.parametrize("pressure", [55.0, 115.0])
def test_pressure_alone_never_trips(env, pressure):
    _, state = _reset(env)
    tripped, _ = check_is_terminal(
        state.replace(pressure=jnp.asarray(pressure)), PARAMS
    )
    assert not bool(tripped)


def test_steam_on_plan_costs_nothing(env):
    _, state = _reset(env)
    assert float(compute_reward(state, PARAMS)) == 0.0


def test_steam_off_plan_within_the_meters_resolution_costs_nothing(env):
    _, state = _reset(env)
    state = state.replace(q_demand=jnp.asarray(80.0), q_steam=jnp.asarray(80.04))
    assert float(compute_reward(state, PARAMS)) == 0.0


def test_steam_off_plan_costs_its_generation_at_the_imbalance_price(env):
    """Valve passing 80 kg/s at nominal; 2 bar low passes 80 * 83 / 85: 1.882
    kg/s short, 1.835 past the meter, 3.671 kg in a 2 s step, 1.049 kWh at
    3.5 kg/kWh, GBP 0.0839 at GBP 80/MWh."""
    _, state = _reset(env)
    q = 80.0 * 83.0 / 85.0
    state = state.replace(q_demand=jnp.asarray(80.0), q_steam=jnp.asarray(q))
    expected = ((80.0 - q) - 0.047) * 2.0 / 3.5 / 1000.0 * 80.0
    np.testing.assert_allclose(
        -float(compute_reward(state, PARAMS)), expected, rtol=1e-4
    )


def test_a_trip_costs_a_hot_restarts_generation(env):
    _, state = _reset(env)
    tripped = _held(state, 0.30, 12.0)
    np.testing.assert_allclose(
        -float(compute_reward(tripped, PARAMS)), 19_203.4, rtol=1e-4
    )


def _upset(env, kind, seed=1):
    params = PARAMS.replace(event_kind=kind, load_range=(0.9, 0.9))
    _, state = env.reset_env(jax.random.PRNGKey(seed), params)
    state = state.replace(ramp_rate=jnp.asarray(0.0))  # no planned ramp
    return params, state


def _roll(env, params, state, steps):
    """Roll the shipped three-element expert, which follows the runback, so the
    plant does not trip and restart on a new upset. Its commands sit above the
    caps while the unit is above the runback load, so the caps bind."""
    from target_gym.boiler_drum.experts import make_boiler_drum_pid_controller

    ctrl = make_boiler_drum_pid_controller()
    act, step = jax.jit(ctrl.act), jax.jit(env.step_env)
    memory, obs = ctrl.init_state, env.get_obs(state, params)
    out = []
    for k in range(steps):
        action, memory = act(memory, obs)
        obs, state, _, _, info = step(jax.random.PRNGKey(k), state, action, params)
        assert not bool(info["tripped"]), k
        out.append((obs, state))
    return out


def test_an_id_fan_runback_caps_firing_and_runs_the_plan_back(env):
    params, state = _upset(env, 1)
    t0 = float(state.event_time)
    nominal = params.q_steam_nominal
    rate = float(state.event_rate)
    out = _roll(env, params, state, int((t0 + 600.0) / params.delta_t))
    before = [s for o, s in out if s.time * params.delta_t < t0 - 2.0]
    after = [(o, s) for o, s in out if s.time * params.delta_t > t0 + 2.0]
    assert all(float(o[10]) == pytest.approx(100.0) for o, s in out[: len(before)])
    cap = 0.5 * nominal * (2.751e6 - params.h_feedwater)  # IAPWS h_s at 85 bar
    assert all(float(s.Q_fuel) <= cap * 1.001 for _, s in after)
    assert any(float(s.Q_fuel) > cap * 1.01 for s in before)  # the cap binds
    # The plan runs back at the drawn rate, then holds at half nominal.
    obs, s = after[10]
    seconds = float(s.time) * params.delta_t
    expected = max(0.5 * nominal, 0.9 * nominal - rate * (seconds - t0))
    assert float(s.q_demand) == pytest.approx(expected, rel=1e-4)
    assert float(after[-1][1].q_demand) == pytest.approx(0.5 * nominal, rel=1e-4)
    # Read as it strikes: the plan's load and the firing cap in the observation.
    assert float(obs[8]) == pytest.approx(0.5 * nominal, rel=1e-4)
    assert float(obs[10]) == pytest.approx(100.0 * cap / params.Q_max, rel=1e-3)
    assert float(before[-1].q_demand) == pytest.approx(0.9 * nominal, rel=1e-4)


def test_a_feed_pump_trip_halves_feed_until_the_standby_delivers(env):
    """Feed is held at half the nominal steam while one pump is out, read as
    the feed cap; the pumps' full span returns when the standby delivers. The
    feed's own rate limit (0.05 of span per s) keeps it from jumping."""
    params, state = _upset(env, 2)
    t0, wait = float(state.event_time), float(state.standby_delay)
    assert 30.0 <= wait <= 60.0
    out = _roll(env, params, state, int((t0 + wait + 60.0) / params.delta_t))
    half = 0.5 * params.q_steam_nominal
    half_pct = 100.0 * half / params.q_feed_max
    for obs, s in out:
        t = float(s.time) * params.delta_t  # the reading's time
        if t0 + 1.0 < t < t0 + wait - 1.0:
            assert float(obs[11]) == pytest.approx(half_pct, rel=1e-4)
        if t0 + 3.0 < t < t0 + wait + 1.0:  # the feed this step was set at t - 2 s
            assert float(s.q_feed) <= half * (1.0 + 1e-4)
        if t < t0 - 1.0 or t > t0 + wait + 1.0:
            assert float(obs[11]) == pytest.approx(100.0, rel=1e-4)
    assert float(out[-1][1].q_demand) < 0.9 * params.q_steam_nominal


def test_the_scenarios_are_the_familys_and_each_runs():
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    spec = REGISTRY["boiler_drum"]
    suite = scenarios.suite(spec)
    assert [s.name for s in suite] == [
        "normal",
        "industrial_swing",
        "low_load",
        "id_fan_runback",
        "feed_pump_trip",
    ]
    env = spec.make_env()
    for sc in suite:
        assert "[S" in sc.source, sc.name
        p = scenarios.params_for(spec, sc, sensor_noise_scale=0.0)
        obs, state = env.reset_env(jax.random.PRNGKey(0), p)
        for k in range(5):
            obs, state, r, _, _ = env.step_env(
                jax.random.PRNGKey(k), state, env.initial_action(state, p), p
            )
        assert np.all(np.isfinite(np.asarray(obs))) and np.isfinite(float(r))


@pytest.mark.parametrize("seed", range(8))
def test_episode_starts_in_established_operation_on_a_plan(env, seed):
    obs, state = _reset(env, seed)
    nominal = PARAMS.q_steam_nominal
    assert 0.4 * nominal - 1e-4 <= float(state.load_from) <= nominal + 1e-4
    assert 0.4 * nominal - 1e-4 <= float(state.load_to) <= nominal + 1e-4
    rate = float(state.ramp_rate) * 60.0 / nominal * 100.0  # %/min
    assert 1.0 - 1e-4 <= rate <= 6.0 + 1e-4
    assert float(state.level) == 0.0
    # Balanced: no derivative at the reset's firing, feedwater and load.
    d, _ = compute_velocity(
        jnp.array([state.pressure, state.V_wt, state.alpha_r, state.V_sd]),
        (state.Q_fuel, state.q_feed),
        state.q_demand,
        PARAMS,
    )
    np.testing.assert_allclose(np.asarray(d), 0.0, atol=1e-5)
    # The plan is read: load now, load planned, seconds to the ramp.
    np.testing.assert_allclose(
        np.asarray(obs[7:10]),
        [float(state.load_from), float(state.load_to), float(state.ramp_start)],
        rtol=1e-5,
    )
    # No upset in the normal scenario: both caps read 100 %.
    np.testing.assert_allclose(np.asarray(obs[10:]), [100.0, 100.0], rtol=1e-6)


def test_the_valve_follows_the_plan(env):
    """The load moves at the planned rate after the ramp starts, and stops at
    the planned load."""
    _, state = _reset(env, 3)
    action = env.initial_action(state, PARAMS)
    step = jax.jit(env.step_env)
    key = jax.random.PRNGKey(0)
    loads = []
    for _ in range(PARAMS.max_steps_in_episode):
        _, state, _, _, _ = step(key, state, action, PARAMS)
        loads.append(float(state.q_demand))
    span = float(state.load_to - state.load_from)
    expected_end = float(state.ramp_start) + abs(span) / float(state.ramp_rate)
    assert expected_end < PARAMS.max_steps_in_episode * PARAMS.delta_t
    np.testing.assert_allclose(loads[-1], float(state.load_to), rtol=1e-5)


def test_a_normal_window_weighs_most_and_every_weight_is_positive():
    """Charter J4: weights are real operating time; normal operation is most of
    it, and every scenario keeps a positive share."""
    from target_gym import scenarios
    from target_gym.registry import REGISTRY

    shares = scenarios.weights(REGISTRY["boiler_drum"])
    assert max(shares, key=shares.get) == "normal"
    assert all(v > 0.0 for v in shares.values())
    assert sum(shares.values()) == pytest.approx(1.0)
