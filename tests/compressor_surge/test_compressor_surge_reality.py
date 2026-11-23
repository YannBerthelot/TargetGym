"""The compressor against its plant family (REALITY.md, COLD_REVIEW.md, Amendment 3).

Contract, from the sources in REALITY.md, the family range the owner's review
verified, and Amendment 3, not from the code:
1. deep surge at closed throttle cycles at 0.4 to 8 Hz, a period of 0.125 to
   2.5 s (Almasi, Plant Engineering 2014, pooled with C1's 1 to 3 s and
   Pinelli's 0.25 to 3.5 Hz);
2. flow reverses within 0.06 s, from +50 % of the cycle's maximum to -50 % of
   its minimum [C1, C2];
3. sustained surge at closed throttle swings flow between forward and
   slightly reverse values [C8, Fig. 5]: it goes negative every cycle, and its
   cycle mean lies within 50 % of its half-amplitude from zero (the audit's 25 %
   widened by the family's spread, ours);
4. the surge period rises with the discharge volume [C8, p. 4];
6. the recycle valve reaches 95 % within 2.5 s [C1], and has moved less than
   5 % of its stroke at 0.15 s and more by 0.30 s [C7, 150 to 300 ms];
   The dead time is the same whenever the command is issued: no travel before
   0.15 s, on every schedule clock.
7. surge is counted: one short surge does not trip; a third within the window
   (10 to 20 s [C5]) does.
And the job [C3, pp. 1-2, 4]: a constant header pressure; inside its zone the
run on less recycle scores better.
8. each scenario starts in established operation, not in a capture (charter B5;
   PHYSICS.md, version 4, "Episode"): the header at its setpoint, the plant at
   rest, on or right of a control line 10 % right of the surge line [C6; C1],
   the recycle no more open than that line needs, so shut on a normal day
   [C3, p. 2]. A trip's restart puts the plant back the same way, for the
   demand it then meets.
Test 5 (signal directions [C4, Table 1]) needs a suction volume, an energy
balance and drive inertia, none modelled: a known deviation, not tested.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.compressor_surge.env import (
    CompressorSurgeParams,
    compute_next_state,
    consumer_flow,
    demand_opening,
    recycle_flow,
    schedule_ramp,
)
from target_gym.compressor_surge.env_jax import CompressorSurge
from target_gym.registry import REGISTRY

FINE = 0.005  # s, the audit's step for reading a cycle


def raw_speed(N, p):
    return 2.0 * (N - p.N_min) / (p.N_max - p.N_min) - 1.0


def deep_surge(p, seconds=30.0, recycle=-1.0):
    """Rated speed, recycle shut, the consumers at their 0.2 floor, stepped by
    ``compute_next_state`` so no trip restarts it. Returns m_c per FINE s."""
    p = p.replace(delta_t=FINE, demand_range=(0.2, 0.2), demand_sigma=0.0)
    env = CompressorSurge()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    action = jnp.array([raw_speed(1.0, p), recycle])

    def body(s, _):
        s = compute_next_state(action, s, p, jax.random.PRNGKey(0), "rk4_1")[0]
        return s, s.m_c

    return np.asarray(jax.lax.scan(body, s, None, length=int(seconds / FINE))[1])


def cycles(m):
    """Indices where flow turns from forward to reverse, after a settle."""
    m = m[len(m) // 3 :]
    return m, np.nonzero((m[:-1] >= 0) & (m[1:] < 0))[0]


@pytest.fixture(scope="module")
def surge():
    return deep_surge(CompressorSurgeParams())


def test_deep_surge_cycles_in_the_family_range(surge):
    m, down = cycles(surge)
    period = np.diff(down[-11:]).mean() * FINE
    assert 0.125 <= period <= 2.5, period


def test_flow_reverses_within_a_twentieth_of_a_second(surge):
    m, down = cycles(surge)
    a, b = down[-3], down[-2]
    cycle = m[a:b]
    hi, lo = 0.5 * cycle.max(), 0.5 * cycle.min()
    # Read on the down-going branch around the reversal at ``b``.
    seg = m[b - 400 : b + 400]
    start = np.nonzero(seg >= hi)[0]
    end = np.nonzero(seg <= lo)[0]
    last_hi = start[start < 400].max()
    first_lo = end[end > 400].min()
    assert (first_lo - last_hi) * FINE <= 0.06


def test_sustained_surge_swings_about_zero_flow(surge):
    m, down = cycles(surge)
    for a, b in zip(down[-6:-1], down[-5:]):
        cycle = m[a:b]
        assert cycle.min() < 0.0
        half = 0.5 * (cycle.max() - cycle.min())
        assert abs(cycle.mean()) <= 0.5 * half


def test_the_surge_period_rises_with_the_discharge_volume():
    periods = []
    for scale in (0.5, 1.0, 2.0):
        p = CompressorSurgeParams()
        m, down = cycles(deep_surge(p.replace(V_p=p.V_p * scale)))
        periods.append(np.diff(down[-6:]).mean() * FINE)
    assert periods[0] < periods[1] < periods[2], periods


def test_the_recycle_valve_strokes_with_its_positioner_lag():
    """[C1]: T95 under 2.5 s; [C7]: 150 to 300 ms of positioner and actuator
    lag before visible motion (5 % of stroke)."""
    p = CompressorSurgeParams(delta_t=FINE, demand_sigma=0.0)
    env = CompressorSurge()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(
        x=jnp.asarray(0.0), N_ramp=s.N, valve_cmds=jnp.zeros_like(s.valve_cmds)
    )
    action = jnp.array([raw_speed(float(s.N), p), 1.0])
    xs = []
    for _ in range(int(3.0 / FINE)):
        s = compute_next_state(action, s, p, jax.random.PRNGKey(0), "rk4_1")[0]
        xs.append(float(s.x))
    xs = np.asarray(xs)
    t = (np.arange(len(xs)) + 1) * FINE
    assert xs[t <= 0.15 + 1e-9].max() < 0.05
    assert xs[np.abs(t - 0.30) < 1e-9][0] > 0.05
    assert t[np.argmax(xs >= 0.95)] <= 2.5


def test_the_valve_waits_out_its_dead_time_on_every_clock():
    """[C7]: 150 to 300 ms pass before the valve moves; the task's is 0.15 s.
    Whatever the schedule's clock reads when the command is issued, the valve
    has not travelled 0.1 s after it, and has travelled (0.2 - 0.15) s x
    0.5 /s = 0.025 of its stroke 0.2 s after it."""
    p = CompressorSurgeParams(demand_sigma=0.0)
    env = CompressorSurge()
    _, s0 = env.reset_env(jax.random.PRNGKey(0), p)
    action = jnp.array([raw_speed(float(s0.N), p), 1.0])

    def travel(clock):
        s = s0.replace(
            block_clock=clock,
            x=jnp.float32(0.2),
            valve_cmds=jnp.full_like(s0.valve_cmds, 0.2),
        )
        s1 = compute_next_state(action, s, p, jax.random.PRNGKey(0))[0]
        s2 = compute_next_state(action, s1, p, jax.random.PRNGKey(0))[0]
        return s1.x - 0.2, s2.x - 0.2

    first, second = jax.vmap(travel)(jnp.arange(1200))
    # float32 holds an opening near 0.2 to 1.5e-8.
    np.testing.assert_allclose(np.asarray(first), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(second), 0.025, atol=1e-6)


def _run_counting(p, action_of, seconds):
    """Step ``step_env`` with ``continuing`` off; returns per step whether it
    tripped and the surge count the state reports."""
    env = CompressorSurge()
    p = p.replace(continuing=False)
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    # The recycle starts shut, so the low demand surges at once.
    s = s.replace(x=jnp.asarray(0.0), valve_cmds=jnp.zeros_like(s.valve_cmds))
    step = jax.jit(env.step_env)
    trips, counts = [], []
    for k in range(int(seconds / p.delta_t)):
        _, s, _, done, info = step(jax.random.PRNGKey(k), s, action_of(k, s), p)
        trips.append(bool(info["tripped"]))
        counts.append(int(jnp.sum(s.surge_times > s.time * p.delta_t - p.surge_window)))
        if bool(done):
            break
    return np.asarray(trips), np.asarray(counts)


def test_one_short_surge_does_not_trip_and_three_in_the_window_do():
    """[C5]: 3 counts within 10 to 20 s trip a centrifugal machine [C3, p. 3:
    shutdown at the 3rd or 4th event]."""
    p = CompressorSurgeParams(demand_range=(0.2, 0.2), demand_sigma=0.0)
    N = raw_speed(1.0, p)

    # Recycle shut until the first surge is counted, then wide open.
    def once(k, s):
        surged = float(jnp.max(s.surge_times)) > 0.0
        return jnp.array([N, 1.0 if surged else -1.0])

    trips, counts = _run_counting(p, once, 20.0)
    assert counts.max() >= 1 and not trips.any()
    # Recycle shut throughout: deep surge, counted, trips at the third.
    trips, counts = _run_counting(p, lambda k, s: jnp.array([N, -1.0]), 20.0)
    assert trips.any()
    at = int(np.argmax(trips))
    assert counts[at - 1] == 2 if at > 0 else True


def test_inside_the_header_zone_less_recycle_scores_better():
    """[C3, p. 1]: as little recycle as possible; [C3, p. 4]: a constant
    header pressure. Two states in the zone, one recycling more."""
    env = CompressorSurge()
    p = CompressorSurgeParams()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    s = s.replace(dp=jnp.asarray(p.header_setpoint))
    lean = s.replace(x=jnp.asarray(0.1))
    heavy = s.replace(x=jnp.asarray(0.6))
    assert float(env.compute_reward(lean, p)) > float(env.compute_reward(heavy, p))


def test_the_header_setpoint_is_constant():
    """[C3, p. 4]: the usual target is a constant suction or discharge
    pressure; nothing in the sources moves it."""
    env = CompressorSurge()
    p = CompressorSurgeParams()
    _, s = env.reset_env(jax.random.PRNGKey(0), p)
    step = jax.jit(env.step_env)
    seen = []
    for k in range(300):
        obs, s, *_ = step(jax.random.PRNGKey(k), s, jnp.zeros(2), p)
        seen.append(float(obs[6]))
    # float32 holds 24 kPa to about 2e-6.
    np.testing.assert_allclose(seen, p.header_setpoint / 1000.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. Established operation at the reset and after a trip (charter B5)
# ---------------------------------------------------------------------------

#: The pressure transmitter's own sigma is 9.17 Pa (PHYSICS.md, Readings): the
#: reset holds the setpoint to what the plant can read.
AT_SETPOINT = 10.0  # Pa
SCENARIOS = ("normal", "slow_approach", "fast_approach", "load_swings")


#: The map's 10 % line passes 11.711 kg/s at 24 kPa (hand-derived below).
MAP_LINE_FLOW = 11.711


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_each_scenario_starts_in_established_operation(scenario):
    """Charter B5: a scenario starts in established operation. Over 32 resets
    of the scenario's own demand: the header is at its setpoint, the compressor
    passes at least the flow of the map's 10 % control line [C6; C1], and the
    recycle is shut unless the compressor passes just that flow [C3, p. 2].
    The line is the map's: each machine's true line is a hidden fraction off it
    (test_compressor_surge_hidden_line.py)."""
    spec = REGISTRY["compressor_surge"]
    suite = {x.name: x for x in scenarios.suite(spec)}
    p = scenarios.params_for(spec, suite[scenario])
    env = CompressorSurge()
    keys = jax.random.split(jax.random.PRNGKey(0), 32)
    s = jax.vmap(lambda k: env.reset_env(k, p)[1])(keys)
    np.testing.assert_allclose(np.asarray(s.dp), p.header_setpoint, atol=AT_SETPOINT)
    flow = np.asarray(s.m_c)
    assert flow.min() >= MAP_LINE_FLOW - 0.01
    on_line = np.abs(flow - MAP_LINE_FLOW) <= 0.01
    assert np.all(on_line | (np.asarray(s.x) <= 1e-6))


def test_the_reset_recycles_what_the_ten_percent_line_needs():
    """Hand-derived. rho01 = 101325 / (287.05 x 288.15) = 1.22501 kg/m^3. The
    10 % line is Phi = 0.55, where the cubic gives Psi = 0.30 + 0.18 (1 + 1.5 x
    1.2 - 0.5 x 1.2^3) = 0.64848; at 24 kPa its speed is sqrt(24000 / (1.22501 x
    200^2 x 0.64848)) = 0.86908 of rated and its flow 1.22501 x 0.10 x 200 x
    0.86908 x 0.55 = 11.711 kg/s. Consumers at opening 0.4 take 0.12 x 0.4 x
    sqrt(24000 - 10000) = 5.679 kg/s and the full-open recycle passes 0.15 x
    sqrt(24000) = 23.238 kg/s, so the recycle stands at (11.711 - 5.679) /
    23.238 = 0.2596. At opening 0.9 the consumers take 12.779 kg/s, more than
    the line's flow: the recycle is shut and the plant right of the line."""
    env = CompressorSurge()
    # The map's own machine: the speed derived above is the map's.
    low = CompressorSurgeParams(
        demand_range=(0.4, 0.4), demand_sigma=0.0, line_shift_max=0.0
    )
    _, s = env.reset_env(jax.random.PRNGKey(0), low)
    assert float(s.x) == pytest.approx(0.2596, abs=1e-3)
    assert float(s.N) == pytest.approx(0.8691, abs=1e-3)
    assert float(s.m_c) == pytest.approx(11.711, abs=0.01)
    assert float(s.dp) == pytest.approx(24000.0, abs=AT_SETPOINT)

    high = low.replace(demand_range=(0.9, 0.9))
    _, s = env.reset_env(jax.random.PRNGKey(0), high)
    assert float(s.x) == 0.0
    assert float(s.dp) == pytest.approx(24000.0, abs=AT_SETPOINT)
    assert float(s.m_c) > MAP_LINE_FLOW


def test_the_reset_is_at_rest():
    """Established operation is a steady state: held where the reset left
    them, with the demand still, the actuators keep the header at its setpoint
    for 10 s (twenty Helmholtz periods)."""
    env = CompressorSurge()
    for opening in (0.3, 0.6, 0.9):
        p = CompressorSurgeParams(demand_range=(opening, opening), demand_sigma=0.0)
        _, s = env.reset_env(jax.random.PRNGKey(0), p)
        hold = env.initial_action(s, p)
        for _ in range(100):
            s = compute_next_state(hold, s, p, jax.random.PRNGKey(0))[0]
        assert float(s.dp) == pytest.approx(p.header_setpoint, abs=AT_SETPOINT)


def test_a_trip_puts_the_plant_back_in_established_operation_for_its_demand():
    """The restart is for the demand the world carries on with, not for a
    fresh draw: after a trip the header is at its setpoint, the plant is on or
    right of the 10 % line, and its flows balance at the opening it now meets
    (compressor flow = consumers' + recycle, to 0.01 kg/s of about 11)."""
    env = CompressorSurge()
    p = CompressorSurgeParams(demand_range=(0.2, 0.6), demand_sigma=0.0)
    _, s = env.reset_env(jax.random.PRNGKey(1), p)
    # Slowed with the recycle shut at low demand, the plant surges until its
    # counter trips it [C5].
    low = jnp.array([-1.0, -1.0])
    step = jax.jit(env.step_env)
    for k in range(1200):
        _, s, _, _, info = step(jax.random.PRNGKey(k), s, low, p)
        if bool(info["tripped"]):
            break
    assert bool(info["tripped"])
    assert float(s.dp) == pytest.approx(p.header_setpoint, abs=AT_SETPOINT)
    assert float(s.m_c) >= MAP_LINE_FLOW - 0.01
    opening = demand_opening(
        s.demand_levels, s.block_clock * p.delta_t, s.demand_dev, p,
        ramp=schedule_ramp(s, p),
    )  # fmt: skip
    out = consumer_flow(s.dp, opening, p) + recycle_flow(s.dp, s.x, p)
    assert float(s.m_c) == pytest.approx(float(out), abs=0.01)
