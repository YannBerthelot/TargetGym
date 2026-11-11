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
7. surge is counted: one short surge does not trip; a third within the window
   (10 to 20 s [C5]) does.
And the job [C3, pp. 1-2, 4]: a constant header pressure; inside its zone the
run on less recycle scores better.
Test 5 (signal directions [C4, Table 1]) needs a suction volume, an energy
balance and drive inertia, none modelled: a known deviation, not tested.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.compressor_surge.env import (
    CompressorSurgeParams,
    compute_next_state,
)
from target_gym.compressor_surge.env_jax import CompressorSurge

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
