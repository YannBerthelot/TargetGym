"""The jacketed CSTR against its plant family (REALITY.md, COLD_REVIEW.md T1 to
T8, Amendment 3).

Contract, from the sources and the audit's tolerances, not from the code:
- the design point is PARSEX's: at 58 C with feed at 24 C it converts 74 to 79 %
  (CA about 0.31 to 0.39 mol/L [S1, Tables 3, 4]) with a cooler near 47 C, on
  fresh water "around 120 g/s" [S1, p. 6]: from 10 % under that to the 136 g/s
  Figure 2 logs [S1, Fig. 2], and inside the valve's range;
- T1: the rate doubles per about 10 C: a ratio 1.5 to 3.0 [S5, p. 1];
- T2: a 40 % loss of activity cools the reactor by at least 1.7 K, open loop
  [S1, Fig. 3];
- T3: the jacket delays a cooling move: at least 60 s to the first 10 % of the
  reactor's response [S6];
- T4: the reactor thermowell's 63 % time is 4 to 45 s [S10, Fig. 2];
- T5: no composition reading: no online analyser on such a reactor was found
  and Bayer sampled once a day [S3]; the observation carries none;
- T6 restated on the family (Amendment 3): loss of circulation heats the
  reactor at 0.75 to 8 K/min over its first 3 min, pooled from Morton's 1 to
  2 C/min [S4] and Bayer's over 2 C/min [S3], with the audit's factor 4 on the
  top and 25 % below the slow end;
- T7: it rises 25 K within 3 to 48 min [S2; S3];
- T8: open loop, a cooling step's 95 % time is 1.6 to 6.4 residence times
  [S1, Fig. 2];
- the interlock cuts the feed on high temperature [S3], so a runaway trips;
- the feed sits at room temperature, 23.1 and 24.9 C on S1's two days
  [S1, Tables 3, 4], so within an episode it drifts and never leaves that
  range; the drift is slow (ours: no faster than that 1.8 K swing over a
  day's cycle allows, under 0.01 K a step) but moves the reactor off its
  setpoint when nothing acts, so holding the reset's water is not free;
- the catalyst_loss scenario loses S1's 40 % of activity [S1, Fig. 3] as a
  decay, since activity "is always expected to decay throughout the operation"
  and S1's instant loss "is not likely to occur" [S1, p. 6]: it falls
  monotonically over 90 min (ours: no source gives a rate inside 4 h, so the
  slowest decay that ends inside the episode from the window's latest event,
  240 - 150 min) and then stays at 60 %;
- the episode starts on its steady state at the setpoint (rule 17).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.pc_gym.cstr.env import (
    CSTRParams,
    compute_next_state,
    events,
    rate,
    steady_state,
)
from target_gym.pc_gym.cstr.env_jax import CSTR
from target_gym.registry import REGISTRY

ENV = CSTR()


def quiet(**kw):
    return CSTRParams(sensor_noise_scale=0.0, **kw)


def rest(p, key=0):
    """The reset state with no event, on its steady state."""
    _, s = ENV.reset_env(jax.random.PRNGKey(key), p)
    return s.replace(event_at=jnp.asarray(1e9))


def hold(p, s, raw, steps):
    """True states over ``steps`` at a constant raw water command, by
    ``compute_next_state`` (no trip restarts)."""
    a = jnp.asarray(raw)

    def body(s, _):
        s = compute_next_state(a, s, p)[0]
        return s, jnp.stack([s.C_a, s.T, s.T_j])

    return np.asarray(jax.lax.scan(body, s, None, length=steps)[1])


def test_the_design_point_is_parsexs():
    p = quiet()
    C_a, T_j, F_w = (float(v) for v in steady_state(331.15, 297.15, p))
    assert 1.0 - 0.79 - 0.01 <= C_a / p.Caf <= 1.0 - 0.74 + 0.01
    assert T_j - 273.15 == pytest.approx(47.0, abs=1.0)
    assert 0.9 * 120.0 <= F_w / 60.0 <= 136.0
    assert p.Fw_min <= F_w <= p.Fw_max


def test_the_rate_doubles_per_ten_kelvin():
    p = quiet()
    for T in (320.0, 331.15, 340.0):
        ratio = float(rate(T + 10.0, 1.0, p) / rate(T, 1.0, p))
        assert 1.5 <= ratio <= 3.0, (T, ratio)


def test_a_loss_of_activity_cools_the_reactor():
    p = quiet()
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    weak = p.replace(k0=0.6 * p.k0)
    traj = hold(weak, s, raw, int(30 * 34 / p.delta_t))
    assert float(s.T) - traj[-1, 1] >= 1.7


def test_the_catalyst_loss_is_a_90_minute_decay_to_60_percent():
    (loss,) = [
        s for s in scenarios.suite(REGISTRY["cstr"]) if s.name == "catalyst_loss"
    ]
    p = scenarios.params_for(REGISTRY["cstr"], loss)
    at = 60.0
    s = rest(p).replace(event_at=jnp.asarray(at))

    def activity(minute):
        return float(events(s, p, minute)[2]) / p.k0

    assert activity(at - p.delta_t) == pytest.approx(1.0)
    during = [activity(at + m) for m in np.arange(p.delta_t, 90.0 + 1e-9, p.delta_t)]
    assert all(b < a for a, b in zip([activity(at)] + during, during))
    assert during[0] > 0.99  # not a step: one step in, under 1 % of it gone
    assert during[-1] == pytest.approx(0.6)
    assert activity(at + 90.0 + 60.0) == pytest.approx(0.6)
    # The latest event's decay ends inside the episode.
    end = p.max_steps_in_episode * p.delta_t
    late = s.replace(event_at=jnp.asarray(p.event_window[1]))
    assert float(events(late, p, end)[2]) / p.k0 == pytest.approx(0.6)


def test_the_jacket_delays_a_cooling_move():
    p = quiet()
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    traj = hold(p, s, raw + 0.5, 2000)
    d = traj[:, 1] - float(s.T)
    first = np.argmax(np.abs(d) >= 0.1 * np.abs(d).max())
    assert (first + 1) * p.delta_t * 60.0 >= 60.0


def test_the_thermowell_lags_four_to_forty_five_seconds():
    """The reading's lag on the true reactor temperature, fitted as the 63 %
    time of its response to a jump."""
    p = quiet()
    s = rest(p)
    raw = jnp.asarray(ENV.initial_action(s, p))
    s = s.replace(T=s.T + 5.0)
    step = jax.jit(ENV.step_env)
    readings, truth = [], []
    for k in range(12):
        obs, s, *_ = step(jax.random.PRNGKey(k), s, raw, p)
        readings.append(float(obs[0]))
        truth.append(float(s.T))
    # The reading closes 1 - exp(-dt / lag) of its gap to the truth per step.
    gap0 = abs(readings[0] - truth[0])
    closed = 1.0 - abs(readings[1] - truth[1]) / max(abs(readings[0] - truth[1]), 1e-9)
    lag_s = -p.delta_t * 60.0 / np.log(1.0 - closed)
    assert gap0 > 0.0 and 4.0 <= lag_s <= 45.0, lag_s


def test_no_composition_is_observed():
    """No channel of the observation moves with C_A alone."""
    p = quiet()
    s = rest(p)
    a = np.asarray(ENV.get_obs(s, p))
    b = np.asarray(ENV.get_obs(s.replace(C_a=s.C_a * 0.5), p))
    np.testing.assert_array_equal(a, b)


def _circulation_loss(p, minutes):
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    return s, hold(p.replace(UA=0.0), s, raw, int(minutes / p.delta_t))


def test_loss_of_circulation_heats_at_the_familys_rate():
    p = quiet()
    s, traj = _circulation_loss(p, 3.0)
    rise = (traj[-1, 1] - float(s.T)) / 3.0
    assert 0.75 <= rise <= 8.0, rise


def test_loss_of_circulation_rises_25_kelvin_in_minutes():
    p = quiet()
    s, traj = _circulation_loss(p, 60.0)
    hit = np.nonzero(traj[:, 1] >= float(s.T) + 25.0)[0]
    assert hit.size and 3.0 <= (hit[0] + 1) * p.delta_t <= 48.0


def test_an_open_loop_cooling_step_settles_in_a_few_residence_times():
    p = quiet()
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    traj = hold(p, s, raw - 0.1, 2000)
    d = traj[:, 1] - float(s.T)
    t95 = (np.argmax(np.abs(d) >= 0.95 * abs(d[-1])) + 1) * p.delta_t
    residence = p.V / p.q
    assert 1.6 <= t95 / residence <= 6.4, t95 / residence


def test_a_runaway_trips_the_interlock():
    """[S3]: high temperature cuts the feed. Loss of circulation runs the
    reactor away, and the env trips before 60 min."""
    p = quiet(UA=0.0, continuing=False)
    s = rest(p)
    raw = jnp.asarray(ENV.initial_action(s, p))
    step = jax.jit(ENV.step_env)
    tripped = False
    for k in range(int(60 / p.delta_t)):
        _, s, _, done, info = step(jax.random.PRNGKey(k), s, raw, p)
        if bool(info["tripped"]):
            tripped = True
            break
    assert tripped


def test_an_episode_starts_at_rest_on_its_setpoint():
    p = quiet()
    s = rest(p)
    raw = float(ENV.initial_action(s, p)[0])
    traj = hold(p, s, raw, 40)
    assert float(s.T) == pytest.approx(p.T_setpoint, abs=1e-3)
    assert np.abs(traj[:, 1] - p.T_setpoint).max() < 0.05


def test_the_feed_drifts_inside_s1s_range_and_moves_a_held_reactor():
    p = quiet()
    lo, hi = p.feed_T_range
    for seed in range(10):
        s = rest(p, seed)
        raw = jnp.asarray(ENV.initial_action(s, p)[0])

        def body(s, _):
            s = compute_next_state(raw, s, p)[0]
            return s, jnp.stack([s.T_feed, s.T])

        feed, T = np.asarray(
            jax.lax.scan(body, s, None, length=int(p.max_steps_in_episode))[1]
        ).T
        assert lo - 1e-6 <= feed.min() and feed.max() <= hi + 1e-6
        assert np.abs(np.diff(feed)).max() < 0.01
        assert feed.max() - feed.min() > 0.05, seed
        assert np.abs(T - p.T_setpoint).max() > 0.01, seed
