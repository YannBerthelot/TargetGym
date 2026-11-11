"""The grid battery as a continuing plant (long-strata draft 4, S1, E4; owner's call
of 2026-11-11 on the BMS estimate).

Contract:
- the BMS recalibrates at a voltage knee: once the true state of charge leaves the
  LFP plateau (ARX-LFP: roughly 20 % to 80 %), the estimate's error is drawn afresh
  from the family's spread (``soc_error_std``) and its drift counts again from then;
- the drift counts from the last recalibration, not from when the plant started;
- the day-ahead prices never run out: a price event is cleared past the visible
  horizon, shows in the forecast, holds its periods, and does not come back a day
  later;
- gate closure keeps exact time however long the plant has run;
- the long deviation is a stream event: it starts now, holds one way at a size in
  ``bias_pu_range`` for its length, then ends;
- a stream starts with neither the window's price event nor its deviation.

Oracles: the plateau's 20 % and 80 % (REALITY.md, ARX-LFP), the scenarios' ranges,
a 4 h block of eight 30 min periods, the signal's 300 s relaxation, and the sampling
error of a standard deviation for the fresh errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.stream import make_stream

SIGNAL, PRICE_NOW, PRICE_OPEN = 5, 13, 16  # observation channels
PERIOD = 1800  # steps in a settlement period at 1 s
DAY = 48 * PERIOD


@pytest.fixture(scope="module")
def battery():
    """The stream's battery, its scenarios' params as the stream runs them, and its
    reset, event hook, one step and an ``n``-step run holding an action."""
    stream = make_stream("battery")
    env = stream.env
    names = [s.name for s in stream.scenarios]

    def params(name, **kw):
        p = stream.scenario_params(stream.default_params, names.index(name))
        return p.replace(**kw)

    @jax.jit
    def step(state, p, action, key=jax.random.PRNGKey(7)):
        obs, state, *_ = env.step_env(key, state, action, p)
        return obs, state

    def run(state, p, n, action=0.0):
        @jax.jit
        def go(state, p):
            def body(s, k):
                obs, s, *_ = env.step_env(k, s, jnp.array([action]), p)
                return s, obs

            return jax.lax.scan(body, state, jax.random.split(jax.random.PRNGKey(8), n))

        state, obs = go(state, p)
        return state, np.asarray(obs)

    return SimpleNamespace(
        env=env,
        params=params,
        reset=jax.jit(env.reset_env),
        start=jax.jit(env.start_event),
        step=step,
        run=run,
    )


def _into_the_top_knee(battery, p, key, time=10):
    """A pack at 79.99 % true charge, charging at full power, so one step takes it
    past the plateau's 80 % (ARX-LFP); with an estimate 10 % off."""
    _, s = battery.reset(key, p)
    return s.replace(
        time=jnp.int32(time),
        soc=jnp.float32(0.7999),
        soc_error=jnp.float32(0.10),
        target_power=jnp.float32(-p.power_max),
    )


def test_a_knee_recalibrates_the_estimate_to_the_family_s_spread(battery):
    # With the family's spread at zero, a recalibration leaves the estimate exact.
    exact = battery.params("normal", soc_error_std=0.0)
    s = _into_the_top_knee(battery, exact, jax.random.PRNGKey(0))
    action = jnp.array([0.0])
    _, s = battery.step(s, exact, action)  # into the knee
    _, s = battery.step(s, exact, action)  # the corrected estimate reports
    assert float(s.soc) >= 0.80
    # One step's drift is 8e-8: nothing else is left of the 10 % error.
    np.testing.assert_allclose(float(s.soc_reported - s.soc), 0.0, atol=1e-6)

    # With the family's spread, each recalibration draws the error afresh.
    # Each plant its own world: a stream's step keys come from its own seed.
    p = battery.params("normal")
    keys = jax.random.split(jax.random.PRNGKey(1), 512)
    starts = jax.vmap(lambda k: _into_the_top_knee(battery, p, k))(keys)

    def two(s, k):
        _, s = battery.step(s, p, action, k)
        return battery.step(s, p, action, jax.random.fold_in(k, 1))[1]

    ends = jax.vmap(two)(starts, keys)
    gaps = np.asarray(ends.soc_reported - ends.soc)
    # The standard deviation of 512 normal draws scatters by sd / sqrt(2 n), 3 %:
    # 10 % is over three of those. The reported value is clipped to [0, 1], which
    # an error of 0.075 from 0.80 does not reach.
    np.testing.assert_allclose(gaps.std(), p.soc_error_std, rtol=0.10)
    assert abs(gaps.mean()) < 3 * p.soc_error_std / np.sqrt(gaps.size)


def test_the_drift_counts_from_the_last_recalibration(battery):
    # No signal, so the pack rests where it is put and both clocks see one plant.
    exact = battery.params(
        "normal", soc_error_std=0.0, signal_std=0.0, neutrality_tau=1.0e12
    )
    action = jnp.array([0.0])
    gaps = []
    for time in (10, 1_000_000_000):  # about 32 years on: drift from then would be 83
        s = _into_the_top_knee(battery, exact, jax.random.PRNGKey(2), time)
        s = s.replace(signal_x=jnp.float32(0.0))
        _, s = battery.step(s, exact, action)  # into the knee: recalibrated
        s = s.replace(target_power=jnp.float32(exact.power_max))
        _, s = battery.step(s, exact, action)  # back onto the plateau
        assert float(s.soc) < 0.80
        s, _ = battery.run(s, exact, 3600)  # an hour on the plateau
        gaps.append(float(s.soc_reported - s.soc))
    assert gaps[0] == gaps[1]
    # An hour at 10 % in 14 days, either way (MPS-ACCURE).
    np.testing.assert_allclose(abs(gaps[0]), 0.10 / (14 * 86400) * 3600, rtol=0.05)


def test_a_price_event_shows_ahead_holds_and_does_not_come_back(battery):
    spike = battery.params("price spike")
    _, s = battery.reset(jax.random.PRNGKey(3), spike)
    s = battery.start(
        s.replace(time=jnp.int32(5 * PERIOD + 100)), jax.random.PRNGKey(4), spike
    )

    def price_at_period(s, k):
        """Step into period ``k`` from its last step before: the boundary renews
        the ring."""
        obs, s = battery.step(
            s.replace(time=jnp.int32(k * PERIOD - 1)), spike, jnp.array([0.0])
        )
        return np.asarray(obs), s

    obs = np.asarray(battery.env.get_obs(s, spike))
    np.testing.assert_array_equal(obs[PRICE_NOW : PRICE_OPEN + 1], spike.price_base)
    obs, s = price_at_period(s, 6)  # the event is cleared for period 9, now in view
    level = obs[PRICE_OPEN]
    assert spike.price_event_range[0] <= level <= spike.price_event_range[1]
    for k, want in ((9, level), (10, level), (11, spike.price_base)):
        obs, s = price_at_period(s, k)
        assert obs[PRICE_NOW] == want, k
    obs, s = price_at_period(s, 9 + 48)  # a day later
    np.testing.assert_array_equal(obs[PRICE_NOW : PRICE_OPEN + 1], spike.price_base)


def test_gate_closure_keeps_exact_time_however_long_the_plant_has_run(battery):
    p = battery.params("normal")
    _, s = battery.reset(jax.random.PRNGKey(5), p)
    declared = []
    # The last second before a gate closes, and the same second about 31.7 years on,
    # where float32 counts seconds in 64 s steps and would round across the gate.
    for time in (8_999, 11_574 * DAY + 8_999):
        _, after = battery.step(s.replace(time=jnp.int32(time)), p, jnp.array([1.0]))
        declared.append(np.asarray(after.baselines))
    np.testing.assert_array_equal(declared[0], declared[1])
    assert np.count_nonzero(declared[0]) == 1  # one period's baseline, declared


def test_a_long_deviation_starts_holds_one_way_and_ends(battery):
    # No noise and no neutrality pull, so the signal follows its mean alone.
    dev = battery.params(
        "long deviation",
        signal_std=0.0,
        neutrality_tau=1.0e12,
        bias_duration_range=(600.0, 600.0),
    )
    _, s = battery.reset(jax.random.PRNGKey(6), dev)
    s = s.replace(signal_x=jnp.float32(0.0), time=jnp.int32(3 * DAY + 77))
    s = battery.start(s, jax.random.PRNGKey(7), dev)
    _, obs = battery.run(s, dev, 1800)
    signal = obs[:, SIGNAL]
    lo, hi = dev.bias_pu_range
    # Relaxing at 300 s, it reaches 1 - e^-2 of the deviation by 600 s.
    reached = abs(signal[595])
    assert lo * (1 - np.exp(-2.0)) * 0.95 <= reached <= hi
    held = signal[60:600]
    assert (np.sign(held) == np.sign(signal[595])).all()  # one way throughout
    assert abs(signal[-1]) < 0.05 * reached / (1 - np.exp(-2.0))  # ended: 4 tau on


@pytest.mark.parametrize("name", ["price spike", "long deviation"])
def test_a_stream_starts_without_the_window_s_events(battery, name):
    # The window would raise two periods' price inside the block, or run a long
    # deviation through it.
    p = battery.params(name, signal_std=0.0, neutrality_tau=1.0e12)
    _, s = battery.reset(jax.random.PRNGKey(8), p)
    s = s.replace(signal_x=jnp.float32(0.0))
    _, obs = battery.run(s, p, 8 * PERIOD)
    np.testing.assert_array_equal(obs[:, PRICE_NOW : PRICE_OPEN + 1], p.price_base)
    np.testing.assert_array_equal(obs[:, SIGNAL], 0.0)
