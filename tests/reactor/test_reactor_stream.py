"""The reactor as a continuing plant (long-strata draft 4, S1).

Contract, from the design, the scenario suite and REALITY.md, not from the code:
- a load-follow day is an event: the dispatch announces it, the observation
  counts the hours to the ramp down from a notice inside ``ramp_down_range``,
  the load ramps down at the day's rate to its minimum, holds a time inside
  ``hold_range`` and ramps back. The day carries its rate and depth, so it runs
  to its end on another scenario's params;
- one day's program at a time: a day drawn while one is under way is absorbed,
  and one drawn after it ended is announced anew;
- the plan after an onset is the same however long the plant has run;
- a stream starts with no ramp planned: nothing ahead, full load;
- the axial offset's allowance is 1 h outside its band in the previous 24 h
  (REALITY.md section 4): time outside more than a day ago stops counting.

Oracles: the scenarios' declared rates, depths and ranges (REALITY.md section 6)
and the allowance's 1 h in 24 h; the frequency wobble is off, so the load is the
day's program exactly.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.stream import StreamEnv, make_stream

LOAD, TO_DOWN, TO_UP = 3, 8, 9
HOUR_STEPS = 120  # 30 s control steps
#: Rods parked (3 of 228 steps in), no boron flow: the reset's own line-out.
PARKED = jnp.array([1.0 - 2.0 * 3.0 / 228.0, 0.0])


@pytest.fixture(scope="module")
def stream():
    return make_stream("reactor")


def _params(stream, name):
    names = [s.name for s in stream.scenarios]
    p = stream.scenario_params(stream.default_params, names.index(name))
    return p.replace(demand_sigma=0.0, sensor_noise_scale=0.0)


def _reset(stream, onset_steps=40):
    """A stream reset on baseload, advanced ``onset_steps`` control steps."""
    base = _params(stream, "baseload")
    _, s = stream.env.reset_env(jax.random.PRNGKey(0), base)
    sub = stream.env.control_period
    return (
        s.replace(
            time=jnp.int32(onset_steps), physics_time=jnp.int32(onset_steps * sub)
        ),
        base,
    )


def _later(state, steps, env):
    return state.replace(
        time=state.time + steps,
        physics_time=state.physics_time + steps * env.control_period,
    )


@pytest.fixture(scope="module")
def run():
    """``run(stream, state, params, steps)``: the loads and hours-ahead readings
    of ``steps`` control steps with the rods parked."""

    def make(stream):
        @jax.jit
        def roll(state, params, keys):
            def body(s, k):
                obs, s, *_ = stream.env.step_env(k, s, PARKED, params)
                return s, obs[jnp.array([LOAD, TO_DOWN, TO_UP])]

            return jax.lax.scan(body, state, keys)[1]

        return roll

    cache = {}

    def call(stream, state, params, steps):
        roll = cache.setdefault(id(stream), make(stream))
        keys = jax.random.split(jax.random.PRNGKey(9), steps)
        return np.asarray(roll(state, params, keys))

    return call


def test_a_load_follow_day_is_announced_and_runs_its_shape_on_another_scenario(
    stream, run
):
    s, base = _reset(stream)
    slow = _params(stream, "krsko_slow")  # 100-50-100 at 8 %/h, no hold
    s = stream.env.start_event(s, jax.random.PRNGKey(1), slow)
    notice = float(stream.env.get_obs(s, base)[TO_DOWN])
    lo, hi = (x / 3600.0 for x in slow.day_notice_range)
    assert lo <= notice <= hi
    out = run(stream, s, base, 15 * HOUR_STEPS)  # stepped on baseload's params
    load, t = out[:, 0], (np.arange(len(out)) + 1) / HOUR_STEPS  # h after onset
    # No hold: the 30 s samples straddle the bottom, one step's ramp at most.
    assert load.min() == pytest.approx(0.5, abs=0.08 / HOUR_STEPS)
    falling = (t > notice + 0.5) & (t < notice + 6.0)
    slope = np.polyfit(t[falling], load[falling], 1)[0]
    assert slope == pytest.approx(-0.08, rel=1e-3)
    bottom = t[np.argmin(load)]
    assert bottom == pytest.approx(notice + 0.5 / 0.08, abs=1.0 / HOUR_STEPS)
    assert load[t > notice + 2 * 0.5 / 0.08 + 0.1] == pytest.approx(1.0, abs=1e-6)
    # The hours ahead count down to the ramps, and read 0 once they have started.
    assert out[0, 1] == pytest.approx(notice - 1.0 / HOUR_STEPS, abs=1e-4)
    assert np.all(out[t > notice + 0.01, 1] == 0.0)


def test_a_day_s_hold_is_drawn_inside_its_range(stream):
    s, base = _reset(stream)
    daily = _params(stream, "edf_daily")
    lo, hi = (x / 3600.0 for x in daily.hold_range)
    ramp = (1.0 - daily.load_min) / daily.load_ramp_rate / 3600.0
    for k in range(6):
        started = stream.env.start_event(s, jax.random.PRNGKey(k), daily)
        obs = stream.env.get_obs(started, base)
        hold = float(obs[TO_UP] - obs[TO_DOWN]) - ramp
        assert lo - 1e-3 <= hold <= hi + 1e-3


def test_one_day_s_program_at_a_time_and_a_new_one_after_it_ends(stream):
    s, base = _reset(stream)
    daily = _params(stream, "edf_daily")
    first = stream.env.start_event(s, jax.random.PRNGKey(1), daily)
    ahead = stream.env.get_obs(first, base)
    end_h = float(ahead[TO_UP]) + (1.0 - daily.load_min) / daily.load_ramp_rate / 3600
    # Drawn an hour in: absorbed, the hours ahead keep counting down as before.
    mid = _later(first, HOUR_STEPS, stream.env)
    again = stream.env.start_event(mid, jax.random.PRNGKey(2), daily)
    np.testing.assert_array_equal(
        np.asarray(stream.env.get_obs(again, base)[TO_DOWN : TO_UP + 1]),
        np.asarray(stream.env.get_obs(mid, base)[TO_DOWN : TO_UP + 1]),
    )
    # Drawn after the day's program ended: announced anew.
    done = _later(first, int(np.ceil(end_h * HOUR_STEPS)) + 1, stream.env)
    assert float(stream.env.get_obs(done, base)[TO_UP]) == 0.0
    anew = stream.env.start_event(done, jax.random.PRNGKey(3), daily)
    assert float(stream.env.get_obs(anew, base)[TO_DOWN]) > 0.0


def test_the_plan_after_an_onset_is_the_same_however_long_the_plant_has_run(
    stream, run
):
    daily = _params(stream, "edf_daily")
    seen = []
    for onset in (40, 40 + 2**22):
        s, base = _reset(stream, onset)
        s = stream.env.start_event(s, jax.random.PRNGKey(1), daily)
        seen.append(run(stream, s, base, 3 * HOUR_STEPS))
    np.testing.assert_array_equal(seen[0], seen[1])


def test_a_stream_starts_with_no_ramp_planned(stream, run):
    quiet = StreamEnv(
        stream.env,
        stream.default_params,
        [dataclasses.replace(s, events_per_h=0.0) for s in stream.scenarios],
    )
    for seed in range(6):
        _, state = quiet.reset(jax.random.PRNGKey(seed), quiet.default_params)
        p = quiet.scenario_params(quiet.default_params, state.scenario)
        p = p.replace(demand_sigma=0.0)
        # The window would start its ramp down within 2 h.
        out = run(stream, state.env, p, 4 * HOUR_STEPS)
        np.testing.assert_array_equal(out[:, 0], 1.0)
        np.testing.assert_array_equal(out[:, 1:], 0.0)


@pytest.mark.medium
def test_time_outside_the_offset_band_a_day_ago_stops_counting(stream):
    """Two hours outside the band (a zero-width band), then a day inside (a band
    wider than any offset): the derate stays charged while the 1 h allowance is
    spent within the last 24 h, and lifts once those hours are a day old."""
    from target_gym.reactor.experts import make_reactor_pid_controller

    expert = make_reactor_pid_controller()
    base = _params(stream, "baseload")
    _, s = stream.env.reset_env(jax.random.PRNGKey(0), base)

    @jax.jit
    def roll(state, memory, obs, params, keys):
        def body(carry, k):
            s, m, o = carry
            a, m = expert.act(m, o)
            o, s, _, _, info = stream.env.step_env(k, s, a, params)
            return (s, m, o), (info["reward_terms"]["tracking"], info["tripped"])

        return jax.lax.scan(body, (state, memory, obs), keys)

    obs = stream.env.get_obs(s, base)
    keys = jax.random.split(jax.random.PRNGKey(4), 26 * HOUR_STEPS)
    (s, m, obs), (cost_out, trip_out) = roll(
        s, expert.init_state, obs, base.replace(ao_band=0.0), keys[: 2 * HOUR_STEPS]
    )
    _, (cost_in, trip_in) = roll(
        s, m, obs, base.replace(ao_band=1.0), keys[2 * HOUR_STEPS :]
    )
    assert not bool(trip_out.any()) and not bool(trip_in.any())
    cost = np.concatenate([np.asarray(cost_out), np.asarray(cost_in)])
    hours = np.arange(len(cost)) // HOUR_STEPS
    # The derate: 1 GWe x (n - 0.5) at $80/MWh over a 30 s step, about $333.
    derate = 1000.0 * 80.0 / 3600.0 * 0.5 * 30.0
    assert np.all(cost[(hours >= 2) & (hours < 24)] > 0.8 * derate)
    assert np.all(cost[hours >= 25] < 0.2 * derate)


# ---------------------------------------------------------------------------
# Restarts and refuelling (long-strata R8; owner, 2026-11-11).
#
# Contract:
# - a trip restarts the plant back on its plan once the lump's time has passed:
#   at the plan's load, lined out on its program, rods parked, critical on boron
#   with the burnup since the last refuelling diluted, and with iodine and xenon
#   (means and tilts) as that time at zero power leaves them: 48 h after a scram
#   (``restart_steps``), 15 h after a timed limit's hot standby (``standby_steps``);
# - a stream refuels the core every ``renewal_days``: renewed at its cycle point
#   as at reset, uncharged, the outside world carrying on; a window never does.
#
# Oracles: the zero-power iodine and xenon equations (PHYSICS.md section 1, the
# point form at n = 0) integrated numerically; the lump's times; and a critical
# core keeps its power with the rods parked and no boron flow.
# ---------------------------------------------------------------------------


def _zero_power(I, Xe, seconds, p):
    """Iodine and xenon after ``seconds`` with no flux, by RK45 on
    dI/dt = -l_I I, dXe/dt = (l_X + s) / (1 + r) I - l_X Xe."""
    from scipy.integrate import solve_ivp

    from target_gym.reactor.env import LAMBDA_IODINE, LAMBDA_XENON

    feed = (LAMBDA_XENON + p.sigma_phi0) / (1.0 + p.gamma_ratio)

    def rates(_, y):
        return [-LAMBDA_IODINE * y[0], feed * y[0] - LAMBDA_XENON * y[1]]

    sol = solve_ivp(rates, (0.0, seconds), [I, Xe], rtol=1e-10, atol=1e-12)
    return sol.y[0, -1], sol.y[1, -1]


def _tripped(stream, standby=False, days=3.0):
    """A baseload plant ``days`` into its fuel's cycle, with a load-follow day
    under way and a tilted xenon, proposed past its envelope."""
    s, base = _reset(stream, int(days * 24 * HOUR_STEPS))
    s = stream.env.start_event(s, jax.random.PRNGKey(1), _params(stream, "edf_daily"))
    s = s.replace(I_tilt=jnp.float32(0.004), Xe_tilt=jnp.float32(-0.003))
    if standby:  # Tavg under its floor for longer than allowed: hot standby
        return s.replace(below_floor=jnp.float32(base.floor_allowed + 5.0)), base
    return s.replace(n=jnp.float32(1.2)), base  # past the high-flux trip: a scram


def _restart(stream, tripped, params):
    """``restart_plant`` as the failure kernel applies it (``base.failure_kernel``):
    the plant it returns, on the trip's clock, with the outside world carried."""
    env = stream.env
    kept = (*env.OUTSIDE_WORLD, *getattr(env, "RECORDS", ()))
    plant = env.restart_plant(jax.random.PRNGKey(5), tripped, params)
    return plant.replace(
        time=tripped.time,
        sensor_key=tripped.sensor_key,
        **{f: getattr(tripped, f) for f in kept},
    )


@pytest.mark.parametrize("standby", [False, True])
def test_a_trip_restarts_with_the_xenon_its_dead_time_leaves(stream, standby):
    tripped, base = _tripped(stream, standby)
    assert bool(stream.env.is_terminated(tripped, base))
    restarted = _restart(stream, tripped, base)
    steps = base.standby_steps if standby else base.restart_steps
    seconds = steps * 30.0  # the lump's time, in 30 s control steps
    I, Xe = _zero_power(float(tripped.I_hat), float(tripped.Xe_hat), seconds, base)
    It, Xt = _zero_power(float(tripped.I_tilt), float(tripped.Xe_tilt), seconds, base)
    np.testing.assert_allclose(float(restarted.I_hat), I, rtol=1e-4)
    np.testing.assert_allclose(float(restarted.Xe_hat), Xe, rtol=1e-4)
    np.testing.assert_allclose(float(restarted.I_tilt), It, rtol=1e-3, atol=1e-7)
    np.testing.assert_allclose(float(restarted.Xe_tilt), Xt, rtol=1e-3, atol=1e-7)
    # From balance at full power: 48 h out, xenon is about a fifth of balance and
    # builds back; 15 h out, it is still above balance and burns out.
    assert (Xe < 0.3) if not standby else (Xe > 1.2)


def test_a_restarted_plant_is_on_its_plan_lined_out_and_critical(stream):
    tripped, base = _tripped(stream, days=12.0)
    restarted = _restart(stream, tripped, base)
    obs = np.asarray(stream.env.get_obs(restarted, base))
    # At the plan's load where the day stands, carried over the trip.
    assert float(restarted.n) == pytest.approx(float(tripped.target_n), abs=0.02)
    assert obs[4] == pytest.approx(obs[5], abs=0.05)  # Tavg on its program
    assert obs[2] == pytest.approx(float(PARKED[0]), abs=1e-6)  # rods parked
    # Critical: with the rods parked and no boron flow, the power holds over a
    # minute, the burnup of 12 days diluted in its boron. Uncompensated, those
    # 12 days are about 270 pcm, which moves the power by percents in a minute.
    step = jax.jit(stream.env.step_env)
    s = restarted.replace(target_n=jnp.asarray(float(restarted.n), jnp.float32))
    for k in range(2):
        obs, s, _, _, info = step(jax.random.PRNGKey(k), s, PARKED, base)
        assert not bool(info["tripped"])
    assert float(s.n) == pytest.approx(float(restarted.n), abs=2e-3)


def test_the_kernel_restarts_a_trip_through_the_restart(stream):
    tripped, base = _tripped(stream)
    # The whole primary and the steam side at 700 K: the hot leg stays past its
    # 610 K limit through the step's first sub-step, a scram.
    hot = jnp.float32(700.0)
    before = tripped.replace(n=jnp.float32(1.0), T_coolant=hot, T_cold=hot, T_sg=hot)
    _, after, _, _, info = jax.jit(stream.env.step_env)(
        jax.random.PRNGKey(0), before, jnp.array([1.0, -1.0]), base
    )
    assert bool(info["tripped"])
    assert float(after.Xe_hat) < 0.3  # a scram's 48 h of decay, not a fresh balance
    assert float(after.rod_in) == pytest.approx(3.0 / 228.0, abs=1e-6)


def test_a_stream_refuels_the_core_at_its_cycle_point_and_a_window_never(stream):
    from target_gym.reactor.env import burnt

    renew = _params(stream, "baseload")
    assert renew.renewal_days > 0
    per = int(round(renew.renewal_days * 24 * HOUR_STEPS))
    s, _ = _reset(stream, per - 1)
    s = s.replace(Xe_hat=jnp.float32(1.1))  # a xenon no reset would draw
    assert float(burnt(s, renew)) > 10.0  # ppm of burnup since the stream began
    step = jax.jit(stream.env.step_env)
    _, renewed, *_ = step(jax.random.PRNGKey(0), s, PARKED, renew)
    assert int(renewed.time) == per and float(burnt(renewed, renew)) == 0.0
    assert float(renewed.Xe_hat) <= 1.0 + 1e-6  # balanced at a recent 0.9 to 1
    assert int(renewed.physics_time) == int(s.physics_time) + 6  # the world goes on
    window = stream.env.default_params.replace(demand_sigma=0.0)
    assert window.renewal_days == 0.0
    _, kept, *_ = step(jax.random.PRNGKey(0), s, PARKED, window)
    assert float(burnt(kept, window)) > 10.0
