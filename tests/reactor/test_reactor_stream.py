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
#: An envelope nothing reaches, so no trip restarts the plant (a restart reloads
#: the turbine, which moves the load off the plan).
WIDE = dict(
    n_max=50.0,
    T_coolant_max=5000.0,
    T_fuel_max=1.0e5,
    floor_allowed=1.0e9,
    ril_allowed=1.0e9,
    dnb_allowed=1.0e9,
)


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
    # Stepped on baseload's params, with the envelope lifted so the parked rods
    # cannot trip the plant (owner's approval, 2026-11-12).
    out = run(stream, s, base.replace(**WIDE), 15 * HOUR_STEPS)
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
        # Burnup counts from the jumped clock, as a refuelling there would leave
        # it, so the core is not tripped by years of it (owner's approval,
        # 2026-11-12).
        s = s.replace(burn_t0=s.physics_time)
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
# - a trip restarts the plant once the lump's dead time has passed: at the restart
#   load (the turbine's reload, below), lined out on its program, rods parked,
#   critical on boron with the burnup since the last refuelling diluted, and with
#   iodine and xenon (means and tilts) as that time at zero power leaves them: 48 h
#   after a scram (``restart_steps``), 6 h after a timed limit's hot standby
#   (``standby_steps``; owner, 2026-11-12);
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
    # builds back; 6 h out, it is near its peak, above balance, and burns out.
    assert (Xe < 0.3) if not standby else (Xe > 1.2)


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


# ---------------------------------------------------------------------------
# The turbine's reload after a restart (re-record pass, step 2; owner,
# 2026-11-11).
#
# Contract:
# - a trip's restart brings the plant back at reduced power, lined out on its
#   program at that load and critical with the rods parked, and the turbine
#   reloads: it takes at most a pickup that starts at the restart load and climbs
#   at the fuel's conditioning ramps until it meets the plan, then follows it;
# - the plan the turbine does not take while it reloads is lost output, charged
#   by the dynamics at spot, as the trip's lump is (R8: the lump prices the dead
#   time, the simulation the reload, once).
#
# Oracles: Doel-3 held its return at 15 % after a 6 h scram (INIS 1ypr9-dk595);
# conditioning ramps of 3 %/h above 50 % of rated power (KNS 15A-200) and Exelon's
# practice of about 15 %/h below (REALITY.md [ST3]); 1 GWe at $80/MWh; the hand
# arithmetic in each test.
# ---------------------------------------------------------------------------

RESTART_LOAD = 0.15
KNEE = 0.5
FAST, SLOW = 0.15, 0.03  # per hour


def _restarted_on_baseload(stream, standby, days=12.0):
    """A baseload plant with no day planned, tripped ``days`` into its cycle and
    restarted: the plan stays at full load."""
    s, base = _reset(stream, int(days * 24 * HOUR_STEPS))
    if standby:
        s = s.replace(below_floor=jnp.float32(base.floor_allowed + 5.0))
    else:
        s = s.replace(n=jnp.float32(1.2))
    return _restart(stream, s, base), base


@pytest.mark.parametrize("standby", [False, True])
def test_a_restart_comes_back_at_the_restart_load_lined_out_and_critical(
    stream, standby
):
    restarted, base = _restarted_on_baseload(stream, standby)
    obs = np.asarray(stream.env.get_obs(restarted, base))
    assert float(restarted.n) == pytest.approx(RESTART_LOAD, abs=0.005)
    assert obs[LOAD] == pytest.approx(RESTART_LOAD, abs=1e-6)  # the turbine's load
    assert obs[4] == pytest.approx(obs[5], abs=0.05)  # Tavg on its program there
    assert obs[2] == pytest.approx(float(PARKED[0]), abs=1e-6)  # rods parked
    # Critical: the pickup held, rods parked and no boron flow, the power holds
    # over a minute, the burnup of 12 days diluted in its boron. Uncompensated,
    # those 12 days are about 270 pcm, which moves the power by percents.
    held = base.replace(reload_fast_rate=0.0, reload_rate=0.0)
    step = jax.jit(stream.env.step_env)
    s = restarted
    for k in range(2):
        obs, s, _, _, info = step(jax.random.PRNGKey(k), s, PARKED, held)
        assert not bool(info["tripped"])
    assert float(s.n) == pytest.approx(float(restarted.n), abs=2e-3)


def test_the_turbine_reloads_at_the_conditioning_ramps_until_it_meets_the_plan(
    stream, run
):
    restarted, base = _restarted_on_baseload(stream, standby=True)
    out = run(stream, restarted, base.replace(**WIDE), 20 * HOUR_STEPS)
    load, t = out[:, 0], (np.arange(len(out)) + 1) / HOUR_STEPS  # h after restart
    # 15 % to the 50 % knee at 15 %/h: 2.33 h; then 3 %/h to full load: 16.67 h.
    knee_h = (KNEE - RESTART_LOAD) / FAST
    full_h = knee_h + (1.0 - KNEE) / SLOW
    fast = (t > 0.05) & (t < knee_h - 0.05)
    slow = (t > knee_h + 0.5) & (t < full_h - 0.5)
    assert np.polyfit(t[fast], load[fast], 1)[0] == pytest.approx(FAST, rel=1e-3)
    assert np.polyfit(t[slow], load[slow], 1)[0] == pytest.approx(SLOW, rel=1e-3)
    met = t[np.argmax(load >= 1.0)]
    assert met == pytest.approx(full_h, abs=2.0 / HOUR_STEPS)
    np.testing.assert_array_equal(load[t >= met], 1.0)  # on the plan from there


def test_a_reload_that_meets_the_plan_follows_it(stream, run):
    """Tripped in a load-follow day's 50 % hold: the pickup meets the plan at the
    knee, and the plan's own ramp back up then runs at the day's 2 %/min."""
    s, base = _reset(stream, int(12 * 24 * HOUR_STEPS))
    daily = _params(stream, "edf_daily")
    s = stream.env.start_event(s, jax.random.PRNGKey(1), daily)
    obs = stream.env.get_obs(s, base)
    to_down, to_up = float(obs[TO_DOWN]), float(obs[TO_UP])
    held = int((to_down + 0.5) * HOUR_STEPS)  # in the hold: past the 25 min ramp
    s = _later(s, held, stream.env).replace(n=jnp.float32(1.2))
    restarted = _restart(stream, s, base)
    hours = to_up - held / HOUR_STEPS + 1.0  # through the plan's ramp back up
    out = run(stream, restarted, base.replace(**WIDE), int(hours * HOUR_STEPS))
    load, t = out[:, 0], (np.arange(len(out)) + 1) / HOUR_STEPS
    knee_h = (KNEE - RESTART_LOAD) / FAST
    up_h = to_up - held / HOUR_STEPS  # the plan's ramp up, from the restart
    assert float(load[0]) < 0.2
    hold = (t > knee_h + 2.0 / HOUR_STEPS) & (t < up_h)
    np.testing.assert_allclose(load[hold], 0.5, atol=1e-6)
    rising = (t > up_h + 0.05) & (t < up_h + 0.35)
    slope = np.polyfit(t[rising], load[rising], 1)[0]
    assert slope == pytest.approx(0.02 * 60.0, rel=1e-3)  # 2 %/min, per hour
    assert float(load[-1]) == pytest.approx(1.0, abs=1e-6)


def test_the_reload_s_lost_output_is_charged_once(stream):
    """In tracking, as the offset's derate is (owner, 2026-11-12)."""
    from target_gym.reactor.env import NOT_RELOADING

    restarted, base = _restarted_on_baseload(stream, standby=True)
    wide = base.replace(**WIDE)

    @jax.jit
    def roll(state, keys):
        def body(s, k):
            obs, s, _, _, info = stream.env.step_env(k, s, PARKED, wide)
            off = stream.env.reward_terms(s.replace(pickup=NOT_RELOADING), wide)
            reload = stream.env.reward_terms(s, wide)["tracking"] - off["tracking"]
            out = (info["reward_terms"]["tracking"], reload, obs[LOAD], info["tripped"])
            return s, out

        return jax.lax.scan(body, state, keys)[1]

    keys = jax.random.split(jax.random.PRNGKey(2), 20 * HOUR_STEPS)
    step_cost, reload, load, tripped = (
        np.asarray(x, np.float64) for x in roll(restarted, keys)
    )
    assert not tripped.any()
    per_sub = 1000.0 * 80.0 / 3600.0 * 5.0  # $ per 5 s sub-step at full shortfall
    # The first step, Tavg still inside its dead band: the pickup climbs
    # 0.15/3600 x 5 s a sub-step from 0.15, and each of the 6 sub-steps charges
    # 1 - pickup: 6 x 0.85 - 21 x 2.083e-4.
    climb = FAST / 3600.0 * 5.0
    assert step_cost[0] == pytest.approx(per_sub * (6 * 0.85 - 21 * climb), rel=1e-4)
    # At every step's end, what the reload adds to a sub-step is the plan (full
    # load) the turbine does not take, and nothing once it has met the plan.
    np.testing.assert_allclose(reload, per_sub * (1.0 - load), rtol=1e-4, atol=1e-3)
    assert reload[-HOUR_STEPS:].max() == 0.0
    # The whole reload: 2.33 h at a mean shortfall of 0.675 and 16.67 h at 0.25,
    # 5.742 h of full output at $80,000 an hour. Summed at the steps' ends, the
    # pickup leads each step's mean by 2.5 sub-steps: under 1e-3 of the total.
    lost_h = (0.35 / FAST) * 0.675 + (0.5 / SLOW) * 0.25
    assert 6.0 * reload.sum() == pytest.approx(lost_h * 80000.0, rel=2e-3)


# ---------------------------------------------------------------------------
# The dispatch's rate and depth (re-record pass, step 2; owner, 2026-11-12).
#
# Contract: the observation carries the day's program as a dispatch gives it to
# an operator: the hours to each ramp (obs 8, 9), the ramp's rate in % of rated
# load a minute (obs 11) and the day's minimum load (obs 12), the day's own,
# whatever scenario the stream has moved to. Oracles: the scenarios' declared
# rates and minima (REALITY.md section 6).
# ---------------------------------------------------------------------------

RATE, LOW = 11, 12


@pytest.mark.parametrize(
    "scenario, per_min, low",
    [
        ("edf_daily", 2.0, 0.5),
        ("krsko_slow", 8.0 / 60.0, 0.5),
        ("krsko_shallow", 4.0 / 60.0, 0.75),
        ("fast_return", 5.0, 0.5),
    ],
)
def test_the_observation_carries_the_day_s_rate_and_minimum(
    stream, scenario, per_min, low
):
    s, base = _reset(stream)
    s = stream.env.start_event(s, jax.random.PRNGKey(1), _params(stream, scenario))
    obs = np.asarray(stream.env.get_obs(s, base))  # read on baseload's params
    assert obs.shape == (13,)
    assert obs[RATE] == pytest.approx(per_min, rel=1e-5)
    assert obs[LOW] == pytest.approx(low, abs=1e-6)
    # Plan values, not sensors: the instruments' noise leaves them exact.
    names = [sc.name for sc in stream.scenarios]
    noisy = stream.scenario_params(stream.default_params, names.index("baseload"))
    read = np.asarray(stream.env.get_obs(s, noisy))
    assert read[0] != obs[0]  # the power reading scatters
    np.testing.assert_array_equal(read[RATE:], obs[RATE:])
