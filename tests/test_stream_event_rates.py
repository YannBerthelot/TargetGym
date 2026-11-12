"""A stream delivers each task's events at their sourced rate (re-record pass, step 2,
"Absorbed event draws"; owner, 2026-11-11).

Contract, from the owner's decision and each task's scenarios and REALITY.md, not from
the code:

- a stream's events arrive at the rate their sources give, counted per hour spent in
  the scenarios that bring them. A draw while an event of its kind is under way is
  absorbed and is not an event, so it does not count;
- glass_furnace: gas swings come 1.5 a day in a blended-gas week ([ST1]: LNG three
  times in about two days); planned pull changes come one a week-long pull regime
  (ours: "a planned change and its return in one visit");
- reactor: one load-follow day a load-follow visit of 24 h. That is EDF's 250 h a
  year off full power at 6.33 h a day, 39.5 days a year, over the load-follow share's
  39.5 visits a year (REALITY.md sections 7 and 8);
- boiler_drum: one ID fan runback or feed pump trip a 2.775 h upset visit, so 0.5
  runbacks and one feed pump trip a unit-year at the rows' weights ([ST1, ST2]);
- a scoring run pinned to a scenario (R2) sees its events at the stream's density in
  that scenario, per scored hour, so that J's per-scenario gains see the stream's
  events (owner, 2026-11-12). Boiler upsets are the known exception: a pinned upset
  regime hosts at most one upset per 3 h busy, 0.33 an hour against the stream's
  0.36, and keeps the stream's draws, 0.71 of it (owner, 2026-11-12).

Oracle: the sourced rates above. A stream's events are the world's and do not depend
on the policy (``OUTSIDE_WORLD``, tests/test_stream_tasks.py), so a held action drives
the plant. A drawn step is an event when it moves the start of the event of its kind:
the time a day or a pull change is announced, or a swing or an upset starts, which
each event carries in the state (long-strata S1). A rebase moves a clock, never that
time.

Tolerance: four standard errors of the pooled rate over independent seeds, plus 2 %
for what the correction leaves out: it takes each visit to start with nothing of its
kind under way, and an event that runs on into the next visit of its kind absorbs a
little more (1 to 2 % on these suites, from an event-only simulation). Each run is
long enough that the tolerance stays under 20 % of the rate, so the shortfall the
correction removed (24 % for the reactor to 46 % for the gas swings) cannot pass. A
pinned run has no next visit, so its allowance is 1 %, for its fixed count at uniform
times against Poisson draws (0.6 % on the glass gas run).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import scenarios
from target_gym.registry import REGISTRY, control_step_seconds
from target_gym.stream import StreamEnv, make_stream

pytestmark = pytest.mark.large

#: A drawn step moved an event's start by more than this, in hours. Float32 rounding
#: of the starts is under 0.001 h on these runs; the nearest real move is an upset's
#: 2.775 h hold.
MOVED_H = 0.05
#: A start further on than this, in plant hours, stands for none yet: the windows'
#: own events sit at 1e9. Each glass rebase moves such a start by float32 rounding,
#: tens of hours at 1e9, and that is not an event.
NONE_H = 1.0e6


def _boiler_onset(s, p):
    """When the running upset struck, in plant hours."""
    return (s.plan_t0 * p.delta_t + s.event_time) / 3600.0


def _reactor_onset(s, p):
    """When the planned day's ramp down starts, in plant hours."""
    return (s.plan_t0 * p.delta_t + s.ramp_down_at) / 3600.0


def _glass_onset(s, p):
    """When each kind's change (pull, gas) starts, in plant hours."""
    return s.event_t0 * p.delta_t / 3600.0 + s.events


def _measure(stream, onset, seeds, steps, first_visit=False, seed=20261112):
    """Per seed and scenario, the hours spent and the events started, under a held
    action. With ``first_visit`` only the visit a seed starts in counts, up to its
    first switch."""
    params = stream.default_params
    n = len(stream.scenarios)
    action = jnp.zeros(stream.action_space(params).shape)
    dt_h = control_step_seconds(stream.env, params) / 3600.0

    def one(key):
        _, state = stream.reset(key, params)

        def body(carry, _):
            state, alive, hours, started = carry
            _, new, _, _, info = stream.step_env(key, state, action, params)
            start = onset(new.env, params)
            moved = jnp.abs(start - onset(state.env, params)) > MOVED_H
            got = info["event"] & jnp.any(moved & (start < NONE_H))
            # A scoring run counts its scored part only; a free stream, every step.
            live = info["scored"] & ~stream.is_truncated(state, params)
            hot = jax.nn.one_hot(info["scenario"], n) * alive * live
            if first_visit:
                alive = alive & (new.scenario == state.scenario)
            return (new, alive, hours + hot * dt_h, started + hot * got), None

        zero = jnp.zeros(n)
        carry = (state, jnp.array(True), zero, zero)
        return jax.lax.scan(body, carry, None, length=steps)[0][2:]

    keys = jax.random.split(jax.random.PRNGKey(seed), seeds)
    hours, started = jax.jit(jax.vmap(one))(keys)
    return np.asarray(hours, np.float64), np.asarray(started, np.float64)


def _miss(label, n, h, rate, allowance):
    """The pooled rate of ``n`` events in ``h`` hours, one each a seed, against
    ``rate``: the miss as text, or None."""
    est = n.sum() / h.sum()
    se = np.sqrt(((n - est * h) ** 2).sum()) / h.sum()
    tol = 4.0 * se + allowance * rate
    line = (
        f"{label}: {n.sum():.0f} events in {h.sum():.0f} h, {est:.5f}/h, "
        f"sourced {rate:.5f}/h, tolerance {tol:.5f}"
    )
    if tol >= 0.2 * rate:
        return f"{line}: the run is too short to tell"
    return line if abs(est - rate) > tol else None


def _shortfalls(stream, hours, started, groups, allowance=0.02):
    """Each group's pooled rate against its sourced rate; the misses, as text."""
    names = [s.name for s in stream.scenarios]
    misses = []
    for label, (rows, rate) in groups.items():
        cols = [names.index(r) for r in rows]
        n, h = started[:, cols].sum(1), hours[:, cols].sum(1)
        misses.append(_miss(label, n, h, rate, allowance))
    return [m for m in misses if m]


def test_glass_gas_swings_and_pull_changes_arrive_at_their_rates():
    stream = make_stream("glass_furnace")
    hours, started = _measure(stream, _glass_onset, seeds=128, steps=250_000)
    groups = {
        "gas swings": (("gas_rich", "gas_lean"), 1.5 / 24.0),
        "pull changes": (("pull_up", "pull_down"), 1.0 / 168.0),
    }
    assert not _shortfalls(stream, hours, started, groups)


def test_reactor_load_follow_days_arrive_at_edf_s_count():
    stream = make_stream("reactor")
    hours, started = _measure(stream, _reactor_onset, seeds=64, steps=260_000)
    days = ("edf_daily", "edf_short_hold", "krsko_slow", "krsko_shallow", "fast_return")
    groups = {"load-follow days": (days, 1.0 / 24.0)}
    assert not _shortfalls(stream, hours, started, groups)


def _fresh_upset_visits():
    """The boiler's two upset rows as the shipped suite runs them, each seed starting
    in one with nothing under way and leaving it, at its first switch, for a quiet
    row. The shipped suite spends 4.2 h a year in them, so a run that saw hundreds
    of upsets would be millions of plant hours; there too each visit starts long
    after the last upset ended."""
    spec = REGISTRY["boiler_drum"]
    rows = {s.name: s for s in scenarios.suite(spec)}
    upsets = [
        dataclasses.replace(rows[r], weight=0.49)
        for r in ("id_fan_runback", "feed_pump_trip")
    ]
    quiet = dataclasses.replace(
        rows["normal"], weight=0.02, dwell_h=1e-3, events_per_h=0.0
    )
    return StreamEnv(spec.make_env(), spec.make_test_params(), (*upsets, quiet))


def test_boiler_upsets_arrive_one_a_visit():
    stream = _fresh_upset_visits()
    # Six mean visits a seed, 2 s steps: a visit outlasts the run once in 400.
    hours, started = _measure(
        stream, _boiler_onset, seeds=1024, steps=30_000, first_visit=True
    )
    groups = {"upsets": (("id_fan_runback", "feed_pump_trip"), 1.0 / 2.775)}
    assert not _shortfalls(stream, hours, started, groups)


def _pinned(task, name, onset, seeds, b_min_h, scored_h):
    """Scoring runs pinned to scenario ``name`` (R2), the scored part long enough to
    read the pinned gain's event density; their hours and events, per seed and
    scenario, scored part only."""
    spec = REGISTRY[task]
    env, params = spec.make_env(), spec.make_test_params()
    rows = scenarios.suite(spec)
    pin = [s.name for s in rows].index(name)
    dt_h = control_step_seconds(env, params) / 3600.0
    b_min, scored = round(b_min_h / dt_h), round(scored_h / dt_h)
    # Room for the planned count, whichever of the row's rates the run plans from.
    pinned = getattr(rows[pin], "pinned_events_per_h", None) or 0.0
    fastest = max(rows[pin].events_per_h, pinned)
    stream = StreamEnv(
        env,
        params,
        rows,
        pin=pin,
        b_min=b_min,
        scored=scored,
        max_events=math.ceil(fastest * scored_h) + 1,
    )
    return stream, _measure(stream, onset, seeds, 2 * b_min + scored)


def _pinned_miss(task, name, onset, burn_h, scored_h, rate):
    """``name``'s pinned runs against ``rate``: the miss or None, and per seed the
    events and scored hours."""
    stream, (hours, started) = _pinned(task, name, onset, 32, burn_h, scored_h)
    col = [s.name for s in stream.scenarios].index(name)
    n, h = started[:, col], hours[:, col]
    return _miss(f"pinned {name}", n, h, rate, 0.01), n, h


def test_glass_pinned_runs_see_the_stream_s_gas_swings_and_pull_changes():
    # gas_lean and pull_down share these rows' rates and busy times.
    misses = [
        _pinned_miss(
            "glass_furnace", "gas_rich", _glass_onset, 50.0, 2_000.0, 1.5 / 24
        )[0],
        _pinned_miss(
            "glass_furnace", "pull_up", _glass_onset, 200.0, 10_000.0, 1 / 168
        )[0],
    ]
    assert not [m for m in misses if m]


def test_reactor_pinned_runs_see_the_stream_s_load_follow_days():
    # krsko_shallow shares krsko_slow's rate and busy time.
    rows = ("edf_daily", "edf_short_hold", "krsko_slow", "fast_return")
    runs = [
        _pinned_miss("reactor", r, _reactor_onset, 30.0, 1_000.0, 1 / 24) for r in rows
    ]
    # Pooled over the rows, the runs resolve a few percent.
    n = np.concatenate([run[1] for run in runs])
    h = np.concatenate([run[2] for run in runs])
    misses = [run[0] for run in runs] + [_miss("pinned days", n, h, 1 / 24, 0.01)]
    assert not [m for m in misses if m]
