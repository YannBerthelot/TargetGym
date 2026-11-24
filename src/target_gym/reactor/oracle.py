"""The reactor's oracle (the MPC slot): one plan for the day ahead, solved again at
every knot from the true state.

The oracle is the problem definition solved (charter J3). The task's trade is slow:
how fast the turbine may load is set by the boron line (5 to 15 min late, hours to
dilute late in the cycle), by the xenon the day's own power history builds and
burns, and by what the rods may do to the axial offset. A planner that sees five
minutes sees none of it. This one plans the next ``LOOK_HOURS`` (the whole day of a
window, shrinking to its end) on the knot model of ``oracle_model.py``, by the
shared interior-point solver (``oracles.shooting``), and applies its first knot.

The planner problem:

- moves, one of each a knot: the rods' travel in and out, the dilution and the
  boration *arriving* in the core, the turbine's load change. The oracle sends a
  flow one line's delay before the knot it is planned to arrive on, so the line's
  delay is in the data (what is already in the line), not in the state. Travel
  and flows are split by direction so their costs are linear and no move has a
  corner at rest;
- cost, the env's own terms in dollars: the plan the turbine does not take at the
  lost-output price, the rods' travel, the effluent, and Tavg past its free band.
  The plan's Tavg is the model's on the wobble's mean path, and the plant's wobbles
  about it with the load, so the plan pays the mean of that cost over the wobble
  (``oracle_model.noisy_excess``): a plan that rode the band's edge paid for
  every crossing on replay (a baseload day lost to the reference, 2026-11-15);
- limits, each backed off by the model's error on it: the turbine under its plan
  (or its conditioning ramps) and under what its valves pass wide open on the
  steam it has, the rods inside their insertion limit, Tavg between
  its floor and its DNB limit, no further under its program than the turbine's
  loading stop, the steam under the safety valves' set temperature, delta I inside
  the acceptable-operation limits, and the offset's penalty time inside its hour. The timed limits (floor, DNB,
  insertion, the 30 min at 90 % and above) are planned as hard: a plan that used
  their allowances could be cheaper, so this bounds the best cost from above;
- the future: the plan's load from the state (the day it carries), the burnup, and
  the load's wobble: its mean path, or with ``seed`` the path the episode's key
  draws, averaged over each knot (the ceiling's foresight). Foreseen only at the
  knots it did worse than on its mean path (2026-11-15): it chased a 5 min sample
  of a noise that turns over in 5 min;
- the tail: a window is a cut in a plant that runs on, so the plan runs
  ``TAIL_HOURS`` past the cut under the same costs, as the glass furnace's does.

The horizon protocol's T_c and P (``data/oracle_horizons/reactor.json``) describe
the rods' loop on Tavg, seconds long. They are kept as the record of that loop;
this planner's knot and look-ahead are the slow loops' and are stated here.
"""

from __future__ import annotations

import functools
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import target_gym.reactor.oracle_model as M
from target_gym.oracles.controller import enable_x64, float64
from target_gym.oracles.shooting import Limit, Shooting, StageModel
from target_gym.reactor.env import (
    NOT_RELOADING,
    plan_seconds,
    planned_load,
    tavg_dnb_limit,
)
from target_gym.reactor.oracle_model import (
    BORATE,
    DILUTE,
    KNOT,
    ROD_IN,
    ROD_OUT,
    TURBINE,
    Day,
)
from target_gym.reward import trip_cost
from target_gym.utils import ou_decay, stationary_innovation

#: How far a plan looks, and how far past a window's cut it is charged.
LOOK_HOURS, TAIL_HOURS = 24.0, 3.0
#: A solve runs in segments of the shared solver's own iteration cap, each from
#: the plan the last one reached, up to this many: the solver hands back its start
#: when a line search fails, which after several hundred good iterations of a cold
#: solve of the day would throw them away. A segment that fails ends the solve on
#: the plan before it.
SEGMENT, COLD_SEGMENTS, WARM_SEGMENTS = 150, 8, 3
#: A plan is dropped for a cold solve when the turbine is this far, of rated load,
#: from where the plan had it: a trip's restart, not a model error.
JUMP = 0.05
#: Knots between two solves: the plan is followed for half an hour, which the
#: model holds to the plant (``tests/reactor/test_reactor_oracle_model.py``).
REPLAN = 6
#: The planner's worst error on Tavg (K) and on delta I (of rated) at a knot: what
#: its limits are backed off by (the smoothing protocol's K11). Measured by
#: following a plan on the env with no solve in between
#: (``test_the_plant_follows_a_plan_as_the_model_says``): 0.68 K at the start of
#: the cycle, where a ramp's corner falls inside a knot before the boron arrives,
#: and 0.2 % of rated delta I; each rounded up.
TAVG_BACKOFF, OFFSET_BACKOFF = 0.8, 0.005
#: Minutes of the offset's hour the plan leaves unused, for the crossings it
#: times a knot apart.
CLOCK_BACKOFF = 10.0
#: The rounding of the penalty time's corners, of a knot.
WIDTH = 0.1
#: Tavg's room is the plant's own on both sides. Under its program the turbine
#: stops loading 10 F under (``ReactorParams.loading_stop``, the family's
#: interlock C-16): that is the plan's room there (``tavg_room`` overrides it).
#: Over its program the steam generators' safety valves relieve the steam at
#: their set pressure (``ReactorParams.relief_temperature``), which the model
#: does not play, so the plan keeps the steam under it. Both are backed off by
#: ``TAVG_BACKOFF``. Until the plant carried the valves (2026-11-17) the plan
#: kept Tavg inside its free band over the program on its own account.
#: The offset's hour is planned in two solves. With Tavg held close the plan
#: leans on the rods, rides the offset's band for hours, and the count of the
#: time past it is too sharp there for the solver, which stalls (2026-11-15: the
#: hour and a band of 1.5 K together; either alone converges). So a cold solve
#: first finds where the hour goes, with Tavg's room under its program at least
#: this wide (K) and the time counted (it converges); the knots that plan spends outside the
#: band are then given to every later solve as data, the band a limit on all the
#: others, and no time counted. A plan from given knots bounds the best cost
#: from above.
DISCOVERY_ROOM = 6.0
#: In percent of rated delta I, so the offset's rows are of order one.
PERCENT = 100.0


def stage_model(env, p, stages: int) -> StageModel:
    """The day's planner problem for the shared solver, ``stages`` knots long.
    Its data ``d`` holds the ``day`` ahead (``oracle_model.Day``), ``room``, the
    most the plan lets Tavg sit under its program before its back-off (K), and
    ``sigma``, the standard deviation of the load's wobble about the path the plan
    has for it."""
    period = env.control_period
    seconds = KNOT * period * float(p.delta_t)
    per_K = p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW / 3.6
    per_load = p.P_electric_GW * 1000.0 * p.derate_price_per_MWh / 3600.0
    rod_price = float(p.rod_step_price * p.rod_stroke_steps)
    flow_price = float(p.effluent_price * period * p.delta_t)  # $ per kg/s a step
    full = float(p.T_avg_full_load)

    def step(d, x, u, k):
        x1, rows = M.knot_step(
            p, d["day"], x, u, k, OFFSET_BACKOFF, WIDTH, period=period
        )
        error, T_avg, off, room, inside, under, steam, sway, relief = rows
        # Tavg wobbles with the load about the plan's, no finer than it is read.
        spread = jnp.maximum(d["sigma"] * sway, p.e_floor)
        flows = u[DILUTE] * p.dilution_flow + u[BORATE] * p.boration_flow
        cost = (
            per_load * seconds * 0.5 * (x[M.SHORT] + x1[M.SHORT])
            + rod_price * (u[ROD_IN] + u[ROD_OUT])
            + flow_price * M.planned_flows(d["day"], k) * flows
            + per_K * seconds * M.noisy_excess(error, p.e_tol, spread)
        )
        y = jnp.stack([
            under, x1[M.RODS], inside, T_avg - full,
            PERCENT * (room - off), PERCENT * (room + off), x1[M.CLOCK],
            d["room"] - error, relief, steam,
        ])  # fmt: skip
        return x1, cost, y

    price = float(trip_cost(p))
    limits = (
        Limit(lo=0.0, price=price),  # the turbine under its cap
        Limit(lo=0.0, price=price),  # the rods no further out than out
        Limit(lo=0.0, price=price),  # and inside their insertion limit
        Limit(float(p.T_avg_min) - full, float(tavg_dnb_limit(p)) - full, price,
              TAVG_BACKOFF),
        Limit(lo=0.0, price=price, backoff=PERCENT * OFFSET_BACKOFF),
        Limit(lo=0.0, price=price, backoff=PERCENT * OFFSET_BACKOFF),
        Limit(hi=(float(p.ao_allowed) - 60.0 * CLOCK_BACKOFF) / 3600.0, price=price),
        # Tavg's room under its program (the loading stop) and the steam's under
        # the safety valves' set temperature: the model plays neither, so the
        # plan stays clear of both. A kelvin past either costs the plant's whole
        # output for the knot: more than any load it buys, and not a trip, which
        # noise must not be planned as.
        Limit(lo=0.0, price=float(per_load * seconds), backoff=TAVG_BACKOFF),
        Limit(lo=0.0, price=float(per_load * seconds), backoff=TAVG_BACKOFF),
        Limit(lo=0.0, price=price),  # the turbine under what its valves pass
    )  # fmt: skip
    return StageModel(
        step=step, block=1, lo=(0.0, 0.0, 0.0, 0.0, -1.0), hi=(1.0,) * M.NU,
        limits=limits, move_scale=(0.1, 0.1, 1.0, 1.0, 1.0),
        tie=1e-3 / (M.NU * stages),
    )  # fmt: skip


def plant_x(state, p):
    """The planner's state of the true ``state``."""
    plan = planned_load(plan_seconds(state.physics_time, state, p), state)
    short = jnp.where(state.pickup > 0.5 * NOT_RELOADING, 0.0, plan - state.pickup)
    return jnp.stack([
        state.I_hat, state.Xe_hat, state.I_tilt * M.TILT, state.Xe_tilt * M.TILT,
        state.boron / M.PPM, state.rod_in, jnp.maximum(short, 0.0),
        state.ao_outside / 3600.0,
    ])  # fmt: skip


def look_ahead(state, p, knots: int, wobble, period: int = 6, free=None) -> Day:
    """What a solve from ``state`` knows of the next ``knots`` knots; ``wobble`` is
    the load's wobble at each of them, the first being now. ``free`` marks the
    knots the plan may spend outside the offset's band; with none given it may
    spend any, and counts the time itself."""
    sub = KNOT * period
    at = state.physics_time + sub * jnp.arange(knots + 1)
    plan = planned_load(plan_seconds(at, state, p), state)
    # After a restart the turbine loads no faster than the conditioning ramps.
    held = jnp.minimum(plan[0], state.pickup)

    def climb(held, target):
        rate = jnp.where(held < p.reload_knee, p.reload_fast_rate, p.reload_rate)
        nxt = jnp.minimum(held + rate * sub * p.delta_t, target)
        return nxt, nxt

    ramps = jnp.concatenate([held[None], jax.lax.scan(climb, held, plan[1:])[1]])
    delay = jnp.asarray(state.boron_delay, jnp.int32) - 1  # steps still in the line
    g = jnp.arange(3 * KNOT)
    arriving = state.boron_queue[jnp.clip(delay - 1 - g, 0, None)]
    return Day(
        plan=plan,
        cap=jnp.where(state.conditioning > 0.5, ramps, plan),
        wobble=wobble,
        burnt=state.burn_rate * (at - state.burn_t0) * p.delta_t,
        arriving=jnp.where(g < delay, arriving, 0.0),
        committed=delay,
        cycle0=state.cycle0,
        free=jnp.ones(knots + 1) if free is None else free,
        count=jnp.asarray(1.0 if free is None else 0.0),
    )


def first_guess(p, d: Day, x0, stages: int, period: int = 6):
    """The moves a cold solve starts from, by the model alone (never an expert,
    G8), each output brought to its target by the move that reaches it: the rods
    where delta I is on its target at the next knot (by bisection over the top
    half of the stroke, where their tilt grows with insertion); the flow that
    brings the balance's Tavg onto its program there, as far as the line carries
    it; and the turbine on its cap, or as far under it as keeps Tavg inside half
    its free band where the line cannot follow."""
    step = period * p.delta_t
    span = p.T_avg_full_load - p.T_avg_no_load
    fuel = p.P_thermal_ref * (0.5 / p.m_dot_cp + 1.0 / p.UA)  # K over Tavg at full load

    def one(x, k):
        go = lambda u: M.knot_step(  # noqa: E731
            p, d, x, u, k, OFFSET_BACKOFF, WIDTH, period=period
        )
        load = d.plan[k] - x[M.SHORT]
        reach = p.turbine_rate * KNOT * step
        u = jnp.zeros(M.NU, x.dtype)
        u = u.at[TURBINE].set(jnp.clip((d.cap[k + 1] - load) / reach, -1.0, 1.0))
        x1, rows = go(u)
        core = M.core_at(d.cycle0 + d.burnt[k + 1] / p.boron_ref, p)
        n1 = d.cap[k + 1] + d.wobble[k + 1]

        def halve(band, _):
            lo, hi = band
            mid = 0.5 * (lo + hi)
            off = M.offset(n1, mid, x1[M.XENON], x1[M.XE_TILT] / M.TILT, core)
            high = off > n1 * p.ao_ref  # more rods bring it down
            return (jnp.where(high, mid, lo), jnp.where(high, hi, mid)), None

        top = jnp.minimum(0.5, M.rod_insertion_limit(n1, p))
        (lo, hi), _ = jax.lax.scan(halve, (0.0 * top, top), jnp.arange(14))
        move = 0.5 * (lo + hi) - x[M.RODS]
        u = u.at[ROD_IN].set(jnp.maximum(move, 0.0))
        u = u.at[ROD_OUT].set(jnp.maximum(-move, 0.0))
        # Tavg's error is linear in the boron: the ppm to add that would close it.
        x1, rows = go(u)
        both = core.alpha_coolant + core.alpha_fuel
        need = -rows[0] * both / core.boron_worth
        steps = M.planned_flows(d, k)
        took = jnp.maximum(steps, 1) * step
        boron = x1[M.BORON] * M.PPM
        water = boron - M.mixed(boron, 1.0, 0.0, took, p)
        acid = M.mixed(boron, 0.0, 1.0, took, p) - boron
        planned = steps > 0
        u = u.at[DILUTE].set(jnp.where(planned, jnp.clip(-need / water, 0.0, 1.0), 0.0))
        u = u.at[BORATE].set(jnp.where(planned, jnp.clip(need / acid, 0.0, 1.0), 0.0))
        # And in the load: a unit of it cools Tavg under its program by the
        # program's span and the fuel's Doppler over both feedbacks.
        cold = go(u)[1][0] - 0.5 * p.e_tol
        per_load = span + core.alpha_fuel * fuel / both
        back = jnp.maximum(cold, 0.0) / per_load / reach
        u = u.at[TURBINE].set(jnp.clip(u[TURBINE] - back, -1.0, 1.0))
        return go(u)[0], u

    return jax.lax.scan(one, x0, jnp.arange(stages))[1]


def roll(p, d: Day, x0, moves, period: int = 6):
    """The model under ``moves`` from ``x0``: the states at every knot, the first
    being ``x0``, and each later knot's rows (``oracle_model.knot_step``)."""

    def one(x, ku):
        k, u = ku
        x1, rows = M.knot_step(p, d, x, u, k, OFFSET_BACKOFF, WIDTH, period=period)
        return x1, (x1, rows)

    ks = jnp.arange(moves.shape[0])
    xs, rows = jax.lax.scan(one, x0, (ks, moves))[1]
    return jnp.concatenate([x0[None], xs]), rows


class Followed(NamedTuple):
    """A plan being followed: the control step it starts on, its moves, the
    model's states and the turbine's load at every knot, and the control steps a
    flow spends in the line."""

    t0: int
    moves: np.ndarray
    states: np.ndarray
    load: np.ndarray
    delay: int


def action(p, plan: Followed, state, period: int = 6):
    """The raw action on the true ``state`` that follows ``plan``: every move on a
    straight line between its knots, as the model has it.

    - The rods are asked for where the line puts them at the end of the step.
    - A flow is sent one line's delay before the knot it is planned to arrive on.
    - The turbine is asked for the rate that puts its load on the planned line,
      and for its fastest where the plan keeps it on its plan, which is the most
      it takes."""
    t = int(state.time) - plan.t0
    k, i = divmod(t, KNOT)
    u, share = plan.moves[k], (i + 1) / KNOT
    rods = plan.states[k, M.RODS] + share * (u[ROD_IN] - u[ROD_OUT])
    arrives = plan.moves[(t + plan.delay) // KNOT]
    on = (float(state.physics_time) - float(state.plan_t0)) * float(p.delta_t)
    now = min(float(planned_load(on, _floats(state), np)), float(state.pickup))
    target = plan.load[k] + share * (plan.load[k + 1] - plan.load[k])
    turbine = (target - now) / (float(p.turbine_rate) * period * float(p.delta_t))
    if plan.states[k + 1, M.SHORT] < 1e-5:
        turbine = 1.0
    flow = arrives[BORATE] - arrives[DILUTE]
    return np.clip(np.array([1.0 - 2.0 * rods, flow, turbine]), -1.0, 1.0)


def _floats(state):
    """The day ``state`` carries, as floats for a host call of ``planned_load``."""
    names = ("ramp_down_at", "ramp_up_at", "plan_rate", "plan_depth")
    return type("Plan", (), {k: float(getattr(state, k)) for k in names})


def knot_means(path, sub: int):
    """The mean of ``path`` over the knot centred on each knot, by the trapezoid
    rule; the knots are ``sub`` samples apart, the first on the first sample, and
    the first and the last have the half knot inside the path. A plan holds one
    load a knot, so this is what it can follow of a wobble it foresees."""
    path, half, out = np.asarray(path, float), sub // 2, []
    for k in range(0, len(path), sub):
        w = path[max(k - half, 0) : min(k + half, len(path) - 1) + 1]
        out.append((w.sum() - 0.5 * (w[0] + w[-1])) / (len(w) - 1))
    return np.array(out)


def spread_left(window: float) -> float:
    """Of a wobble's spread, the share left about its mean over ``window``
    reversion times. The mean of an Ornstein-Uhlenbeck path over a window keeps
    ``2 (w - 1 + exp(-w)) / w^2`` of its variance, and the rest is about it."""
    kept = 2.0 * (window - 1.0 + np.exp(-window)) / window**2
    return float(np.sqrt(1.0 - kept))


@functools.partial(jax.jit, static_argnames=("n",))
def wobble_path(key, physics_time, wobble, n: int, theta, sigma, dt):
    """The load's wobble over the next ``n`` sub-steps as the env will draw it
    (``env.compute_next_state``), in the scored run's float32."""
    at = physics_time + 1 + jnp.arange(n)
    noise = jax.vmap(
        lambda i: jax.random.normal(jax.random.fold_in(key, i), dtype=jnp.float32)
    )(at)

    decay = ou_decay(theta, dt)
    innovation = stationary_innovation(sigma / jnp.sqrt(2.0 * theta), decay)

    def one(w, z):
        w = decay * w + innovation * z
        return w, w

    return jax.lax.scan(one, wobble, noise)[1]


class DayOracle:
    """The registry's oracle: ``step(obs, state)`` reads the true state.

    ``params`` are the planner's: with ``seed`` None the noise zeroed
    (``plan_params``) and the wobble planned on its mean path; with a seed the
    scored ones, and the wobble the episode's own. ``certificates`` keeps each
    solve's, and how far the last plan's next knot was from the plant."""

    oracle = True

    def __init__(
        self, env, params, seed=None, replan=REPLAN, tavg_room=None,
        segment=SEGMENT,
    ):  # fmt: skip
        self.env, self.foresight, self.replan = env, seed is not None, int(replan)
        self.period = int(env.control_period)
        self.raw = params
        hour = 3600.0 / (KNOT * self.period * float(params.delta_t))
        self.knots = -(-int(params.max_steps_in_episode) // KNOT)
        # No further ahead than the window is long: a short episode plans a
        # short problem.
        self.look = min(round(LOOK_HOURS * hour), self.knots)
        self.tail = round(TAIL_HOURS * hour)
        self.S = self.look + self.tail
        with enable_x64():
            self.p = p = float64(params)
            n, per = self.S + 1, self.period
            self.shooting = Shooting(
                stage_model(env, p, self.S), self.S, M.NX, max_iter=segment
            )
            room = float(p.loading_stop) if tavg_room is None else tavg_room
            self.room = float(room)
            self._x = jax.jit(lambda s: plant_x(s, p))
            self._day = jax.jit(lambda s, w, f=None: look_ahead(s, p, n, w, per, f))
            self._guess = jax.jit(lambda d, x: first_guess(p, d, x, n, per))
            self._roll = jax.jit(lambda d, x, u: roll(p, d, x, u, per))
        # The wobble's spread about the plan's path. ``plan_params`` zero its
        # noise for the model's mean path, so its size is the task's own; with the
        # wobble foreseen, the plan is given its mean over each knot
        # (``knot_means``) and pays for what that leaves (``spread_left``).
        task = params if float(params.demand_sigma) > 0.0 else type(params)()
        theta = float(task.demand_theta)
        self.sigma = float(task.demand_sigma) / (2.0 * theta) ** 0.5
        if self.foresight:
            knot = KNOT * self.period * float(params.delta_t)
            self.sigma *= spread_left(theta * knot)
        self.certificates: list[dict] = []
        self.reset()

    def reset(self):
        """Forget the plan: the next step solves cold."""
        self._plan, self._followed, self._rows = None, None, None
        self._outside: set[int] = set()  # the knots of the episode spent outside

    def _wobble(self, state):
        """The wobble at each knot ahead: the episode's own, or its mean path."""
        sub, p = KNOT * self.period, self.raw
        n = self.S + 2
        if self.foresight:
            with enable_x64(False):
                path = wobble_path(
                    state.demand_key, jnp.asarray(state.physics_time, jnp.int32),
                    jnp.asarray(state.load_wobble, jnp.float32), n * sub,
                    jnp.float32(p.demand_theta), jnp.float32(p.demand_sigma),
                    jnp.float32(p.delta_t),
                )  # fmt: skip
            path = np.concatenate([[float(state.load_wobble)], np.asarray(path)])
            return knot_means(path, sub)[:n]
        decay = float(np.exp(-float(p.demand_theta) * float(p.delta_t) * sub))
        return float(state.load_wobble) * decay ** np.arange(n)

    def _solve(self, state, t: int):
        left = max(self.knots - t // KNOT, 0)
        active = min(self.look, left) + self.tail
        wobble = self._wobble(state)
        with enable_x64():
            st = float64(state)
            wobble = jnp.asarray(wobble)
            x0 = self._x(st)
            self._at = (x0, jnp.zeros(M.NU))
            last = self._followed
            cold = last is None or t != last.t0 + self.replan * KNOT
            gap = {} if cold else self._gap(x0, st)
            cold = cold or gap["turbine_gap"] > JUMP
            room, sigma = jnp.asarray(self.room), jnp.asarray(self.sigma)
            if cold:
                self._discover(st, wobble, t, active)
            day = self._day(st, wobble, self._free(t, x0))
            d = {"day": day, "room": room, "sigma": sigma}
            # The free knots are already counted: this plan carries no clock.
            x0 = x0.at[M.CLOCK].set(0.0)
            self._at = (x0, self._at[1])
            plan = self._attempt(d, active, cold)
            self.certificates[-1].update(gap, applies=True)
            self._plan = plan
            moves = self.shooting.moves(plan)
            states, rows = self._roll(d["day"], x0, moves)
            states = np.asarray(states)
            load = np.asarray(d["day"].plan)[: len(states)] - states[:, M.SHORT]
            self._rows = np.asarray(rows)
            self._followed = Followed(
                t, np.asarray(moves), states, load, int(d["day"].committed)
            )

    def _gap(self, x0, st) -> dict:
        """How far the last plan was from the plant at the knot this solve starts
        from (K8, one hold ahead): Tavg in K, delta I of rated, the planned state."""
        k = self.replan
        x1, rows = self._followed.states[k], self._rows[k - 1]
        T_avg = 0.5 * (st.T_coolant + st.T_cold)
        off = st.delta_I - st.n * self.p.ao_ref
        return {
            "tavg_gap": float(abs(rows[1] - T_avg)),
            "offset_gap": float(abs(rows[2] - off)),
            "state_gap": float(
                np.max((np.abs(x1 - x0) / np.maximum(1.0, x0))[: M.CLOCK])
            ),
            "turbine_gap": float(abs(x1[M.SHORT] - x0[M.SHORT])),
        }

    def _discover(self, st, wobble, t: int, active: int):
        """Where the offset's hour goes (``DISCOVERY_ROOM``): the knots a plan
        that counts its own penalty time, with Tavg's room wide, spends past the
        band's backed-off edge."""
        d = {
            "day": self._day(st, wobble),
            "room": jnp.asarray(max(self.room, DISCOVERY_ROOM + TAVG_BACKOFF)),
            "sigma": jnp.asarray(self.sigma),
        }
        plan = self._segments(d, active, COLD_SEGMENTS)
        _, rows = self._roll(d["day"], self._at[0], self.shooting.moves(plan))
        past = np.abs(np.asarray(rows)[:, 2]) > float(self.p.ao_band) - OFFSET_BACKOFF
        self._outside = {t // KNOT + 1 + int(k) for k in np.flatnonzero(past)}

    def _free(self, t: int, x0):
        """The knots ahead the plan may spend outside the band: the discovered
        ones, the latest dropped until they fit what is left of the hour."""
        p, now = self.p, t // KNOT
        left = (float(p.ao_allowed) - 60.0 * CLOCK_BACKOFF) / 3600.0 - float(
            x0[M.CLOCK]
        )
        hours = KNOT * self.period * float(p.delta_t) / 3600.0
        keep = sorted(k for k in self._outside if k > now)[: max(int(left / hours), 0)]
        free = np.zeros(self.S + 2)
        free[[k - now for k in keep if k - now < len(free)]] = 1.0
        return jnp.asarray(free)

    def _attempt(self, d, active: int, cold: bool):
        """One solve, its last plan: warm from the plan being followed, a knot
        on, or cold from the first guess."""
        if cold:
            return self._segments(d, active, COLD_SEGMENTS)
        last = self._plan
        for _ in range(self.replan - 1):  # the solve shifts it once more
            z, lam, w = self.shooting.shift(last)
            last = last._replace(z=z, lam=lam, w=w)
        return self._segments(d, active, WARM_SEGMENTS, last, shift=True)

    def _segments(self, d, active: int, most: int, plan=None, shift=False):
        """Up to ``most`` segments of the solver on ``d``, each from the plan the
        last reached: from ``plan`` (shifted a knot first if ``shift``), or with
        none from the first guess. Ends at a converged plan, or on the plan before
        a segment that failed."""
        x0, hold = self._at
        for i in range(most):
            t0 = time.perf_counter()
            if plan is None:
                start = self._guess(d["day"], x0)
                new = self.shooting.solve(d, x0, hold, active=active, start=start)
            elif shift and i == 0:
                new = self.shooting.solve(d, x0, hold, plan, active=active)
            else:
                new = self.shooting.resolve(d, x0, hold, plan, active=active)
            jax.block_until_ready(new.z)
            c = new.certificate
            failed, capped = bool(c.failed), bool(c.capped)
            self.certificates.append({
                "seconds": time.perf_counter() - t0, "iterations": int(c.iterations),
                "error": float(c.error), "capped": capped, "failed": failed,
                "violation": float(c.violation), "cost": float(new.cost),
                "room": float(d["room"]), "applies": False,
            })  # fmt: skip
            if failed and plan is not None:
                if shift and i == 0:  # the plan being followed, a knot on
                    z, lam, w = self.shooting.shift(plan)
                    plan = plan._replace(z=z, lam=lam, w=w)
                break
            plan = new
            if failed or not capped:
                break
        return plan

    def step(self, _obs, state):
        """The raw action for the true ``state``; ``_obs`` is ignored."""
        t, last = int(state.time), self._followed
        if last is None or t < last.t0 or t >= last.t0 + self.replan * KNOT:
            self._solve(state, t)
        return action(self.raw, self._followed, state, self.period)

    def solver_report(self) -> dict:
        """The certificates summed for the record. A solve is its segments up to
        the one that applied: it failed if that one did, and was capped if that
        one stopped at the cap above the tolerance."""
        every = self.certificates
        applied = [x for x in every if x["applies"]]

        def worst(key, rows=applied):
            return max((v for x in rows if np.isfinite(v := x.get(key, -1.0))),
                       default=-1.0)  # fmt: skip

        return {
            "solver_calls": len(applied),
            "solver_segments": len(every),
            "solver_failures": sum(x["failed"] for x in applied),
            "solver_capped": sum(x["capped"] for x in applied),
            "solver_mean_iters": round(
                sum(x["iterations"] for x in every) / max(len(applied), 1), 1
            ),
            "solver_max_error": worst("error", [x for x in applied if not x["failed"]]),
            "solver_max_violation": worst("violation"),
            "solver_max_tavg_gap": worst("tavg_gap"),
            "solver_max_offset_gap": worst("offset_gap"),
        }


def make_reactor_mpc(env, params, seed=None, **kwargs) -> DayOracle:
    """The oracle for one episode: with ``seed``, the wobble it draws foreseen."""
    return DayOracle(env, params, seed=seed, **kwargs)
