"""The oracle (the MPC slot) for the reactor task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own stiff step, sub-step by sub-step, with the rods' raw move
  clip removed and the move bounded instead (T1); the rod demand within the rods'
  travel over the hold (T1r, ``reach``); the boron line's queue in the state; the
  corner left inside the reach smoothed by softplus at the K5 width (T5,
  ``data/oracle_widths/reactor.json``); the load the env applies: the day the
  state carries (its rate, depth and plan clock, which a stream rebases), at most
  the reload's ``pickup`` after a restart, and the wobble on its noise-free mean
  path; the core at its cycle point (``env.core_at``), with the burnup since the
  state's last refuelling (``burn_t0``);
- the turbine is not one of its moves: it is asked to load as fast as it may
  (the third action at +1), so it follows the plan, climbs at the conditioning
  ramps after a restart, and at its own fastest rate from any other load under
  the plan, as the env does;
- objective: |Tref - Tavg| past the rods' dead band, and delta I past the offset's
  band on either side, 5 % of rated at every power (a surrogate for the env's
  offset clocks, which K8's replay leaves out), through epigraph rows on every sub-step, and the rods' travel and the
  effluent through rows a step (T2);
- limits: power, fuel temperature and the hot leg on every sub-step's state; Tavg's
  floor has time to act, so only the replay charges it;
  elastic at the trip cost per unit (T3), backed off by the planner's error on
  them (K11, in the widths file);
- dropped guard (T4): the rods' range clip after the rate limit, since the demand
  is already in range.

Its horizon is shorter than the boron line, so this solver cannot see the task's
main trade. The ceiling comes from ``scripts/reactor_foresight.py``, which plans the
whole day; this slot stays so the shared runners have an oracle.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from target_gym.oracles.controller import Oracle, TaskModel, softplus_clip
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reactor.env import (
    NOT_RELOADING,
    ON_PLAN,
    _reactor_stiff_step,
    core_at,
    flux_tilt,
    moderator_tilt,
    planned_load,
    rod_tilt,
    rod_worth,
    tavg_dnb_limit,
    tavg_program,
    turbine_capacity,
)
from target_gym.reward import trip_cost

#: Where the plan's rods, boron, the turbine's load under its plan, whether the
#: conditioning ramps bind it, and the boron queue sit in ``x``, after the plant's
#: 15 states.
RODS, BORON, PICKUP, CONDITIONING, QUEUE = 15, 16, 17, 18, 19


def rod_travel(p):
    """The rods' raw travel in one physics sub-step at their one speed."""
    return 2.0 * p.rod_speed * p.delta_t


def reach(env, p, x, u, steps):
    """T1r, rows >= 0: the demand within what the rods cover over a hold of
    ``steps`` control steps: past it they run at full speed every sub-step."""
    room = steps * env.control_period * rod_travel(p)
    rows = jnp.stack([room - (u[0] - x[RODS]), room + (u[0] - x[RODS])])
    return rows


def reach_scale(env, p, steps):
    """Each move's typical size: half the span of the raw rod demand's reach over
    ``steps`` held steps, and the boron flow's range."""
    return (float(2.0 * steps * env.control_period * rod_travel(p)), 1.0)


def boron_step(arriving, boron, p, width):
    """The boron over one sub-step under the arriving flow command: below 0 it
    dilutes with water, above 0 it borates. The two branches' slopes differ, a corner
    right where a resting plan sits, so ``width`` smooths the split as K2 does; 0.0 is
    the env's exact step."""
    down = softplus_clip(-arriving, 0.0, 2.0, width) * p.dilution_flow
    up = softplus_clip(arriving, 0.0, 2.0, width) * p.boration_flow
    flow = down + up
    feed = up * p.boric_acid_ppm / jnp.where(flow > 0.0, flow, 1.0)
    return boron + (feed - boron) * (1.0 - jnp.exp(-flow / p.primary_mass * p.delta_t))


def reactor_step(env, p, x, u, t, task, width, deltas=None, boron_width=0.0):
    """One control step from ``x`` = (the plant, the rods in raw units, the boron,
    the turbine's load under its plan, whether the conditioning ramps bind it, the
    boron queue) under the raw demand ``u``: ``(x_next, rows)``, one row (Tref -
    Tavg, n, T_fuel, T_hot, Tavg, delta I, the day's load the turbine does not
    take) per sub-step.
    ``width`` is the rate corner's (0.0 is the exact clip) and ``deltas`` are added
    to its output, one per sub-step (K5's probe). The load is the env's
    (``env.compute_next_state``) with the third action at +1: the day ``task``
    carries on its plan clock, capped by the turbine's own load while it is under
    the day, which climbs at the conditioning ramps while they bind and at the
    turbine's fastest rate otherwise, until it meets the day."""
    start, ramp_down_at, ramp_up_at, wobble, delay, burn_rate = task[:6]
    plan_t0, plan_rate, plan_depth, burn_t0, cycle0 = task[6:]
    renewed_at = p.boron_ref * (1.0 - cycle0)  # critical boron at the last renewal
    day = SimpleNamespace(
        ramp_down_at=ramp_down_at, ramp_up_at=ramp_up_at, plan_rate=plan_rate,
        plan_depth=plan_depth,
    )  # fmt: skip
    most = rod_travel(p)
    period = env.control_period
    substeps = int(env.integration_method.split("_")[-1])
    decay = 1.0 - p.demand_theta * p.delta_t  # the wobble's mean per sub-step
    deltas = jnp.zeros(period, x.dtype) if deltas is None else deltas
    queue = jnp.concatenate([u[1:], x[QUEUE:-1]])
    arriving = queue[delay.astype(jnp.int32) - 1]

    def day_at(i):
        return planned_load((start + t * period + i - plan_t0) * p.delta_t, day)

    def load_at(i, pickup):
        return jnp.minimum(day_at(i), pickup) + wobble * decay ** (t * period + i)

    def sub_step(carry, i):
        plant, rods, boron, pickup, conditioning = carry
        rods = rods + softplus_clip(u[0] - rods, -most, most, width) + deltas[i]
        rod_in = 0.5 * (1.0 - rods)
        boron = boron_step(arriving, boron, p, boron_width)
        burnt = burn_rate * (start + t * period + i - burn_t0) * p.delta_t
        core = core_at(cycle0 + burnt / p.boron_ref, p)  # the core's coefficients now
        rho = rod_worth(rod_in, p) - rod_worth(p.rod_park, p)
        rho = rho + core.boron_worth * (boron - renewed_at)
        rho = rho + core.boron_worth * burnt
        eps = rod_tilt(rod_in, p)
        load = load_at(i, pickup)
        capacity = turbine_capacity(plant[10], p)  # on the steam before the step
        plant, _ = _reactor_stiff_step(plant, rho, core, substeps, load, eps)
        # The env's turbine with the third action at +1.
        held = jnp.where(pickup > 0.5 * NOT_RELOADING, day_at(i), pickup)
        climb = jnp.where(held < p.reload_knee, p.reload_fast_rate, p.reload_rate)
        fastest = jnp.where(conditioning > 0.5, climb, p.turbine_rate)
        pickup = jnp.maximum(
            held + fastest * p.delta_t, jnp.minimum(held, p.restart_load)
        )
        pickup = jnp.minimum(pickup, capacity)
        meets = pickup >= day_at(i + 1) - ON_PLAN
        pickup = jnp.where(meets, NOT_RELOADING, pickup)
        conditioning = jnp.where(meets, 0.0, conditioning)
        T_hot, T_cold = plant[8], plant[9]
        T_avg = 0.5 * (T_hot + T_cold)
        error = tavg_program(load_at(i + 1, pickup), p) - T_avg
        tilt = eps + moderator_tilt(T_hot, T_cold, p)
        delta_I = plant[0] * flux_tilt(plant[14], plant[12], plant[0], tilt, p)
        short = jnp.maximum(day_at(i + 1) - pickup, 0.0)
        rows = jnp.stack([error, plant[0], plant[7], T_hot, T_avg, delta_I, short])
        return (plant, rods, boron, pickup, conditioning), rows

    start_of_step = (x[:RODS], x[RODS], x[BORON], x[PICKUP], x[CONDITIONING])
    (plant, rods, boron, pickup, conditioning), rows = jax.lax.scan(
        sub_step, start_of_step, jnp.arange(period)
    )
    tail = jnp.stack([rods, boron, pickup, conditioning])
    return jnp.concatenate([plant, tail, queue]), rows


#: $/h per 1 % of rated delta I past the band, the plan's surrogate (ours).
PLAN_BAND_WEIGHT = 1000.0


def reactor_model(env, p) -> TaskModel:
    assert float(p.tracking_exponent) == 1.0
    assert env.integration_method.startswith("tr_bdf2_")
    per_K_s = p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW / 3.6
    per_K = float(per_K_s * p.delta_t)  # a sub-step's dollars
    period = env.control_period
    # The plan's surrogate for the offset's penalty clock, which has no gradient:
    # past the band, each 1 % of rated delta I weighs as 1 % of 1 GWe at $100/MWh.
    # Plans are scored on replay, against the clock itself, so K8's replay leaves
    # the surrogate out of the planned reward.
    per_delta_I = float(PLAN_BAND_WEIGHT * 100.0 * p.delta_t / 3600.0)
    flow_step = float(p.effluent_price * p.delta_t * period)
    # The reload's lost output, a sub-step's dollars per unit of load not taken: no
    # move changes it, but the plan's reward is the env's.
    per_GW_s = p.P_electric_GW * 1000.0 * p.derate_price_per_MWh / 3600.0
    per_short = float(per_GW_s * p.delta_t)

    def step(d, x, u, t):
        boron_width = d.get("boron_width", 0.0)
        x1, rows = reactor_step(
            env, p, x, u, t, d["task"], d["widths"][0], boron_width=boron_width
        )
        # The offset's band: rated delta I either side of a target that scales
        # with the power.
        off = rows[:, 5] - rows[:, 1] * p.ao_ref
        y = jnp.concatenate([
            rows[:, 0] / p.e_floor, off - p.ao_band,
            -off - p.ao_band, (x1[RODS:RODS + 1] - x[RODS:RODS + 1]) / 2.0,
            -u[1:], u[1:], rows[:, 1:5].ravel(),
        ])  # fmt: skip
        return x1, per_short * jnp.sum(rows[:, 6]), y, rows[-1:, 4], ()

    price = float(trip_cost(p))
    band = (
        Limit(p.n_min, p.n_max, price),
        Limit(p.T_fuel_min, p.T_fuel_max, price),
        Limit(hi=p.T_coolant_max, price=price),
        # Tavg's DNB limit is timed (2 h), planned as a hard bound: a plan that
        # used the allowance could be cheaper, so the ceiling stays an upper bound.
        Limit(hi=float(tavg_dnb_limit(p)), price=price),
    )  # Tavg's floor is a limit with time to act, charged on replay, not planned
    rod_raw = lambda s: 1.0 - 2.0 * s.rod_in  # noqa: E731
    return TaskModel(
        nx=49, nu=2, step=step,
        state_x=lambda s: jnp.concatenate([
            s.n[None], s.C,
            jnp.stack([
                s.T_fuel, s.T_coolant, s.T_cold, s.T_sg, s.I_hat, s.Xe_hat, s.I_tilt,
                s.Xe_tilt, rod_raw(s), s.boron, s.pickup, s.conditioning,
            ]),
            s.boron_queue,
        ]),
        data=lambda s: (
            s.physics_time, s.ramp_down_at, s.ramp_up_at, s.load_wobble, s.boron_delay,
            s.burn_rate, s.plan_t0, s.plan_rate, s.plan_depth, s.burn_t0, s.cycle0,
        ),
        hold=lambda s: jnp.stack([rod_raw(s), jnp.zeros_like(s.rod_in)]),
        tracked=lambda s: 0.5 * (s.T_coolant + s.T_cold)[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=per_K * float(p.e_floor), absolute=True,
                      tol=float(p.e_tol / p.e_floor)),) * period
        + (Hinge(weight=per_delta_I, surrogate=True),) * (2 * period)
        + (
            Hinge(weight=float(p.rod_step_price * p.rod_stroke_steps), absolute=True),
            Hinge(weight=flow_step * float(p.dilution_flow)),
            Hinge(weight=flow_step * float(p.boration_flow)),
        ),
        limits=band * period,
        # The turbine loads as fast as it may: the third action, not a move.
        action=lambda s, u: jnp.concatenate([u, jnp.ones(1, u.dtype)]),
        move_rows=lambda x, u, steps: reach(env, p, x, u, steps),
        move_scale=lambda steps: reach_scale(env, p, steps),
        tie_budget=per_K * period * 0.5 * float(p.precision_floor),
    )  # fmt: skip


def make_reactor_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``reactor_model``."""
    return Oracle(env, params, "reactor", reactor_model)
