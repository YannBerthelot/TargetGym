"""The oracle (the MPC slot) for the reactor task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's own stiff step, sub-step by sub-step, with the rods' raw move
  clip removed and the move bounded instead (T1); the rod demand within the rods'
  travel over the hold (T1r, ``reach``); the boron line's queue in the state; the
  corner left inside the reach smoothed by softplus at the K5 width (T5,
  ``data/oracle_widths/reactor.json``); the load on the day's schedule, with its
  wobble on its noise-free mean path;
- objective: |Tref - Tavg| and delta I past its band through epigraph rows on every
  sub-step, and the rods' travel and the effluent through rows a step (T2);
- limits: power, fuel temperature, the hot leg and Tavg on every sub-step's state,
  elastic at the trip cost per unit (T3), backed off by the planner's error on
  them (K11, in the widths file);
- dropped guard (T4): the rods' range clip after the rate limit, since the demand
  is already in range.

Its horizon is shorter than the boron line, so this solver cannot see the task's
main trade. The ceiling comes from ``scripts/reactor_foresight.py``, which plans the
whole day; this slot stays so the shared runners have an oracle.
"""

import jax
import jax.numpy as jnp

from target_gym.oracles.controller import Oracle, TaskModel, softplus_clip
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reactor.env import (
    _reactor_stiff_step,
    flux_tilt,
    moderator_tilt,
    rod_tilt,
    rod_worth,
    scheduled_load,
    tavg_program,
)
from target_gym.reward import trip_cost

#: Where the plan's rods, boron and queue sit in ``x``, after the plant's 15 states.
RODS, BORON, QUEUE = 15, 16, 17


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
    """Half the span of the raw demand's reach over ``steps`` held steps."""
    return (float(2.0 * steps * env.control_period * rod_travel(p)),)


def reactor_step(env, p, x, u, t, task, width, deltas=None):
    """One control step from ``x`` = (the plant, the rods in raw units, the boron,
    its queue) under the raw demand ``u``: ``(x_next, rows)``, one row (Tref - Tavg,
    n, T_fuel, T_hot, Tavg, delta I) per sub-step. ``width`` is the rate corner's
    (0.0 is the exact clip) and ``deltas`` are added to its output, one per sub-step
    (K5's probe)."""
    start, ramp_down_at, ramp_up_at, wobble, delay = task
    most = rod_travel(p)
    period = env.control_period
    substeps = int(env.integration_method.split("_")[-1])
    decay = 1.0 - p.demand_theta * p.delta_t  # the wobble's mean per sub-step
    deltas = jnp.zeros(period, x.dtype) if deltas is None else deltas
    queue = jnp.concatenate([u[1:], x[QUEUE:-1]])
    arriving = queue[delay.astype(jnp.int32) - 1]
    flow = jnp.where(
        arriving < 0, -arriving * p.dilution_flow, arriving * p.boration_flow
    )
    feed = jnp.where(arriving < 0, 0.0, p.boric_acid_ppm)
    closes = 1.0 - jnp.exp(-flow / p.primary_mass * p.delta_t)

    def load_at(i):
        k = start + t * period + i
        day = scheduled_load(k * p.delta_t, ramp_down_at, ramp_up_at, p)
        return day + wobble * decay ** (t * period + i)

    def sub_step(carry, i):
        plant, rods, boron = carry
        rods = rods + softplus_clip(u[0] - rods, -most, most, width) + deltas[i]
        rod_in = 0.5 * (1.0 - rods)
        boron = boron + (feed - boron) * closes
        rho = rod_worth(rod_in, p) - rod_worth(p.rod_park, p)
        rho = rho + p.boron_worth * (boron - p.boron_ref)
        eps = rod_tilt(rod_in, p)
        plant, _ = _reactor_stiff_step(plant, rho, p, substeps, load_at(i), eps)
        T_hot, T_cold = plant[8], plant[9]
        T_avg = 0.5 * (T_hot + T_cold)
        error = tavg_program(load_at(i + 1), p) - T_avg
        tilt = eps + moderator_tilt(T_hot, T_cold, p)
        delta_I = plant[0] * flux_tilt(plant[14], plant[12], plant[0], tilt, p)
        rows = jnp.stack([error, plant[0], plant[7], T_hot, T_avg, delta_I])
        return (plant, rods, boron), rows

    (plant, rods, boron), rows = jax.lax.scan(
        sub_step, (x[:RODS], x[RODS], x[BORON]), jnp.arange(period)
    )
    return jnp.concatenate([plant, jnp.stack([rods, boron]), queue]), rows


def reactor_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 1.0
    assert env.integration_method.startswith("tr_bdf2_")
    per_K_s = p.imbalance_multiple * p.spot_price_per_MWh * p.P_electric_GW / 3.6
    per_K = float(per_K_s * p.delta_t)  # a sub-step's dollars
    period = env.control_period
    per_delta_I = float(p.delta_I_price * 100.0 * p.delta_t / 3600.0)
    flow_step = float(p.effluent_price * p.delta_t * period)

    def step(d, x, u, t):
        x1, rows = reactor_step(env, p, x, u, t, d["task"], d["widths"][0])
        y = jnp.concatenate([
            rows[:, 0] / p.e_floor, rows[:, 5], (x1[RODS:RODS + 1] - x[RODS:RODS + 1]) / 2.0,
            -u[1:], u[1:], rows[:, 1:5].ravel(),
        ])  # fmt: skip
        return x1, jnp.zeros((), x.dtype), y, rows[-1:, 4], ()

    price = float(trip_cost(p))
    band = (
        Limit(p.n_min, p.n_max, price),
        Limit(p.T_fuel_min, p.T_fuel_max, price),
        Limit(hi=p.T_coolant_max, price=price),
        Limit(lo=p.T_avg_min, price=price),
    )
    rod_raw = lambda s: 1.0 - 2.0 * s.rod_in  # noqa: E731
    return TaskModel(
        nx=47, nu=2, step=step,
        state_x=lambda s: jnp.concatenate([
            s.n[None], s.C,
            jnp.stack([
                s.T_fuel, s.T_coolant, s.T_cold, s.T_sg, s.I_hat, s.Xe_hat, s.I_tilt,
                s.Xe_tilt, rod_raw(s), s.boron,
            ]),
            s.boron_queue,
        ]),
        data=lambda s: (
            s.physics_time, s.ramp_down_at, s.ramp_up_at, s.load_wobble, s.boron_delay,
        ),
        hold=lambda s: jnp.stack([rod_raw(s), jnp.zeros_like(s.rod_in)]),
        tracked=lambda s: 0.5 * (s.T_coolant + s.T_cold)[None],
        resolution=(float(p.precision_floor),),
        hinges=(Hinge(weight=per_K * float(p.e_floor), absolute=True),) * period
        + (Hinge(weight=per_delta_I, absolute=True, tol=float(p.delta_I_band)),) * period
        + (
            Hinge(weight=float(p.rod_step_price * p.rod_stroke_steps), absolute=True),
            Hinge(weight=flow_step * float(p.dilution_flow)),
            Hinge(weight=flow_step * float(p.boration_flow)),
        ),
        limits=band * period,
        move_rows=lambda x, u, steps: reach(env, p, x, u, steps),
        move_scale=lambda steps: reach_scale(env, p, steps),
        tie_budget=per_K * period * 0.5 * float(p.precision_floor),
    )  # fmt: skip


def make_reactor_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``reactor_model``."""
    return Oracle(env, params, "reactor", reactor_model)
