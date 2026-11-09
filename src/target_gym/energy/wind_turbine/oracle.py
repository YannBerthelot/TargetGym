"""The oracle (the MPC slot) for the wind turbine task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``wind_step``) with its own right-hand side,
  ``velocity``: the rotor, the tower's fore-aft mode, Oye's lagging inflow
  and the actuators. Each move is an actuator's rate, a fraction of its limit,
  sent as the command that ramps there in one step (``command``), so the rate
  clip never binds and the move rows keep the command in its range (T1r);
  ROSCO's Cp and Ct tables read through K2's clip, cell by cell, which is their
  bilinear lookup at width 0 (``table_weights``); the wind after every step:
  on its noise-free mean path in the registry's receding slot, and on the
  episode's own path for the ceiling (``ceiling_data``), which knows the
  turbulence ahead as a ceiling may;
- objective: the score's lost production against the available power (the
  rating, timed curtailment included, or the wind at peak Cp), a one-sided
  epigraph row, and the pitch system's wear, an absolute row on the pitch's
  travel, both in dollars on every visited state (T2). The tower's fatigue,
  a half-cycle count between turning points of the base moment, is planned
  by online rainflow counting (Loew and Bottasso, WES 7, 1605, 2022): the
  turning points are counted outside the solver on a plan's own moments
  (``turning_points``) and frozen in the data, where each one's half-cycle,
  0.5 (range / 1 MN m)^m, is an absolute epigraph row of power m from the
  last frozen turning point, carried as a seventh state; a driver re-solves
  on the new plan's turning points until they stop changing. With no frozen
  points (the registry's receding slot) the row is 0 and the fatigue is not
  planned. The receding slot keeps the speed's resting target at rated
  (G12); the ceiling plans the score alone;
- limit: the rotor speed under the safety system's overspeed level on every
  visited state, elastic at the trip cost per unit (T3);
- dropped guards (T4): the speed and wind floors, and the pitch and torque
  clips after the step, since a ramp toward an in-range command stays in range.

The cold start holds the pitch and the torque where they are.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial as partial

from target_gym.energy.wind_turbine.cp_table import (
    CP_TABLE_FEATHER,
    CT_TABLE_FEATHER,
    PITCH_STEP,
    TSR_FIRST,
    TSR_STEP,
)
from target_gym.energy.wind_turbine.env import (
    drivetrain_inertia,
    electrical_power,
    gust,
    life_steps,
    omega_rated,
    pitch_system_cost,
    rotor_area,
    tower_base_moment,
    tower_cost,
    tower_damping,
    tower_stiffness,
    turbulence_time_constant,
    wind_power,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import (
    Oracle,
    TaskModel,
    raw_of,
    scored_run,
    softplus_clip,
)
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost


def pitch_reach(p, steps):
    """How far the pitch moves at its rate limit over ``steps`` task steps."""
    return p.pitch_rate_max * steps * p.delta_t


def torque_reach(p, steps):
    return p.torque_rate_max * steps * p.delta_t


def travel(p):
    """How far each actuator moves in one task step at its rate limit."""
    return jnp.stack([pitch_reach(p, 1), torque_reach(p, 1)])


def command(p, held, v):
    """The pitch and torque that ramp from ``held`` by ``v`` of the rate limit
    in one step."""
    return held + v * travel(p)


def raw_command(p, held, v):
    """``command`` as the env's raw action."""
    pitch, torque = command(p, held, v)
    return jnp.stack(
        [raw_of(pitch, p.pitch_min, p.pitch_max), raw_of(torque, 0.0, p.torque_max)]
    )


def reach(p, held, v, steps):
    """T1r, rows >= 0: every command of a ramp held ``steps`` steps stays in
    its range."""
    last = held + v * travel(p) * steps
    lo = jnp.stack([p.pitch_min, 0.0])
    hi = jnp.stack([p.pitch_max, p.torque_max])
    span = hi - lo
    return jnp.concatenate([(hi - last) / span, (last - lo) / span])


def table_weights(z, nodes, width):
    """Each node's share of ``z``, in cells from the first: the table's bilinear
    weights, held at its edges, from one clip per cell, smoothed at ``width`` (K2).
    Softplus sums to 1 here, as the clip does."""
    ramp = softplus_clip(z - jnp.arange(nodes - 1.0), 0.0, 1.0, width)
    return jnp.concatenate([1.0 - ramp[:1], ramp[:-1] - ramp[1:], ramp[-1:]])


def table_read(table, lam, pitch, width_tsr, width_pitch):
    """A coefficient table's lookup by ``table_weights``."""
    table = jnp.asarray(table, lam.dtype)
    tsr = table_weights((lam - TSR_FIRST) / TSR_STEP, table.shape[0], width_tsr)
    beta = table_weights(pitch / PITCH_STEP, table.shape[1], width_pitch)
    return tsr @ table @ beta


def table_cp(lam, pitch, width_tsr, width_pitch):
    """``power_coefficient``'s lookup by ``table_weights``."""
    return table_read(CP_TABLE_FEATHER, lam, pitch, width_tsr, width_pitch)


def velocity(position, action, v_wind, widths, p):
    """``compute_velocity`` with its sixth state, the tables read at
    ``widths`` (in the widths file's order: unused, pitch axis, tip-speed
    axis), the guards dropped. ``action`` is the step's (pitch rate, torque
    rate)."""
    omega, pitch, torque, x, xdot, w = (position[i] for i in range(6))
    v_rel = v_wind - xdot

    def tables(v):
        lam = omega * p.R / v
        return (
            table_read(CP_TABLE_FEATHER, lam, pitch, widths[2], widths[1]),
            table_read(CT_TABLE_FEATHER, lam, pitch, widths[2], widths[1]),
        )

    # Oye's quasi-steady induction (``induction``), on the smoothed thrust.
    ct_qs = tables(v_rel)[1]
    a = jnp.where(
        ct_qs <= 0.96,
        0.5 * (1.0 - jnp.sqrt(1.0 - jnp.minimum(ct_qs, 0.96))),
        0.143 + jnp.sqrt(0.0203 - 0.6427 * (0.889 - jnp.maximum(ct_qs, 0.96))),
    )
    tau = p.oye_k1 / (1.0 - p.oye_k2 * jnp.minimum(a, 0.5)) * p.R / v_rel
    w_qs = a * v_rel
    v_eff = v_rel - p.dynamic_inflow * (w - w_qs)
    cp, ct = tables(v_eff)
    q = 0.5 * p.rho * rotor_area(p) * v_eff**2
    tau_aero = q * v_eff * cp / omega
    accel = (q * ct - tower_damping(p) * xdot - tower_stiffness(p) * x) / (
        p.tower_modal_mass
    )
    return jnp.stack([
        (tau_aero - p.N_gear * torque) / drivetrain_inertia(p),
        action[0], action[1], xdot, accel, (w_qs - w) / tau,
    ]), None  # fmt: skip


def wind_step(env, p, x, v, t, task, widths):
    """One env step from ``x`` = (omega, pitch, torque, x_tower, v_tower,
    w_induced) at the rates ``v`` (fractions of the limits): (x_next, the
    step's wind, the rating's power)."""
    v_mean, v_turb, now, gust_t0, derate_t0, path = task[1:7]
    after = (now + t + 1) * p.delta_t
    if path is None:
        decay = jnp.exp(-p.delta_t / turbulence_time_constant(v_mean, p))
        v_wind = v_mean + v_turb * decay ** (t + 1) + gust(after, v_mean, gust_t0, p)
        derated = (derate_t0 >= 0.0) & (after >= derate_t0)
        rating = jnp.where(derated, p.derate_to, p.rating) * p.P_rated
    else:  # the episode's own wind and rating, t + 1 steps after the start
        k = jnp.minimum(t, path.shape[0] - 1)
        v_wind, rating = path[k, 0], path[k, 1]
    rates = v * travel(p) / p.delta_t
    x1, _ = integrate_dynamics(
        positions=x, delta_t=p.delta_t, method=env.integration_method,
        compute_velocity=partial(
            velocity, action=rates, v_wind=v_wind, widths=widths, p=p
        ),
    )  # fmt: skip
    return x1, v_wind, rating


def turning_points(m0, turn0, dir0, moments):
    """The turning points of a plan's base moments ``moments`` (N m, after
    each step) as the env counts them (``env.count_half_cycles``) from the
    state's moment ``m0``, last turning point ``turn0`` and direction ``dir0``:
    (a mask over the steps, True where the moment after that step is a
    turning point; the last turning point before the first step). A turn is
    seen one step late, so the moment after the last step is never one."""
    m = np.concatenate([[float(m0)], np.asarray(moments, float)])
    mask, z0, prev = np.zeros(len(m) - 1, bool), float(turn0), float(dir0)
    for j in range(len(m) - 1):
        d = float(np.sign(m[j + 1] - m[j]))
        if prev != 0.0 and d != 0.0 and d != prev:
            if j == 0:
                z0 = float(m[0])
            else:
                mask[j - 1] = True
        if d != 0.0:
            prev = d
    return mask, z0


def wind_model(env, p, rest: bool = True) -> TaskModel:
    per_W = float(1.0e-6 * p.delta_t / 3600.0)  # MWh per W per step
    rated = omega_rated(p)
    per_deg = float(pitch_system_cost(p) / (p.pitch_travel_design * life_steps(p)))
    # $ per (MN m)^m of a half-cycle's range (``compute_reward_terms``).
    per_range = float(0.5 * tower_cost(p) / (p.tower_damage_design * life_steps(p)))

    def step(d, x, u, t):
        task = d["task"]
        x1, v_wind, rating = wind_step(env, p, x[:6], u, t, task, d["widths"])
        power = electrical_power(x1[0], x1[2], p)
        available = jnp.minimum(rating, wind_power(v_wind, p))
        moment = tower_base_moment(x1[3], p)
        if len(task) > 7:  # turning points frozen by the outer loop
            mask, z0 = task[7]
            mask = jnp.asarray(mask)
            z = jnp.where(t == 0, jnp.asarray(z0, x.dtype), x[6])
            hit = mask[jnp.minimum(t, mask.shape[0] - 1)]
            half = jnp.where(hit, (moment - z) / 1.0e6, 0.0)
            z1 = jnp.where(hit, moment, z)
        else:
            half, z1 = jnp.zeros((), x.dtype), x[6]
        y = jnp.stack([available - power, x1[1] - x[1], half, x1[0]])
        x1 = jnp.concatenate([x1, z1[None]])
        return x1, jnp.zeros((), x.dtype), y, power[None], ()

    return TaskModel(
        nx=7, nu=2, step=step,
        state_x=lambda s: jnp.stack([
            s.omega, s.pitch, s.torque, s.x_tower, s.v_tower, s.w_induced,
            s.moment_turn,
        ]),  # fmt: skip
        data=lambda s: (
            s.target_power, s.v_mean, s.v_turb, s.time, s.gust_t0, s.derate_t0, None
        ),
        hold=lambda s: jnp.zeros(2, s.omega.dtype),
        action=lambda s, v: raw_command(p, jnp.stack([s.pitch, s.torque]), v),
        tracked=lambda s: electrical_power(s.omega, s.torque, p)[None],
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(weight=per_W * float(p.energy_price)),
            Hinge(weight=per_deg, absolute=True),
            Hinge(
                weight=per_range, absolute=True, power=int(p.fatigue_exponent)
            ),
        ),
        limits=(Limit(hi=p.overspeed_factor * rated, price=float(trip_cost(p))),),
        move_rows=lambda x, v, steps: reach(p, x[1:3], v, steps),
        move_scale=lambda steps: (1.0, 1.0),
        tie_budget=per_W * float(p.energy_price) * 0.5 * float(p.precision_floor),
        rest=(
            lambda x: (x[0] - rated) / (rated * (p.overspeed_factor - 1.0))
        ) if rest else None,
    )  # fmt: skip


def ceiling_backoffs(env, p) -> tuple:
    """How far inside each limit the ceiling plans: one sigma of the rotor
    speed's reading (5.2e-4 rad/s). With the gusts known ahead the plan stores
    energy in the rotor at the overspeed level, and the float32 plant the score
    runs ends a few ulps off a float64 plan, which trips it."""
    (sigma,) = [r.sigma for r in env.readings if r.field == "omega"]
    return (float(sigma),)


def ceiling_data(env, p, state, step_keys, run_as=scored_run):
    """The data of a plan from ``state`` over the steps of ``step_keys``: the
    wind and the rating's power after every step, from those keys.
    Neither depends on the moves, so they are read off a run of the env at
    any move (here the reset's pitch and torque held) that cannot trip: a trip
    restarts the plant on a fresh draw of the turbulence, which is not the
    episode's wind (the overspeed level is lifted for this run alone). The run
    is the scored one's, in float32 (``scored_run``; a test of the
    model on x64 data passes its own ``run_as``): under x64 the turbulence
    draws other values."""

    def run(state, p, step_keys):
        held = env.initial_action(state, p)
        free = p.replace(overspeed_factor=1e9)

        def advance(s, k):
            s1 = env.step_env(k, s, held, free)[1]
            return s1, jnp.stack([s1.v_wind, s1.target_power])

        return jax.lax.scan(advance, state, step_keys)[1]

    path = jnp.asarray(run_as(run, state, p, step_keys), jnp.result_type(state.omega))
    return (
        state.target_power, state.v_mean, state.v_turb, state.time, state.gust_t0,
        state.derate_t0, path,
    )  # fmt: skip


def make_wind_turbine_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``wind_model``."""
    return Oracle(env, params, "wind_turbine", wind_model)
