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
  travel, both in dollars on every visited state (T2). The tower's fatigue is
  not in the plan: its half-cycle count reads turning points of the history,
  which no smooth planner has got (PHYSICS.md, what the oracle needs); the
  ceiling's replay charges it. The receding slot keeps the speed's resting
  target at rated (G12); the ceiling plans the score alone;
- limit: the rotor speed under the safety system's overspeed level on every
  visited state, elastic at the trip cost per unit (T3);
- dropped guards (T4): the speed and wind floors, and the pitch and torque
  clips after the step, since a ramp toward an in-range command stays in range.

The cold start holds the pitch and the torque where they are.
"""

import jax
import jax.numpy as jnp
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
    tower_damping,
    tower_stiffness,
    turbulence_time_constant,
    wind_power,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of, softplus_clip
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
    v_mean, v_turb, now, gust_t0, derate_t0, path = task[1:]
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


def wind_model(env, p, rest: bool = True) -> TaskModel:
    per_W = float(1.0e-6 * p.delta_t / 3600.0)  # MWh per W per step
    rated = omega_rated(p)
    per_deg = float(pitch_system_cost(p) / (p.pitch_travel_design * life_steps(p)))

    def step(d, x, u, t):
        x1, v_wind, rating = wind_step(env, p, x, u, t, d["task"], d["widths"])
        power = electrical_power(x1[0], x1[2], p)
        available = jnp.minimum(rating, wind_power(v_wind, p))
        y = jnp.stack([available - power, x1[1] - x[1], x1[0]])
        return x1, jnp.zeros((), x.dtype), y, power[None], ()

    return TaskModel(
        nx=6, nu=2, step=step,
        state_x=lambda s: jnp.stack(
            [s.omega, s.pitch, s.torque, s.x_tower, s.v_tower, s.w_induced]
        ),
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


def ceiling_data(env, p, state, step_keys):
    """The data of a plan from ``state`` over the steps of ``step_keys``: the
    wind and the rating's power after every step, from those keys.
    Neither depends on the moves, so they are read off a run of the env at
    any move (here the reset's pitch and torque held)."""
    held = env.initial_action(state, p)

    def advance(s, k):
        s1 = env.step_env(k, s, held, p)[1]
        return s1, jnp.stack([s1.v_wind, s1.target_power])

    path = jax.lax.scan(advance, state, step_keys)[1]
    return (
        state.target_power, state.v_mean, state.v_turb, state.time, state.gust_t0,
        state.derate_t0, path,
    )  # fmt: skip


def make_wind_turbine_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``wind_model``."""
    return Oracle(env, params, "wind_turbine", wind_model)
