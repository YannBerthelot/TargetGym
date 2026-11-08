"""The oracle (the MPC slot) for the wind turbine task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``wind_step``) with its own right-hand side,
  ``velocity``: the rotor, the tower's fore-aft mode and the actuators' ramps at
  their rate limits, the rate clips smoothed by softplus at the K5 widths (T5);
  ROSCO's Cp and Ct tables read through K2's clip, cell by cell, which is their
  bilinear lookup at width 0 (``table_weights``); the wind on its noise-free
  mean path (the turbulence's deviation decaying at the Kaimal time scale, the
  gust if one is set), which the oracle knows ahead;
- objective: the score's lost production against the available power (the
  rating, timed curtailment included, or the wind at peak Cp), a one-sided
  epigraph row, and the pitch system's wear, an absolute row on the pitch's
  travel, both in dollars on every visited state (T2); the speed's resting
  target at rated (G12), since the score leaves the rotor's stored energy
  free. The tower's fatigue is not in the plan: its half-cycle count reads
  turning points of the history, which the planner's smooth model has not
  got (PHYSICS.md, what the oracle needs);
- limit: the rotor speed under the safety system's overspeed level on every
  visited state, elastic at the trip cost per unit (T3);
- dropped guards (T4): the speed and wind floors, and the pitch and torque
  clips after the step, since a ramp toward an in-range command stays in range.

Not yet run on this plant (rule 9's room is "not measured"). The cold start
holds the pitch and the torque where they are.
"""

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


def commands(p, u):
    """The pitch and torque the raw move ``u`` commands, unclipped (T1)."""
    pitch = p.pitch_min + 0.5 * (u[0] + 1.0) * (p.pitch_max - p.pitch_min)
    return pitch, 0.5 * (u[1] + 1.0) * p.torque_max


def pitch_reach(p, steps):
    """How far an actuator moves at its rate limit over ``steps`` task steps."""
    return p.pitch_rate_max * steps * p.delta_t


def torque_reach(p, steps):
    return p.torque_rate_max * steps * p.delta_t


def reach(p, x, u, steps):
    """T1r, rows >= 0: each command within its reach from its actuator."""
    pitch, torque = commands(p, u)
    span = p.pitch_max - p.pitch_min
    gap = jnp.stack([(pitch - x[1]) / span, (torque - x[2]) / p.torque_max])
    room = jnp.stack(
        [pitch_reach(p, steps) / span, torque_reach(p, steps) / p.torque_max]
    )
    return jnp.concatenate([room - gap, room + gap])


def reach_scale(p, steps):
    """Half the span of each raw move's reach over ``steps`` held steps."""
    return (
        float(2.0 * pitch_reach(p, steps) / (p.pitch_max - p.pitch_min)),
        float(2.0 * torque_reach(p, steps) / p.torque_max),
    )


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
    """``compute_velocity`` with the tables read at ``widths`` (in the widths
    file's order: unused, pitch axis, tip-speed axis), the guards dropped.
    ``action`` is the step's (pitch rate, torque rate)."""
    omega, pitch, torque, x, xdot = position
    v_rel = v_wind - xdot
    lam = omega * p.R / v_rel
    cp = table_read(CP_TABLE_FEATHER, lam, pitch, widths[2], widths[1])
    ct = table_read(CT_TABLE_FEATHER, lam, pitch, widths[2], widths[1])
    q = 0.5 * p.rho * rotor_area(p) * v_rel**2
    tau_aero = q * v_rel * cp / omega
    accel = (q * ct - tower_damping(p) * xdot - tower_stiffness(p) * x) / (
        p.tower_modal_mass
    )
    return jnp.stack([
        (tau_aero - p.N_gear * torque) / drivetrain_inertia(p),
        action[0], action[1], xdot, accel,
    ]), None  # fmt: skip


def wind_step(env, p, x, u, t, task, widths):
    """One env step from ``x`` = (omega, pitch, torque, x_tower, v_tower) under
    the raw move ``u``: (x_next, the step's wind)."""
    _, v_mean, v_turb, now, gust_t0, _ = task
    pitch_cmd, torque_cmd = commands(p, u)
    rates = (
        softplus_clip(
            (pitch_cmd - x[1]) / p.delta_t, -p.pitch_rate_max, p.pitch_rate_max,
            widths[3],
        ),
        softplus_clip(
            (torque_cmd - x[2]) / p.delta_t, -p.torque_rate_max, p.torque_rate_max,
            widths[4],
        ),
    )  # fmt: skip
    decay = jnp.exp(-p.delta_t / turbulence_time_constant(v_mean, p))
    v_wind = (
        v_mean + v_turb * decay ** (t + 1)
        + gust((now + t + 1) * p.delta_t, v_mean, gust_t0, p)
    )  # fmt: skip
    x1, _ = integrate_dynamics(
        positions=x, delta_t=p.delta_t, method=env.integration_method,
        compute_velocity=partial(
            velocity, action=rates, v_wind=v_wind, widths=widths, p=p
        ),
    )  # fmt: skip
    return x1, v_wind


def wind_model(env, p) -> TaskModel:
    per_W = float(1.0e-6 * p.delta_t / 3600.0)  # MWh per W per step
    rated = omega_rated(p)
    per_deg = float(pitch_system_cost(p) / (p.pitch_travel_design * life_steps(p)))

    def step(d, x, u, t):
        task = d["task"]
        x1, v_wind = wind_step(env, p, x, u, t, task, d["widths"])
        power = electrical_power(x1[0], x1[2], p)
        now, derate_t0 = task[3], task[5]
        derated = (derate_t0 >= 0.0) & ((now + t + 1) * p.delta_t >= derate_t0)
        rating = jnp.where(derated, p.derate_to, p.rating) * p.P_rated
        available = jnp.minimum(rating, wind_power(v_wind, p))
        y = jnp.stack([available - power, x1[1] - x[1], x1[0]])
        return x1, jnp.zeros((), x.dtype), y, power[None], ()

    return TaskModel(
        nx=5, nu=2, step=step,
        state_x=lambda s: jnp.stack(
            [s.omega, s.pitch, s.torque, s.x_tower, s.v_tower]
        ),
        data=lambda s: (
            s.target_power, s.v_mean, s.v_turb, s.time, s.gust_t0, s.derate_t0
        ),
        hold=lambda s: jnp.stack([
            raw_of(s.pitch, p.pitch_min, p.pitch_max),
            raw_of(s.torque, 0.0, p.torque_max),
        ]),
        tracked=lambda s: electrical_power(s.omega, s.torque, p)[None],
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(weight=per_W * float(p.energy_price)),
            Hinge(weight=per_deg, absolute=True),
        ),
        limits=(Limit(hi=p.overspeed_factor * rated, price=float(trip_cost(p))),),
        move_rows=lambda x, u, steps: reach(p, x, u, steps),
        move_scale=lambda steps: reach_scale(p, steps),
        tie_budget=per_W * float(p.energy_price) * 0.5 * float(p.precision_floor),
        rest=lambda x: (x[0] - rated) / (rated * (p.overspeed_factor - 1.0)),
    )  # fmt: skip


def make_wind_turbine_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``wind_model``."""
    return Oracle(env, params, "wind_turbine", wind_model)
