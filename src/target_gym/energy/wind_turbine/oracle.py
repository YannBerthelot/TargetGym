"""The oracle (the MPC slot) for the wind turbine task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4, and Amendment 3), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``wind_step``) with its own right-hand side,
  ``velocity``: the raw moves' clips removed and the moves bounded (T1); both
  commands within their actuator's reach (T1r, ``reach``); the corners left
  inside the reach and Cp's floor smoothed by softplus at the K5 widths (T5,
  ``data/oracle_widths/wind_turbine.json``); ROSCO's table read through K2's
  clip, cell by cell, which is its bilinear lookup at width 0 (T5, one site per
  axis, ``table_weights``); the wind on its noise-free mean path, updated before
  the step as the env does; the dispatch's levels along the plan;
- objective: the imbalance, the pitch's travel above its hold level and the
  speed above its band through epigraph rows on every visited state, linear
  since e_tol = 0 and p = 1 (T2); the speed's resting target at rated (G12),
  since the reward prices only power at a steady state and leaves the rotor's
  stored energy free;
- limit: the rotor speed, underspeed < omega / rated < overspeed on every
  visited state, elastic at the trip cost per unit (T3), backed off by the
  planner's error on it (K11, in the widths file);
- dropped guards (T4): the tip-speed and speed guards, and the pitch and
  torque clips after the step, since a lag toward an in-range command stays
  in range.

The reward is in dollars, so the tie-break's budget is what half a resolution
costs in one step in dollars (G10). The cold start holds the pitch and the
torque where they are.
"""

import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.energy.wind_turbine.cp_table import (
    CP_TABLE,
    PITCH_STEP,
    TSR_FIRST,
    TSR_STEP,
)
from target_gym.energy.wind_turbine.env import (
    N_TARGET_BLOCKS,
    WIND_OU_THETA,
    drivetrain_inertia,
    electrical_power,
    omega_rated,
    rotor_area,
)
from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import Oracle, TaskModel, raw_of, softplus_clip
from target_gym.oracles.shooting import Hinge, Limit
from target_gym.reward import trip_cost

#: Cp's ceiling (``power_coefficient``), which no reachable state meets.
BETZ = 0.593


def commands(p, u):
    """The pitch and torque the raw move ``u`` commands, unclipped (T1)."""
    pitch = p.pitch_min + 0.5 * (u[0] + 1.0) * (p.pitch_max - p.pitch_min)
    return pitch, 0.5 * (u[1] + 1.0) * p.torque_max


def rate_reach(rate, tau, p, steps):
    """How far past its position a command keeps an actuator at full rate over
    the whole hold of ``steps`` task steps: it closes the gap at the rate limit
    until the gap is rate * tau, where the lag takes over (``compute_velocity``)."""
    return rate * (steps * p.delta_t + tau)


def pitch_reach(p, steps):
    return rate_reach(p.pitch_rate_max, p.pitch_tau, p, steps)


def torque_reach(p, steps):
    return rate_reach(p.torque_rate_max, p.torque_tau, p, steps)


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


def table_cp(lam, pitch, width_tsr, width_pitch):
    """``power_coefficient``'s lookup, before its clip, by ``table_weights``."""
    table = jnp.asarray(CP_TABLE, lam.dtype)
    tsr = table_weights((lam - TSR_FIRST) / TSR_STEP, table.shape[0], width_tsr)
    beta = table_weights(pitch / PITCH_STEP, table.shape[1], width_pitch)
    return tsr @ table @ beta


def velocity(position, action, v_wind, widths, deltas, p):
    """``compute_velocity`` with its kinks smoothed at ``widths`` (in name order:
    Cp's floor, the table's pitch axis and tip-speed axis, the pitch rate, the
    torque rate; 0.0 is the exact clip) and ``deltas`` added to the outputs of Cp,
    the pitch rate and the torque rate at every evaluation (K5's probe), the
    guards dropped."""
    omega, pitch, torque = position[0], position[1], position[2]
    pitch_cmd, torque_cmd = action
    lam = omega * p.R / v_wind
    fit = table_cp(lam, pitch, widths[2], widths[1])
    cp = softplus_clip(fit, 0.0, BETZ, widths[0]) + deltas[0]
    tau_aero = 0.5 * p.rho * rotor_area(p) * v_wind**3 * cp / omega
    dpitch = softplus_clip(
        (pitch_cmd - pitch) / p.pitch_tau, -p.pitch_rate_max, p.pitch_rate_max,
        widths[3],
    )  # fmt: skip
    dtorque = softplus_clip(
        (torque_cmd - torque) / p.torque_tau, -p.torque_rate_max, p.torque_rate_max,
        widths[4],
    )  # fmt: skip
    return jnp.stack([
        (tau_aero - p.N_gear * torque) / drivetrain_inertia(p),
        dpitch + deltas[1],
        dtorque + deltas[2],
    ]), None  # fmt: skip


def wind_step(env, p, x, u, t, task, widths, deltas=(0.0, 0.0, 0.0)):
    """One env step from ``x`` = (omega, pitch, torque) under the raw move
    ``u``: x_next."""
    _, _, v_mean, deviation = task
    action = commands(p, u)
    v_wind = v_mean + deviation * (1.0 - WIND_OU_THETA * p.delta_t) ** (t + 1)
    x1, _ = integrate_dynamics(
        positions=x, delta_t=p.delta_t, method=env.integration_method,
        compute_velocity=partial(
            velocity, action=action, v_wind=v_wind, widths=widths, deltas=deltas,
            p=p,
        ),
    )  # fmt: skip
    return x1


def wind_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 1.0
    per_W = float(p.imbalance_price / 1.0e6 * p.delta_t / 3600.0)
    floor = per_W * float(p.e_floor)
    rated = omega_rated(p)

    def step(d, x, u, t):
        task, w = d["task"], d["widths"]
        levels, clock = task[0], task[1]
        x1 = wind_step(env, p, x, u, t, task, w)
        power = electrical_power(x1[0], x1[2], p)
        block = jnp.minimum(
            (clock + t + 1) // p.target_block_steps, N_TARGET_BLOCKS - 1
        )
        y = jnp.stack([
            (levels[block] - power) / p.e_floor,
            (x1[1] - x[1]) / (p.pitch_max * p.c_hold),
            x1[0] / rated - p.speed_band,
            x1[0],
        ])  # fmt: skip
        return x1, jnp.zeros((), x.dtype), y, power[None], ()

    return TaskModel(
        nx=3, nu=2, step=step,
        state_x=lambda s: jnp.stack([s.omega, s.pitch, s.torque]),
        data=lambda s: (s.target_levels, s.block_clock, s.v_mean, s.v_wind - s.v_mean),
        hold=lambda s: jnp.stack([
            raw_of(s.pitch, p.pitch_min, p.pitch_max),
            raw_of(s.torque, 0.0, p.torque_max),
        ]),
        tracked=lambda s: electrical_power(s.omega, s.torque, p)[None],
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(weight=floor, absolute=True),
            Hinge(weight=floor * float(p.fatigue_weight), absolute=True, tol=1.0),
            Hinge(weight=floor * float(p.speed_weight)),
        ),
        limits=(
            Limit(
                p.underspeed_factor * rated, p.overspeed_factor * rated,
                float(trip_cost(p)),
            ),
        ),
        move_rows=lambda x, u, steps: reach(p, x, u, steps),
        move_scale=lambda steps: reach_scale(p, steps),
        tie_budget=per_W * 0.5 * float(p.precision_floor),
        rest=lambda x: (x[0] - rated) / (rated * (p.overspeed_factor - 1.0)),
    )  # fmt: skip


def make_wind_turbine_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``wind_model``."""
    return Oracle(env, params, "wind_turbine", wind_model)
