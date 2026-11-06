"""The oracle (the MPC slot) for the wind turbine task, on the shared solver.

Kept apart from the expert (``experts.py``) so that each has a fingerprint of
its own (``provenance.oracle_fingerprint``). The planner problem, by the
smoothing protocol's inventory (docs/proposals/oracle-smoothing-protocol.md,
section 5.4), run by ``target_gym.oracles.controller.Oracle``:

- model: the env's step restated (``wind_step``) with its own right-hand side,
  ``velocity``: the raw moves' clips removed and the moves bounded (T1); the
  pitch command within the actuator's reach (T1r, ``reach``); the corner left
  inside the reach and the fit's floor at Cp = 0 smoothed by softplus at the
  K5 widths (T5, ``data/oracle_widths/wind_turbine.json``); the wind on its
  noise-free mean path, updated before the step as the env does;
- objective: the imbalance and the pitch activity above its hold level
  through epigraph rows on every visited state, linear since e_tol = 0 and
  p = 1 (T2); the speed's resting target at rated (G12), since the reward
  prices only power and leaves the rotor's stored energy free;
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

from target_gym.energy.wind_turbine.env import (
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

#: The Cp fit's ceiling (env.py:271), which no reachable state meets.
BETZ = 0.593


def commands(p, u):
    """The pitch and torque the raw move ``u`` commands, unclipped (T1)."""
    pitch = p.pitch_min + 0.5 * (u[0] + 1.0) * (p.pitch_max - p.pitch_min)
    return pitch, 0.5 * (u[1] + 1.0) * p.torque_max


def pitch_reach(p, steps):
    """How far past the pitch a command stays at full rate over the whole hold
    of ``steps`` task steps: the actuator closes the gap at the rate limit
    until the gap is rate * tau, where the lag takes over (env.py:309)."""
    return p.pitch_rate_max * (steps * p.delta_t + p.pitch_tau)


def reach(p, x, u, steps):
    """T1r, rows >= 0: the pitch command within its reach from the pitch."""
    gap = (commands(p, u)[0] - x[1]) / (p.pitch_max - p.pitch_min)
    room = pitch_reach(p, steps) / (p.pitch_max - p.pitch_min)
    return jnp.stack([room - gap, room + gap])


def reach_scale(p, steps):
    """Half the span of each raw move's reach over ``steps`` held steps."""
    return (float(2.0 * pitch_reach(p, steps) / (p.pitch_max - p.pitch_min)), 1.0)


def velocity(position, action, v_wind, widths, deltas, p):
    """``compute_velocity`` (env.py) with its two kinks smoothed at ``widths``
    (Cp's floor, the pitch rate; 0.0 is the exact clip) and ``deltas`` added
    to their outputs at every evaluation (K5's probe), the guards dropped."""
    omega, pitch, torque = position[0], position[1], position[2]
    pitch_cmd, torque_cmd = action
    lam = omega * p.R / v_wind
    inv_lam_i = 1.0 / (lam + 0.08 * pitch) - 0.035 / (pitch**3 + 1.0)
    fit = (
        p.cp_c1
        * (p.cp_c2 * inv_lam_i - p.cp_c3 * pitch - p.cp_c4)
        * jnp.exp(-p.cp_c5 * inv_lam_i)
        + p.cp_c6 * lam
    )
    cp = softplus_clip(fit, 0.0, BETZ, widths[0]) + deltas[0]
    tau_aero = 0.5 * p.rho * rotor_area(p) * v_wind**3 * cp / omega
    dpitch = softplus_clip(
        (pitch_cmd - pitch) / p.pitch_tau, -p.pitch_rate_max, p.pitch_rate_max,
        widths[1],
    )  # fmt: skip
    return jnp.stack([
        (tau_aero - p.N_gear * torque) / drivetrain_inertia(p),
        dpitch + deltas[1],
        (torque_cmd - torque) / p.torque_tau,
    ]), None  # fmt: skip


def wind_step(env, p, x, u, t, task, widths, deltas=(0.0, 0.0)):
    """One env step from ``x`` = (omega, pitch, torque) under the raw move
    ``u``: ``(x_next, pitch command)``."""
    _, v_mean, deviation = task
    action = commands(p, u)
    v_wind = v_mean + deviation * (1.0 - WIND_OU_THETA * p.delta_t) ** (t + 1)
    x1, _ = integrate_dynamics(
        positions=x, delta_t=p.delta_t, method=env.integration_method,
        compute_velocity=partial(
            velocity, action=action, v_wind=v_wind, widths=widths, deltas=deltas,
            p=p,
        ),
    )  # fmt: skip
    return x1, action[0]


def wind_model(env, p) -> TaskModel:
    assert float(p.e_tol) == 0.0 and float(p.tracking_exponent) == 1.0
    per_W = float(p.imbalance_price / 1.0e6 * p.delta_t / 3600.0)
    floor = per_W * float(p.e_floor)
    rated = omega_rated(p)

    def step(d, x, u, t):
        task, w = d["task"], d["widths"]  # in name order: cp_floor, pitch_rate
        x1, pitch_cmd = wind_step(env, p, x, u, t, task, (w[0], w[1]))
        power = electrical_power(x1[0], x1[2], p)
        y = jnp.stack([
            (task[0] - power) / p.e_floor,
            (pitch_cmd - x1[1]) / (p.pitch_max * p.c_hold),
            x1[0],
        ])  # fmt: skip
        return x1, jnp.zeros((), x.dtype), y, power[None], ()

    return TaskModel(
        nx=3, nu=2, step=step,
        state_x=lambda s: jnp.stack([s.omega, s.pitch, s.torque]),
        data=lambda s: (s.target_power, s.v_mean, s.v_wind - s.v_mean),
        hold=lambda s: jnp.stack([
            raw_of(s.pitch, p.pitch_min, p.pitch_max),
            raw_of(s.torque, 0.0, p.torque_max),
        ]),
        tracked=lambda s: electrical_power(s.omega, s.torque, p)[None],
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(weight=floor, absolute=True),
            Hinge(weight=floor * float(p.fatigue_weight), absolute=True, tol=1.0),
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
