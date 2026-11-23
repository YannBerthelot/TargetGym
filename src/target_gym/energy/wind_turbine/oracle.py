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
  on its noise-free mean path in the registry's receding slot, the hub's
  turbulence decaying and the rotor's spatial filter following it
  (``rotor_ahead``), and on the episode's own path for the ceiling
  (``ceiling_data``), which knows the turbulence ahead as a ceiling may;
- objective: the score's lost production against the available power (the
  rating, timed curtailment included, or the wind at peak Cp), a one-sided
  epigraph row, and the pitch system's wear, an absolute row on the pitch's
  travel, both in dollars on every visited state (T2). The tower's fatigue is
  the env's rainflow count of the base moment (ASTM E1049), which reads the
  history's turning points and so can't be planned through by a gradient
  planner. It is planned by online rainflow counting (Loew and Bottasso, WES 7,
  1605, 2022): the count is run outside the solver on a plan's own moments
  (``rainflow_rows``), which pairs up the record's turning points, and frozen
  in the data as rows. Each pair's range is charged at its two ends: about the
  partner's moment where the partner is already past, exactly, and about the
  pair's midpoint where both ends are in the plan, |a - b|^m <= 2^(m-1) (|a -
  c|^m + |b - c|^m), which is exact at the plan the rows were counted on and
  never under the count elsewhere. A driver re-solves on the new plan's rows
  until they stop changing. With no rows (the registry's receding slot) the
  fatigue is not planned. The receding slot keeps the speed's resting target
  at rated (G12); the ceiling plans the score alone;
- limit: the rotor speed under the safety system's overspeed level on every
  visited state, elastic at the trip cost per unit (T3);
- the converter's power limit: the env loads the rotor with no more torque than
  delivers the set point (``env.compute_velocity``). The model keeps its torque
  free and pays for power above the set point at ten times the energy price, a
  surrogate (the env charges nothing there: it never delivers it), so a plan
  stays where the model and the env agree. Ten times is ours: an exact penalty
  only has to exceed what shedding power can buy;
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
    rotor_blend,
    rotor_filter,
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


def rotor_ahead(hub, lags, steps, v_mean, p):
    """The turbulence the rotor sees ``steps`` steps on, with no draw to come,
    from the hub's turbulence ``hub`` and the filter's (fast, slow) ``lags`` now.
    The hub's decays by d a step, and each lag's recursion ``lag' = e lag + (1 -
    e) u'`` sums in closed form, ``e^k lag + (1 - e) u d (e^k - d^k) / (e - d)``."""
    lag_fast, lag_slow = lags[0], lags[1]
    t_fast, t_slow, _ = rotor_filter(v_mean, p)
    d = jnp.exp(-p.delta_t / turbulence_time_constant(v_mean, p))

    def ahead(lag, t_lag):
        e = jnp.exp(-p.delta_t / t_lag)
        return e**steps * lag + (1.0 - e) * hub * d * (e**steps - d**steps) / (e - d)

    return rotor_blend(
        ahead(lag_fast, t_fast), ahead(lag_slow, t_slow), hub * d**steps, v_mean, p
    )


def wind_step(env, p, x, v, t, task, widths):
    """One env step from ``x`` = (omega, pitch, torque, x_tower, v_tower,
    w_induced) at the rates ``v`` (fractions of the limits): (x_next, the
    step's wind, the rating's power)."""
    v_mean, v_turb, now, gust_t0, derate_t0, lags, path = task[1:8]
    after = (now + t + 1) * p.delta_t
    if path is None:
        seen = rotor_ahead(v_turb, lags, t + 1, v_mean, p)
        v_wind = v_mean + seen + gust(after, v_mean, gust_t0, p)
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


def filter_lags(state):
    """The rotor's spatial filter as it stands, for the plan's data: (its fast
    lag, its slow lag) of the hub's turbulence. Kept apart from the turbulence
    itself, the deviation the noise drives (``measure_oracle_widths``, K11), and
    ahead of the wind path, which stays the data's last entry."""
    return jnp.stack([state.v_lag_fast, state.v_lag_slow])


def rainflow_rows(stack, depth, moment, moments, exponent):
    """The plan's tower rows, counted on a plan's own base moments ``moments``
    (N m, after each step) from the state's open turning points (``stack``,
    bottom first, ``depth`` of them) and its moment now, as the env will count
    them (``env.rainflow_step``, without its full-stack rule).

    Rainflow on the whole record, open ranges as half cycles, pairs up turning
    points, and every point ends at most two counted ranges. Returns (centres,
    scales, constant), the first two of shape (steps, 2): row k of step t is
    ``scales[t, k] * (moment_t - centres[t, k]) / 1e6``, charged ``|row|^m`` at
    the tower's price per (MN m)^m.

    - Partner in the past (an open point, or the moment now): the centre is
      its moment and the scale ``count^(1/m)``, so the row is the range itself.
    - Partner at another step of the plan: the centre is the pair's midpoint on
      this plan and the scale ``(2^(m-1) count)^(1/m)`` at both ends, which
      adds up to the range's own charge at this plan.
    - ``constant``, (MN m)^m: the ranges between two past points, less what the
      record has been charged already (its open ranges as half cycles).

    So the rows and the constant are what the env charges from here on."""
    m = float(exponent)
    stack = np.asarray(stack, float)[: int(depth)]
    past = [(float(v), -1) for v in stack] + [(float(moment), -1)]
    record = past + [(float(v), t) for t, v in enumerate(np.asarray(moments, float))]
    points = [record[0]]
    for value, step in record[1:]:
        if value == points[-1][0]:
            continue
        rising = points[-1][0] - points[-2][0] if len(points) >= 2 else 0.0
        if rising * (value - points[-1][0]) > 0.0:
            points[-1] = (value, step)
        else:
            points.append((value, step))
    pairs, open_points = [], []
    for point in points:
        open_points.append(point)
        while len(open_points) >= 3:
            a, b, c = open_points[-3], open_points[-2], open_points[-1]
            if abs(c[0] - b[0]) < abs(b[0] - a[0]):
                break
            if len(open_points) == 3:
                pairs.append((a, b, 0.5))
                open_points.pop(0)
            else:
                pairs.append((a, b, 1.0))
                del open_points[-3:-1]
    pairs += [(a, b, 0.5) for a, b in zip(open_points[:-1], open_points[1:])]

    steps = len(moments)
    centres, scales = np.zeros((steps, 2)), np.zeros((steps, 2))
    used = np.zeros(steps, int)

    def row(step, centre, scale):
        centres[step, used[step]], scales[step, used[step]] = centre, scale
        used[step] += 1

    charged = [abs(b[0] - a[0]) / 1.0e6 for a, b in zip(past[:-1], past[1:])]
    constant = -0.5 * sum(r**m for r in charged)
    for (va, ta), (vb, tb), count in pairs:
        if ta < 0 and tb < 0:
            constant += count * (abs(vb - va) / 1.0e6) ** m
        elif ta < 0:
            row(tb, va, count ** (1.0 / m))
        else:
            both = (2.0 ** (m - 1.0) * count) ** (1.0 / m)
            row(ta, 0.5 * (va + vb), both)
            row(tb, 0.5 * (va + vb), both)
    return centres, scales, float(constant)


def wind_model(env, p, rest: bool = True) -> TaskModel:
    per_W = float(1.0e-6 * p.delta_t / 3600.0)  # MWh per W per step
    rated = omega_rated(p)
    per_deg = float(pitch_system_cost(p) / (p.pitch_travel_design * life_steps(p)))
    # $ per (MN m)^m of the tower's rainflow count (``compute_reward_terms``).
    per_damage = float(tower_cost(p) / (p.tower_damage_design * life_steps(p)))
    # $ per W above the set point for a step: the plan's price for power the
    # converter will not deliver.
    per_excess = 10.0 * per_W * float(p.energy_price)

    def step(d, x, u, t):
        task = d["task"]
        x1, v_wind, rating = wind_step(env, p, x, u, t, task, d["widths"])
        power = electrical_power(x1[0], x1[2], p)
        available = jnp.minimum(rating, wind_power(v_wind, p))
        moment = tower_base_moment(x1[3], p)
        if len(task) > 8:  # the tower's rows, frozen by the outer loop
            centres, scales, constant = task[8]
            centres, scales = jnp.asarray(centres, x.dtype), jnp.asarray(
                scales, x.dtype
            )
            k = jnp.minimum(t, centres.shape[0] - 1)
            rows = scales[k] * (moment - centres[k]) / 1.0e6
            cost = jnp.where(t == 0, per_damage * constant, 0.0).astype(x.dtype)
        else:
            rows, cost = jnp.zeros(2, x.dtype), jnp.zeros((), x.dtype)
        y = jnp.stack(
            [available - power, x1[1] - x[1], rows[0], rows[1], power - rating, x1[0]]
        )
        return x1, cost, y, power[None], ()

    return TaskModel(
        nx=6, nu=2, step=step,
        state_x=lambda s: jnp.stack([
            s.omega, s.pitch, s.torque, s.x_tower, s.v_tower, s.w_induced,
        ]),  # fmt: skip
        data=lambda s: (
            s.target_power, s.v_mean, s.v_turb, s.time, s.gust_t0, s.derate_t0,
            filter_lags(s), None,
        ),  # fmt: skip
        hold=lambda s: jnp.zeros(2, s.omega.dtype),
        action=lambda s, v: raw_command(p, jnp.stack([s.pitch, s.torque]), v),
        tracked=lambda s: electrical_power(s.omega, s.torque, p)[None],
        resolution=(float(p.precision_floor),),
        hinges=(
            Hinge(weight=per_W * float(p.energy_price)),
            Hinge(weight=per_deg, absolute=True),
            Hinge(
                weight=per_damage, absolute=True, power=int(p.fatigue_exponent)
            ),
            Hinge(
                weight=per_damage, absolute=True, power=int(p.fatigue_exponent)
            ),
            Hinge(weight=per_excess, surrogate=True),
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
        state.derate_t0, filter_lags(state), path,
    )  # fmt: skip


def make_wind_turbine_mpc(env, params) -> Oracle:
    """Registry factory: the shared solver's oracle on ``wind_model``."""
    return Oracle(env, params, "wind_turbine", wind_model)
