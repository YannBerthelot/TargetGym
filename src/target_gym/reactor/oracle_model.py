"""The reactor planner's model: the plant from knot to knot, its heat in balance.

The task's trade lives on the boron line and the xenon, over hours; the fuel, the
coolant and the neutrons settle in under a minute. So the planner (``oracle.py``)
does not integrate them. Between two knots, ``KNOT`` control steps apart (5 min), it
carries only what is slow:

- iodine and xenon, their means and their axial tilts, by one RK4 step on the env's
  own rates (``env._slow_rates``) with the power and the rods on a straight line
  between the knots;
- the boron, by the env's exact mixing, control step by control step, under the
  flow that *arrives*: the commands already in the line (``Day.arriving``, the
  first ``Day.committed`` control steps), then the planned one;
- the rods, where the planned move leaves them;
- the turbine's shortfall, its plan's load less the load it holds;
- the offset's penalty time, the share of the knot a straight line between the two
  knots' offsets spends past the band.

At a knot the temperatures are the ones that leave the core critical at the
turbine's load (``balance``): the env's reactivity (``env._reactivity``) set to
zero with the power equal to the load, solved for Tavg in closed form. The axial
offset is the env's (``env.flux_tilt``) at those temperatures.

What the model leaves out is the lag of that balance: the loop's heat capacity
warms on the power the core makes over the load, which closes the gap in 76 s at
the start of the cycle and 22 s at 80 % of it. A first-order correction for it was
tried and did not bring the model nearer the env on a followed plan (0.80 K worst
with it, 0.68 K without), so the planner backs its Tavg limits off by the gap
instead (``oracle_day.TAVG_BACKOFF``).
``tests/reactor/test_reactor_oracle_model.py`` holds the model to the env at the
knots.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from target_gym.oracles.controller import softplus_clip
from target_gym.reactor.env import (
    _slow_rates,
    core_at,
    flux_tilt,
    moderator_tilt,
    rod_insertion_limit,
    rod_tilt,
    rod_worth,
    tavg_program,
    turbine_capacity,
)

#: Control steps between two knots of the plan: 5 min.
KNOT = 10
#: Where each planned state sits in ``x``.
IODINE, XENON, I_TILT, XE_TILT, BORON, RODS, SHORT, CLOCK = range(8)
NX = 8
#: Where each planned move sits in ``u``: the rods' travel in and out over the
#: knot (of the stroke), the dilution and boration that arrive (of full flow), and
#: the turbine's load change over the knot (of its fastest).
ROD_IN, ROD_OUT, DILUTE, BORATE, TURBINE = range(5)
NU = 5
#: The planner's units, so every state is of order one: the tilts in percent, the
#: boron in hundreds of ppm, the penalty time in hours.
TILT, PPM = 100.0, 100.0


class Day(NamedTuple):
    """What a solve knows of the time ahead, one value per knot from the solve's
    start: the plan's load, the most the turbine may hold (the plan, or the
    conditioning ramps under it after a restart), the load's wobble, the boron
    burnt since the last refuelling (ppm), and whether the knot is one the plan may
    spend outside the offset's band (``free``, 1 or 0). And the boron line: the
    commands sent and not yet arrived, by the control step they arrive on, and how
    many. ``count`` is 1 where the plan counts its own penalty time (every knot is
    then free), 0 where the free knots are given and already counted."""

    plan: jnp.ndarray
    cap: jnp.ndarray
    wobble: jnp.ndarray
    burnt: jnp.ndarray
    arriving: jnp.ndarray
    committed: jnp.ndarray
    cycle0: jnp.ndarray
    free: jnp.ndarray
    count: jnp.ndarray


def balance(n, rod_in, boron, xenon, burnt, cycle0, p):
    """``(T_avg, T_hot, T_cold)`` that leave the core critical at power ``n`` with
    the heat in balance: the fuel ``n P / UA`` above the hot leg, the hot leg half
    the core's rise above Tavg, and the reactivity of the rods, the boron, the
    burnup, both temperature feedbacks and the xenon summing to zero."""
    core = core_at(cycle0 + burnt / p.boron_ref, p)
    renewed_at = p.boron_ref * (1.0 - cycle0)
    rho = rod_worth(rod_in, p) - rod_worth(p.rod_park, p)
    rho = rho + core.boron_worth * (boron - renewed_at + burnt)
    power = p.P_thermal_ref * n
    rise = 0.5 * power / p.m_dot_cp
    above = rise + power / p.UA  # the fuel over Tavg
    a_f, a_c = core.alpha_fuel, core.alpha_coolant
    T_avg = (
        a_f * (p.T_fuel_ref - above)
        + a_c * p.T_coolant_ref
        + p.rho_Xe_full * (xenon - 1.0)
        - rho
    ) / (a_f + a_c)
    return T_avg, T_avg + rise, T_avg - rise


def offset(n, rod_in, xenon, xe_tilt, p):
    """Delta I at power ``n`` with the heat in balance: the env's flux tilt under
    the rods' and the coolant rise's absorption difference."""
    rise = 0.5 * p.P_thermal_ref * n / p.m_dot_cp
    eps = rod_tilt(rod_in, p) + moderator_tilt(rise, -rise, p)
    return n * flux_tilt(xe_tilt, xenon, n, eps, p)


def xenon_rates(X, n, rod_in, p):
    """Rates of ``[I, Xe, I_tilt, Xe_tilt]`` at power ``n``: the env's."""
    rise = 0.5 * p.P_thermal_ref * n / p.m_dot_cp
    slow = jnp.concatenate([jnp.stack([0.0 * n, rise, -rise, 0.0 * n]), X])
    return _slow_rates(slow, n, n, rod_tilt(rod_in, p), p)[4:]


def mixed(boron, dilute, borate, seconds, p):
    """The boron after ``seconds`` of water at ``dilute`` and acid at ``borate`` of
    their full flows: the env's feed and bleed of a well-mixed primary, written so
    no flow is no 0/0."""
    acid = borate * p.boration_flow
    z = (dilute * p.dilution_flow + acid) * seconds / p.primary_mass
    safe = jnp.where(z > 1e-12, z, 1.0)
    share = jnp.where(z > 1e-12, -jnp.expm1(-safe) / safe, 1.0 - 0.5 * z)
    inflow = acid * p.boric_acid_ppm * seconds / p.primary_mass
    return boron + (inflow - z * boron) * share


def noisy_excess(error, tol, spread):
    """The mean of ``max(|error + e| - tol, 0)`` for a normal ``e`` of standard
    deviation ``spread``: how far past its free band a Tavg that wobbles about the
    plan's sits on average, each side ``spread phi(z) + (m - tol) Phi(z)`` at ``z =
    (m - tol) / spread``. The plan's Tavg is the model's at the wobble's mean; the
    plant's wobbles with the load."""
    norm = jax.scipy.stats.norm

    def side(m):
        z = (m - tol) / spread
        return spread * norm.pdf(z) + (m - tol) * norm.cdf(z)

    return side(error) + side(-error)


def time_outside(before, after, band, backoff, width):
    """The share of a knot spent past ``band``, counted from ``backoff`` inside it,
    by an offset on a straight line from ``|before|`` to ``|after|``: 0 with both
    ends inside, 1 with both outside, the crossing's share between. The corners
    are rounded at ``width`` (of the share), so a plan can move a crossing."""
    a, b = jnp.abs(before), jnp.abs(after)
    span = jnp.sqrt((a - b) ** 2 + backoff**2)
    high = 0.5 * (a + b + span)
    return softplus_clip((high - band + backoff) / span, 0.0, 1.0, width)


def penalty_share(n):
    """How much of the time outside counts at power ``n``: all of it from 47 % of
    rated power up, half of it at 42 % and below, never less than the env counts
    (all at 50 % and above, half below)."""
    s = jnp.clip((n - 0.42) / 0.05, 0.0, 1.0)
    return 0.5 + 0.5 * s * s * (3.0 - 2.0 * s)


def offset_room(n, p, width=0.02):
    """The most |delta I - target| the plan allows at power ``n``, inside the env's
    limits (``env.offset_limit``): the band from 90 % of rated power up, where the
    env gives 30 min to be back in it, and the acceptable-operation line below,
    reached over the 5 % of power under 90 %. The line stops rising at 95 % of its
    way to 50 % of rated power, rounded at ``width`` of that way, so the plan has
    no corner at a hold there; under 50 % the env has no limit."""
    way = (p.ao_strict_power - n) / (p.ao_strict_power - p.ao_half_power)
    way = 0.95 - width * jax.nn.softplus((0.95 - jnp.maximum(way, 0.0)) / width)
    line = p.ao_limit_high + (p.ao_limit_low - p.ao_limit_high) * way
    s = jnp.clip((p.ao_strict_power - n) / 0.05, 0.0, 1.0)
    return p.ao_band + (line - p.ao_band) * s * s * (3.0 - 2.0 * s)


def knot_step(p, d: Day, x, u, k, backoff, width, knot=KNOT, period=6):
    """One knot from ``x`` under ``u``, ``k`` knots into the plan: ``(x_next,
    rows)``, with ``rows`` the next knot's (Tref - Tavg, Tavg, delta I less its
    target, the offset's room, the rods' room inside their insertion limit, the
    turbine's room under its cap, and under what its valves pass wide open on the
    steam the balance makes, and how far a unit of load moves Tavg's error). ``backoff`` is the planner's error on delta I
    and ``width`` the corners' (``time_outside``)."""
    step = period * p.delta_t  # a control step, s
    seconds = knot * step
    X0 = jnp.stack([x[IODINE], x[XENON], x[I_TILT] / TILT, x[XE_TILT] / TILT])
    B0, r0, s0 = x[BORON] * PPM, x[RODS], x[SHORT]
    r1 = r0 + u[ROD_IN] - u[ROD_OUT]
    L0 = d.plan[k] - s0
    L1 = L0 + u[TURBINE] * p.turbine_rate * seconds
    n0, n1 = L0 + d.wobble[k], L1 + d.wobble[k + 1]

    def mix(B, i):
        g = k * knot + i
        sent = g < d.committed
        c = d.arriving[jnp.minimum(g, d.arriving.shape[0] - 1)]
        dilute = jnp.where(sent, jnp.maximum(-c, 0.0), u[DILUTE])
        borate = jnp.where(sent, jnp.maximum(c, 0.0), u[BORATE])
        return mixed(B, dilute, borate, step, p), None

    B1, _ = jax.lax.scan(mix, B0, jnp.arange(knot))

    def rates(X, a):
        core = core_at(d.cycle0 + d.burnt[k] / p.boron_ref, p)
        return xenon_rates(X, n0 + a * (n1 - n0), r0 + a * (r1 - r0), core)

    k1 = rates(X0, 0.0)
    k2 = rates(X0 + 0.5 * seconds * k1, 0.5)
    k3 = rates(X0 + 0.5 * seconds * k2, 0.5)
    k4 = rates(X0 + seconds * k3, 1.0)
    X1 = X0 + seconds / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    core = core_at(d.cycle0 + d.burnt[k + 1] / p.boron_ref, p)
    T_avg, _, _ = balance(n1, r1, B1, X1[1], d.burnt[k + 1], d.cycle0, p)
    off0 = offset(n0, r0, X0[1], X0[3], core) - n0 * p.ao_ref
    off1 = offset(n1, r1, X1[1], X1[3], core) - n1 * p.ao_ref
    share = time_outside(off0, off1, p.ao_band, backoff, width)
    counted = d.count * penalty_share(0.5 * (n0 + n1)) * share
    clock = x[CLOCK] + seconds / 3600.0 * counted
    # Outside the knots the plan may spend past the band, the room is the band.
    room = p.ao_band + d.free[k + 1] * (offset_room(n1, p) - p.ao_band)

    x1 = jnp.stack([
        X1[0], X1[1], X1[2] * TILT, X1[3] * TILT, B1 / PPM, r1,
        d.plan[k + 1] - L1, clock,
    ])  # fmt: skip
    steam = T_avg - p.P_thermal_ref * n1 / p.UA_sg  # the steam generator's, settled
    # Tavg's error per unit of load, rods and boron still: the program's span and
    # the fuel's Doppler over both feedbacks.
    fuel = p.P_thermal_ref * (0.5 / p.m_dot_cp + 1.0 / p.UA)
    sway = (p.T_avg_full_load - p.T_avg_no_load) + core.alpha_fuel * fuel / (
        core.alpha_fuel + core.alpha_coolant
    )
    rows = jnp.stack([
        tavg_program(n1, p) - T_avg, T_avg, off1, room,
        rod_insertion_limit(n1, p) - r1, d.cap[k + 1] - L1,
        turbine_capacity(steam, p) - L1, sway,
    ])  # fmt: skip
    return x1, rows


def planned_flows(d: Day, k, knot=KNOT):
    """How many of knot ``k``'s control steps the planned flow arrives on."""
    return knot - jnp.clip(d.committed - k * knot, 0, knot)
