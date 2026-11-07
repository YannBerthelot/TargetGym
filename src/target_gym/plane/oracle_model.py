"""The aircraft's equations as the oracles plan on them.

``plane.dynamics`` restated with its kinks treated by the smoothing protocol's
inventory (docs/proposals/oracle-smoothing-protocol.md, section 5.2), so that the
2D, 3D, patrol and formation oracles share one statement of them. At width 0 and
with no dropped guard crossed, every function here equals the env's to rounding,
which K8's replay checks on every plan.

The sites smoothed by K2's softplus (T5), in the order of ``Sites``:

- ``buffet``, the wing's lift capped at the buffet line;
- ``cl_stall``, the stall clip of the section lift, read by wing, tail and elevator;
- ``separation``, ``abs(incidence)`` in the separated-flow blend;
- ``surface_clip``, the attitude loop's output, and its integral's freeze, which
  blends by the softplus's slope (K2's switch);
- ``surface_rate``, the rate limit of a surface. T1r's bound is empty: the reach
  over a hold of one step, the rate times the step and the lag, already passes a
  surface's whole travel, so any command is inside it (``surface_reach``);
- ``tropopause``, the switch of the sound speed at 11 km.

Dropped guards (T4), which ``Flown.guards`` checks along every plan: the Mach
switches past ``M_crit`` (shock stall, drag rise), the section's lift and drag
caps, the incidence wrap, the angular acceleration clip, the scaler's clip and
the tanks. The altitude clip of the density and the Mach branch of the speed
schedule are the env's own functions, called off their edges.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.tree_util import Partial as partial

from target_gym.integration import integrate_dynamics
from target_gym.oracles.controller import TaskModel, raw_of, softplus_clip
from target_gym.oracles.shooting import Limit
from target_gym.plane.dynamics import (
    AILERON_TRAVEL,
    BANK_DEMAND_MAX,
    INNER_STEPS,
    PITCH_DEMAND_MAX,
    STICK_TRAVEL,
    airspeed_scaler,
    buffet_line,
    compute_air_density_from_altitude,
    compute_drag,
    compute_thrust_output,
    scheduled_airspeed,
    total_wind_3d,
)
from target_gym.reward import trip_cost

#: The smoothed sites, name order, as in ``data/oracle_widths``.
SITE_NAMES = ("buffet", "cl_stall", "separation", "surface_clip", "surface_rate")
SITE_NAMES += ("tropopause",)
#: The sound speed's temperature at the tropopause (K).
T11 = 216.65
#: Past these the env's clips of the angular accelerations bind (rad/s^2) and the
#: section's lift cap does.
ANGULAR_CLIP, SECTION_CAP = 1.5, 2.0


class Sites(NamedTuple):
    """One value per smoothed site: a width, or a delta added at its output."""

    buffet: jnp.ndarray
    cl_stall: jnp.ndarray
    separation: jnp.ndarray
    surface_clip: jnp.ndarray
    surface_rate: jnp.ndarray
    tropopause: jnp.ndarray


def sites(values) -> Sites:
    return Sites(*(values[i] for i in range(len(SITE_NAMES))))


NO_DELTA = Sites(*(0.0 for _ in SITE_NAMES))


def plus(z, w):
    """K2's p_w(z), with its width guarded (JAX FAQ); ``max(z, 0)`` at 0."""
    v = jnp.where(w > 0.0, w, 1.0)
    return jnp.where(w > 0.0, v * jax.nn.softplus(z / v), jnp.maximum(z, 0.0))


def soft_abs(z, w):
    """K2's abs(z), exact at width 0."""
    return jnp.where(w > 0.0, plus(z, w) + plus(-z, w), jnp.abs(z))


def surface_reach(rate_limit, response_rate, dt):
    """T1r for a surface, as for the wind's pitch: the command, from its position,
    past which the surface moves at its full rate over a hold of ``dt``."""
    return rate_limit * (dt + 1.0 / response_rate)


def sound_speed(z, w, d):
    """``compute_speed_of_sound_from_altitude`` with its switch smoothed."""
    temperature = T11 + plus(288.15 - 0.0065 * z - T11, w.tropopause) + d.tropopause
    return jnp.sqrt(1.4 * 287.0 * temperature)


def section(aoa_deg, mach, p, w, d):
    """``aero_coefficients`` below ``M_crit``: ``(CL, CD)``, its guards dropped."""
    centre = p.aoa_stall + p.aoa_stall_width
    sharp = jnp.log(99.0) / p.aoa_stall_width
    lift = p.cl0 + p.cl_alpha * aoa_deg
    lift = lift * jax.nn.sigmoid(-(aoa_deg - centre) * sharp)
    lift = softplus_clip(lift, p.CL_min, p.CL_max, w.cl_stall) + d.cl_stall
    drag = p.cd0 + p.k * lift**2
    lift = lift / jnp.sqrt(1.0 - mach**2)
    incidence = soft_abs(aoa_deg, w.separation) + d.separation
    separated = jax.nn.sigmoid((incidence - centre) * sharp)
    rad = jnp.deg2rad(aoa_deg)
    cn_max = 1.11 + 0.018 * p.aspect_ratio
    plate_lift = 0.5 * cn_max * jnp.sin(2.0 * rad)
    plate_drag = p.cd0 + cn_max * jnp.sin(rad) ** 2
    return (
        (1.0 - separated) * lift + separated * plate_lift,
        (1.0 - separated) * drag + separated * plate_drag,
    )


class Airframe(NamedTuple):
    """What the longitudinal aerodynamics give: forces, pitching moment and the
    largest section lift, for its guard."""

    lift: jnp.ndarray
    drag: jnp.ndarray
    moment: jnp.ndarray
    cl_peak: jnp.ndarray


def airframe(alpha, speed, mach, rho, q, stick, p, w, d) -> Airframe:
    """The wing, the stabiliser and the elevator (``compute_acceleration``)."""
    deg = jnp.rad2deg(alpha)
    cz_w, cx_w = section(deg, mach, p, w, d)
    cap = buffet_line(mach, p)
    v = jnp.where(w.buffet > 0.0, w.buffet, 1.0)
    capped = cz_w - v * jax.nn.softplus((cz_w - cap) / v)
    cz_w = jnp.where(w.buffet > 0.0, capped, jnp.minimum(cz_w, cap)) + d.buffet
    lift_w = compute_drag(S=p.wings_surface, C=cz_w, V=speed, rho=rho)
    drag_w = compute_drag(S=p.wings_surface, C=cx_w, V=speed, rho=rho)
    moment = (lift_w * jnp.cos(alpha) + drag_w * jnp.sin(alpha)) * p.moment_arm_wings

    tail_deg = jnp.rad2deg(jnp.arctan2(q * p.moment_arm_stabilizer, speed))
    aoa_s = deg - 3.0 + tail_deg
    cz_s, cx_s = section(aoa_s, mach, p, w, d)
    lift_s = compute_drag(S=p.stabilizer_surface, C=cz_s, V=speed, rho=rho)
    drag_s = compute_drag(S=p.stabilizer_surface, C=cx_s, V=speed, rho=rho)
    rad_s = jnp.deg2rad(aoa_s)
    normal_s = lift_s * jnp.cos(rad_s) + drag_s * jnp.sin(rad_s)

    aoa_e = aoa_s - jnp.rad2deg(stick)
    cz_e, cx_e = section(aoa_e, mach, p, w, d)
    lift_e = compute_drag(S=p.elevator_surface, C=cz_e, V=speed, rho=rho)
    drag_e = compute_drag(S=p.elevator_surface, C=cx_e, V=speed, rho=rho)
    rad_e = jnp.deg2rad(aoa_e)
    normal_e = lift_e * jnp.cos(rad_e) + drag_e * jnp.sin(rad_e)

    moment = moment - (normal_s + normal_e) * p.moment_arm_stabilizer
    peak = jnp.max(jnp.abs(jnp.stack([cz_s, cz_e])))
    return Airframe(lift_w + lift_s + lift_e, drag_w + drag_s + drag_e, moment, peak)


def attitude(kp, ki, kd, error, rate, integral, scaler, dt, width, delta):
    """``attitude_axis``, its clip smoothed, its integral growing by the clip's
    slope: 1 inside, 0 outside, ``output == surface`` at width 0."""
    output = kp * scaler * error + integral - kd * scaler**2 * rate
    surface = softplus_clip(output, -1.0, 1.0, width) + delta
    v = jnp.where(width > 0.0, width, 1.0)
    slope = 1.0 - jax.nn.sigmoid((output - 1.0) / v)
    slope = slope - jax.nn.sigmoid((-1.0 - output) / v)
    free = jnp.where(width > 0.0, slope, jnp.where(jnp.abs(output) <= 1.0, 1.0, 0.0))
    return surface, integral + free * ki * scaler**2 * error * dt


def surface_step(requested, current, dt, response_rate, rate_limit, width, delta):
    """``compute_next_surface``, its rate limit smoothed."""
    move = (requested - current) * (1.0 - jnp.exp(-response_rate * dt))
    most = rate_limit * dt
    return current + softplus_clip(move, -most, most, width) + delta


def engine(lever, power, dt, p):
    """``compute_next_power`` with the lever's clip removed (T1)."""
    requested = p.idle_power + (1.0 - p.idle_power) * lever
    return requested + (power - requested) * jnp.exp(-p.power_response_rate * dt)


#: The track's place in a plan's state with the position kept.
TRACK = 4
#: The speed schedule of the 2D tasks and of the 3D autopilot's cruise, which is
#: the design point of a task that does not price its speed (G12).
DESIGN_CAS, DESIGN_MACH = 290.0 * 1852.0 / 3600.0, 0.78
#: The 3D flight state, in the order of ``flight_step``'s ``x``.
FLIGHT_FIELDS = ("x", "y", "z", "x_dot", "y_dot", "z_dot", "theta", "phi")
FLIGHT_FIELDS += ("theta_dot", "phi_dot", "power", "stick", "aileron")
FLIGHT_FIELDS += ("pitch_integ", "roll_integ", "fuel")


def flight_x(state):
    """The planner's 3D state of an aircraft's env state."""
    return jnp.stack([getattr(state, f) for f in FLIGHT_FIELDS])


def to_polar(x):
    """``x`` with its horizontal velocity as (speed, track): a plan's states are its
    variables, and one that reverses course in Cartesian velocity passes a cold
    start's Newton steps through zero speed, where nothing in the model holds."""
    return x.at[3].set(jnp.hypot(x[3], x[4])).at[4].set(jnp.arctan2(x[4], x[3]))


def to_cartesian(x):
    """The inverse of ``to_polar``, for any branch of the track."""
    return x.at[3].set(x[3] * jnp.cos(x[4])).at[4].set(x[3] * jnp.sin(x[4]))


def polar_step(x1, track):
    """``x1`` in ``to_polar``'s form, its track on the branch of ``track``, a step
    earlier, by the turn over that step, which never reaches the half-turn."""
    sin_t, cos_t = jnp.sin(track), jnp.cos(track)
    turned = jnp.arctan2(x1[4] * cos_t - x1[3] * sin_t, x1[3] * cos_t + x1[4] * sin_t)
    return x1.at[3].set(jnp.sqrt(x1[3] ** 2 + x1[4] ** 2)).at[4].set(track + turned)


def on_branch(angle, reference):
    """``angle`` plus the whole turns that put it nearest ``reference``."""
    return angle + 2.0 * jnp.pi * jnp.round((reference - angle) / (2.0 * jnp.pi))


def decayed(gust, t, p):
    """The noise-free gust ``t + 1`` steps on from the measured ``gust``."""
    return gust * (1.0 - p.turbulence_theta * p.delta_t) ** (t + 1)


class Flown(NamedTuple):
    """A task step's outputs for the limits and the guards: the loads at the inner
    sub-steps, EAS and Mach on the state reached, each guard's smallest argument."""

    loads: jnp.ndarray
    eas: jnp.ndarray
    mach: jnp.ndarray
    guards: jnp.ndarray


def accelerate_3d(velocities, positions, thrust, stick, aileron, air, mass, p, w, d):
    """``compute_acceleration_3d``, its guards dropped. ``air`` is the wind and the
    gust's roll rate. The flight-path and track angles enter by their sines and
    cosines, formed from the velocity, so no cut of an arctan2 is met."""
    vx, vy, vz, q, roll_rate = velocities
    _, _, z, theta, phi = positions
    ax, ay, az = vx - air[0], vy - air[1], vz - air[2]
    horizontal = jnp.sqrt(ax**2 + ay**2)
    speed = jnp.sqrt(ax**2 + ay**2 + az**2)
    sin_g, cos_g, sin_t, cos_t = (
        az / speed,
        horizontal / speed,
        ay / horizontal,
        ax / horizontal,
    )
    alpha = theta - jnp.arctan2(az, horizontal)
    mach = speed / sound_speed(z, w, d)
    rho = compute_air_density_from_altitude(z)
    a = airframe(alpha, speed, mach, rho, q, stick, p, w, d)
    dynamic = 0.5 * rho * speed**2
    roll_moment = (
        dynamic * p.aileron_surface * p.cl_alpha * p.aileron_effectiveness
        * jnp.rad2deg(aileron) * p.moment_arm_aileron
    )  # fmt: skip
    roll_moment = roll_moment + (
        p.C_lp * ((roll_rate - air[3]) * p.wingspan / (2.0 * speed))
        * dynamic * p.wings_surface * p.wingspan
    )  # fmt: skip
    sin_b, cos_b = jnp.sin(phi), jnp.cos(phi)
    sin_p, cos_p = jnp.sin(theta), jnp.cos(theta)
    up = jnp.stack([-sin_g * cos_t, -sin_g * sin_t, cos_g])
    right = jnp.stack([-sin_t, cos_t, jnp.zeros_like(cos_t)])
    along = jnp.stack([cos_g * cos_t, cos_g * sin_t, sin_g])
    body = jnp.stack([cos_p * cos_t, cos_p * sin_t, sin_p])
    force = thrust * body - a.drag * along + a.lift * (cos_b * up + sin_b * right)
    force = force.at[2].add(-mass * p.gravity)
    accel = jnp.concatenate(
        [force / mass, jnp.stack([a.moment / p.I, roll_moment / p.I_x])]
    )
    return accel, (a.lift, mach, a.cl_peak, accel[3:], alpha)


def flight_step(p, x, u, t, gust, method, widths, deltas=NO_DELTA, roll_gust=None):
    """One task step of a 3D aircraft: ``(x_next, Flown)``, ``x`` in the order of
    ``FLIGHT_FIELDS``, ``u`` (throttle, pitch demand, bank demand) raw, unclipped
    (T1), ``gust`` the air it flies in this step, and ``roll_gust`` added to the
    gust's roll rate, as a wake's is."""
    dt = p.delta_t / INNER_STEPS
    wind = total_wind_3d(x[2], gust[0], gust[1], gust[2], p)
    roll = p.gust_roll_rate if roll_gust is None else p.gust_roll_rate + roll_gust
    air = jnp.stack([*wind, jnp.asarray(roll, x.dtype)])
    mass = p.initial_mass - (p.initial_fuel_quantity - x[15])
    lever = 0.5 * (u[0] + 1.0)
    pitch_demand, bank_demand = u[1] * PITCH_DEMAND_MAX, u[2] * BANK_DEMAND_MAX
    g = p.inner_loop
    w = sites(widths)

    def inner(carry, _):
        velocities, positions, rest = carry
        power, stick, aileron, pitch_integ, roll_integ, burned = rest
        vx, vy, vz, q, roll_rate = velocities
        z, theta, phi = positions[2], positions[3], positions[4]
        air_speed = jnp.sqrt(
            (vx - wind[0]) ** 2 + (vy - wind[1]) ** 2 + (vz - wind[2]) ** 2
        )
        rho = compute_air_density_from_altitude(z)
        eas = air_speed * jnp.sqrt(rho / p.air_density_at_sea_level)
        scaler = airspeed_scaler(eas, g)
        stick_out, pitch_integ = attitude(
            g.Kp_theta, g.Ki_theta, g.Kd_theta, pitch_demand - theta, q, pitch_integ,
            scaler, dt, w.surface_clip, deltas.surface_clip,
        )  # fmt: skip
        aileron_out, roll_integ = attitude(
            g.Kp_roll, g.Ki_roll, g.Kd_roll, bank_demand - phi, roll_rate, roll_integ,
            scaler, dt, w.surface_clip, deltas.surface_clip,
        )  # fmt: skip
        stick = surface_step(
            stick_out * STICK_TRAVEL, stick, dt, p.stick_response_rate,
            p.surface_rate_limit, w.surface_rate, deltas.surface_rate,
        )  # fmt: skip
        aileron = surface_step(
            aileron_out * AILERON_TRAVEL, aileron, dt, p.aileron_response_rate,
            p.surface_rate_limit, w.surface_rate, deltas.surface_rate,
        )  # fmt: skip
        power = engine(lever, power, dt, p)
        thrust = compute_thrust_output(
            power=power,
            thrust_output_at_sea_level=p.thrust_output_at_sea_level,
            rho=rho,
            M=air_speed / sound_speed(z, w, deltas),
        )
        velocities, positions, metrics = integrate_dynamics(
            velocities=velocities,
            positions=positions,
            delta_t=dt,
            compute_acceleration=partial(
                accelerate_3d,
                thrust=thrust,
                stick=stick,
                aileron=aileron,
                air=air,
                mass=mass,
                p=p,
                w=w,
                d=deltas,
            ),
            method=method,
        )
        burned = burned + p.specific_fuel_consumption * 1e-3 * thrust * dt
        lift, mach, peak, angular, alpha = jax.tree_util.tree_map(
            lambda m: m[-1], metrics
        )
        guards = jnp.stack([
            p.M_crit - mach,
            SECTION_CAP - peak,
            ANGULAR_CLIP - jnp.max(jnp.abs(angular)),
            jnp.pi - jnp.abs(alpha),
            eas - g.scaling_speed / g.scale_max,
        ])  # fmt: skip
        rest = (power, stick, aileron, pitch_integ, roll_integ, burned)
        return (velocities, positions, rest), (lift / (mass * p.gravity), guards)

    start = (
        jnp.stack([x[3], x[4], x[5], x[8], x[9]]),
        jnp.stack([x[0], x[1], x[2], x[6], x[7]]),
        (x[10], x[11], x[12], x[13], x[14], jnp.zeros_like(x[0])),
    )
    end, (loads, guards) = jax.lax.scan(inner, start, None, length=INNER_STEPS)
    (vx, vy, vz, q, roll_rate), (px, py, z, theta, phi) = end[0], end[1]
    power, stick, aileron, pitch_integ, roll_integ, burned = end[2]
    fuel = x[15] - burned
    x1 = jnp.stack([
        px, py, z, vx, vy, vz, theta, phi, q, roll_rate, power, stick, aileron,
        pitch_integ, roll_integ, fuel,
    ])  # fmt: skip
    wind1 = total_wind_3d(z, gust[0], gust[1], gust[2], p)
    air_speed = jnp.sqrt(
        (vx - wind1[0]) ** 2 + (vy - wind1[1]) ** 2 + (vz - wind1[2]) ** 2
    )
    rho = compute_air_density_from_altitude(z)
    eas = air_speed * jnp.sqrt(rho / p.air_density_at_sea_level)
    mach = air_speed / sound_speed(z, w, deltas)
    return x1, Flown(loads, eas, mach, jnp.append(jnp.min(guards, axis=0), fuel))


def turn_distance(sin_e, cos_e, width, delta=0.0):
    """How far the track is from a target, over the shorter turn (S4): the min of
    the two turn directions, ``l`` left and ``2 pi - l`` right, by K2's softplus,
    from the error's sine and cosine. ``l``'s cut falls where the two swap, so the
    value is smooth there, and it is ``abs(wrapped error)`` at width 0."""
    left = jnp.pi - jnp.arctan2(-sin_e, -cos_e)
    right = 2.0 * jnp.pi - left
    return left - plus(left - right, width) + delta


def flight_limits(p, price):
    """T3: the band, the load at each sub-step, the bank, VMO and MMO, in the order
    of ``flight_outputs``."""
    return (
        Limit(float(p.min_alt), float(p.max_alt), price),
        *(
            Limit(float(p.load_factor_min), float(p.load_factor_max), price)
            for _ in range(INNER_STEPS)
        ),
        Limit(-float(p.bank_limit), float(p.bank_limit), price),
        Limit(hi=float(p.vmo_eas), price=price),
        Limit(hi=float(p.mmo), price=price),
    )


def flight_outputs(x1, flown):
    """The limited quantities of ``flight_limits`` on a step's outputs."""
    return jnp.concatenate(
        [x1[2:3], flown.loads, x1[7:8], flown.eas[None], flown.mach[None]]
    )


def hold_flight(state, p):
    """The raw moves that hold an aircraft's throttle and attitude."""
    lever = (state.power - p.idle_power) / (1.0 - p.idle_power)
    return jnp.stack([
        raw_of(lever, 0.0, 1.0),
        raw_of(state.theta, -PITCH_DEMAND_MAX, PITCH_DEMAND_MAX),
        raw_of(state.phi, -BANK_DEMAND_MAX, BANK_DEMAND_MAX),
    ])  # fmt: skip


def speed_rest(x1, p):
    """G12's distance from the design speed, the schedule's, 1 at VMO, for a task
    whose reward leaves the speed free."""
    speed = jnp.sqrt(x1[3] ** 2 + x1[4] ** 2 + x1[5] ** 2)
    design = scheduled_airspeed(x1[2], DESIGN_CAS, DESIGN_MACH)
    rho = compute_air_density_from_altitude(x1[2])
    edge = p.vmo_eas * jnp.sqrt(p.air_density_at_sea_level / rho)
    return (speed - design) / (edge - design)


def flight_model(env, p, tracking, *, data, tracked, resolution, drop=0, **fields):
    """The ``TaskModel`` of a task flown by one 3D aircraft, its velocity in polar
    form. ``tracking(d, x1, u, t, widths)`` is the reward on the state reached:
    ``(cost, tracked outputs, further limited quantities)``. ``drop`` leading
    position fields stay out of the plan, where nothing depends on them. ``air(d, x,
    t)`` is the step's ``(gust, roll gust)``, the last field of the task's data
    decaying if not given. Further ``limits`` follow the aircraft's own."""
    method = env.integration_method
    track = TRACK - drop
    air = fields.pop("air", lambda d, x, t: (decayed(d["task"][-1], t, p), None))
    limits = fields.pop("limits", ())
    rest = fields.pop("rest", True)

    def full(x):
        return to_cartesian(jnp.concatenate([jnp.zeros(drop, x.dtype), x]))

    def step(d, x, u, t):
        widths = d["widths"]
        gust, roll = air(d, x, t)
        x1, flown = flight_step(
            p, full(x), u[:3], t, gust, method, widths[: len(SITE_NAMES)],
            roll_gust=roll,
        )  # fmt: skip
        x1 = polar_step(x1, x[track])
        cost, outputs, more = tracking(d, x1, u, t, widths)
        y = jnp.concatenate([flight_outputs(x1, flown), jnp.asarray(more, x.dtype)])
        return x1[drop:], cost, y, outputs, flown.guards

    def align(x, planned):
        return x.at[track].set(on_branch(x[track], planned[track]))

    return TaskModel(
        nx=16 - drop, step=step, data=data, tracked=tracked, resolution=resolution,
        state_x=fields.pop("state_x", lambda s: to_polar(flight_x(s))[drop:]),
        limits=(*flight_limits(p, float(trip_cost(p))), *limits),
        rest=(lambda x1: speed_rest(full(x1), p)) if rest else None,
        align=align,
        **{"nu": 3, "hold": lambda s: hold_flight(s, p), **fields},
    )  # fmt: skip
