"""The aircraft's equations as the oracles plan on them.

``plane.dynamics`` restated with its kinks treated by the smoothing protocol's
inventory (docs/proposals/oracle-smoothing-protocol.md, section 5.2), so that the
2D, 3D, patrol and formation oracles share one statement of them. At width 0 and
with no dropped guard crossed, every function here equals the env's to rounding,
which K8's replay checks on every plan.

The sites smoothed by K2's softplus (T5), in the order of ``Sites``:

- ``buffet``, the wing's lift capped at the buffet line;
- ``cl_stall``, the stall clip of the section lift, read by wing and tail;
- ``separation``, ``abs(incidence)`` in the separated-flow blend;
- ``surface_clip``, the attitude loop's output, and its integral's freeze, which
  blends by the softplus's slope (K2's switch);
- ``surface_rate``, the rate limit of a surface. T1r's bound is empty: the reach
  over a hold of one step, the rate times the step and the lag, already passes a
  surface's whole travel, so any command is inside it (``surface_reach``);
- ``tropopause``, the switch of the sound speed at 11 km;
- ``protection``, the clip of the pitch demand to its protected bounds
  (``pitch_demand_bounds``), which the autopilot's 0.2 g binds on every capture.

Dropped guards (T4), which each task's step checks along every plan: the Mach
switches past ``M_crit`` (shock stall, drag rise), the section's lift and drag
caps, the incidence wrap, the angular acceleration clip, the scaler's clip and
the tanks. The altitude clip of the density and the Mach branch of the speed
schedule are the env's own functions, called off their edges.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from target_gym.oracles.controller import softplus_clip
from target_gym.plane.dynamics import (
    buffet_line,
    compressibility,
    compute_drag,
    downwash,
    engine_response_rate,
    pitch_demand_bounds,
    tail_section,
)

#: The smoothed sites, name order, as in ``data/oracle_widths``.
SITE_NAMES = ("buffet", "cl_stall", "separation", "surface_clip", "surface_rate")
SITE_NAMES += ("tropopause", "protection")
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
    protection: jnp.ndarray


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
    lift = lift * compressibility(mach, p.aspect_ratio, p.sweep_half_chord)
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
    """The wing-body and the horizontal tail (``longitudinal_airframe``)."""
    cz_w, cx_w = section(jnp.rad2deg(alpha), mach, p, w, d)
    cap = buffet_line(mach, p)
    v = jnp.where(w.buffet > 0.0, w.buffet, 1.0)
    capped = cz_w - v * jax.nn.softplus((cz_w - cap) / v)
    cz_w = jnp.where(w.buffet > 0.0, capped, jnp.minimum(cz_w, cap)) + d.buffet
    lift_w = compute_drag(S=p.wings_surface, C=cz_w, V=speed, rho=rho)
    drag_w = compute_drag(S=p.wings_surface, C=cx_w, V=speed, rho=rho)
    moment = (lift_w * jnp.cos(alpha) + drag_w * jnp.sin(alpha)) * p.moment_arm_wings

    swept = jnp.arctan2(q * p.moment_arm_stabilizer, speed)
    tail = (
        alpha
        - downwash(cz_w, p)
        + p.stabilizer_setting
        - p.elevator_effectiveness * stick
        + swept
    )
    cz_t, cx_t = section(jnp.rad2deg(tail), mach, tail_section(p), w, d)
    area = p.tail_efficiency * p.stabilizer_surface
    lift_t = compute_drag(S=area, C=cz_t, V=speed, rho=rho)
    drag_t = compute_drag(S=area, C=cx_t, V=speed, rho=rho)
    moment = moment - (lift_t * jnp.cos(tail) + drag_t * jnp.sin(tail)) * (
        p.moment_arm_stabilizer
    )
    return Airframe(lift_w + lift_t, drag_w + drag_t, moment, jnp.abs(cz_t))


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


def protected(demand, gamma_air, speed, z, mass, p, w, d):
    """``protected_pitch_demand``, its clip smoothed."""
    lo, hi = pitch_demand_bounds(gamma_air, speed, z, mass, p)
    return softplus_clip(demand, lo, hi, w.protection) + d.protection


def surface_step(requested, current, dt, response_rate, rate_limit, width, delta):
    """``compute_next_surface``, its rate limit smoothed."""
    move = (requested - current) * (1.0 - jnp.exp(-response_rate * dt))
    most = rate_limit * dt
    return current + softplus_clip(move, -most, most, width) + delta


def engine(lever, power, dt, p, z):
    """``compute_next_power`` with the lever's clip removed (T1), at height ``z``."""
    requested = p.idle_power + (1.0 - p.idle_power) * lever
    rate = engine_response_rate(z, p.power_response_rate)
    return requested + (power - requested) * jnp.exp(-rate * dt)
