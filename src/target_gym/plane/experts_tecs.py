"""Total energy control (TECS), the longitudinal autopilot PX4 and ArduPilot fly.

Lambregts' observation: throttle changes an aircraft's total energy, height
plus speed, while the elevator only trades one for the other. So throttle
closes the total specific energy error and pitch closes the balance between
potential and kinetic energy, instead of one loop per quantity fighting over
both. Written after ArduPilot's ``AP_TECS``, with its parameter names, as one
pure step over whichever array module is passed (NumPy on the host, jnp
under jit), so the 2D and 3D aircraft share it.

The aircraft's own numbers, as ArduPilot's tuning guide has a pilot measure
them in flight: ``climb_max``, the climb full throttle sustains at cruise
speed, and ``sink_min``, the sink at idle. ``thr_trim`` is the throttle that
holds level cruise and ``pitch_trim`` the pitch that does. ``AIRLINER`` holds
this aircraft's, which the 2D and 3D tasks share, and ``pitch_hold`` is the
attitude loop TECS's pitch demand goes to.
"""

from __future__ import annotations

from flax import struct


@struct.dataclass
class TECSParams:
    """``TECS_*`` parameters, in SI units and radians."""

    climb_max: float  # m/s, TECS_CLMB_MAX
    sink_min: float  # m/s, TECS_SINK_MIN
    sink_max: float  # m/s, TECS_SINK_MAX
    thr_trim: float  # throttle, TRIM_THROTTLE
    time_const: float = 5.0  # s, TECS_TIME_CONST
    thr_damp: float = 0.5  # TECS_THR_DAMP
    ptch_damp: float = 0.3  # TECS_PTCH_DAMP
    integ_gain: float = 0.1  # TECS_INTEG_GAIN
    spd_weight: float = 1.0  # TECS_SPDWEIGHT: 0 pitch on height only, 2 on speed only
    roll_comp: float = 10.0  # TECS_RLL2THR, m^2/s^3 of induced drag per unit
    thr_min: float = 0.0
    thr_max: float = 1.0
    pitch_trim: float = 0.032  # rad
    pitch_max: float = 0.26  # rad
    accel_max: float = 2.0  # m/s^2, the speed rate demand's limit
    k_vdot: float = 0.3  # smoothing of the speed rate, differenced from speed
    gravity: float = 9.81
    dt: float = 1.0


@struct.dataclass
class TECSState:
    """The two integrators, the slewed demands and the speed-rate estimate."""

    thr_integ: float = 0.0
    pitch_integ: float = 0.0
    h_ref: float = 0.0
    v_ref: float = 0.0
    v_prev: float = 0.0
    v_dot: float = 0.0
    started: float = 0.0


def tecs_step(p, state, h, h_dot, v, h_dem, v_dem, phi, xp):
    """One TECS step: ``(throttle, pitch demand, new state)``.

    ``h_dot`` is the measured climb rate and ``phi`` the bank, whose induced
    drag the throttle pays for ahead of the error it would cause.
    """
    g = p.gravity
    started = state.started > 0.0
    v_dot_now = xp.where(started, (v - state.v_prev) / p.dt, 0.0)
    v_dot = state.v_dot + p.k_vdot * (v_dot_now - state.v_dot)

    # The demands move toward their targets no faster than the aircraft can
    # climb, sink and accelerate, from wherever it starts, and the rate they
    # move at is the rate demanded (AP_TECS's height and speed demand slews).
    h_ref = xp.where(started, state.h_ref, h)
    v_ref = xp.where(started, state.v_ref, v)
    h_step = xp.clip(h_dem - h_ref, -p.sink_max * p.dt, p.climb_max * p.dt)
    v_step = xp.clip(v_dem - v_ref, -p.accel_max * p.dt, p.accel_max * p.dt)
    h_ref, v_ref = h_ref + h_step, v_ref + v_step
    h_dot_dem = h_step / p.dt
    v_dot_dem = v_step / p.dt

    spe_err = g * (h_ref - h)
    ske_err = 0.5 * (v_ref**2 - v**2)
    spe_dot_dem, ske_dot_dem = g * h_dot_dem, v * v_dot_dem
    spe_dot, ske_dot = g * h_dot, v * v_dot

    # Throttle on the total: feedforward of the demanded rate from the
    # throttle-to-energy-rate slope, plus proportional and integral terms.
    ste_err, ste_dot = spe_err + ske_err, spe_dot + ske_dot
    ste_dot_dem = spe_dot_dem + ske_dot_dem
    ste_dot_dem = ste_dot_dem + p.roll_comp * (
        1.0 / xp.maximum(xp.cos(phi) ** 2, 0.1) - 1.0
    )
    k_thr2ste = g * (p.climb_max + p.sink_min) / (p.thr_max - p.thr_min)
    k_ste2thr = 1.0 / (p.time_const * k_thr2ste)
    ff = p.thr_trim + ste_dot_dem / k_thr2ste
    unclipped = (
        (ste_err + (ste_dot_dem - ste_dot) * p.thr_damp) * k_ste2thr
        + ff
        + state.thr_integ
    )
    thr = xp.clip(unclipped, p.thr_min, p.thr_max)
    # Integrate only while the throttle is free to act on it.
    thr_integ = xp.where(
        unclipped == thr,
        state.thr_integ + ste_err * p.integ_gain * p.dt * k_ste2thr,
        state.thr_integ,
    )

    # Pitch on the balance, potential minus kinetic, scaled to an angle.
    w_spe, w_ske = 2.0 - p.spd_weight, p.spd_weight
    seb_err = spe_err * w_spe - ske_err * w_ske
    seb_dot_dem = spe_dot_dem * w_spe - ske_dot_dem * w_ske
    seb_dot = spe_dot * w_spe - ske_dot * w_ske
    gain_inv = xp.maximum(v, 1.0) * p.time_const * g
    unclipped = (
        seb_err
        + seb_dot_dem * p.time_const
        + (seb_dot_dem - seb_dot) * p.ptch_damp
        + state.pitch_integ
    ) / gain_inv + p.pitch_trim
    pitch = xp.clip(unclipped, -p.pitch_max, p.pitch_max)
    pitch_integ = xp.where(
        unclipped == pitch,
        state.pitch_integ + seb_err * p.integ_gain * p.dt,
        state.pitch_integ,
    )

    new_state = TECSState(
        thr_integ=thr_integ,
        pitch_integ=pitch_integ,
        h_ref=h_ref,
        v_ref=v_ref,
        v_prev=v,
        v_dot=v_dot,
        started=xp.ones_like(h),
    )
    return thr, pitch, new_state


def pitch_hold(p, integ, theta_cmd, theta, theta_dot, xp):
    """The pitch attitude loop under TECS: PI on the pitch error, damped on
    pitch rate, the integral frozen while the stick is saturated. ``p``
    carries ``Kp_theta``, ``Ki_theta``, ``Kd_theta`` and ``dt``. Returns
    ``(stick, new integral)``."""
    err = theta_cmd - theta
    unclipped = p.Kp_theta * err + p.Ki_theta * integ - p.Kd_theta * theta_dot
    stick = xp.clip(unclipped, -1.0, 1.0)
    return stick, xp.where(unclipped == stick, integ + err * p.dt, integ)


# The airliner's numbers, from a flight test of the model at 230 m/s with no
# turbulence, pitch holding the speed at a fixed throttle
# (``scripts/tune_aircraft_guidance.py``). Full throttle climbs 10.2, 10.1 and
# 7.9 m/s at 3500, 5500 and 7500 m, idle sinks 27, 25 and 23 m/s, and level
# flight trims at 0.15 to 0.36 of the throttle action, which spans [-1, 1].
# The climb limit is the highest altitude's, as ArduPilot's guide sets it
# below what full throttle gives everywhere; the integrators take up the trim.
# The time constant, damping and integrator gain are tuned on training seeds
# with the pitch loop, and ``spd_weight`` is set per task, as the mission's
# priority between height and speed.
AIRLINER = TECSParams(
    climb_max=8.0,
    sink_min=23.0,
    sink_max=10.0,
    thr_trim=0.2,
    thr_min=-1.0,
    thr_max=1.0,
    pitch_trim=0.0,
    time_const=2.0,
    ptch_damp=1.0,
    integ_gain=0.1,
    spd_weight=0.0,
)
