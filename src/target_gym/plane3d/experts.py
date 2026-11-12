"""The expert (the PID slot) for
the 3D aircraft tasks (plane3d_heading, plane3d_circle, plane3d_racetrack, plane3d_figure8).

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.pid import (
    Controller,
    _wrap_angle,
    register_learnable_gains,
)
from target_gym.plane.dynamics import (
    BANK_DEMAND_MAX,
    PITCH_DEMAND_MAX,
    scheduled_airspeed,
)
from target_gym.plane.experts_tecs import (
    AIRLINER,
    TECSParams,
    TECSState,
    tecs_step,
)

# ---------------------------------------------------------------------------
# The experts: an autopilot's loops under the field's guidance laws
# ---------------------------------------------------------------------------
#
# Each task's expert is what a fixed-wing autopilot flies (PX4, ArduPilot):
# TECS for height and speed (``plane.experts_tecs``) and a bank demand from a
# lateral guidance law, both held by the plant's fixed inner loop. The heading
# task holds a course; the circle, racetrack and figure-8 share L1 guidance (Park, Deyst and
# How, 2004), the law ArduPilot flies paths and loiters with, plus the path's
# curvature as a coordinated-turn feedforward. Each is one pure
# ``step(params, state, obs, xp)`` for ``experts.pid.Controller``: NumPy on
# the host, jnp under jit.
#
# Signs: a positive bank turns toward increasing heading. ``cross`` is the offset from the path, positive on
# that same side, and ``curvature`` is positive where the path turns that way.

_BANK_45 = float(np.deg2rad(45.0))
_SIN_45 = float(np.sin(np.deg2rad(45.0)))
_HALF_PI = float(np.pi / 2.0)


@struct.dataclass
class Plane3DGuidanceParams:
    """Lateral guidance and TECS, sending the plant's inner loop its demands.

    ``tau_hdg`` is the course hold's time constant (heading task). ``period``
    and ``damping`` are L1's, as ArduPilot names them ``NAVL1_PERIOD`` and
    ``NAVL1_DAMPING``. ``max_bank_rad`` is ArduPilot's default roll limit and
    the edge of the model's validity (PHYSICS.md). A curved path is flown at
    the speed whose steady bank in its tightest turn is ``turn_bank_margin``
    of the limit, leaving the rest for L1's corrections; otherwise at
    ``cruise_cas`` and ``cruise_mach``, the 2D tasks' schedule. The figure-8, the
    only path whose curvature the observation does not give, reads it from
    the pattern it was given: ``figure8_altitude_amplitude``, the scored
    task's, and ``curvature_lead`` seconds ahead, the bank loop's lag. The
    pitch loop holds TECS's pitch demand, with the extra pitch a banked turn
    needs and the angle of attack kept inside ``alpha_max``. ``v_min_eas`` is the slowest equivalent
    airspeed the autopilot flies (ArduPilot's ``ARSPD_FBW_MIN``): 1.3 times
    the model's clean 1 g stall, 80 m/s at 73.5 t, CL_max 1.5 and 122.6 m^2.
    """

    max_bank_rad: float = _BANK_45
    turn_bank_margin: float = 0.85
    tau_hdg: float = 10.0
    period: float = 35.0
    damping: float = 0.75
    figure8_altitude_amplitude: float = 200.0
    curvature_lead: float = 1.0
    alpha_max: float = 0.19
    k_bank_comp: float = 0.15
    cruise_cas: float = 290.0 * 1852.0 / 3600.0
    cruise_mach: float = 0.78
    target_speed: float = 230.0  # the formation lead's, until PR 7b
    v_min_eas: float = 104.0
    gravity: float = 9.81
    dt: float = 1.0
    tecs: TECSParams = AIRLINER


@struct.dataclass
class Plane3DGuidanceState:
    """TECS's state, and on the figure-8 the last path point's altitude, the way the path turns, and the pattern's
    orientation averaged over ``samples`` observations."""

    tecs: TECSState = struct.field(default_factory=TECSState)
    path_z: float = 0.0
    turn_sign: float = 1.0
    orientation: float = 0.0
    samples: float = 0.0


def _ground_speed(obs, xp):
    return xp.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-6)


def _cruise_speed(params, obs, xp):
    """The schedule's true airspeed at the observed height."""
    return scheduled_airspeed(obs[..., 2], params.cruise_cas, params.cruise_mach, xp)


def _turn_speed(params, radius, obs, xp):
    """Cruise speed, or slower where the tightest turn of radius ``radius``
    would otherwise need more bank than the limit leaves in hand."""
    usable = params.turn_bank_margin * params.max_bank_rad
    limit = xp.sqrt(params.gravity * xp.maximum(radius, 1.0) * xp.tan(usable))
    return xp.minimum(_cruise_speed(params, obs, xp), limit)


def _v_min(params, z, xp):
    """``v_min_eas`` as a true airspeed at height ``z``, by the standard
    atmosphere's density ratio below 11 km."""
    sigma = xp.maximum(1.0 - 2.2558e-5 * z, 0.2) ** 4.2559
    return params.v_min_eas / xp.sqrt(sigma)


def _engaged(tecs, state, power, pitch, v, xp):
    """``state`` with, before TECS's first step, the integrators that make its
    outputs the observed ``power`` and ``pitch``, its errors being zero there.
    The throttle action spans [-1, 1] for power in [0, 1]. The pitch integral
    is in the units of master's pitch loop (``experts_tecs.tecs_step``), an
    energy-balance rate: pitch above trim times TAS g."""
    throttle = 2.0 * power - 1.0
    fresh = state.started <= 0.0
    gain_inv = xp.maximum(v, 1.0) * tecs.gravity
    return state.replace(
        thr_integ=xp.where(fresh, throttle - tecs.thr_trim, state.thr_integ),
        pitch_integ=xp.where(
            fresh, (pitch - tecs.pitch_trim) * gain_inv, state.pitch_integ
        ),
    )


def l1_lateral_accel(params, cross, tangent, curvature, obs, xp):
    """L1 guidance's lateral acceleration (Park, Deyst and How, 2004), as
    ArduPilot computes it, plus ``v^2 * curvature`` so a curved path needs no
    standing offset to turn."""
    v = _ground_speed(obs, xp)
    l1 = params.damping * params.period * v / np.pi
    nu1 = -xp.arcsin(xp.clip(cross / l1, -_SIN_45, _SIN_45))
    nu2 = _wrap_angle(tangent - xp.arctan2(obs[..., 1], obs[..., 0]), xp)
    nu = xp.clip(nu1 + nu2, -_HALF_PI, _HALF_PI)
    k = 4.0 * params.damping**2
    return k * v**2 / l1 * xp.sin(nu) + v**2 * curvature


def _fly(
    params,
    state,
    obs,
    xp,
    h_dem,
    accel,
    target_speed,
    airspeed=None,
    power=None,
    tecs_fn=None,
    **carry,
):
    """The autopilot under a guidance law: TECS on height and speed, and the
    bank that makes ``accel`` in a coordinated turn. Returns the demands the
    plant's inner loop holds, ``[throttle, pitch, bank]`` in [-1, 1].

    With ``airspeed`` and ``power`` (the amended tasks' air data and engine
    reading), TECS reads the airspeed and engages on the trim found, and the
    speed floor carries the load factor of the bank demanded, not of the full
    limit. Without it, the earlier expert: ground speed, and the floor at the
    full bank's load factor (the Extra tasks, patrol, formation).

    ``tecs_fn`` is the TECS step, ``experts_tecs.tecs_step`` unless given: the
    patrol family's scripted lead, part of its plant, flies the 4.0.9 port
    (``tecs_step_409``)."""
    z, z_dot = obs[..., 2], obs[..., 3]
    phi, gamma = obs[..., 6], obs[..., 8]
    ground = xp.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2 + z_dot**2)
    v = ground if airspeed is None else airspeed
    # Stall prevention, as ArduPilot's STALL_PREVENTION: "roll limits based on
    # aerodynamic load factor in turns" [P409], never more bank than the speed
    # supports, n = (v / v_min)^2; and never slower than the speed at which the
    # bank flown is safe.
    v_min = _v_min(params, z, xp)
    max_bank = xp.arccos(xp.clip((v_min / v) ** 2, xp.cos(params.max_bank_rad), 1.0))
    demanded = xp.clip(xp.arctan(accel / params.gravity), -max_bank, max_bank)
    floor_bank = params.max_bank_rad if airspeed is None else xp.abs(demanded)
    target_speed = xp.maximum(target_speed, v_min / xp.sqrt(xp.cos(floor_bank)))
    tecs_state = state.tecs
    if airspeed is not None:
        # The amended tasks start established, the autopilot already flying
        # (rule 17): on its first step TECS's integrators take up the trim it
        # finds, the observed throttle and pitch, so it engages bumpless.
        tecs_state = _engaged(params.tecs, tecs_state, power, obs[..., 4], v, xp)
    step = tecs_step if tecs_fn is None else tecs_fn
    power, pitch, tecs = step(
        params.tecs, tecs_state, z, z_dot, v, h_dem, target_speed, phi, xp
    )
    load_factor = 1.0 / xp.maximum(xp.cos(phi), 0.35)
    theta_cmd = pitch + params.k_bank_comp * (load_factor - 1.0)
    theta_cmd = xp.clip(theta_cmd, gamma - params.alpha_max, gamma + params.alpha_max)
    action = [power, theta_cmd / PITCH_DEMAND_MAX, demanded / BANK_DEMAND_MAX]
    return xp.stack(action, axis=-1), state.replace(tecs=tecs, **carry)


def plane3d_heading_step(params, state, obs, xp=jnp, tecs_fn=None):
    """Course hold: the turn rate that closes the heading error in ``tau_hdg``.
    ``tecs_fn`` as :func:`_fly`'s."""
    v = _ground_speed(obs, xp)
    turn_rate = _wrap_angle(obs[..., 11] - obs[..., 9], xp) / params.tau_hdg
    target = _cruise_speed(params, obs, xp)
    return _fly(
        params, state, obs, xp, obs[..., 10], v * turn_rate, target, tecs_fn=tecs_fn
    )


def plane3d_heading_select_step(params, state, obs, xp=jnp):
    """The heading task's heading hold (Amendment 3): as ``plane3d_heading_step``,
    which the patrol's lead also flies, with TECS on the observed airspeed and
    the turn on the IRS heading, a coordinated turn's rate being g tan(bank)
    over the airspeed."""
    v = obs[..., 16]
    turn_rate = _wrap_angle(obs[..., 11] - obs[..., 18], xp) / params.tau_hdg
    target = _cruise_speed(params, obs, xp)
    return _fly(
        params,
        state,
        obs,
        xp,
        obs[..., 10],
        v * turn_rate,
        target,
        airspeed=obs[..., 16],
        power=obs[..., 12],
    )


def plane3d_circle_step(params, state, obs, xp=jnp):
    """L1 on the circle, flown with heading increasing."""
    rx, ry = obs[..., 11], obs[..., 12]
    radius = xp.maximum(obs[..., 13], 1.0)
    cross = radius - xp.sqrt(rx**2 + ry**2 + 1e-6)
    tangent = xp.arctan2(ry, rx) + _HALF_PI
    accel = l1_lateral_accel(params, cross, tangent, 1.0 / radius, obs, xp)
    target = _turn_speed(params, radius, obs, xp)
    return _fly(params, state, obs, xp, obs[..., 10], accel, target)


def plane3d_racetrack_step(params, state, obs, xp=jnp):
    """L1 on the holding pattern, whose guidance the observation gives."""
    cross, tangent, curvature = obs[..., 18], obs[..., 19], obs[..., 20]
    accel = l1_lateral_accel(params, cross, tangent, curvature, obs, xp)
    # The assigned holding speed the observation gives (Amendment 2).
    target, airspeed, power = obs[..., 22], obs[..., 23], obs[..., 15]
    return _fly(
        params,
        state,
        obs,
        xp,
        obs[..., 10],
        accel,
        target,
        airspeed=airspeed,
        power=power,
    )


def plane3d_figure8_step(params, state, obs, xp=jnp):
    """L1 on the figure-8, whose curvature is not observed but follows from
    where the aircraft is on it.

    The pattern is Bernoulli's lemniscate of half-width ``a``, its altitude
    z_mean + dz sin(tau) along its parameter tau (PHYSICS.md), so the nearest
    point's altitude gives sin(tau). The curvature there is
    3 cos(tau) / (a sqrt(1 + sin^2 tau)), and the path turns the way it
    climbs. The tangent is 3 arctan(sin tau) from its heading at the lobe
    tip, which is constant over the flight; the observed tangent, read off
    the env's sampled curve, gives it to within the sampling, so the expert
    averages it."""
    a = xp.maximum(obs[..., 11], 1.0)
    dx, dy, dz = obs[..., 12], obs[..., 13], obs[..., 14]
    path_z = obs[..., 2] + dz
    sin_tau = xp.clip(
        (path_z - obs[..., 10]) / params.figure8_altitude_amplitude, -1.0, 1.0
    )

    # The first observation sets the orientation; later ones are averaged in
    # modulo pi, since the observed tangent flips to follow the velocity.
    shape = _HALF_PI + 3.0 * xp.arctan(sin_tau)
    error = _wrap_angle(obs[..., 15] - shape - state.orientation, xp)
    started = state.samples > 0.0
    error = xp.where(started, 0.5 * _wrap_angle(2.0 * error, xp), error)
    orientation = state.orientation + error / (state.samples + 1.0)
    tangent = orientation + shape
    cross = dx * xp.sin(tangent) - dy * xp.cos(tangent)

    # Read ``curvature_lead`` seconds ahead along tau, so the sign flips at a
    # crossover before the aircraft reaches it.
    climb = path_z - state.path_z
    turn_sign = xp.where(
        started & (xp.abs(climb) > 1e-3), xp.sign(climb), state.turn_sign
    )
    tau_rate = _ground_speed(obs, xp) * xp.sqrt(1.0 + sin_tau**2) / a
    tau = xp.arcsin(sin_tau) + turn_sign * tau_rate * params.curvature_lead
    curvature = turn_sign * 3.0 * xp.cos(tau) / (a * xp.sqrt(1.0 + xp.sin(tau) ** 2))

    accel = l1_lateral_accel(params, cross, tangent, curvature, obs, xp)
    target = _turn_speed(params, a / 3.0, obs, xp)
    return _fly(
        params,
        state,
        obs,
        xp,
        path_z,
        accel,
        target,
        path_z=path_z,
        turn_sign=turn_sign,
        orientation=orientation,
        samples=state.samples + 1.0,
    )


register_learnable_gains(plane3d_heading_step, ("tau_hdg",))
register_learnable_gains(plane3d_heading_select_step, ("tau_hdg",))
for _step in (plane3d_circle_step, plane3d_racetrack_step, plane3d_figure8_step):
    register_learnable_gains(_step, ("period", "damping"))


def make_plane3d_heading_pid_controller() -> Controller:
    """The heading task's expert: course hold over the autopilot's loops."""
    return Controller(_AMENDED, Plane3DGuidanceState(), plane3d_heading_select_step)


def make_plane3d_circle_pid_controller() -> Controller:
    """The circle's expert: L1 guidance over the autopilot's loops."""
    return Controller(_PATH, Plane3DGuidanceState(), plane3d_circle_step)


def make_plane3d_racetrack_pid_controller() -> Controller:
    """The holding pattern's expert: L1 guidance over the autopilot's loops."""
    return Controller(_AMENDED, Plane3DGuidanceState(), plane3d_racetrack_step)


def make_plane3d_figure8_pid_controller() -> Controller:
    """The figure-8's expert: L1 guidance over the autopilot's loops."""
    return Controller(_FIGURE8, Plane3DGuidanceState(), plane3d_figure8_step)


# Tuned inner loop first, as ArduPilot's tuning guide orders it, on training
# seeds 100-109 and checked on held-out seeds 200-209, never on the scored
# seeds 0-9 (scripts/tune_aircraft_guidance.py).
_HEADING = Plane3DGuidanceParams()
# The circle flies the amended plant since Amendment 4: its protected 30 deg
# bank and the amended speed floor, as the hold's.
_PATH = Plane3DGuidanceParams(max_bank_rad=float(np.deg2rad(30.0)), v_min_eas=96.0)
# The amended tasks (heading, racetrack; Amendment 3): the plant's protected
# 30 deg bank (plane3d/PHYSICS.md), and the slowest speed 1.2 times the model's
# clean 1 g stall, 80 m/s EAS: "20% higher than level flight stall speed"
# (ARSPD_FBW_MIN [P409]).
_AMENDED = Plane3DGuidanceParams(max_bank_rad=float(np.deg2rad(30.0)), v_min_eas=96.0)
# The figure-8 flies slower and higher than the other tasks, banked most of
# the way, so it gets a faster, better damped TECS, chosen to hold with any one
# gain 25% off. Its lobe tips are flown with a quarter of the bank in hand. PR 9
# replaces these with the shared tune.
_FIGURE8 = Plane3DGuidanceParams(
    max_bank_rad=float(np.deg2rad(30.0)),
    v_min_eas=96.0,
    turn_bank_margin=0.75,
    tecs=AIRLINER.replace(time_const=1.5, ptch_damp=2.0),
)
