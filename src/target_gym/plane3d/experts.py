"""Reference controller (the MPC slot) for
the 3D aircraft tasks (plane3d_heading, plane3d_circle, plane3d_racetrack, plane3d_figure8).

Each task keeps its oracle in its own package, so changing one re-records
only the tasks of that package. The shared machinery (GradientMPC,
CasadiMPC, SamplingMPC and the v2 objective helpers) stays in
``target_gym.experts.mpc``, which is in every task's baseline fingerprint.
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.mpc import (
    GradientMPC,
    _done_value,
)
from target_gym.experts.pid import (
    _PLANE3D_HEADING_LEARNABLE,
    Controller,
    _g3d,
    _load_gains,
    _wrap_angle,
    register_learnable_gains,
)

# Stand-in for "the task's own default" in the factory's keyword arguments,
# so an explicit ``lr_end=None`` (a fixed step) stays expressible.
_TASK_DEFAULT = object()

# Fraction of the bank limit a steady turn may use, leaving the rest as control
# authority for the radial correction that sits on top of the feedforward.
_TURN_BANK_MARGIN = 0.85

#: Solver settings per task, keyed by the environment class. They live here,
#: not in the registry, because this file is in the four tasks' baseline
#: fingerprint and the registry is not: a setting passed from the registry
#: would change the oracle without staling its recorded scores.
#:
#: The step decays geometrically from ``lr`` (0.05) to ``lr_end`` over each
#: solve. A normalised descent cannot place an action closer than about its
#: step to the optimum. At a fixed 0.05 for 50 iterations the planner held
#: the circle 4.1 to 4.5 m off its altitude and 14.3 to 15.5 m off its path
#: (the earlier hold rows). On the circle's protocol seeds 0 and 1, under
#: the -v2 floors, the mean errors after the burn-in: the decay at 50
#: iterations took them
#: to 1.6 to 1.7 m and 4.6 to 7.6 m, and at 200 iterations to 0.86 to 1.0 m
#: and 0.42 to 2.4 m. The figure-8 needs the 200: at 50 its protocol seed 0
#: cost 3.3x the fixed step's (40 m excursions at the lobe tips, where the
#: fixed step held a steady 15 m), at 200 it held 3.3 m. So does the
#: racetrack: under the -v3 floors its protocol cost on seeds 0 to 2 is 4.58
#: at 50 iterations (2.23, 7.54, 3.97) and 1.07 at 200 (0.960, 1.29, 0.962),
#: zero trips either way, for 3.5x the compute per seed (11 to 14 min against
#: 3.5 in the paired runs). The holds under the -v3 floors, the instrument
#: resolutions, are in PHYSICS.md.
_TASK_SETTINGS = {
    "Plane3DHeading": dict(n_iter=200, lr_end=0.002),
    "Plane3DCircle": dict(n_iter=200, lr_end=0.002),
    "Plane3DRacetrack": dict(n_iter=200, lr_end=0.002),
    "Plane3DFigureEight": dict(n_iter=200, lr_end=0.002),
}


def task_settings(env) -> dict:
    """The solver settings ``make_plane3d_mpc`` uses for ``env``'s task.

    An environment class not in ``_TASK_SETTINGS`` gets the settings every
    task used before the audit: 50 iterations at a fixed step.
    """
    return dict(_TASK_SETTINGS.get(type(env).__name__, dict(n_iter=50, lr_end=None)))


def make_plane3d_mpc(
    env,
    params,
    horizon: int = 30,
    n_iter=_TASK_DEFAULT,
    lr: float = 0.05,
    lr_end=_TASK_DEFAULT,
):
    """Gradient MPC for the 3D plane tasks, optimising [power, stick, aileron] in [-1, 1].

    Same rationale as the 2D Plane MPC: the 3D dynamics extend the 2D
    aerodynamic model with roll, so it remains differentiable JAX but not
    expressible in CasADi. Works for all four task variants (Heading,
    Circle, Racetrack, FigureEight) since they share step_env.

    The objective is the environment's own reward. Under the version-2
    reward a trip is charged inside the rollout itself (the step that leaves
    the envelope pays the trip cost, ``base.failure_kernel``) and
    ``step_env`` never raises ``terminated``, so ``done_value``, the failure
    charge in floor units, is never applied (version 1 keeps 0). The plan
    therefore does not read ``rho_floor_tracking``; the floors and
    ``failure_cost`` weight it through the reward.

    ``n_iter`` and ``lr_end`` default to the task's entry in
    ``_TASK_SETTINGS``; pass them to override (``lr_end=None`` is the fixed
    step the planner used before the audit).
    """
    settings = task_settings(env)
    if n_iter is _TASK_DEFAULT:
        n_iter = settings["n_iter"]
    if lr_end is _TASK_DEFAULT:
        lr_end = settings["lr_end"]
    return GradientMPC(
        env,
        params,
        action_dim=3,
        action_lb=-1.0,
        action_ub=1.0,
        done_value=_done_value(params),
        horizon=horizon,
        n_iter=n_iter,
        lr=lr,
        lr_end=lr_end,
    )


# ---------------------------------------------------------------------------
# Scored PIDs as pure functions (one step, NumPy on the host, jnp under jit)
# ---------------------------------------------------------------------------
#
# These are the scored plane3d PIDs, the bank-angle cascade (heading, circle,
# racetrack) and the figure-8 controller, each written once as
# ``step(params, state, obs, xp)`` for ``experts.pid.Controller``. Under
# ``xp=numpy`` they reproduce the scored baselines bit for bit, so every
# expression keeps the operand order those were recorded with, and every
# constant and initial state is a Python float: NumPy treats those as weakly
# typed, so the arithmetic stays in the observation's float32 as the recorded
# runs had it. The one place those runs left float32 is kept too: ``np.tan`` of
# the Python-float bank margin is a float64 scalar, which put the circle and
# racetrack airspeed loops in float64; ``xp.tan`` of the same Python float does
# the same on the host and stays float32 under jnp.

_BANK_25 = float(np.deg2rad(25.0))
_BANK_30 = float(np.deg2rad(30.0))
_INTERCEPT_45 = float(np.deg2rad(45.0))


@struct.dataclass
class Plane3DVerticalParams:
    """Altitude -> vertical speed -> pitch -> elevator (``_Plane3DVerticalChannel``)."""

    Kp_alt: float = 0.03
    vs_max: float = 12.0
    Kp_vs: float = 0.010
    Ki_vs: float = 4.0e-4
    theta_trim: float = 0.032
    theta_max: float = 0.26
    Kp_theta: float = 4.0
    Kd_theta: float = 3.0
    alpha_max: float = 0.19
    k_bank_comp: float = 0.15
    dt: float = 1.0


@struct.dataclass
class Plane3DAirspeedParams:
    """Throttle loop holding a target airspeed (``_AirspeedChannel``)."""

    target_speed: float = 230.0
    cruise_power: float = 0.2
    Kp: float = 0.025
    Ki: float = 3.0e-4
    dt: float = 1.0


@struct.dataclass
class Plane3DCascadeState:
    """What the cascaded autopilot carries: three integrators, one previous error."""

    vs_integral: float = 0.0
    speed_integral: float = 0.0
    lat_integral: float = 0.0
    lat_prev: float = 0.0


@struct.dataclass
class Plane3DHeadingCascadeParams:
    """Cascaded heading autopilot (``_HeadingLateral``)."""

    Kp_hdg: float
    Ki_hdg: float
    Kd_hdg: float
    Kp_bank: float
    Kd_bank: float = 0.5
    max_bank_rad: float = _BANK_25
    dt: float = 1.0
    vertical: Plane3DVerticalParams = struct.field(
        default_factory=Plane3DVerticalParams
    )
    airspeed: Plane3DAirspeedParams = struct.field(
        default_factory=Plane3DAirspeedParams
    )


@struct.dataclass
class Plane3DCircleCascadeParams:
    """Cascaded circle autopilot (``_CircleLateral``)."""

    Kp_rad: float
    Ki_rad: float
    Kd_rad: float
    Kp_bank: float
    Kd_bank: float = 0.5
    max_bank_rad: float = _BANK_30
    gravity: float = 9.81
    dt: float = 1.0
    vertical: Plane3DVerticalParams = struct.field(
        default_factory=Plane3DVerticalParams
    )
    airspeed: Plane3DAirspeedParams = struct.field(
        default_factory=Plane3DAirspeedParams
    )


@struct.dataclass
class Plane3DRacetrackCascadeParams:
    """Cascaded holding-pattern autopilot (``_RacetrackLateral``)."""

    Kp_track: float
    Ki_track: float
    Kd_track: float
    Kp_bank: float
    Kd_bank: float = -1.5
    max_bank_rad: float = _BANK_30
    max_intercept_rad: float = _INTERCEPT_45
    gravity: float = 9.81
    dt: float = 1.0
    hdg_gain: float = 1.0
    vertical: Plane3DVerticalParams = struct.field(
        default_factory=Plane3DVerticalParams
    )
    airspeed: Plane3DAirspeedParams = struct.field(
        default_factory=Plane3DAirspeedParams
    )


def _vertical_channel(p, vs_integral, obs, xp):
    """One step of the altitude cascade. Returns ``(stick, new_vs_integral)``."""
    z, z_dot = obs[..., 2], obs[..., 3]
    theta, theta_dot = obs[..., 4], obs[..., 5]
    phi = obs[..., 6]
    gamma = obs[..., 8]
    alt_error = obs[..., 10] - z

    vs_cmd = xp.clip(p.Kp_alt * alt_error, -p.vs_max, p.vs_max)
    vs_err = vs_cmd - z_dot
    vs_integral = vs_integral + vs_err * p.dt

    # Extra pitch needed to hold altitude while banked.
    load_factor = 1.0 / xp.maximum(xp.cos(phi), 0.35)
    bank_comp = p.k_bank_comp * (load_factor - 1.0)

    theta_cmd = p.theta_trim + bank_comp + p.Kp_vs * vs_err + p.Ki_vs * vs_integral
    theta_cmd = xp.clip(theta_cmd, -p.theta_max, p.theta_max)
    theta_cmd = xp.clip(theta_cmd, gamma - p.alpha_max, gamma + p.alpha_max)

    limited = xp.abs(theta_cmd - p.theta_trim) >= p.theta_max - 1e-6
    vs_integral = xp.where(limited, vs_integral - vs_err * p.dt, vs_integral)

    stick = xp.clip(
        p.Kp_theta * (theta_cmd - theta) - p.Kd_theta * theta_dot, -1.0, 1.0
    )
    return stick, vs_integral


def _airspeed_channel(p, integral, speed, target, xp):
    """One step of the throttle loop. Returns ``(power, new_integral)``."""
    err = target - speed
    integral = integral + err * p.dt
    power = p.cruise_power + p.Kp * err + p.Ki * integral
    clipped = xp.clip(power, -1.0, 1.0)
    integral = xp.where(power != clipped, integral - err * p.dt, integral)
    return clipped, integral


def _turn_speed_limit(params, obs, xp):
    """Fastest speed the commanded radius admits, with bank in hand."""
    radius = xp.maximum(obs[..., 13], 1.0)
    usable = _TURN_BANK_MARGIN * params.max_bank_rad
    return xp.sqrt(params.gravity * radius * xp.tan(usable))


def _bank_to_aileron(params, desired_bank, obs, xp):
    phi, phi_dot = obs[..., 6], obs[..., 7]
    return xp.clip(
        params.Kp_bank * (phi - desired_bank) + params.Kd_bank * phi_dot, -1.0, 1.0
    )


def _cascade(params, state, obs, xp, target_speed, lateral):
    """The shared cascade: vertical, airspeed, then the task's lateral law.

    ``lateral(params, state, obs, xp) -> (aileron, lat_integral, lat_prev)``.
    """
    stick, vs_integral = _vertical_channel(params.vertical, state.vs_integral, obs, xp)
    speed = xp.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-9)
    power, speed_integral = _airspeed_channel(
        params.airspeed, state.speed_integral, speed, target_speed, xp
    )
    aileron, lat_integral, lat_prev = lateral(params, state, obs, xp)
    new_state = Plane3DCascadeState(
        vs_integral=vs_integral,
        speed_integral=speed_integral,
        lat_integral=lat_integral,
        lat_prev=lat_prev,
    )
    return xp.stack([power, stick, aileron], axis=-1), new_state


def _heading_lateral(params, state, obs, xp):
    err = _wrap_angle(obs[..., 11] - obs[..., 9], xp)
    integral = state.lat_integral + err * params.dt
    deriv = (err - state.lat_prev) / params.dt
    desired_bank = xp.clip(
        params.Kp_hdg * err + params.Ki_hdg * integral + params.Kd_hdg * deriv,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    aileron = _bank_to_aileron(params, desired_bank, obs, xp)
    integral = xp.where(xp.abs(aileron) >= 1.0, integral - err * params.dt, integral)
    return aileron, integral, err


def _circle_lateral(params, state, obs, xp):
    speed_sq = obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-6
    radius = xp.maximum(obs[..., 13], 1.0)
    # Bank that sustains the turn with no radial error: tan(phi) = v^2/(g r).
    ideal_bank = xp.arctan2(speed_sq, params.gravity * radius)
    dist = xp.sqrt(obs[..., 11] ** 2 + obs[..., 12] ** 2)
    err = dist - obs[..., 13]
    integral = state.lat_integral + err * params.dt
    deriv = (err - state.lat_prev) / params.dt
    desired_bank = xp.clip(
        ideal_bank
        + params.Kp_rad * err
        + params.Ki_rad * integral
        + params.Kd_rad * deriv,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    aileron = _bank_to_aileron(params, desired_bank, obs, xp)
    integral = xp.where(xp.abs(aileron) >= 1.0, integral - err * params.dt, integral)
    return aileron, integral, err


def _racetrack_lateral(params, state, obs, xp):
    cross, tangent, curvature = obs[..., 18], obs[..., 19], obs[..., 20]
    integral = state.lat_integral + cross * params.dt
    deriv = (cross - state.lat_prev) / params.dt
    intercept = xp.clip(
        params.Kp_track * cross + params.Ki_track * integral + params.Kd_track * deriv,
        -params.max_intercept_rad,
        params.max_intercept_rad,
    )
    desired_heading = tangent - intercept
    err = _wrap_angle(desired_heading - obs[..., 9], xp)
    # Coordinated-turn feedforward, tan(phi) = v^2 * kappa / g.
    speed_sq = obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-6
    ideal_bank = xp.arctan(speed_sq * curvature / params.gravity)
    desired_bank = xp.clip(
        ideal_bank + params.hdg_gain * err,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    aileron = _bank_to_aileron(params, desired_bank, obs, xp)
    integral = xp.where(xp.abs(aileron) >= 1.0, integral - cross * params.dt, integral)
    return aileron, integral, cross


def plane3d_heading_cascade_step(params, state, obs, xp=jnp):
    """Cascaded heading autopilot: ``(action [power, stick, aileron], state)``."""
    target = params.airspeed.target_speed
    return _cascade(params, state, obs, xp, target, _heading_lateral)


def plane3d_circle_cascade_step(params, state, obs, xp=jnp):
    """Cascaded circle autopilot, trading airspeed for the commanded radius."""
    target = _turn_speed_limit(params, obs, xp)
    return _cascade(params, state, obs, xp, target, _circle_lateral)


def plane3d_racetrack_cascade_step(params, state, obs, xp=jnp):
    """Cascaded holding-pattern autopilot, trading airspeed for the turn radius."""
    target = _turn_speed_limit(params, obs, xp)
    return _cascade(params, state, obs, xp, target, _racetrack_lateral)


@struct.dataclass
class Plane3DFigure8Params:
    """Figure-8 PID: fixed cruise throttle."""

    Kp_alt: float
    Ki_alt: float
    Kd_alt: float
    Kp_hdg: float
    Ki_hdg: float
    Kd_hdg: float
    Kp_bank: float
    power: float = 0.6
    max_bank_rad: float = _BANK_25
    dt: float = 1.0


@struct.dataclass
class Plane3DFigure8State:
    alt_integral: float = 0.0
    alt_prev: float = 0.0
    hdg_integral: float = 0.0
    hdg_prev: float = 0.0


def plane3d_figure8_step(params, state, obs, xp=jnp):
    """Figure-8 PID: altitude PID on ``nearest_dz``, blended-heading PID on bank."""
    psi = obs[..., 9]
    phi = obs[..., 6]
    target_radius = obs[..., 11]
    nearest_dx = obs[..., 12]
    nearest_dy = obs[..., 13]
    nearest_dz = obs[..., 14]
    tangent_heading = obs[..., 15]

    # Altitude: PID on nearest_dz (curve_z - aircraft_z).
    alt_err = nearest_dz
    alt_integral = state.alt_integral + alt_err * params.dt
    alt_d = (alt_err - state.alt_prev) / params.dt
    stick = xp.clip(
        params.Kp_alt * alt_err + params.Ki_alt * alt_integral + params.Kd_alt * alt_d,
        -1.0,
        1.0,
    )
    alt_integral = xp.where(
        xp.abs(stick) >= 1.0, alt_integral - alt_err * params.dt, alt_integral
    )

    # Heading: blend tangent (on-curve) with correction (off-curve).
    lateral_dist = xp.sqrt(nearest_dx**2 + nearest_dy**2 + 1e-6)
    blend = xp.clip(lateral_dist / (0.05 * xp.maximum(target_radius, 1.0)), 0.0, 1.0)
    correction_heading = xp.arctan2(nearest_dy, nearest_dx)
    bx = blend * xp.cos(correction_heading) + (1.0 - blend) * xp.cos(tangent_heading)
    by = blend * xp.sin(correction_heading) + (1.0 - blend) * xp.sin(tangent_heading)
    desired_heading = xp.arctan2(by, bx)

    hdg_err = _wrap_angle(desired_heading - psi, xp)
    hdg_integral = state.hdg_integral + hdg_err * params.dt
    hdg_d = (hdg_err - state.hdg_prev) / params.dt
    desired_bank = xp.clip(
        params.Kp_hdg * hdg_err + params.Ki_hdg * hdg_integral + params.Kd_hdg * hdg_d,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    bank_err = phi - desired_bank
    aileron = xp.clip(params.Kp_bank * bank_err, -1.0, 1.0)
    hdg_integral = xp.where(
        xp.abs(aileron) >= 1.0, hdg_integral - hdg_err * params.dt, hdg_integral
    )
    # The class broadcast its cruise throttle with jnp, which made it float32;
    # ``full_like`` gives the same float32 on the host and traces under jnp.
    power = xp.full_like(stick, params.power)
    new_state = Plane3DFigure8State(
        alt_integral=alt_integral,
        alt_prev=alt_err,
        hdg_integral=hdg_integral,
        hdg_prev=hdg_err,
    )
    return xp.stack([power, stick, aileron], axis=-1), new_state


# The figure-8 carries the same seven gains, under the same names, as the
# functional heading PID they were registered for.
register_learnable_gains(plane3d_figure8_step, _PLANE3D_HEADING_LEARNABLE)


def make_plane3d_heading_pid_controller() -> Controller:
    """``make_plane3d_heading_cascaded_pid`` as a :class:`Controller`."""
    g = _load_gains().get("plane3d_heading", {})
    hdg = g.get("hdg", {})
    params = Plane3DHeadingCascadeParams(
        Kp_hdg=float(hdg.get("Kp", 2.33)),
        Ki_hdg=float(hdg.get("Ki", 0.0)),
        Kd_hdg=float(hdg.get("Kd", 3.77)),
        Kp_bank=float(g.get("bank", {}).get("Kp", -3.24)),
    )
    return Controller(params, Plane3DCascadeState(), plane3d_heading_cascade_step)


def make_plane3d_circle_pid_controller() -> Controller:
    """``make_plane3d_circle_cascaded_pid`` as a :class:`Controller`."""
    g = _load_gains().get("plane3d_circle", {})
    rad = g.get("rad", {})
    params = Plane3DCircleCascadeParams(
        Kp_rad=float(rad.get("Kp", 2.8e-6)),
        Ki_rad=float(rad.get("Ki", 1e-8)),
        Kd_rad=float(rad.get("Kd", 3.2e-4)),
        Kp_bank=float(g.get("bank", {}).get("Kp", -3.24)),
    )
    return Controller(params, Plane3DCascadeState(), plane3d_circle_cascade_step)


def make_plane3d_racetrack_pid_controller() -> Controller:
    """``make_plane3d_racetrack_cascaded_pid`` as a :class:`Controller`."""
    params = Plane3DRacetrackCascadeParams(
        Kp_track=_g3d("racetrack", "track", "Kp", 6e-4),
        Ki_track=_g3d("racetrack", "track", "Ki", 0.0),
        Kd_track=_g3d("racetrack", "track", "Kd", 0.0),
        Kp_bank=_g3d("racetrack", "bank", "Kp", -2.0),
        Kd_bank=_g3d("racetrack", "bank", "Kd", -1.5),
    )
    return Controller(params, Plane3DCascadeState(), plane3d_racetrack_cascade_step)


def make_plane3d_figure8_pid_controller() -> Controller:
    """``make_plane3d_figure8_stateful_pid`` as a :class:`Controller`."""
    params = Plane3DFigure8Params(
        Kp_alt=_g3d("figure8", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("figure8", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("figure8", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("figure8", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("figure8", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("figure8", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("figure8", "bank", "Kp", -2.0),
        power=float(_load_gains().get("plane3d_figure8", {}).get("power", 0.6)),
    )
    return Controller(params, Plane3DFigure8State(), plane3d_figure8_step)
