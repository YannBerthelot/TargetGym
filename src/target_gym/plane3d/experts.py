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
    Controller,
    _wrap_angle,
    register_learnable_gains,
)
from target_gym.plane.experts_tecs import (
    AIRLINER,
    TECSParams,
    TECSState,
    pitch_hold,
    tecs_step,
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
# The experts: an autopilot's loops under the field's guidance laws
# ---------------------------------------------------------------------------
#
# Each task's expert is what a fixed-wing autopilot flies (PX4, ArduPilot):
# TECS for height and speed (``plane.experts_tecs``) over a pitch loop, and a
# damped bank loop under a lateral guidance law. The heading task holds a
# course; the circle, racetrack and figure-8 share L1 guidance (Park, Deyst and
# How, 2004), the law ArduPilot flies paths and loiters with, plus the path's
# curvature as a coordinated-turn feedforward. Each is one pure
# ``step(params, state, obs, xp)`` for ``experts.pid.Controller``: NumPy on
# the host, jnp under jit.
#
# Signs: a positive bank turns toward increasing heading, and a positive
# aileron rolls that way. ``cross`` is the offset from the path, positive on
# that same side, and ``curvature`` is positive where the path turns that way.

_BANK_45 = float(np.deg2rad(45.0))
_SIN_45 = float(np.sin(np.deg2rad(45.0)))
_HALF_PI = float(np.pi / 2.0)


@struct.dataclass
class Plane3DGuidanceParams:
    """Lateral guidance and its bank loop, over TECS and a pitch loop.

    ``tau_hdg`` is the course hold's time constant (heading task). ``period``
    and ``damping`` are L1's, as ArduPilot names them ``NAVL1_PERIOD`` and
    ``NAVL1_DAMPING``. ``max_bank_rad`` is ArduPilot's default roll limit and
    the edge of the model's validity (PHYSICS.md). ``k_curv`` smooths the
    figure-8's curvature estimate, the only path whose geometry
    the observation does not give in closed form. The pitch loop holds TECS's
    pitch demand, with the extra pitch a banked turn needs and the angle of
    attack kept inside ``alpha_max``. ``v_min_eas`` is the slowest equivalent
    airspeed the autopilot flies (ArduPilot's ``ARSPD_FBW_MIN``): 1.3 times
    the model's clean 1 g stall, 80 m/s at 73.5 t, CL_max 1.5 and 122.6 m^2.
    """

    Kp_roll: float = 6.0
    Kd_roll: float = 3.0
    max_bank_rad: float = _BANK_45
    tau_hdg: float = 10.0
    period: float = 35.0
    damping: float = 0.75
    k_curv: float = 0.3
    Kp_theta: float = 8.0
    Ki_theta: float = 2.0
    Kd_theta: float = 0.3
    alpha_max: float = 0.19
    k_bank_comp: float = 0.15
    target_speed: float = 230.0
    v_min_eas: float = 104.0
    gravity: float = 9.81
    dt: float = 1.0
    tecs: TECSParams = AIRLINER


@struct.dataclass
class Plane3DGuidanceState:
    """TECS's state, the pitch loop's integral, and the figure-8's last path
    point and estimates."""

    tecs: TECSState = struct.field(default_factory=TECSState)
    theta_integ: float = 0.0
    tangent_prev: float = 0.0
    curvature: float = 0.0
    started: float = 0.0


def _ground_speed(obs, xp):
    return xp.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-6)


def _turn_speed(params, radius, xp):
    """Cruise speed, or slower where the tightest turn of radius ``radius``
    would otherwise need more bank than the limit leaves in hand."""
    usable = _TURN_BANK_MARGIN * params.max_bank_rad
    limit = xp.sqrt(params.gravity * xp.maximum(radius, 1.0) * xp.tan(usable))
    return xp.minimum(params.target_speed, limit)


def _v_min(params, z, xp):
    """``v_min_eas`` as a true airspeed at height ``z``, by the standard
    atmosphere's density ratio below 11 km."""
    sigma = xp.maximum(1.0 - 2.2558e-5 * z, 0.2) ** 4.2559
    return params.v_min_eas / xp.sqrt(sigma)


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


def _fly(params, state, obs, xp, h_dem, accel, target_speed, **carry):
    """The autopilot under a guidance law: TECS on height and speed, a pitch
    loop holding its demand, and the bank that makes ``accel`` in a
    coordinated turn, held by a loop damped on roll rate."""
    z, z_dot = obs[..., 2], obs[..., 3]
    theta, theta_dot = obs[..., 4], obs[..., 5]
    phi, phi_dot = obs[..., 6], obs[..., 7]
    gamma = obs[..., 8]
    v = xp.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2 + z_dot**2)
    # Stall prevention, as ArduPilot's STALL_PREVENTION: never slower than
    # the speed at which the full bank is safe, and never more bank than the
    # speed supports, n = (v / v_min)^2.
    v_min = _v_min(params, z, xp)
    target_speed = xp.maximum(
        target_speed, v_min / xp.sqrt(xp.cos(params.max_bank_rad))
    )
    max_bank = xp.arccos(xp.clip((v_min / v) ** 2, xp.cos(params.max_bank_rad), 1.0))
    power, pitch, tecs = tecs_step(
        params.tecs, state.tecs, z, z_dot, v, h_dem, target_speed, phi, xp
    )
    load_factor = 1.0 / xp.maximum(xp.cos(phi), 0.35)
    theta_cmd = pitch + params.k_bank_comp * (load_factor - 1.0)
    theta_cmd = xp.clip(theta_cmd, gamma - params.alpha_max, gamma + params.alpha_max)
    stick, theta_integ = pitch_hold(
        params, state.theta_integ, theta_cmd, theta, theta_dot, xp
    )
    bank = xp.clip(xp.arctan(accel / params.gravity), -max_bank, max_bank)
    aileron = xp.clip(
        params.Kp_roll * (bank - phi) - params.Kd_roll * phi_dot, -1.0, 1.0
    )
    new_state = state.replace(tecs=tecs, theta_integ=theta_integ, **carry)
    return xp.stack([power, stick, aileron], axis=-1), new_state


def plane3d_heading_step(params, state, obs, xp=jnp):
    """Course hold: the turn rate that closes the heading error in ``tau_hdg``."""
    v = _ground_speed(obs, xp)
    turn_rate = _wrap_angle(obs[..., 11] - obs[..., 9], xp) / params.tau_hdg
    target = params.target_speed
    return _fly(params, state, obs, xp, obs[..., 10], v * turn_rate, target)


def plane3d_circle_step(params, state, obs, xp=jnp):
    """L1 on the circle, flown with heading increasing."""
    rx, ry = obs[..., 11], obs[..., 12]
    radius = xp.maximum(obs[..., 13], 1.0)
    cross = radius - xp.sqrt(rx**2 + ry**2 + 1e-6)
    tangent = xp.arctan2(ry, rx) + _HALF_PI
    accel = l1_lateral_accel(params, cross, tangent, 1.0 / radius, obs, xp)
    target = _turn_speed(params, radius, xp)
    return _fly(params, state, obs, xp, obs[..., 10], accel, target)


def plane3d_racetrack_step(params, state, obs, xp=jnp):
    """L1 on the holding pattern, whose guidance the observation gives."""
    cross, tangent, curvature = obs[..., 18], obs[..., 19], obs[..., 20]
    accel = l1_lateral_accel(params, cross, tangent, curvature, obs, xp)
    target = _turn_speed(params, obs[..., 13], xp)
    return _fly(params, state, obs, xp, obs[..., 10], accel, target)


def plane3d_figure8_step(params, state, obs, xp=jnp):
    """L1 on the figure-8. Its curvature is not observed, so it is estimated
    from how the nearest point's tangent turns."""
    a = xp.maximum(obs[..., 11], 1.0)
    dx, dy, dz, tangent = obs[..., 12], obs[..., 13], obs[..., 14], obs[..., 15]
    cross = dx * xp.sin(tangent) - dy * xp.cos(tangent)
    curve_z = obs[..., 2] + dz

    # A lemniscate's tightest turn has radius a / 3, at its tips.
    k_max = 4.5 / a
    turned = _wrap_angle(tangent - state.tangent_prev, xp)
    curvature_now = xp.clip(
        turned / (_ground_speed(obs, xp) * params.dt), -k_max, k_max
    )
    started = state.started > 0.0
    k = params.k_curv
    curvature = xp.where(
        started, state.curvature + k * (curvature_now - state.curvature), 0.0
    )

    accel = l1_lateral_accel(params, cross, tangent, curvature, obs, xp)
    target = _turn_speed(params, a / 3.0, xp)
    return _fly(
        params,
        state,
        obs,
        xp,
        curve_z,
        accel,
        target,
        tangent_prev=tangent,
        curvature=curvature,
        started=xp.ones_like(dz),
    )


_LATERAL_GAINS = ("Kp_roll", "Kd_roll")
register_learnable_gains(plane3d_heading_step, _LATERAL_GAINS + ("tau_hdg",))
for _step in (plane3d_circle_step, plane3d_racetrack_step, plane3d_figure8_step):
    register_learnable_gains(_step, _LATERAL_GAINS + ("period", "damping"))


def make_plane3d_heading_pid_controller() -> Controller:
    """The heading task's expert: course hold over the autopilot's loops."""
    return Controller(_HEADING, Plane3DGuidanceState(), plane3d_heading_step)


def make_plane3d_circle_pid_controller() -> Controller:
    """The circle's expert: L1 guidance over the autopilot's loops."""
    return Controller(_PATH, Plane3DGuidanceState(), plane3d_circle_step)


def make_plane3d_racetrack_pid_controller() -> Controller:
    """The holding pattern's expert: L1 guidance over the autopilot's loops."""
    return Controller(_PATH, Plane3DGuidanceState(), plane3d_racetrack_step)


def make_plane3d_figure8_pid_controller() -> Controller:
    """The figure-8's expert: L1 guidance over the autopilot's loops."""
    return Controller(_PATH, Plane3DGuidanceState(), plane3d_figure8_step)


# Tuned inner loop first, as ArduPilot's tuning guide orders it, on training
# seeds 100-109 and checked on held-out seeds 200-209, never on the scored
# seeds 0-9 (scripts/tune_aircraft_guidance.py).
_HEADING = Plane3DGuidanceParams()
_PATH = Plane3DGuidanceParams()
