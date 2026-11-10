"""The expert (the PID slot) for the wind turbine task.

The oracle (the MPC slot) is in ``oracle.py``, so that the expert and
the oracle each have a fingerprint of their own
(``provenance.expert_fingerprint``, ``provenance.oracle_fingerprint``).
"""

import jax.numpy as jnp
import numpy as np
from flax import struct

from target_gym.experts.pid import Controller, gain

# ---------------------------------------------------------------------------
# The expert: the NREL 5 MW baseline controller (Jonkman et al. 2009, ch. 7)
# ---------------------------------------------------------------------------


@struct.dataclass
class NRELBaselineParams:
    """The baseline controller's constants, as published, and the plant's.

    Speeds are generator-side, in rad/s; pitch in rad; torque in N m. The
    pitch gains come from Jonkman's pole placement on the Cp table
    (``scripts/wind_turbine_gains.py``), with his schedule ``GK = 1 / (1 + theta
    / theta_K)``. The published ones (0.01882681 s, 0.008068634, 6.302336 deg)
    linearise with the wake frozen. On the steady table the sensitivity runs
    almost through the origin, so ``Kp`` and ``Ki`` here are the line's value at
    0 deg, far larger, and ``theta_K`` far smaller: only their product with
    ``GK`` means anything, and ``schedule_lo_deg`` and ``schedule_hi_deg`` hold
    the schedule on the curve the line was fitted to.
    """

    Kp: float
    Ki: float
    theta_K_deg: float
    schedule_lo_deg: float
    schedule_hi_deg: float
    N_gear: float
    eta_gen: float
    torque_max_plant: float
    pitch_max_deg: float
    dt: float = 0.25
    filter_hz: float = 0.25
    rated_speed: float = 1173.7 * 2.0 * np.pi / 60.0
    region3_speed: float = 1161.963 * 2.0 * np.pi / 60.0
    region15_lo: float = 670.0 * 2.0 * np.pi / 60.0
    region15_hi: float = 871.0 * 2.0 * np.pi / 60.0
    k_region2: float = 0.0255764 * (60.0 / (2.0 * np.pi)) ** 2
    slip: float = 0.10
    region3_pitch_deg: float = 1.0
    torque_max: float = 47_402.91
    torque_rate: float = 15_000.0
    pitch_rate_deg: float = 8.0


@struct.dataclass
class NRELBaselineState:
    """The filtered speed, the pitch integral (rad of pitch, as ROSCO keeps it),
    the previous commands, and whether the first observation has been read."""

    speed_f: float = 0.0
    integral: float = 0.0
    pitch_prev: float = 0.0
    torque_prev: float = 0.0
    started: float = 0.0


def nrel_baseline_step(params, state, obs, xp=jnp):
    """One 0.25 s step of the NREL 5 MW baseline controller.

    obs: ``[omega_rpm, pitch_deg, torque_pct, P_MW, target_P_MW]``.
    """
    p = params
    w = obs[..., 0] * p.N_gear * (2.0 * np.pi / 60.0)
    pitch = xp.radians(obs[..., 1])
    torque = obs[..., 2] / 100.0 * p.torque_max_plant
    first = state.started < 0.5

    # Single-pole low-pass on the generator speed; starts at the reading.
    alpha = np.exp(-2.0 * np.pi * p.dt * p.filter_hz)
    speed_f = xp.where(first, w, alpha * state.speed_f + (1.0 - alpha) * w)
    pitch_prev = xp.where(first, pitch, state.pitch_prev)
    torque_prev = xp.where(first, torque, state.torque_prev)

    # Torque. Region 3 is constant power at the dispatch (ours: the target
    # over the generator efficiency); below it the baseline table, capped at
    # region 3's torque so a curtailed turbine never exceeds its target.
    power = obs[..., 4] * 1.0e6 / p.eta_gen
    region3 = power / xp.maximum(speed_f, 1e-3)
    sync = p.region3_speed / (1.0 + p.slip)
    line = (power / p.region3_speed) / (p.region3_speed - sync) * (speed_f - sync)
    curve = p.k_region2 * speed_f**2
    ramp = (
        p.k_region2
        * p.region15_hi**2
        * (speed_f - p.region15_lo)
        / (p.region15_hi - p.region15_lo)
    )
    below = xp.where(
        speed_f < p.region15_lo,
        0.0,
        xp.where(speed_f < p.region15_hi, ramp, xp.maximum(curve, line)),
    )
    in_region3 = (pitch_prev >= np.radians(p.region3_pitch_deg)) | (
        speed_f >= p.region3_speed
    )
    tq = xp.where(in_region3, region3, xp.minimum(below, region3))
    tq = xp.clip(tq, 0.0, p.torque_max)
    step_tq = p.torque_rate * p.dt
    tq = xp.clip(tq, torque_prev - step_tq, torque_prev + step_tq)

    # Pitch: PI on the filtered speed error, scheduled on the last command, the
    # integral saturated at the pitch limits and started bumpless from the
    # measured pitch. It adds the scheduled rate, in pitch units, so the poles
    # sit where they were placed at every pitch.
    pitch_max = np.radians(p.pitch_max_deg)
    error = speed_f - p.rated_speed
    read = xp.clip(xp.degrees(pitch_prev), p.schedule_lo_deg, p.schedule_hi_deg)
    gk = 1.0 / (1.0 + read / p.theta_K_deg)
    integral = xp.where(first, pitch, state.integral) + gk * p.Ki * error * p.dt
    integral = xp.clip(integral, 0.0, pitch_max)
    command = xp.clip(gk * p.Kp * error + integral, 0.0, pitch_max)
    step_pitch = np.radians(p.pitch_rate_deg) * p.dt
    command = xp.clip(command, pitch - step_pitch, pitch + step_pitch)

    action = xp.stack(
        [
            2.0 * command / pitch_max - 1.0,
            2.0 * xp.clip(tq / p.torque_max_plant, 0.0, 1.0) - 1.0,
        ],
        axis=-1,
    )
    new_state = NRELBaselineState(
        speed_f=speed_f,
        integral=integral,
        pitch_prev=command,
        torque_prev=tq,
        started=1.0 + 0.0 * w,
    )
    return action, new_state


def make_wind_turbine_pid_controller() -> Controller:
    """The expert: the NREL 5 MW baseline controller as a
    :class:`~target_gym.experts.pid.Controller`. The pitch gains are stored in
    ``pid_gains.json`` by ``scripts/wind_turbine_gains.py``."""
    from target_gym.energy.wind_turbine.env import WindTurbineParams

    wp = WindTurbineParams()
    params = NRELBaselineParams(
        Kp=gain("wind_turbine", "Kp"),
        Ki=gain("wind_turbine", "Ki"),
        theta_K_deg=gain("wind_turbine", "theta_K_deg"),
        schedule_lo_deg=gain("wind_turbine", "schedule_lo_deg"),
        schedule_hi_deg=gain("wind_turbine", "schedule_hi_deg"),
        N_gear=float(wp.N_gear),
        eta_gen=float(wp.eta_gen),
        torque_max_plant=float(wp.torque_max),
        pitch_max_deg=float(wp.pitch_max),
        dt=float(wp.delta_t),
    )
    return Controller(params, NRELBaselineState(), nrel_baseline_step)
