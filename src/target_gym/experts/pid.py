"""
JAX-compatible PID expert controllers for target_gym environments.

Design mirrors gymnax: explicit state structs + pure functions so everything
is jit/vmap/scan-able.

Usage pattern (single episode with jax.lax.scan):

    from target_gym.experts.pid import make_cstr_pid, pid_step

    params, state0 = make_cstr_pid()

    def step_fn(carry, obs):
        pid_state = carry
        action, new_pid_state = pid_step(params, pid_state, obs)
        return new_pid_state, action

    final_state, actions = jax.lax.scan(step_fn, state0, obs_sequence)

Observation layouts (matches each env's get_obs):
    CSTR       : [Ca, T, target_Ca]
    FirstOrder : [x, target_x]
    Nonsmooth  : [x1, x2, target_x1]
    FourTank   : [h1, h2, h3, h4, target_h1, target_h2]
    Car        : [velocity, target_velocity, ...]  (uses only first two)
    Plane      : uses [z, target_altitude] (minimal obs in runner)

All actions are raw-normalized in [-1, 1] (what step_env expects).

Tuning methodology
------------------
Gains are optimised by gradient descent (Adam) on an ITAE loss averaged over
many uniformly-spaced setpoints, via JAX autodiff through the full closed-loop
rollout.  See ``target_gym.experts.pid_tuning`` to re-run the optimisation.

Notes per environment:
- CSTR       : L2 regularisation prevents Kp runaway in the saturation regime.
- Nonsmooth  : targets (1.0–2.0) lie outside the actuator's steady-state range
               (u∈[−1,1] → x1_ss∈[−0.5, 0.5]); gains cannot solve this.
- FourTank   : cross-coupling (v1↔h2, v2↔h1) limits independent-loop PID; the
               asymmetric gains (pid2 stronger) reflect the coupling structure.
"""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

# ---------------------------------------------------------------------------
# Persistent gains store
# ---------------------------------------------------------------------------

# Resolved relative to this file so it works regardless of cwd.
_GAINS_FILE = pathlib.Path(__file__).resolve().parents[3] / "data" / "pid_gains.json"
_gains_cache: dict | None = None


def _load_gains() -> dict:
    """Load gains from data/pid_gains.json, running tuning first if the file is absent."""
    global _gains_cache
    if _gains_cache is None:
        if not _GAINS_FILE.exists():
            print(
                f"[target_gym] No PID gains file found at {_GAINS_FILE}. "
                "Running gradient-based tuning — this may take a few minutes..."
            )
            from target_gym.experts.pid_tuning import tune_all_and_save

            tune_all_and_save(verbose=True)
        with open(_GAINS_FILE) as f:
            _gains_cache = json.load(f)
    return _gains_cache


def _g(env: str, key: str, default: float) -> float:
    """Return gain ``key`` for ``env`` from the JSON store, or ``default``."""
    return float(_load_gains().get(env, {}).get(key, default))


def _load_gain_schedule(env: str) -> dict | None:
    """Return the ``gain_schedule`` sub-dict for *env*, or None if absent."""
    return _load_gains().get(env, {}).get("gain_schedule", None)


# ---------------------------------------------------------------------------
# JAX state and params structs (jit/vmap/scan compatible)
# ---------------------------------------------------------------------------


@struct.dataclass
class PIDState:
    """Carried state for one SISO PID loop."""

    integral: float
    prev_error: float


@struct.dataclass
class PIDParams:
    """Static configuration for one SISO PID loop."""

    Kp: float
    Ki: float
    Kd: float
    dt: float
    state_index: int
    setpoint_index: int
    action_min: float
    action_max: float


@struct.dataclass
class GainSchedulePIDParams:
    """PID with gains interpolated from a table of operating points.

    At each step, the current setpoint is read from the observation and
    (Kp, Ki, Kd) are looked up via ``jnp.interp``. This makes the
    controller adapt to the operating regime — equivalent to classical
    gain scheduling used in industrial process control.

    All array fields must have the same length N (number of grid points).
    ``operating_points`` must be sorted in ascending order.
    """

    operating_points: jnp.ndarray  # (N,) — sorted setpoint values
    Kp_table: jnp.ndarray  # (N,)
    Ki_table: jnp.ndarray  # (N,)
    Kd_table: jnp.ndarray  # (N,)
    dt: float
    state_index: int
    setpoint_index: int
    action_min: float = -1.0
    action_max: float = 1.0


@struct.dataclass
class MIMOPIDState:
    """Carried state for a 2×2 independent-loop PID (e.g. Four Tank)."""

    state1: PIDState
    state2: PIDState


@struct.dataclass
class MIMOPIDParams:
    """Static configuration for a 2×2 independent-loop PID."""

    pid1: PIDParams
    pid2: PIDParams


# ---------------------------------------------------------------------------
# Pure JAX step functions — jit/vmap/scan compatible
# ---------------------------------------------------------------------------


def pid_reset(params: PIDParams) -> PIDState:  # noqa: ARG001
    """Return a zeroed initial state for a SISO PID."""
    return PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def pid_step(
    params: PIDParams,
    state: PIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, PIDState]:
    """
    One discrete PID update step.

    Returns (action [shape (1,)], new_state).
    Anti-windup: the integral is not updated when the output is saturated.
    """
    e = obs[params.setpoint_index] - obs[params.state_index]

    new_integral = state.integral + e * params.dt
    derivative = (e - state.prev_error) / params.dt
    u = params.Kp * e + params.Ki * new_integral + params.Kd * derivative
    u_clipped = jnp.clip(u, params.action_min, params.action_max)

    # Anti-windup: undo integral accumulation on saturation
    new_integral = jnp.where(u == u_clipped, new_integral, state.integral)

    new_state = PIDState(integral=new_integral, prev_error=e)
    return jnp.array([u_clipped]), new_state


def gain_scheduled_pid_step(
    params: GainSchedulePIDParams,
    state: PIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, PIDState]:
    """One PID update with gains interpolated from the operating-point table.

    Identical to ``pid_step`` except Kp/Ki/Kd are looked up via
    ``jnp.interp`` based on the current setpoint in the observation.
    Fully JIT-compatible.
    """
    sp = obs[params.setpoint_index]
    Kp = jnp.interp(sp, params.operating_points, params.Kp_table)
    Ki = jnp.interp(sp, params.operating_points, params.Ki_table)
    Kd = jnp.interp(sp, params.operating_points, params.Kd_table)

    e = sp - obs[params.state_index]
    new_integral = state.integral + e * params.dt
    derivative = (e - state.prev_error) / params.dt
    u = Kp * e + Ki * new_integral + Kd * derivative
    u_clipped = jnp.clip(u, params.action_min, params.action_max)
    new_integral = jnp.where(u == u_clipped, new_integral, state.integral)
    return jnp.array([u_clipped]), PIDState(integral=new_integral, prev_error=e)


@struct.dataclass
class MIMOGainSchedulePIDParams:
    """Two independent gain-scheduled PID loops (e.g. Four Tank, Plane)."""

    pid1: GainSchedulePIDParams
    pid2: GainSchedulePIDParams


def mimo_gain_scheduled_pid_step(
    params: MIMOGainSchedulePIDParams,
    state: MIMOPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, MIMOPIDState]:
    """One step of two independent gain-scheduled PIDs. Returns (action [2,], new_state)."""
    u1, s1 = gain_scheduled_pid_step(params.pid1, state.state1, obs)
    u2, s2 = gain_scheduled_pid_step(params.pid2, state.state2, obs)
    return jnp.concatenate([u1, u2]), MIMOPIDState(state1=s1, state2=s2)


def mimo_pid_reset(params: MIMOPIDParams) -> MIMOPIDState:
    """Return a zeroed initial state for a 2×2 MIMO PID."""
    return MIMOPIDState(
        state1=pid_reset(params.pid1),
        state2=pid_reset(params.pid2),
    )


def mimo_pid_step(
    params: MIMOPIDParams,
    state: MIMOPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, MIMOPIDState]:
    """
    One step of two independent SISO PIDs. Returns (action [shape (2,)], new_state).
    """
    u1, new_state1 = pid_step(params.pid1, state.state1, obs)
    u2, new_state2 = pid_step(params.pid2, state.state2, obs)
    new_state = MIMOPIDState(state1=new_state1, state2=new_state2)
    return jnp.concatenate([u1, u2]), new_state


# ---------------------------------------------------------------------------
# JAX-functional 3D plane PIDs  (jit/vmap/scan compatible)
# ---------------------------------------------------------------------------


@struct.dataclass
class Plane3DPIDState:
    """Carry state shared by all 3D plane PID variants."""

    alt_integral: float
    alt_prev: float
    track_integral: (
        float  # heading integral (heading/fig-8) or radial integral (circle)
    )
    track_prev: float
    # Separate integrator for the power loop (heading task MIMO altitude control)
    power_integral: float
    power_prev: float


@struct.dataclass
class Plane3DHeadingPIDParams:
    """Static params for the heading and figure-8 task PIDs (identical structure).

    The altitude loop is MIMO: altitude error drives BOTH stick (fast pitch) and
    power (slow thrust), mirroring the 2D Airplane2D PID. `power` is a cruise
    bias added on top of the power PID output.
    """

    Kp_alt: float
    Ki_alt: float
    Kd_alt: float
    Kp_hdg: float
    Ki_hdg: float
    Kd_hdg: float
    Kp_bank: float
    power: float  # cruise throttle bias, added to Kp_power·err + ...
    max_bank_rad: float
    dt: float
    Kp_power: float
    Ki_power: float
    Kd_power: float
    Kd_bank: float = 1.5  # roll-rate (phi_dot) damping — kills the bank wobble


@struct.dataclass
class Plane3DCirclePIDParams:
    """Static params for the circle task PID.

    Altitude is MIMO: alt_err drives both stick and power loops.
    """

    Kp_alt: float
    Ki_alt: float
    Kd_alt: float
    Kp_rad: float
    Ki_rad: float
    Kd_rad: float
    Kp_bank: float
    power: float  # cruise throttle bias
    max_bank_rad: float
    dt: float
    gravity: float
    Kp_power: float
    Ki_power: float
    Kd_power: float
    Kd_bank: float = 1.5  # roll-rate (phi_dot) damping — kills the bank wobble


def plane3d_pid_reset(params) -> Plane3DPIDState:  # noqa: ARG001
    z = jnp.zeros(())
    return Plane3DPIDState(
        alt_integral=z,
        alt_prev=z,
        track_integral=z,
        track_prev=z,
        power_integral=z,
        power_prev=z,
    )


def plane3d_heading_pid_step(
    params: Plane3DHeadingPIDParams,
    state: Plane3DPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, Plane3DPIDState]:
    """One step of the heading-task PID. obs must be 1-D (vmap handles batching).

    MIMO altitude control: alt_err drives both stick (fast pitch) and power
    (slow throttle, on top of a cruise bias). Heading controlled via bank.
    """
    alt_err = obs[10] - obs[2]
    new_alt_int = state.alt_integral + alt_err * params.dt
    alt_d = (alt_err - state.alt_prev) / params.dt
    stick_u = (
        params.Kp_alt * alt_err + params.Ki_alt * new_alt_int + params.Kd_alt * alt_d
    )
    stick = jnp.clip(stick_u, -1.0, 1.0)
    new_alt_int = jnp.where(stick_u == stick, new_alt_int, state.alt_integral)

    # Power loop: separate integrator, same altitude error signal.
    new_power_int = state.power_integral + alt_err * params.dt
    power_d = (alt_err - state.power_prev) / params.dt
    power_u = (
        params.power
        + params.Kp_power * alt_err
        + params.Ki_power * new_power_int
        + params.Kd_power * power_d
    )
    power = jnp.clip(power_u, -1.0, 1.0)
    new_power_int = jnp.where(power_u == power, new_power_int, state.power_integral)

    psi = obs[9]
    target_heading = obs[11]
    phi = obs[6]
    phi_dot = obs[7]  # roll rate
    hdg_err = _wrap_angle_jnp(target_heading - psi)
    new_hdg_int = state.track_integral + hdg_err * params.dt
    hdg_d = (hdg_err - state.track_prev) / params.dt
    desired_bank = jnp.clip(
        params.Kp_hdg * hdg_err + params.Ki_hdg * new_hdg_int + params.Kd_hdg * hdg_d,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    bank_err = phi - desired_bank
    aileron = jnp.clip(params.Kp_bank * bank_err - params.Kd_bank * phi_dot, -1.0, 1.0)
    new_hdg_int = jnp.where(jnp.abs(aileron) >= 1.0, state.track_integral, new_hdg_int)

    new_state = Plane3DPIDState(
        alt_integral=new_alt_int,
        alt_prev=alt_err,
        track_integral=new_hdg_int,
        track_prev=hdg_err,
        power_integral=new_power_int,
        power_prev=alt_err,
    )
    return jnp.array([power, stick, aileron]), new_state


def plane3d_circle_pid_step(
    params: Plane3DCirclePIDParams,
    state: Plane3DPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, Plane3DPIDState]:
    """One step of the circle-task PID. obs must be 1-D (vmap handles batching)."""
    alt_err = obs[10] - obs[2]
    new_alt_int = state.alt_integral + alt_err * params.dt
    alt_d = (alt_err - state.alt_prev) / params.dt
    stick_u = (
        params.Kp_alt * alt_err + params.Ki_alt * new_alt_int + params.Kd_alt * alt_d
    )
    stick = jnp.clip(stick_u, -1.0, 1.0)
    new_alt_int = jnp.where(stick_u == stick, new_alt_int, state.alt_integral)

    # Power loop (MIMO altitude control).
    new_power_int = state.power_integral + alt_err * params.dt
    power_d = (alt_err - state.power_prev) / params.dt
    power_u = (
        params.power
        + params.Kp_power * alt_err
        + params.Ki_power * new_power_int
        + params.Kd_power * power_d
    )
    power = jnp.clip(power_u, -1.0, 1.0)
    new_power_int = jnp.where(power_u == power, new_power_int, state.power_integral)

    x_dot = obs[0]
    y_dot = obs[1]
    phi = obs[6]
    phi_dot = obs[7]  # roll rate
    rel_x = obs[11]
    rel_y = obs[12]
    radius = obs[13]
    speed_sq = x_dot**2 + y_dot**2 + 1e-6
    ideal_bank = jnp.arctan2(speed_sq, params.gravity * jnp.maximum(radius, 1.0))
    dist = jnp.sqrt(rel_x**2 + rel_y**2)
    rad_err = dist - radius
    new_rad_int = state.track_integral + rad_err * params.dt
    rad_d = (rad_err - state.track_prev) / params.dt
    bank_corr = (
        params.Kp_rad * rad_err + params.Ki_rad * new_rad_int + params.Kd_rad * rad_d
    )
    desired_bank = jnp.clip(
        ideal_bank + bank_corr, -params.max_bank_rad, params.max_bank_rad
    )
    bank_err = phi - desired_bank
    aileron = jnp.clip(params.Kp_bank * bank_err - params.Kd_bank * phi_dot, -1.0, 1.0)
    new_rad_int = jnp.where(jnp.abs(aileron) >= 1.0, state.track_integral, new_rad_int)

    new_state = Plane3DPIDState(
        alt_integral=new_alt_int,
        alt_prev=alt_err,
        track_integral=new_rad_int,
        track_prev=rad_err,
        power_integral=new_power_int,
        power_prev=alt_err,
    )
    return jnp.array([power, stick, aileron]), new_state


def plane3d_figure8_pid_step(
    params: Plane3DHeadingPIDParams,
    state: Plane3DPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, Plane3DPIDState]:
    """One step of the figure-8 task PID. obs must be 1-D (vmap handles batching)."""
    psi = obs[9]
    phi = obs[6]
    phi_dot = obs[7]  # roll rate
    target_radius = obs[11]
    nearest_dx = obs[12]
    nearest_dy = obs[13]
    nearest_dz = obs[14]
    tangent_heading = obs[15]

    alt_err = nearest_dz
    new_alt_int = state.alt_integral + alt_err * params.dt
    alt_d = (alt_err - state.alt_prev) / params.dt
    stick_u = (
        params.Kp_alt * alt_err + params.Ki_alt * new_alt_int + params.Kd_alt * alt_d
    )
    stick = jnp.clip(stick_u, -1.0, 1.0)
    new_alt_int = jnp.where(jnp.abs(stick_u) >= 1.0, state.alt_integral, new_alt_int)

    # Power loop (MIMO altitude control on nearest_dz).
    new_power_int = state.power_integral + alt_err * params.dt
    power_d = (alt_err - state.power_prev) / params.dt
    power_u = (
        params.power
        + params.Kp_power * alt_err
        + params.Ki_power * new_power_int
        + params.Kd_power * power_d
    )
    power = jnp.clip(power_u, -1.0, 1.0)
    new_power_int = jnp.where(power_u == power, new_power_int, state.power_integral)

    lateral_dist = jnp.sqrt(nearest_dx**2 + nearest_dy**2 + 1e-6)
    blend = jnp.clip(lateral_dist / (0.05 * jnp.maximum(target_radius, 1.0)), 0.0, 1.0)
    correction_heading = jnp.arctan2(nearest_dy, nearest_dx)
    bx = blend * jnp.cos(correction_heading) + (1.0 - blend) * jnp.cos(tangent_heading)
    by = blend * jnp.sin(correction_heading) + (1.0 - blend) * jnp.sin(tangent_heading)
    hdg_err = _wrap_angle_jnp(jnp.arctan2(by, bx) - psi)
    new_hdg_int = state.track_integral + hdg_err * params.dt
    hdg_d = (hdg_err - state.track_prev) / params.dt
    desired_bank = jnp.clip(
        params.Kp_hdg * hdg_err + params.Ki_hdg * new_hdg_int + params.Kd_hdg * hdg_d,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    bank_err = phi - desired_bank
    aileron = jnp.clip(params.Kp_bank * bank_err - params.Kd_bank * phi_dot, -1.0, 1.0)
    new_hdg_int = jnp.where(jnp.abs(aileron) >= 1.0, state.track_integral, new_hdg_int)

    new_state = Plane3DPIDState(
        alt_integral=new_alt_int,
        alt_prev=alt_err,
        track_integral=new_hdg_int,
        track_prev=hdg_err,
        power_integral=new_power_int,
        power_prev=alt_err,
    )
    return jnp.array([power, stick, aileron]), new_state


# ---------------------------------------------------------------------------
# Close-patrol (formation-keeping) pursuit-guidance PID
# ---------------------------------------------------------------------------


@struct.dataclass
class PatrolPIDParams:
    """Static params for the close-patrol follower PID (full-obs layout).

    Pursuit guidance drives the follower into the slot:
      - vertical   : slot up-error -> stick (pitch)
      - along-track: slot back-error -> power (throttle, on a cruise bias)
      - horizontal : slot in-plane error -> desired heading -> bank -> aileron

    The horizontal loop is a *pursuit* controller (position error steers a
    desired heading, not the bank directly), mirroring the circle/figure-8
    experts — direct position->bank bleeds energy in turns and destabilises
    formation-keeping.  When the follower is far from the slot it points at
    it; as it closes (within ``blend_dist``) it blends onto the lead heading
    so the two fly parallel.

    Reads the full-obs vector from
    :func:`target_gym.patrol.env.get_obs_full` (indices documented there).
    """

    Kp_alt: float
    Ki_alt: float
    Kd_alt: float
    Kp_power: float
    Ki_power: float
    Kd_power: float
    cruise: float  # throttle bias (raw [-1, 1])
    Kp_hdg: float
    Ki_hdg: float
    Kd_hdg: float
    Kp_bank: float
    Kd_bank: float  # roll-rate (phi_dot) damping — kills the bank wobble
    max_bank_rad: float
    blend_dist: float  # in-plane distance (m) over which to blend to lead heading
    dt: float


def patrol_pid_step(
    params: PatrolPIDParams,
    state: Plane3DPIDState,
    obs: jnp.ndarray,
) -> tuple[jnp.ndarray, Plane3DPIDState]:
    """One step of the pursuit-guidance formation PID. obs must be 1-D.

    Carries three integrators via :class:`Plane3DPIDState`:
    ``alt_*`` (vertical), ``power_*`` (along-track), ``track_*`` (heading).
    """
    e_back = obs[10]
    e_right = obs[11]
    e_up = obs[12]
    rv_up = obs[15]
    phi = obs[6]
    phi_dot = obs[7]  # roll rate
    psi = obs[9]  # follower heading
    rel_heading = obs[19]  # wrap(lead.psi - follower.psi)

    # Vertical loop: altitude error is -e_up (positive => must climb).
    alt_err = -e_up
    alt_rate = -rv_up  # d(alt_err)/dt
    new_alt_int = state.alt_integral + alt_err * params.dt
    stick_u = (
        params.Kp_alt * alt_err + params.Ki_alt * new_alt_int + params.Kd_alt * alt_rate
    )
    stick = jnp.clip(stick_u, -1.0, 1.0)
    new_alt_int = jnp.where(stick_u == stick, new_alt_int, state.alt_integral)

    # Along-track loop: too far back (e_back > 0) => add power.
    new_power_int = state.power_integral + e_back * params.dt
    power_u = (
        params.cruise
        + params.Kp_power * e_back
        + params.Ki_power * new_power_int
        + params.Kd_power * (e_back - state.power_prev) / params.dt
    )
    power = jnp.clip(power_u, -1.0, 1.0)
    new_power_int = jnp.where(power_u == power, new_power_int, state.power_integral)

    # Horizontal pursuit.  Reconstruct the world-frame vector from follower to
    # slot from the lead-frame errors: with lead heading psi_L, forward_L =
    # (cos, sin), right_L = (sin, -cos), the follower->slot vector works out to
    #   V = e_back * forward_L - e_right * right_L.
    # Its atan2 gives a desired heading in the same right-handed convention as
    # the (tuned) bank loop.  Far from the slot we point at it; close in, we
    # blend onto the lead heading so the pair fly parallel.
    psi_lead = psi + rel_heading
    fwd = jnp.array([jnp.cos(psi_lead), jnp.sin(psi_lead)])
    rgt = jnp.array([jnp.sin(psi_lead), -jnp.cos(psi_lead)])
    v = e_back * fwd - e_right * rgt
    dist_h = jnp.sqrt(e_back**2 + e_right**2 + 1e-6)
    pursuit_err = _wrap_angle_jnp(jnp.arctan2(v[1], v[0]) - psi)
    parallel_err = rel_heading  # wrap(psi_lead - psi)
    blend = jnp.clip(dist_h / params.blend_dist, 0.0, 1.0)
    heading_err = _wrap_angle_jnp(blend * pursuit_err + (1.0 - blend) * parallel_err)

    new_hdg_int = state.track_integral + heading_err * params.dt
    hdg_d = (heading_err - state.track_prev) / params.dt
    desired_bank = jnp.clip(
        params.Kp_hdg * heading_err
        + params.Ki_hdg * new_hdg_int
        + params.Kd_hdg * hdg_d,
        -params.max_bank_rad,
        params.max_bank_rad,
    )
    bank_err = phi - desired_bank
    aileron = jnp.clip(params.Kp_bank * bank_err - params.Kd_bank * phi_dot, -1.0, 1.0)
    new_hdg_int = jnp.where(jnp.abs(aileron) >= 1.0, state.track_integral, new_hdg_int)

    new_state = Plane3DPIDState(
        alt_integral=new_alt_int,
        alt_prev=alt_err,
        track_integral=new_hdg_int,
        track_prev=heading_err,
        power_integral=new_power_int,
        power_prev=e_back,
    )
    return jnp.array([power, stick, aileron]), new_state


def make_patrol_pid() -> tuple[PatrolPIDParams, Plane3DPIDState]:
    """JAX-functional close-patrol follower PID (pursuit guidance).

    Gains default to hand-set values that hold a trailing slot; they can be
    overridden from ``data/pid_gains.json`` under the ``patrol`` key.
    """
    _p = _load_gains().get("patrol", {})
    params = PatrolPIDParams(
        Kp_alt=float(_p.get("Kp_alt", 0.02)),
        Ki_alt=float(_p.get("Ki_alt", 1e-4)),
        Kd_alt=float(_p.get("Kd_alt", 0.05)),
        Kp_power=float(_p.get("Kp_power", 0.003)),
        Ki_power=float(_p.get("Ki_power", 3e-5)),
        Kd_power=float(_p.get("Kd_power", 0.15)),
        cruise=float(_p.get("cruise", 0.6)),
        Kp_hdg=float(_p.get("Kp_hdg", 0.6)),
        Ki_hdg=float(_p.get("Ki_hdg", 0.0)),
        Kd_hdg=float(_p.get("Kd_hdg", 2.5)),
        Kp_bank=float(_p.get("Kp_bank", -2.0)),
        Kd_bank=float(_p.get("Kd_bank", 1.5)),
        max_bank_rad=float(np.deg2rad(30.0)),
        blend_dist=float(_p.get("blend_dist", 500.0)),
        dt=1.0,
    )
    return params, plane3d_pid_reset(params)


# ---------------------------------------------------------------------------
# FunctionalExpertPolicy — JAX-compatible, stateless expert wrapper
# ---------------------------------------------------------------------------


# Registry: which PID param fields are learnable gains for each step_fn.
# Other fields (power, max_bank_rad, dt, gravity) are treated as structural
# constants. The order here is the canonical ordering used by the gain-policy
# action pipeline and by anchor_gains.
_LEARNABLE_GAINS_BY_STEP_FN: dict[str, tuple[str, ...]] = {}


def register_learnable_gains(step_fn, fields: tuple[str, ...]) -> None:
    _LEARNABLE_GAINS_BY_STEP_FN[step_fn.__qualname__] = fields


class FunctionalExpertPolicy:
    """Wraps a functional (params, step_fn) PID pair as a JAX-compatible expert policy.

    Interface::

        policy = FunctionalExpertPolicy(params, zero_state, step_fn)
        pid_state = policy.init_state(num_envs)         # batched initial state
        actions, pid_state = policy(pid_state, obs)     # (num_envs, action_dim)

    The step_fn is vmapped over the env batch dimension automatically.
    Pass as ``eval_expert_policy`` to Ajax agents; thread ``pid_state``
    through the while-loop carry in ``step_environment_expert``.

    Gain-policy interface (used when SAC's actor outputs PID gains):

        expert.learnable_fields        # tuple of gain field names
        expert.anchor_gains            # jnp.ndarray of expert gain values
        expert.step_with_gains(state, obs, gains)  # per-env gains
    """

    def __init__(self, params, zero_state, step_fn):
        self._params = params
        self._zero_state = zero_state
        self._step_fn = step_fn
        self._vmapped_step = jax.vmap(step_fn, in_axes=(None, 0, 0))
        # vmap over (params, state, obs) for per-env gain overrides.
        self._vmapped_step_per_env_params = jax.vmap(step_fn, in_axes=(0, 0, 0))

    @property
    def learnable_fields(self) -> tuple[str, ...]:
        key = self._step_fn.__qualname__
        if key not in _LEARNABLE_GAINS_BY_STEP_FN:
            raise ValueError(
                f"No learnable gain registry entry for step_fn {key}. "
                "Register with register_learnable_gains(step_fn, fields)."
            )
        return _LEARNABLE_GAINS_BY_STEP_FN[key]

    @property
    def anchor_gains(self) -> jnp.ndarray:
        return jnp.array(
            [float(getattr(self._params, f)) for f in self.learnable_fields]
        )

    def step_with_gains(self, state, obs, gains):
        """Step the PID with per-env learnable gains.

        state: pytree batched (num_envs, ...)
        obs:   (num_envs, obs_dim)
        gains: (num_envs, len(learnable_fields))
        Returns (actions, new_state).
        """
        fields = self.learnable_fields
        n = len(fields)
        # Build per-env params by broadcasting self._params and overriding learnable fields.
        num_envs = gains.shape[0]
        base = jax.tree.map(
            lambda x: jnp.broadcast_to(jnp.asarray(x), (num_envs,) + jnp.shape(x)),
            self._params,
        )
        overrides = {fields[i]: gains[:, i] for i in range(n)}
        per_env_params = base.replace(**overrides)
        return self._vmapped_step_per_env_params(per_env_params, state, obs)

    def init_state(self, num_envs: int):
        """Return zero-initialised state with a leading batch dimension."""
        return jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_envs,) + jnp.shape(x)),
            self._zero_state,
        )

    def __call__(self, *args):
        # Two-arg form: (state, obs) -> (action, new_state). State threaded by caller.
        # One-arg form: (obs,) -> action. Stateless per call (fresh zero state),
        # matching the pre-state-threading interface still used across Ajax agents.
        if len(args) == 2:
            state, obs = args
            return self._vmapped_step(self._params, state, obs)
        (obs,) = args
        batch_shape = obs.shape[:-1]
        flat_batch = 1
        for d in batch_shape:
            flat_batch *= d
        obs_flat = obs.reshape((flat_batch, obs.shape[-1]))
        zero = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (flat_batch,) + jnp.shape(x)),
            self._zero_state,
        )
        action, _ = self._vmapped_step(self._params, zero, obs_flat)
        return action.reshape(batch_shape + action.shape[1:])

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


# ---------------------------------------------------------------------------
# Python stateful PIDs — for video rollouts (not JAX-traced)
# ---------------------------------------------------------------------------


class StatefulPID:
    """
    Python stateful PID controller, suitable for select_action closures in
    save_video (which runs in a Python while-loop, not under JAX tracing).

    If fixed_setpoint is given, the setpoint is a constant rather than read
    from obs[setpoint_index].  This is useful when the target is a fixed constant (e.g. 0).
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        dt: float,
        state_index: int,
        action_min: float = -1.0,
        action_max: float = 1.0,
        setpoint_index: int | None = None,
        fixed_setpoint: float | None = None,
    ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.state_index = state_index
        self.setpoint_index = setpoint_index
        self.fixed_setpoint = fixed_setpoint
        self.action_min = action_min
        self.action_max = action_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, obs):
        state_val = obs[..., self.state_index]
        sp = (
            self.fixed_setpoint
            if self.fixed_setpoint is not None
            else obs[..., self.setpoint_index]
        )
        e = sp - state_val
        self.integral = self.integral + e * self.dt
        derivative = (e - self.prev_error) / self.dt
        u = self.Kp * e + self.Ki * self.integral + self.Kd * derivative
        u_clipped = jnp.clip(u, self.action_min, self.action_max)
        self.integral = jnp.where(
            u != u_clipped, self.integral - e * self.dt, self.integral
        )
        self.prev_error = e
        return u_clipped

    __call__ = step


class StatefulGainScheduledPID:
    """Gain-scheduled PID for video/gymnasium rollouts (not JAX-traced).

    At each step, interpolates (Kp, Ki, Kd) from a table of operating
    points based on the current setpoint.
    """

    def __init__(
        self,
        operating_points: np.ndarray,
        Kp_table: np.ndarray,
        Ki_table: np.ndarray,
        Kd_table: np.ndarray,
        dt: float,
        state_index: int,
        setpoint_index: int,
        action_min: float = -1.0,
        action_max: float = 1.0,
    ):
        self.operating_points = np.asarray(operating_points)
        self.Kp_table = np.asarray(Kp_table)
        self.Ki_table = np.asarray(Ki_table)
        self.Kd_table = np.asarray(Kd_table)
        self.dt = dt
        self.state_index = state_index
        self.setpoint_index = setpoint_index
        self.action_min = action_min
        self.action_max = action_max
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, obs):
        sp = obs[..., self.setpoint_index]
        Kp = jnp.interp(sp, self.operating_points, self.Kp_table)
        Ki = jnp.interp(sp, self.operating_points, self.Ki_table)
        Kd = jnp.interp(sp, self.operating_points, self.Kd_table)

        e = sp - obs[..., self.state_index]
        self.integral = self.integral + e * self.dt
        derivative = (e - self.prev_error) / self.dt
        u = Kp * e + Ki * self.integral + Kd * derivative
        u_clipped = jnp.clip(u, self.action_min, self.action_max)
        self.integral = jnp.where(
            u != u_clipped, self.integral - e * self.dt, self.integral
        )
        self.prev_error = e
        return u_clipped

    __call__ = step


class StatefulMIMOPID:
    """Two independent SISO PIDs packaged as a MIMO controller (e.g. FourTank)."""

    def __init__(self, pid1: StatefulPID, pid2: StatefulPID):
        self.pid1 = pid1
        self.pid2 = pid2

    def reset(self):
        self.pid1.reset()
        self.pid2.reset()

    def step(self, obs):
        return jnp.stack([self.pid1.step(obs), self.pid2.step(obs)], axis=-1)

    __call__ = step


# ---------------------------------------------------------------------------
# Per-environment JAX factories  →  (PIDParams, PIDState)
# ---------------------------------------------------------------------------


def make_cstr_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for CSTR — tracks Ca by controlling Tc (coolant temperature).

    Observation : [Ca, T, target_Ca]
    Action      : raw Tc in [-1, 1] → physical [295, 302] K inside the env

    Sign: Ca increases when Tc decreases (more cooling → lower T → slower
    reaction → higher Ca), so all gains are negative.
    dt = env delta_t = 0.25 (matches PC-gym: tsim=25s, N=100).

    Gains are read from data/pid_gains.json if present; keyword arguments
    override the file (useful for ablation / manual testing).
    """
    Kp = Kp if Kp is not None else _g("cstr", "Kp", -103.6)
    Ki = Ki if Ki is not None else _g("cstr", "Ki", -1.86)
    Kd = Kd if Kd is not None else _g("cstr", "Kd", -26.87)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=0.25,
        state_index=0,  # Ca
        setpoint_index=2,  # target_Ca
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_first_order_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for FirstOrderSystem — tracks x by controlling u.

    Observation : [x, target_x]
    Action      : raw u in [-1, 1] → physical [-2, 2] inside the env

    Steady state: x_ss = K * u_actual = u_actual (K=1),
    u_actual = 2 * u_raw, so u_raw_ss = x_target / 2.
    dt = env delta_t = 0.05.
    """
    Kp = Kp if Kp is not None else _g("first_order", "Kp", 2.11)
    Ki = Ki if Ki is not None else _g("first_order", "Ki", 9.72)
    Kd = Kd if Kd is not None else _g("first_order", "Kd", 0.0039)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=0.05,
        state_index=0,  # x
        setpoint_index=1,  # target_x
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_four_tank_pid(
    Kp1: float | None = None,
    Ki1: float | None = None,
    Kd1: float | None = None,
    Kp2: float | None = None,
    Ki2: float | None = None,
    Kd2: float | None = None,
) -> tuple[MIMOPIDParams, MIMOPIDState]:
    """
    Two independent PID loops for FourTank on the CROSS pairing.

    Observation : [h1, h2, h3, h4, target_h1, target_h2]
    Action      : [raw_v1, raw_v2] each in [-1, 1] → physical [0, 10] V

    Pump v1 regulates tank 2 and pump v2 regulates tank 1, because with
    γ1 + γ2 = 0.4 the steady-state RGA element λ11 is *negative* (−0.067) and
    the diagonal pairing is unusable: closing one loop reverses the sign of the
    other. 80 % of each pump goes to the diagonal upper tank, which then drains
    into the lower tank on the opposite side, so the cross pairing is also the
    one the plumbing implies. See :func:`make_four_tank_stateful_pid`.
    dt = env delta_t = 1.0.
    """
    _ft = _load_gains().get("four_tank", {})
    _p1 = _ft.get("pid1", {})
    _p2 = _ft.get("pid2", {})
    Kp1 = Kp1 if Kp1 is not None else float(_p1.get("Kp", 40.0))
    Ki1 = Ki1 if Ki1 is not None else float(_p1.get("Ki", 0.6))
    Kd1 = Kd1 if Kd1 is not None else float(_p1.get("Kd", 0.0))
    Kp2 = Kp2 if Kp2 is not None else float(_p2.get("Kp", 40.0))
    Ki2 = Ki2 if Ki2 is not None else float(_p2.get("Ki", 0.6))
    Kd2 = Kd2 if Kd2 is not None else float(_p2.get("Kd", 0.0))
    params = MIMOPIDParams(
        pid1=PIDParams(
            Kp=Kp1,
            Ki=Ki1,
            Kd=Kd1,
            dt=1.0,
            state_index=1,  # h2  -- v1 fills tank 4, which drains into tank 2
            setpoint_index=5,  # target_h2
            action_min=-1.0,
            action_max=1.0,
        ),
        pid2=PIDParams(
            Kp=Kp2,
            Ki=Ki2,
            Kd=Kd2,
            dt=1.0,
            state_index=0,  # h1  -- v2 fills tank 3, which drains into tank 1
            setpoint_index=4,  # target_h1
            action_min=-1.0,
            action_max=1.0,
        ),
    )
    return params, mimo_pid_reset(params)


def make_four_tank_gs_pid() -> tuple[MIMOGainSchedulePIDParams, MIMOPIDState]:
    """Functional gain-scheduled MIMO PID for FourTank.

    Mirrors :func:`make_four_tank_pid` but reads the per-loop
    ``gain_schedule_pid1`` / ``gain_schedule_pid2`` tables from
    ``data/pid_gains.json``. Pair this with
    :func:`mimo_gain_scheduled_pid_step` and wrap in a
    ``FunctionalExpertPolicy`` to run as a JIT-compatible expert.
    """
    _ft = _load_gains().get("four_tank", {})
    gs1 = _ft.get("gain_schedule_pid1") or {}
    gs2 = _ft.get("gain_schedule_pid2") or {}
    if not gs1 or not gs2:
        raise RuntimeError(
            "make_four_tank_gs_pid requires four_tank.gain_schedule_pid1 "
            "and gain_schedule_pid2 in data/pid_gains.json. Run "
            "`python scripts/tune_pid.py --envs four_tank` first."
        )

    def _to_gs_params(gs: dict, state_index: int, setpoint_index: int):
        ops = jnp.asarray(gs["operating_points"], dtype=jnp.float32)
        return GainSchedulePIDParams(
            operating_points=ops,
            Kp_table=jnp.asarray(gs["Kp"], dtype=jnp.float32),
            Ki_table=jnp.asarray(gs["Ki"], dtype=jnp.float32),
            Kd_table=jnp.asarray(gs.get("Kd", [0.0] * len(ops)), dtype=jnp.float32),
            dt=1.0,
            state_index=state_index,
            setpoint_index=setpoint_index,
            action_min=-1.0,
            action_max=1.0,
        )

    # CROSS pairing, as in make_four_tank_pid: v1 regulates h2, v2 regulates h1.
    params = MIMOGainSchedulePIDParams(
        pid1=_to_gs_params(gs1, state_index=1, setpoint_index=5),
        pid2=_to_gs_params(gs2, state_index=0, setpoint_index=4),
    )
    state = MIMOPIDState(
        state1=PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(())),
        state2=PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(())),
    )
    return params, state


def make_glass_furnace_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for GlassFurnace — tracks T_crown by controlling fuel flow.

    Observation : [T_crown, T_air_preheat, fuel_pct, reversal_phase, target_T_crown]
    Action      : raw fuel in [-1, 1] → physical [fuel_min, fuel_max] kg/s

    Sign: more fuel → more heat → higher T_crown → Kp > 0.
    dt = env delta_t = 30.0 s.
    """
    Kp = Kp if Kp is not None else _g("glass_furnace", "Kp", 0.01)
    Ki = Ki if Ki is not None else _g("glass_furnace", "Ki", 0.001)
    Kd = Kd if Kd is not None else _g("glass_furnace", "Kd", 0.0)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=30.0,
        state_index=0,  # T_crown
        setpoint_index=4,  # target_T_crown
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_reactor_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """
    PID for Reactor — tracks normalised neutron density ``n`` by moving the
    control rod (raw action → rho_ext).

    Observation : [n, T_coolant, rho_ext_norm, target_n]
    Action      : raw rho_ext in [-1, 1] → physical rho_ext in
                  [rho_ext_min, rho_ext_max] ([-0.010, +0.003] by default).

    Sign: withdrawing rods (positive rho_ext) raises dn/dt → Kp > 0.
    The asymmetric action range means the closed-loop gain is larger
    when inserting rods than when withdrawing, which is physically
    correct (rod insertion is always authorised, withdrawal is capped
    below prompt-critical).
    dt matches the control period (delta_t × control_period = 10 s by default).
    """
    Kp = Kp if Kp is not None else _g("reactor", "Kp", 5.0)
    Ki = Ki if Ki is not None else _g("reactor", "Ki", 0.5)
    Kd = Kd if Kd is not None else _g("reactor", "Kd", 0.0)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=10.0,
        state_index=0,  # n
        setpoint_index=3,  # target_n
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_plane_pid(
    Kp1: float | None = None,
    Ki1: float | None = None,
    Kd1: float | None = None,
    Kp2: float | None = None,
    Ki2: float | None = None,
    Kd2: float | None = None,
) -> tuple[MIMOPIDParams, MIMOPIDState]:
    """
    MIMO PID for Airplane2D — controls both power and stick to track altitude.

    Observation (full get_obs): [x_dot, z, z_dot, theta, theta_dot, gamma,
                                  target_altitude, power, stick]
    Action: (power, stick) both in [-1, 1].

    Both loops respond to the altitude error (target_altitude - z):
    - pid1: altitude error → power  (coarse, slow)
    - pid2: altitude error → stick  (fine, fast pitch correction)

    Sign: both positive — higher altitude error → more power / nose-up stick.
    dt = env delta_t = 1.0.
    Gains read from data/pid_gains.json; keyword arguments override.
    """
    _pl = _load_gains().get("plane", {})
    _p1 = _pl.get("pid1", {})
    _p2 = _pl.get("pid2", {})
    Kp1 = Kp1 if Kp1 is not None else float(_p1.get("Kp", 0.0002))
    Ki1 = Ki1 if Ki1 is not None else float(_p1.get("Ki", 0.000005))
    Kd1 = Kd1 if Kd1 is not None else float(_p1.get("Kd", 0.0))
    Kp2 = Kp2 if Kp2 is not None else float(_p2.get("Kp", 0.0005))
    Ki2 = Ki2 if Ki2 is not None else float(_p2.get("Ki", 0.00001))
    Kd2 = Kd2 if Kd2 is not None else float(_p2.get("Kd", 0.001))
    params = MIMOPIDParams(
        pid1=PIDParams(
            Kp=Kp1,
            Ki=Ki1,
            Kd=Kd1,
            dt=1.0,
            state_index=1,  # z (altitude)
            setpoint_index=6,  # target_altitude
            action_min=-1.0,
            action_max=1.0,
        ),
        pid2=PIDParams(
            Kp=Kp2,
            Ki=Ki2,
            Kd=Kd2,
            dt=1.0,
            state_index=1,  # z (altitude)
            setpoint_index=6,  # target_altitude
            action_min=-1.0,
            action_max=1.0,
        ),
    )
    return params, mimo_pid_reset(params)


_PLANE3D_HEADING_LEARNABLE = (
    "Kp_alt",
    "Ki_alt",
    "Kd_alt",
    "Kp_hdg",
    "Ki_hdg",
    "Kd_hdg",
    "Kp_bank",
)
register_learnable_gains(plane3d_heading_pid_step, _PLANE3D_HEADING_LEARNABLE)


def make_plane3d_heading_pid() -> tuple[Plane3DHeadingPIDParams, Plane3DPIDState]:
    """JAX-functional heading-task PID. Gains from data/pid_gains.json.

    Altitude is MIMO-controlled: alt_err drives both stick (pitch) and power
    (throttle, added on top of a cruise bias), matching the 2D Airplane2D PID.
    """
    params = Plane3DHeadingPIDParams(
        Kp_alt=_g3d("heading", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("heading", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("heading", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("heading", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("heading", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("heading", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("heading", "bank", "Kp", -2.0),
        power=float(_load_gains().get("plane3d_heading", {}).get("power", 0.6)),
        max_bank_rad=float(np.deg2rad(25.0)),
        dt=1.0,
        Kp_power=_g3d("heading", "power_pid", "Kp", 2e-4),
        Ki_power=_g3d("heading", "power_pid", "Ki", 5e-6),
        Kd_power=_g3d("heading", "power_pid", "Kd", 0.0),
    )
    return params, plane3d_pid_reset(params)


def make_plane3d_circle_pid() -> tuple[Plane3DCirclePIDParams, Plane3DPIDState]:
    """JAX-functional circle-task PID. Gains from data/pid_gains.json."""
    params = Plane3DCirclePIDParams(
        Kp_alt=_g3d("circle", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("circle", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("circle", "alt", "Kd", 0.001),
        Kp_rad=_g3d("circle", "rad", "Kp", 1e-5),
        Ki_rad=_g3d("circle", "rad", "Ki", 0.0),
        Kd_rad=_g3d("circle", "rad", "Kd", 0.0),
        Kp_bank=_g3d("circle", "bank", "Kp", -2.0),
        power=float(_load_gains().get("plane3d_circle", {}).get("power", 0.6)),
        max_bank_rad=float(np.deg2rad(30.0)),
        dt=1.0,
        gravity=9.81,
        Kp_power=_g3d("circle", "power_pid", "Kp", 2e-4),
        Ki_power=_g3d("circle", "power_pid", "Ki", 5e-6),
        Kd_power=_g3d("circle", "power_pid", "Kd", 0.0),
    )
    return params, plane3d_pid_reset(params)


def make_plane3d_figure8_pid() -> tuple[Plane3DHeadingPIDParams, Plane3DPIDState]:
    """JAX-functional figure-8 task PID. Gains from data/pid_gains.json."""
    params = Plane3DHeadingPIDParams(
        Kp_alt=_g3d("figure8", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("figure8", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("figure8", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("figure8", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("figure8", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("figure8", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("figure8", "bank", "Kp", -2.0),
        power=float(_load_gains().get("plane3d_figure8", {}).get("power", 0.6)),
        max_bank_rad=float(np.deg2rad(25.0)),
        dt=1.0,
        Kp_power=_g3d("figure8", "power_pid", "Kp", 2e-4),
        Ki_power=_g3d("figure8", "power_pid", "Ki", 5e-6),
        Kd_power=_g3d("figure8", "power_pid", "Kd", 0.0),
    )
    return params, plane3d_pid_reset(params)


# ---------------------------------------------------------------------------
# Per-environment Python stateful factories  →  StatefulPID / StatefulMIMOPID
# (for video / gymnasium rollouts)
# ---------------------------------------------------------------------------


def make_cstr_stateful_pid() -> StatefulPID:
    """obs: [Ca, T, target_Ca]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("cstr", "Kp", -103.6),
        Ki=_g("cstr", "Ki", -1.86),
        Kd=_g("cstr", "Kd", -26.87),
        dt=0.25,
        state_index=0,
        setpoint_index=2,
    )


def make_first_order_stateful_pid() -> StatefulPID:
    """obs: [x, target_x]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("first_order", "Kp", 2.11),
        Ki=_g("first_order", "Ki", 9.72),
        Kd=_g("first_order", "Kd", 0.0039),
        dt=0.05,
        state_index=0,
        setpoint_index=1,
    )


def make_four_tank_stateful_pid() -> StatefulMIMOPID:
    """Diagonal PID on the CROSS pairing: v1 -> h2 and v2 -> h1.

    obs: [h1, h2, h3, h4, target_h1, target_h2]

    The pairing is the whole point of this plant. With gamma1 + gamma2 = 0.4
    the process is in Johansson's non-minimum-phase configuration and its
    steady-state RGA is::

        lambda_11 = gamma1 gamma2 / (gamma1 + gamma2 - 1) = -0.067

    A *negative* RGA element means the sign of the v1 -> h1 loop gain flips
    once the v2 -> h2 loop is closed, so integral action on the obvious
    diagonal pairing drives the plant unstable -- which it duly did, tripping
    on the low-level limit within 40 steps. The measured gain matrix says the
    same thing directly: at the operating point dh2/dv1 and dh1/dv2 are about
    four times larger than the diagonal terms.

    So each pump is paired with the tank it actually fills: the flow a pump
    diverts to the *diagonal* upper tank dominates, and that tank drains into
    the lower tank on the other side.

    Gains from data/pid_gains.json.
    """
    _ft = _load_gains().get("four_tank", {})
    _p1 = _ft.get("pid1", {})
    _p2 = _ft.get("pid2", {})
    pid1 = StatefulPID(  # pump v1 regulates tank 2
        Kp=float(_p1.get("Kp", 40.0)),
        Ki=float(_p1.get("Ki", 0.6)),
        Kd=float(_p1.get("Kd", 0.0)),
        dt=1.0,
        state_index=1,
        setpoint_index=5,
    )
    pid2 = StatefulPID(  # pump v2 regulates tank 1
        Kp=float(_p2.get("Kp", 40.0)),
        Ki=float(_p2.get("Ki", 0.6)),
        Kd=float(_p2.get("Kd", 0.0)),
        dt=1.0,
        state_index=0,
        setpoint_index=4,
    )
    return StatefulMIMOPID(pid1, pid2)


def make_glass_furnace_stateful_pid() -> StatefulPID:
    """obs: [T_crown, T_air_preheat, fuel_pct, reversal_phase, target_T_crown]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("glass_furnace", "Kp", 0.01),
        Ki=_g("glass_furnace", "Ki", 0.001),
        Kd=_g("glass_furnace", "Kd", 0.0),
        dt=30.0,
        state_index=0,
        setpoint_index=4,
    )


def make_reactor_stateful_pid() -> StatefulPID:
    """obs: [n, T_coolant, rho_ext_norm, target_n]  (full get_obs layout). Gains from data/pid_gains.json."""
    return StatefulPID(
        Kp=_g("reactor", "Kp", 5.0),
        Ki=_g("reactor", "Ki", 0.5),
        Kd=_g("reactor", "Kd", 0.0),
        dt=0.5,
        state_index=0,
        setpoint_index=3,
    )


def make_plane_stateful_pid() -> StatefulMIMOPID:
    """
    obs: [x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick]
    MIMO PID: both power (pid1) and stick (pid2) track altitude error.
    Gains read from data/pid_gains.json.
    """
    _pl = _load_gains().get("plane", {})
    _p1 = _pl.get("pid1", {})
    _p2 = _pl.get("pid2", {})
    pid1 = StatefulPID(
        Kp=float(_p1.get("Kp", 0.0002)),
        Ki=float(_p1.get("Ki", 0.000005)),
        Kd=float(_p1.get("Kd", 0.0)),
        dt=1.0,
        state_index=1,
        setpoint_index=6,
    )
    pid2 = StatefulPID(
        Kp=float(_p2.get("Kp", 0.0005)),
        Ki=float(_p2.get("Ki", 0.00001)),
        Kd=float(_p2.get("Kd", 0.001)),
        dt=1.0,
        state_index=1,
        setpoint_index=6,
    )
    return StatefulMIMOPID(pid1, pid2)


# ---------------------------------------------------------------------------
# Per-environment gain-scheduled factories (relay autotuning)
# ---------------------------------------------------------------------------
#
# These read from the ``gain_schedule`` sub-dict in data/pid_gains.json.
# If no gain schedule is available, they fall back to the flat-gain factories
# above (wrapping them in a 1-point "schedule" for API compatibility).


def _gs_params_from_json(
    env_name: str,
    dt: float,
    state_index: int,
    setpoint_index: int,
    fallback_Kp: float = 0.0,
    fallback_Ki: float = 0.0,
    fallback_Kd: float = 0.0,
) -> GainSchedulePIDParams:
    """Build a GainSchedulePIDParams from the JSON gain_schedule entry."""
    gs = _load_gain_schedule(env_name)
    if gs is not None:
        ops = jnp.array(gs["operating_points"])
        Kp = jnp.array(gs["Kp"])
        Ki = jnp.array(gs["Ki"])
        Kd = jnp.array(gs["Kd"])
    else:
        # Fallback: single-point schedule from flat gains
        ops = jnp.array([0.0])
        Kp = jnp.array([_g(env_name, "Kp", fallback_Kp)])
        Ki = jnp.array([_g(env_name, "Ki", fallback_Ki)])
        Kd = jnp.array([_g(env_name, "Kd", fallback_Kd)])
    return GainSchedulePIDParams(
        operating_points=ops,
        Kp_table=Kp,
        Ki_table=Ki,
        Kd_table=Kd,
        dt=dt,
        state_index=state_index,
        setpoint_index=setpoint_index,
    )


def _stateful_gs_from_json(
    env_name: str,
    dt: float,
    state_index: int,
    setpoint_index: int,
    fallback_Kp: float = 0.0,
    fallback_Ki: float = 0.0,
    fallback_Kd: float = 0.0,
) -> StatefulGainScheduledPID:
    """Build a StatefulGainScheduledPID from the JSON gain_schedule entry."""
    gs = _load_gain_schedule(env_name)
    if gs is not None:
        ops = np.array(gs["operating_points"])
        Kp = np.array(gs["Kp"])
        Ki = np.array(gs["Ki"])
        Kd = np.array(gs["Kd"])
    else:
        ops = np.array([0.0])
        Kp = np.array([_g(env_name, "Kp", fallback_Kp)])
        Ki = np.array([_g(env_name, "Ki", fallback_Ki)])
        Kd = np.array([_g(env_name, "Kd", fallback_Kd)])
    return StatefulGainScheduledPID(
        operating_points=ops,
        Kp_table=Kp,
        Ki_table=Ki,
        Kd_table=Kd,
        dt=dt,
        state_index=state_index,
        setpoint_index=setpoint_index,
    )


def make_cstr_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "cstr",
        dt=0.25,
        state_index=0,
        setpoint_index=2,
        fallback_Kp=-103.6,
        fallback_Ki=-1.86,
        fallback_Kd=-26.87,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_first_order_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "first_order",
        dt=0.05,
        state_index=0,
        setpoint_index=1,
        fallback_Kp=2.11,
        fallback_Ki=9.72,
        fallback_Kd=0.0039,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_reactor_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "reactor",
        dt=0.5,
        state_index=0,
        setpoint_index=3,
        fallback_Kp=5.0,
        fallback_Ki=0.5,
        fallback_Kd=0.0,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_glass_furnace_gain_scheduled_pid() -> tuple[GainSchedulePIDParams, PIDState]:
    p = _gs_params_from_json(
        "glass_furnace",
        dt=30.0,
        state_index=0,
        setpoint_index=4,
        fallback_Kp=0.01,
        fallback_Ki=0.001,
        fallback_Kd=0.0,
    )
    return p, PIDState(integral=jnp.zeros(()), prev_error=jnp.zeros(()))


def make_cstr_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "cstr",
        dt=0.25,
        state_index=0,
        setpoint_index=2,
        fallback_Kp=-103.6,
        fallback_Ki=-1.86,
        fallback_Kd=-26.87,
    )


def make_first_order_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "first_order",
        dt=0.05,
        state_index=0,
        setpoint_index=1,
        fallback_Kp=2.11,
        fallback_Ki=9.72,
        fallback_Kd=0.0039,
    )


def make_reactor_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "reactor",
        dt=0.5,
        state_index=0,
        setpoint_index=3,
        fallback_Kp=5.0,
        fallback_Ki=0.5,
        fallback_Kd=0.0,
    )


def make_glass_furnace_stateful_gs_pid() -> StatefulGainScheduledPID:
    return _stateful_gs_from_json(
        "glass_furnace",
        dt=30.0,
        state_index=0,
        setpoint_index=4,
        fallback_Kp=0.01,
        fallback_Ki=0.001,
        fallback_Kd=0.0,
    )


# ---------------------------------------------------------------------------
# MIMO gain-scheduled factories (four_tank, plane)
# ---------------------------------------------------------------------------
#
# These read from ``gain_schedule_pid1`` / ``gain_schedule_pid2`` sub-dicts.
# If no gain schedule data is present, they fall back to the flat-gain MIMO
# factories above.


def _load_mimo_gain_schedule(env: str, pid_key: str) -> dict | None:
    """Return ``gain_schedule_pid1`` or ``gain_schedule_pid2`` sub-dict, or None."""
    return _load_gains().get(env, {}).get(pid_key, None)


def _stateful_gs_mimo_from_json(
    env_name: str,
    dt: float,
    state_index_1: int,
    setpoint_index_1: int,
    state_index_2: int,
    setpoint_index_2: int,
    fallback_pid1: tuple[float, float, float],
    fallback_pid2: tuple[float, float, float],
    gs_key1: str = "gain_schedule_pid1",
    gs_key2: str = "gain_schedule_pid2",
) -> StatefulMIMOPID:
    """Build a StatefulMIMOPID with gain-scheduled sub-PIDs from JSON."""
    gs1 = _load_mimo_gain_schedule(env_name, gs_key1)
    gs2 = _load_mimo_gain_schedule(env_name, gs_key2)

    if gs1 is not None:
        pid1 = StatefulGainScheduledPID(
            operating_points=np.array(gs1["operating_points"]),
            Kp_table=np.array(gs1["Kp"]),
            Ki_table=np.array(gs1["Ki"]),
            Kd_table=np.array(gs1["Kd"]),
            dt=dt,
            state_index=state_index_1,
            setpoint_index=setpoint_index_1,
        )
    else:
        _env = _load_gains().get(env_name, {})
        _p1 = _env.get("pid1", {})
        pid1 = StatefulPID(
            Kp=float(_p1.get("Kp", fallback_pid1[0])),
            Ki=float(_p1.get("Ki", fallback_pid1[1])),
            Kd=float(_p1.get("Kd", fallback_pid1[2])),
            dt=dt,
            state_index=state_index_1,
            setpoint_index=setpoint_index_1,
        )

    if gs2 is not None:
        pid2 = StatefulGainScheduledPID(
            operating_points=np.array(gs2["operating_points"]),
            Kp_table=np.array(gs2["Kp"]),
            Ki_table=np.array(gs2["Ki"]),
            Kd_table=np.array(gs2["Kd"]),
            dt=dt,
            state_index=state_index_2,
            setpoint_index=setpoint_index_2,
        )
    else:
        _env = _load_gains().get(env_name, {})
        _p2 = _env.get("pid2", {})
        pid2 = StatefulPID(
            Kp=float(_p2.get("Kp", fallback_pid2[0])),
            Ki=float(_p2.get("Ki", fallback_pid2[1])),
            Kd=float(_p2.get("Kd", fallback_pid2[2])),
            dt=dt,
            state_index=state_index_2,
            setpoint_index=setpoint_index_2,
        )

    return StatefulMIMOPID(pid1, pid2)


def make_four_tank_stateful_gs_pid() -> StatefulMIMOPID:
    """Gain-scheduled MIMO PID for FourTank, on the CROSS pairing.

    obs: [h1, h2, h3, h4, target_h1, target_h2]. See
    :func:`make_four_tank_stateful_pid` for why the loops are crossed.
    """
    return _stateful_gs_mimo_from_json(
        "four_tank",
        dt=1.0,
        state_index_1=1,  # v1 regulates h2
        setpoint_index_1=5,
        state_index_2=0,  # v2 regulates h1
        setpoint_index_2=4,
        fallback_pid1=(9.0, 0.05, 0.0),
        fallback_pid2=(9.0, 0.05, 0.0),
    )


def make_plane_stateful_gs_pid() -> StatefulMIMOPID:
    """Gain-scheduled MIMO PID for Airplane2D. pid1=power, pid2=stick, both on altitude."""
    return _stateful_gs_mimo_from_json(
        "plane",
        dt=1.0,
        state_index_1=1,
        setpoint_index_1=6,  # z, target_altitude
        state_index_2=1,
        setpoint_index_2=6,
        fallback_pid1=(0.0002, 0.000005, 0.0),
        fallback_pid2=(0.0005, 0.00001, 0.001),
    )


# ---------------------------------------------------------------------------
# 3D plane controllers (3 actions: power, stick, aileron)
# ---------------------------------------------------------------------------
#
# All three tasks share an altitude loop (PID on z → stick) and use a
# task-specific lateral loop (heading / circle radius / figure-8 phase →
# desired bank → aileron). Gains are stored under "plane3d_<task>" in
# data/pid_gains.json so each task can be tuned independently.
#
# Structure of the gains dict for each task::
#
#   {
#     "alt":   {"Kp": ..., "Ki": ..., "Kd": ...},   # altitude → stick
#     "lat":   {"Kp": ..., "Ki": ..., "Kd": ...},   # task error → desired bank
#     "bank":  {"Kp": ...},                         # bank error → aileron
#     "power": <float>,                             # fixed cruise throttle
#   }
#
# For the figure-8 task an additional ``period`` field controls the bank
# oscillation period.


def _wrap_angle_jnp(a):
    return jnp.arctan2(jnp.sin(a), jnp.cos(a))


# ── obs index conventions (see plane3d/env.py) ────────────────────────────
# heading obs: [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
#               gamma, psi, target_altitude, target_heading, power, stick, aileron]
#                 0     1   2    3      4         5       6      7
#                 8     9       10              11           12     13     14
# circle  obs: [..., target_altitude, rel_x, rel_y, target_radius, ...]
#                                 10     11     12       13
# fig8    obs: [..., target_altitude, rel_x, rel_y, target_radius, dist, ...]
#                                 10     11     12       13         14


class StatefulPlane3DHeadingPID:
    """
    Heading task PID:
      stick   = altitude PID(z, target_altitude)
      aileron = bank-error PD where desired_bank = lateral PID(heading_err)
      power   = fixed cruise throttle
    """

    def __init__(
        self,
        Kp_alt: float,
        Ki_alt: float,
        Kd_alt: float,
        Kp_hdg: float,
        Ki_hdg: float,
        Kd_hdg: float,
        Kp_bank: float,
        power: float = 0.6,
        max_bank_rad: float = np.deg2rad(25.0),
        dt: float = 1.0,
    ):
        self.alt = StatefulPID(
            Kp=Kp_alt,
            Ki=Ki_alt,
            Kd=Kd_alt,
            dt=dt,
            state_index=2,
            setpoint_index=10,
            action_min=-1.0,
            action_max=1.0,
        )
        self.Kp_hdg, self.Ki_hdg, self.Kd_hdg = Kp_hdg, Ki_hdg, Kd_hdg
        self.Kp_bank = Kp_bank
        self.power = power
        self.max_bank_rad = float(max_bank_rad)
        self.dt = dt
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def reset(self):
        self.alt.reset()
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def step(self, obs):
        stick = self.alt.step(obs)
        psi = obs[..., 9]
        target_heading = obs[..., 11]
        phi = obs[..., 6]

        hdg_err = _wrap_angle_jnp(target_heading - psi)
        self._hdg_int = self._hdg_int + hdg_err * self.dt
        deriv = (hdg_err - self._hdg_prev) / self.dt
        desired_bank = (
            self.Kp_hdg * hdg_err + self.Ki_hdg * self._hdg_int + self.Kd_hdg * deriv
        )
        desired_bank = jnp.clip(desired_bank, -self.max_bank_rad, self.max_bank_rad)
        bank_err = phi - desired_bank
        aileron = jnp.clip(self.Kp_bank * bank_err, -1.0, 1.0)
        self._hdg_int = jnp.where(
            jnp.abs(aileron) >= 1.0, self._hdg_int - hdg_err * self.dt, self._hdg_int
        )
        self._hdg_prev = hdg_err
        return jnp.stack(
            [jnp.broadcast_to(self.power, jnp.shape(stick)), stick, aileron], axis=-1
        )

    __call__ = step


class StatefulPlane3DCirclePID:
    """
    Circle task PID:
      stick   = altitude PID
      aileron = bank-error P controller where desired_bank is computed from
                coordinated-turn physics (φ = atan(v²/gr)) plus a PID
                correction on radial error to stay on the target circle.
      power   = fixed cruise throttle
    """

    def __init__(
        self,
        Kp_alt: float,
        Ki_alt: float,
        Kd_alt: float,
        Kp_rad: float,
        Ki_rad: float,
        Kd_rad: float,
        Kp_bank: float,
        power: float = 0.6,
        target_bank_rad: float = np.deg2rad(15.0),
        max_bank_rad: float = np.deg2rad(30.0),
        dt: float = 1.0,
        gravity: float = 9.81,
    ):
        self.alt = StatefulPID(
            Kp=Kp_alt,
            Ki=Ki_alt,
            Kd=Kd_alt,
            dt=dt,
            state_index=2,
            setpoint_index=10,
            action_min=-1.0,
            action_max=1.0,
        )
        self.Kp_rad, self.Ki_rad, self.Kd_rad = Kp_rad, Ki_rad, Kd_rad
        self.Kp_bank = Kp_bank
        self.power = power
        self.max_bank_rad = float(max_bank_rad)
        self.dt = dt
        self.gravity = gravity
        self._rad_int = 0.0
        self._rad_prev = 0.0

    def reset(self):
        self.alt.reset()
        self._rad_int = 0.0
        self._rad_prev = 0.0

    def step(self, obs):
        stick = self.alt.step(obs)
        x_dot = obs[..., 0]
        y_dot = obs[..., 1]
        rel_x = obs[..., 11]
        rel_y = obs[..., 12]
        radius = obs[..., 13]
        phi = obs[..., 6]

        speed_sq = x_dot**2 + y_dot**2 + 1e-6
        ideal_bank = jnp.arctan2(speed_sq, self.gravity * jnp.maximum(radius, 1.0))

        dist = jnp.sqrt(rel_x**2 + rel_y**2)
        rad_err = dist - radius

        self._rad_int = self._rad_int + rad_err * self.dt
        deriv = (rad_err - self._rad_prev) / self.dt
        bank_correction = (
            self.Kp_rad * rad_err + self.Ki_rad * self._rad_int + self.Kd_rad * deriv
        )
        desired_bank = jnp.clip(
            ideal_bank + bank_correction, -self.max_bank_rad, self.max_bank_rad
        )
        bank_err = phi - desired_bank
        aileron = jnp.clip(self.Kp_bank * bank_err, -1.0, 1.0)
        self._rad_int = jnp.where(
            jnp.abs(aileron) >= 1.0, self._rad_int - rad_err * self.dt, self._rad_int
        )
        self._rad_prev = rad_err
        return jnp.stack(
            [jnp.broadcast_to(self.power, jnp.shape(stick)), stick, aileron], axis=-1
        )

    __call__ = step


class StatefulPlane3DFigureEightPID:
    """
    Figure-8 task PID for the twisted 3D lemniscate:
      stick   = PID on nearest_dz (altitude error to curve)
      aileron = heading PID that blends tangent heading (when on-curve)
                with correction heading (when off-curve), via bank angle
      power   = fixed cruise throttle

    Obs layout (19 values):
      [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
       gamma, psi, target_altitude, target_radius,
       nearest_dx, nearest_dy, nearest_dz, tangent_heading,
       power, stick, aileron]
    """

    def __init__(
        self,
        Kp_alt: float,
        Ki_alt: float,
        Kd_alt: float,
        Kp_hdg: float,
        Ki_hdg: float,
        Kd_hdg: float,
        Kp_bank: float,
        power: float = 0.6,
        max_bank_rad: float = np.deg2rad(25.0),
        dt: float = 1.0,
    ):
        self.Kp_alt = Kp_alt
        self.Ki_alt = Ki_alt
        self.Kd_alt = Kd_alt
        self.Kp_hdg = Kp_hdg
        self.Ki_hdg = Ki_hdg
        self.Kd_hdg = Kd_hdg
        self.Kp_bank = Kp_bank
        self.power = power
        self.max_bank_rad = float(max_bank_rad)
        self.dt = dt
        self._alt_int = 0.0
        self._alt_prev = 0.0
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def reset(self):
        self._alt_int = 0.0
        self._alt_prev = 0.0
        self._hdg_int = 0.0
        self._hdg_prev = 0.0

    def step(self, obs):
        psi = obs[..., 9]
        phi = obs[..., 6]
        target_radius = obs[..., 11]
        nearest_dx = obs[..., 12]
        nearest_dy = obs[..., 13]
        nearest_dz = obs[..., 14]
        tangent_heading = obs[..., 15]

        # ── Altitude: PID on nearest_dz (curve_z - aircraft_z) ──
        alt_err = nearest_dz
        self._alt_int = self._alt_int + alt_err * self.dt
        alt_d = (alt_err - self._alt_prev) / self.dt
        stick = jnp.clip(
            self.Kp_alt * alt_err + self.Ki_alt * self._alt_int + self.Kd_alt * alt_d,
            -1.0,
            1.0,
        )
        self._alt_int = jnp.where(
            jnp.abs(stick) >= 1.0, self._alt_int - alt_err * self.dt, self._alt_int
        )
        self._alt_prev = alt_err

        # ── Heading: blend tangent (on-curve) with correction (off-curve) ──
        lateral_dist = jnp.sqrt(nearest_dx**2 + nearest_dy**2 + 1e-6)
        blend = jnp.clip(
            lateral_dist / (0.05 * jnp.maximum(target_radius, 1.0)), 0.0, 1.0
        )
        correction_heading = jnp.arctan2(nearest_dy, nearest_dx)
        bx = blend * jnp.cos(correction_heading) + (1.0 - blend) * jnp.cos(
            tangent_heading
        )
        by = blend * jnp.sin(correction_heading) + (1.0 - blend) * jnp.sin(
            tangent_heading
        )
        desired_heading = jnp.arctan2(by, bx)

        hdg_err = _wrap_angle_jnp(desired_heading - psi)
        self._hdg_int = self._hdg_int + hdg_err * self.dt
        hdg_d = (hdg_err - self._hdg_prev) / self.dt
        desired_bank = jnp.clip(
            self.Kp_hdg * hdg_err + self.Ki_hdg * self._hdg_int + self.Kd_hdg * hdg_d,
            -self.max_bank_rad,
            self.max_bank_rad,
        )
        bank_err = phi - desired_bank
        aileron = jnp.clip(self.Kp_bank * bank_err, -1.0, 1.0)
        self._hdg_int = jnp.where(
            jnp.abs(aileron) >= 1.0, self._hdg_int - hdg_err * self.dt, self._hdg_int
        )
        self._hdg_prev = hdg_err
        return jnp.stack(
            [jnp.broadcast_to(self.power, jnp.shape(stick)), stick, aileron], axis=-1
        )

    __call__ = step


def _g3d(task: str, group: str, key: str, default: float) -> float:
    """Read a plane3d gain: gains["plane3d_<task>"][group][key]."""
    return float(
        _load_gains().get(f"plane3d_{task}", {}).get(group, {}).get(key, default)
    )


def make_plane3d_heading_stateful_pid() -> StatefulPlane3DHeadingPID:
    """Heading task PID. Gains read from data/pid_gains.json under "plane3d_heading"."""
    cruise = float(_load_gains().get("plane3d_heading", {}).get("power", 0.6))
    return StatefulPlane3DHeadingPID(
        Kp_alt=_g3d("heading", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("heading", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("heading", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("heading", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("heading", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("heading", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("heading", "bank", "Kp", -2.0),
        power=cruise,
    )


def make_plane3d_circle_stateful_pid() -> StatefulPlane3DCirclePID:
    """Circle task PID. Gains read from data/pid_gains.json under "plane3d_circle"."""
    cruise = float(_load_gains().get("plane3d_circle", {}).get("power", 0.6))
    return StatefulPlane3DCirclePID(
        Kp_alt=_g3d("circle", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("circle", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("circle", "alt", "Kd", 0.001),
        Kp_rad=_g3d("circle", "rad", "Kp", 1e-5),
        Ki_rad=_g3d("circle", "rad", "Ki", 0.0),
        Kd_rad=_g3d("circle", "rad", "Kd", 0.0),
        Kp_bank=_g3d("circle", "bank", "Kp", -2.0),
        power=cruise,
    )


def make_plane3d_figure8_stateful_pid() -> StatefulPlane3DFigureEightPID:
    """Figure-8 task PID. Gains read from data/pid_gains.json under "plane3d_figure8".

    Now uses heading-chasing (same structure as heading PID) to follow the
    moving reference point.  Falls back to heading gains if figure8-specific
    gains are absent.
    """
    cruise = float(_load_gains().get("plane3d_figure8", {}).get("power", 0.6))
    return StatefulPlane3DFigureEightPID(
        Kp_alt=_g3d("figure8", "alt", "Kp", 0.0005),
        Ki_alt=_g3d("figure8", "alt", "Ki", 1e-5),
        Kd_alt=_g3d("figure8", "alt", "Kd", 0.001),
        Kp_hdg=_g3d("figure8", "hdg", "Kp", 0.5),
        Ki_hdg=_g3d("figure8", "hdg", "Ki", 0.0),
        Kd_hdg=_g3d("figure8", "hdg", "Kd", 0.0),
        Kp_bank=_g3d("figure8", "bank", "Kp", -2.0),
        power=cruise,
    )


# ---------------------------------------------------------------------------
# Cascaded altitude-hold autopilot (2D plane)
# ---------------------------------------------------------------------------


class StatefulCascadedAltitudePID:
    """Three-loop altitude-hold autopilot with attitude and AoA limiting.

    ``make_plane_stateful_pid`` maps altitude error *directly* to elevator.
    That is not how an altitude hold works, and it stops being survivable once
    the aerodynamics are right: a large altitude error saturates the stick,
    drives angle of attack past the stall and departs controlled flight. The
    airframe is not the limit -- full power with neutral elevator climbs to
    12 200 m at 3.1 deg AoA -- the single loop is.

    A real autopilot cascades, with a limit between each stage::

        altitude error -> commanded vertical speed   (limited, |vs| <= vs_max)
        vertical-speed error -> commanded pitch      (limited, |theta| <= theta_max)
        pitch error -> elevator                      (with pitch-rate damping)

    Each limit converts an unbounded demand into a bounded one, so a large
    setpoint change becomes a *sustained* climb at a safe attitude rather than
    an immediate full-deflection pull. Airspeed is held by a separate throttle
    loop, since pitching for altitude trades speed away.

    On top of that sits **alpha protection**, the same idea as an A320's
    alpha-max: angle of attack is ``theta - gamma`` and both are observable, so
    the commanded pitch is capped at ``gamma + alpha_max``. The aircraft
    physically cannot be commanded to stall.

    Observation layout (``plane.env.get_obs``)::

        [x_dot, z, z_dot, theta, theta_dot, gamma, target_altitude, power, stick]
    """

    def __init__(
        self,
        Kp_alt: float = 0.03,  # altitude error (m) -> vertical speed (m/s)
        vs_max: float = 15.0,  # m/s climb/descent limit (~3000 ft/min)
        Kp_vs: float = 0.010,  # vertical-speed error (m/s) -> pitch (rad)
        Ki_vs: float = 4.0e-4,
        theta_trim: float = 0.032,  # rad, ~1.85 deg cruise trim
        theta_max: float = 0.26,  # rad, ~15 deg pitch limit
        Kp_theta: float = 4.0,  # pitch error (rad) -> elevator (raw)
        Kd_theta: float = 3.0,  # pitch-rate damping
        alpha_max: float = 0.19,  # rad, ~11 deg -- inside the 15 deg stall
        target_speed: float = 230.0,  # m/s, held by the throttle loop
        cruise_power: float = -0.65,  # raw, ~0.175 throttle at cruise
        Kp_speed: float = 0.025,
        Ki_speed: float = 3.0e-4,
        dt: float = 1.0,
    ):
        self.Kp_alt, self.vs_max = Kp_alt, vs_max
        self.Kp_vs, self.Ki_vs = Kp_vs, Ki_vs
        self.theta_trim, self.theta_max = theta_trim, theta_max
        self.Kp_theta, self.Kd_theta = Kp_theta, Kd_theta
        self.alpha_max = alpha_max
        self.target_speed, self.cruise_power = target_speed, cruise_power
        self.Kp_speed, self.Ki_speed = Kp_speed, Ki_speed
        self.dt = dt
        self.reset()

    def reset(self):
        self._vs_integral = 0.0
        self._speed_integral = 0.0

    def step(self, obs):
        x_dot = obs[..., 0]
        z = obs[..., 1]
        z_dot = obs[..., 2]
        theta = obs[..., 3]
        theta_dot = obs[..., 4]
        gamma = obs[..., 5]
        target_altitude = obs[..., 6]

        # --- Outer loop: altitude -> commanded vertical speed -------------
        vs_cmd = jnp.clip(
            self.Kp_alt * (target_altitude - z), -self.vs_max, self.vs_max
        )

        # --- Middle loop: vertical speed -> commanded pitch ----------------
        vs_err = vs_cmd - z_dot
        self._vs_integral = self._vs_integral + vs_err * self.dt
        theta_cmd = (
            self.theta_trim + self.Kp_vs * vs_err + self.Ki_vs * self._vs_integral
        )
        theta_cmd = jnp.clip(theta_cmd, -self.theta_max, self.theta_max)

        # --- Alpha protection: alpha = theta - gamma ----------------------
        theta_cmd = jnp.minimum(theta_cmd, gamma + self.alpha_max)
        theta_cmd = jnp.maximum(theta_cmd, gamma - self.alpha_max)

        # Anti-windup: stop integrating once the pitch demand is limited.
        limited = jnp.abs(theta_cmd - self.theta_trim) >= self.theta_max - 1e-6
        self._vs_integral = jnp.where(
            limited, self._vs_integral - vs_err * self.dt, self._vs_integral
        )

        # --- Inner loop: pitch -> elevator, with rate damping -------------
        stick = self.Kp_theta * (theta_cmd - theta) - self.Kd_theta * theta_dot
        stick = jnp.clip(stick, -1.0, 1.0)

        # --- Throttle loop: hold airspeed ---------------------------------
        speed_err = self.target_speed - x_dot
        self._speed_integral = self._speed_integral + speed_err * self.dt
        power = (
            self.cruise_power
            + self.Kp_speed * speed_err
            + self.Ki_speed * self._speed_integral
        )
        power_clipped = jnp.clip(power, -1.0, 1.0)
        self._speed_integral = jnp.where(
            power != power_clipped,
            self._speed_integral - speed_err * self.dt,
            self._speed_integral,
        )

        return jnp.stack([power_clipped, stick], axis=-1)

    __call__ = step


def make_plane_cascaded_pid() -> StatefulCascadedAltitudePID:
    """Cascaded altitude-hold autopilot for Airplane2D. Gains from JSON if tuned."""
    _p = _load_gains().get("plane_cascaded", {})
    return StatefulCascadedAltitudePID(
        **{k: float(v) for k, v in _p.items() if not isinstance(v, (str, dict, list))}
    )


# ---------------------------------------------------------------------------
# Building HVAC
# ---------------------------------------------------------------------------


def make_hvac_pid(
    Kp: float | None = None,
    Ki: float | None = None,
    Kd: float | None = None,
) -> tuple[PIDParams, PIDState]:
    """PID for BuildingHVAC — tracks zone air temperature with heating power.

    Observation : [T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]
    Action      : raw in [-1, 1] -> [0, Q_heat_max] W

    Sign: more heat raises T_air, so Kp > 0. dt = 900 s (15 min).
    Scale: the envelope loses ~159 W/K, and one raw unit is half of
    Q_heat_max = 3600 W, so a 1 K error calling for ~20 % of full output
    implies Kp ~ 0.4.
    """
    Kp = Kp if Kp is not None else _g("hvac", "Kp", 0.40)
    Ki = Ki if Ki is not None else _g("hvac", "Ki", 1.0e-4)
    Kd = Kd if Kd is not None else _g("hvac", "Kd", 0.0)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=900.0,
        state_index=0,  # T_air
        setpoint_index=6,  # target_T
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_hvac_stateful_pid() -> StatefulPID:
    """obs: [T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]."""
    _p = _load_gains().get("hvac", {})
    return StatefulPID(
        Kp=float(_p.get("Kp", 0.40)),
        Ki=float(_p.get("Ki", 1.0e-4)),
        Kd=float(_p.get("Kd", 0.0)),
        dt=900.0,
        state_index=0,
        setpoint_index=6,
    )


# ---------------------------------------------------------------------------
# Cascaded 3D autopilot
# ---------------------------------------------------------------------------


class _Plane3DVerticalChannel:
    """Altitude -> vertical speed -> pitch -> elevator, with limits.

    The same structural fix as :class:`StatefulCascadedAltitudePID`, for the 3D
    tasks: the original 3D controllers mapped altitude error *directly* to
    elevator, which departs controlled flight once the lift curve is correct.

    Two additions the 2D version does not need:

    * **Bank compensation.** Holding altitude in a banked turn needs lift
      scaled by ``1/cos(phi)``, so the commanded pitch is fed forward by
      ``k_bank_comp * (1/cos(phi) - 1)``. Without it the aircraft sinks
      whenever it turns, and the altitude loop has to chase the loss.
    * **Alpha protection referenced to the flight path**, as in 2D: alpha is
      ``theta - gamma`` and both are observable, so commanded pitch is capped
      at ``gamma + alpha_max``.

    All three 3D tasks share observation indices 0-10, so this channel is
    identical across heading, circle and figure-8.
    """

    def __init__(
        self,
        Kp_alt: float = 0.03,
        vs_max: float = 12.0,
        Kp_vs: float = 0.010,
        Ki_vs: float = 4.0e-4,
        theta_trim: float = 0.032,
        theta_max: float = 0.26,
        Kp_theta: float = 4.0,
        Kd_theta: float = 3.0,
        alpha_max: float = 0.19,
        k_bank_comp: float = 0.15,
        dt: float = 1.0,
    ):
        self.Kp_alt, self.vs_max = Kp_alt, vs_max
        self.Kp_vs, self.Ki_vs = Kp_vs, Ki_vs
        self.theta_trim, self.theta_max = theta_trim, theta_max
        self.Kp_theta, self.Kd_theta = Kp_theta, Kd_theta
        self.alpha_max, self.k_bank_comp = alpha_max, k_bank_comp
        self.dt = dt
        self.reset()

    def reset(self):
        self._vs_integral = 0.0

    def __call__(self, alt_error, z_dot, theta, theta_dot, gamma, phi):
        """``alt_error`` is (desired - actual) altitude, in metres.

        Taking an error rather than (z, target) lets the same channel serve
        absolute altitude tracking and *relative* station keeping, where the
        reference is a moving slot rather than a fixed altitude.
        """
        vs_cmd = jnp.clip(self.Kp_alt * alt_error, -self.vs_max, self.vs_max)
        vs_err = vs_cmd - z_dot
        self._vs_integral = self._vs_integral + vs_err * self.dt

        # Extra pitch needed to hold altitude while banked.
        load_factor = 1.0 / jnp.maximum(jnp.cos(phi), 0.35)
        bank_comp = self.k_bank_comp * (load_factor - 1.0)

        theta_cmd = (
            self.theta_trim
            + bank_comp
            + self.Kp_vs * vs_err
            + self.Ki_vs * self._vs_integral
        )
        theta_cmd = jnp.clip(theta_cmd, -self.theta_max, self.theta_max)
        theta_cmd = jnp.clip(theta_cmd, gamma - self.alpha_max, gamma + self.alpha_max)

        limited = jnp.abs(theta_cmd - self.theta_trim) >= self.theta_max - 1e-6
        self._vs_integral = jnp.where(
            limited, self._vs_integral - vs_err * self.dt, self._vs_integral
        )

        return jnp.clip(
            self.Kp_theta * (theta_cmd - theta) - self.Kd_theta * theta_dot, -1.0, 1.0
        )


class _AirspeedChannel:
    """Throttle loop holding a target airspeed.

    The 3D controllers previously held a *fixed* cruise throttle. Turning and
    climbing both bleed speed, so a fixed setting lets the aircraft decelerate
    toward the stall exactly when it is most loaded.
    """

    def __init__(
        self,
        target_speed: float = 230.0,
        cruise_power: float = 0.2,
        Kp: float = 0.025,
        Ki: float = 3.0e-4,
        dt: float = 1.0,
    ):
        self.target_speed, self.cruise_power = target_speed, cruise_power
        self.Kp, self.Ki, self.dt = Kp, Ki, dt
        self.reset()

    def reset(self):
        self._integral = 0.0

    def __call__(self, speed):
        err = self.target_speed - speed
        self._integral = self._integral + err * self.dt
        power = self.cruise_power + self.Kp * err + self.Ki * self._integral
        clipped = jnp.clip(power, -1.0, 1.0)
        self._integral = jnp.where(
            power != clipped, self._integral - err * self.dt, self._integral
        )
        return clipped


class StatefulCascadedPlane3DPID:
    """Cascaded 3D autopilot: shared vertical + airspeed channels, task lateral law.

    ``lateral_fn(obs, phi, phi_dot, state) -> aileron`` supplies the
    task-specific guidance (hold a heading, hold a circle, follow a
    lemniscate). Everything else -- the altitude cascade, alpha protection,
    bank compensation and airspeed hold -- is shared, because all three 3D
    tasks expose observation indices 0-10 identically.
    """

    def __init__(self, lateral, vertical=None, airspeed=None):
        self.lateral = lateral
        self.vertical = vertical or _Plane3DVerticalChannel()
        self.airspeed = airspeed or _AirspeedChannel()

    def reset(self):
        self.vertical.reset()
        self.airspeed.reset()
        if hasattr(self.lateral, "reset"):
            self.lateral.reset()

    def step(self, obs):
        z, z_dot = obs[..., 2], obs[..., 3]
        theta, theta_dot = obs[..., 4], obs[..., 5]
        phi, phi_dot = obs[..., 6], obs[..., 7]
        gamma = obs[..., 8]
        target_altitude = obs[..., 10]

        stick = self.vertical(target_altitude - z, z_dot, theta, theta_dot, gamma, phi)
        speed = jnp.sqrt(obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-9)
        power = self.airspeed(speed)
        aileron = self.lateral(obs, phi, phi_dot)
        return jnp.stack([power, stick, aileron], axis=-1)

    __call__ = step


class _HeadingLateral:
    """Heading error -> limited bank command -> aileron."""

    def __init__(
        self,
        Kp_hdg,
        Ki_hdg,
        Kd_hdg,
        Kp_bank,
        Kd_bank=0.5,
        max_bank_rad=np.deg2rad(25.0),
        dt=1.0,
    ):
        self.Kp_hdg, self.Ki_hdg, self.Kd_hdg = Kp_hdg, Ki_hdg, Kd_hdg
        self.Kp_bank, self.Kd_bank = Kp_bank, Kd_bank
        self.max_bank_rad, self.dt = float(max_bank_rad), dt
        self.reset()

    def reset(self):
        self._int = 0.0
        self._prev = 0.0

    def __call__(self, obs, phi, phi_dot):
        err = _wrap_angle_jnp(obs[..., 11] - obs[..., 9])
        self._int = self._int + err * self.dt
        deriv = (err - self._prev) / self.dt
        self._prev = err
        desired_bank = jnp.clip(
            self.Kp_hdg * err + self.Ki_hdg * self._int + self.Kd_hdg * deriv,
            -self.max_bank_rad,
            self.max_bank_rad,
        )
        aileron = jnp.clip(
            self.Kp_bank * (phi - desired_bank) + self.Kd_bank * phi_dot, -1.0, 1.0
        )
        self._int = jnp.where(
            jnp.abs(aileron) >= 1.0, self._int - err * self.dt, self._int
        )
        return aileron


class _CircleLateral:
    """Coordinated-turn feedforward plus a radial-error correction."""

    def __init__(
        self,
        Kp_rad,
        Ki_rad,
        Kd_rad,
        Kp_bank,
        Kd_bank=0.5,
        max_bank_rad=np.deg2rad(30.0),
        gravity=9.81,
        dt=1.0,
    ):
        self.Kp_rad, self.Ki_rad, self.Kd_rad = Kp_rad, Ki_rad, Kd_rad
        self.Kp_bank, self.Kd_bank = Kp_bank, Kd_bank
        self.max_bank_rad, self.gravity, self.dt = float(max_bank_rad), gravity, dt
        self.reset()

    def reset(self):
        self._int = 0.0
        self._prev = 0.0

    def __call__(self, obs, phi, phi_dot):
        speed_sq = obs[..., 0] ** 2 + obs[..., 1] ** 2 + 1e-6
        radius = jnp.maximum(obs[..., 13], 1.0)
        # Bank that sustains the turn with no radial error: tan(phi) = v^2/(g r).
        ideal_bank = jnp.arctan2(speed_sq, self.gravity * radius)
        dist = jnp.sqrt(obs[..., 11] ** 2 + obs[..., 12] ** 2)
        err = dist - obs[..., 13]
        self._int = self._int + err * self.dt
        deriv = (err - self._prev) / self.dt
        self._prev = err
        desired_bank = jnp.clip(
            ideal_bank
            + self.Kp_rad * err
            + self.Ki_rad * self._int
            + self.Kd_rad * deriv,
            -self.max_bank_rad,
            self.max_bank_rad,
        )
        aileron = jnp.clip(
            self.Kp_bank * (phi - desired_bank) + self.Kd_bank * phi_dot, -1.0, 1.0
        )
        self._int = jnp.where(
            jnp.abs(aileron) >= 1.0, self._int - err * self.dt, self._int
        )
        return aileron


def make_plane3d_heading_cascaded_pid() -> StatefulCascadedPlane3DPID:
    """Cascaded autopilot for the 3D heading task."""
    g = _load_gains().get("plane3d_heading", {})
    hdg = g.get("hdg", {})
    return StatefulCascadedPlane3DPID(
        lateral=_HeadingLateral(
            Kp_hdg=float(hdg.get("Kp", 2.33)),
            Ki_hdg=float(hdg.get("Ki", 0.0)),
            Kd_hdg=float(hdg.get("Kd", 3.77)),
            Kp_bank=float(g.get("bank", {}).get("Kp", -3.24)),
        )
    )


def make_plane3d_circle_cascaded_pid() -> StatefulCascadedPlane3DPID:
    """Cascaded autopilot for the 3D circle task."""
    g = _load_gains().get("plane3d_circle", {})
    rad = g.get("rad", {})
    return StatefulCascadedPlane3DPID(
        lateral=_CircleLateral(
            Kp_rad=float(rad.get("Kp", 2.8e-6)),
            Ki_rad=float(rad.get("Ki", 1e-8)),
            Kd_rad=float(rad.get("Kd", 3.2e-4)),
            Kp_bank=float(g.get("bank", {}).get("Kp", -3.24)),
        )
    )


def _wrap_angle_np(a):
    """Wrap an angle to (-pi, pi] using numpy, for the stateful controllers."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class StatefulPatrolPID:
    """Stateful wrapper around the functional close-patrol expert.

    ``make_patrol_pid`` / ``patrol_pid_step`` already implement pursuit
    guidance toward the slot and hold formation; what was missing was a
    stateful adapter so Python rollouts, the registry and the conformance
    suite can use them like every other baseline.

    Writing a fresh cascaded controller for this task was tried first and is
    the wrong approach: decomposing the slot error into independent channels,
    and even bearing-pursuit with a heading-match term, settles more than a
    kilometre from the slot against a 60 m tolerance, where the functional
    expert settles at 72 m on a well-behaved seed. Formation flight against a
    manoeuvring lead needs the guidance law that already exists here.

    Valid only for the *full* patrol observation. The bearing-only variant
    withholds the decomposed slot error this law reads (indices 10-12) -- that
    withholding is the point of the variant -- so it needs its own estimator
    rather than this wrapper.
    """

    def __init__(self, params=None, zero_state=None):
        if params is None or zero_state is None:
            params, zero_state = make_patrol_pid()
        self.params = params
        self._zero_state = zero_state
        self.reset()

    def reset(self):
        self.state = self._zero_state

    def step(self, obs):
        action, self.state = patrol_pid_step(self.params, self.state, obs)
        return jnp.asarray(action)

    __call__ = step


class StatefulBearingOnlyPatrolPID:
    """Close-patrol expert for the bearing-only variant, via a lead estimator.

    The full-observation expert reads the slot error already decomposed in the
    lead body frame (obs 10-12). This variant withholds exactly that: a passive
    sensor gives range, azimuth and elevation, and no lead heading or velocity.
    Withholding it is the point of the task, so the fix is not to re-index the
    other controller but to *estimate what is missing* and then reuse the
    guidance law that already works.

    What is actually missing turns out to be small. Range with azimuth and
    elevation is a complete relative-position measurement, so the geometry is
    recoverable in the follower's own frame:

        horiz = range cos(el),  dz = range sin(el)
        d_fwd = horiz cos(az),  d_rgt = horiz sin(az)

    and rotating by the follower's own heading (which it does observe) puts the
    lead in the world frame. The vertical channel then needs no estimation at
    all: ``e_up = -dz - slot_up`` exactly.

    The one genuinely unobservable quantity is the **lead's heading**, which
    the commanded slot needs because the slot is expressed in the lead's frame.
    It is recovered from the lead's motion:

        lead_velocity = follower_velocity + d(relative position)/dt

    differenced across steps and low-pass filtered, since differencing a
    measurement is noisy. Its heading follows by ``atan2``. The estimator is
    seeded with the follower's own heading, which is right to within the
    formation's own angular error at t = 0.

    With that estimate the eight quantities the pursuit law reads are
    reconstructed and handed to :func:`patrol_pid_step` unchanged.
    """

    # Indices into the bearing-only observation.
    _X_DOT, _Y_DOT, _Z_DOT = 0, 1, 3
    _PHI, _PHI_DOT, _PSI = 6, 7, 9
    _RANGE, _AZ, _EL = 10, 11, 12
    _SLOT_BACK, _SLOT_RIGHT, _SLOT_UP = 13, 14, 15

    def __init__(
        self,
        params=None,
        zero_state=None,
        dt: float = 0.1,
        psi_tau: float = 0.6,
        rate_tau: float = 0.4,
    ):
        if params is None or zero_state is None:
            params, zero_state = make_patrol_pid()
        self.params = params
        self._zero_state = zero_state
        self.dt = dt
        # Filter constants, as fractions of a second of memory.
        self.psi_alpha = float(np.exp(-dt / max(psi_tau, 1e-6)))
        self.rate_alpha = float(np.exp(-dt / max(rate_tau, 1e-6)))
        self.reset()

    def reset(self):
        self.state = self._zero_state
        self._prev_rel = None  # (dx, dy, dz) lead minus follower, world frame
        self._psi_lead = None  # filtered lead-heading estimate
        self._lead_zdot = 0.0  # filtered lead vertical rate

    @staticmethod
    def _relative_world(obs):
        """Lead-minus-follower position in the world frame, from the bearing."""
        rng = obs[..., StatefulBearingOnlyPatrolPID._RANGE]
        az = obs[..., StatefulBearingOnlyPatrolPID._AZ]
        el = obs[..., StatefulBearingOnlyPatrolPID._EL]
        psi = obs[..., StatefulBearingOnlyPatrolPID._PSI]
        horiz = rng * jnp.cos(el)
        d_fwd = horiz * jnp.cos(az)
        d_rgt = horiz * jnp.sin(az)
        dz = rng * jnp.sin(el)
        # Follower body axes, matching patrol.env._lead_frame's convention:
        # fwd = (cos, sin), rgt = (sin, -cos).
        dx = d_fwd * jnp.cos(psi) + d_rgt * jnp.sin(psi)
        dy = d_fwd * jnp.sin(psi) - d_rgt * jnp.cos(psi)
        return dx, dy, dz

    def step(self, obs):
        obs = jnp.atleast_1d(jnp.asarray(obs))
        psi = float(obs[self._PSI])
        dx, dy, dz = (float(v) for v in self._relative_world(obs))

        # --- estimate the lead's motion -----------------------------------
        if self._prev_rel is None:
            psi_lead = psi  # seed: formations start near-parallel
            lead_zdot = float(obs[self._Z_DOT])
        else:
            pdx, pdy, pdz = self._prev_rel
            rel_rate = ((dx - pdx) / self.dt, (dy - pdy) / self.dt)
            lead_vx = float(obs[self._X_DOT]) + rel_rate[0]
            lead_vy = float(obs[self._Y_DOT]) + rel_rate[1]
            raw_psi = float(np.arctan2(lead_vy, lead_vx))
            prev = self._psi_lead if self._psi_lead is not None else psi
            # Filter on the *innovation* so the estimate wraps correctly.
            innov = float(_wrap_angle_np(raw_psi - prev))
            psi_lead = prev + (1.0 - self.psi_alpha) * innov
            lead_zdot = self.rate_alpha * self._lead_zdot + (1.0 - self.rate_alpha) * (
                float(obs[self._Z_DOT]) + (dz - pdz) / self.dt
            )
        self._prev_rel = (dx, dy, dz)
        self._psi_lead = psi_lead
        self._lead_zdot = lead_zdot

        # --- rebuild what the pursuit law reads ---------------------------
        fwd = np.array([np.cos(psi_lead), np.sin(psi_lead)])
        rgt = np.array([np.sin(psi_lead), -np.cos(psi_lead)])
        d = np.array([-dx, -dy])  # follower minus lead
        back = -float(d @ fwd)
        lateral = float(d @ rgt)
        up = -dz

        e_back = back - float(obs[self._SLOT_BACK])
        e_right = lateral - float(obs[self._SLOT_RIGHT])
        e_up = up - float(obs[self._SLOT_UP])
        rv_up = float(obs[self._Z_DOT]) - lead_zdot
        rel_heading = float(_wrap_angle_np(psi_lead - psi))

        full = np.zeros(26, dtype=np.float32)
        # Indices 0-9 are identical in both observation layouts (own state),
        # so they pass straight through: the pursuit law reads roll, roll rate
        # and heading from here directly.
        full[0:10] = np.asarray(obs[0:10], dtype=np.float32)
        full[10], full[11], full[12] = e_back, e_right, e_up
        full[15] = rv_up
        full[19] = rel_heading

        action, self.state = patrol_pid_step(self.params, self.state, jnp.asarray(full))
        return jnp.asarray(action)

    __call__ = step


def make_patrol_bearing_only_stateful_pid() -> StatefulBearingOnlyPatrolPID:
    """Close-patrol expert for the bearing-only variant."""
    from target_gym.patrol.env import PatrolParams

    return StatefulBearingOnlyPatrolPID(dt=float(PatrolParams().delta_t))


def make_patrol_stateful_pid() -> StatefulPatrolPID:
    """Stateful close-patrol expert for PlanePatrol (full observation)."""
    return StatefulPatrolPID()


def make_ph_pid(
    Kp: float | None = None, Ki: float | None = None, Kd: float | None = None
) -> tuple[PIDParams, PIDState]:
    """PID for PHNeutralization -- tracks pH by manipulating base flow.

    Observation : [pH, q3_pct, target_pH]
    Action      : raw in [-1, 1] -> [q3_min, q3_max] mL/s

    Sign: more base raises pH, so Kp > 0. dt = 5 s.
    Scale: the raw action spans 12 mL/s and the steady gain near neutrality is
    ~0.5 pH per mL/s, so a 1 pH error calling for ~2 mL/s of base implies
    Kp ~ 0.35. Deliberately detuned relative to that: the titration gain varies
    ~45x across the range, so a controller tuned for the steep middle is
    unstable, and one tuned for the shoulders is sluggish.
    """
    Kp = Kp if Kp is not None else _g("ph_neutralization", "Kp", 0.18)
    Ki = Ki if Ki is not None else _g("ph_neutralization", "Ki", 0.004)
    Kd = Kd if Kd is not None else _g("ph_neutralization", "Kd", 0.0)
    params = PIDParams(
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        dt=5.0,
        state_index=0,
        setpoint_index=2,
        action_min=-1.0,
        action_max=1.0,
    )
    return params, pid_reset(params)


def make_ph_stateful_pid() -> StatefulPID:
    """obs: [pH, q3_pct, target_pH]."""
    _p = _load_gains().get("ph_neutralization", {})
    return StatefulPID(
        Kp=float(_p.get("Kp", 0.18)),
        Ki=float(_p.get("Ki", 0.004)),
        Kd=float(_p.get("Kd", 0.0)),
        dt=5.0,
        state_index=0,
        setpoint_index=2,
    )


def make_distillation_stateful_pid() -> StatefulMIMOPID:
    """Dual-composition PID for the distillation column, LV pairing.

    obs: [yD, xB, L_pct, V_pct, target_yD, target_xB]

    Loop 1: yD -> L (reflux).  dyD/dL ~ +0.92, so the gain is positive.
    Loop 2: xB -> V (boilup).  dxB/dV ~ -1.03 -- more boilup strips the bottoms
    cleaner, *lowering* xB -- so the gain is negative.

    Deliberately detuned. The steady-state gain matrix has an RGA around 50 and
    a condition number around 200: reflux and boilup move both compositions
    almost identically, and the useful direction is a small difference between
    two large, nearly-cancelling effects. Diagonal PID on an ill-conditioned
    plant must be slow, or the loops fight each other.
    """
    _p = _load_gains().get("distillation", {})
    _p1, _p2 = _p.get("pid1", {}), _p.get("pid2", {})
    pid1 = StatefulPID(
        Kp=float(_p1.get("Kp", 4.0)),
        Ki=float(_p1.get("Ki", 0.03)),
        Kd=float(_p1.get("Kd", 0.0)),
        dt=1.0,
        state_index=0,  # yD
        setpoint_index=4,  # target_yD
    )
    pid2 = StatefulPID(
        Kp=float(_p2.get("Kp", -4.0)),
        Ki=float(_p2.get("Ki", -0.03)),
        Kd=float(_p2.get("Kd", 0.0)),
        dt=1.0,
        state_index=1,  # xB
        setpoint_index=5,  # target_xB
    )
    return StatefulMIMOPID(pid1, pid2)


class StatefulWindTurbinePID:
    """Standard above-rated wind turbine controller.

    The classical industrial split, and the reason it is the right baseline:

    * **Generator torque sets power.** Above rated, the torque that delivers
      the setpoint follows directly from ``P = eta N tau w`` -- a feedforward,
      not a loop, because rotor speed is measured and the relation is exact.
    * **Collective pitch regulates rotor speed.** Whatever aerodynamic power
      the torque demand does not absorb has to be spilled, or the rotor
      accelerates. A PI on rotor-speed error does that.

    obs: [omega_rpm, pitch_deg, torque_pct, P_MW, target_P_MW]

    Sign: rotor above rated means excess aerodynamic power, so pitch must
    *increase* to spill it -- hence a positive gain on ``omega - omega_rated``.
    """

    def __init__(
        self,
        Kp_pitch: float = 3.0,
        Ki_pitch: float = 0.6,
        Kd_pitch: float = 0.0,
        omega_rated_rpm: float = 12.1,
        eta_gen: float = 0.944,
        N_gear: float = 97.0,
        torque_max: float = 47_400.0,
        pitch_max: float = 40.0,
        protect_lo: float = 0.60,
        protect_hi: float = 0.90,
        dt: float = 0.25,
    ):
        self.Kp_pitch, self.Ki_pitch, self.Kd_pitch = Kp_pitch, Ki_pitch, Kd_pitch
        self.omega_rated_rpm = omega_rated_rpm
        self.eta_gen, self.N_gear = eta_gen, N_gear
        self.torque_max, self.pitch_max = torque_max, pitch_max
        # Torque is backed off linearly between these fractions of rated speed.
        self.protect_lo, self.protect_hi = protect_lo, protect_hi
        self.dt = dt
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0

    def step(self, obs):
        omega_rpm = obs[..., 0]
        target_MW = obs[..., 4]

        # --- Torque: power feedforward, limited by the Region 2 law ---
        omega = omega_rpm * 2.0 * jnp.pi / 60.0
        torque = (target_MW * 1.0e6) / (
            self.eta_gen * self.N_gear * jnp.maximum(omega, 1e-3)
        )
        # Rotor-speed protection. Below rated wind the setpoint is simply not
        # available, and demanding it anyway drags the rotor down until it
        # stalls. Backing the torque demand off as speed falls lets the rotor
        # re-accelerate. It is scaled to be inactive at and above rated speed,
        # so it never interferes with normal Region 3 regulation -- capping at
        # the Region 2 law tau = K omega^2 instead does interfere, because at
        # rated speed that law sits well below rated torque and the rotor runs
        # away.
        speed_ratio = omega / (self.omega_rated_rpm * 2.0 * jnp.pi / 60.0)
        protection = jnp.clip(
            (speed_ratio - self.protect_lo) / (self.protect_hi - self.protect_lo),
            0.0,
            1.0,
        )
        torque = torque * protection
        torque_raw = 2.0 * jnp.clip(torque / self.torque_max, 0.0, 1.0) - 1.0

        # --- Pitch: PI on rotor-speed error ---
        err = omega_rpm - self.omega_rated_rpm
        self._integral = self._integral + err * self.dt
        deriv = (err - self._prev_err) / self.dt
        self._prev_err = err
        pitch = (
            self.Kp_pitch * err + self.Ki_pitch * self._integral + self.Kd_pitch * deriv
        )
        pitch_clipped = jnp.clip(pitch, 0.0, self.pitch_max)
        # Anti-windup: stop integrating once pitch is against a stop.
        self._integral = jnp.where(
            pitch != pitch_clipped, self._integral - err * self.dt, self._integral
        )
        pitch_raw = 2.0 * (pitch_clipped / self.pitch_max) - 1.0

        return jnp.stack([pitch_raw, torque_raw], axis=-1)

    __call__ = step


class StatefulBoilerDrumPID:
    """Three-element drum level control plus a pressure loop.

    Three-element control is the standard industrial answer to shrink and
    swell, and it is the right baseline precisely because it does not try to
    fight the inverse response with tuning:

    * **Feedwater follows measured steam flow directly.** That is the mass
      balance, closed as a feedforward rather than through the level gauge, so
      it is immune to the level lying during a transient. This is the element
      that makes the loop stable.
    * **The level PI only trims that feedforward.** It is deliberately slow.
      A fast level loop reacts to swell by *cutting* feedwater exactly when
      mass is leaving fastest, which is how single-element control drives a
      boiler into a low-level trip.
    * **Firing regulates pressure.** Pressure is what the burners actually
      control; a PI on pressure error suffices because the energy balance has
      no inverse response.

    obs: [level, pressure, q_steam, fuel_pct, feed_pct, target_level, target_pressure]
    """

    def __init__(
        self,
        Kp_level: float,
        Ki_level: float,
        Kp_pressure: float,
        Ki_pressure: float,
        Kd_pressure: float,
        q_feed_max: float,
        Q_max: float,
        h_feedwater: float,
        dt: float = 2.0,
        trim_max: float = 25.0,
    ):
        self.Kp_level, self.Ki_level = Kp_level, Ki_level
        self.Kp_pressure = Kp_pressure
        self.Ki_pressure = Ki_pressure
        self.Kd_pressure = Kd_pressure
        self.q_feed_max = q_feed_max
        self.Q_max = Q_max
        self.h_feedwater = h_feedwater
        self.dt = dt
        # The level trim is bounded so the feedforward always dominates.
        self.trim_max = trim_max
        self.reset()

    def reset(self):
        self._level_int = 0.0
        self._press_int = 0.0
        self._prev_press_err = 0.0

    def step(self, obs):
        level = obs[..., 0]
        pressure = obs[..., 1]
        q_steam = obs[..., 2]
        target_level = obs[..., 5]
        target_pressure = obs[..., 6]

        # --- Feedwater: steam-flow feedforward, trimmed by a slow level PI ---
        err_l = target_level - level
        self._level_int = self._level_int + err_l * self.dt
        trim = self.Kp_level * err_l + self.Ki_level * self._level_int
        trim_clipped = jnp.clip(trim, -self.trim_max, self.trim_max)
        self._level_int = jnp.where(
            trim != trim_clipped, self._level_int - err_l * self.dt, self._level_int
        )
        q_feed = jnp.clip(q_steam + trim_clipped, 0.0, self.q_feed_max)
        feed_raw = 2.0 * (q_feed / self.q_feed_max) - 1.0

        # --- Firing: enthalpy feedforward plus a PI on pressure ---
        # The heat needed to turn the measured feedwater into steam is known,
        # so the loop only has to correct the balance rather than find it.
        Q_ff = q_steam * (2.75e6 - self.h_feedwater)
        err_p = target_pressure - pressure
        self._press_int = self._press_int + err_p * self.dt
        deriv = (err_p - self._prev_press_err) / self.dt
        self._prev_press_err = err_p
        Q = Q_ff + self.Q_max * (
            self.Kp_pressure * err_p
            + self.Ki_pressure * self._press_int
            + self.Kd_pressure * deriv
        )
        Q_clipped = jnp.clip(Q, 0.0, self.Q_max)
        self._press_int = jnp.where(
            Q != Q_clipped, self._press_int - err_p * self.dt, self._press_int
        )
        fuel_raw = 2.0 * (Q_clipped / self.Q_max) - 1.0

        return jnp.stack([fuel_raw, feed_raw], axis=-1)

    __call__ = step


def make_boiler_drum_stateful_pid() -> StatefulBoilerDrumPID:
    """Three-element level control plus pressure control for the drum boiler."""
    from target_gym.boiler_drum.env import BoilerDrumParams

    pr = BoilerDrumParams()
    _p = _load_gains().get("boiler_drum", {})
    return StatefulBoilerDrumPID(
        Kp_level=float(_p.get("Kp_level", 60.0)),
        Ki_level=float(_p.get("Ki_level", 0.6)),
        Kp_pressure=float(_p.get("Kp_pressure", 0.05)),
        Ki_pressure=float(_p.get("Ki_pressure", 0.002)),
        Kd_pressure=float(_p.get("Kd_pressure", 0.0)),
        q_feed_max=pr.q_feed_max,
        Q_max=pr.Q_max,
        h_feedwater=pr.h_feedwater,
        dt=pr.delta_t,
    )


def make_boiler_drum_pid():
    """JAX-functional 2x2 variant, for ``env.expert_policy``.

    Two independent loops: level -> feedwater, pressure -> firing. It has no
    steam-flow feedforward -- ``PIDParams`` cannot carry one -- so it is the
    weaker single-element controller, and the stateful three-element version
    above is the real baseline.
    """
    _p = _load_gains().get("boiler_drum", {})
    params = MIMOPIDParams(
        pid1=PIDParams(
            Kp=float(_p.get("Kp_fuel", 0.4)),
            Ki=float(_p.get("Ki_fuel", 0.01)),
            Kd=0.0,
            dt=2.0,
            state_index=1,  # pressure
            setpoint_index=6,  # target_pressure
            action_min=-1.0,
            action_max=1.0,
        ),
        pid2=PIDParams(
            Kp=float(_p.get("Kp_feed", 4.0)),
            Ki=float(_p.get("Ki_feed", 0.05)),
            Kd=0.0,
            dt=2.0,
            state_index=0,  # level
            setpoint_index=5,  # target_level
            action_min=-1.0,
            action_max=1.0,
        ),
    )
    return params, mimo_pid_reset(params)


class StatefulCementKilnPID:
    """Cascade control for the rotary kiln, plus a speed feedforward.

    Cascade is the standard answer to a long dead time, and it is the right
    baseline for exactly that reason:

    * **Inner loop: fuel -> burning-zone temperature.** The pyrometer responds
      in minutes and carries no transport delay, so this loop can be tuned
      reasonably fast.
    * **Outer loop: free lime -> burning-zone setpoint.** Free lime is what the
      plant is paid for, but it is half an hour behind the fuel. Closing a fast
      loop directly around it would oscillate with the delay period; the outer
      loop is therefore deliberately slow and only trims the inner setpoint.
    * **Kiln speed follows feed.** Holdup is feed x residence and residence
      goes as 1/speed, so running speed proportional to the measured feed rate
      holds the bed depth constant. That is real practice, and it keeps the
      transport delay from wandering when the feed does.

    obs: [lime_pct, T_bz, T_exhaust, T_back_end, feed, fuel_pct, speed_pct,
          target_lime_pct]
    """

    def __init__(
        self,
        Kp_outer: float,
        Ki_outer: float,
        Kp_inner: float,
        Ki_inner: float,
        T_bz_base: float,
        fuel_min: float,
        fuel_max: float,
        fuel_nominal: float,
        rpm_min: float,
        rpm_max: float,
        rpm_nominal: float,
        feed_nominal: float,
        dt: float = 30.0,
        sp_trim_max: float = 120.0,
    ):
        self.Kp_outer, self.Ki_outer = Kp_outer, Ki_outer
        self.Kp_inner, self.Ki_inner = Kp_inner, Ki_inner
        self.T_bz_base = T_bz_base
        self.fuel_min, self.fuel_max = fuel_min, fuel_max
        self.fuel_nominal = fuel_nominal
        self.rpm_min, self.rpm_max = rpm_min, rpm_max
        self.rpm_nominal, self.feed_nominal = rpm_nominal, feed_nominal
        self.dt = dt
        # The setpoint trim is bounded so the outer loop cannot drive the kiln
        # to a temperature that would form rings or let it go cold.
        self.sp_trim_max = sp_trim_max
        self.reset()

    def reset(self):
        self._outer_int = 0.0
        self._inner_int = 0.0

    def step(self, obs):
        lime = obs[..., 0] / 100.0
        T_bz = obs[..., 1]
        feed = obs[..., 4]
        target_lime = obs[..., 7] / 100.0

        # --- Outer: free lime sets the burning-zone target ---
        # Lime above target means the charge is under-burnt, so the setpoint
        # must go UP. Hence a positive gain on (lime - target).
        err_o = lime - target_lime
        self._outer_int = self._outer_int + err_o * self.dt
        trim = self.Kp_outer * err_o + self.Ki_outer * self._outer_int
        trim_clipped = jnp.clip(trim, -self.sp_trim_max, self.sp_trim_max)
        self._outer_int = jnp.where(
            trim != trim_clipped, self._outer_int - err_o * self.dt, self._outer_int
        )
        T_bz_sp = self.T_bz_base + trim_clipped

        # --- Inner: fuel holds the burning-zone temperature ---
        err_i = T_bz_sp - T_bz
        self._inner_int = self._inner_int + err_i * self.dt
        fuel = (
            self.fuel_nominal + self.Kp_inner * err_i + self.Ki_inner * self._inner_int
        )
        fuel_clipped = jnp.clip(fuel, self.fuel_min, self.fuel_max)
        self._inner_int = jnp.where(
            fuel != fuel_clipped, self._inner_int - err_i * self.dt, self._inner_int
        )
        fuel_raw = (
            2.0 * (fuel_clipped - self.fuel_min) / (self.fuel_max - self.fuel_min) - 1.0
        )

        # --- Speed follows feed, holding bed depth constant ---
        rpm = self.rpm_nominal * feed / self.feed_nominal
        rpm = jnp.clip(rpm, self.rpm_min, self.rpm_max)
        rpm_raw = 2.0 * (rpm - self.rpm_min) / (self.rpm_max - self.rpm_min) - 1.0

        return jnp.stack([fuel_raw, rpm_raw], axis=-1)

    __call__ = step


def make_cement_kiln_stateful_pid() -> StatefulCementKilnPID:
    """Cascade free-lime / burning-zone controller for the rotary kiln."""
    from target_gym.cement_kiln.env import CementKilnParams

    pr = CementKilnParams()
    _p = _load_gains().get("cement_kiln", {})
    return StatefulCementKilnPID(
        Kp_outer=float(_p.get("Kp_outer", 12000.0)),
        Ki_outer=float(_p.get("Ki_outer", 6.0)),
        Kp_inner=float(_p.get("Kp_inner", 0.025)),
        Ki_inner=float(_p.get("Ki_inner", 5.0e-6)),
        T_bz_base=float(_p.get("T_bz_base", 1765.0)),
        fuel_min=pr.fuel_min,
        fuel_max=pr.fuel_max,
        fuel_nominal=pr.fuel_nominal,
        rpm_min=pr.rpm_min,
        rpm_max=pr.rpm_max,
        rpm_nominal=pr.rpm_nominal,
        feed_nominal=pr.raw_meal_nominal,
        dt=pr.delta_t,
    )


def make_cement_kiln_pid():
    """JAX-functional 2x2 variant, for ``env.expert_policy``.

    Single-loop rather than cascade: fuel is driven straight off the delayed
    free-lime measurement, which is the naive structure the cascade exists to
    avoid. It is the weaker controller by design.
    """
    _p = _load_gains().get("cement_kiln", {})
    params = MIMOPIDParams(
        pid1=PIDParams(
            Kp=float(_p.get("Kp_fuel", -20.0)),
            Ki=float(_p.get("Ki_fuel", -0.05)),
            Kd=0.0,
            dt=30.0,
            state_index=0,  # lime_pct
            setpoint_index=7,  # target_lime_pct
            action_min=-1.0,
            action_max=1.0,
        ),
        pid2=PIDParams(
            Kp=0.0,
            Ki=0.0,
            Kd=0.0,
            dt=30.0,
            state_index=6,  # speed_pct -- held at its reset value
            setpoint_index=6,
            action_min=-1.0,
            action_max=1.0,
        ),
    )
    return params, mimo_pid_reset(params)


def make_wind_turbine_stateful_pid() -> StatefulWindTurbinePID:
    """Above-rated controller for the NREL 5 MW turbine."""
    _p = _load_gains().get("wind_turbine", {})
    return StatefulWindTurbinePID(
        Kp_pitch=float(_p.get("Kp_pitch", 3.0)),
        Ki_pitch=float(_p.get("Ki_pitch", 0.6)),
        Kd_pitch=float(_p.get("Kd_pitch", 0.0)),
    )


class StatefulBatteryPID:
    """Dispatch-following controller with state-of-charge guarding.

    Power is close to a direct feedthrough -- commanding it delivers it, minus
    conversion losses -- so the tracking loop is mostly feedforward with a
    small proportional trim for the losses.

    The interesting part is the guard. A battery cannot hold a setpoint
    indefinitely, and hitting either state-of-charge limit ends the episode
    irrecoverably, so the demand is faded out as the pack approaches a limit
    *in the direction that would breach it*. Discharging is throttled near
    empty and charging near full; neither is touched in the middle. Without
    this the controller follows dispatch straight into a terminal state.

    obs: [soc, V_cell, T_cell, P_MW, target_P_MW]
    """

    def __init__(
        self,
        Kp: float = 0.5,
        Ki: float = 0.02,
        guard_margin: float = 0.12,
        soc_min: float = 0.05,
        soc_max: float = 0.95,
        power_max_MW: float = 1.0,
        dt: float = 5.0,
    ):
        self.Kp, self.Ki = Kp, Ki
        self.guard_margin = guard_margin
        self.soc_min, self.soc_max = soc_min, soc_max
        self.power_max_MW = power_max_MW
        self.dt = dt
        self.reset()

    def reset(self):
        self._integral = 0.0

    def step(self, obs):
        soc = obs[..., 0]
        delivered_MW = obs[..., 3]
        target_MW = obs[..., 4]

        err = target_MW - delivered_MW
        self._integral = self._integral + err * self.dt
        demand = target_MW + self.Kp * err + self.Ki * self._integral

        # Fade the demand out only in the direction that would breach a limit.
        discharge_room = jnp.clip((soc - self.soc_min) / self.guard_margin, 0.0, 1.0)
        charge_room = jnp.clip((self.soc_max - soc) / self.guard_margin, 0.0, 1.0)
        demand = jnp.where(demand > 0.0, demand * discharge_room, demand * charge_room)

        raw = jnp.clip(demand / self.power_max_MW, -1.0, 1.0)
        # Anti-windup: stop integrating once the demand is clipped or guarded.
        self._integral = jnp.where(
            jnp.abs(raw) >= 1.0, self._integral - err * self.dt, self._integral
        )
        return raw

    __call__ = step


def make_battery_stateful_pid() -> StatefulBatteryPID:
    """Dispatch-following controller for the grid battery."""
    _p = _load_gains().get("battery", {})
    return StatefulBatteryPID(
        Kp=float(_p.get("Kp", 0.5)),
        Ki=float(_p.get("Ki", 0.02)),
        guard_margin=float(_p.get("guard_margin", 0.12)),
    )
