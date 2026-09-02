"""
Close-patrol (formation-keeping) environment: state, parameters and transition.

Two aircraft share the exact 3D physics of :mod:`target_gym.plane3d`:

  - the **lead** flies a scripted patrol pattern (straight-and-level, or a
    constant-rate turn) driven by the tuned 3D heading autopilot;
  - the **follower** is the learning agent and must reach and hold a *slot* —
    a target position expressed in the lead's body frame (so many metres
    behind, to one side and above/below the lead).

This is a **dynamic target MDP**: the target subset of states is "follower is
in the slot", but the slot moves with the (maneuvering) lead.  New challenges
relative to the single-aircraft tasks:

  - a non-stationary, maneuvering reference;
  - relative-frame observations;
  - collision as a new irrecoverable state (the follower may not hit the lead).

The physics core (:func:`compute_next_state_3d`) is reused verbatim for both
aircraft — this module only *composes* two of them and adds the slot geometry,
reward, observation and termination logic.  Task-specific observation layouts
live in the environment classes (``env_jax.py``); everything here is a pure,
jit/vmap/scan-able function over structs.
"""

from typing import Tuple

import jax.numpy as jnp
from flax import struct

from target_gym.base import EnvState
from target_gym.experts.pid import Plane3DPIDState, plane3d_heading_pid_step
from target_gym.plane.dynamics import advance_gust
from target_gym.plane3d.dynamics import compute_velocity_3d
from target_gym.plane3d.env import (
    PlaneParams3D,
    PlaneState3D,
    compute_next_state_3d,
    get_obs_heading,
    wrap_angle,
)

# ─── State & parameters ─────────────────────────────────


@struct.dataclass
class PatrolState(EnvState):
    """Combined state of the two aircraft plus the slot definition.

    ``follower`` is the learner; ``lead`` is scripted.  ``lead_pid`` carries
    the lead autopilot's integrator state so the scripted lead tracks its
    (possibly moving) heading/altitude setpoints without steady-state drift.
    The slot offsets are constant over an episode (sampled at reset) and are
    expressed in the lead body frame: *back* = metres behind the lead along
    its velocity, *right* = metres to the lead's right, *up* = metres above.
    """

    follower: PlaneState3D
    lead: PlaneState3D
    lead_pid: Plane3DPIDState
    slot_back: float
    slot_right: float
    slot_up: float
    lead_turn_rate: float  # rad per step commanded onto the lead heading
    # Shared formation turbulence gust (m/s): all aircraft are in the same air
    # mass, so they feel one common OU gust on top of the steady params.wind.
    gust_x: float = 0.0
    gust_y: float = 0.0
    gust_z: float = 0.0


@struct.dataclass
class PatrolParams(PlaneParams3D):
    """Physics params (inherited from :class:`PlaneParams3D`) + patrol config."""

    # Slot geometry sampled at reset (metres).  Defaults put the follower in a
    # classic trailing echelon: ~200 m back, ~120 m to one side, co-altitude.
    slot_back_range: Tuple[float, float] = (150.0, 300.0)
    slot_right_range: Tuple[float, float] = (-150.0, 150.0)
    slot_up_range: Tuple[float, float] = (-60.0, 60.0)

    # Reward / termination shaping.
    slot_tolerance: float = 60.0  # sigma of the Gaussian slot reward (m)
    # Heading-alignment tolerance (rad): the follower should fly roughly
    # parallel to the lead (like a real wingman), not merely occupy the slot
    # position.  30 deg sigma nudges toward parallel flight without dominating.
    heading_tolerance: float = 0.5236
    min_separation: float = 25.0  # collision distance (m) -> terminal
    max_slot_error: float = 1500.0  # follower lost the formation (m) -> terminal

    # Lead behaviour.  Turn rate is sampled in [-r, r] rad/step; 0 => straight
    # and level.  At delta_t = 1 s, 0.003 rad/step ~ 0.17 deg/s ~ a very gentle
    # standard-rate-ish orbit for an airliner.
    lead_turn_rate_range: Tuple[float, float] = (-0.003, 0.003)

    # Follower is spawned near the slot with this much isotropic position noise
    # (m) so the episode starts solvable but not perfectly trimmed.
    follower_spawn_noise: float = 40.0

    delta_t: float = 1.0


def make_cruise_state(
    x: float,
    y: float,
    z: float,
    psi: float,
    speed: float,
    params: PatrolParams,
    target_altitude: float,
    target_heading: float,
) -> PlaneState3D:
    """Build a trimmed, straight-and-level :class:`PlaneState3D` at cruise.

    Used to spawn both aircraft at reset: level flight, wings level, velocity
    of magnitude ``speed`` along heading ``psi``.
    """
    x_dot = speed * jnp.cos(psi)
    y_dot = speed * jnp.sin(psi)
    theta = jnp.deg2rad(params.initial_theta)
    return PlaneState3D(
        x=x,
        x_dot=x_dot,
        y=y,
        y_dot=y_dot,
        z=z,
        z_dot=0.0,
        theta=theta,
        theta_dot=0.0,
        phi=0.0,
        phi_dot=0.0,
        psi=psi,
        alpha=theta,  # gamma = 0 at level flight
        gamma=0.0,
        m=params.initial_mass,  # fuel is a component of it, not an addition
        power=params.initial_power,
        stick=jnp.deg2rad(params.initial_stick),
        aileron=jnp.deg2rad(params.initial_aileron),
        fuel=params.initial_fuel_quantity,
        time=0,
        target_altitude=target_altitude,
        target_heading=target_heading,
        target_x=0.0,
        target_y=0.0,
        target_radius=0.0,
    )


# ─── Slot geometry ──────────────────────────────────────


def _lead_frame(lead: PlaneState3D):
    """Return (forward, right) horizontal unit vectors of the lead body frame.

    ``forward`` points along the lead heading, ``right`` is 90 deg clockwise
    from it (the lead's right wing), both in the world x-y plane.
    """
    psi = lead.psi
    forward = jnp.array([jnp.cos(psi), jnp.sin(psi)])
    right = jnp.array([jnp.sin(psi), -jnp.cos(psi)])
    return forward, right


def desired_slot_position(state: PatrolState):
    """World-frame (x, y, z) of the slot the follower should occupy."""
    forward, right = _lead_frame(state.lead)
    lead_xy = jnp.array([state.lead.x, state.lead.y])
    slot_xy = lead_xy - state.slot_back * forward + state.slot_right * right
    slot_z = state.lead.z + state.slot_up
    return slot_xy[0], slot_xy[1], slot_z


def slot_error_vector(state: PatrolState):
    """Follower-minus-slot error, decomposed in the lead body frame.

    Returns (e_back, e_right, e_up): the signed distances by which the
    follower's *back/right/up* offset from the lead differs from the commanded
    slot.  All three are zero exactly in the slot.
    """
    forward, right = _lead_frame(state.lead)
    d = jnp.array([state.follower.x - state.lead.x, state.follower.y - state.lead.y])
    back = -jnp.dot(d, forward)  # positive = behind the lead
    lateral = jnp.dot(d, right)  # positive = to the lead's right
    up = state.follower.z - state.lead.z
    return back - state.slot_back, lateral - state.slot_right, up - state.slot_up


def slot_error(state: PatrolState):
    """Euclidean magnitude of :func:`slot_error_vector` (metres)."""
    eb, er, eu = slot_error_vector(state)
    return jnp.sqrt(eb**2 + er**2 + eu**2 + 1e-8)


def separation(state: PatrolState):
    """3D distance between the two aircraft (metres)."""
    dx = state.follower.x - state.lead.x
    dy = state.follower.y - state.lead.y
    dz = state.follower.z - state.lead.z
    return jnp.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)


# ─── Termination & reward ───────────────────────────────


def check_is_terminal_patrol(state: PatrolState, params: PatrolParams, xp=jnp):
    """Return (terminated, truncated).

    Terminal (irrecoverable) if either aircraft leaves the altitude envelope,
    the two collide, or the follower falls out of formation past
    ``max_slot_error``.
    """
    follower_crash = xp.logical_or(
        state.follower.z <= params.min_alt, state.follower.z >= params.max_alt
    )
    lead_crash = xp.logical_or(
        state.lead.z <= params.min_alt, state.lead.z >= params.max_alt
    )
    collision = separation(state) <= params.min_separation
    lost = slot_error(state) >= params.max_slot_error
    terminated = follower_crash | lead_crash | collision | lost
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def heading_alignment(state: PatrolState, params: PatrolParams, xp=jnp):
    """Gaussian on the follower-vs-lead heading difference (1 when parallel).

    Encodes the *patrol* geometry: a wingman flies parallel to the lead, not
    just at the right position.
    """
    dpsi = wrap_angle(state.follower.psi - state.lead.psi)
    return xp.exp(-0.5 * (dpsi / params.heading_tolerance) ** 2)


def compute_reward_patrol(state: PatrolState, params: PatrolParams, xp=jnp):
    """Slot-position Gaussian * heading-alignment, with a hard terminal penalty.

    Mirrors the shaping style of the path-following 3D tasks (a Gaussian in the
    tracking error) and the crash-penalty convention of the whole suite
    (``-max_steps_in_episode`` on an irrecoverable state).  The multiplicative
    heading factor makes the target "fly the slot *parallel* to the lead".
    """
    terminated, _ = check_is_terminal_patrol(state, params, xp)
    err = slot_error(state)
    track_r = xp.exp(-0.5 * (err / params.slot_tolerance) ** 2)
    align_r = heading_alignment(state, params, xp)
    return xp.where(terminated, -1.0 * params.max_steps_in_episode, track_r * align_r)


# ─── Observations ───────────────────────────────────────


def _relative_velocity_lead_frame(state: PatrolState):
    """Follower-minus-lead velocity decomposed in the lead body frame."""
    forward, right = _lead_frame(state.lead)
    dv = jnp.array(
        [
            state.follower.x_dot - state.lead.x_dot,
            state.follower.y_dot - state.lead.y_dot,
        ]
    )
    rv_forward = jnp.dot(dv, forward)
    rv_right = jnp.dot(dv, right)
    rv_up = state.follower.z_dot - state.lead.z_dot
    return rv_forward, rv_right, rv_up


def get_obs_full(state: PatrolState, xp=jnp):
    """Full-state observation (26 values).

    The follower sees its own attitude/velocity, the relative position and
    velocity of the lead in the lead body frame, the commanded slot, the
    relative heading and lead speed, its own actuators, and the scalar slot
    error (value) against a constant-zero target.

    Layout::

        [ x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot, gamma, psi,   # 0-9
          e_back, e_right, e_up,                                                # 10-12
          rv_back, rv_right, rv_up,                                             # 13-15
          slot_back, slot_right, slot_up,                                       # 16-18
          rel_heading, lead_speed,                                             # 19-20
          power, stick, aileron,                                              # 21-23
          slot_error, target_slot_error(=0) ]                                  # 24-25
    """
    f = state.follower
    eb, er, eu = slot_error_vector(state)
    rvb, rvr, rvu = _relative_velocity_lead_frame(state)
    rel_heading = wrap_angle(state.lead.psi - f.psi)
    lead_speed = compute_velocity_3d(
        state.lead.x_dot, state.lead.y_dot, state.lead.z_dot
    )
    return xp.stack(
        [
            f.x_dot,
            f.y_dot,
            f.z,
            f.z_dot,
            f.theta,
            f.theta_dot,
            f.phi,
            f.phi_dot,
            f.gamma,
            f.psi,
            eb,
            er,
            eu,
            rvb,
            rvr,
            rvu,
            state.slot_back,
            state.slot_right,
            state.slot_up,
            rel_heading,
            lead_speed,
            f.power,
            f.stick,
            f.aileron,
            slot_error(state),
            jnp.zeros(()),
        ]
    )


def get_obs_bearing_only(state: PatrolState, xp=jnp):
    """Partial (bearing + range) observation (21 values).

    A passive sensor on the follower yields only the range and the
    azimuth/elevation *bearing* to the lead — no lead velocity, heading or the
    decomposed slot error.  Recovering the slot then requires the follower to
    infer the lead's motion.  The tracked scalar is measured range against the
    commanded slot range.

    Layout::

        [ x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot, gamma, psi,   # 0-9
          range, azimuth, elevation,                                           # 10-12
          slot_back, slot_right, slot_up,                                       # 13-15
          power, stick, aileron,                                              # 16-18
          range, slot_range ]                                                  # 19-20 (value, target)
    """
    f = state.follower
    # Bearing measured in the *follower* body frame.
    fwd = jnp.array([jnp.cos(f.psi), jnp.sin(f.psi)])
    rgt = jnp.array([jnp.sin(f.psi), -jnp.cos(f.psi)])
    d = jnp.array([state.lead.x - f.x, state.lead.y - f.y])
    d_fwd = jnp.dot(d, fwd)
    d_rgt = jnp.dot(d, rgt)
    dz = state.lead.z - f.z
    horiz = jnp.sqrt(d_fwd**2 + d_rgt**2 + 1e-8)
    rng = separation(state)
    azimuth = jnp.arctan2(d_rgt, d_fwd)
    elevation = jnp.arctan2(dz, horiz)
    slot_range = jnp.sqrt(
        state.slot_back**2 + state.slot_right**2 + state.slot_up**2 + 1e-8
    )
    return xp.stack(
        [
            f.x_dot,
            f.y_dot,
            f.z,
            f.z_dot,
            f.theta,
            f.theta_dot,
            f.phi,
            f.phi_dot,
            f.gamma,
            f.psi,
            rng,
            azimuth,
            elevation,
            state.slot_back,
            state.slot_right,
            state.slot_up,
            f.power,
            f.stick,
            f.aileron,
            rng,
            slot_range,
        ]
    )


# ─── Action decoding (shared with Plane3D convention) ───


def decode_action(action: jnp.ndarray):
    """Map a raw [-1, 1]^3 action to physical (power[0,1], stick[rad], aileron[rad])."""
    power, stick, aileron = action
    power = (power + 1) / 2
    stick = jnp.deg2rad(stick * 15)
    aileron = jnp.deg2rad(aileron * 25)
    return power, stick, aileron


# ─── Combined transition ────────────────────────────────


def step_lead(state: PatrolState, params: PatrolParams, lead_pid_params, method: str):
    """Advance the scripted lead one step with its heading autopilot.

    The lead's commanded heading is stored in ``lead.target_heading`` and
    advanced by ``lead_turn_rate`` each step, producing straight-and-level
    flight (rate 0) or a constant-rate orbit.  Returns (new_lead, new_lead_pid).
    """
    commanded_heading = wrap_angle(state.lead.target_heading + state.lead_turn_rate)
    lead_cmd = state.lead.replace(target_heading=commanded_heading)
    lead_obs = get_obs_heading(lead_cmd)
    lead_action, new_pid = plane3d_heading_pid_step(
        lead_pid_params, state.lead_pid, lead_obs
    )
    power, stick, aileron = decode_action(lead_action)
    new_lead, _ = compute_next_state_3d(
        power, stick, aileron, lead_cmd, params, integration_method=method
    )
    return new_lead, new_pid


def compute_next_state_patrol(
    action: jnp.ndarray,
    state: PatrolState,
    params: PatrolParams,
    lead_pid_params,
    integration_method: str = "rk4_2",
    key=None,
):
    """One environment step: advance the follower (from ``action``) and the lead.

    A single **shared** turbulence gust (advanced with ``key``) is applied to
    every aircraft through ``eff_params`` — all planes are in the same air mass —
    while each still gets its own altitude-dependent wind shear inside
    :func:`compute_next_state_3d`.  ``key=None`` keeps a constant (steady) wind.
    """
    gust = advance_gust(
        jnp.array([state.gust_x, state.gust_y, state.gust_z]),
        params.turbulence_theta,
        params.turbulence_sigma,
        params.delta_t,
        key,
    )
    eff_params = params.replace(
        wind_x=params.wind_x + gust[0],
        wind_y=params.wind_y + gust[1],
        wind_z=params.wind_z + gust[2],
    )

    # Follower (the learner) — key=None so it does not advance its own per-plane
    # gust; the shared gust is already folded into eff_params.wind.
    power, stick, aileron = decode_action(action)
    new_follower, metrics = compute_next_state_3d(
        power,
        stick,
        aileron,
        state.follower,
        eff_params,
        integration_method=integration_method,
    )

    # Lead (scripted autopilot), same shared wind.
    new_lead, new_pid = step_lead(
        state, eff_params, lead_pid_params, integration_method
    )

    new_state = state.replace(
        follower=new_follower,
        lead=new_lead,
        lead_pid=new_pid,
        time=state.time + 1,
        gust_x=gust[0],
        gust_y=gust[1],
        gust_z=gust[2],
    )
    return new_state, metrics
