"""
Close-patrol (formation-keeping) environment: state, parameters and transition.

Two aircraft share the exact 3D physics of :mod:`target_gym.plane3d`:

  - the **lead** flies a scripted patrol route (a turn rate per leg) under the
    3D heading expert, on the speed schedule the 3D tasks fly;
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

import jax
import jax.numpy as jnp
from flax import struct

from target_gym import reward as R
from target_gym.base import EnvState, actuator_field
from target_gym.plane.dynamics import (
    BANK_DEMAND_MAX,
    PITCH_DEMAND_MAX,
    STICK_TRAVEL,
    advance_gust,
    scheduled_airspeed,
    step_key,
    total_wind_3d,
    trim_stick,
)
from target_gym.plane.experts_tecs import TECSParams
from target_gym.plane3d.dynamics import compute_velocity_3d, rolling_gust
from target_gym.plane3d.env import (
    PlaneParams3D,
    PlaneState3D,
    check_is_terminal_3d,
    compute_next_state_3d,
    get_obs_heading,
    wrap_angle,
)
from target_gym.plane3d.experts import (
    Plane3DGuidanceParams,
    Plane3DGuidanceState,
    plane3d_heading_step,
)
from target_gym.utils import log_scaled_reward

# ─── State & parameters ─────────────────────────────────


#: Legs in the lead's route, cycled. A patrol *is* a repeating circuit: fly a
#: leg, turn onto the next, fly it, turn again. Eight of them at the default
#: leg length gives a 200-step episode three turn onsets and a 1500-step one
#: about twenty-five.
#:
#: The lead used to draw one turn rate at reset and hold it for the whole
#: episode, which made the task unable to measure what it claimed. Tracking a
#: reference that rotates at a *constant, exactly observable* rate is cancelled
#: outright by feedforward -- with one added, the follower's settled error went
#: from 77.8 m at the hardest turn to 2.4 m, against a 3 m reward precision
#: floor -- and a planner has nothing to anticipate, because the future is a
#: linear extrapolation of the present at every instant.
#:
#: A route puts the difficulty back where the task says it is. Feedforward is
#: momentarily wrong at every turn onset, which is exactly the lag a follower
#: should have to work for, and a planner rolling ``step_env`` forward sees the
#: next turn coming because the schedule is in the state.
N_LEAD_LEGS = 8

#: The scripted lead: the 3D heading expert on its 2026-11-06 values, the values
#: ``plane3d.experts`` and ``plane.experts_tecs`` shipped when the lead moved onto
#: it. The lead is part of the plant, so they are task constants: a retune of that
#: expert does not reach this task, and moving them is a task change.
LEAD_AUTOPILOT = Plane3DGuidanceParams(
    max_bank_rad=45.0 * jnp.pi / 180.0,
    tau_hdg=10.0,
    alpha_max=0.19,
    k_bank_comp=0.15,
    cruise_cas=290.0 * 1852.0 / 3600.0,
    cruise_mach=0.78,
    v_min_eas=104.0,
    gravity=9.81,
    dt=1.0,
    tecs=TECSParams(
        climb_max=8.0,
        sink_min=23.0,
        sink_max=10.0,
        thr_trim=0.2,
        time_const=2.0,
        thr_damp=0.5,
        ptch_damp=1.0,
        integ_gain=0.1,
        spd_weight=0.0,
        roll_comp=10.0,
        thr_min=-1.0,
        thr_max=1.0,
        pitch_trim=0.0,
        pitch_max=0.26,
        accel_max=2.0,
        k_vdot=0.3,
        gravity=9.81,
        dt=1.0,
    ),
)

#: Samples of the lead's gust the air keeps, newest first. The largest V's last
#: rank sits 600 m back, 3.5 steps at the schedule's slowest 170 m/s.
GUST_MEMORY = 5


def gust_memory_field():
    """The state field holding the lead's recent gust, ``(GUST_MEMORY, 3)``."""
    return struct.field(
        default_factory=lambda: jnp.zeros((GUST_MEMORY, 3)), kw_only=True
    )


def own_gust_field(aircraft: int):
    """The state field holding each follower's own gust draw, ``(aircraft, 3)``."""
    return struct.field(default_factory=lambda: jnp.zeros((aircraft, 3)), kw_only=True)


def lead_leg(time, params, xp=jnp):
    """Which leg of the lead's route is being flown at *time*."""
    per_leg = xp.maximum(params.lead_leg_seconds / params.delta_t, 1.0)
    return ((time / per_leg).astype(int)) % N_LEAD_LEGS


@struct.dataclass
class PatrolState(EnvState):
    """Combined state of the two aircraft plus the slot definition.

    ``follower`` is the learner; ``lead`` is scripted.  ``lead_memory`` carries
    the lead autopilot's integrators, so it tracks its moving heading and its
    level without steady-state drift.
    The slot offsets are constant over an episode (sampled at reset) and are
    expressed in the lead body frame: *back* = metres behind the lead along
    its velocity, *right* = metres to the lead's right, *up* = metres above.
    """

    follower: PlaneState3D
    lead: PlaneState3D
    lead_memory: Plane3DGuidanceState
    slot_back: float
    slot_right: float
    slot_up: float
    lead_turn_rate: float  # rad per step commanded onto the lead heading, this leg
    #: Turn rate for each leg of the route, cycled. Held in the state, not
    #: derived from a key, so a planner that rolls the environment forward can
    #: read the lead's future out of the state it is handed.
    lead_turn_schedule: jnp.ndarray
    # The lead's OU gust over the last steps and the follower's own draw
    # (``advance_air``). Each aircraft holds the gust it flies in.
    gust_history: jax.Array = gust_memory_field()
    gust_own: jax.Array = own_gust_field(1)

    actuator_held: jax.Array = actuator_field(3)


@struct.dataclass
class PatrolParams(PlaneParams3D):
    """Physics params (inherited from :class:`PlaneParams3D`) + patrol config."""

    # Slot geometry sampled at reset (metres).  Defaults put the follower in a
    # classic trailing echelon: ~200 m back, ~120 m to one side, co-altitude.
    slot_back_range: Tuple[float, float] = (150.0, 300.0)
    slot_right_range: Tuple[float, float] = (-150.0, 150.0)
    slot_up_range: Tuple[float, float] = (-60.0, 60.0)

    # Reward / termination shaping.
    #: The formation tolerance, in metres: what "in the slot" means when a
    #: test or a report judges whether the follower held station. Not a reward
    #: parameter -- the reward is log-scaled against ``max_slot_error`` with a
    #: floor at ``slot_precision_floor``.
    slot_tolerance: float = 60.0
    # Heading-alignment tolerance (rad): the follower should fly roughly
    # parallel to the lead (like a real wingman), not merely occupy the slot
    # position.  30 deg sigma nudges toward parallel flight without dominating.
    heading_tolerance: float = 0.5236
    #: Collision distance between centres, in metres, a trip. Over the A320's
    #: 34.1 to 35.8 m span: at 25 m the wingtips of two aircraft abeam overlapped.
    min_separation: float = 40.0
    max_slot_error: float = 1500.0  # follower lost the formation (m) -> terminal
    #: Slot error below which the reward stops paying, in metres. Relative
    #: position between two aircraft comes from differencing GPS fixes, so a
    #: few metres is the honest resolution; asking for better is measuring
    #: noise.
    slot_precision_floor: float = 3.0

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Slot position, quadratic outside a tolerance of ``e_tol_slot`` (0
    # provisionally: the formation's station-keeping radius is a procedural
    # number to be supplied), normalised by the 3 m relative-GPS resolution
    # above; heading alignment with the lead, quadratic, normalised by the
    # inherited 0.0087 rad AHRS resolution. The oracle holds both below their
    # instruments in the test turbulence (scripts/measure_hold.py), so the
    # instruments set the scale. Losing the formation, a collision or a crash
    # costs, per step, twice the 1500 m slot-loss bound plus the heading term.
    #
    # patrol-v3: the slot floor was 18.6 m, an earlier planner's lowest
    # per-seed hold. The twin-autopilot
    # oracle that replaced it held 0.41 m under that floor and holds 0.10 m
    # under this one (lowest per-seed holds), so the floor is now the
    # instrument rather than the next oracle's hold.
    e_floor_slot: float = 3.0  # m, relative-GPS resolution (slot_precision_floor)
    e_tol_slot: float = 0.0  # provisional; station-keeping radius to be supplied
    # Overrides the inherited aircraft value: twice the slot-loss bound's cost.
    failure_cost: float = 2.0 * ((1500.0 / 3.0) ** 2 + (3.14159 / 0.0087) ** 2)
    #: Restart time priced into a trip (``reward.trip_cost``): a crash, collision
    #: or lost formation loses the sortie, 1 h of flight at 1 s steps (provisional).
    restart_steps: int = 3600
    #: Two terms (slot, heading). The oracle's best hold: the lowest per-seed hold
    #: cost the oracle demonstrated, in the reward's units (its slot hold in
    #: 3 m units plus its alignment hold in 0.5 deg units).
    rho_floor_tracking: float = (0.1021 / 3.0) ** 2 + (1.631e-3 / 0.0087) ** 2
    rho_floor: float = (0.1021 / 3.0) ** 2 + (1.631e-3 / 0.0087) ** 2

    # Lead behaviour.  Turn rate is sampled in [-r, r] rad/step; 0 => straight
    # and level.  At delta_t = 1 s, 0.03 rad/step is 1.7 deg/s, under the 3 deg/s
    # of a standard-rate turn: 28 deg of bank at 171 m/s and 34 deg at 219 m/s.
    lead_turn_rate_range: Tuple[float, float] = (-0.03, 0.03)
    #: Seconds the lead holds each leg of its route before turning onto the
    #: next. 60 s at the 1 s step, so the follower meets a turn onset roughly
    #: every minute rather than once per episode.
    lead_leg_seconds: float = 60.0

    # Follower is spawned near the slot with this much isotropic position noise
    # (m) so the episode starts solvable but not perfectly trimmed.
    follower_spawn_noise: float = 40.0

    delta_t: float = 1.0


#: Units of the PatrolParams fields, for the generated environment pages.
PARAM_UNITS = {
    "slot_tolerance": "m",
    "heading_tolerance": "rad",
    "min_separation": "m",
    "max_slot_error": "m",
    "slot_precision_floor": "m",
    "e_floor_slot": "m",
    "e_tol_slot": "m",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
    "lead_leg_seconds": "s",
    "follower_spawn_noise": "m",
    "delta_t": "s",
}


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
    trim = trim_stick(speed, z, theta, params)
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
        stick=trim + jnp.deg2rad(params.initial_stick),
        aileron=jnp.deg2rad(params.initial_aileron),
        fuel=params.initial_fuel_quantity,
        time=0,
        target_altitude=target_altitude,
        target_heading=target_heading,
        target_x=0.0,
        target_y=0.0,
        target_radius=0.0,
        pitch_integ=trim / STICK_TRAVEL,
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

    Terminal (irrecoverable) if either aircraft leaves the 3D tasks' envelope
    (altitude, bank, load factor, EAS, Mach), the two collide, or the follower
    falls out of formation past ``max_slot_error``.
    """
    follower_crash, _ = check_is_terminal_3d(state.follower, params, xp)
    lead_crash, _ = check_is_terminal_3d(state.lead, params, xp)
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


def compute_reward_terms_patrol(
    state: PatrolState, params: PatrolParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    slot = R.tracking_cost(
        slot_error(state), p.e_floor_slot, p.e_tol_slot, p.tracking_exponent, xp
    )
    heading = R.tracking_cost(
        wrap_angle(state.follower.psi - state.lead.psi),
        p.e_floor_heading,
        0.0,
        p.tracking_exponent,
        xp,
    )
    if tripped is None:
        tripped, _ = check_is_terminal_patrol(state, p, xp)
    terms = {
        "tracking": slot + heading,
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward_patrol_v1(state: PatrolState, params: PatrolParams, xp=jnp):
    """Slot-position tracking times heading alignment.

    The multiplicative heading factor makes the target "fly the slot *parallel*
    to the lead" rather than merely occupy the point.

    Tracking is log-scaled, as everywhere else in the suite. This was the last
    environment still using a Gaussian, ``exp(-0.5 (e/sigma)^2)``, and with a
    60 m sigma against a 1500 m terminal bound -- twenty-five sigma -- it was
    flat at 1.4e-6 from roughly 250 m outward. The whole reachable range beyond
    a couple of slot widths carried no gradient at all, which is the defect
    ``docs/reward-shaping.md`` records for every other plant that has since been
    migrated.

    That flatness is also a candidate explanation for the tuning pathology in
    D1 of PHYSICS.md: an objective that is a flat floor with a spike near zero
    is one whose landscape *would* look chaotic under small gain perturbations.
    Migrating it separates reward shape from guidance law as the cause, which
    the deviation could not do while both were suspect.

    No explicit crash penalty. Termination already costs the agent every step
    it would otherwise have earned, and since the reward is non-negative
    everywhere that is strictly worse than flying on. A large negative spike
    bought nothing the forgone reward did not, and left this family on a
    different contract from the twelve process plants, which have always relied
    on forgone reward alone.
    """
    err = slot_error(state)
    track_r = log_scaled_reward(
        err, params.slot_precision_floor, params.max_slot_error, xp
    )
    align_r = heading_alignment(state, params, xp)
    return track_r * align_r


# ─── Observations ───────────────────────────────────────


def compute_reward_patrol(state: PatrolState, params: PatrolParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_patrol_v1(state, params, xp),
        R.total(compute_reward_terms_patrol(state, params, xp), xp),
        xp,
    )


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


def get_obs_range_bearing(state: PatrolState, xp=jnp):
    """Partial (bearing + range) observation (21 values).

    A TCAS-style sensor on the follower yields only the range and the
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
    """Map a raw [-1, 1]^3 action to physical (lever[0,1], pitch[rad], bank[rad]).

    Not every learner squashes, so a raw action past the box acts as the bound
    rather than as a command the plant could never receive.
    """
    power, pitch, bank = jnp.clip(jnp.asarray(action), -1.0, 1.0)
    return (power + 1) / 2, pitch * PITCH_DEMAND_MAX, bank * BANK_DEMAND_MAX


# ─── The air ────────────────────────────────────────────


def lead_cruise_speed(z):
    """The true airspeed of the lead's schedule at height ``z``, m/s."""
    return scheduled_airspeed(z, LEAD_AUTOPILOT.cruise_cas, LEAD_AUTOPILOT.cruise_mach)


def air_speed(aircraft, gust, params):
    """An aircraft's speed through the air, in the gust ``gust``."""
    wind = total_wind_3d(aircraft.z, gust[0], gust[1], gust[2], params)
    return compute_velocity_3d(
        aircraft.x_dot - wind[0], aircraft.y_dot - wind[1], aircraft.z_dot - wind[2]
    )


def advance_air(history, own, back, right, up, speed, params, key, time):
    """One step of the air around the lead and its ``n`` followers.

    The lead draws an OU gust, ``history`` being its last ``GUST_MEMORY`` values,
    newest first. The field is frozen as it passes (Taylor), so a follower
    ``back`` metres behind meets that gust ``back / speed`` later, read between
    step samples. Dryden's correlation over its offset ``r = hypot(right, up)``
    from the lead's path leaves ``rho = exp(-r / L)`` of it shared, on MIL-F-8785C's
    scale ``L``; the rest is the follower's own OU draw, ``own`` of shape ``(n, 3)``,
    weighted so its gust keeps the lead's variance. The rolling gust's scale is the
    span, so every wing draws its own.

    ``key=None`` keeps a constant wind. Returns ``(history, own, gusts, rolls)``,
    ``gusts`` of shape ``(n + 1, 3)`` and ``rolls`` of shape ``(n + 1,)``, lead first.
    """
    step = step_key(key, time)

    def draw(gust, seat):
        if step is None:
            return gust, jnp.zeros(())
        gust_key, roll_key = jax.random.split(jax.random.fold_in(step, seat))
        theta, sigma = params.turbulence_theta, params.turbulence_sigma
        new = advance_gust(gust, theta, sigma, params.delta_t, gust_key)
        return new, rolling_gust(roll_key, speed, params)

    seats = jnp.arange(own.shape[0] + 1)
    drawn, rolls = jax.vmap(draw)(jnp.concatenate([history[:1], own]), seats)
    history = jnp.concatenate([drawn[:1], history[:-1]])
    delay = jnp.maximum(back, 0.0) / (jnp.maximum(speed, 1.0) * params.delta_t)
    sample = jnp.clip(jnp.floor(delay).astype(int), 0, GUST_MEMORY - 2)
    part = jnp.clip(delay - sample, 0.0, 1.0)[:, None]
    delayed = (1.0 - part) * history[sample] + part * history[sample + 1]
    rho = jnp.exp(-jnp.hypot(right, up) / params.turbulence_scale_length)[:, None]
    gusts = rho * delayed + jnp.sqrt(1.0 - rho**2) * drawn[1:]
    return history, drawn[1:], jnp.concatenate([drawn[:1], gusts]), rolls


def in_air(aircraft, gust):
    """``aircraft`` holding ``gust``. :func:`compute_next_state_3d` with no key
    flies in the gust it is given, and the trips read it."""
    return aircraft.replace(
        gust_x=gust[..., 0], gust_y=gust[..., 1], gust_z=gust[..., 2]
    )


# ─── Combined transition ────────────────────────────────


def step_lead(state: PatrolState, params: PatrolParams, method: str = "rk4_1"):
    """Advance the scripted lead one step under the 3D heading expert.

    The lead's commanded heading is stored in ``lead.target_heading`` and
    advanced by ``lead_turn_rate`` each step, producing straight-and-level
    flight (rate 0) or a turn. Returns (new_lead, new_lead_memory).
    """
    commanded_heading = wrap_angle(state.lead.target_heading + state.lead_turn_rate)
    lead_cmd = state.lead.replace(target_heading=commanded_heading)
    lead_action, memory = plane3d_heading_step(
        LEAD_AUTOPILOT, state.lead_memory, get_obs_heading(lead_cmd)
    )
    power, pitch, bank = decode_action(lead_action)
    new_lead, _ = compute_next_state_3d(
        power, pitch, bank, lead_cmd, params, integration_method=method
    )
    return new_lead, memory


def compute_next_state_patrol(
    action: jnp.ndarray,
    state: PatrolState,
    params: PatrolParams,
    integration_method: str = "rk4_1",
    key=None,
):
    """One environment step: advance the follower (from ``action``) and the lead.

    Each aircraft flies in its own gust and rolling gust (:func:`advance_air`) and
    its own altitude-dependent wind shear inside :func:`compute_next_state_3d`.
    ``key=None`` keeps a constant (steady) wind.
    """
    history, own, gusts, rolls = advance_air(
        state.gust_history,
        state.gust_own,
        jnp.atleast_1d(state.slot_back),
        jnp.atleast_1d(state.slot_right),
        jnp.atleast_1d(state.slot_up),
        air_speed(state.lead, state.gust_history[0], params),
        params,
        key,
        state.time,
    )

    power, pitch, bank = decode_action(action)
    new_follower, metrics = compute_next_state_3d(
        power,
        pitch,
        bank,
        in_air(state.follower, gusts[1]),
        params.replace(gust_roll_rate=params.gust_roll_rate + rolls[1]),
        integration_method=integration_method,
    )
    new_lead, memory = step_lead(
        state.replace(lead=in_air(state.lead, gusts[0])),
        params.replace(gust_roll_rate=params.gust_roll_rate + rolls[0]),
        integration_method,
    )

    new_state = state.replace(
        follower=new_follower,
        lead=new_lead,
        lead_memory=memory,
        # The leg live at the *next* step, so the rate the lead is commanded
        # with and the time it is commanded at stay in step.
        lead_turn_rate=state.lead_turn_schedule[lead_leg(state.time + 1, params)],
        time=state.time + 1,
        gust_history=history,
        gust_own=own,
    )
    return new_state, metrics
