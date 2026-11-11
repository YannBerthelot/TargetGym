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
import numpy as np
from flax import struct

from target_gym import reward as R
from target_gym.base import EnvState, actuator_field
from target_gym.plane.dynamics import (
    BANK_DEMAND_MAX,
    PITCH_DEMAND_MAX,
    STICK_TRAVEL,
    InnerLoop,
    advance_gust,
    compute_air_density_from_altitude,
    economy_cost,
    scheduled_airspeed,
    step_key,
    total_wind_3d,
    trim_level,
)
from target_gym.plane.experts_tecs import TECSParams
from target_gym.plane3d import hornet
from target_gym.plane3d.dynamics import compute_velocity_3d, rolling_gust
from target_gym.plane3d.env import (
    PlaneParams3D,
    PlaneState3D,
    airspeed_3d,
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
    # The Hornet lead flies AFF's Class B speed, 250 KCAS [AFF], the Mach
    # branch past it ours and never reached in the band.
    cruise_cas=hornet.CLASS_B_CAS,
    cruise_mach=0.70,
    # 1.2 times the Hornet's clean 1 g stall, 61.9 m/s EAS at its 15.1 t and
    # CL_max 1.70 [HARV] (ArduPilot's ARSPD_FBW_MIN rule [P409]).
    v_min_eas=74.0,
    gravity=9.81,
    dt=1.0,
    tecs=TECSParams(
        # Full military power climbs 25 m/s and idle sinks 15 m/s at the AFF
        # point (the model's own excess thrust, 23 kN and -12 kN, over 148 kN).
        climb_max=25.0,
        sink_min=15.0,
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

#: An airliner lead, for the wake-surfing task: the 3D heading expert on its
#: 2026-11-06 values (the lead the patrol flew before it moved to the Hornet).
AIRLINER_LEAD = LEAD_AUTOPILOT.replace(
    cruise_cas=290.0 * 1852.0 / 3600.0,
    cruise_mach=0.78,
    v_min_eas=104.0,
    tecs=LEAD_AUTOPILOT.tecs.replace(climb_max=8.0, sink_min=23.0),
)


def lead_autopilot(params):
    """The scripted lead's autopilot for the airframe ``params`` flies."""
    airliner = getattr(params, "airliner_lead", False)
    return jax.tree.map(
        lambda a, h: jnp.where(airliner, a, h), AIRLINER_LEAD, LEAD_AUTOPILOT
    )


#: Samples of the lead's gust and wake the air keeps, newest first: 20 s, so a
#: wake-surfing follower 3.7 km behind an airliner at 230 m/s (16 s) still finds
#: the air its lead left.
GUST_MEMORY = 20

#: What the lead leaves in the air each step: x, y, z, its right wing's horizontal
#: direction (2), its bank and its circulation (``wake_sample``).
WAKE_CHANNELS = 7
#: Stations across a follower's wing and their elliptic lift weights, Gauss-Chebyshev
#: of the second kind. 31 keep a span-wise core inside 0.2% of a fine integral.
_ANGLE = np.pi * np.arange(1, 32) / 32.0
_STATION = np.cos(_ANGLE) / 2.0
_LOAD = 2.0 / 32.0 * np.sin(_ANGLE) ** 2


def gust_memory_field():
    """The state field holding the lead's recent gust, ``(GUST_MEMORY, 3)``."""
    return struct.field(
        default_factory=lambda: jnp.zeros((GUST_MEMORY, 3)), kw_only=True
    )


def wake_memory_field():
    """The state field holding the lead's wake, ``(GUST_MEMORY, WAKE_CHANNELS)``."""
    return struct.field(
        default_factory=lambda: jnp.zeros((GUST_MEMORY, WAKE_CHANNELS)), kw_only=True
    )


def own_gust_field(aircraft: int):
    """The state field holding each follower's own gust draw, ``(aircraft, 3)``."""
    return struct.field(default_factory=lambda: jnp.zeros((aircraft, 3)), kw_only=True)


def lead_leg(time, params, xp=jnp):
    """Which leg of the lead's route is being flown at *time*."""
    per_leg = xp.maximum(params.lead_leg_seconds / params.delta_t, 1.0)
    # The clock wrapped on whole routes: the same count inside a window, and in
    # a stream a count float32 divides exactly however long the plant runs
    # (long-strata S1).
    route = xp.asarray(xp.round(per_leg * N_LEAD_LEGS), jnp.int32)
    return ((xp.mod(time, route) / per_leg).astype(int)) % N_LEAD_LEGS


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
    wake_history: jax.Array = wake_memory_field()

    actuator_held: jax.Array = actuator_field(3)


@struct.dataclass
class PatrolParams(PlaneParams3D):
    """Physics params (inherited from :class:`PlaneParams3D`, with the F/A-18
    Hornet's airframe, ``plane3d.hornet``) + patrol config."""

    # ---- The airframe: an F/A-18 Hornet (owner, 2026-11-10; hornet.py) ----
    initial_mass: float = hornet.MASS
    initial_fuel_quantity: float = hornet.FUEL
    wings_surface: float = hornet.WING_AREA
    wingspan: float = hornet.SPAN
    aspect_ratio: float = hornet.ASPECT_RATIO
    cl_alpha: float = hornet.CL_ALPHA
    cl0: float = hornet.CL0
    CL_max: float = hornet.CL_MAX
    aoa_stall: float = hornet.AOA_STALL
    sweep_half_chord: float = hornet.SWEEP_HALF_CHORD
    cd0: float = hornet.CD0
    k: float = hornet.K_INDUCED
    M_crit: float = hornet.M_CRIT
    pitch_gyration_radius: float = hornet.PITCH_GYRATION
    roll_gyration_radius: float = hornet.ROLL_GYRATION
    moment_arm_wings: float = hornet.WING_ARM
    moment_arm_stabilizer: float = hornet.TAIL_ARM
    stabilizer_surface: float = hornet.TAIL_AREA
    tail_cl_alpha: float = hornet.TAIL_CL_ALPHA
    tail_aspect_ratio: float = hornet.TAIL_ASPECT_RATIO
    tail_sweep_half_chord: float = hornet.TAIL_SWEEP_HALF_CHORD
    tail_k: float = 1.0 / (3.14159265 * hornet.TAIL_ASPECT_RATIO * 0.8)
    tail_efficiency: float = hornet.TAIL_EFFICIENCY
    elevator_effectiveness: float = 1.0
    downwash_ratio: float = hornet.DOWNWASH_RATIO
    stabilizer_setting: float = hornet.STABILIZER_SETTING
    C_lp: float = hornet.C_LP
    aileron_surface: float = hornet.AILERON_AREA
    aileron_effectiveness: float = hornet.AILERON_EFFECTIVENESS
    moment_arm_aileron: float = hornet.AILERON_ARM
    thrust_output_at_sea_level: float = hornet.THRUST_SEA_LEVEL
    specific_fuel_consumption: float = hornet.SFC
    power_response_rate: float = hornet.POWER_RESPONSE_RATE
    load_factor_max: float = hornet.LOAD_FACTOR_MAX
    load_factor_min: float = hornet.LOAD_FACTOR_MIN
    ap_load_increment: float = hornet.AP_LOAD_INCREMENT
    protected_bank: float = hornet.PROTECTED_BANK
    mmo: float = hornet.MMO
    vmo_eas: float = hornet.VMO_EAS
    #: The Hornet's attitude loop, tuned once by the A320's procedure at the
    #: AFF point, M 0.56 at 25 000 ft (``scripts/tune_inner_loop.py --airframe
    #: hornet``), its scaler about that point's 116 m/s EAS. Its roll D is
    #: capped at the Hornet's own roll damper, not the guide's 0.1 [HARV IV.A.2].
    inner_loop: InnerLoop = InnerLoop(
        Kp_theta=2.2918,
        Ki_theta=0.0318,
        Kd_theta=0.1273,
        Kp_roll=0.6366,
        Ki_roll=0.0318,
        Kd_roll=0.3438,
        scaling_speed=116.08,
    )
    #: The plant every scored aircraft task flies: limits protected, the
    #: ground the one trip (plane3d/PHYSICS.md, Amendment 2).
    amended: bool = True
    #: The AFF test band, 20 000 to 32 000 ft [AFF].
    target_altitude_range: Tuple[float, float] = hornet.ALTITUDE_BAND
    initial_altitude_range: Tuple[float, float] = hornet.ALTITUDE_BAND
    #: Light turbulence, as AFF flew ("no turbulence" to "light chop" [AFF]):
    #: the OU innovation that holds a 0.6 m/s gust (ours, the 3D tasks' light).
    turbulence_sigma: float = 0.36

    # Slot geometry sampled at reset (metres): the cruise position, on a
    # bearing line 3 to 6 plane widths out with a step down [CNATRA, cruise
    # position], never closer than AFF's 56 ft nose to tail, one length, so
    # 34 m centre to centre at least [AFF]. The lateral offset's size is drawn
    # in ``slot_right_range`` and its side at random.
    slot_back_range: Tuple[float, float] = (34.0, 70.0)
    slot_right_range: Tuple[float, float] = (17.0, 45.0)
    slot_up_range: Tuple[float, float] = (-10.0, 0.0)

    # Reward / termination shaping.
    #: The formation tolerance, in metres: what "in the slot" means when a
    #: test or a report judges whether the follower held station. Not a reward
    #: parameter -- the reward is log-scaled against ``max_slot_error`` with a
    #: floor at ``slot_precision_floor``.
    slot_tolerance: float = 10.0 * 0.3048  # 10 ft one sigma [AFF]
    # Heading-alignment tolerance (rad): the follower should fly roughly
    # parallel to the lead (like a real wingman), not merely occupy the slot
    # position.  30 deg sigma nudges toward parallel flight without dominating.
    heading_tolerance: float = 0.5236
    #: Collision distance between centres, in metres, a trip: one Hornet
    #: length, 56 ft, the nose-to-tail limit AFF flew to [AFF].
    min_separation: float = 56.0 * 0.3048
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
    # and level.
    #: 0.0303 rad/s is the parade turn's 30 deg of bank [CNATRA] at the lead's
    #: 250 KCAS at 25 000 ft (186.9 m/s): 28 deg at the band's foot and 33 deg at
    #: its top, inside the 45 deg the lead's and the wingmen's autopilots fly.
    lead_turn_rate_range: Tuple[float, float] = (-0.0303, 0.0303)
    #: Seconds the lead holds each leg of its route before turning onto the
    #: next. 60 s at the 1 s step, so the follower meets a turn onset roughly
    #: every minute rather than once per episode.
    lead_leg_seconds: float = 60.0

    # The lead's wake on its followers (``wake_air``, PHYSICS.md section 1).
    #: Scales the wake: 0 is the plant without it, how a run measures what it costs.
    wake_strength: float = 1.0
    #: Burnham core radius over the span, the 0.0582 of NASA's wake model review.
    wake_core_ratio: float = 0.0582
    #: The circulation's decay per ``t0 = 2 pi b0^2 / Gamma``: our fit, ``exp(-0.2 t /
    #: t0)``, to the lidar median of Holzapfel and Steen (2007), figure 3.
    wake_decay_rate: float = 0.2

    # Follower is spawned near the slot with this much isotropic position noise
    # (m) so the episode starts solvable but not perfectly trimmed.
    follower_spawn_noise: float = 10.0  # ours: inside the cruise position

    #: Fly the scripted lead on the airliner's autopilot (``AIRLINER_LEAD``),
    #: for an A320 lead (``wake_surf``).
    airliner_lead: bool = False
    #: Bank past which the follower is upset, a trip, rad; 0 leaves it out. A
    #: bank of more than 60 deg en route is a Hazardous effect [EASA AMC
    #: 25.1329, A4.2]; ``wake_surf`` sets it, where a vortex can roll the
    #: follower past what its ailerons hold.
    upset_bank: float = 0.0
    #: A sortie's all-up mass in a stream, kg (long-strata S1b): the patrol's
    #: zero-fuel mass with a quarter of the internal 4 926 kg to all of it
    #: aboard (ours; the scored window carries 3 000 kg, ``hornet.FUEL``).
    sortie_mass_range: Tuple[float, float] = (
        hornet.MASS - hornet.FUEL + 0.25 * 4_926.0,
        hornet.MASS - hornet.FUEL + 4_926.0,
    )

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
    "wake_strength": "-",
    "wake_core_ratio": "-",
    "wake_decay_rate": "-",
    "follower_spawn_noise": "m",
    "airliner_lead": "-",
    "upset_bank": "rad",
    "delta_t": "s",
}


_A320 = PlaneParams3D


@struct.dataclass
class WakeSurfParams(PatrolParams):
    """``wake_surf``: an A320 follower surfing an A320 lead's wake (owner,
    2026-11-10), wake-energy retrieval as Airbus's fello'fly flew it, two A350s
    3 km apart, and the C-17 $AVE trials 3 000 to 6 000 ft in trail.

    The airframe is the A320 of the transport tasks (``PlaneParams3D``), its
    attitude loop and protections included. The station sits where the
    model's own wake gives the most upwash for a level lead: 0.87 of a span
    outboard, NASA's 13 % tip overlap, and below by the vortex's sink, 21 m at
    3 km (plane3d/PHYSICS.md, wake surfing).
    """

    initial_mass: float = _A320.initial_mass
    initial_fuel_quantity: float = _A320.initial_fuel_quantity
    wings_surface: float = _A320.wings_surface
    wingspan: float = _A320.wingspan
    aspect_ratio: float = _A320.aspect_ratio
    cl_alpha: float = _A320.cl_alpha
    cl0: float = _A320.cl0
    CL_max: float = _A320.CL_max
    aoa_stall: float = _A320.aoa_stall
    sweep_half_chord: float = _A320.sweep_half_chord
    cd0: float = _A320.cd0
    k: float = _A320.k
    M_crit: float = _A320.M_crit
    pitch_gyration_radius: float = _A320.pitch_gyration_radius
    roll_gyration_radius: float = _A320.roll_gyration_radius
    moment_arm_wings: float = _A320.moment_arm_wings
    moment_arm_stabilizer: float = _A320.moment_arm_stabilizer
    stabilizer_surface: float = _A320.stabilizer_surface
    tail_cl_alpha: float = _A320.tail_cl_alpha
    tail_aspect_ratio: float = _A320.tail_aspect_ratio
    tail_sweep_half_chord: float = _A320.tail_sweep_half_chord
    tail_k: float = _A320.tail_k
    tail_efficiency: float = _A320.tail_efficiency
    elevator_effectiveness: float = _A320.elevator_effectiveness
    downwash_ratio: float = _A320.downwash_ratio
    stabilizer_setting: float = _A320.stabilizer_setting
    C_lp: float = _A320.C_lp
    aileron_surface: float = _A320.aileron_surface
    aileron_effectiveness: float = _A320.aileron_effectiveness
    moment_arm_aileron: float = _A320.moment_arm_aileron
    thrust_output_at_sea_level: float = _A320.thrust_output_at_sea_level
    specific_fuel_consumption: float = _A320.specific_fuel_consumption
    power_response_rate: float = _A320.power_response_rate
    load_factor_max: float = _A320.load_factor_max
    load_factor_min: float = _A320.load_factor_min
    ap_load_increment: float = _A320.ap_load_increment
    protected_bank: float = _A320.protected_bank
    mmo: float = _A320.mmo
    vmo_eas: float = _A320.vmo_eas
    inner_loop: InnerLoop = InnerLoop()
    airliner_lead: bool = True
    sortie_mass_range: Tuple[float, float] = _A320.sortie_mass_range

    #: Transatlantic cruise levels, FL330 to FL370 (ours, the A320's cruise).
    target_altitude_range: Tuple[float, float] = (33_000.0 * 0.3048, 37_000.0 * 0.3048)
    initial_altitude_range: Tuple[float, float] = (33_000.0 * 0.3048, 37_000.0 * 0.3048)
    #: The station: 2.8 to 3.7 km in trail (fello'fly's 3 km; ours, about it),
    #: 0.87 of the span outboard on a side drawn at random, and the vortex's
    #: sink below, 19 to 26 m over 12 to 16 s.
    slot_back_range: Tuple[float, float] = (2_800.0, 3_700.0)
    slot_right_range: Tuple[float, float] = (31.0, 31.0)
    slot_up_range: Tuple[float, float] = (-26.0, -19.0)
    #: The box about the station, m (ours, a span: no public tolerance for a
    #: fello'fly station was found). Inside it only the economy counts.
    e_floor_slot: float = 30.0
    e_tol_slot: float = 30.0
    slot_tolerance: float = 30.0
    min_separation: float = 40.0  # over the A320's span
    follower_spawn_noise: float = 10.0
    #: Straight and level, as fello'fly's oceanic track (ours): the station is
    #: held in the lead's frame, and in a turn the wake stays on the lead's
    #: past track, 3.7 deg of turn and 100 m off that frame at 0.3 deg/s
    #: (plane3d/PHYSICS.md, known deviations).
    lead_turn_rate_range: Tuple[float, float] = (0.0, 0.0)
    lead_leg_seconds: float = 300.0
    upset_bank: float = 60.0 * 3.14159265 / 180.0
    economy_weight: float = 1.0
    rho_floor_tracking: float = 0.0
    rho_floor: float = 0.0
    #: As ``plane``: the ground and an upset are the trips, priced by us.
    failure_cost: float = ((12192.0 - 19.812) / 19.812) ** 2


def make_cruise_state(
    x: float,
    y: float,
    z: float,
    psi: float,
    speed: float,
    params: PatrolParams,
    target_altitude: float,
    target_heading: float,
    mass=None,
    fuel=None,
) -> PlaneState3D:
    """Build a trimmed, straight-and-level :class:`PlaneState3D` at cruise.

    Used to spawn both aircraft at reset: level flight, wings level, velocity
    of magnitude ``speed`` along heading ``psi``, in level trim (``trim_level``:
    no net force, no pitching moment, the power that holds the speed), with the
    attitude loop's integral at that trim. Spawning at ``initial_theta`` with the
    elevator merely balanced there left a Hornet, which trims near 6 deg, 6 deg
    short of the lift it needs.

    ``mass`` and ``fuel`` default to the scored configuration's; a stream's
    sortie passes its own (long-strata S1b).
    """
    mass = params.initial_mass if mass is None else mass
    fuel = params.initial_fuel_quantity if fuel is None else fuel
    x_dot = speed * jnp.cos(psi)
    y_dot = speed * jnp.sin(psi)
    theta, trim, power = trim_level(speed, z, mass, params)
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
        m=mass,  # fuel is a component of it, not an addition
        power=power,
        stick=trim,
        aileron=jnp.deg2rad(params.initial_aileron),
        fuel=fuel,
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
    upset = (params.upset_bank > 0.0) & (xp.abs(state.follower.phi) > params.upset_bank)
    terminated = follower_crash | lead_crash | collision | lost | upset
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


def compute_reward_terms_wake_surf(
    state: PatrolState, params: PatrolParams, xp=jnp, tripped=None
):
    """``wake_surf``'s cost terms, each >= 0: the station box, and what the
    follower's operator pays, fuel and time by the cost index per air distance
    (``plane.env.economy_cost``). Inside the box the economy is all that counts,
    so the cost falls where the follower finds the lead's upwash."""
    p = params
    box = R.tracking_cost(slot_error(state), p.e_floor_slot, p.e_tol_slot, 2.0, xp)
    follower = state.follower
    terms = {
        "tracking": box,
        "economy": economy_cost(
            follower.power,
            follower.z,
            airspeed_3d(follower, p, xp),
            lead_cruise_speed(follower.z, p),
            p,
            xp,
        ),
    }
    if tripped is None:
        tripped, _ = check_is_terminal_patrol(state, p, xp)
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


def compute_reward_v1(state: PatrolState, params: PatrolParams, xp=jnp):
    """Version 1 of the package's tasks, by the name TargetFoundation looks up:
    the patrol's log-scaled reward (``compute_reward_patrol_v1``), which
    ``wake_surf``, the task added to the package, shares."""
    return compute_reward_patrol_v1(state, params, xp)


def compute_reward_wake_surf(state: PatrolState, params: PatrolParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms_wake_surf(state, params, xp), xp),
        xp,
    )


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

    The wingman's sight of the lead (its range) and a TCAS-style sensor yield
    only the range and the azimuth/elevation *bearing* to the lead — no lead
    velocity, heading or the decomposed slot error.  Recovering the slot then requires the follower to
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


def lead_cruise_speed(z, params=None):
    """The true airspeed of the lead's schedule at height ``z``, m/s."""
    lead = LEAD_AUTOPILOT if params is None else lead_autopilot(params)
    return scheduled_airspeed(z, lead.cruise_cas, lead.cruise_mach)


def air_speed(aircraft, gust, params):
    """An aircraft's speed through the air, in the gust ``gust``."""
    wind = total_wind_3d(aircraft.z, gust[0], gust[1], gust[2], params)
    return compute_velocity_3d(
        aircraft.x_dot - wind[0], aircraft.y_dot - wind[1], aircraft.z_dot - wind[2]
    )


def slot_delay(back, speed, params):
    """Steps a follower ``back`` metres behind lags the lead's air by."""
    return jnp.maximum(back, 0.0) / (jnp.maximum(speed, 1.0) * params.delta_t)


def at_delay(history, delay):
    """``history``, newest first, read ``delay`` steps ago between its samples."""
    sample = jnp.clip(jnp.floor(delay).astype(int), 0, GUST_MEMORY - 2)
    part = jnp.clip(delay - sample, 0.0, 1.0)[:, None]
    return (1.0 - part) * history[sample] + part * history[sample + 1]


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
    delayed = at_delay(history, slot_delay(back, speed, params))
    rho = jnp.exp(-jnp.hypot(right, up) / params.turbulence_scale_length)[:, None]
    gusts = rho * delayed + jnp.sqrt(1.0 - rho**2) * drawn[1:]
    return history, drawn[1:], jnp.concatenate([drawn[:1], gusts]), rolls


def wake_sample(lead, lift, speed, params):
    """What the lead leaves in the air: where it is, which way its wing lies, and the
    circulation of a pair ``b0 = pi b / 4`` apart that carries its ``lift``."""
    rho = compute_air_density_from_altitude(lead.z)
    gamma = lift / (rho * speed * jnp.pi / 4.0 * params.wingspan)
    right = (jnp.sin(lead.psi), -jnp.cos(lead.psi))
    wake = (lead.x, lead.y, lead.z, *right, lead.phi, params.wake_strength * gamma)
    return jnp.stack(wake)


def shed_wake(history, sample):
    """``history`` with ``sample`` as its newest entry."""
    return jnp.concatenate([sample[None], history[:-1]])


def initial_wake(lead, params):
    """The wake of a lead already in steady level flight, laid out behind it."""
    speed = compute_velocity_3d(lead.x_dot, lead.y_dot, lead.z_dot)
    sample = wake_sample(lead, lead.m * params.gravity, speed, params)
    velocity = jnp.zeros(WAKE_CHANNELS).at[:2].set(jnp.stack([lead.x_dot, lead.y_dot]))
    steps = jnp.arange(GUST_MEMORY, dtype=jnp.float32)[:, None]
    return sample - steps * params.delta_t * velocity


def wake_air(history, x, y, z, delay, params):
    """The lead's wake on ``n`` followers: ``(gusts of shape (n, 3), rolling gusts)``.

    Blake and Multhopp's horseshoe vortex: two legs ``b0`` apart with Burnham's core,
    far enough back to read as straight lines in the follower's cross-plane. The
    wake lies where the lead was ``delay`` steps ago, tilted as it was banked, and has
    sunk along its lift at ``w0 = Gamma / (2 pi b0)`` by then, with its circulation
    decayed on ``t0 = b0 / w0``.

    Dogan's effective wind and gradient, weighted by a follower's elliptic lift: the
    upwash and sidewash go in as a gust, so the lift rotates forward by ``w / V`` and
    the drag saving needs no term of its own. The gradient goes in as the plant's
    ``p_g``: ``16 / b^2`` of the first moment, which gives back ``p`` from ``w = p y``.
    Each is read across a wing lying along the pair. A follower past the lead is clear.
    """
    px, py, pz, rx, ry, phi, gamma = jnp.moveaxis(at_delay(history, delay), -1, 0)
    span = params.wingspan
    b0 = jnp.pi / 4.0 * span
    sink = gamma / (2.0 * jnp.pi * b0) * delay * params.delta_t
    gamma = gamma * jnp.exp(-params.wake_decay_rate * sink / b0)
    across, height = (x - px) * rx + (y - py) * ry, z - pz
    cos, sin = jnp.cos(phi), jnp.sin(phi)
    lateral = (across * cos + height * sin)[:, None] + span * _STATION
    normal = (height * cos - across * sin + sink)[:, None]
    upwash, sidewash = 0.0, 0.0
    for leg, turn in ((b0 / 2.0, 1.0), (-b0 / 2.0, -1.0)):
        offset = lateral - leg
        radius2 = offset**2 + normal**2 + (params.wake_core_ratio * span) ** 2
        upwash = upwash + turn * offset / radius2
        sidewash = sidewash - turn * normal / radius2
    strength = gamma / (2.0 * jnp.pi)
    w = strength * jnp.sum(_LOAD * upwash, axis=-1)
    v = strength * jnp.sum(_LOAD * sidewash, axis=-1)
    roll = strength * 16.0 / span * jnp.sum(_LOAD * _STATION * upwash, axis=-1)
    side = v * cos - w * sin
    gusts = jnp.stack([side * rx, side * ry, v * sin + w * cos], axis=-1)
    now = history[0]
    ahead = (x - now[0]) * -now[4] + (y - now[1]) * now[3] > 0.0
    return jnp.where(ahead[:, None], 0.0, gusts), jnp.where(ahead, 0.0, roll)


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
    flight (rate 0) or a turn. Returns (new_lead, new_lead_memory, its lift).
    """
    commanded_heading = wrap_angle(state.lead.target_heading + state.lead_turn_rate)
    lead_cmd = state.lead.replace(target_heading=commanded_heading)
    lead_action, memory = plane3d_heading_step(
        lead_autopilot(params), state.lead_memory, get_obs_heading(lead_cmd)
    )
    power, pitch, bank = decode_action(lead_action)
    new_lead, metrics = compute_next_state_3d(
        power, pitch, bank, lead_cmd, params, integration_method=method
    )
    return new_lead, memory, metrics[1][-1]


def compute_next_state_patrol(
    action: jnp.ndarray,
    state: PatrolState,
    params: PatrolParams,
    integration_method: str = "rk4_1",
    key=None,
):
    """One environment step: advance the follower (from ``action``) and the lead.

    Each aircraft flies in its own gust and rolling gust (:func:`advance_air`) and
    its own altitude-dependent wind shear inside :func:`compute_next_state_3d`, and
    the follower in the lead's wake (:func:`wake_air`). ``key=None`` keeps a constant
    (steady) wind.
    """
    speed = air_speed(state.lead, state.gust_history[0], params)
    back = jnp.atleast_1d(state.slot_back)
    history, own, gusts, rolls = advance_air(
        state.gust_history,
        state.gust_own,
        back,
        jnp.atleast_1d(state.slot_right),
        jnp.atleast_1d(state.slot_up),
        speed,
        params,
        key,
        state.time,
    )
    wake, wake_rolls = wake_air(
        state.wake_history,
        jnp.atleast_1d(state.follower.x),
        jnp.atleast_1d(state.follower.y),
        jnp.atleast_1d(state.follower.z),
        slot_delay(back, speed, params),
        params,
    )

    power, pitch, bank = decode_action(action)
    new_follower, metrics = compute_next_state_3d(
        power,
        pitch,
        bank,
        in_air(state.follower, gusts[1] + wake[0]),
        params.replace(gust_roll_rate=params.gust_roll_rate + rolls[1] + wake_rolls[0]),
        integration_method=integration_method,
    )
    new_lead, memory, lift = step_lead(
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
        wake_history=shed_wake(
            state.wake_history, wake_sample(new_lead, lift, speed, params)
        ),
    )
    return new_state, metrics
