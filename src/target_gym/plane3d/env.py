"""
3D airplane environment state, parameters, and transition logic.

The physics (state transition) are shared across all 3D tasks.
Task-specific reward, observation, and reset logic live in the
individual environment classes (env_jax.py).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.plane.dynamics import (
    AILERON_TRAVEL,
    INNER_STEPS,
    STICK_RESPONSE_RATE,
    STICK_TRAVEL,
    InnerLoop,
    advance_dryden_gust,
    advance_gust,
    airspeed_scaler,
    attitude_axis,
    compute_air_density_from_altitude,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_surface,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    economy_cost,
    engine_response_rate,
    in_zone,
    outside_envelope,
    protected_pitch_demand,
    scheduled_airspeed,
    speed_band_excess,
    step_key,
    total_wind_3d,
)
from target_gym.plane3d.dynamics import (
    compute_acceleration_3d,
    compute_alpha_3d,
    compute_psi,
    compute_velocity_3d,
    rolling_gust,
)
from target_gym.utils import log_scaled_reward


@struct.dataclass
class PlaneState3D(EnvState):
    x: float
    x_dot: float
    y: float
    y_dot: float
    z: float
    z_dot: float
    theta: float
    theta_dot: float
    phi: float
    phi_dot: float
    psi: float
    alpha: float
    gamma: float
    m: float
    power: float
    stick: float
    aileron: float
    fuel: float
    # Task targets — semantics depend on the task variant
    target_altitude: float
    target_heading: (
        float  # heading: desired heading; fig8: orientation angle; circle: unused
    )
    target_x: float  # circle/fig8: center x; heading: unused (0)
    target_y: float  # circle/fig8: center y; heading: unused (0)
    target_radius: float  # circle: radius; fig8: lobe radius; heading: unused (0)
    # Ornstein-Uhlenbeck turbulence gust (m/s); total wind = params.wind_* + gust.
    gust_x: float = 0.0
    gust_y: float = 0.0
    gust_z: float = 0.0
    # The inner loop's integrals, in surface units.
    pitch_integ: float = 0.0
    roll_integ: float = 0.0
    # The step's load factor furthest from 1 g, which the limits read.
    load_factor: float = 1.0
    # Amendment 2 (``params.amended``; plane3d/PHYSICS.md): the racetrack's
    # half leg, m; the heading clearance's size, rad; the normal load factor
    # an accelerometer reads at the end of the step; the turbulence zone, s.
    target_half_leg: float = 0.0
    heading_change: float = 0.0
    n_z: float = 1.0
    zone_start: float = 0.0
    zone_end: float = 0.0
    #: The step of the heading clearance in force (Amendment 3).
    clearance_time: float = 0.0

    actuator_held: jax.Array = actuator_field(3)

    @property
    def rho(self):
        return compute_air_density_from_altitude(self.z)

    @property
    def speed_of_sound(self):
        return compute_speed_of_sound_from_altitude(self.z)

    @property
    def M(self):
        return compute_Mach_from_velocity_and_speed_of_sound(
            compute_velocity_3d(self.x_dot, self.y_dot, self.z_dot),
            self.speed_of_sound,
        )


@struct.dataclass
class PlaneParams3D(EnvParams):
    gravity: float = 9.81
    initial_mass: float = 73_500.0
    thrust_output_at_sea_level: float = 240_000.0
    air_density_at_sea_level: float = 1.225
    frontal_surface: float = 12.6
    wings_surface: float = 122.6
    C_x0: float = 0.095
    C_z0: float = 0.9
    initial_fuel_quantity: float = 23860 / 1.25
    specific_fuel_consumption: float = 17.5 / 1000

    # PlaneParams3D does not inherit from PlaneParams, so the engine, the
    # surfaces and the inner loop are declared here as well; the values match
    # the 2D aircraft's.
    # A 1.7 s engine above 5% idle. 14 CFR 33.73(b) gives idle to 95% in 5 s, and
    # a first-order lag takes ln 19 of its time constant to do that (5.006 s).
    power_response_rate: float = 1.0 / 1.7
    idle_power: float = 0.05
    stick_response_rate: float = STICK_RESPONSE_RATE
    aileron_response_rate: float = STICK_RESPONSE_RATE
    surface_rate_limit: float = jnp.pi / 6.0
    inner_loop: InnerLoop = InnerLoop()
    # The envelope (PHYSICS.md section 3): the limit load factors of 14 CFR
    # 25.337, the A320's VMO 350 kt and MMO 0.82, and the buffet line, CL_max
    # falling from ``buffet_mach`` at ``buffet_slope`` per unit of Mach (ours).
    load_factor_max: float = 2.5
    load_factor_min: float = -1.0
    ap_load_increment: float = 0.2
    vmo_eas: float = 350.0 * 1852.0 / 3600.0
    mmo: float = 0.82
    buffet_mach: float = 0.3
    buffet_slope: float = 1.4
    # 14 CFR 91.307(c)'s bank limit, 60 deg: the demand stops at 45.
    bank_limit: float = jnp.pi / 3.0

    # span^2 / S for the A320. PlaneParams3D does not inherit from PlaneParams,
    # so the shared aerodynamics needs its own copy; the value matches.
    aspect_ratio: float = 9.48

    # Control-surface effectiveness (the flap parameter tau). A surface
    # deflected by delta does not change the section's incidence by delta: only
    # the hinged part of the chord turns, so the effective change is tau*delta.
    # Thin-aerofoil theory gives tau ~ 0.4 for a control surface occupying a
    # quarter of the chord, which is the usual aileron proportion.
    #
    # Without it the model applied the *wing's* lift-curve slope directly to the
    # deflection, implying a section lift change of 2.20 at full aileron --
    # larger than the whole wing's CL_max of 1.5, and giving a roll rate of
    # 84 deg/s against an A320's 25-30.
    aileron_effectiveness: float = 0.4

    # Aero coefficients and the longitudinal airframe: the 2D aircraft's, read
    # from ``PlaneParams`` so the two cannot drift (plane/PHYSICS.md 3 and 4).
    cl_alpha: float = 0.08378
    sweep_half_chord: float = 0.38186
    cl0: float = 0.2
    cd0: float = 0.02
    k: float = 0.045
    aoa_stall: float = 15.5
    CL_max: float = 1.5
    # Negative-stall limit. A cambered wing stalls asymmetrically: the negative
    # branch bottoms out near -10 deg AoA. Without this the corrected (steeper)
    # lift slope lets CL run away negative -- it was previously masked by the
    # too-shallow slope, not actually bounded.
    CL_min: float = -1.0
    # Degrees beyond `aoa_stall` at which the separation sigmoid is centred, so
    # that lift *peaks at* aoa_stall rather than being halved there.
    aoa_stall_width: float = 3.0
    M_crit: float = 0.8
    # Shock stall above M_crit (PHYSICS.md D3): the attainable lift falls
    # once shocks form, rather than rising with the Prandtl-Glauert factor.
    # k = 4.0 puts CL_max back to its low-speed value by M 0.9 and at 40 %
    # of it by M 0.95, which is the right order for lift divergence.
    k_shock_stall: float = 4.0
    shock_stall_floor: float = 0.25
    k_drag: float = 5.0

    pitch_gyration_radius: float = 8.666
    #: Roll radius of gyration, m: JSBSim's A320, 1.28e6 kg m^2 over its empty
    #: 50.3 t, applied at the all-up mass (1.9e6 at 73.5 t).
    roll_gyration_radius: float = 5.039
    moment_arm_stabilizer: float = 13.36
    moment_arm_wings: float = 0.822
    stabilizer_surface: float = 31.0
    tail_cl_alpha: float = 0.07021
    tail_aspect_ratio: float = 5.0
    tail_sweep_half_chord: float = 0.411
    tail_k: float = 0.0796
    tail_CL_max: float = 1.0
    tail_aoa_stall: float = 15.0
    tail_efficiency: float = 0.9
    elevator_effectiveness: float = 0.5
    downwash_ratio: float = 1.0
    stabilizer_setting: float = 0.02417

    # Roll-specific A320 parameters
    wingspan: float = 35.8
    aileron_surface: float = 6.0
    moment_arm_aileron: float = 14.0
    C_lp: float = -0.4  # roll damping derivative

    max_steps_in_episode: int = 10_000
    min_alt: float = 0.0
    max_alt: float = 40_000.0 / 3.281
    target_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    target_heading_range: Tuple[float, float] = (-3.14159, 3.14159)
    target_radius_range: Tuple[float, float] = (8_000.0, 12_000.0)  # m

    # Reward precision floors -- the finest error each sensor can resolve.
    # Below these the reward saturates, because "improving" further is noise.
    precision_floor: float = 1.0  # m, barometric altimeter resolution
    heading_precision_floor: float = 0.0087  # rad (~0.5 deg), AHRS/compass
    position_precision_floor: float = 3.0  # m, civil GPS horizontal accuracy

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # One quadratic term per tracked output, summed, each normalised by its
    # floor. The test configurations fly in the turbulence below. Each 3D
    # task's registry entry sets its own floors: they
    # are the instrument resolutions above (1 m altitude,
    # 0.0087 rad heading, 3 m path), which the oracle holds below on every
    # task (`scripts/measure_hold.py`; the holds are in PHYSICS.md). No
    # tolerance and no running cost on these tasks. Leaving the envelope
    # costs, per step, twice the altitude envelope's cost (on the figure-8,
    # which scores no altitude, twice the path's).
    #
    # The class defaults below are plane3d_heading-v2's floors, set from an
    # earlier oracle's holds. They stay because PatrolParams inherits them:
    # patrol scores neither altitude nor path, but these values are in its
    # fingerprint, so moving them would re-version patrol for nothing.
    reward_version: int = 2
    e_floor_altitude: float = 1.44  # m, heading-v2's; the 3D tasks set 1.0
    e_tol_altitude: float = 0.0  # m; a +-30 m band made the altitude hold vacuous
    e_floor_heading: float = 0.0087  # rad, 0.5 deg AHRS resolution
    e_floor_path: float = 3.0  # m, civil GPS accuracy; the path tasks set it too
    tracking_exponent: float = 2.0
    failure_cost: float = 2.0 * (12192.0 / 1.44) ** 2  # per task in the registry
    #: Restart time priced into a trip (``reward.trip_cost``): a crash loses the
    #: sortie, 1 h of flight at 1 s steps (provisional).
    restart_steps: int = 3600
    #: The oracle's best hold: the lowest per-seed hold cost the oracle
    #: demonstrated, in the reward's units. These defaults are heading-v2's
    #: (altitude at its floor, an earlier oracle's 1e-4 rad heading hold in
    #: 0.5 deg units); each 3D task sets its own in the registry.
    rho_floor_tracking: float = 1.0 + (1.0e-4 / 0.0087) ** 2
    rho_floor: float = 1.0 + (1.0e-4 / 0.0087) ** 2
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False
    # Figure-8: half-amplitude of the altitude twist (meters).  The curve
    # altitude is z_mean ± this value, so the two crossover passes differ
    # by 2× this.  200 m ≈ 660 ft — gentle enough for an A320 but enough
    # to require coordinated altitude+heading control.
    figure8_altitude_amplitude: float = 200.0
    # Random orientation of the lemniscate (radians).  ±15° by default.
    figure8_angle_range: Tuple[float, float] = (-0.26, 0.26)
    initial_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    #: How far from the commanded altitude the episode may start. The initial
    #: altitude used to be drawn independently of the target over the same
    #: 5 000 m band, so the median start was 1 590 m off its assigned level and
    #: the episode opened with several minutes of climb before the pattern
    #: could be flown at all. An aircraft handed a heading, a circle or a hold
    #: is at or near the level it was given. ``initial_altitude_range`` still
    #: bounds the result.
    initial_altitude_offset_range: Tuple[float, float] = (-600.0, 600.0)
    initial_z_dot: float = 0.0
    initial_x_dot: float = 200.0
    initial_y_dot: float = 0.0
    initial_theta_dot: float = 0.0
    initial_theta: float = 0.0
    initial_phi: float = 0.0
    initial_phi_dot: float = 0.0
    initial_heading: float = 0.0
    initial_power: float = 1.0
    initial_stick: float = 0.0
    initial_aileron: float = 0.0

    # Steady mean wind (world frame, m/s).  Aerodynamics use the air-relative
    # velocity V_ground - (wind + gust); wind is not observed (crab in crosswind).
    wind_x: float = 0.0
    wind_y: float = 0.0
    wind_z: float = 0.0
    # Ornstein-Uhlenbeck turbulence: sigma = gust std (m/s), theta = mean-
    # reversion rate (1/s).  sigma = 0 (default) => steady wind, no turbulence.
    # 1.2 m/s per 1 s step is a 2 m/s stationary gust std at theta = 0.2:
    # light-to-moderate turbulence (provisional; see PHYSICS.md).
    turbulence_sigma: float = 1.2
    turbulence_theta: float = 0.2
    # MIL-F-8785C's scale length above 2000 ft, 1750 ft, which sizes the rolling
    # gust (``plane3d.dynamics.rolling_gust``); ``gust_roll_rate`` is that gust,
    # set per step as ``wind_x`` is, and 0 in still air.
    turbulence_scale_length: float = 533.4
    gust_roll_rate: float = 0.0
    # Linear wind shear: horizontal wind gains ``wind_shear_x``/``wind_shear_y``
    # m/s per metre of altitude above ``shear_ref_alt`` (0 => no shear).
    wind_shear_x: float = 0.0
    wind_shear_y: float = 0.0
    shear_ref_alt: float = 0.0

    delta_t: float = 1.0

    # ---- Amendment 2 (plane3d/PHYSICS.md), on the heading and racetrack tasks
    #: Protected limits, ground-only trips, a Dryden gust in one zone with
    #: ``turbulence_sigma`` its stationary std, an established start and the
    #: job's bands. Off, the earlier plant (the Extra tasks, patrol, formation).
    amended: bool = False
    #: The speed schedule, as the 2D aircraft's (unsourced, plane/PHYSICS.md).
    target_cas: float = 290.0 * 1852.0 / 3600.0
    target_mach: float = 0.78
    #: +-10 kt or +-0.02 M of the assigned speed [AIM 4-4-12; ACS IR.III.B.S4].
    speed_band: float = 10.0 * 1852.0 / 3600.0
    mach_band: float = 0.02
    #: Tolerances past which heading and path cost (``e_tol_altitude`` above).
    e_tol_heading: float = 0.0
    e_tol_path: float = 0.0
    #: The turbulence zone, s: onset (ours) and length, 3 to 30 min [BEA447].
    zone_onset_range: Tuple[float, float] = (0.0, 600.0)
    zone_duration_range: Tuple[float, float] = (180.0, 1800.0)
    #: 0: the cruise schedule; 1: the holding speed below the cap [IFH ch. 10].
    assigned_speed_mode: int = 0
    #: Holding turns: 3 deg/s or 30 deg of bank, whichever is less [IFH ch. 10].
    holding_rate: float = 3.0 * 3.14159265 / 180.0
    holding_bank: float = 30.0 * 3.14159265 / 180.0
    #: The bank a hold's turns are sized for, rad: an RNAV hold's radius is
    #: set at the maximum ground speed, the airspeed plus the wind, by the
    #: AIM's 25 deg, so the downwind turn takes 25 deg and the upwind one less
    #: (Garmin, RNAV holding; AIM). In still air, the IFH's holding turn.
    hold_design_bank: float = 25.0 * 3.14159265 / 180.0
    #: What the operator pays inside the bands (plane.dynamics.economy_cost).
    cost_index: float = 30.0
    economy_weight: float = 0.0
    economy_scale: float = 1.242
    #: Heading select is vectoring (rule 17, Amendment 3): a new heading every
    #: this many seconds, drawn over ``target_heading_range`` (ours: vectors come
    #: one after another; no source gives their spacing). 0: one clearance.
    heading_clearance_period: float = 0.0
    #: The protected bank: the demand stops here, for agent and reference alike
    #: (rule 14, Amendment 3). Ours, family range 25 to 30 deg [IFH ch. 10's 30
    #: deg holding bank; ICAO Doc 8168 and the A320 HDG mode's 25, unconfirmed].
    protected_bank: float = 30.0 * 3.14159265 / 180.0
    #: Inbound leg 1 min at or below 14 000 ft, 1.5 min above [IFH ch. 10].
    holding_split_altitude: float = 14_000.0 * 0.3048
    holding_leg_low: float = 60.0
    holding_leg_high: float = 90.0
    #: Holding speed caps, 230 KIAS to 14 000 ft and 265 above [IFH ch. 10];
    #: assigned 10 kt under, so the band's edge is the cap (ours).
    holding_cap_low: float = 230.0 * 1852.0 / 3600.0
    holding_cap_high: float = 265.0 * 1852.0 / 3600.0


#: Units of the PlaneParams3D fields, for the generated environment pages.
PARAM_UNITS = {
    "amended": "-",
    "target_cas": "m/s",
    "target_mach": "-",
    "speed_band": "m/s",
    "mach_band": "-",
    "e_tol_heading": "rad",
    "e_tol_path": "m",
    "zone_onset_range": "s",
    "zone_duration_range": "s",
    "assigned_speed_mode": "-",
    "holding_rate": "rad/s",
    "holding_bank": "rad",
    "hold_design_bank": "rad",
    "cost_index": "kg/min",
    "economy_weight": "-",
    "economy_scale": "kg/s",
    "protected_bank": "rad",
    "heading_clearance_period": "s",
    "holding_split_altitude": "m",
    "holding_leg_low": "s",
    "holding_leg_high": "s",
    "holding_cap_low": "m/s",
    "holding_cap_high": "m/s",
    "gravity": "m/s^2",
    "initial_mass": "kg",
    "thrust_output_at_sea_level": "N",
    "air_density_at_sea_level": "kg/m^3",
    "frontal_surface": "m^2",
    "wings_surface": "m^2",
    "C_x0": "-",
    "C_z0": "-",
    "initial_fuel_quantity": "kg",
    "specific_fuel_consumption": "kg/(kN.s)",
    "cl_alpha": "1/deg",
    "cl0": "-",
    "cd0": "-",
    "k": "-",
    "aoa_stall": "deg",
    "CL_max": "-",
    "CL_min": "-",
    "aoa_stall_width": "deg",
    "aspect_ratio": "-",
    "M_crit": "-",
    "k_shock_stall": "-",
    "shock_stall_floor": "-",
    "power_response_rate": "1/s",
    "idle_power": "-",
    "stick_response_rate": "1/s",
    "surface_rate_limit": "rad/s",
    "load_factor_max": "g",
    "load_factor_min": "g",
    "vmo_eas": "m/s",
    "mmo": "-",
    "buffet_mach": "-",
    "buffet_slope": "-",
    "bank_limit": "rad",
    "k_drag": "-",
    "pitch_gyration_radius": "m",
    "moment_arm_stabilizer": "m",
    "moment_arm_wings": "m",
    "stabilizer_surface": "m^2",
    "tail_cl_alpha": "1/deg",
    "tail_aspect_ratio": "-",
    "tail_sweep_half_chord": "rad",
    "tail_k": "-",
    "tail_CL_max": "-",
    "tail_aoa_stall": "deg",
    "tail_efficiency": "-",
    "elevator_effectiveness": "-",
    "downwash_ratio": "-",
    "stabilizer_setting": "rad",
    "sweep_half_chord": "rad",
    "ap_load_increment": "g",
    "precision_floor": "m",
    "reward_version": "-",
    "tracking_exponent": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
    "min_alt": "m",
    "max_alt": "m",
    "initial_z_dot": "m/s",
    "initial_x_dot": "m/s",
    "initial_theta_dot": "deg/s",
    "initial_theta": "deg",
    "initial_power": "-",
    "initial_stick": "deg",
    "wind_x": "m/s",
    "wind_z": "m/s",
    "turbulence_sigma": "m/s",
    "turbulence_theta": "1/s",
    "turbulence_scale_length": "m",
    "gust_roll_rate": "rad/s",
    "wind_shear_x": "1/s",
    "shear_ref_alt": "m",
    "aileron_response_rate": "1/s",
    "aileron_effectiveness": "-",
    "roll_gyration_radius": "m",
    "wingspan": "m",
    "aileron_surface": "m^2",
    "moment_arm_aileron": "m",
    "C_lp": "-",
    "heading_precision_floor": "rad",
    "position_precision_floor": "m",
    "e_floor_altitude": "m",
    "e_tol_altitude": "m",
    "e_floor_heading": "rad",
    "e_floor_path": "m",
    "figure8_altitude_amplitude": "m",
    "initial_y_dot": "m/s",
    "initial_phi": "deg",
    "initial_phi_dot": "deg/s",
    "initial_heading": "rad",
    "initial_aileron": "deg",
    "wind_y": "m/s",
    "wind_shear_y": "1/s",
}


# ─── Shared helpers ──────────────────────────────────────


def check_is_terminal_3d(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Return (terminated, truncated) flags. The aircraft trips when its altitude
    leaves ``[min_alt, max_alt]``, past the bank limit, or out of the envelope
    (``plane.dynamics.outside_envelope``)."""
    wind_x, wind_y, wind_z = total_wind_3d(
        state.z, state.gust_x, state.gust_y, state.gust_z, params
    )
    air_speed = compute_velocity_3d(
        state.x_dot - wind_x, state.y_dot - wind_y, state.z_dot - wind_z
    )
    envelope = (
        (state.z >= params.max_alt)
        | (jnp.abs(state.phi) > params.bank_limit)
        | outside_envelope(state.load_factor, air_speed, state.z, params)
    )
    # Amended: load factor, AoA and bank are protected and VMO is a warning
    # (rule 14; plane/REALITY.md, limits); the ground is the one trip.
    terminated = (state.z <= params.min_alt) | (
        envelope & jnp.logical_not(params.amended)
    )
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def airspeed_3d(state, params, xp=jnp):
    """True airspeed against the wind and gust, what air data reads."""
    wind_x, wind_y, wind_z = total_wind_3d(
        state.z, state.gust_x, state.gust_y, state.gust_z, params
    )
    return compute_velocity_3d(
        state.x_dot - wind_x, state.y_dot - wind_y, state.z_dot - wind_z
    )


def assigned_cas_3d(target_altitude, params, xp=jnp):
    """The CAS assigned: the cruise schedule's, or on a hold 10 kt under the
    holding cap for the level [IFH ch. 10]."""
    low = target_altitude <= params.holding_split_altitude
    cap = xp.where(low, params.holding_cap_low, params.holding_cap_high)
    hold = cap - params.speed_band
    return xp.where(params.assigned_speed_mode == 1, hold, params.target_cas)


def assigned_tas_3d(z, target_altitude, params, xp=jnp):
    """The true airspeed of the assigned speed at height ``z``."""
    cas = assigned_cas_3d(target_altitude, params, xp)
    return scheduled_airspeed(z, cas, params.target_mach, xp)


def holding_turn_rate(speed, params, xp=jnp):
    """A holding turn's rate at ``speed``: 3 deg/s or 30 deg of bank, whichever
    is less [IFH ch. 10]."""
    banked = params.gravity * xp.tan(params.holding_bank) / xp.maximum(speed, 1.0)
    return xp.minimum(params.holding_rate, banked)


def speed_cost_3d(state, params, xp=jnp):
    """The assigned speed's band cost (``plane.dynamics.speed_band_excess``)."""
    cas = assigned_cas_3d(state.target_altitude, params, xp)
    excess = speed_band_excess(state.z, airspeed_3d(state, params, xp), params, xp, cas)
    return xp.where(params.amended, excess**2, 0.0)


def wrap_angle(angle):
    """Wrap angle to [-pi, pi]."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def altitude_reward(state, params, xp=jnp):
    """Altitude tracking component, shared by all tasks."""
    return log_scaled_reward(
        xp.abs(state.target_altitude - state.z),
        params.precision_floor,
        params.max_alt - params.min_alt,
        xp,
    )


def heading_reward(state, params, xp=jnp):
    """Heading tracking component, log-scaled over the full +/-pi range."""
    return log_scaled_reward(
        xp.abs(wrap_angle(state.psi - state.target_heading)),
        params.heading_precision_floor,
        jnp.pi,
        xp,
    )


def path_reward(dist, state, params, xp=jnp):
    """Path-proximity component, normalised by the radius.

    Using ``target_radius`` as the envelope keeps the reward independent of the
    size of the commanded path, and the aircraft starts *on* the path in both
    path-following tasks, so the ``dist >= radius`` floor is only reached by a
    controller that has already lost the shape entirely.
    """
    return log_scaled_reward(
        dist, params.position_precision_floor, state.target_radius, xp
    )


# ─── Heading task reward ────────────────────────────────


def compute_reward_heading_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: multiplicative altitude * heading, both log-scaled in the error.

    Mirrors the Plane (2D) altitude reward and extends it with a heading
    factor, so the task is "Plane + heading" rather than an additive
    simplification.  Multiplicative composition requires *both* objectives to
    be met.  Each factor is log-scaled (see ``log_scaled_reward``) so the
    reward keeps discriminating all the way down to sensor resolution instead
    of saturating a few hundred metres from the target.  Crash penalty matches
    Plane 2D.
    """
    reward = altitude_reward(state, params, xp) * heading_reward(state, params, xp)
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    return reward


# ─── Circle task reward ─────────────────────────────────


def compute_reward_heading(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_heading_v1(state, params, xp),
        R.total(compute_reward_terms_heading(state, params, xp), xp),
        xp,
    )


def distance_to_circle(state: PlaneState3D):
    """Signed distance from aircraft to the target circle (positive = outside)."""
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    dist_to_center = jnp.sqrt(dx**2 + dy**2)
    return dist_to_center - state.target_radius


def _half_leg(state):
    """The pattern's half leg: the hold's own (Amendment 2), or the earlier
    ``_RACETRACK_LEG`` radii."""
    return jnp.where(
        state.target_half_leg > 0.0,
        state.target_half_leg,
        state.target_radius * _RACETRACK_LEG,
    )


def distance_to_racetrack(state: PlaneState3D):
    """Distance from the aircraft to a racetrack holding pattern.

    A holding pattern is two straight legs joined by two 180 degree turns --
    what "holding" actually means in aviation, and the shape this library is
    named for. It is the union of two half-circles of radius ``target_radius``
    centred at the ends of a straight segment of length ``2 * target_radius *
    _RACETRACK_LEG``, oriented by ``target_heading``.

    Unlike the figure-8's, this distance is computed in closed form rather than
    searched over samples, so it has no resolution floor of its own -- check 11
    of the model review checklist exists because the figure-8's argmin metric
    quantised its own reward.
    """
    r = state.target_radius
    half_leg = _half_leg(state)

    # Into pattern frame: origin at the centre, x along the straight legs.
    c, s_ = jnp.cos(state.target_heading), jnp.sin(state.target_heading)
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    u = c * dx + s_ * dy
    v = -s_ * dx + c * dy

    # Straight legs where |u| <= half_leg, the turn circles beyond.
    on_leg = jnp.abs(jnp.abs(v) - r)
    cap_u = jnp.abs(u) - half_leg
    on_cap = jnp.abs(jnp.sqrt(cap_u**2 + v**2) - r)
    return jnp.where(jnp.abs(u) <= half_leg, on_leg, on_cap)


def racetrack_guidance(state: PlaneState3D):
    """Signed cross-track error and tangent heading for the holding pattern.

    Closed form, unlike the figure-8's, whose nearest point is an argmin over
    400 samples and therefore quantised -- check 11 of the model review
    checklist exists because that quantisation was larger than the expert's own
    tracking error. Here both come out of the geometry exactly.

    Returns ``(cross_track, tangent_heading, curvature)``. ``cross_track`` is
    positive outside the pattern, so a controller steers to drive it to zero.
    ``curvature`` is signed, zero on the straight legs and 1/r through the
    turns: without it a controller can only chase the tangent, and chasing a
    tangent lags a curved path by construction -- which is exactly how the
    circle expert used to fail, and why it now carries a coordinated-turn
    feedforward.
    """
    r = state.target_radius
    half_leg = _half_leg(state)
    c, s_ = jnp.cos(state.target_heading), jnp.sin(state.target_heading)
    dx = state.x - state.target_x
    dy = state.y - state.target_y
    u = c * dx + s_ * dy
    v = -s_ * dx + c * dy

    on_leg = jnp.abs(u) <= half_leg
    side = jnp.sign(v) + (v == 0.0)

    # Nearest point on the pattern, and the tangent there. On a leg the pattern
    # runs parallel to +u on the far side and -u on the near one, so a circuit
    # goes up one and back down the other; on a cap it is a circle about the leg
    # end, turning the same way at both ends -- that is what makes it a
    # racetrack rather than a figure-8.
    leg_n = jnp.stack([u, side * r])
    leg_tan = state.target_heading + jnp.where(side > 0, 0.0, jnp.pi)

    cu = u - jnp.sign(u) * half_leg
    rho = jnp.sqrt(cu**2 + v**2) + 1e-6
    cap_n = jnp.stack([jnp.sign(u) * half_leg + r * cu / rho, r * v / rho])
    # Clockwise about either end, so a quarter turn behind the radius on both:
    # the sign of ``u`` belongs to the centre, not to the direction of travel.
    cap_tan = state.target_heading + jnp.arctan2(v, cu) - jnp.pi / 2.0

    nearest = jnp.where(on_leg, leg_n, cap_n)
    tangent = jnp.where(on_leg, leg_tan, cap_tan)

    # Signed in the tangent's own frame, so the sign means the same thing
    # everywhere: positive when the path lies to the aircraft's right. Taking it
    # from the geometry directly -- ``side * (|v| - r)`` -- reverses meaning
    # between the two legs, and a controller using it steers away from the path
    # on the return leg.
    local = state.target_heading + tangent * 0.0  # keep shapes aligned
    del local
    offset = jnp.stack([u, v]) - nearest
    tangent_local = tangent - state.target_heading
    normal = jnp.stack([-jnp.sin(tangent_local), jnp.cos(tangent_local)])
    cross = jnp.dot(offset, normal)

    # Straight legs have no curvature. Both caps turn the same way, so the sign
    # is constant rather than a function of which end the aircraft is at.
    curvature = jnp.where(on_leg, 0.0, -1.0 / r)
    return cross, jnp.arctan2(jnp.sin(tangent), jnp.cos(tangent)), curvature


def compute_reward_racetrack_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Altitude tracking * proximity to the holding pattern, both log-scaled."""
    alt_r = altitude_reward(state, params, xp)
    d = xp.abs(distance_to_racetrack(state))
    track_r = path_reward(d, state, params, xp)
    return alt_r * track_r


def compute_reward_racetrack(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_racetrack_v1(state, params, xp),
        R.total(compute_reward_terms_racetrack(state, params, xp), xp),
        xp,
    )


def compute_reward_circle_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: altitude tracking * proximity to the circle path, both log-scaled."""
    alt_r = altitude_reward(state, params, xp)
    d = xp.abs(distance_to_circle(state))
    circle_r = path_reward(d, state, params, xp)
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    return alt_r * circle_r


# ─── Figure-8 task: twisted 3D lemniscate ───────────────
#
# The lemniscate of Bernoulli is parametrised as:
#   x(τ) = a·cos(τ) / (1 + sin²τ)
#   y(τ) = a·sin(τ)·cos(τ) / (1 + sin²τ)
# for τ ∈ [0, 2π].  The 3D twist adds a sinusoidal altitude:
#   z(τ) = z_mean + Δz·sin(τ)
# so the two crossover passes (τ=π/2 at z_mean+Δz and τ=3π/2 at z_mean-Δz)
# are at different altitudes.  Viewed from above it is still a figure-8,
# but in 3D the path is unambiguous — no two branches share the same (x,y,z).
#
# The whole curve is rotated in the horizontal plane by target_heading
# (the orientation angle, randomised at reset).


def compute_reward_circle(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_circle_v1(state, params, xp),
        R.total(compute_reward_terms_circle(state, params, xp), xp),
        xp,
    )


#: Fine enough to leave a sagitta under 1 m at the 35 km tips.
_N_CURVE_SAMPLES = 800

# Straight-leg half-length of the holding pattern, in turn radii. A standard
# civil hold is a one-minute leg, which at these speeds and bank limits is
# close to two turn radii each side.
_RACETRACK_LEG = 2.0


def _sample_twisted_lemniscate(state: PlaneState3D, params: PlaneParams3D):
    """Return (curve_x, curve_y, curve_z) arrays for the twisted lemniscate."""
    a = state.target_radius
    cx, cy = state.target_x, state.target_y
    z_mean = state.target_altitude
    dz = params.figure8_altitude_amplitude
    orientation = state.target_heading  # repurposed for figure-8

    tau = jnp.linspace(0, 2.0 * jnp.pi, _N_CURVE_SAMPLES, endpoint=False)
    denom = 1.0 + jnp.sin(tau) ** 2
    base_x = a * jnp.cos(tau) / denom
    base_y = a * jnp.sin(tau) * jnp.cos(tau) / denom

    # Rotate by orientation angle
    cos_o = jnp.cos(orientation)
    sin_o = jnp.sin(orientation)
    curve_x = cx + base_x * cos_o - base_y * sin_o
    curve_y = cy + base_x * sin_o + base_y * cos_o
    curve_z = z_mean + dz * jnp.sin(tau)
    return curve_x, curve_y, curve_z


def nearest_point_on_twisted_lemniscate(state: PlaneState3D, params: PlaneParams3D):
    """Find nearest point on the 3D twisted lemniscate.

    Returns (nearest_dx, nearest_dy, nearest_dz, dist, tangent_heading)
    where (dx, dy, dz) is the vector from aircraft to nearest curve point
    and tangent_heading is the heading of the curve tangent at that point
    (flipped if anti-aligned with aircraft velocity for consistent direction).
    """
    curve_x, curve_y, curve_z = _sample_twisted_lemniscate(state, params)

    dx = curve_x - state.x
    dy = curve_y - state.y
    dz = curve_z - state.z
    dists_sq = dx**2 + dy**2 + dz**2
    idx = jnp.argmin(dists_sq)

    # Sub-sample refinement.  The coarse argmin is quantised by the sample
    # spacing (~230 m for a 35 km lemniscate at 800 samples), so an aircraft
    # flying *exactly* along the commanded curve is reported up to half a
    # spacing off it -- an order of magnitude more than the tracking error the
    # reward is meant to resolve, which floors how precisely any controller can
    # be scored.  Projecting onto the two adjacent chords instead leaves only
    # the curve's sagitta over one segment, which is sub-metre here.
    pts = jnp.stack([curve_x, curve_y, curve_z], axis=-1)
    pos = jnp.array([state.x, state.y, state.z])
    anchor = pts[idx]

    def _project_onto_chord(other):
        chord = other - anchor
        t = jnp.clip(
            jnp.dot(pos - anchor, chord) / (jnp.dot(chord, chord) + 1e-12), 0.0, 1.0
        )
        return anchor + t * chord

    idx_fwd = (idx + 1) % _N_CURVE_SAMPLES
    idx_bwd = (idx - 1) % _N_CURVE_SAMPLES
    candidates = jnp.stack(
        [_project_onto_chord(pts[idx_fwd]), _project_onto_chord(pts[idx_bwd])]
    )
    delta = candidates[jnp.argmin(jnp.sum((candidates - pos) ** 2, axis=-1))] - pos

    nearest_dx, nearest_dy, nearest_dz = delta[0], delta[1], delta[2]
    dist = jnp.sqrt(jnp.sum(delta**2) + 1e-8)

    # Tangent via central finite differences (wrapping around)
    idx_next = (idx + 1) % _N_CURVE_SAMPLES
    idx_prev = (idx - 1) % _N_CURVE_SAMPLES
    tx = curve_x[idx_next] - curve_x[idx_prev]
    ty = curve_y[idx_next] - curve_y[idx_prev]

    # Flip tangent if anti-aligned with aircraft velocity
    dot = tx * state.x_dot + ty * state.y_dot
    sign = jnp.where(dot >= 0, 1.0, -1.0)
    tangent_heading = jnp.arctan2(sign * ty, sign * tx)

    return nearest_dx, nearest_dy, nearest_dz, dist, tangent_heading


def compute_reward_figure8_v1(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """Reward: log-scaled 3D distance to the twisted lemniscate.

    Pure shape tracking — no moving reference, no shape backstop.  The 3D
    twist makes crossovers unambiguous (different altitudes), so the reward
    has a single global optimum: fly along the curve.
    """
    _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
    track_r = path_reward(dist, state, params, xp)
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    return track_r


# ─── Observation helpers ────────────────────────────────


def compute_reward_figure8(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_figure8_v1(state, params, xp),
        R.total(compute_reward_terms_figure8(state, params, xp), xp),
        xp,
    )


def heading_3d(state, params, xp=jnp):
    """The aircraft's heading, rad: where its nose points. With no sideslip in
    the model (PHYSICS.md D1) that is the direction of the velocity through the
    air, gust included; ``psi`` is the track over the ground, which a wind
    turns away from it by the drift angle. ATC vectors are headings [IFH ch. 7],
    and an IRS reads both."""
    wind_x, wind_y, _ = total_wind_3d(
        state.z, state.gust_x, state.gust_y, state.gust_z, params
    )
    return xp.arctan2(state.y_dot - wind_y, state.x_dot - wind_x)


def air_data_3d(state, params, xp=jnp):
    """What the autopilot reads besides the state (rule 16): the assigned
    speed as a true airspeed, the airspeed and the normal load factor."""
    return [
        assigned_tas_3d(state.z, state.target_altitude, params, xp),
        airspeed_3d(state, params, xp),
        state.n_z,
    ]


def get_obs_heading(state: PlaneState3D, xp=jnp, params=None):
    """Observation for heading task (15 values, 19 with the air data and the
    IRS heading, which the vector is flown on)."""
    extra = (
        []
        if params is None
        else air_data_3d(state, params, xp) + [heading_3d(state, params, xp)]
    )
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.target_heading,
            state.power,
            state.stick,
            state.aileron,
            *extra,
        ]
    )


def get_obs_racetrack(state: PlaneState3D, xp=jnp, params=None):
    """Observation for the holding pattern (21 values; 25 with the half leg,
    which the hold's timing sets, and the air data).

    The circle's observation plus ``target_heading``, which orients the pattern
    and without which the straight legs are unobservable -- the aircraft would
    have to infer which way the hold lies from its own history.
    """
    extra = (
        []
        if params is None
        else [state.target_half_leg, *air_data_3d(state, params, xp)]
    )
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.x - state.target_x,
            state.y - state.target_y,
            state.target_radius,
            state.target_heading,
            state.power,
            state.stick,
            state.aileron,
            # The guidance the pattern's geometry already determines, in closed
            # form: signed cross-track error and the tangent heading at the
            # nearest point. Provided rather than left to the controller because
            # it is exact here -- the figure-8 has to search for its nearest
            # point and pays a quantisation floor for it (check 11) -- and
            # because a policy that had to rediscover the pattern's shape from
            # position alone would be solving a different problem.
            *racetrack_guidance(state),
            *extra,
        ]
    )


def get_obs_circle(state: PlaneState3D, xp=jnp):
    """
    Observation for circle task (17 values).
    Includes relative position to circle center and target radius.
    """
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.x - state.target_x,  # relative x to center
            state.y - state.target_y,  # relative y to center
            state.target_radius,
            state.power,
            state.stick,
            state.aileron,
        ]
    )


def get_obs_figure8(state: PlaneState3D, params: PlaneParams3D, xp=jnp):
    """
    Observation for figure-8 task (21 values).

    Provides the vector from aircraft to the nearest point on the 3D
    twisted lemniscate (nearest_dx, nearest_dy, nearest_dz) plus the
    tangent heading at that point, and the tracked output: the distance to
    the curve (value) against a constant-zero target.

    Layout:
      [x_dot, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot,
       gamma, psi, target_altitude, target_radius,
       nearest_dx, nearest_dy, nearest_dz, tangent_heading,
       power, stick, aileron, path_distance, target_path_distance(=0)]
    """
    ndx, ndy, ndz, dist, tang_hdg = nearest_point_on_twisted_lemniscate(state, params)
    return xp.stack(
        [
            state.x_dot,
            state.y_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.phi,
            state.phi_dot,
            state.gamma,
            state.psi,
            state.target_altitude,
            state.target_radius,
            ndx,
            ndy,
            ndz,
            tang_hdg,
            state.power,
            state.stick,
            state.aileron,
            dist,
            jnp.zeros(()),
        ]
    )


# ─── Shared state transition ────────────────────────────

#: Salt for the rolling gust's key, apart from the three gust axes' draws.
_ROLLING_GUST = 0x524F4C4C


@partial(jax.jit, static_argnames=["integration_method", "inner_steps"])
def compute_next_state_3d(
    power_requested: float,
    pitch_demand: float,
    bank_demand: float,
    state: PlaneState3D,
    params: PlaneParams3D,
    integration_method: str = "rk4_1",
    key=None,
    inner_steps: int = INNER_STEPS,
):
    """One step: the demands held while the inner loop flies ``inner_steps``
    sub-steps, each moving the surfaces and the engine and integrating the
    dynamics. The surfaces are the loop's, never the caller's.

    Wind is a physics-engine property: total wind = steady ``params.wind_x/y/z``
    plus an Ornstein-Uhlenbeck turbulence gust (advanced with ``key`` when
    ``params.turbulence_sigma > 0``).  Aerodynamics/engine-Mach use the
    air-relative velocity while position/observations stay in the ground frame.
    ``wind = 0`` with ``turbulence_sigma = 0`` reproduces the original behaviour;
    ``key=None`` freezes the gust (constant wind).
    """
    dt = params.delta_t / inner_steps
    # Amended: bank is a protected limit (rule 14), the demand clipped to it.
    protected = jnp.clip(bank_demand, -params.protected_bank, params.protected_bank)
    bank_demand = jnp.where(params.amended, protected, bank_demand)

    # Total wind = steady mean + altitude shear + OU turbulence gust.
    previous = jnp.array([state.gust_x, state.gust_y, state.gust_z])
    kicked = advance_gust(
        previous,
        params.turbulence_theta,
        params.turbulence_sigma,
        params.delta_t,
        step_key(key, state.time),
    )
    # Amended: a Dryden gust in the episode's zone, calm outside it
    # (plane/PHYSICS.md, turbulence), with ``turbulence_sigma`` its std.
    zone = in_zone(state.time * params.delta_t, state.zone_start, state.zone_end)
    dryden = advance_dryden_gust(
        previous,
        compute_velocity_3d(
            state.x_dot - params.wind_x - previous[0],
            state.y_dot - params.wind_y - previous[1],
            state.z_dot - params.wind_z - previous[2],
        ),
        jnp.where(zone, params.turbulence_sigma, 0.0),
        params.turbulence_scale_length,
        params.delta_t,
        step_key(key, state.time),
    )
    gust = jnp.where(params.amended, dryden, kicked)
    calm = params.amended & jnp.logical_not(zone)
    total_wind_x, total_wind_y, total_wind_z = total_wind_3d(
        state.z, gust[0], gust[1], gust[2], params
    )
    # With no key the caller owns the air (a patrol's shared gust), as above.
    gust_roll_rate = params.gust_roll_rate
    if key is not None:
        roll_key = jax.random.fold_in(step_key(key, state.time), _ROLLING_GUST)
        speed = compute_velocity_3d(
            state.x_dot - total_wind_x,
            state.y_dot - total_wind_y,
            state.z_dot - total_wind_z,
        )
        rolled = rolling_gust(roll_key, speed, params)
        gust_roll_rate = gust_roll_rate + jnp.where(calm, 0.0, rolled)
    # All-up mass for this step: fuel is a component of ``initial_mass``, so
    # burning it subtracts. Threaded through ``eff_params`` because the
    # dynamics read ``params.initial_mass``.
    mass_now = params.initial_mass - (params.initial_fuel_quantity - state.fuel)
    eff_params = params.replace(
        wind_x=total_wind_x,
        wind_y=total_wind_y,
        wind_z=total_wind_z,
        gust_roll_rate=gust_roll_rate,
        initial_mass=mass_now,
    )
    gains = params.inner_loop

    def inner(carry, _):
        velocities, positions, power, surfaces, integrals, burned, peak = carry
        x_dot, y_dot, z_dot, theta_dot, phi_dot = velocities
        _, _, z, theta, phi = positions
        stick, aileron = surfaces
        pitch_integ, roll_integ = integrals
        # Air data and the engine's ram/Mach effects read airspeed.
        air_speed = compute_velocity_3d(
            x_dot - total_wind_x, y_dot - total_wind_y, z_dot - total_wind_z
        )
        rho = compute_air_density_from_altitude(z)
        eas = air_speed * jnp.sqrt(rho / params.air_density_at_sea_level)
        scaler = airspeed_scaler(eas, gains)
        # Amended: normal law's protection of load factor and AoA (rule 14).
        air_up = z_dot - total_wind_z
        protected = protected_pitch_demand(
            pitch_demand,
            jnp.arcsin(jnp.clip(air_up / jnp.maximum(air_speed, 1.0), -1.0, 1.0)),
            air_speed,
            z,
            mass_now,
            params,
            # A level turn needs 1 / cos(bank) g; the autopilot's margin is
            # about that load, not about 1 g.
            load_reference=1.0 / jnp.maximum(jnp.cos(phi), 0.1),
        )
        demand = jnp.where(params.amended, protected, pitch_demand)
        stick_output, pitch_integ = attitude_axis(
            gains.Kp_theta,
            gains.Ki_theta,
            gains.Kd_theta,
            demand - theta,
            theta_dot,
            pitch_integ,
            scaler,
            dt,
        )
        aileron_output, roll_integ = attitude_axis(
            gains.Kp_roll,
            gains.Ki_roll,
            gains.Kd_roll,
            bank_demand - phi,
            phi_dot,
            roll_integ,
            scaler,
            dt,
        )
        stick = compute_next_surface(
            stick_output * STICK_TRAVEL,
            stick,
            dt,
            params.stick_response_rate,
            params.surface_rate_limit,
        )
        aileron = compute_next_surface(
            aileron_output * AILERON_TRAVEL,
            aileron,
            dt,
            params.aileron_response_rate,
            params.surface_rate_limit,
        )
        power = compute_next_power(
            power_requested,
            power,
            dt,
            engine_response_rate(z, params.power_response_rate),
            params.idle_power,
        )
        thrust = compute_thrust_output(
            power=power,
            thrust_output_at_sea_level=params.thrust_output_at_sea_level,
            rho=rho,
            M=compute_Mach_from_velocity_and_speed_of_sound(
                air_speed, compute_speed_of_sound_from_altitude(z)
            ),
        )
        accelerate = partial(
            compute_acceleration_3d,
            action=(thrust, stick, aileron),
            params=eff_params,
            clip=True,
            min_clip_boundaries=(-100, -100, -100, -1.5, -1.5),
            max_clip_boundaries=(100, 100, 100, 1.5, 1.5),
        )
        velocities, positions, metrics = integrate_dynamics(
            velocities=velocities,
            positions=positions,
            delta_t=dt,
            compute_acceleration=accelerate,
            method=integration_method,
        )
        # Fuel burned over the sub-step, charged against the thrust produced.
        burned = burned + params.specific_fuel_consumption * 1e-3 * thrust * dt
        # Trips read the worst load inside the step, not the one it ends on.
        load = metrics[1][-1] / (mass_now * params.gravity)
        peak = jnp.where(jnp.abs(load - 1.0) > jnp.abs(peak - 1.0), load, peak)
        return (
            velocities,
            positions,
            power,
            (stick, aileron),
            (pitch_integ, roll_integ),
            burned,
            peak,
        ), metrics

    start = (
        jnp.array(
            [state.x_dot, state.y_dot, state.z_dot, state.theta_dot, state.phi_dot]
        ),
        jnp.array([state.x, state.y, state.z, state.theta, state.phi]),
        state.power,
        (state.stick, state.aileron),
        (state.pitch_integ, state.roll_integ),
        jnp.zeros_like(state.fuel),
        jnp.ones_like(state.fuel),
    )
    end, metrics = jax.lax.scan(inner, start, None, length=inner_steps)
    x_dot, y_dot, z_dot, theta_dot, phi_dot = end[0]
    x, y, z, theta, phi = end[1]
    power, (stick, aileron), (pitch_integ, roll_integ), burned, load_factor = end[2:]
    metrics = jax.tree_util.tree_map(lambda m: m[-1], metrics)

    alpha, gamma = compute_alpha_3d(theta, x_dot, y_dot, z_dot)
    psi = compute_psi(x_dot, y_dot)
    # Total all-up mass: fuel is a component of ``initial_mass``.
    fuel = jnp.clip(state.fuel - burned, 0.0, params.initial_fuel_quantity)
    m = params.initial_mass - (params.initial_fuel_quantity - fuel)

    new_state = PlaneState3D(
        x=x,
        x_dot=x_dot,
        y=y,
        y_dot=y_dot,
        z=z,
        z_dot=z_dot,
        theta=theta,
        theta_dot=theta_dot,
        phi=phi,
        phi_dot=phi_dot,
        psi=psi,
        alpha=alpha,
        gamma=gamma,
        m=m,
        power=power,
        stick=stick,
        aileron=aileron,
        fuel=fuel,
        time=state.time + 1,
        sensor_key=state.sensor_key,
        actuator_held=state.actuator_held,
        target_altitude=state.target_altitude,
        target_heading=state.target_heading,
        target_x=state.target_x,
        target_y=state.target_y,
        target_radius=state.target_radius,
        gust_x=gust[0],
        gust_y=gust[1],
        gust_z=gust[2],
        pitch_integ=pitch_integ,
        roll_integ=roll_integ,
        load_factor=load_factor,
        target_half_leg=state.target_half_leg,
        heading_change=state.heading_change,
        n_z=metrics[1][-1] / (mass_now * params.gravity),
        zone_start=state.zone_start,
        zone_end=state.zone_end,
        clearance_time=state.clearance_time,
    )
    return new_state, metrics


def altitude_cost(state, params, xp=jnp):
    return R.tracking_cost(
        state.target_altitude - state.z,
        params.e_floor_altitude,
        params.e_tol_altitude,
        params.tracking_exponent,
        xp,
    )


def _downtime(terms, state, params, xp=jnp, tripped=None):
    """``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``)."""
    if tripped is None:
        tripped, _ = check_is_terminal_3d(state, params, xp)
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def compute_reward_terms_heading(
    state: PlaneState3D, params: PlaneParams3D, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    # Amended: the heading band, +-10 deg [ACS], about a heading that turns
    # toward the target at a holding turn's rate from the clearance [IFH ch.
    # 10], so the band travels with that turn; ahead of it costs nothing. Unamended, the allowance
    # is ``e_tol_heading`` (0) and this is plain tracking.
    # The vector is a heading, so the error is the heading's, not the track's.
    error = xp.abs(wrap_angle(heading_3d(state, params, xp) - state.target_heading))
    rate = holding_turn_rate(
        assigned_tas_3d(state.target_altitude, state.target_altitude, params, xp),
        params,
        xp,
    )
    since = (state.time - state.clearance_time) * params.delta_t
    left = state.heading_change - rate * since
    allowance = params.e_tol_heading + xp.maximum(left, 0.0)
    behind = xp.maximum(error - allowance, 0.0) + params.e_tol_heading
    heading = R.tracking_cost(
        behind,
        params.e_floor_heading,
        params.e_tol_heading,
        params.tracking_exponent,
        xp,
    )
    terms = {
        "tracking": altitude_cost(state, params, xp) + heading,
        "running": speed_cost_3d(state, params, xp),
        "economy": economy_cost(
            state.power,
            state.z,
            airspeed_3d(state, params, xp),
            assigned_tas_3d(state.z, state.target_altitude, params, xp),
            params,
            xp,
        ),
    }
    return _downtime(terms, state, params, xp, tripped)


def compute_reward_terms_racetrack(
    state: PlaneState3D, params: PlaneParams3D, xp=jnp, tripped=None
):
    path = R.tracking_cost(
        distance_to_racetrack(state),
        params.e_floor_path,
        params.e_tol_path,
        params.tracking_exponent,
        xp,
    )
    # Amended: headings +-10 deg of the pattern's [ACS IR.III.B.S4], which
    # also tells right turns from left [IFH ch. 10].
    _, tangent, _ = racetrack_guidance(state)
    heading = R.tracking_cost(
        wrap_angle(state.psi - tangent),
        params.e_floor_heading,
        params.e_tol_heading,
        params.tracking_exponent,
        xp,
    )
    terms = {
        "tracking": altitude_cost(state, params, xp)
        + path
        + xp.where(params.amended, heading, 0.0),
        "running": speed_cost_3d(state, params, xp),
    }
    return _downtime(terms, state, params, xp, tripped)


def compute_reward_terms_circle(
    state: PlaneState3D, params: PlaneParams3D, xp=jnp, tripped=None
):
    path = R.tracking_cost(
        distance_to_circle(state),
        params.e_floor_path,
        0.0,
        params.tracking_exponent,
        xp,
    )
    terms = {"tracking": altitude_cost(state, params, xp) + path}
    return _downtime(terms, state, params, xp, tripped)


def compute_reward_terms_figure8(
    state: PlaneState3D, params: PlaneParams3D, xp=jnp, tripped=None
):
    _, _, _, dist, _ = nearest_point_on_twisted_lemniscate(state, params)
    path = R.tracking_cost(dist, params.e_floor_path, 0.0, params.tracking_exponent, xp)
    return _downtime({"tracking": path}, state, params, xp, tripped)
