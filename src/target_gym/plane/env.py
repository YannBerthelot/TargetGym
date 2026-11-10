from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import (
    integrate_dynamics,
)
from target_gym.plane.dynamics import (
    INNER_STEPS,
    STICK_RESPONSE_RATE,
    STICK_TRAVEL,
    InnerLoop,
    advance_dryden_gust,
    advance_gust,
    airspeed_scaler,
    attitude_axis,
    compute_acceleration,
    compute_air_density_from_altitude,
    compute_alpha,
    compute_Mach_from_velocity_and_speed_of_sound,
    compute_next_power,
    compute_next_surface,
    compute_speed_of_sound_from_altitude,
    compute_thrust_output,
    compute_velocity_from_horizontal_and_vertical_speed,
    economy_cost,
    engine_response_rate,
    in_zone,
    protected_pitch_demand,
    scheduled_airspeed,
    speed_band_excess,
    step_key,
    total_wind_2d,
)
from target_gym.utils import log_scaled_reward

DEBUG = False


@struct.dataclass
class EnvMetrics:
    drag: float
    lift: float
    S_x: float
    S_z: float
    C_x: float
    C_z: float
    F_x: float
    F_z: float


@struct.dataclass
class PlaneState(EnvState):
    x: float
    x_dot: float
    z: float
    z_dot: float
    theta: float
    theta_dot: float
    alpha: float
    gamma: float
    m: float
    power: float
    stick: float
    fuel: float
    target_altitude: float
    # The altitude sampled at reset. ``target_altitude`` is what is commanded
    # *now*, which differs once a moving pattern is selected.
    base_target_altitude: float = 0.0
    # Ornstein-Uhlenbeck turbulence gust (m/s), mean-reverting to 0.  Total wind
    # acting on the aircraft is params.wind_* + gust_*.  Default 0 => no gust.
    gust_x: float = 0.0
    gust_z: float = 0.0
    # The inner loop's pitch integral, in surface units.
    pitch_integ: float = 0.0
    # The step's load factor furthest from 1 g, inside its sub-steps.
    load_factor: float = 1.0
    # The normal load factor at the end of the step, what an accelerometer reads.
    n_z: float = 1.0
    # The episode's turbulence zone, in seconds from the start (REALITY.md,
    # time scales: zones of 3 to 30 min with calm between [BEA447 Figure 24]).
    zone_start: float = 0.0
    zone_end: float = 0.0

    actuator_held: jax.Array = actuator_field(2)

    @property
    def rho(self):
        return compute_air_density_from_altitude(self.z)

    @property
    def speed_of_sound(self):
        return compute_speed_of_sound_from_altitude(self.z)

    @property
    def M(self):
        return compute_Mach_from_velocity_and_speed_of_sound(
            compute_velocity_from_horizontal_and_vertical_speed(self.x_dot, self.z_dot),
            self.speed_of_sound,
        )


@struct.dataclass
class PlaneParams(EnvParams):
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

    # The swept wing's lift-curve slope at Mach 0, derived (PHYSICS.md 4):
    # DATCOM's 2 pi A / (2 + sqrt(A^2 (1 + tan^2 L) + 4)) with A = 34.1^2/122.6
    # = 9.48 and the half-chord sweep L below, 4.80 /rad = 0.0838 /deg. It grows
    # with Mach by ``compressibility``. The unswept Prandtl slope it replaced,
    # 0.0879 /deg grown by 1/beta, overstated the slope at M 0.78 by 26 %.
    cl_alpha: float = 0.08378  # per deg
    #: The wing's half-chord sweep, rad: 21.9 deg, from 25 deg at the quarter
    #: chord and a taper of 0.24 [Jenkinson, A320-200].
    sweep_half_chord: float = 0.38186
    cl0: float = 0.2  # zero-lift AoA
    cd0: float = 0.02  # zero-lift drag
    k: float = 0.045  # induced drag factor
    # Ours: where the swept wing's linear lift, 0.2 + 0.0838 x 15.5, reaches
    # CL_max, so the four lift constants describe one wing.
    aoa_stall: float = 15.5  # deg
    CL_max: float = 1.5
    # Negative-stall limit. A cambered wing stalls asymmetrically: the negative
    # branch bottoms out near -10 deg AoA. Without this the corrected (steeper)
    # lift slope lets CL run away negative -- it was previously masked by the
    # too-shallow slope, not actually bounded.
    CL_min: float = -1.0
    # Degrees beyond `aoa_stall` at which the separation sigmoid is centred, so
    # that lift *peaks at* aoa_stall rather than being halved there.
    aoa_stall_width: float = 3.0
    # Wing aspect ratio, span^2 / S = 34.1^2 / 122.6 for the A320. Sourced
    # geometry (PHYSICS.md section 3), and it sets how much force a *finite*
    # wing develops once separated: a 2D flat plate reaches a normal-force
    # coefficient of 2, a finite one less, because the flow relieves around the
    # tips.
    aspect_ratio: float = 9.48
    # Drag-divergence Mach number. NOTE: this field was previously declared
    # twice (0.78 above the aero block, 0.80 here). Python keeps the last
    # definition, so 0.80 was always the effective value and the 0.78 was
    # dead -- it is the A320 *cruise* Mach, not the drag-divergence Mach,
    # which is where the confusion came from. Kept at 0.80 so behaviour is
    # unchanged; see PHYSICS.md for sourcing.
    M_crit: float = 0.80
    # Shock stall above M_crit (PHYSICS.md D3): the attainable lift falls
    # once shocks form, rather than rising with the Prandtl-Glauert factor.
    # k = 4.0 puts CL_max back to its low-speed value by M 0.9 and at 40 %
    # of it by M 0.95, which is the right order for lift divergence.
    k_shock_stall: float = 4.0
    shock_stall_floor: float = 0.25

    # The altitude error below which "more precise" stops being meaningful, in
    # metres. It is a *resolution*, not a tolerance: the reward keeps paying for
    # every halving of the error down to this point, and only flattens beneath
    # it. 1 m is the order of a barometric altimeter's resolution, so tracking
    # tighter than this would be rewarding the controller for chasing noise.
    #
    # This is the parameter that decides how precise the benchmark asks a policy
    # to be, so it is deliberately a physical limit rather than a comfort band:
    # a band rewards reaching a tolerance and then stops caring, which cannot
    # show whether a learned policy holds altitude better than a PID.
    precision_floor: float = 1.0
    # A 1.7 s engine above 5% idle. 14 CFR 33.73(b) gives idle to 95% in 5 s, and
    # a first-order lag takes ln 19 of its time constant to do that (5.006 s).
    power_response_rate: float = 1.0 / 1.7
    idle_power: float = 0.05
    # The elevator's lag and rate limit, inside the inner loop (PHYSICS.md
    # section 3 gives each figure's status).
    stick_response_rate: float = STICK_RESPONSE_RATE
    surface_rate_limit: float = jnp.pi / 6.0
    # The fixed inner loop. Its gains are task constants, not tuned per task.
    inner_loop: InnerLoop = InnerLoop()
    # The envelope (PHYSICS.md section 3): the limit load factors of 14 CFR
    # 25.337, the A320's VMO 350 kt and MMO 0.82, and the buffet line, CL_max
    # falling from ``buffet_mach`` at ``buffet_slope`` per unit of Mach (ours).
    load_factor_max: float = 2.5
    load_factor_min: float = -1.0
    #: The autopilot's manoeuvre limit, g either side of the load the flight
    #: path needs (``protected_pitch_demand``): ours, the top of the 0.1 to
    #: 0.2 g the record's autopilots hold a capture to [KFC 500,
    #: TM 1-1510-225-10]. Every controller flies through it.
    ap_load_increment: float = 0.2
    vmo_eas: float = 350.0 * 1852.0 / 3600.0
    mmo: float = 0.82
    buffet_mach: float = 0.3
    buffet_slope: float = 1.4
    k_drag: float = 5.0

    # ---- The longitudinal airframe (PHYSICS.md 3 and 4) ----
    #: Pitch radius of gyration, m: JSBSim's A320 inertia over its empty mass,
    #: 3.78e6 kg m^2 over 50.3 t, applied at the all-up mass (5.5e6 at 73.5 t).
    pitch_gyration_radius: float = 8.666
    #: The wing-body's aerodynamic centre ahead of the centre of gravity, m:
    #: (0.29 - 0.098) of the 4.29 m MAC [Jenkinson]. The wing's quarter chord
    #: moved forward 0.152 MAC by the fuselage (Torenbeek's 1.8 b h l / (a S c),
    #: its 3.95 x 4.14 m section [Jenkinson] and 13 m of nose ahead of the wing
    #: root, ours), and the centre of gravity at 29 % MAC (ours).
    moment_arm_wings: float = 0.822
    #: The tail's aerodynamic centre behind the centre of gravity, m: the
    #: 13.53 m tail arm from the wing's quarter chord [Jenkinson], less the
    #: centre of gravity's 0.04 MAC behind it.
    moment_arm_stabilizer: float = 13.36
    #: Horizontal tail area, m^2, elevators included [Jenkinson].
    stabilizer_surface: float = 31.0
    #: The tail's slope at Mach 0, per deg: DATCOM's, as the wing's, for its
    #: aspect ratio of 5.0 and 23.6 deg of half-chord sweep (29 deg at the
    #: quarter chord, taper 0.256) [Jenkinson].
    tail_cl_alpha: float = 0.07021
    tail_aspect_ratio: float = 5.0
    tail_sweep_half_chord: float = 0.41100
    #: The tail's induced drag factor, 1 / (pi A e) with e = 0.8 (ours).
    tail_k: float = 0.0796
    #: The tail's stall: a symmetric section at +-15 deg, CL 1.0 (ours).
    tail_CL_max: float = 1.0
    tail_aoa_stall: float = 15.0
    #: The tail's dynamic pressure over the free stream's (ours: 0.85 to 0.95 is
    #: the textbook range for a fuselage-mounted tail).
    tail_efficiency: float = 0.9
    #: How much of an elevator deflection turns the tail's incidence (ours: a
    #: flap of about 30 % chord, 0.66 by thin-aerofoil theory, reduced to the
    #: 0.5 measured flaps reach).
    elevator_effectiveness: float = 0.5
    #: The downwash at the tail over lifting-line's far field (``downwash``):
    #: 1, the far field itself, on the A320's long wing and high-set tail.
    downwash_ratio: float = 1.0
    #: The tail's fixed setting against the body, rad: the trimmable
    #: stabiliser at the reference cruise's trim, so the elevator holds it at
    #: zero there (FL330, M 0.78, 73.5 t; calibrated, PHYSICS.md 4). The
    #: stabiliser's own travel is not modelled: the attitude loop's integral
    #: trims on the elevator.
    stabilizer_setting: float = 0.02417

    max_steps_in_episode: int = 10_000

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Altitude: quadratic outside a +-``e_tol`` tolerance about the commanded
    # altitude, normalised by the altitude floor ``e_floor``. Airspeed: the
    # deviation from ``target_speed`` above the hold-phase deviation of the
    # better shipped controller (``c_hold``, m/s; 5.06 for the altitude hold,
    # 8.15 on the sinusoid, 5.23 on the ladder, all PID) charged at weight 1,
    # a stand-in for fuel. Flying out of the altitude envelope costs, per
    # step, twice the envelope.
    reward_version: int = 2
    # The 1 m barometric resolution. The oracle holds finer than that in the
    # test turbulence, 0.46 m on its best seed (`scripts/measure_hold.py`), so
    # the instrument sets the scale. ``plane_sine``
    # and ``plane_energy`` set their own floors in the registry.
    e_floor: float = 1.0  # m
    e_tol: float = (
        0.0  # m; a +-30 m band made the hold vacuous (both controllers held 0.1 m)
    )
    tracking_exponent: float = 2.0
    c_hold: float = (
        5.06  # m/s airspeed deviation while holding (PID; 8.15 sine, 5.23 ladder)
    )
    running_weight: float = 1.0
    #: ``plane`` sets its own (Amendment 2, ``Airplane2D.default_params``).
    failure_cost: float = 2.0 * (12192.0 / 1.0) ** 2
    #: Restart time priced into a trip (``reward.trip_cost``): a crash loses the
    #: sortie, 1 h of flight at 1 s steps (provisional).
    restart_steps: int = 3600
    #: The oracle's best hold: the lowest per-seed hold cost the shipped MPC
    #: demonstrated, in the reward's units (its 0.4581 m hold in 1 m units,
    #: `scripts/measure_hold.py`; per task in the registry). ``rho_floor``
    #: leaves out the airspeed term, as on the other plants whose running
    #: term is charged above the hold-phase consumption; the oracle still
    #: pays some of it while holding (``hold_running`` in
    #: protocol_results.json). The oracle's planner does not read it
    #: (``objective_scale`` in plane/experts.py).
    rho_floor_tracking: float = (0.4581 / 1.0) ** 2
    rho_floor: float = (0.4581 / 1.0) ** 2
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False
    min_alt: float = 0.0
    # The A320's maximum operating altitude, 39 100 ft [TCDS item 11]. An
    # operating limit, not a trip: no source has the aircraft trip there.
    max_alt: float = 39_100.0 * 0.3048
    # Look-ahead wind: apply the gust ALREADY stored in the state this step, so
    # the sensor that reads it reveals the disturbance *before* it acts. That
    # makes it genuine feedforward an agent can pre-cancel -- and a disturbance a
    # PID structurally cannot, which is the kind of gap this benchmark exists to
    # measure. With False (the default) the gust is advanced then applied, so an
    # ``observe_wind`` sensor only ever reports what has already hit the
    # aircraft, whose effect is in the velocities anyway.
    wind_lookahead: bool = False
    target_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    # Commanded-altitude pattern. 0 = hold one altitude, which is what the
    # shipped baselines are measured against and therefore the default; the rest
    # make the setpoint move, which is a different and harder task -- a P
    # controller cannot hold a ramp with zero error at all, and a sinusoid
    # exposes the closed loop's bandwidth directly, since the amplitude ratio
    # and phase lag against frequency *are* its frequency response.
    #
    #   0  hold      a single altitude, sampled at reset
    #   1  steps     a staircase: discrete changes at fixed intervals
    #   2  ramp      a constant climb or descent rate
    #   3  sinusoid  a continuous oscillation about the sampled altitude
    #   4  chirp     a sinusoid whose frequency rises through the episode
    #   5  clearances  ATC's level and speed clearances (``plane_energy``)
    # Weight on an airspeed target, alongside the altitude one. Zero -- the
    # default -- leaves the task exactly as it was, tracking altitude alone.
    #
    # Above zero it becomes the energy-management problem the aircraft has
    # always physically had and never been scored on. It carries two actuators,
    # thrust and elevator, against one objective, so a controller has a spare
    # degree of freedom and can trade airspeed for altitude freely. Scoring both
    # removes it: thrust is finite, so climbing and accelerating compete, and
    # the exchange between them is precisely the phugoid.
    speed_weight: float = 0.0
    # The airspeed channel's schedule (``plane.dynamics.scheduled_airspeed``):
    # 290 kt CAS, then M 0.78, EUROCONTROL's A320 climb. A fixed 230 m/s was
    # 385 KEAS at 3000 m, past VMO.
    target_cas: float = 290.0 * 1852.0 / 3600.0  # m/s
    target_mach: float = 0.78
    speed_precision_floor: float = 0.5  # m/s, air-data resolution
    speed_envelope: float = 60.0  # m/s, the error at which the term reaches zero
    target_pattern: int = 0
    target_amplitude: float = 800.0  # m, half-range of the moving patterns
    target_period: float = 240.0  # s, one cycle of the sinusoid
    target_steps: int = 4  # how many treads in the staircase
    target_chirp_octaves: float = 2.0  # frequency multiple across a chirp
    initial_altitude_range: Tuple[float, float] = (3_000.0, 8_000.0)
    #: How far from the commanded altitude the episode may start. The initial
    #: altitude used to be drawn independently of the target over the same
    #: 5 000 m band, so the expected gap was 1 667 m and could reach 5 000 m --
    #: minutes of open-loop climb at the aircraft's climb rate before any
    #: tracking began, and on a short clip that was the entire episode. Drawing
    #: it around the target instead keeps the sampling varied while making the
    #: episode about holding the altitude, which is what the reward scores.
    #: ``initial_altitude_range`` still bounds the result.
    #: Zero on the scored tasks: the hold starts established at its level
    #: [BEA447, ALT CRZ; REALITY.md]. Extra tasks may still widen it.
    initial_altitude_offset_range: Tuple[float, float] = (0.0, 0.0)
    initial_z_dot: float = 0.0
    initial_x_dot: float = 200.0
    initial_theta_dot: float = 0.0
    initial_theta: float = 0.0
    initial_power: float = 1.0
    initial_stick: float = 0.0

    # Steady mean wind (world frame, m/s).  Aerodynamics use the air-relative
    # velocity V_ground - (wind + gust).
    wind_x: float = 0.0
    wind_z: float = 0.0
    # Ornstein-Uhlenbeck turbulence on top of the mean wind: sigma is the gust
    # std (m/s), theta the mean-reversion rate (1/s; correlation time ~ 1/theta).
    # sigma = 0 (default) => no turbulence, wind is exactly the steady mean.
    # 1.2 m/s per 1 s step is a 2 m/s stationary gust std at theta = 0.2:
    # light-to-moderate turbulence (provisional; see PHYSICS.md).
    #
    # Now (Amendment 2): ``turbulence_sigma`` is the gust's stationary standard
    # deviation inside a zone, on every axis (isotropic above 2 000 ft
    # [MADDEN]), 0.55 m/s: the intensity whose sharp-edged gust load at the
    # schedule's speed at 5 500 m spans the 0.2 g peak to peak per minute that
    # BEA447 Figure 24 records most (PHYSICS.md, turbulence: derived).
    # ``turbulence_theta`` is not read by the Dryden gust, whose correlation is
    # set by ``gust_scale_length`` and the airspeed.
    turbulence_sigma: float = 0.55
    turbulence_theta: float = 0.2
    #: 1 750 ft above 2 000 ft AGL [MADDEN, MIL-F-8785C].
    gust_scale_length: float = 1750.0 * 0.3048
    #: When the zone starts, s: anywhere in a 30 min episode (ours: so every
    #: episode holds one onset; real calm lasts over an hour [BEA447 Fig. 24]).
    zone_onset_range: Tuple[float, float] = (0.0, 1800.0)
    #: How long it lasts, s: 3 to 30 min [BEA447 Figure 24].
    zone_duration_range: Tuple[float, float] = (180.0, 1800.0)
    # ---- The job's bands (Amendment 2, rule 15) ----
    #: Score the speed as the band below rather than as ``c_hold``'s excess.
    #: ``plane`` sets it, with ``e_tol = e_floor = 65 ft`` [RVSM A.4.1.4].
    speed_band_cost: bool = False
    #: +-10 kt of an assigned speed below the crossover, +-0.02 Mach above it
    #: [AIM 4-4-12].
    speed_band: float = 10.0 * 1852.0 / 3600.0
    mach_band: float = 0.02
    # ---- What the operator pays inside the bands (owner, 2026-11-10) ----
    #: Cost index, kg of fuel a minute of time is worth: ours, family range
    #: 6 (an operator's) to 40 (the A320 QRH's long-range cruise equivalent),
    #: both secondary (pprune, plane/REALITY.md).
    cost_index: float = 30.0
    #: Weight of the economy term (``economy_cost``): 0 leaves it out; ``plane``
    #: sets 1, so a second of the reference cruise costs 1, as a second one
    #: band width past a band does.
    economy_weight: float = 0.0
    #: The reference cruise's operating cost, kg/s: fuel at FL330, M 0.78,
    #: 73.5 t (0.742 kg/s in the model) plus the cost index's 0.5.
    economy_scale: float = 1.242
    # ---- Clearances (``target_pattern = 5``, ``plane_energy``) ----
    #: One clearance per tread, s (ours: a 2 000 ft change at the 500 fpm
    #: minimum takes 240 s, plus a minute held at the new level).
    clearance_period: float = 300.0
    #: Climb or descend at no less than 500 fpm [AIM 4-4-10].
    level_change_rate_min: float = 500.0 * 0.3048 / 60.0
    #: Within 1 000 ft of the level, no more than 1 500 fpm [AIM 4-4-10] ...
    level_off_height: float = 1000.0 * 0.3048
    level_off_rate_max: float = 1500.0 * 0.3048 / 60.0
    #: ... charged past it in widths of 500 fpm, half the AIM's 500 to
    #: 1 500 fpm range (ours: the band's half-width, as the other bands).
    level_off_rate_band: float = 500.0 * 0.3048 / 60.0
    # Impulse gust mode. impulse_prob = 0 (default) => the OU turbulence above. When
    # impulse_prob > 0 the gust is instead a rare, memoryless "kick": each step a kick
    # arrives with per-step probability impulse_prob and jumps the gust by ~sigma*N,
    # then it decays at rate theta between kicks. Arrivals are white (no autocorrelation),
    # but the kick has temporal extent, so a lookahead sensor can anticipate it while
    # memory can only track its decay -- the impulsive-disturbance analog of a robot shove.
    impulse_prob: float = 0.0
    # Linear wind shear: the horizontal wind gains ``wind_shear_x`` m/s per metre
    # of altitude above ``shear_ref_alt`` (0 => no shear).  Makes the wind
    # altitude-dependent, so climbing/descending is itself a disturbance.
    wind_shear_x: float = 0.0
    shear_ref_alt: float = 0.0

    delta_t: float = 1.0


#: Units of the PlaneParams fields, for the generated environment pages.
PARAM_UNITS = {
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
    "wind_shear_x": "1/s",
    "shear_ref_alt": "m",
    "e_floor": "m",
    "e_tol": "m",
    "c_hold": "m/s",
    "running_weight": "-",
    "speed_weight": "-",
    "target_cas": "m/s",
    "target_mach": "-",
    "speed_precision_floor": "m/s",
    "speed_envelope": "m/s",
    "target_pattern": "-",
    "target_amplitude": "m",
    "target_period": "s",
    "target_steps": "-",
    "target_chirp_octaves": "-",
    "impulse_prob": "1/step",
    "gust_scale_length": "m",
    "zone_onset_range": "s",
    "zone_duration_range": "s",
    "speed_band_cost": "-",
    "speed_band": "m/s",
    "cost_index": "kg/min",
    "economy_weight": "-",
    "economy_scale": "kg/s",
    "mach_band": "-",
    "clearance_period": "s",
    "level_change_rate_min": "m/s",
    "level_off_height": "m",
    "level_off_rate_max": "m/s",
    "level_off_rate_band": "m/s",
}


def check_mass_does_not_increase(old_mass, new_mass, xp=jnp):
    """Check that mass does not increase. Safe for JIT if wrapped in callback."""
    if jax is not None and xp is jnp:
        jax.debug.callback(
            lambda o, n: None if o >= n else AssertionError("Mass increased"),
            old_mass,
            new_mass,
        )
    else:
        assert old_mass >= new_mass


def check_is_terminal(state: PlaneState, params: PlaneParams, xp=jnp):
    """The aircraft trips only when it reaches the ground.

    Returns ``(tripped, truncated)``. A trip does not end the episode:
    ``base.failure_kernel`` charges it and restarts the aircraft.
    """
    # Load factor, AoA and overspeed are protected or warned, never trips
    # (rule 14; REALITY.md, limits); the ceiling is an operating limit. The
    # ground is the one loss.
    terminated = state.z <= params.min_alt
    truncated = state.time >= params.max_steps_in_episode

    # done = xp.logical_or(done_alt, done_steps)
    return terminated, truncated


def check_no_nan(x, id=None):
    """Assert that no NaNs are present in arrays, scalars, or PlaneState."""
    if isinstance(x, PlaneState):
        # Iterate over fields of the dataclass
        for name, value in x.__dict__.items():
            try:
                check_no_nan(value, id=f"{id}.{name}" if id else name)
            except AssertionError as e:
                raise AssertionError(str(e)) from None
    else:
        if jnp.isnan(x).any():
            raise AssertionError(f"NaN detected in {id}: {x}")


#: ``plane_energy``'s clearances, one per ``clearance_period`` from the start:
#: the level, ft, and the speed, kt, above the starting ones. Ours: the order
#: and the sizes, chosen so each kind comes in both directions and the levels
#: stay inside the altitude range. Sourced: levels move in whole 1 000 ft
#: [RVSM's separation minimum], speeds in 10 kt [AIM 4-4-12].
CLEARANCE_LEVELS_FT = (0.0, 2000.0, 2000.0, 1000.0, 1000.0, -1000.0)
CLEARANCE_SPEEDS_KT = (0.0, 0.0, 10.0, 10.0, 0.0, 0.0)


def _level_changes():
    """For each tread, the tread of the latest level change and its size, ft."""
    last, size = [0], [0.0]
    for k in range(1, len(CLEARANCE_LEVELS_FT)):
        moved = CLEARANCE_LEVELS_FT[k] != CLEARANCE_LEVELS_FT[k - 1]
        last.append(k if moved else last[-1])
        size.append(
            abs(CLEARANCE_LEVELS_FT[k] - CLEARANCE_LEVELS_FT[k - 1])
            if moved
            else size[-1]
        )
    return jnp.array(last, jnp.float32), jnp.array(size, jnp.float32)


_LAST_LEVEL_CHANGE, _LEVEL_CHANGE_FT = _level_changes()


def clearance_tread(time, params, xp=jnp):
    """Which clearance is in force at step ``time``."""
    t = xp.asarray(time, dtype=jnp.float32) * params.delta_t
    k = xp.floor(t / params.clearance_period)
    return xp.asarray(xp.clip(k, 0, len(CLEARANCE_LEVELS_FT) - 1), dtype=jnp.int32)


def assigned_cas(time, params, xp=jnp):
    """The CAS assigned at step ``time``: the schedule's, plus the clearance's
    change on ``plane_energy``."""
    extra = jnp.array(CLEARANCE_SPEEDS_KT)[clearance_tread(time, params, xp)]
    return params.target_cas + xp.where(
        params.target_pattern == 5, extra * 1852.0 / 3600.0, 0.0
    )


def level_change_allowance(time, params, xp=jnp):
    """How far from its level the aircraft may still be at step ``time``, m:
    the band about a level that closes at 500 fpm from the latest level
    clearance [AIM 4-4-10], so the band travels with that profile and a prompt
    start costs nothing."""
    k = clearance_tread(time, params, xp)
    since = xp.asarray(time, jnp.float32) * params.delta_t - (
        _LAST_LEVEL_CHANGE[k] * params.clearance_period
    )
    left = _LEVEL_CHANGE_FT[k] * 0.3048 - params.level_change_rate_min * since
    on_clearances = params.target_pattern == 5
    return params.e_tol + xp.where(on_clearances, xp.maximum(left, 0.0), 0.0)


def target_airspeed(state: PlaneState, params: PlaneParams):
    """The airspeed demanded at the aircraft's altitude."""
    cas = assigned_cas(state.time, params)
    return scheduled_airspeed(state.z, cas, params.target_mach)


def compute_reward_terms(state: PlaneState, params: PlaneParams, xp=jnp, tripped=None):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    # With ``speed_band_cost`` (``plane``, Amendment 2) the score is the job
    # (rule 15; REALITY.md, the job): the altitude inside +-65 ft [RVSM], which
    # is ``e_tol = e_floor = 65 ft``, and the assigned speed inside +-10 kt or
    # +-0.02 M [AIM 4-4-12], both charged as the squared excess past the edge
    # in band widths, per second. Without it, the earlier ground-speed excess
    # over ``c_hold`` (``plane_sine``, ``plane_energy`` until their change).
    ground = xp.sqrt(state.x_dot**2 + state.z_dot**2)
    legacy = R.excess_cost(
        xp.abs(ground - target_airspeed(state, p)), p.c_hold, p.running_weight, xp
    )
    cas = assigned_cas(state.time, p, xp)
    band = speed_band_excess(state.z, airspeed(state, p, xp), p, xp, cas=cas) ** 2
    # On the clearances (``plane_energy``) the altitude band trails a 500 fpm
    # climb or descent from each level clearance [AIM 4-4-10]; elsewhere the
    # allowance is the band itself and this is plain tracking past ``e_tol``.
    error = xp.abs(state.target_altitude - state.z)
    # Past the allowance, measured from ``e_tol`` so the shared cost reads it;
    # ahead of the profile is zero, never wrapped by the cost's absolute value.
    allowance = level_change_allowance(state.time, p, xp)
    behind = xp.maximum(error - allowance, 0.0) + p.e_tol
    terms = {
        "tracking": R.tracking_cost(
            behind, p.e_floor, p.e_tol, p.tracking_exponent, xp
        ),
        "running": xp.where(p.speed_band_cost, band, legacy),
    }
    # Within 1 000 ft of the level, no more than 1 500 fpm [AIM 4-4-10].
    fast = xp.maximum(xp.abs(state.z_dot) - p.level_off_rate_max, 0.0)
    near = error < p.level_off_height
    terms["level_off"] = xp.where(
        (p.target_pattern == 5) & near, (fast / p.level_off_rate_band) ** 2, 0.0
    )
    terms["economy"] = economy_cost(
        state.power, state.z, airspeed(state, p, xp), target_airspeed(state, p), p, xp
    )
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def airspeed(state, params, xp=jnp):
    """True airspeed, against the wind the aircraft is in, gust included: what
    the air data computer reads [BEA447, ADR], which sees a horizontal gust at
    once (COLD_REVIEW A6). Its lag is not found (REALITY.md), so none."""
    wind_x, wind_z = total_wind_2d(state.z, state.gust_x, state.gust_z, params)
    return xp.sqrt((state.x_dot - wind_x) ** 2 + (state.z_dot - wind_z) ** 2)


def compute_reward_v1(state: PlaneState, params: PlaneParams, xp=jnp):
    """Log-scaled altitude tracking, optionally coupled to an airspeed hold.

    One over the commanded altitude when it is held exactly, decaying so that
    every halving of the error is worth the same. When ``speed_weight`` is
    above zero the airspeed term multiplies it, so the two must be satisfied
    together; at ``speed_weight = 0`` the factor is exactly 1.0 and the task is
    altitude alone. Safe for JIT.
    """
    xp = jnp
    # Log-scaled tracking: every halving of the error is worth the same, so
    # holding 1 m is rewarded over 2 m exactly as much as 100 m is over 200 m.
    # The altitude envelope only normalises the result into [0, 1]; unlike the
    # old reward it does not set the sensitivity, because the shape is
    # logarithmic and therefore scale-free.
    tracking = log_scaled_reward(
        xp.abs(state.target_altitude - state.z),
        params.precision_floor,
        params.max_alt - params.min_alt,
        xp,
    )
    # No explicit crash penalty. Termination already costs the agent every
    # step it would otherwise have earned, and since the reward is
    # non-negative everywhere that is strictly worse than flying on. A
    # large negative spike bought nothing the forgone reward did not, and
    # left this family on a different contract from the twelve process
    # plants, which have always relied on forgone reward alone.
    #
    # Airspeed, when it is being scored, multiplies rather than adds -- the
    # convention every environment here follows. Holding the speed while
    # abandoning the altitude earns nothing, which is the point: the two are
    # one objective the aircraft must satisfy together, not two it can pick
    # between. At speed_weight = 0 this is exactly 1.0 and the task is
    # unchanged.
    speed = xp.sqrt(state.x_dot**2 + state.z_dot**2)
    speed_term = log_scaled_reward(
        xp.abs(target_airspeed(state, params) - speed),
        params.speed_precision_floor,
        params.speed_envelope,
        xp,
    )
    return tracking * (1.0 - params.speed_weight * (1.0 - speed_term))


def compute_reward(state: PlaneState, params: PlaneParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def get_obs(state: PlaneState, params: PlaneParams = None, xp=jnp):
    """Applies observation function to state."""
    if params is None:
        raise ValueError(
            "get_obs needs the episode's params: the airspeed schedule is a "
            "parameter, so falling back to PlaneParams() would silently report "
            "the class default instead of what this episode is commanding."
        )
    params_target_speed = jnp.asarray(target_airspeed(state, params), jnp.float32)
    return xp.stack(
        [
            state.x_dot,
            state.z,
            state.z_dot,
            state.theta,
            state.theta_dot,
            state.gamma,
            state.target_altitude,
            state.power,
            state.stick,
            # Commanded airspeed, appended rather than inserted. Putting it
            # beside target_altitude would have shifted power and stick from
            # indices 7 and 8 to 8 and 9, silently breaking every consumer that
            # reads the observation positionally -- the PIDs do. It is present
            # whether or not it is scored, so the shape does not depend on a
            # reward weight.
            params_target_speed,
            # What the autopilot reads from air data and the IRS (rule 16;
            # REALITY.md, readings): airspeed and normal acceleration.
            airspeed(state, params, xp),
            state.n_z,
        ]
    )


@partial(jax.jit, static_argnames=["min", "max"])
def clip_acceleration(a: jnp.ndarray, min: tuple, max: tuple):
    return jnp.clip(a, min=jnp.array(min), max=jnp.array(max))


#: The levels the ``steps`` pattern walks, as fractions of ``target_amplitude``.
#: Eight of them, so the schedule does not repeat inside an episode, with
#: adjacent changes between 0.2 and 0.8 of the amplitude.
_STEP_LEVELS = jnp.array([0.0, 0.5, 0.1, -0.3, 0.3, -0.5, -0.1, 0.4])


def commanded_altitude(base: float, time, params: PlaneParams, xp=jnp):
    """The altitude being commanded at this step.

    ``base`` is the altitude sampled at reset; every moving pattern is written
    as an excursion about it, so the aircraft always starts being asked for
    something it can reach and the patterns stay inside the altitude envelope.

    Branch-free on purpose: all five are evaluated and one is selected, so the
    function traces to a single ``select`` and the pattern can live in params
    rather than forcing a separate environment class per shape.
    """
    t = xp.asarray(time, dtype=jnp.float32) * params.delta_t
    span = params.target_amplitude
    total = xp.maximum(params.max_steps_in_episode * params.delta_t, 1.0)

    hold = base
    tread = xp.floor(t / (total / params.target_steps))
    # A ladder of levels, not a square wave. Alternating between two altitudes
    # is a wave with a flat top, and at the excursion this used to run it read
    # as an aircraft being thrown between two extremes rather than an aircraft
    # being given new levels to hold. The ladder revisits neither level nor
    # direction on a two-tread cycle, so consecutive changes differ in size and
    # sign the way a real clearance sequence does, and the largest adjacent
    # change is 0.8 of the amplitude rather than 1.2.
    idx = xp.asarray(xp.mod(tread, float(len(_STEP_LEVELS))), dtype=jnp.int32)
    steps = base + span * _STEP_LEVELS[idx]
    ramp = base + span * (2.0 * t / total - 1.0)
    sine = base + span * xp.sin(2.0 * xp.pi * t / params.target_period)
    rate = 1.0 + (params.target_chirp_octaves - 1.0) * (t / total)
    chirp = base + span * xp.sin(2.0 * xp.pi * t * rate / params.target_period)

    level = jnp.array(CLEARANCE_LEVELS_FT)[clearance_tread(time, params, xp)]
    cleared = base + level * 0.3048
    options = xp.stack([hold * xp.ones_like(t), steps, ramp, sine, chirp, cleared])
    target = options[params.target_pattern]
    return xp.clip(target, params.min_alt + span * 0.2, params.max_alt - span * 0.2)


@partial(jax.jit, static_argnames=["integration_method", "inner_steps"])
def compute_next_state(
    power_requested: float,
    pitch_demand: float,
    state: PlaneState,
    params: PlaneParams,
    integration_method: str = "rk4_1",
    key=None,
    inner_steps: int = INNER_STEPS,
):
    """One step: the demands held while the inner loop flies ``inner_steps``
    sub-steps, each moving the elevator and the engine and integrating the
    dynamics. The surfaces are the loop's, never the caller's.

    Wind is a physics-engine property: the total wind acting on the aircraft is
    the steady ``params.wind_*`` plus an Ornstein-Uhlenbeck turbulence gust
    (advanced with ``key`` when ``params.turbulence_sigma > 0``), and the
    aerodynamics/engine-Mach use the air-relative velocity while
    position/observations stay in the ground frame.  With ``wind = 0`` and
    ``turbulence_sigma = 0`` the behaviour is unchanged; ``key=None`` freezes the
    gust (so callers that don't model turbulence keep a constant wind).
    """
    dt = params.delta_t / inner_steps

    # Total wind = steady mean + altitude shear + OU turbulence gust. `gust` is the
    # ADVANCED gust (stored for next step). With wind_lookahead we APPLY the gust
    # already in the state (which the sensor revealed before this action -> the
    # agent can pre-cancel it); otherwise we apply the advanced gust (post-hoc).
    # Dryden gust in a zone, calm outside it (PHYSICS.md, turbulence). The
    # impulse mode is the old kick model, kept for the Extra variants.
    previous = jnp.array([state.gust_x, state.gust_z])
    zone = in_zone(state.time * params.delta_t, state.zone_start, state.zone_end)
    dryden = advance_dryden_gust(
        previous,
        compute_velocity_from_horizontal_and_vertical_speed(
            state.x_dot - params.wind_x - previous[0],
            state.z_dot - params.wind_z - previous[1],
        ),
        jnp.where(zone, params.turbulence_sigma, 0.0),
        params.gust_scale_length,
        params.delta_t,
        step_key(key, state.time),
    )
    kicked = advance_gust(
        previous,
        params.turbulence_theta,
        params.turbulence_sigma,
        params.delta_t,
        step_key(key, state.time),
        params.impulse_prob,
    )
    gust = jnp.where(params.impulse_prob > 0.0, kicked, dryden)
    # With look-ahead the gust already in the state is the one that acts now;
    # the freshly advanced one is stored for the next step.
    applied = jnp.where(
        params.wind_lookahead, jnp.array([state.gust_x, state.gust_z]), gust
    )
    total_wind_x, total_wind_z = total_wind_2d(state.z, applied[0], applied[1], params)
    # All-up mass for this step. ``initial_fuel_quantity`` is a component of
    # ``initial_mass``, not an addition, so burning fuel subtracts from it.
    mass_now = params.initial_mass - (params.initial_fuel_quantity - state.fuel)
    eff_params = params.replace(
        wind_x=total_wind_x, wind_z=total_wind_z, initial_mass=mass_now
    )
    gains = params.inner_loop

    def inner(carry, _):
        velocities, positions, power, stick, integ, burned, peak = carry
        x_dot, z_dot, theta_dot = velocities
        _, z, theta = positions
        # Air data and the engine's ram/Mach effects read airspeed.
        air_speed = compute_velocity_from_horizontal_and_vertical_speed(
            x_dot - total_wind_x, z_dot - total_wind_z
        )
        rho = compute_air_density_from_altitude(z)
        eas = air_speed * jnp.sqrt(rho / params.air_density_at_sea_level)
        demand = protected_pitch_demand(
            pitch_demand,
            jnp.arctan2(z_dot - total_wind_z, x_dot - total_wind_x),
            air_speed,
            z,
            mass_now,
            params,
        )
        surface, integ = attitude_axis(
            gains.Kp_theta,
            gains.Ki_theta,
            gains.Kd_theta,
            demand - theta,
            theta_dot,
            integ,
            airspeed_scaler(eas, gains),
            dt,
        )
        stick = compute_next_surface(
            surface * STICK_TRAVEL,
            stick,
            dt,
            params.stick_response_rate,
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
            compute_acceleration,
            action=(thrust, stick),
            params=eff_params,
            clip=True,
            min_clip_boundaries=(-100, -100, -1.5),
            max_clip_boundaries=(100, 100, 1.5),
        )
        velocities, positions, metrics = integrate_dynamics(
            velocities=velocities,
            positions=positions,
            delta_t=dt,
            compute_acceleration=accelerate,
            method=integration_method,
        )
        # Fuel burned over the sub-step. ``specific_fuel_consumption`` is
        # thrust specific: kg per kilonewton-second, so 17.5/1000 is 1.75e-5
        # kg/(N s), the right order for a modern high-bypass turbofan. Burn is
        # charged against the thrust actually produced, which already carries
        # the altitude lapse and the Mach terms.
        burned = burned + params.specific_fuel_consumption * 1e-3 * thrust * dt
        # Trips read the worst load inside the step, not the one it ends on.
        load = metrics[1][-1] / (mass_now * params.gravity)
        peak = jnp.where(jnp.abs(load - 1.0) > jnp.abs(peak - 1.0), load, peak)
        return (velocities, positions, power, stick, integ, burned, peak), metrics

    start = (
        jnp.array([state.x_dot, state.z_dot, state.theta_dot]),
        jnp.array([state.x, state.z, state.theta]),
        state.power,
        state.stick,
        state.pitch_integ,
        jnp.zeros_like(state.fuel),
        jnp.ones_like(state.fuel),
    )
    end, metrics = jax.lax.scan(inner, start, None, length=inner_steps)
    (x_dot, z_dot, theta_dot), (x, z, theta) = end[0], end[1]
    power, stick, pitch_integ, burned, load_factor = end[2:]
    metrics = jax.tree_util.tree_map(lambda m: m[-1], metrics)

    alpha, gamma = compute_alpha(theta, x_dot, z_dot)
    fuel = jnp.clip(state.fuel - burned, 0.0, params.initial_fuel_quantity)
    # Below the tanks running dry the engines cannot produce thrust; the mass
    # simply stops falling, and the aircraft is a glider from there.
    m = params.initial_mass - (params.initial_fuel_quantity - fuel)

    new_state = PlaneState(
        x=x,
        x_dot=x_dot,
        z=z,
        z_dot=z_dot,
        theta=theta,
        theta_dot=theta_dot,
        alpha=alpha,
        gamma=gamma,
        m=m,
        power=power,
        stick=stick,
        fuel=fuel,
        time=state.time + 1,
        sensor_key=state.sensor_key,
        actuator_held=state.actuator_held,
        target_altitude=commanded_altitude(
            state.base_target_altitude, state.time + 1, params
        ),
        base_target_altitude=state.base_target_altitude,
        gust_x=gust[0],
        gust_z=gust[1],
        pitch_integ=pitch_integ,
        load_factor=load_factor,
        n_z=metrics[1][-1] / (mass_now * params.gravity),
        zone_start=state.zone_start,
        zone_end=state.zone_end,
    )
    return new_state, metrics
