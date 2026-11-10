"""
Wind turbine: the NREL 5 MW reference turbine above rated wind, with its tower.

See ``PHYSICS.md`` in this directory for every number's source, and
``REALITY.md`` for the plant record ([Sn] tags below point there).

Model
-----
A rigid drivetrain on a tower whose top moves fore-aft in its first bending
mode, both driven by the rotor's aerodynamics in the wind the rotor sees, the
free wind less the tower top's velocity::

    J dw/dt   = tau_aero - N tau_gen
    m x''     = F_thrust - c x' - k x
    v_rel     = v - x'
    tau_aero  = 0.5 rho A v_rel^3 Cp(lambda, beta) / w      lambda = w R / v_rel
    F_thrust  = 0.5 rho A v_rel^2 Ct(lambda, beta)
    P_elec    = eta N tau_gen w

Cp and Ct are ROSCO's published tables for this rotor (``cp_table``). The
tower is S1's first fore-aft mode (0.3240 Hz, 1 % structural damping) on the
modal mass of S16. Pitch and generator torque move toward their commands at
S1's rate limits. Above about 20 % overspeed [S17] the safety system feathers
the blades at the pitch rate, whatever the controller sends [S4], and the
turbine is lost to a shutdown and restart.

Task
----
Above rated wind, hold rated power with pitch and torque without overspeeding,
while the thrust swings the tower (REALITY.md sec. 6). The score is the energy
delivered off the rating, in MWh, and a trip costs the production of a restart.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.scipy.ndimage import map_coordinates
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.energy.wind_turbine.cp_table import (
    CP_MAX,
    CP_TABLE_FEATHER,
    CT_TABLE_FEATHER,
    PITCH_STEP,
    TSR_FIRST,
    TSR_STEP,
)
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

RPM_PER_RAD_S = 60.0 / (2.0 * jnp.pi)

#: The extreme operating gust's duration, s (IEC 61400-1 via [S13]).
EOG_SECONDS = 10.5
#: Salt for the mean wind's draw, apart from the turbulence's ``fold_in(key, t)``.
_MEAN_WIND = 0x4D45414E


@struct.dataclass
class WindTurbineParams(EnvParams):
    # ---- Rotor [S1 Table 1-1] ----
    R: float = 63.0  # m, rotor radius (126 m diameter)
    rho: float = 1.225  # kg/m^3, air density

    # ---- Dynamic inflow: Oye's lag on the rotor's induced velocity ----
    # tau1 = 1.1 / (1 - 1.3 min(a, 0.5)) R / U0, as OpenFAST's AeroDyn DBEMT
    # writes it (REALITY.md [S22]); one lumped rotor-average lag, so tau2 and
    # its radial split are not modelled. 1 on, 0 the quasi-steady tables.
    dynamic_inflow: float = 1.0
    oye_k1: float = 1.1
    oye_k2: float = 1.3

    # ---- Drivetrain [S1 Table 5-1] ----
    J_rotor: float = 38_759_228.0  # kg m^2
    J_gen: float = 534.116  # kg m^2, generator side
    N_gear: float = 97.0  # gearbox ratio
    eta_gen: float = 0.944  # generator efficiency

    # ---- Ratings [S1 Table 1-1, p. 19] ----
    P_rated: float = 5.0e6  # W
    v_rated: float = 11.4  # m/s
    omega_rated_rpm: float = 12.1
    v_cut_out: float = 25.0  # m/s

    # ---- Tower, first fore-aft mode ----
    #: First tower fore-aft frequency, Hz [S1 Table 9-1, FAST].
    tower_frequency_hz: float = 0.3240
    #: Rotor plus nacelle plus a quarter of the tower, 110,000 + 240,000 +
    #: 0.25 x 347,460 kg [S16 Sect. 2.1.1 and App. B; masses from S1 Table 1-1].
    tower_modal_mass: float = 436_865.0
    #: Structural damping, fraction of critical [S1 Sec. 6, Table 6-2].
    tower_damping_ratio: float = 0.01

    # ---- Actuators [S1 p. 26, Table 7-2] ----
    pitch_min: float = 0.0  # deg
    pitch_max: float = 90.0  # deg, feather
    pitch_rate_max: float = 8.0  # deg/s
    torque_max: float = 47_402.91  # N m, generator side, 10 % above rated
    torque_rate_max: float = 15_000.0  # N m/s

    # ---- Wind (IEC 61400-1 normal turbulence, via [S13], [S19]) ----
    #: Mean hub wind range, m/s: S2's above-rated load cases, 13 to 25 m/s.
    v_mean_range: Tuple[float, float] = (13.0, 25.0)
    #: Turbulence intensity reference (ours: class B, the NREL 5 MW's class is
    #: not in REALITY.md; the owner picks the class).
    turbulence_iref: float = 0.14
    #: Turbulence scale parameter Lambda1, m, above 60 m hub height [S13].
    turbulence_lambda1: float = 42.0
    #: Kaimal longitudinal length scale over Lambda1 [S19].
    kaimal_factor: float = 8.1
    #: The mean wind's correlation time in a stream, s: its score is an
    #: Ornstein-Uhlenbeck process of this time, and the mean wind its quantile
    #: on ``v_mean_range``, uniform in the long run as a window draws it
    #: (``mean_wind``). 0, as a window has it: the mean wind holds its draw.
    v_mean_tau: float = 0.0
    #: Reference wind of the class, m/s (ours: class I, 50 m/s [S13]); only
    #: the gust's 1.35 (Ve1 - V) branch reads it, which never binds here.
    v_ref: float = 50.0
    #: 1 for one extreme operating gust in the episode, 0 for none. A 1-year
    #: event [S13], so none on a normal day; a scenario of its own.
    gust: float = 0.0
    #: When it starts, s, drawn by the seed (ours: anywhere a full gust fits
    #: after the first minute).
    gust_window: Tuple[float, float] = (60.0, 580.0)

    # ---- Safety system ----
    #: Overspeed level as a multiple of rated speed: "approximately 20 %
    #: above the rated speed of the generator" [S17]. Past it the pitch
    #: system feathers independently of the controller [S4]: a trip.
    overspeed_factor: float = 1.20

    # ---- Episode ----
    #: Fraction of rated power the turbine is asked for (ROSCO's rating R,
    #: [S9 Ex. 29]); 1 on a normal day [S1 Sec. 7]. Read by the agent.
    rating: float = 1.0
    #: A curtailment: the rating falls to ``derate_to`` at a time drawn in
    #: ``derate_window`` (s), a de-rate step [S11 Fig. ES-8] to a set point
    #: followed within a TSO-set time [S10 Art. 13(6)]. 1 for none. The size
    #: is ours: 10 %, S11's continuous AGC de-rate (p. xix).
    derate_to: float = 1.0
    derate_window: Tuple[float, float] = (60.0, 300.0)
    #: How long a curtailment holds before the rating returns, s. Only a stream
    #: sets it (scenarios.py); in a window the de-rate holds to the end.
    derate_hold: float = 1.0e9
    #: 0.25 s steps (ours: controllers run at 80 to 100 Hz [S1, S3]; 12 per
    #: tower period).
    delta_t: float = 0.25
    #: 10 min: one SCADA interval [S7] and one load case [S2 Table 2].
    max_steps_in_episode: int = 2400

    # ---- Score (rule 15), in dollars: lost production, tower and pitch wear ----
    reward_version: int = 4
    #: Value of lost production, $/MWh: ours, inside the family's LCOE range,
    #: 60 to 140 $/MWh for new onshore plants [S20 Table 1].
    energy_price: float = 60.0
    #: Installed cost, $/kW: ours, inside 1,300 to 2,200 [S20 Table 1].
    capex_per_kw: float = 1750.0
    #: The turbine's share of it: ours, inside 65 to 84 % [S20 Table 4.1].
    turbine_share: float = 0.75
    #: The tower's and the pitch system's shares of the turbine [S20, REpower
    #: MM92 breakdown]: 26.3 % and 2.66 %.
    tower_share: float = 0.263
    pitch_share: float = 0.0266
    #: Design life, years [S20, the LCOE's 20 year lifetime].
    design_life_years: float = 20.0
    #: Tower top height, m [S1 Table 6-2], the lever of the base moment.
    tower_height: float = 87.6
    #: S-N slope for the damage-equivalent load: the average of the welded
    #: tower curve's two slopes, 3 and 5 (DNV-RP-C203 type E) [S23].
    fatigue_exponent: float = 4.0
    #: The design rates, (MN m)^m and deg per step: 20 years at these use the
    #: tower and the pitch system up exactly. Ours, measured once from S1's
    #: controller on seeds 100 to 107 of the normal scenario (PHYSICS.md).
    tower_damage_design: float = 2.126e5
    pitch_travel_design: float = 0.1754
    #: A trip's restart: an overspeed trip needs a manual reset [S21]; ours,
    #: 2 h, between ROSCO's automatic 400 s start-up [S9 Ex. 32] and a site
    #: visit. At 0.25 s steps.
    restart_steps: int = 28800
    #: The manual reset's callout, $ (ours: no family figure found).
    reset_callout: float = 1000.0
    #: Per step of a restart, $: rated production lost at ``energy_price``,
    #: plus the callout spread over the restart.
    failure_cost: float = 5.0 * 0.25 / 3600.0 * 60.0 + 1000.0 / 28800.0
    # Kept for the time-in-band KPI tables (benchmark._OUTPUT_FLOORS) and the
    # oracle's resolution; the score does not read them.
    e_floor: float = 1680.0  # W
    precision_floor: float = 1.7e3  # W, a class 0.1S meter at 5 MW


#: Units of the WindTurbineParams fields, for the generated environment pages.
PARAM_UNITS = {
    "R": "m",
    "dynamic_inflow": "-",
    "oye_k1": "-",
    "oye_k2": "-",
    "rho": "kg/m^3",
    "J_rotor": "kg m^2",
    "J_gen": "kg m^2",
    "N_gear": "-",
    "eta_gen": "-",
    "P_rated": "W",
    "v_rated": "m/s",
    "omega_rated_rpm": "rpm",
    "v_cut_out": "m/s",
    "tower_frequency_hz": "Hz",
    "tower_modal_mass": "kg",
    "tower_damping_ratio": "-",
    "pitch_min": "deg",
    "pitch_max": "deg",
    "pitch_rate_max": "deg/s",
    "torque_max": "N m",
    "torque_rate_max": "N m/s",
    "turbulence_iref": "-",
    "turbulence_lambda1": "m",
    "kaimal_factor": "-",
    "v_mean_tau": "s",
    "v_ref": "m/s",
    "gust": "-",
    "derate_to": "-",
    "derate_hold": "s",
    "energy_price": "$/MWh",
    "capex_per_kw": "$/kW",
    "turbine_share": "-",
    "tower_share": "-",
    "pitch_share": "-",
    "design_life_years": "years",
    "tower_height": "m",
    "fatigue_exponent": "-",
    "tower_damage_design": "(MN m)^m per step",
    "reset_callout": "$",
    "pitch_travel_design": "deg per step",
    "overspeed_factor": "-",
    "rating": "-",
    "reward_version": "-",
    "restart_steps": "steps",
    "failure_cost": "$ per step",
    "e_floor": "W",
    "precision_floor": "W",
}


@struct.dataclass
class WindTurbineState(EnvState):
    omega: float  # rotor speed (rad/s)
    pitch: float  # blade pitch (deg)
    torque: float  # generator torque (N m)

    v_wind: float  # rotor-effective free wind (m/s)
    v_mean: float  # the episode's mean wind (m/s)

    torque_cmd: float
    #: The rating's power, ``rating * P_rated`` (W).
    target_power: float
    #: Electrical output (W), always ``electrical_power(omega, torque)``. A field
    #: so the meter can be its own reading.
    power: float
    #: Tower-top fore-aft displacement (m), velocity (m/s), acceleration (m/s^2).
    x_tower: float = 0.0
    v_tower: float = 0.0
    a_tower: float = 0.0
    #: The rotor-average induced velocity (m/s), which lags its quasi-steady
    #: value by Oye's tau1 (``induction``).
    w_induced: float = 0.0
    #: The turbulence's deviation from the mean (m/s), without the gust.
    v_turb: float = 0.0
    #: 1 once the safety system has fed the blades to feather (a trip).
    feathered: float = 0.0
    #: When this episode's gust and curtailment start, s (-1 for none).
    gust_t0: float = -1.0
    derate_t0: float = -1.0
    #: The mean wind's standard-normal score: ``v_mean`` is its quantile on the
    #: scenario's ``v_mean_range`` (``mean_wind``).
    v_mean_z: float = 0.0
    #: Each event carries its own size, so it runs to its end through a switch
    #: (docs/proposals/long-strata.md, S1): the gust's V_gust as it started (m/s;
    #: 0 reads it off the params, as a window does), the de-rate's rating and when
    #: it returns (s).
    gust_size: float = 0.0
    derate_level: float = 1.0
    derate_until: float = 1.0e9
    #: The steps the gust's and the de-rate's times count from: 0 in a window; a
    #: stream sets each at its event's onset, so float32 keeps them exact.
    gust_c0: jax.Array = 0
    derate_c0: jax.Array = 0
    #: Tower-base moment (N m) last step and at the last turning point, its
    #: direction (+1, -1, 0), and the damage the last step closed, (MN m)^m:
    #: half-cycles counted between turning points (PHYSICS.md, Score).
    moment: float = 0.0
    moment_turn: float = 0.0
    moment_dir: float = 0.0
    damage: float = 0.0
    #: How far the pitch moved over the last step (deg).
    pitch_travel: float = 0.0

    actuator_held: jax.Array = actuator_field(2)


def rotor_area(params: WindTurbineParams):
    return jnp.pi * params.R**2


def omega_rated(params: WindTurbineParams):
    return params.omega_rated_rpm * 2.0 * jnp.pi / 60.0


def tower_stiffness(params: WindTurbineParams):
    """k = m (2 pi f)^2: the modal stiffness that puts the mode at f."""
    return params.tower_modal_mass * (2.0 * jnp.pi * params.tower_frequency_hz) ** 2


def tower_damping(params: WindTurbineParams):
    """c = 2 zeta sqrt(k m), N s/m."""
    return (
        2.0
        * params.tower_damping_ratio
        * jnp.sqrt(tower_stiffness(params) * params.tower_modal_mass)
    )


def _table(table, tip_speed_ratio, pitch_deg):
    """Bilinear between the table's nodes, held at its edges (beyond 30 deg
    and outside 2 to 14.5 in tip speed ratio, PHYSICS.md, Deviations)."""
    lam, beta = jnp.broadcast_arrays(tip_speed_ratio, pitch_deg)
    return map_coordinates(
        jnp.asarray(table, lam.dtype),
        [(lam - TSR_FIRST) / TSR_STEP, beta / PITCH_STEP],
        order=1,
        mode="nearest",
    )


def power_coefficient(tip_speed_ratio, pitch_deg, params: WindTurbineParams):
    """Cp(lambda, beta) from ROSCO's table. Negative where the rotor brakes."""
    return _table(CP_TABLE_FEATHER, tip_speed_ratio, pitch_deg)


def thrust_coefficient(tip_speed_ratio, pitch_deg, params: WindTurbineParams):
    """Ct(lambda, beta) from ROSCO's table."""
    return _table(CT_TABLE_FEATHER, tip_speed_ratio, pitch_deg)


def aerodynamic_torque(omega, v_wind, pitch_deg, params: WindTurbineParams):
    """Rotor torque and power in the wind ``v_wind`` the rotor sees."""
    w = jnp.maximum(omega, 1e-3)
    v = jnp.maximum(v_wind, 1e-3)
    cp = power_coefficient(w * params.R / v, pitch_deg, params)
    power = 0.5 * params.rho * rotor_area(params) * v**3 * cp
    return power / w, power


def rotor_thrust(omega, v_wind, pitch_deg, params: WindTurbineParams):
    """Rotor thrust (N) in the wind ``v_wind`` the rotor sees."""
    w = jnp.maximum(omega, 1e-3)
    v = jnp.maximum(v_wind, 1e-3)
    ct = thrust_coefficient(w * params.R / v, pitch_deg, params)
    return 0.5 * params.rho * rotor_area(params) * v**2 * ct


def induction(omega, v_wind, pitch_deg, params: WindTurbineParams):
    """The quasi-steady rotor-average induced velocity (m/s) and Oye's time
    constant (s), from the thrust table: momentum theory, a = (1 - sqrt(1 -
    Ct)) / 2, up to Ct = 0.96, and above it Buhl's empirical correction, a =
    0.143 + sqrt(0.0203 - 0.6427 (0.889 - Ct)), which meets it there in value
    and slope (as AeroDyn applies it, REALITY.md [S22])."""
    w = jnp.maximum(omega, 1e-3)
    v = jnp.maximum(v_wind, 1e-3)
    # No floor at Ct = 0: momentum theory runs smoothly into the propeller
    # state (negative Ct and induction) at high tip-speed ratios.
    ct = thrust_coefficient(w * params.R / v, pitch_deg, params)
    low = jnp.minimum(ct, 0.96)
    high = jnp.maximum(ct, 0.96)
    a = jnp.where(
        ct <= 0.96,
        0.5 * (1.0 - jnp.sqrt(1.0 - low)),
        0.143 + jnp.sqrt(0.0203 - 0.6427 * (0.889 - high)),
    )
    tau = params.oye_k1 / (1.0 - params.oye_k2 * jnp.minimum(a, 0.5)) * params.R / v
    return a * v, tau


def balance_pitch(omega, v_wind, power, params: WindTurbineParams):
    """The feather-side pitch leaving ``power`` (W) in the wind at this speed,
    to 0.01 deg: scanning up from fine pitch by 0.1 deg, then by 0.01 deg in
    the cell found. Two short scans, since every step's restart runs it."""
    coarse = jnp.arange(301.0) * 0.1
    _, available = aerodynamic_torque(omega, v_wind, coarse, params)
    top = coarse[jnp.argmax(available < power)]
    fine = jnp.maximum(top - 0.1, 0.0) + jnp.arange(11.0) * 0.01
    _, available = aerodynamic_torque(omega, v_wind, fine, params)
    return fine[jnp.argmax(available < power)]


def electrical_power(omega, torque, params: WindTurbineParams):
    """Generator electrical output; torque is generator-side."""
    return params.eta_gen * params.N_gear * torque * omega


def drivetrain_inertia(params: WindTurbineParams):
    """Inertia referred to the rotor shaft."""
    return params.J_rotor + params.N_gear**2 * params.J_gen


def rated_generator_torque(params: WindTurbineParams):
    """Generator torque that delivers rated power at rated speed."""
    return params.P_rated / (params.eta_gen * params.N_gear * omega_rated(params))


def turbulence_sigma(v_mean, params: WindTurbineParams):
    """sigma1 = Iref (0.75 V + 5.6), m/s [S13]."""
    return params.turbulence_iref * (0.75 * v_mean + 5.6)


def turbulence_time_constant(v_mean, params: WindTurbineParams):
    """The Kaimal spectrum's integral time scale, L_u / V, s: its value at zero
    frequency, 4 sigma^2 L_u / V, is an Ornstein-Uhlenbeck process's 4 sigma^2 T."""
    return params.kaimal_factor * params.turbulence_lambda1 / v_mean


def gust_magnitude(v_hub, params: WindTurbineParams):
    """The extreme operating gust's V_gust, m/s [S13]."""
    v_e1 = 0.8 * 1.4 * params.v_ref
    diameter = 2.0 * params.R
    return jnp.minimum(
        1.35 * (v_e1 - v_hub),
        3.3
        * turbulence_sigma(v_hub, params)
        / (1.0 + 0.1 * diameter / params.turbulence_lambda1),
    )


def gust(t, v_hub, t0, params: WindTurbineParams, size=0.0):
    """The gust's departure from the mean at time ``t`` (s), for a gust that
    starts at ``t0`` (none if negative) [S13]. ``size`` is its V_gust as it
    started (m/s); 0 reads it off the params at ``v_hub``."""
    tau = t - t0
    on = (t0 >= 0.0) & (tau >= 0.0) & (tau <= EOG_SECONDS)
    shape = jnp.sin(3.0 * jnp.pi * tau / EOG_SECONDS) * (
        1.0 - jnp.cos(2.0 * jnp.pi * tau / EOG_SECONDS)
    )
    magnitude = jnp.where(size > 0.0, size, gust_magnitude(v_hub, params))
    return jnp.where(on, -0.37 * magnitude * shape, 0.0)


def mean_wind(state: "WindTurbineState", key, params: WindTurbineParams):
    """The mean wind's next ``(score, value)``. In a stream (``v_mean_tau`` > 0)
    the score is an Ornstein-Uhlenbeck process, stationary N(0, 1), and the mean
    wind its quantile on ``v_mean_range``: uniform on the range in the long run,
    the law a window draws it from (docs/proposals/long-strata.md, E4). In a
    window both hold."""
    p = params
    tau = jnp.maximum(p.v_mean_tau, p.delta_t)
    a = jnp.exp(-p.delta_t / tau)
    eps = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, state.time), _MEAN_WIND)
    )
    z = a * state.v_mean_z + jnp.sqrt(-jnp.expm1(-2.0 * p.delta_t / tau)) * eps
    lo, hi = p.v_mean_range
    v = lo + (hi - lo) * jax.scipy.special.ndtr(z)
    moves = p.v_mean_tau > 0.0
    return jnp.where(moves, z, state.v_mean_z), jnp.where(moves, v, state.v_mean)


def tower_base_moment(x_tower, params: WindTurbineParams):
    """Fore-aft bending moment at the tower base, N m: the mode's elastic force
    at the top times the tower's height."""
    return tower_stiffness(params) * x_tower * params.tower_height


def count_half_cycles(state, moment, params):
    """Half-cycle counting between turning points of the base moment: when the
    moment turns, the half-cycle from the last turning point closes, with
    damage 0.5 (range / 1 MN m)^m. Returns (moment, last turn, direction,
    damage this step)."""
    step = moment - state.moment
    direction = jnp.sign(step)
    turned = (
        (state.moment_dir != 0.0) & (direction != 0.0) & (direction != state.moment_dir)
    )
    span = jnp.abs(state.moment - state.moment_turn) / 1.0e6
    damage = jnp.where(turned, 0.5 * span**params.fatigue_exponent, 0.0)
    turn = jnp.where(turned, state.moment, state.moment_turn)
    direction = jnp.where(direction == 0.0, state.moment_dir, direction)
    return moment, turn, direction, damage


def compute_velocity(position, action, v_wind, params: WindTurbineParams):
    """RHS for ``[omega, pitch, torque, x_tower, v_tower]``, and the induced
    velocity as a sixth state when given. ``action`` is the step's ``(pitch
    rate, torque rate)``, constant over the step.

    With the sixth state the tables, which hold equilibrium induction, are read
    at the wind that gives the rotor plane its lagging inflow, ``v_rel - (w -
    w_qs)``: an up-step in wind or a pitch move changes the loads at once and
    then relaxes as the wake catches up, Oye's overshoot.
    """
    p = params
    omega, pitch, torque, x, xdot = (position[i] for i in range(5))
    v_rel = v_wind - xdot
    rates = [action[0], action[1]]
    if position.shape[0] > 5:
        w_qs, tau = induction(omega, v_rel, pitch, p)
        w = position[5]
        v_eff = v_rel - p.dynamic_inflow * (w - w_qs)
        dw = (w_qs - w) / tau
    else:
        v_eff = v_rel
    tau_aero, _ = aerodynamic_torque(omega, v_eff, pitch, p)
    thrust = rotor_thrust(omega, v_eff, pitch, p)
    domega = (tau_aero - p.N_gear * torque) / drivetrain_inertia(p)
    accel = (
        thrust - tower_damping(p) * xdot - tower_stiffness(p) * x
    ) / p.tower_modal_mass
    out = [domega, *rates, xdot, accel]
    if position.shape[0] > 5:
        out.append(dw)
    return jnp.stack(out), None


def _ramp(command, now, rate, dt):
    """The constant rate that moves an actuator toward ``command`` at most at
    ``rate``, ending on it if it is within reach this step [S1 Table 7-2]."""
    return jnp.clip((command - now) / dt, -rate, rate)


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: jnp.ndarray,
    state: WindTurbineState,
    params: WindTurbineParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` is ``[pitch_raw, torque_raw]`` in [-1, 1]."""
    p = params
    pitch_cmd = convert_raw_action_to_range(
        action_raw[0], min_action=p.pitch_min, max_action=p.pitch_max
    )
    torque_cmd = convert_raw_action_to_range(
        action_raw[1], min_action=0.0, max_action=p.torque_max
    )
    # The safety system, once tripped, feathers whatever the controller sends [S4].
    tripped = state.feathered > 0.5
    pitch_cmd = jnp.where(tripped, p.pitch_max, pitch_cmd)

    # Normal turbulence about the mean: an Ornstein-Uhlenbeck process with the
    # Kaimal integral time scale and IEC's sigma, stepped exactly. Drawn from a
    # key folded with ``state.time``, so a constant key still gives a process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    decay = jnp.exp(-p.delta_t / turbulence_time_constant(state.v_mean, p))
    sigma = turbulence_sigma(state.v_mean, p)
    v_turb = state.v_turb * decay + sigma * jnp.sqrt(1.0 - decay**2) * noise
    t_gust = (state.time + 1 - state.gust_c0) * p.delta_t
    v_wind = jnp.maximum(
        state.v_mean
        + v_turb
        + gust(t_gust, state.v_mean, state.gust_t0, p, state.gust_size),
        0.1,
    )  # 0.1 m/s: a numerical floor, never reached

    rates = (
        _ramp(pitch_cmd, state.pitch, p.pitch_rate_max, p.delta_t),
        _ramp(torque_cmd, state.torque, p.torque_rate_max, p.delta_t),
    )
    new, _ = integrate_dynamics(
        positions=jnp.stack(
            [
                state.omega,
                state.pitch,
                state.torque,
                state.x_tower,
                state.v_tower,
                state.w_induced,
            ]
        ),
        delta_t=p.delta_t,
        compute_velocity=partial(
            compute_velocity, action=rates, v_wind=v_wind, params=p
        ),
        method=integration_method,
    )
    omega = jnp.maximum(new[0], 1e-3)  # numerical floor: tau_aero = P / omega
    pitch = jnp.clip(new[1], p.pitch_min, p.pitch_max)
    torque = jnp.clip(new[2], 0.0, p.torque_max)
    position = jnp.stack([omega, pitch, torque, new[3], new[4], new[5]])
    accel = compute_velocity(position, (0.0, 0.0), v_wind, p)[0][4]
    over = omega >= p.overspeed_factor * omega_rated(p)
    moment, turn, direction, damage = count_half_cycles(
        state, tower_base_moment(new[3], p), p
    )
    t_derate = (state.time + 1 - state.derate_c0) * p.delta_t
    derated = (
        (state.derate_t0 >= 0.0)
        & (t_derate >= state.derate_t0)
        & (t_derate < state.derate_until)
    )
    rating = jnp.where(derated, state.derate_level, p.rating)
    v_mean_z, v_mean = mean_wind(state, key, p)

    return (
        state.replace(
            omega=omega,
            pitch=pitch,
            torque=torque,
            x_tower=new[3],
            v_tower=new[4],
            w_induced=new[5],
            a_tower=accel,
            v_wind=v_wind,
            v_turb=v_turb,
            v_mean=v_mean,
            v_mean_z=v_mean_z,
            torque_cmd=torque_cmd,
            power=electrical_power(omega, torque, p),
            target_power=rating * p.P_rated,
            feathered=jnp.where(tripped | over, 1.0, 0.0),
            moment=moment,
            moment_turn=turn,
            moment_dir=direction,
            damage=damage,
            pitch_travel=jnp.abs(pitch - state.pitch),
            time=state.time + 1,
        ),
        None,
    )


# Deliberately not jitted: callers that want it fused jit ``step_env``.
def get_obs(state: WindTurbineState, params: WindTurbineParams):
    """``[omega_rpm, pitch_deg, torque_pct, P_MW, rating_P_MW, a_fa, v_nacelle]``.

    The turbine controller's readings: rotor speed, pitch, generator torque and
    power [S1 Sec. 7]; the rating's power, a set point known ahead [S9 Ex. 29];
    the nacelle accelerometer's fore-aft reading, m/s^2 [S3, S5]; the nacelle
    anemometer, m/s [S6].
    """
    p = params
    return jnp.array(
        [
            state.omega * RPM_PER_RAD_S,
            state.pitch,
            100.0 * state.torque / p.torque_max,
            state.power / 1.0e6,
            state.target_power / 1.0e6,
            state.a_tower,
            state.v_wind,
        ]
    )


def check_is_terminal(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    """A trip is the safety system's feather [S4]: no other limit trips."""
    terminated = state.feathered > 0.5
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def wind_power(v_wind, params: WindTurbineParams):
    """Electrical power the wind carries at the table's peak Cp, W."""
    p = params
    return p.eta_gen * 0.5 * p.rho * rotor_area(p) * v_wind**3 * CP_MAX


def available_power(state, params: WindTurbineParams, xp=jnp):
    """What the wind makes available, W: the rating's power, or less where the
    wind at peak Cp falls short of it."""
    return xp.minimum(state.target_power, wind_power(state.v_wind, params))


def lost_production(state, params: WindTurbineParams, power, xp=jnp):
    """MWh short of the available power over one step; over-production is free."""
    short = xp.maximum(available_power(state, params, xp) - power, 0.0)
    return short / 1.0e6 * params.delta_t / 3600.0


def tower_cost(params: WindTurbineParams):
    """The tower's cost, $."""
    p = params
    return p.capex_per_kw * p.P_rated / 1e3 * p.turbine_share * p.tower_share


def pitch_system_cost(params: WindTurbineParams):
    """The pitch system's cost, $."""
    p = params
    return p.capex_per_kw * p.P_rated / 1e3 * p.turbine_share * p.pitch_share


def life_steps(params: WindTurbineParams):
    return params.design_life_years * 365.25 * 24.0 * 3600.0 / params.delta_t


def compute_reward_terms(
    state: WindTurbineState, params: WindTurbineParams, xp=jnp, tripped=None
):
    """The score's terms in dollars, each >= 0: production lost against what
    the wind made available, the tower's fatigue and the pitch system's wear,
    each at its share of the turbine's cost over its design life.

    ``tripped`` marks a measured copy (``base.measured_reward``): its trip comes
    from the true state, and its power from the meter.
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
        power = electrical_power(state.omega, state.torque, p)
    else:
        power = state.power
    life = life_steps(p)
    terms = {
        "lost": lost_production(state, p, power, xp) * p.energy_price,
        "tower": tower_cost(p) * state.damage / (p.tower_damage_design * life),
        "pitch": pitch_system_cost(p)
        * state.pitch_travel
        / (p.pitch_travel_design * life),
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    return R.total(compute_reward_terms(state, params, xp), xp)
