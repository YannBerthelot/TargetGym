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
    CP_TABLE,
    CT_TABLE,
    PITCH_STEP,
    TSR_FIRST,
    TSR_STEP,
)
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

RPM_PER_RAD_S = 60.0 / (2.0 * jnp.pi)

#: The extreme operating gust's duration, s (IEC 61400-1 via [S13]).
EOG_SECONDS = 10.5


@struct.dataclass
class WindTurbineParams(EnvParams):
    # ---- Rotor [S1 Table 1-1] ----
    R: float = 63.0  # m, rotor radius (126 m diameter)
    rho: float = 1.225  # kg/m^3, air density

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
    #: Reference wind of the class, m/s (ours: class I, 50 m/s [S13]); only
    #: the gust's 1.35 (Ve1 - V) branch reads it, which never binds here.
    v_ref: float = 50.0
    #: Start of one extreme operating gust, s; negative for none. A 1-year
    #: event [S13], so none in a normal 10 min episode.
    gust_start: float = -1.0

    # ---- Safety system ----
    #: Overspeed level as a multiple of rated speed: "approximately 20 %
    #: above the rated speed of the generator" [S17]. Past it the pitch
    #: system feathers independently of the controller [S4]: a trip.
    overspeed_factor: float = 1.20

    # ---- Episode ----
    #: Fraction of rated power the turbine is asked for (ROSCO's rating R,
    #: [S9 Ex. 29]); 1 on a normal day [S1 Sec. 7]. Read by the agent.
    rating: float = 1.0
    #: 0.25 s steps (ours: commercial rates not found; 12 per tower period).
    delta_t: float = 0.25
    #: 10 min: one SCADA interval [S7] and one load case [S2 Table 2].
    max_steps_in_episode: int = 2400

    # ---- Score (rule 15): energy off the rating, in MWh ----
    reward_version: int = 3
    #: A trip's restart: ROSCO's start-up from feather, 400 s [S9 Ex. 32], at
    #: 0.25 s steps.
    restart_steps: int = 1600
    #: Production lost per step of a restart, MWh: rated power for one step.
    failure_cost: float = 5.0 * 0.25 / 3600.0
    # Kept for the time-in-band KPI tables (benchmark._OUTPUT_FLOORS) and the
    # oracle's resolution; the score does not read them.
    e_floor: float = 1680.0  # W
    precision_floor: float = 1.7e3  # W, a class 0.1S meter at 5 MW


#: Units of the WindTurbineParams fields, for the generated environment pages.
PARAM_UNITS = {
    "R": "m",
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
    "v_ref": "m/s",
    "gust_start": "s",
    "overspeed_factor": "-",
    "rating": "-",
    "reward_version": "-",
    "restart_steps": "steps",
    "failure_cost": "MWh per step",
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
    #: The turbulence's deviation from the mean (m/s), without the gust.
    v_turb: float = 0.0
    #: 1 once the safety system has fed the blades to feather (a trip).
    feathered: float = 0.0

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
    return _table(CP_TABLE, tip_speed_ratio, pitch_deg)


def thrust_coefficient(tip_speed_ratio, pitch_deg, params: WindTurbineParams):
    """Ct(lambda, beta) from ROSCO's table."""
    return _table(CT_TABLE, tip_speed_ratio, pitch_deg)


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


def gust(t, v_hub, params: WindTurbineParams):
    """The gust's departure from the mean at time ``t`` (s) [S13]."""
    tau = t - params.gust_start
    on = (params.gust_start >= 0.0) & (tau >= 0.0) & (tau <= EOG_SECONDS)
    shape = jnp.sin(3.0 * jnp.pi * tau / EOG_SECONDS) * (
        1.0 - jnp.cos(2.0 * jnp.pi * tau / EOG_SECONDS)
    )
    return jnp.where(on, -0.37 * gust_magnitude(v_hub, params) * shape, 0.0)


def compute_velocity(position, action, v_wind, params: WindTurbineParams):
    """RHS for ``[omega, pitch, torque, x_tower, v_tower]``. ``action`` is the
    step's ``(pitch rate, torque rate)``, constant over the step."""
    p = params
    omega, pitch, torque, x, xdot = position
    v_rel = v_wind - xdot
    tau_aero, _ = aerodynamic_torque(omega, v_rel, pitch, p)
    thrust = rotor_thrust(omega, v_rel, pitch, p)
    domega = (tau_aero - p.N_gear * torque) / drivetrain_inertia(p)
    accel = (
        thrust - tower_damping(p) * xdot - tower_stiffness(p) * x
    ) / p.tower_modal_mass
    return jnp.stack([domega, action[0], action[1], xdot, accel]), None


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
    t_new = (state.time + 1) * p.delta_t
    v_wind = jnp.maximum(
        state.v_mean + v_turb + gust(t_new, state.v_mean, p), 0.1
    )  # 0.1 m/s: a numerical floor, never reached

    rates = (
        _ramp(pitch_cmd, state.pitch, p.pitch_rate_max, p.delta_t),
        _ramp(torque_cmd, state.torque, p.torque_rate_max, p.delta_t),
    )
    new, _ = integrate_dynamics(
        positions=jnp.stack(
            [state.omega, state.pitch, state.torque, state.x_tower, state.v_tower]
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
    position = jnp.stack([omega, pitch, torque, new[3], new[4]])
    accel = compute_velocity(position, (0.0, 0.0), v_wind, p)[0][4]
    over = omega >= p.overspeed_factor * omega_rated(p)

    return (
        state.replace(
            omega=omega,
            pitch=pitch,
            torque=torque,
            x_tower=new[3],
            v_tower=new[4],
            a_tower=accel,
            v_wind=v_wind,
            v_turb=v_turb,
            torque_cmd=torque_cmd,
            power=electrical_power(omega, torque, p),
            target_power=p.rating * p.P_rated,
            feathered=jnp.where(tripped | over, 1.0, 0.0),
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


def energy_off_rating(state, params: WindTurbineParams, power, xp=jnp):
    """MWh delivered off the rating over one step, above or below alike."""
    return xp.abs(state.target_power - power) / 1.0e6 * params.delta_t / 3600.0


def compute_reward_terms(
    state: WindTurbineState, params: WindTurbineParams, xp=jnp, tripped=None
):
    """The score's terms, in MWh, each >= 0.

    ``tripped`` marks a measured copy (``base.measured_reward``): its trip comes
    from the true state, and its power from the meter.
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
        power = electrical_power(state.omega, state.torque, p)
    else:
        power = state.power
    terms = {"off_rating": energy_off_rating(state, p, power, xp)}
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    return R.total(compute_reward_terms(state, params, xp), xp)
