"""
Wind turbine — NREL 5 MW reference turbine, collective-pitch power regulation.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
Single-mass drivetrain driven by rotor aerodynamics, with pitch and generator
torque as the two manipulated variables::

    J dw/dt = tau_aero - N tau_gen
    tau_aero = 0.5 rho A v^3 Cp(lambda, beta) / w        lambda = w R / v
    P_elec   = eta N tau_gen w

``J`` is the drivetrain inertia referred to the rotor, ``N`` the gearbox ratio.
Pitch and generator torque follow their demands through first-order actuators;
pitch is additionally rate-limited, which is the binding constraint during
gusts.

Task
----
Track a **power setpoint** in above-rated wind -- the curtailed-operation
problem a modern turbine actually faces when the grid dispatches it below its
available power. The turbine must hold that setpoint while turbulence moves
the available power around it, without overspeeding the rotor.

Why it is a hard target MDP
---------------------------
* **The rotor-effective wind is hidden.** A nacelle anemometer sits in the
  rotor's own wake and is famously unreliable, so the controller has to infer
  what the wind is doing from rotor speed and power. That is a real estimation
  problem, not an artificial restriction.
* **Two inputs, coupled through one inertia.** Pitch spills aerodynamic power;
  generator torque converts it. Both change rotor speed, and the useful
  combination depends on where the turbine sits on the Cp surface.
* **Pitch is rate-limited.** During a gust the controller cannot pitch as fast
  as the wind changes, so it must anticipate rather than react.
* **Overspeed is irrecoverable.** Losing rotor-speed regulation trips the
  turbine, and the reward cannot buy that back.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

RPM_PER_RAD_S = 60.0 / (2.0 * jnp.pi)

# Ornstein-Uhlenbeck turbulence: mean-reversion rate (1/s). 1/theta ~ 20 s, a
# gust timescale rather than white noise.
WIND_OU_THETA = 0.05


@struct.dataclass
class WindTurbineParams(EnvParams):
    # ---- Rotor ----
    R: float = 63.0  # m, rotor radius (126 m diameter)
    rho: float = 1.225  # kg/m^3, air density

    # ---- Drivetrain ----
    J_rotor: float = 38_759_228.0  # kg m^2
    J_gen: float = 534.116  # kg m^2, generator side
    N_gear: float = 97.0  # gearbox ratio
    eta_gen: float = 0.944  # generator efficiency

    # ---- Ratings ----
    P_rated: float = 5.0e6  # W
    v_rated: float = 11.4  # m/s
    omega_rated_rpm: float = 12.1
    v_cut_out: float = 25.0  # m/s

    # ---- Cp surface (analytic fit; see PHYSICS.md) ----
    cp_c1: float = 0.5176
    cp_c2: float = 116.0
    cp_c3: float = 0.4
    cp_c4: float = 5.0
    cp_c5: float = 21.0
    cp_c6: float = 0.0068

    # ---- Actuators ----
    pitch_min: float = 0.0  # deg
    pitch_max: float = 40.0  # deg
    pitch_rate_max: float = 8.0  # deg/s -- the binding constraint in gusts
    pitch_tau: float = 0.2  # s, actuator lag
    torque_max: float = 47_400.0  # N m, generator side (~110 % of rated)
    torque_tau: float = 0.1  # s, power-electronics lag

    # ---- Wind ----
    # Comfortably above the 11.4 m/s rated wind: with 1.2 m/s turbulence a
    # mean of 12 dips below rated regularly, where the power setpoint is
    # physically unachievable and the rotor stalls no matter what the
    # controller does. 13.5 keeps the task well-posed.
    v_mean_range: Tuple[float, float] = (13.5, 20.0)
    turbulence_std: float = 1.2  # m/s, OU stationary std
    v_min: float = 4.0
    v_max: float = 28.0

    # ---- Operating / termination bounds ----
    overspeed_factor: float = 1.25  # trip above this multiple of rated speed
    underspeed_factor: float = 0.40

    # ---- Reward shaping ----
    power_band: float = 0.5e6  # W, error at which tracking reward reaches 0
    pitch_activity_weight: float = 0.02  # fatigue proxy

    # ---- Targets ----
    target_power_range: Tuple[float, float] = (3.5e6, 5.0e6)

    # ---- Time discretization ----
    # Rotor time constant J w^2 / P ~ 14 s, so dt = 0.25 s gives ~56 steps per
    # time constant. 1200 steps = 5 min.
    delta_t: float = 0.25
    max_steps_in_episode: int = 1200


@struct.dataclass
class WindTurbineState(EnvState):
    omega: float  # rotor speed (rad/s)
    pitch: float  # blade pitch (deg), actuator output
    torque: float  # generator torque (N m), actuator output

    v_wind: float  # rotor-effective wind (m/s) -- HIDDEN
    v_mean: float  # episode mean wind (m/s) -- HIDDEN

    pitch_cmd: float
    torque_cmd: float
    target_power: float


def rotor_area(params: WindTurbineParams):
    return jnp.pi * params.R**2


def omega_rated(params: WindTurbineParams):
    return params.omega_rated_rpm * 2.0 * jnp.pi / 60.0


def power_coefficient(tip_speed_ratio, pitch_deg, params: WindTurbineParams):
    """Cp(lambda, beta) from the standard analytic fit.

    Reproduces Cp_max = 0.480 at TSR 8.1, against the reference turbine's
    0.482 at 7.55 -- the peak value matches closely, its location is ~7 %
    high. Clipped at zero because the fit goes negative outside its valid
    region, which is not physical for a turbine extracting power.
    """
    p = params
    lam = jnp.maximum(tip_speed_ratio, 1e-3)
    inv_lam_i = 1.0 / (lam + 0.08 * pitch_deg) - 0.035 / (pitch_deg**3 + 1.0)
    lam_i = 1.0 / jnp.where(jnp.abs(inv_lam_i) < 1e-6, 1e-6, inv_lam_i)
    cp = (
        p.cp_c1
        * (p.cp_c2 / lam_i - p.cp_c3 * pitch_deg - p.cp_c4)
        * jnp.exp(-p.cp_c5 / lam_i)
        + p.cp_c6 * lam
    )
    return jnp.clip(cp, 0.0, 0.593)  # Betz limit as a hard ceiling


def aerodynamic_torque(omega, v_wind, pitch_deg, params: WindTurbineParams):
    """Rotor torque from the momentum balance."""
    w = jnp.maximum(omega, 1e-3)
    lam = w * params.R / jnp.maximum(v_wind, 1e-3)
    cp = power_coefficient(lam, pitch_deg, params)
    power = 0.5 * params.rho * rotor_area(params) * v_wind**3 * cp
    return power / w, power


def electrical_power(omega, torque, params: WindTurbineParams):
    """Generator electrical output; torque is generator-side."""
    return params.eta_gen * params.N_gear * torque * omega


def drivetrain_inertia(params: WindTurbineParams):
    """Inertia referred to the rotor shaft."""
    return params.J_rotor + params.N_gear**2 * params.J_gen


def rated_generator_torque(params: WindTurbineParams):
    """Generator torque that delivers rated power at rated speed."""
    return params.P_rated / (params.eta_gen * params.N_gear * omega_rated(params))


def compute_velocity(position, action, v_wind, params: WindTurbineParams):
    """RHS for ``[omega, pitch, torque]``. ``action`` is ``(pitch_cmd, torque_cmd)``."""
    p = params
    omega, pitch, torque = position[0], position[1], position[2]
    pitch_cmd, torque_cmd = action

    tau_aero, _ = aerodynamic_torque(omega, v_wind, pitch, p)
    domega = (tau_aero - p.N_gear * torque) / drivetrain_inertia(p)

    # Pitch actuator: first-order toward the demand, then rate-limited. The
    # rate limit is the binding constraint during a gust.
    dpitch = jnp.clip(
        (pitch_cmd - pitch) / p.pitch_tau, -p.pitch_rate_max, p.pitch_rate_max
    )
    dtorque = (torque_cmd - torque) / p.torque_tau
    return jnp.array([domega, dpitch, dtorque]), None


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

    # OU turbulence about the episode mean. Drawn from a key folded with
    # ``state.time`` so a caller passing a constant key -- which every rollout
    # helper here does -- still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.turbulence_std * jnp.sqrt(2.0 * WIND_OU_THETA * p.delta_t)
    v_wind = jnp.clip(
        state.v_wind
        + WIND_OU_THETA * (state.v_mean - state.v_wind) * p.delta_t
        + sigma * noise,
        p.v_min,
        p.v_max,
    )

    _compute_velocity = partial(
        compute_velocity, action=(pitch_cmd, torque_cmd), v_wind=v_wind, params=p
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.omega, state.pitch, state.torque]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    omega = jnp.maximum(new_positions[0], 1e-3)
    pitch = jnp.clip(new_positions[1], p.pitch_min, p.pitch_max)
    torque = jnp.clip(new_positions[2], 0.0, p.torque_max)

    return (
        state.replace(
            omega=omega,
            pitch=pitch,
            torque=torque,
            v_wind=v_wind,
            pitch_cmd=pitch_cmd,
            torque_cmd=torque_cmd,
            time=state.time + 1,
        ),
        None,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: WindTurbineState, params: WindTurbineParams):
    """``[omega_rpm, pitch_deg, torque_pct, P_MW, target_P_MW]``.

    A turbine measures rotor speed, pitch, generator torque and electrical
    power. It does *not* reliably measure the rotor-effective wind -- a
    nacelle anemometer sits in the rotor's own wake -- so ``v_wind`` is hidden
    and must be inferred from how the rotor is behaving.
    """
    p = params
    return jnp.array(
        [
            state.omega * 60.0 / (2.0 * jnp.pi),
            state.pitch,
            100.0 * state.torque / p.torque_max,
            electrical_power(state.omega, state.torque, p) / 1.0e6,
            state.target_power / 1.0e6,
        ]
    )


def check_is_terminal(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    w_rated = omega_rated(params)
    terminated = xp.logical_or(
        state.omega >= params.overspeed_factor * w_rated,
        state.omega <= params.underspeed_factor * w_rated,
    )
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: WindTurbineState, params: WindTurbineParams, xp=jnp):
    """Power tracking minus a pitch-activity penalty (a fatigue proxy)."""
    power = electrical_power(state.omega, state.torque, params)
    err = xp.abs(state.target_power - power)
    tracking = xp.clip(1.0 - err / params.power_band, 0.0, 1.0) ** 2
    activity = xp.abs(state.pitch_cmd - state.pitch) / params.pitch_max
    return tracking - params.pitch_activity_weight * activity


def available_power(v_wind, omega, params: WindTurbineParams):
    """Best electrical power extractable at this wind and rotor speed."""
    lam = omega * params.R / jnp.maximum(v_wind, 1e-3)
    cp = power_coefficient(lam, 0.0, params)
    return params.eta_gen * 0.5 * params.rho * rotor_area(params) * v_wind**3 * cp
