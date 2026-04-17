"""
Glass furnace (float-glass process) — 3-zone lumped thermal model.

All temperatures are expressed in degrees Celsius.  The Stefan-Boltzmann
radiation term internally converts to Kelvin (T_C + 273.15) because T^4
requires an absolute temperature scale.  Every other heat-transfer term
uses a *temperature difference*, which is scale-invariant.

Layout (side/end-port fired):

                 Fuel in (action)
                         |
                         v
    +------------------------------------------------------+
    |            CROWN (combustion space)                  |  <- T_crown  [target]
    +------------------------------------------------------+
    |   MELT ZONE (front)       |   WORKING END (back)     |
    |   T_melt (hidden)         |   T_work (hidden)        |
    +------------------------------------------------------+
        <- glass flows this way at constant pull rate ->

States (ODEs):
    - T_crown: combustion-space gas + refractory lumped temperature [°C]
    - T_melt:  glass melt temperature in melting zone [°C]
    - T_work:  glass temperature in working end [°C]

Only T_crown is observed by the agent. The glass temperatures are hidden —
changing fuel flow affects T_crown in minutes but T_work only over hours
(because heat must advect through the glass at the pull rate).

Task-hardening features
-----------------------
1. **Setpoint schedule**: the target T_crown is piecewise constant with
   ``N_SETPOINTS`` segments of equal duration.  Each segment's setpoint is
   sampled uniformly from ``target_T_crown_range`` at reset.  This rewards
   *anticipation* (MPC can look ahead, PID cannot).
2. **Fuel cost in reward**: a small fuel penalty makes the controller
   prefer efficient operation, not just tight tracking.  Large setpoint
   jumps therefore trade off tracking speed against fuel burn.
3. **Pull-rate disturbance**: the actual pull rate ``m_pull`` drifts around
   its nominal value as a slowly-varying AR(1) process.  This breaks the
   fixed-steady-state assumption that PID tuning relies on.

Physics (per-zone energy balance):
    Q_comb      = m_fuel * LHV
    Q_rad_CX    = eps * sigma * A_X * ((T_crown+273.15)**4 - (T_X+273.15)**4)
    Q_conv_CX   = h * A_X * (T_crown - T_X)
    Q_wall_i    = U * A_wall_i * (T_i - T_ambient)
    Q_fusion    = m_pull * dH_fusion         (batch -> glass, endothermic)
    Q_exhaust   = m_gas * c_p_gas * (T_crown - T_ambient)
    m_gas       = (1 + AFR) * m_fuel         (combustion stoichiometry)

Energy balances:
    m_crown * c_p_gas  * dT_crown/dt = Q_comb - Q_rad_CM - Q_rad_CW
                                     - Q_conv_CM - Q_conv_CW
                                     - Q_wall_crown - Q_exhaust
    m_melt  * c_p_g    * dT_melt/dt  = Q_rad_CM + Q_conv_CM - Q_wall_melt
                                     - Q_fusion + m_pull*c_p_g*(T_batch_in - T_melt)
    m_work  * c_p_g    * dT_work/dt  = Q_rad_CW + Q_conv_CW - Q_wall_work
                                     + m_pull*c_p_g*(T_melt - T_work)
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann (W m^-2 K^-4)
KELVIN_OFFSET = 273.15  # °C -> K conversion

# Number of piecewise-constant setpoints in an episode.
# Transitions are evenly spaced at i * max_steps_in_episode / N_SETPOINTS.
# More setpoints → more transients → bigger MPC-over-PID gap (MPC anticipates
# each step change; PID can only react to it after it happens).
N_SETPOINTS = 5

# AR(1) correlation coefficient for the pull-rate disturbance.
# With dt=30s and rho=0.99, the disturbance has a ~50 min memory — slow
# enough to appear as a drifting operating point rather than white noise.
M_PULL_AR_RHO = 0.99


@struct.dataclass
class GlassFurnaceParams(EnvParams):
    # ---- Combustion ----
    LHV: float = 50.0e6  # natural gas LHV (J/kg)
    AFR: float = 17.0  # stoichiometric air-fuel mass ratio (approx)
    fuel_min: float = 0.3  # kg/s  (~15 MW)
    fuel_max: float = 2.0  # kg/s  (~100 MW)

    # ---- Glass flow (nominal pull rate — disturbed during simulation) ----
    m_pull: float = 5.79  # kg/s (~500 t/day float line)
    T_batch_in: float = 25.0  # °C (incoming batch — cold)
    dH_fusion: float = 0.8e6  # J/kg (latent + endothermic reactions)

    # ---- Pull-rate disturbance (AR(1), stationary std-dev in kg/s) ----
    # Set to 0.0 to disable noise (deterministic physics, e.g. for MPC
    # prediction models).
    m_pull_noise_std: float = 0.4  # kg/s  (~7 % of nominal m_pull)

    # ---- Thermal masses (product of m*c_p, in J/K) ----
    C_crown: float = 2.0e8  # refractory + hot gas lumped
    C_melt: float = 7.8e8  # ~625 t glass at c_p=1250 J/(kg.K)
    C_work: float = 7.8e8

    # ---- Specific heats ----
    c_p_glass: float = 1250.0  # J/(kg.K)
    c_p_gas: float = 1200.0  # J/(kg.K) (hot flue gas)

    # ---- Radiation / convection surface areas (crown -> glass interface) ----
    # A_work << A_melt because the working end is narrowed (throat / breast walls
    # partially shield it from direct flame radiation).
    eps_rad: float = 0.8  # effective emissivity (flame + crown)
    A_melt: float = 200.0  # m^2 (full exposure to flames)
    A_work: float = 30.0  # m^2 (partially shielded by throat/narrowing)
    h_conv: float = 30.0  # W/(m^2.K)

    # ---- Wall losses ----
    U_wall: float = 1.0  # W/(m^2.K) through insulated refractory
    A_wall_crown: float = 200.0  # m^2
    A_wall_melt: float = 200.0  # m^2
    A_wall_work: float = 200.0  # m^2
    T_ambient: float = 25.0  # °C

    # ---- Operating / termination bounds (crown temperature) ----
    T_crown_min: float = 1427.0  # °C  (too cold — incomplete melting)
    T_crown_max: float = 1677.0  # °C  (too hot — refractory damage)

    # Glass temperature safety bounds (terminate if wildly off — protects integrator)
    T_glass_min: float = 927.0  # °C
    T_glass_max: float = 1727.0  # °C

    # ---- Reward shaping ----
    # Small fuel penalty so the controller prefers efficient operation.
    # Weight is relative to the (already normalised) tracking reward ∈ [0, 1].
    fuel_cost_weight: float = 0.1

    # ---- Initial / target ranges ----
    # Wide target range (150 °C span) so that schedule samples create genuinely
    # different operating points and MPC's anticipation advantage becomes
    # visible (wider range → bigger step changes → longer transients).
    target_T_crown_range: Tuple[float, float] = (1500.0, 1650.0)
    initial_T_crown_range: Tuple[float, float] = (1527.0, 1607.0)
    initial_T_melt: float = 1497.0
    initial_T_work: float = 1427.0

    # ---- Time discretization ----
    delta_t: float = 30.0  # s per step
    max_steps_in_episode: int = 5760  # 48 hours @ 30s


@struct.dataclass
class GlassFurnaceState(EnvState):
    T_crown: float
    T_melt: float
    T_work: float

    # Current active target (derived from ``target_schedule`` and ``time``).
    # Kept as an explicit field so existing consumers (obs, rendering, MPC
    # setpoint update) can read it without replaying the schedule.
    target_T_crown: float

    # Piecewise-constant setpoint schedule (shape (N_SETPOINTS,)), sampled
    # at reset.  Segment i (of length max_steps/N_SETPOINTS) uses
    # ``target_schedule[i]`` as the active target.
    target_schedule: jnp.ndarray

    # AR(1) disturbance on the pull rate (kg/s, additive to params.m_pull).
    m_pull_disturbance: float

    # For rendering / observation
    fuel_flow: float


def get_target_from_schedule(
    target_schedule: jnp.ndarray, time: int, params: GlassFurnaceParams
) -> jnp.ndarray:
    """Select the currently-active setpoint from the schedule based on ``time``."""
    slot = jnp.minimum(
        (time * N_SETPOINTS) // params.max_steps_in_episode,
        N_SETPOINTS - 1,
    )
    return target_schedule[slot]


def compute_velocity(position, action, m_pull, params: GlassFurnaceParams):
    """
    Right-hand side of the coupled ODE.

    position = [T_crown, T_melt, T_work]
    action   = fuel flow (kg/s), already in physical units
    m_pull   = effective glass pull rate (kg/s) — noisy during simulation,
               nominal during MPC prediction.
    Returns  = [dT_crown/dt, dT_melt/dt, dT_work/dt], None
    """
    T_crown, T_melt, T_work = position[0], position[1], position[2]
    m_fuel = action

    # Combustion
    Q_comb = m_fuel * params.LHV

    # Exhaust loss (flue gas leaves at T_crown)
    m_gas = (1.0 + params.AFR) * m_fuel
    Q_exhaust = m_gas * params.c_p_gas * (T_crown - params.T_ambient)

    # Radiation crown -> melt / work (Stefan-Boltzmann — requires absolute K)
    T_crown_K = T_crown + KELVIN_OFFSET
    T_melt_K = T_melt + KELVIN_OFFSET
    T_work_K = T_work + KELVIN_OFFSET
    Q_rad_CM = params.eps_rad * SIGMA_SB * params.A_melt * (T_crown_K**4 - T_melt_K**4)
    Q_rad_CW = params.eps_rad * SIGMA_SB * params.A_work * (T_crown_K**4 - T_work_K**4)

    # Convection crown -> melt / work
    Q_conv_CM = params.h_conv * params.A_melt * (T_crown - T_melt)
    Q_conv_CW = params.h_conv * params.A_work * (T_crown - T_work)

    # Wall losses
    Q_wall_crown = params.U_wall * params.A_wall_crown * (T_crown - params.T_ambient)
    Q_wall_melt = params.U_wall * params.A_wall_melt * (T_melt - params.T_ambient)
    Q_wall_work = params.U_wall * params.A_wall_work * (T_work - params.T_ambient)

    # Batch fusion (endothermic, paced to pull rate)
    Q_fusion = m_pull * params.dH_fusion

    # Glass advection between zones at pull rate
    Q_adv_in_melt = m_pull * params.c_p_glass * (params.T_batch_in - T_melt)
    Q_adv_melt_to_work = m_pull * params.c_p_glass * (T_melt - T_work)

    dT_crown = (
        Q_comb - Q_rad_CM - Q_rad_CW - Q_conv_CM - Q_conv_CW - Q_wall_crown - Q_exhaust
    ) / params.C_crown

    dT_melt = (
        Q_rad_CM + Q_conv_CM - Q_wall_melt - Q_fusion + Q_adv_in_melt
    ) / params.C_melt

    dT_work = (Q_rad_CW + Q_conv_CW - Q_wall_work + Q_adv_melt_to_work) / params.C_work

    return jnp.array([dT_crown, dT_melt, dT_work]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    fuel_raw: float,
    state: GlassFurnaceState,
    params: GlassFurnaceParams,
    key: jax.Array,
    integration_method: str = "rk4_1",
):
    """
    fuel_raw : raw action in [-1, 1], mapped to [fuel_min, fuel_max] kg/s.
    key      : PRNG key used for the pull-rate disturbance innovation.
    """
    m_fuel = convert_raw_action_to_range(
        fuel_raw, min_action=params.fuel_min, max_action=params.fuel_max
    )

    # AR(1) pull-rate disturbance (stationary std = params.m_pull_noise_std).
    innovation = (
        jax.random.normal(key)
        * params.m_pull_noise_std
        * jnp.sqrt(1.0 - M_PULL_AR_RHO**2)
    )
    new_disturbance = M_PULL_AR_RHO * state.m_pull_disturbance + innovation
    m_pull_eff = jnp.maximum(params.m_pull + new_disturbance, 0.1)

    _compute_velocity = partial(
        compute_velocity, action=m_fuel, m_pull=m_pull_eff, params=params
    )

    (T_crown, T_melt, T_work), metrics = integrate_dynamics(
        positions=jnp.array([state.T_crown, state.T_melt, state.T_work]),
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )

    new_time = state.time + 1
    new_target = get_target_from_schedule(state.target_schedule, new_time, params)

    return (
        state.replace(
            T_crown=T_crown,
            T_melt=T_melt,
            T_work=T_work,
            target_T_crown=new_target,
            m_pull_disturbance=new_disturbance,
            fuel_flow=m_fuel,
            time=new_time,
        ),
        metrics,
    )


@partial(jax.jit, static_argnames=["params"])
def get_obs(state: GlassFurnaceState, params: GlassFurnaceParams):
    """
    Partially observable: only T_crown is visible (plus fuel_flow and the target).
    Glass temperatures T_melt, T_work are hidden — the agent must infer them
    from T_crown history.  The pull-rate disturbance is also hidden.

    Fuel flow is exposed as a percentage of ``fuel_max`` (0–100) rather than
    kg/s, so the observation is unit-free and stays in a familiar operator
    range regardless of the chosen ``fuel_max``.
    """
    fuel_pct = 100.0 * state.fuel_flow / params.fuel_max
    return jnp.array([state.T_crown, fuel_pct, state.target_T_crown])


def check_is_terminal(state: GlassFurnaceState, params: GlassFurnaceParams, xp=jnp):
    crown_out = xp.logical_or(
        state.T_crown <= params.T_crown_min, state.T_crown >= params.T_crown_max
    )
    glass_out = xp.logical_or(
        xp.logical_or(
            state.T_melt <= params.T_glass_min, state.T_melt >= params.T_glass_max
        ),
        xp.logical_or(
            state.T_work <= params.T_glass_min, state.T_work >= params.T_glass_max
        ),
    )
    terminated = xp.logical_or(crown_out, glass_out)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: GlassFurnaceState, params: GlassFurnaceParams, xp=jnp):
    """
    Reward = squared-normalised tracking term minus a small fuel-cost penalty.

    Tracking : ((max_diff - |target - T_crown|) / max_diff) ** 2 ∈ [0, 1]
    Fuel cost: fuel_cost_weight * (fuel_flow - fuel_min) / (fuel_max - fuel_min)
               ∈ [0, fuel_cost_weight]

    The fuel term encourages the controller to pick the *cheapest* fuel flow
    consistent with tracking — saturating at fuel_max is now penalised.
    """
    max_diff = params.T_crown_max - params.T_crown_min
    tracking = (
        (max_diff - xp.abs(state.target_T_crown - state.T_crown)) / max_diff
    ) ** 2
    fuel_span = params.fuel_max - params.fuel_min
    fuel_norm = (state.fuel_flow - params.fuel_min) / fuel_span
    fuel_penalty = params.fuel_cost_weight * fuel_norm
    return tracking - fuel_penalty
