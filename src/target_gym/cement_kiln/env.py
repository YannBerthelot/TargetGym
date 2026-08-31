"""
Cement rotary kiln — 1-D axial model with counter-current gas.

See ``PHYSICS.md`` in this directory for provenance, the parameter table and
the validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Layout (a 3000 t/day dry-process kiln with preheater and precalciner)::

    hot meal (800 C, 92 % calcined)
        |
        v
    +===========================================================+
    |  solid  ---------------------------------------------->   |   clinker
    |  gas    <----------------------------------------------   | <-- burner
    +===========================================================+
        |                                                       ^
        v exhaust to the precalciner            secondary air from the cooler

The kiln is discretised into ``n_zones`` axial slices. Each carries four
dynamic states -- solid temperature, refractory temperature, calcination extent
and free lime -- and the gas is solved **quasi-steady** by a single sweep from
the burner end, which is justified because gas crosses the kiln in about 8 s
against 25 minutes for the solid.

Why this environment exists
---------------------------
It is the suite's **transport-delay** problem, and the delay is emergent rather
than a delay block:

* **Half an hour of dead time.** Solid states are advected down the kiln, so a
  fuel change reaches the discharge only after the material does. Nothing in
  the model contains a delay parameter -- it falls out of the advection.
* **Kiln speed moves the delay itself.** Speed sets residence time and holdup
  together, so the second input does not just add heat somewhere else: it
  changes the plant's dynamics. That makes it genuinely different from fuel,
  not a second knob on the same response.
* **Free lime is ferociously temperature-sensitive.** The clinkerisation rate
  has an activation energy of 280 kJ/mol, so the reaction time constant runs
  from 20 minutes at 1230 C to under a minute at 1480 C. Small burning-zone
  errors become large quality errors.
* **The kiln is operated nearly blind.** Four states per zone are dynamic and
  the observation exposes a handful of measurements: two pyrometers, a gas
  thermocouple and the discharge assay. The axial profile -- which is what
  actually determines the product -- is hidden.
* **Both failure modes are irrecoverable.** Overheat and the charge sinters
  into rings that block the kiln; go cold and the reaction stops, and a cold
  kiln takes hours to bring back.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

R_GAS = 8.314
SIGMA = 5.67e-8
N_ZONES = 16

# Ornstein-Uhlenbeck raw-meal feed disturbance. 1/theta is about 20 min, so a
# feed swing persists comparably to the kiln's own transport delay.
FEED_OU_THETA = 8.3e-4


@struct.dataclass
class CementKilnParams(EnvParams):
    delta_t: float = 30.0
    max_steps_in_episode: int = 480  # 4 hours

    # -- geometry -------------------------------------------------------------
    diameter: float = 4.0  # m
    length: float = 60.0  # m
    slope: float = 0.035  # -
    # Static: it sizes the state arrays and indexes slices, so it must not be
    # traced. A different value simply retraces.
    n_zones: int = struct.field(pytree_node=False, default=N_ZONES)

    # Cross-section at ~10 % fill. The bed covers 3.25 m of the 12.57 m
    # circumference, and that covered arc is the regenerative wall-to-bed
    # contact path -- a major heat route, not a correction.
    w_bed_gas: float = 2.906  # m  exposed bed chord (gas <-> solid)
    w_wall_gas: float = 9.313  # m  exposed wall arc (gas <-> wall)
    w_wall_bed: float = 3.253  # m  covered arc      (wall <-> solid)

    refractory_thickness: float = 0.20  # m
    rho_refractory: float = 2200.0  # kg/m3
    cp_refractory: float = 900.0  # J/(kg K)
    U_shell: float = 4.0  # W/(m2 K) through shell to ambient

    # -- material -------------------------------------------------------------
    raw_meal_nominal: float = 53.82  # kg/s raw meal to the preheater
    caco3_fraction: float = 0.79  # -    carbonate in raw meal
    calcination_upstream: float = 0.92  # -    done in the precalciner
    cp_solid: float = 1000.0  # J/(kg K)
    cp_gas: float = 1150.0  # J/(kg K)
    bulk_density: float = 1400.0  # kg/m3
    angle_of_repose: float = 35.0  # deg

    # -- thermochemistry ------------------------------------------------------
    h_calcination: float = 1780e3  # J/kg CaCO3
    h_clinkerisation: float = 420e3  # J/kg free lime converted
    A_calcination: float = 1.5199e5  # 1/s
    E_calcination: float = 170e3  # J/mol
    A_clinker: float = 4.7308e6  # 1/s
    E_clinker: float = 280e3  # J/mol

    # -- combustion -----------------------------------------------------------
    fuel_lhv: float = 26e6  # J/kg  coal
    air_fuel_ratio: float = 10.5  # kg gas per kg fuel
    excess_air: float = 1.15  # -
    flame_length: float = 15.0  # m
    emissivity_gas: float = 0.25  # -    CO2/H2O bands
    h_convective: float = 30.0  # W/(m2 K)
    h_wall_bed: float = 500.0  # W/(m2 K) covered-wall contact

    # -- boundary conditions --------------------------------------------------
    T_feed: float = 1073.0  # K  hot meal from the preheater (800 C)
    T_secondary_air: float = 1373.0  # K  from the clinker cooler (1100 C)
    T_ambient: float = 300.0  # K

    # -- actuators ------------------------------------------------------------
    fuel_min: float = 1.20  # kg/s
    fuel_max: float = 2.40  # kg/s
    rpm_min: float = 2.0
    rpm_max: float = 4.5
    fuel_nominal: float = 1.78  # kg/s
    rpm_nominal: float = 3.0

    # -- disturbance ----------------------------------------------------------
    feed_noise_std: float = 2.0  # kg/s stationary std of the feed swing

    # -- task -----------------------------------------------------------------
    target_lime_range: Tuple[float, float] = (0.008, 0.018)
    lime_band: float = 0.006  # tracking band (fractional free lime)
    T_bz_max: float = 1900.0  # K  ring formation / melting
    T_bz_min: float = 1620.0  # K  the kiln has gone cold: the charge stops
    #                                 clinkering and recovery takes hours,
    #                                 longer than an episode
    T_wall_max: float = 2200.0  # K  refractory failure
    fuel_weight: float = 0.05


@struct.dataclass
class CementKilnState(EnvState):
    T_solid: jnp.ndarray  # (n_zones,) K
    T_wall: jnp.ndarray  # (n_zones,) K
    alpha: jnp.ndarray  # (n_zones,) residual calcination extent
    lime: jnp.ndarray  # (n_zones,) free lime fraction
    T_gas: jnp.ndarray  # (n_zones,) K, quasi-steady
    T_exhaust: jnp.ndarray  # K at the kiln inlet (to the precalciner)
    fuel: jnp.ndarray  # kg/s
    rpm: jnp.ndarray
    raw_meal: jnp.ndarray  # kg/s (the disturbance)
    target_lime: jnp.ndarray


# ---------------------------------------------------------------------------
# Geometry and material flow
# ---------------------------------------------------------------------------


def zone_length(params: CementKilnParams) -> float:
    return params.length / params.n_zones


def zone_centres(params: CementKilnParams) -> jnp.ndarray:
    return (jnp.arange(params.n_zones) + 0.5) * zone_length(params)


def kiln_feed_rate(raw_meal, params: CementKilnParams):
    """Hot meal actually entering the kiln.

    Calcination sheds CO2: CaCO3 -> CaO + CO2 loses 44 % of the carbonate mass,
    and 92 % of that has already happened in the precalciner. Feeding the kiln
    the *raw* meal rate overstates its thermal load by nearly 50 %.
    """
    co2 = params.caco3_fraction * 0.44
    return raw_meal * (1.0 - params.calcination_upstream * co2)


def residence_time(rpm, params: CementKilnParams):
    """Sullivan's correlation, the standard for rotary kilns (seconds)."""
    return (
        1.77
        * params.length
        * jnp.sqrt(params.angle_of_repose)
        / (params.slope * params.diameter * jnp.maximum(rpm, 0.1))
    )


def flame_profile(params: CementKilnParams) -> jnp.ndarray:
    """Heat release decaying away from the burner, normalised to one.

    A point source at the burner end dumps the entire firing rate into one
    slice, exhausts the gas immediately and leaves the rest of the kiln cold.
    Real flames extend 15-25 m.
    """
    distance = params.length - zone_centres(params)
    w = jnp.exp(-distance / params.flame_length)
    return w / jnp.sum(w)


# ---------------------------------------------------------------------------
# Heat transfer
# ---------------------------------------------------------------------------


def h_radiative(T_hot, T_cold, params: CementKilnParams):
    """Linearised gray-gas radiation plus a convective floor.

    Radiation dominates everywhere above about 1000 C, which is most of the
    kiln.
    """
    return (
        params.emissivity_gas * SIGMA * (T_hot**2 + T_cold**2) * (T_hot + T_cold)
        + params.h_convective
    )


def gas_sweep(T_solid, T_wall, fuel, params: CementKilnParams):
    """Solve the counter-current gas profile quasi-steadily.

    Marches from the burner end toward the feed end, releasing combustion heat
    along the flame and giving it up to the bed and the wall. Quasi-steady is
    justified by the timescales: gas crosses the kiln in about 8 s against
    25 minutes for the solid, a factor of ~180.
    """
    dz = zone_length(params)
    A_gs = params.w_bed_gas * dz
    A_gw = params.w_wall_gas * dz
    gas_flow = fuel * params.air_fuel_ratio * params.excess_air
    capacity = gas_flow * params.cp_gas
    release = flame_profile(params) * fuel * params.fuel_lhv

    def march(T_g, zone):
        T_s, T_w, q_release = zone
        T_g = jnp.clip(T_g + q_release / capacity, 300.0, 2900.0)
        q_s = h_radiative(T_g, T_s, params) * A_gs * (T_g - T_s)
        q_w = h_radiative(T_g, T_w, params) * A_gw * (T_g - T_w)
        T_next = jnp.clip(T_g - (q_s + q_w) / capacity, 400.0, 2900.0)
        return T_next, (T_g, q_s, q_w)

    T_exhaust, (T_gas, Q_gs, Q_gw) = jax.lax.scan(
        march,
        params.T_secondary_air,
        (T_solid, T_wall, release),
        reverse=True,
    )
    return T_gas, Q_gs, Q_gw, T_exhaust


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def _upwind(x, inlet):
    """Advective increment for material carried from the feed end."""
    return jnp.concatenate([jnp.array([inlet]), x[:-1]]) - x


def compute_velocity(position, action, raw_meal, params: CementKilnParams):
    """Time derivatives of the stacked ``[T_solid, T_wall, alpha, lime]``.

    ``action`` is ``(fuel [kg/s], rpm)``.
    """
    n = params.n_zones
    T_solid = position[0:n]
    T_wall = position[n : 2 * n]
    alpha = jnp.clip(position[2 * n : 3 * n], 0.0, 1.0)
    lime = jnp.clip(position[3 * n : 4 * n], 0.0, 1.0)
    fuel, rpm = action
    p = params

    dz = zone_length(p)
    A_ws = p.w_wall_bed * dz
    A_shell = jnp.pi * p.diameter * dz
    C_wall = A_shell * p.refractory_thickness * p.rho_refractory * p.cp_refractory

    feed = kiln_feed_rate(raw_meal, p)
    tau = residence_time(rpm, p)
    m_zone = feed * tau / p.n_zones
    v_adv = feed / jnp.maximum(m_zone, 1e-3)

    T_solid = jnp.clip(T_solid, 300.0, 2400.0)
    T_wall = jnp.clip(T_wall, 300.0, 2400.0)
    T_gas, Q_gs, Q_gw, _ = gas_sweep(T_solid, T_wall, fuel, p)
    Q_ws = p.h_wall_bed * A_ws * (T_wall - T_solid)

    # -- reactions ------------------------------------------------------------
    k_calc = p.A_calcination * jnp.exp(-p.E_calcination / (R_GAS * T_solid))
    k_sint = p.A_clinker * jnp.exp(-p.E_clinker / (R_GAS * T_solid))
    d_alpha = k_calc * (1.0 - alpha)
    d_lime = -k_sint * lime

    caco3_left = (1.0 - p.calcination_upstream) * p.caco3_fraction
    Q_reaction = m_zone * (
        caco3_left * p.h_calcination * d_alpha - p.h_clinkerisation * d_lime
    )

    dT_solid = (Q_gs + Q_ws - Q_reaction) / (m_zone * p.cp_solid) + v_adv * _upwind(
        T_solid, p.T_feed
    )
    dT_wall = (Q_gw - Q_ws - p.U_shell * A_shell * (T_wall - p.T_ambient)) / C_wall
    dalpha = d_alpha + v_adv * _upwind(alpha, 0.0)
    dlime = d_lime + v_adv * _upwind(lime, 1.0)

    return jnp.concatenate([dT_solid, dT_wall, dalpha, dlime]), None


def compute_next_state(
    action_raw,
    state: CementKilnState,
    params: CementKilnParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` is ``[fuel, kiln_speed]``, each in [-1, 1]."""
    p = params
    action_raw = jnp.atleast_1d(action_raw)
    fuel = convert_raw_action_to_range(
        action_raw[0], min_action=p.fuel_min, max_action=p.fuel_max
    )
    rpm = convert_raw_action_to_range(
        action_raw[1], min_action=p.rpm_min, max_action=p.rpm_max
    )

    # OU raw-meal feed. The innovation is drawn from a key folded with
    # ``state.time`` so a caller passing a constant key -- which every rollout
    # helper in this repo does -- still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.feed_noise_std * jnp.sqrt(2.0 * FEED_OU_THETA * p.delta_t)
    raw_meal = (
        state.raw_meal
        - FEED_OU_THETA * (state.raw_meal - p.raw_meal_nominal) * p.delta_t
        + sigma * noise
    )
    raw_meal = jnp.clip(raw_meal, 0.5 * p.raw_meal_nominal, 1.5 * p.raw_meal_nominal)

    _compute_velocity = partial(
        compute_velocity, action=(fuel, rpm), raw_meal=raw_meal, params=p
    )
    packed = jnp.concatenate([state.T_solid, state.T_wall, state.alpha, state.lime])
    new_packed, _ = integrate_dynamics(
        positions=packed,
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    n = p.n_zones
    T_solid = jnp.clip(new_packed[0:n], 300.0, 2400.0)
    T_wall = jnp.clip(new_packed[n : 2 * n], 300.0, 2400.0)
    alpha = jnp.clip(new_packed[2 * n : 3 * n], 0.0, 1.0)
    lime = jnp.clip(new_packed[3 * n : 4 * n], 0.0, 1.0)
    T_gas, _, _, T_exhaust = gas_sweep(T_solid, T_wall, fuel, p)

    return (
        state.replace(
            T_solid=T_solid,
            T_wall=T_wall,
            alpha=alpha,
            lime=lime,
            T_gas=T_gas,
            T_exhaust=T_exhaust,
            fuel=fuel,
            rpm=rpm,
            raw_meal=raw_meal,
            time=state.time + 1,
        ),
        None,
    )


# ---------------------------------------------------------------------------
# Observation, reward, termination
# ---------------------------------------------------------------------------


def burning_zone_temperature(state: CementKilnState):
    """What the burning-zone pyrometer sees: the hottest part of the charge."""
    return jnp.max(state.T_solid)


def discharge_lime(state: CementKilnState):
    return state.lime[-1]


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: CementKilnState, params: CementKilnParams):
    """Plant instrumentation only.

    ``[lime_pct, T_burning_zone, T_exhaust, T_back_end, feed_rate, fuel_pct,
    speed_pct, target_lime_pct]``

    A kiln has a burning-zone pyrometer, a back-end gas thermocouple, a
    back-end material pyrometer, a weighfeeder on the raw meal and a discharge
    free-lime measurement, and it knows its own fuel and speed demands. It does
    **not** see the axial profile: ``n_zones x 4`` states are dynamic and a
    handful of numbers come out. This is not a modelling convenience -- kiln
    operators really do run the process on a few readings, which is a large
    part of why the job is famously hard.
    """
    return jnp.array(
        [
            100.0 * discharge_lime(state),
            burning_zone_temperature(state),
            state.T_exhaust,
            state.T_solid[0],
            state.raw_meal,
            100.0
            * (state.fuel - params.fuel_min)
            / (params.fuel_max - params.fuel_min),
            100.0 * (state.rpm - params.rpm_min) / (params.rpm_max - params.rpm_min),
            100.0 * state.target_lime,
        ]
    )


def check_is_terminal(state: CementKilnState, params: CementKilnParams, xp=jnp):
    """Both failure modes are irrecoverable in practice.

    Overheating sinters the charge into rings that block the kiln; a kiln that
    goes cold stops reacting and takes hours to bring back.
    """
    T_bz = xp.max(state.T_solid)
    overheat = xp.logical_or(
        T_bz >= params.T_bz_max, xp.max(state.T_wall) >= params.T_wall_max
    )
    gone_cold = T_bz <= params.T_bz_min
    terminated = xp.logical_or(overheat, gone_cold)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: CementKilnState, params: CementKilnParams, xp=jnp):
    """Free-lime tracking minus fuel.

    A squared-normalised band rather than a Gaussian: it stays informative well
    outside the band, so a controller that is far off still sees a gradient
    back toward the setpoint.
    """
    err = xp.abs(discharge_lime(state) - state.target_lime)
    quality = xp.clip(1.0 - err / params.lime_band, 0.0, 1.0) ** 2
    fuel = (state.fuel - params.fuel_min) / (params.fuel_max - params.fuel_min)
    return quality - params.fuel_weight * fuel


def specific_heat_consumption(state: CementKilnState, params: CementKilnParams):
    """Plant heat use in MJ per kg clinker -- for reporting and validation.

    The kiln burner is about 40 % of plant firing; the precalciner takes the
    rest.
    """
    co2 = params.caco3_fraction * 0.44
    clinker = state.raw_meal * (1.0 - co2)
    return state.fuel * params.fuel_lhv / 0.40 / clinker / 1e6


def steady_profile(params: CementKilnParams, fuel=None, rpm=None, n_iter=4000, dt=10.0):
    """March the model to its steady axial profile.

    Used by ``reset_env`` and by the tests. A kiln started from an arbitrary
    profile would spend hours of episode time relaxing, which is an artefact
    rather than a control problem.
    """
    p = params
    fuel = p.fuel_nominal if fuel is None else fuel
    rpm = p.rpm_nominal if rpm is None else rpm
    n = p.n_zones
    packed = jnp.concatenate(
        [
            jnp.linspace(p.T_feed, 1750.0, n),
            jnp.linspace(p.T_feed, 1850.0, n),
            jnp.linspace(0.1, 1.0, n),
            jnp.linspace(1.0, 0.02, n),
        ]
    )

    def body(_, y):
        d, _ = compute_velocity(y, (fuel, rpm), p.raw_meal_nominal, p)
        y = y + dt * d
        return jnp.concatenate(
            [
                jnp.clip(y[0 : 2 * n], 300.0, 2400.0),
                jnp.clip(y[2 * n :], 0.0, 1.0),
            ]
        )

    packed = jax.lax.fori_loop(0, n_iter, body, packed)
    return (
        packed[0:n],
        packed[n : 2 * n],
        packed[2 * n : 3 * n],
        packed[3 * n : 4 * n],
    )
