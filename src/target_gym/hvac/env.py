"""
Building HVAC — single thermal zone, ISO 13790 5R1C reduced-order model.

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and the validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Topology (EN ISO 13790 "simple hourly method"), three nodes and five
conductances::

              Phi_ia + Phi_HC          Phi_st              Phi_m
                    |                    |                   |
     T_sup --H_ve-- T_air --H_tr_is-- T_surface --H_tr_ms-- T_mass
                                         |                   |
                                      H_tr_w              H_tr_em
                                         |                   |
                                       T_out               T_out

Only ``T_mass`` carries capacitance (``C_m``); ``T_air`` and ``T_surface``
are solved algebraically from it each step, which is what makes the model both
faithful and cheap. The building's entire thermal memory therefore lives in a
single **hidden** state -- the agent measures air temperature, never the mass
it is really fighting.

Control problem
---------------
Track a *scheduled* comfort setpoint (occupied 21 C, night setback 17 C)
against weather and occupancy, trading comfort against energy. The structure
that makes it interesting:

* **The thermal mass is hidden and slow** (tau ~43 h). Air temperature responds
  in minutes but is anchored by a mass the controller cannot see.
* **Setbacks reward anticipation.** Recovering from a night setback takes
  hours, so a controller that waits for the setpoint step is already late --
  the gap MPC should exploit over PID.
* **Solar and occupancy gains are free heat** arriving on a schedule, and
  over-heating is penalised, so the controller must anticipate them too.
* **Weather is a genuine disturbance**: a daily temperature cycle plus a
  mean-reverting stochastic deviation.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0

# Ornstein-Uhlenbeck weather deviation: mean-reversion rate (1/s). 1/theta is
# about 12 h, so a warm or cold spell persists across a day rather than
# flickering step to step.
WEATHER_OU_THETA = 2.3e-5


@struct.dataclass
class HVACParams(EnvParams):
    # ---- Zone geometry (ISO 13790 simple hourly method) ----
    A_floor: float = 150.0  # m^2 conditioned floor area
    lambda_at: float = 4.5  # A_tot / A_floor (standard value)
    f_class: float = 2.5  # A_m / A_floor, "medium" construction class
    cm_per_area: float = 165_000.0  # J/(K.m^2), "medium" class
    h_is: float = 3.45  # W/(m^2.K) air <-> surface film
    h_ms: float = 9.1  # W/(m^2.K) surface <-> mass

    # ---- Envelope (U-values x areas) ----
    A_wall: float = 120.0
    U_wall: float = 0.28  # W/(m^2.K), modern insulated wall
    A_roof: float = 150.0
    U_roof: float = 0.18
    A_window: float = 25.0
    U_window: float = 1.30  # double glazing
    g_window: float = 0.55  # solar transmittance

    # ---- Ventilation ----
    air_changes_per_hour: float = 0.5
    room_height: float = 2.6  # m

    # ---- Heating system ----
    Q_heat_max: float = 7200.0  # W (~1.5x the design load)
    emitter_tau: float = 900.0  # s, radiator thermal lag (15 min)

    # ---- Weather ----
    T_out_mean: float = 5.0  # C, seasonal mean (heating season)
    T_out_amplitude: float = 5.0  # C, half daily swing
    T_out_noise_std: float = 3.0  # C, OU deviation std
    solar_peak: float = 350.0  # W/m^2 on the glazing at solar noon

    # ---- Internal gains ----
    gain_occupied: float = 8.0  # W/m^2 (people, lighting, equipment)
    gain_unoccupied: float = 2.0  # W/m^2
    occupied_start_h: float = 7.0
    occupied_end_h: float = 22.0

    # ---- Setpoint schedule ----
    setpoint_occupied: float = 21.0  # C
    setpoint_setback: float = 17.0  # C
    # Sampled per episode so the task is not a single fixed schedule.
    setpoint_occupied_range: Tuple[float, float] = (20.0, 22.5)

    # ---- Comfort / termination bounds ----
    T_air_min: float = 5.0  # C, building left to freeze
    T_air_max: float = 35.0  # C, grossly overheated

    # ---- Reward shaping ----
    comfort_band: float = 1.0  # C, error at which tracking reward halves
    energy_weight: float = 0.15  # relative to the [0,1] comfort term

    # ---- Initial conditions ----
    initial_T_range: Tuple[float, float] = (18.0, 22.0)

    # ---- Time discretization ----
    delta_t: float = 900.0  # s (15 min) -- standard building-simulation step
    max_steps_in_episode: int = 672  # 7 days


@struct.dataclass
class HVACState(EnvState):
    T_mass: float  # thermal mass node -- HIDDEN, holds the building's memory
    Q_emitter: float  # delivered heating power (W), lags the command

    T_air: float  # algebraic, measured
    T_surface: float  # algebraic, hidden
    T_out: float  # outdoor air temperature (C), measured
    weather_dev: float  # OU deviation from the daily cycle (C), hidden

    target_T: float  # active comfort setpoint
    setpoint_occupied: float  # this episode's occupied setpoint
    Q_command: float  # commanded heating power (W)


# ---------------------------------------------------------------------------
# Derived conductances (ISO 13790 §7). Pure functions of params.
# ---------------------------------------------------------------------------


def zone_conductances(params: HVACParams):
    """Return the five 5R1C conductances plus ``A_m`` and ``C_m``.

    ``H_tr_op`` is the opaque envelope's overall conductance; the standard
    splits it in *series* into the mass-to-surface part ``H_tr_ms`` and the
    remaining mass-to-exterior part ``H_tr_em``, hence the reciprocal
    subtraction.
    """
    p = params
    A_tot = p.lambda_at * p.A_floor
    A_m = p.f_class * p.A_floor
    C_m = p.cm_per_area * p.A_floor

    H_tr_is = p.h_is * A_tot
    H_tr_ms = p.h_ms * A_m
    H_tr_op = p.A_wall * p.U_wall + p.A_roof * p.U_roof
    H_tr_w = p.A_window * p.U_window
    H_tr_em = 1.0 / (1.0 / H_tr_op - 1.0 / H_tr_ms)

    volume = p.A_floor * p.room_height
    # 0.34 Wh/(m^3.K) is the volumetric heat capacity of air; the ACH is per
    # hour, so the two hour units cancel and H_ve comes out in W/K.
    H_ve = 0.34 * p.air_changes_per_hour * volume

    return dict(
        A_tot=A_tot,
        A_m=A_m,
        C_m=C_m,
        H_tr_is=H_tr_is,
        H_tr_ms=H_tr_ms,
        H_tr_w=H_tr_w,
        H_tr_em=H_tr_em,
        H_tr_op=H_tr_op,
        H_ve=H_ve,
    )


def total_heat_loss_coefficient(params: HVACParams) -> float:
    """Steady-state W/K from indoors to outdoors -- the headline envelope figure."""
    c = zone_conductances(params)
    return c["H_tr_op"] + c["H_tr_w"] + c["H_ve"]


# ---------------------------------------------------------------------------
# Schedules and disturbances (deterministic functions of time)
# ---------------------------------------------------------------------------


def hour_of_day(time, params: HVACParams):
    return jnp.mod(time * params.delta_t, SECONDS_PER_DAY) / SECONDS_PER_HOUR


def is_occupied(time, params: HVACParams):
    h = hour_of_day(time, params)
    return jnp.logical_and(h >= params.occupied_start_h, h < params.occupied_end_h)


def outdoor_temperature(time, weather_dev, params: HVACParams):
    """Daily sinusoid (coldest ~05:00, warmest ~15:00) plus the OU deviation."""
    h = hour_of_day(time, params)
    # +cos peaks at h = 15 (warmest mid-afternoon) and troughs 12 h later at
    # h = 3 (coldest before dawn). The sign was inverted originally, which put
    # the daily maximum at 3 a.m.
    daily = params.T_out_amplitude * jnp.cos(2.0 * jnp.pi * (h - 15.0) / 24.0)
    return params.T_out_mean + daily + weather_dev


def solar_gain(time, params: HVACParams):
    """Solar power through the glazing (W). Zero at night, peak at solar noon."""
    h = hour_of_day(time, params)
    daylight = jnp.logical_and(h >= 7.0, h <= 19.0)
    shape = jnp.sin(jnp.pi * jnp.clip((h - 7.0) / 12.0, 0.0, 1.0))
    return jnp.where(daylight, params.solar_peak * shape, 0.0) * (
        params.A_window * params.g_window
    )


def internal_gain(time, params: HVACParams):
    """Occupancy / equipment heat (W)."""
    per_area = jnp.where(
        is_occupied(time, params), params.gain_occupied, params.gain_unoccupied
    )
    return per_area * params.A_floor


def scheduled_setpoint(time, setpoint_occupied, params: HVACParams):
    """Occupied setpoint during the day, night setback otherwise."""
    return jnp.where(
        is_occupied(time, params), setpoint_occupied, params.setpoint_setback
    )


# ---------------------------------------------------------------------------
# Node solution
# ---------------------------------------------------------------------------


def solve_air_and_surface(T_mass, T_out, Q_heat, phi_st, phi_ia, params: HVACParams):
    """Algebraic air and surface temperatures given the mass node.

    Neither node has capacitance, so their balances are two linear equations::

        air:     H_tr_is*(T_s - T_air) + H_ve*(T_out - T_air) + phi_ia + Q = 0
        surface: H_tr_ms*(T_m - T_s) + H_tr_w*(T_out - T_s)
                 + H_tr_is*(T_air - T_s) + phi_st = 0

    Substituting the first into the second gives a closed form -- no solver,
    no stiffness, and exactly the reduction that makes 5R1C cheap.
    """
    c = zone_conductances(params)
    H_tr_is, H_tr_ms, H_tr_w, H_ve = (
        c["H_tr_is"],
        c["H_tr_ms"],
        c["H_tr_w"],
        c["H_ve"],
    )

    denom_air = H_tr_is + H_ve
    a = H_tr_is / denom_air
    b = (H_ve * T_out + phi_ia + Q_heat) / denom_air

    T_surface = (H_tr_ms * T_mass + H_tr_w * T_out + phi_st + H_tr_is * b) / (
        H_tr_ms + H_tr_w + H_tr_is * (1.0 - a)
    )
    T_air = a * T_surface + b
    return T_air, T_surface


def split_gains(phi_int, phi_sol, params: HVACParams):
    """Distribute internal and solar gains across the three nodes (ISO 13790).

    Half the internal gain goes to the air node; the rest, with all the solar
    gain, is split between the surface and mass nodes by area ratio.

    Note the three shares deliberately do **not** sum to the total: the
    standard withholds ``H_tr_w / (9.1 * A_tot)`` of the remainder, the radiant
    fraction absorbed by the glazing and re-transmitted straight outdoors
    rather than entering the zone. That is an ISO 13790 convention, not a
    conservation error -- ``test_gain_split_follows_the_iso_convention``
    pins the exact identity.
    """
    c = zone_conductances(params)
    A_tot, A_m, H_tr_w = c["A_tot"], c["A_m"], c["H_tr_w"]
    phi_ia = 0.5 * phi_int
    remainder = 0.5 * phi_int + phi_sol
    phi_m = (A_m / A_tot) * remainder
    phi_st = (1.0 - A_m / A_tot - H_tr_w / (9.1 * A_tot)) * remainder
    return phi_ia, phi_st, phi_m


def compute_velocity(position, action, time, weather_dev, params: HVACParams):
    """RHS for ``[T_mass, Q_emitter]``.

    ``action`` is the commanded heating power in W. The emitter lags it with a
    first-order time constant, representing radiator/water thermal inertia --
    a real heating system cannot deliver a step change in output.
    """
    p = params
    T_mass, Q_emitter = position[0], position[1]
    Q_command = action

    c = zone_conductances(p)
    T_out = outdoor_temperature(time, weather_dev, p)
    phi_int = internal_gain(time, p)
    phi_sol = solar_gain(time, p)
    phi_ia, phi_st, phi_m = split_gains(phi_int, phi_sol, p)

    T_air, T_surface = solve_air_and_surface(
        T_mass, T_out, Q_emitter, phi_st, phi_ia, p
    )

    dT_mass = (
        c["H_tr_ms"] * (T_surface - T_mass) + c["H_tr_em"] * (T_out - T_mass) + phi_m
    ) / c["C_m"]
    dQ_emitter = (Q_command - Q_emitter) / p.emitter_tau

    return jnp.array([dT_mass, dQ_emitter]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: float,
    state: HVACState,
    params: HVACParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` in [-1, 1] maps to heating power in [0, Q_heat_max]."""
    p = params
    Q_command = convert_raw_action_to_range(
        action_raw, min_action=0.0, max_action=p.Q_heat_max
    )

    # OU weather deviation. The innovation is drawn from a key folded with
    # ``state.time`` so a caller passing a constant key -- which every rollout
    # helper in this repo does -- still gets a genuine zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.T_out_noise_std * jnp.sqrt(2.0 * WEATHER_OU_THETA * p.delta_t)
    weather_dev = (
        state.weather_dev
        - WEATHER_OU_THETA * state.weather_dev * p.delta_t
        + sigma * noise
    )

    _compute_velocity = partial(
        compute_velocity,
        action=Q_command,
        time=state.time,
        weather_dev=weather_dev,
        params=p,
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.T_mass, state.Q_emitter]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    T_mass, Q_emitter = new_positions[0], new_positions[1]

    new_time = state.time + 1
    T_out = outdoor_temperature(new_time, weather_dev, p)
    phi_ia, phi_st, _ = split_gains(
        internal_gain(new_time, p), solar_gain(new_time, p), p
    )
    T_air, T_surface = solve_air_and_surface(
        T_mass, T_out, Q_emitter, phi_st, phi_ia, p
    )

    return (
        state.replace(
            T_mass=T_mass,
            Q_emitter=Q_emitter,
            T_air=T_air,
            T_surface=T_surface,
            T_out=T_out,
            weather_dev=weather_dev,
            target_T=scheduled_setpoint(new_time, state.setpoint_occupied, p),
            Q_command=Q_command,
            time=new_time,
        ),
        None,
    )


@partial(jax.jit, static_argnames=["params"])
def get_obs(state: HVACState, params: HVACParams):
    """Plant instrumentation only.

    ``[T_air, T_out, heat_pct, solar_norm, sin_h, cos_h, target_T]``

    A building management system measures zone air temperature, outdoor
    temperature, its own heat output and (usually) a rooftop pyranometer, and
    knows the time of day. It does **not** measure the structural thermal mass
    or the weather deviation -- so ``T_mass``, ``T_surface`` and
    ``weather_dev`` are hidden, and the mass is precisely the state that
    governs the building's multi-hour response.

    Time of day is given as sin/cos so it is continuous across midnight.
    """
    h = hour_of_day(state.time, params)
    angle = 2.0 * jnp.pi * h / 24.0
    heat_pct = 100.0 * state.Q_emitter / params.Q_heat_max
    solar_norm = solar_gain(state.time, params) / (
        params.solar_peak * params.A_window * params.g_window + 1e-9
    )
    return jnp.array(
        [
            state.T_air,
            state.T_out,
            heat_pct,
            solar_norm,
            jnp.sin(angle),
            jnp.cos(angle),
            state.target_T,
        ]
    )


def check_is_terminal(state: HVACState, params: HVACParams, xp=jnp):
    terminated = xp.logical_or(
        state.T_air <= params.T_air_min, state.T_air >= params.T_air_max
    )
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward(state: HVACState, params: HVACParams, xp=jnp):
    """Comfort tracking minus energy use.

    Comfort is a squared-normalised band rather than a Gaussian: it stays
    informative several degrees out, so a controller that is far off still
    sees a gradient back toward the setpoint.
    """
    err = xp.abs(state.target_T - state.T_air)
    comfort = xp.clip(1.0 - err / (2.0 * params.comfort_band), 0.0, 1.0) ** 2
    energy = state.Q_emitter / params.Q_heat_max
    return comfort - params.energy_weight * energy


def energy_use_kwh(state: HVACState, params: HVACParams):
    """Energy delivered in the last step, in kWh -- for reporting and validation."""
    return state.Q_emitter * params.delta_t / 3.6e6
