"""
Building heating with a thermally activated concrete slab (TABS).

See ``PHYSICS.md`` for the model, the numbers and the deviations, and
``REALITY.md`` for how slab-heated buildings are read, moved and protected.
Every design choice below cites ``REALITY.md`` [n] (charter rules 11 to 18).

The zone is ISO 13790's 5R1C network (air, surface and mass nodes) with a
concrete core slab between the heating water and the room, the first-order
TABS model of Gwerder et al. [9, Fig. 3]::

    T_supply --Rt-- T_core (C_slab) --floor and ceiling faces-- T_air

    T_out --H_ve-- T_air --H_tr_is-- T_surface --H_tr_ms-- T_mass --H_tr_em-- T_out
                                         |
                                       H_tr_w -- T_out

The job (REALITY.md sec. 6): choose the water temperature about a day ahead
from the weather, so the room stays inside its comfort band without heating
and cooling on the same day, because the slab cannot correct within the day.
The action is the supply water temperature from the mixing valve [11, 14].
"""

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0

# Ornstein-Uhlenbeck weather deviation: mean-reversion rate (1/s). 1/theta is
# about 12 h, so a warm or cold spell persists across a day. Ours (PHYSICS.md).
WEATHER_OU_THETA = 2.3e-5

#: Steps in the outdoor temperature's 24 h mean, at 15 min steps [9, 13].
DAY_STEPS = 96


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

    # ---- The TABS slab [9, Table 2] ----
    slab_thickness: float = 0.25  # m
    tabs_covering: float = 0.80  # share of the floor with pipes
    R_t: float = 0.08  # m^2 K/W, supply water to core, per covered m^2
    water_flow: float = 15.0  # kg/(h m^2) per covered m^2
    R_flooring: float = 0.125  # m^2 K/W, carpet
    # Ours: concrete as commonly quoted from ISO 10456 (not read).
    concrete_density: float = 2400.0  # kg/m^3
    concrete_heat_capacity: float = 1000.0  # J/(kg K)
    concrete_conductivity: float = 2.0  # W/(m K)
    #: Floor surface to room, from EN 1264's characteristic in [5]: 100 W/m^2
    #: at 9 K and 50 W/m^2 at 4.8 K over a 20 C room, the mean of 11.1 and 10.4.
    h_floor: float = 10.8  # W/(m^2 K)
    #: Ceiling surface to room in heating. Ours: EN 1264-5 Table A.1 as
    #: commonly quoted (not read).
    h_ceiling: float = 6.5  # W/(m^2 K)
    c_water: float = 4186.0  # J/(kg K)

    # ---- Supply water (rule 14) ----
    #: Hard limit at the heating elements, cut by a safety device even without
    #: power, for cement screed [17]: the mixing valve cannot exceed it.
    T_supply_max: float = 55.0  # C
    #: Ours: the valve's lowest setting, below any core temperature here, so
    #: the slab gets no heat.
    T_supply_min: float = 15.0  # C

    # ---- Frost protection, a protected limit [1] ----
    frost_on: float = 4.4  # C, any zone below 40 F
    frost_off: float = 7.2  # C, until all zones exceed 45 F

    # ---- Weather (ours, PHYSICS.md) ----
    T_out_mean: float = 5.0  # C, seasonal mean (heating season)
    T_out_amplitude: float = 5.0  # C, half daily swing
    T_out_noise_std: float = 3.0  # C, OU deviation std
    solar_peak: float = 175.0  # W/m^2 on the glazing at noon

    # ---- Internal gains (ours, PHYSICS.md) ----
    gain_occupied: float = 8.0  # W/m^2 (people, lighting, equipment)
    gain_unoccupied: float = 2.0  # W/m^2
    occupied_start_h: float = 7.0
    occupied_end_h: float = 22.0

    # ---- Comfort band, all hours (rule 14, zone) ----
    #: Room setpoints 21 C heating and 26 C cooling, "used all year" [9, p. 5];
    #: no night setback, as slab buildings run [7, p. 7], [15].
    band_low: float = 21.0  # C
    band_high: float = 26.0  # C
    #: Floor surface, occupied area: 29 C (EN 1264, [5, p. 23]); ASHRAE 55's
    #: comfort range for feet on the floor tops out there too [4].
    floor_surface_max: float = 29.0  # C
    #: Half the band, for the time-in-band KPI tables (benchmark._OUTPUT_FLOORS).
    comfort_tolerance: float = 2.5  # K
    precision_floor: float = 0.1  # K, room temperature sensor resolution

    # ---- Time discretization ----
    delta_t: float = 900.0  # s (15 min)
    #: Two weeks: a heating curve change is judged over two weeks [11, p. 1].
    max_steps_in_episode: int = 1344

    # ---- Score (rule 15), euros per step ----
    # Energy: the heat into the slab's water at ``heat_price``. Comfort: each
    # kelvin-hour outside the band, and of floor surface above 29 C, as [9]
    # counts discomfort (Kh), at ``comfort_price``. Both prices are ours.
    reward_version: int = 3
    #: Ours: a heat price; REALITY.md names energy as the cost but no tariff.
    heat_price: float = 0.10  # EUR per kWh
    #: Ours: twice the heat it takes to hold this building 1 K warmer for 1 h
    #: (159 W/K at EUR 0.10/kWh is EUR 0.016).
    comfort_price: float = 0.03  # EUR per K h
    #: No limit trips this plant (``check_is_terminal``), so a trip costs 0.
    restart_steps: int = 0
    failure_cost: float = 0.0


#: Units of the HVACParams fields, for the generated environment pages.
PARAM_UNITS = {
    "A_floor": "m^2",
    "lambda_at": "-",
    "f_class": "-",
    "cm_per_area": "J/(K m^2)",
    "h_is": "W/(m^2 K)",
    "h_ms": "W/(m^2 K)",
    "A_wall": "m^2",
    "U_wall": "W/(m^2 K)",
    "A_roof": "m^2",
    "U_roof": "W/(m^2 K)",
    "A_window": "m^2",
    "U_window": "W/(m^2 K)",
    "g_window": "-",
    "air_changes_per_hour": "1/h",
    "room_height": "m",
    "slab_thickness": "m",
    "tabs_covering": "-",
    "R_t": "m^2 K/W",
    "water_flow": "kg/(h m^2)",
    "R_flooring": "m^2 K/W",
    "concrete_density": "kg/m^3",
    "concrete_heat_capacity": "J/(kg K)",
    "concrete_conductivity": "W/(m K)",
    "h_floor": "W/(m^2 K)",
    "h_ceiling": "W/(m^2 K)",
    "c_water": "J/(kg K)",
    "T_supply_max": "degC",
    "T_supply_min": "degC",
    "frost_on": "degC",
    "frost_off": "degC",
    "T_out_mean": "degC",
    "T_out_amplitude": "K",
    "T_out_noise_std": "K",
    "solar_peak": "W/m^2",
    "gain_occupied": "W/m^2",
    "gain_unoccupied": "W/m^2",
    "occupied_start_h": "h",
    "occupied_end_h": "h",
    "band_low": "degC",
    "band_high": "degC",
    "floor_surface_max": "degC",
    "comfort_tolerance": "K",
    "precision_floor": "K",
    "reward_version": "-",
    "heat_price": "EUR/kWh",
    "comfort_price": "EUR/(K h)",
    "restart_steps": "steps",
    "failure_cost": "EUR per step",
}


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


def scheduled_band(params: HVACParams):
    """The comfort band, the same at every hour [9, p. 5]."""
    return params.band_low, params.band_high


def band_centre(params: HVACParams):
    return 0.5 * (params.band_low + params.band_high)


def slab_conductances(params: HVACParams):
    """The slab's capacity and its three couplings, W/K and J/K.

    ``H_water``: supply water to core over the covered area, ``1 / R_t`` [9].
    ``H_slab``: core to room through the floor face (half the slab, the
    carpet, the floor's coefficient) and the ceiling face (half the slab, the
    ceiling's coefficient): the zone is one storey of a stack, so both faces
    of a TABS slab heat it, as in [9]'s simulated room.
    ``m_c``: the water flow's heat capacity rate.
    """
    p = params
    half = p.slab_thickness / 2.0 / p.concrete_conductivity
    R_up = half + p.R_flooring + 1.0 / p.h_floor
    R_down = half + 1.0 / p.h_ceiling
    covered = p.tabs_covering * p.A_floor
    return dict(
        C_slab=p.concrete_density
        * p.concrete_heat_capacity
        * p.slab_thickness
        * p.A_floor,
        H_water=covered / p.R_t,
        H_up=p.A_floor / R_up,
        H_down=p.A_floor / R_down,
        H_slab=p.A_floor / R_up + p.A_floor / R_down,
        m_c=p.water_flow * covered / 3600.0 * p.c_water,
        floor_share=(1.0 / p.h_floor) / R_up,
    )


def solve_air_and_surface(T_mass, T_core, T_out, phi_st, phi_ia, params: HVACParams):
    """Air and surface temperatures given the mass and core nodes.

    Neither node has capacitance, so their balances are two linear equations::

        air:     H_tr_is (T_s - T_a) + H_ve (T_out - T_a)
                 + H_slab (T_core - T_a) + phi_ia = 0
        surface: H_tr_ms (T_m - T_s) + H_tr_w (T_out - T_s)
                 + H_tr_is (T_a - T_s) + phi_st = 0
    """
    c = zone_conductances(params)
    H_is, H_ms, H_w, H_ve = c["H_tr_is"], c["H_tr_ms"], c["H_tr_w"], c["H_ve"]
    H_sl = slab_conductances(params)["H_slab"]
    a11 = H_is + H_ve + H_sl
    b1 = H_ve * T_out + phi_ia + H_sl * T_core
    a22 = H_ms + H_w + H_is
    b2 = H_ms * T_mass + H_w * T_out + phi_st
    T_air = (b1 * a22 + H_is * b2) / (a11 * a22 - H_is * H_is)
    T_surface = (b2 + H_is * T_air) / a22
    return T_air, T_surface


def slab_heat(T_supply, T_core, params: HVACParams):
    """Heat from the water into the core, W. Heating only: water below the
    core gives none, since the plant has no chiller (PHYSICS.md)."""
    H_water = slab_conductances(params)["H_water"]
    return H_water * jnp.maximum(T_supply - T_core, 0.0)


def return_temperature(T_supply, T_core, params: HVACParams):
    """Return water, from the heat the flow gave up [12, p. 12]."""
    s = slab_conductances(params)
    return T_supply - slab_heat(T_supply, T_core, params) / s["m_c"]


def floor_surface_temperature(T_core, T_air, params: HVACParams):
    """The floor's surface, on the floor face's resistance chain."""
    s = slab_conductances(params)
    return T_air + (T_core - T_air) * s["floor_share"]


def compute_velocity(position, action, time, weather_dev, params: HVACParams):
    """RHS for ``[T_mass, T_core]``. ``action`` is the supply water temperature."""
    p = params
    T_mass, T_core = position[0], position[1]
    c = zone_conductances(p)
    s = slab_conductances(p)
    T_out = outdoor_temperature(time, weather_dev, p)
    phi_ia, phi_st, phi_m = split_gains(internal_gain(time, p), solar_gain(time, p), p)
    T_air, T_surface = solve_air_and_surface(T_mass, T_core, T_out, phi_st, phi_ia, p)
    dT_mass = (
        c["H_tr_ms"] * (T_surface - T_mass) + c["H_tr_em"] * (T_out - T_mass) + phi_m
    ) / c["C_m"]
    dT_core = (slab_heat(action, T_core, p) + s["H_slab"] * (T_air - T_core)) / s[
        "C_slab"
    ]
    return jnp.array([dT_mass, dT_core]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: float,
    state: "HVACState",
    params: HVACParams,
    key: jax.Array,
    integration_method: str = "rk4_2",
):
    """``action_raw`` in [-1, 1] maps to supply water in [T_supply_min,
    T_supply_max]. Frost protection overrides it with full supply [1]."""
    p = params
    commanded = convert_raw_action_to_range(
        action_raw, min_action=p.T_supply_min, max_action=p.T_supply_max
    )
    T_supply = jnp.where(state.frost > 0.5, p.T_supply_max, commanded)

    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.T_out_noise_std * jnp.sqrt(2.0 * WEATHER_OU_THETA * p.delta_t)
    weather_dev = (
        state.weather_dev
        - WEATHER_OU_THETA * state.weather_dev * p.delta_t
        + sigma * noise
    )
    _compute_velocity = partial(
        compute_velocity,
        action=T_supply,
        time=state.time,
        weather_dev=weather_dev,
        params=p,
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.T_mass, state.T_core]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    T_mass, T_core = new_positions[0], new_positions[1]

    new_time = state.time + 1
    T_out = outdoor_temperature(new_time, weather_dev, p)
    phi_ia, phi_st, _ = split_gains(
        internal_gain(new_time, p), solar_gain(new_time, p), p
    )
    T_air, T_surface = solve_air_and_surface(T_mass, T_core, T_out, phi_st, phi_ia, p)
    frost = jnp.where(
        T_air < p.frost_on, 1.0, jnp.where(T_air > p.frost_off, 0.0, state.frost)
    )
    history = jnp.concatenate([state.T_out_hist[1:], jnp.atleast_1d(T_out)])
    return (
        state.replace(
            T_mass=T_mass,
            T_core=T_core,
            T_air=T_air,
            T_surface=T_surface,
            T_out=T_out,
            T_out_hist=history,
            weather_dev=weather_dev,
            T_supply=T_supply,
            Q_slab=slab_heat(T_supply, T_core, p),
            T_return=return_temperature(T_supply, T_core, p),
            T_floor=floor_surface_temperature(T_core, T_air, p),
            frost=frost,
            time=new_time,
        ),
        None,
    )


def get_obs(state: "HVACState", params: HVACParams):
    """What the building's controller reads (rule 16, REALITY.md sec. 1).

    ``[T_air, T_out, T_out_24h, T_core, T_return, T_supply, sin_h, cos_h,
    band_centre]``

    The room sensor [6, 14]; the outdoor sensor and its mean over the last 24 h,
    which TABS controllers use [9, 13]; the slab's core sensor, which all
    eleven designers fit [7, p. 8], [12, p. 12]; the return water [12, 14]; the
    supply the valve gives; the clock; the band's centre (the band is 21 to 26
    C [9]). The building's other mass is hidden.
    """
    h = hour_of_day(state.time, params)
    angle = 2.0 * jnp.pi * h / 24.0
    return jnp.array(
        [
            state.T_air,
            state.T_out,
            jnp.mean(state.T_out_hist),
            state.T_core,
            state.T_return,
            state.T_supply,
            jnp.sin(angle),
            jnp.cos(angle),
            band_centre(params) + 0.0 * state.T_air,
        ]
    )


def check_is_terminal(state: "HVACState", params: HVACParams, xp=jnp):
    """No trip: frost is a mode change the plant makes itself [1], and the
    supply limit is a safety cut-off [17]. Both are protected limits."""
    terminated = xp.zeros_like(state.T_air, dtype=bool)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def comfort_kelvin_hours(state: "HVACState", params: HVACParams, xp=jnp):
    """Kelvin-hours outside the band and of floor above its limit, one step."""
    p = params
    hours = p.delta_t / 3600.0
    outside = xp.maximum(p.band_low - state.T_air, 0.0) + xp.maximum(
        state.T_air - p.band_high, 0.0
    )
    floor = xp.maximum(state.T_floor - p.floor_surface_max, 0.0)
    return (outside + floor) * hours


def compute_reward_terms(state: "HVACState", params: HVACParams, xp=jnp, tripped=None):
    """The score's terms in euros per step: comfort and heat."""
    p = params
    hours = p.delta_t / 3600.0
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    terms = {
        "comfort": p.comfort_price * comfort_kelvin_hours(state, p, xp),
        "heat": p.heat_price * state.Q_slab / 1000.0 * hours,
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward(state: "HVACState", params: HVACParams, xp=jnp):
    return R.total(compute_reward_terms(state, params, xp), xp)


def energy_use_kwh(state: "HVACState", params: HVACParams):
    """Heat into the slab in the last step, in kWh."""
    return state.Q_slab * params.delta_t / 3.6e6


def steady_state(params: HVACParams, T_out, T_air, phi_int=None, phi_sol=0.0):
    """The settled building holding ``T_air`` at constant ``T_out`` and gains.

    Returns ``(T_mass, T_core, T_surface, T_supply)``: the four node balances,
    solved as one linear system with the water heating the core.
    """
    p = params
    c = zone_conductances(p)
    s = slab_conductances(p)
    if phi_int is None:
        phi_int = p.gain_unoccupied * p.A_floor
    phi_ia, phi_st, phi_m = split_gains(phi_int, phi_sol, p)
    H_is, H_ms, H_w, H_ve, H_em = (
        c["H_tr_is"],
        c["H_tr_ms"],
        c["H_tr_w"],
        c["H_ve"],
        c["H_tr_em"],
    )
    H_sl, H_wat = s["H_slab"], s["H_water"]
    # Unknowns [T_s, T_m, T_c, T_sup].
    A = jnp.array(
        [
            [H_is, 0.0, H_sl, 0.0],
            [-(H_ms + H_w + H_is), H_ms, 0.0, 0.0],
            [H_ms, -(H_ms + H_em), 0.0, 0.0],
            [0.0, 0.0, -(H_wat + H_sl), H_wat],
        ]
    )
    b = jnp.array(
        [
            (H_is + H_ve + H_sl) * T_air - H_ve * T_out - phi_ia,
            -H_w * T_out - H_is * T_air - phi_st,
            -H_em * T_out - phi_m,
            -H_sl * T_air,
        ]
    )
    T_s, T_m, T_c, T_sup = jnp.linalg.solve(A, b)
    return T_m, T_c, T_s, T_sup


@struct.dataclass
class HVACState(EnvState):
    T_mass: float  # building mass node, hidden
    T_core: float  # slab core, measured [7]
    T_air: float  # room air, measured
    T_surface: float  # star node, hidden
    T_out: float  # outdoor air, measured
    T_out_hist: jax.Array  # the last 24 h of outdoor readings, oldest first
    weather_dev: float  # OU deviation from the daily cycle, hidden
    T_supply: float  # supply water the valve gave, C
    Q_slab: float  # heat into the slab from the water, W
    T_return: float  # return water, measured [12]
    T_floor: float  # floor surface, C
    frost: float  # 1 while frost protection holds the supply at its maximum

    actuator_held: jax.Array = actuator_field(1)
