"""
Boiler drum: natural-circulation drum boiler with shrink-and-swell.

See ``PHYSICS.md`` in this directory for provenance, the parameter table and
the known deviations, and ``REALITY.md`` for how real drum boilers are read,
moved and protected. Every change below cites ``REALITY.md`` (charter rules
11 to 18).

Topology::

                    steam out q_s
                         ^
                    +----|----------------+
       feedwater    |    steam space      |
       q_f -------> |=====================|  <- drum water level
                    |  water + bubbles    |
                    +--+--------------+---+
                       |              ^
              downcomer|              | riser (two-phase, heated)
                       v              |
                       +----[ Q ]-----+

The plant is Astrom and Bell's fourth-order drum model, ported from the
authors' own simulation code with the numbers of the P16-G16 unit
(REALITY.md [S1], Appendix B). Four states: drum pressure ``p``, total water
volume ``V_wt``, steam quality at the riser outlet ``alpha_r`` and the steam
volume under the drum's water surface ``V_sd``.

The job (REALITY.md sec. 6): keep the drum water between the low trip that
starves the tubes and the high trip that floods the turbine while the load
moves, and hold the steam pressure, with feedwater and firing. The unit is in
boiler-following mode [S10]: the turbine valve follows the load plan, and the
boiler holds pressure. Steam that leaves off the plan is the cost, and a trip
loses the production of a hot restart.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range

PA_PER_BAR = 1e5


@struct.dataclass
class BoilerDrumParams(EnvParams):
    delta_t: float = 2.0
    #: One hour of operation around a planned load change (PHYSICS.md, Task).
    max_steps_in_episode: int = 1800

    # ---- Score (rule 15, Amendment 3): money at the imbalance price ----
    # REALITY.md sec. 6: the job is to keep the level between the trips while
    # the load moves. In boiler-following mode the boiler holds pressure for the
    # turbine [S10], so a pressure error is steam, and generation, off the
    # plan, which imbalance settlement prices; a trip is "a costly plant trip"
    # [S2], the generation lost until a hot restart. Steam off the plan within
    # the steam meter's resolution is not counted: no settlement can see it.
    # Precision inside the trips earns nothing (rule 15).
    reward_version: int = 4
    #: Ours, family range 2.18 to 4.51 kg/kWh (ideal-cycle examples, REALITY.md
    #: [S23]): the steam a kWh of the unit's generation takes.
    steam_rate: float = 3.5  # kg/kWh
    #: Ours, family range GBP 30 to 156 per MWh: 59 % of 2018's settlement
    #: periods fell in 30 to 60, the short system averaged 78.84 [S22]; the
    #: annual mean was 44.61 over 2017 to 2020 and 156.20 over 2021 to 2022
    #: [S22b].
    imbalance_price: float = 80.0  # GBP/MWh
    #: Steam off the plan below this is not counted: the steam meter's scatter
    #: (the ``q_steam`` reading, PHYSICS.md).
    steam_resolution: float = 0.047  # kg/s
    #: Restart time priced into a trip: a hot start of a hard coal unit, 2.5 h
    #: at the low end of "most commonly used" (Agora Energiewende 2017,
    #: Table 1, p. 47; REALITY.md [S17]), at 2 s steps.
    restart_steps: int = 4500
    #: The generation a restart step does not make, in GBP: nominal steam for
    #: one step, over the steam rate, at the imbalance price.
    failure_cost: float = 93.35 * 2.0 / 3.5 / 1000.0 * 80.0
    # Kept for the time-in-band KPI tables (benchmark._OUTPUT_FLOORS); the
    # score does not read them.
    e_floor_level: float = 2.67e-3  # m
    e_floor_pressure: float = 0.05  # bar
    level_precision_floor: float = 1e-3  # m, drum level transmitter
    pressure_precision_floor: float = 0.05  # bar, pressure transmitter
    level_band: float = 0.10  # m, the expert tools' error scale
    pressure_band: float = 2.0  # bar, likewise

    # -- geometry and masses (P16-G16, REALITY.md [S1], Appendix B) -----------
    V_t: float = 85.0  # m3   total volume, V_d + V_r + V_dc
    V_d: float = 37.0  # m3   drum
    V_r: float = 37.0  # m3   risers
    V_dc: float = 11.0  # m3   downcomers
    A_d: float = 23.0  # m2   drum area at normal water level
    m_metal: float = 300_000.0  # kg   total metal
    m_riser: float = 100_000.0  # kg   riser metal
    m_drum: float = 20_000.0  # kg   drum metal
    C_metal: float = 650.0  # J/(kg K)
    k_e: float = 200.0  # circulation: q_dc^2 = k_e (rho_w - rho_s) V_r alpha_v
    V_sd0: float = 8.0  # m3   steam under the level with no condensation
    beta: float = 0.3  # -    empirical drum flow parameter, [S1] eq. (19)
    T_d_flow: float = 600.0  # kg   residence time T_d = T_d_flow / q_s
    #: Normal water level: the code's level is (V_wd + V_sd) / A_d - 1.0 m.
    normal_level_height: float = 1.0  # m

    # -- saturated-steam property fits, quadratic in pressure [bar] ----------
    # Least-squares fits to IAPWS saturation values at 60-110 bar; residuals
    # are asserted in the tests (all within 0.5 %).
    ts_coef: Tuple[float, float, float] = (-3.8655462e-3, 1.504000e0, 1.9935496e2)
    rho_w_coef: Tuple[float, float, float] = (3.6974790e-3, -2.3474286e0, 8.8550706e2)
    rho_s_coef: Tuple[float, float, float] = (1.9215686e-3, 3.0847619e-1, 5.4157983e0)
    h_w_coef: Tuple[float, float, float] = (-1.3899160e1, 7.0854286e3, 8.3899193e5)
    h_s_coef: Tuple[float, float, float] = (-9.8207283e0, 8.6952381e1, 2.8145190e6)

    # -- operating point ------------------------------------------------------
    h_feedwater: float = 1.037e6  # J/kg feedwater at 240 C
    Q_max: float = 2.4e8  # W    maximum firing rate (150 % of nominal)
    q_feed_max: float = 200.0  # kg/s maximum feedwater flow
    Q_nominal: float = 1.6e8  # W
    q_steam_nominal: float = 93.35  # kg/s
    p_nominal: float = 85.0  # bar, the pressure the boiler master holds

    # -- the load plan (rule 17, PHYSICS.md "Task") ---------------------------
    #: Loads, as fractions of nominal steam: from the top of a hard coal unit's
    #: minimum load (25 to 40 %, [S17] Table 1) to nominal.
    load_range: Tuple[float, float] = (0.40, 1.0)
    #: Load ramp, % of nominal per minute: the family's 1 to 6 %/min, "1% or 2%
    #: per minute" typical [S10] to state-of-the-art coal's 3 to 6 [S17].
    ramp_rate_range: Tuple[float, float] = (1.0, 6.0)
    #: When the planned ramp starts, s after the episode starts. Ours: early
    #: enough that the slowest full ramp ends inside the hour.
    ramp_start_range: Tuple[float, float] = (300.0, 900.0)
    #: The window's one planned ramp. A stream turns it off: it starts on a flat
    #: plan and plans each load change as an event (``start_event``).
    window_ramp: bool = True

    # -- upsets, the family's bad days (REALITY.md sec. 5; scenarios.py) ------
    #: 0 none, 1 an ID fan runback, 2 a feed pump trip.
    event_kind: int = 0
    #: When the upset strikes, s. Ours, inside the window.
    event_time_range: Tuple[float, float] = (300.0, 900.0)
    #: Where a runback takes the unit, of nominal steam. Ours, family range: "set
    #: to a point that was below the three-mill minimum" [S10]; the minimum
    #: load is 25 to 40 % [S17], so 0.40 to 0.60.
    runback_load: float = 0.50
    #: Runback rate, % of nominal per minute. Ours, family range from [S10]:
    #: "The entire runback occurred in just over two minutes" from 75 % load.
    runback_rate_range: Tuple[float, float] = (12.0, 25.0)
    #: Feed flow left after one pump trips, of nominal steam. Ours, family
    #: range 0.5 to 0.6: the second of two 50 or 60 % pumps (the feed span,
    #: ``q_feed_max``, carries both pumps' headroom).
    feed_trip_capacity: float = 0.5
    #: Until the standby pump delivers, s: "may take more than 30 seconds to
    #: start, come up to speed and pressure" [S11]. Ours, 30 to 60 s.
    standby_delay_range: Tuple[float, float] = (30.0, 60.0)
    #: How long a runback holds before the unit ramps back at its plan's rate, s.
    #: Only a stream sets it (scenarios.py); in a window the runback holds to the
    #: end.
    event_hold: float = 1.0e9

    # -- actuators, fuel then feedwater (PHYSICS.md, Actuators) ----------------
    actuator_rate: Tuple[float, float] = (0.007, 0.05)
    actuator_lag: Tuple[float, float] = (20.0, 1.0)  # s

    # -- limits (rule 14, REALITY.md sec. 3) ------------------------------------
    #: Low level trip, master fuel trip: "Drum Level Low-low (-330 mm)" [S9].
    level_trip_low: float = -0.330  # m
    #: High level trip: "Drum Level HH (+254 mm)" (Hub Power Station FSSS
    #: logic, REALITY.md [S20]); it trips the turbine against carryover [S12].
    level_trip_high: float = 0.254  # m
    #: A level trip acts when the level stays past it "for more than ten
    #: seconds" [S9, items 23 and 24]; instantaneous trips "would cause
    #: excessive false trips" [S21].
    level_trip_delay: float = 10.0  # s


#: Units of the BoilerDrumParams fields, for the generated environment pages.
PARAM_UNITS = {
    "reward_version": "-",
    "restart_steps": "steps",
    "failure_cost": "GBP per step",
    "steam_rate": "kg/kWh",
    "imbalance_price": "GBP/MWh",
    "steam_resolution": "kg/s",
    "event_kind": "-",
    "event_time_range": "s",
    "runback_load": "fraction of nominal steam",
    "runback_rate_range": "% of nominal per minute",
    "feed_trip_capacity": "fraction of nominal steam",
    "standby_delay_range": "s",
    "event_hold": "s",
    "window_ramp": "-",
    "e_floor_level": "m",
    "e_floor_pressure": "bar",
    "level_precision_floor": "m",
    "pressure_precision_floor": "bar",
    "level_band": "m",
    "pressure_band": "bar",
    "V_t": "m^3",
    "V_d": "m^3",
    "V_r": "m^3",
    "V_dc": "m^3",
    "A_d": "m^2",
    "m_metal": "kg",
    "m_riser": "kg",
    "m_drum": "kg",
    "C_metal": "J/(kg K)",
    "k_e": "kg/s^2 per kg/m^3 per m^3",
    "V_sd0": "m^3",
    "beta": "-",
    "T_d_flow": "kg",
    "normal_level_height": "m",
    "h_feedwater": "J/kg",
    "Q_max": "W",
    "q_feed_max": "kg/s",
    "Q_nominal": "W",
    "q_steam_nominal": "kg/s",
    "p_nominal": "bar",
    "load_range": "fraction of nominal steam",
    "ramp_rate_range": "% of nominal per minute",
    "ramp_start_range": "s",
    "level_trip_low": "m",
    "level_trip_high": "m",
    "level_trip_delay": "s",
}


@struct.dataclass
class BoilerDrumState(EnvState):
    pressure: jnp.ndarray  # bar
    V_wt: jnp.ndarray  # m3   total water volume
    alpha_r: jnp.ndarray  # -    steam quality at the riser outlet
    V_sd: jnp.ndarray  # m3   steam under the drum's water surface
    level: jnp.ndarray  # m    deviation from normal water level
    q_steam: jnp.ndarray  # kg/s turbine draw, measured
    q_demand: jnp.ndarray  # kg/s what the turbine valve passes at p_nominal
    Q_fuel: jnp.ndarray  # W
    q_feed: jnp.ndarray  # kg/s
    target_pressure: jnp.ndarray  # bar, the boiler master's setpoint
    level_ref: jnp.ndarray  # m3   V_wd + V_sd at normal water level
    # The load plan: from ``load_from`` the valve ramps to ``load_to`` at
    # ``ramp_rate`` from ``ramp_start`` on (``load_plan``).
    load_from: jnp.ndarray  # kg/s
    load_to: jnp.ndarray  # kg/s
    ramp_start: jnp.ndarray  # s
    ramp_rate: jnp.ndarray  # kg/s per s
    # The upset (``event_kind``): when it strikes, how fast the unit runs
    # back, and how long until the standby feed pump delivers.
    event_time: jnp.ndarray  # s, 1e9 with no upset
    event_rate: jnp.ndarray  # kg/s per s
    standby_delay: jnp.ndarray  # s
    # The upset carries its own kind (``event_kind``'s 0, 1, 2) and the time it
    # returns, so it runs to its return whatever the params do meanwhile
    # (docs/proposals/long-strata.md, S1).
    event_kind: jnp.ndarray = struct.field(default=0, kw_only=True)
    event_until: jnp.ndarray = struct.field(default=1.0e9, kw_only=True)  # s
    # The step the plan's clock counts from: plan times above are seconds since
    # it. 0 in a window; a stream rebases it at each event, so they stay small.
    plan_t0: jnp.ndarray = struct.field(default=0, kw_only=True)
    # How long the level has stayed past each trip, s (``level_trip_delay``).
    time_high: jnp.ndarray
    time_low: jnp.ndarray

    actuator_held: jax.Array = actuator_field(2)


# ---------------------------------------------------------------------------
# Saturated steam properties
# ---------------------------------------------------------------------------


def _poly(coef, p):
    return coef[0] * p * p + coef[1] * p + coef[2]


def _dpoly(coef, p):
    """Derivative with respect to pressure, per Pa (the state is in bar)."""
    return (2.0 * coef[0] * p + coef[1]) / PA_PER_BAR


def saturation_temperature(p_bar, params: BoilerDrumParams):
    return _poly(params.ts_coef, p_bar)


def water_density(p_bar, params: BoilerDrumParams):
    return _poly(params.rho_w_coef, p_bar)


def steam_density(p_bar, params: BoilerDrumParams):
    return _poly(params.rho_s_coef, p_bar)


def water_enthalpy(p_bar, params: BoilerDrumParams):
    return _poly(params.h_w_coef, p_bar)


def steam_enthalpy(p_bar, params: BoilerDrumParams):
    return _poly(params.h_s_coef, p_bar)


def latent_heat(p_bar, params: BoilerDrumParams):
    return steam_enthalpy(p_bar, params) - water_enthalpy(p_bar, params)


# ---------------------------------------------------------------------------
# Dynamics: Astrom and Bell's fourth-order model ([S1], Appendix B)
# ---------------------------------------------------------------------------


def _void(alpha_r, rho_w, rho_s, d_rho_w, d_rho_s):
    """Mean volume fraction of steam in the risers and its two derivatives,
    for a linear quality profile ([S1], Appendix B)."""
    z = jnp.maximum(alpha_r, 1e-6) * (rho_w - rho_s) / rho_s
    lz = jnp.log1p(z) / z
    av = rho_w / (rho_w - rho_s) * (1.0 - lz)
    davdar = rho_w / rho_s * (lz - 1.0 / (1.0 + z)) / z
    zp = (rho_s * d_rho_w - rho_w * d_rho_s) * alpha_r / (rho_s * rho_s)
    z1 = (rho_w * d_rho_s - rho_s * d_rho_w) / (rho_w - rho_s) ** 2
    z2 = rho_w / (rho_w - rho_s) * (lz - 1.0 / (1.0 + z)) / z * zp
    davdp = z1 * (1.0 - lz) + z2
    return av, davdar, davdp


def _properties(p_bar, pr):
    return dict(
        rho_w=water_density(p_bar, pr),
        rho_s=steam_density(p_bar, pr),
        h_w=water_enthalpy(p_bar, pr),
        h_s=steam_enthalpy(p_bar, pr),
        d_rho_w=_dpoly(pr.rho_w_coef, p_bar),
        d_rho_s=_dpoly(pr.rho_s_coef, p_bar),
        d_h_w=_dpoly(pr.h_w_coef, p_bar),
        d_h_s=_dpoly(pr.h_s_coef, p_bar),
        d_ts=_dpoly(pr.ts_coef, p_bar),
    )


def void_fraction(alpha_r, p_bar, params: BoilerDrumParams):
    """Mean volumetric steam fraction in the risers."""
    s = _properties(p_bar, params)
    return _void(alpha_r, s["rho_w"], s["rho_s"], s["d_rho_w"], s["d_rho_s"])[0]


def circulation_flow(alpha_v, p_bar, params: BoilerDrumParams):
    """Downcomer flow from the static momentum balance, [S1] eq. (16)."""
    rho_w = water_density(p_bar, params)
    rho_s = steam_density(p_bar, params)
    return jnp.sqrt(
        jnp.maximum(params.k_e * (rho_w - rho_s) * params.V_r * alpha_v, 0.0) + 1e-9
    )


def drum_level(V_wt, alpha_r, V_sd, level_ref, p_bar, params: BoilerDrumParams):
    """Level as a deviation from normal water level, [S1] eqs. (21), (22)."""
    alpha_v = void_fraction(alpha_r, p_bar, params)
    V_wd = V_wt - params.V_dc - (1.0 - alpha_v) * params.V_r
    return (V_wd + V_sd - level_ref) / params.A_d


def steam_draw(q_demand, p_bar, params: BoilerDrumParams):
    """Steam through the turbine valve, proportional to drum pressure.

    ``q_demand`` is what the valve passes at ``p_nominal``.
    """
    return q_demand * p_bar / params.p_nominal


def compute_velocity(position, action, q_demand, params: BoilerDrumParams):
    """Time derivatives of ``[pressure, V_wt, alpha_r, V_sd]``, pressure in
    bar/s. ``action`` is ``(Q_fuel [W], q_feed [kg/s])``.

    The four balances of [S1] Appendix B, ``e @ dx = b``, solved in order: the
    global mass and energy balances for ``(V_wt, p)``, the risers' for
    ``alpha_r`` and the drum's for ``V_sd``.
    """
    p_bar, V_wt, alpha_r, V_sd = position[0], position[1], position[2], position[3]
    Q, q_f = action
    pr = params

    p_bar = jnp.clip(p_bar, 20.0, 200.0)
    alpha_r = jnp.clip(alpha_r, 1e-4, 0.9)
    q_s = steam_draw(q_demand, p_bar, pr)
    s = _properties(p_bar, pr)
    rho_w, rho_s, h_w, h_s = s["rho_w"], s["rho_s"], s["h_w"], s["h_s"]
    d_rho_w, d_rho_s, d_h_w, d_h_s, d_ts = (
        s["d_rho_w"],
        s["d_rho_s"],
        s["d_h_w"],
        s["d_h_s"],
        s["d_ts"],
    )
    h_c = h_s - h_w
    h_f = pr.h_feedwater
    av, davdar, davdp = _void(alpha_r, rho_w, rho_s, d_rho_w, d_rho_s)
    q_dc = circulation_flow(av, p_bar, pr)
    V_st = pr.V_t - V_wt
    V_wd = V_wt - pr.V_dc - (1.0 - av) * pr.V_r
    T_d = pr.T_d_flow / jnp.maximum(q_s, 1.0)

    e11 = rho_w - rho_s
    e12 = V_st * d_rho_s + V_wt * d_rho_w
    e21 = rho_w * h_w - rho_s * h_s
    e22 = (
        V_st * (h_s * d_rho_s + rho_s * d_h_s)
        + V_wt * (h_w * d_rho_w + rho_w * d_h_w)
        - pr.V_t
        + pr.m_metal * pr.C_metal * d_ts
    )
    e32 = (
        ((1.0 - alpha_r) * h_c * d_rho_s + rho_s * d_h_s) * av * pr.V_r
        + (rho_w * d_h_w - alpha_r * h_c * d_rho_w) * (1.0 - av) * pr.V_r
        + (rho_s + (rho_w - rho_s) * alpha_r) * h_c * pr.V_r * davdp
        - pr.V_r
        + pr.m_riser * pr.C_metal * d_ts
    )
    e33 = (rho_s + (rho_w - rho_s) * alpha_r) * h_c * pr.V_r * davdar
    e42 = (
        V_sd * d_rho_s
        + (
            rho_s * V_sd * d_h_s
            + rho_w * V_wd * d_h_w
            - V_sd
            - V_wd
            + pr.m_drum * pr.C_metal * d_ts
        )
        / h_c
        + alpha_r
        * (1.0 + pr.beta)
        * pr.V_r
        * (av * d_rho_s + (1.0 - av) * d_rho_w + (rho_s - rho_w) * davdp)
    )
    e43 = alpha_r * (1.0 + pr.beta) * (rho_s - rho_w) * pr.V_r * davdar
    e44 = rho_s

    b1 = q_f - q_s
    b2 = Q + q_f * h_f - q_s * h_s
    b3 = Q - alpha_r * h_c * q_dc
    b4 = rho_s * (pr.V_sd0 - V_sd) / T_d + (h_f - h_w) * q_f / h_c

    det = e11 * e22 - e12 * e21
    dV_wt = (b1 * e22 - e12 * b2) / det
    dp = (e11 * b2 - e21 * b1) / det  # Pa/s
    dalpha = (b3 - e32 * dp) / e33
    dV_sd = (b4 - e42 * dp - e43 * dalpha) / e44
    return jnp.array([dp / PA_PER_BAR, dV_wt, dalpha, dV_sd]), None


# ---------------------------------------------------------------------------
# The load plan
# ---------------------------------------------------------------------------


def _base_plan(state: BoilerDrumState, seconds):
    span = state.load_to - state.load_from
    moved = state.ramp_rate * jnp.maximum(seconds - state.ramp_start, 0.0)
    return state.load_from + jnp.sign(span) * jnp.minimum(moved, jnp.abs(span))


def plan_seconds(state: BoilerDrumState, params: BoilerDrumParams, ahead=0):
    """Seconds on the plan's clock, ``ahead`` steps on: from ``plan_t0``, so
    float32 counts them exactly however long the plant runs."""
    return (state.time + ahead - state.plan_t0) * params.delta_t


def runback_target(params: BoilerDrumParams):
    return params.runback_load * params.q_steam_nominal


def load_plan(state: BoilerDrumState, seconds, params: BoilerDrumParams):
    """What the turbine valve passes at ``p_nominal`` at ``seconds`` on the
    plan's clock (``plan_seconds``).

    The plan ramps from ``load_from`` to ``load_to`` at ``ramp_rate`` from
    ``ramp_start``. An upset runs the unit back from where the plan stood,
    at ``event_rate``, to ``runback_load``, and holds it there: an ID fan
    runback [S10], or the runback a feed pump trip drives [S10, S9]. At
    ``event_until`` the unit ramps back to its plan at the plan's rate.
    """
    base = _base_plan(state, seconds)
    struck = _base_plan(state, state.event_time)
    held = jnp.minimum(seconds, state.event_until)
    back = jnp.maximum(
        runback_target(params),
        struck - state.event_rate * jnp.maximum(held - state.event_time, 0.0),
    ) + state.ramp_rate * jnp.maximum(seconds - state.event_until, 0.0)
    upset = (state.event_kind > 0) & (seconds >= state.event_time)
    return jnp.where(upset, jnp.minimum(base, back), base)


def actuator_caps(state: BoilerDrumState, seconds, params: BoilerDrumParams):
    """The firing and feedwater the plant can deliver at ``seconds``, W and kg/s.

    After an ID fan trip the fans hold firing to what the runback load needs
    [S10]. After a feed pump trip one pump is left until the standby delivers
    [S10, S11]. The fans are back at ``event_until``.
    """
    pr = params
    after = seconds >= state.event_time
    h_s = steam_enthalpy(pr.p_nominal, pr)
    fan_cap = runback_target(pr) * (h_s - pr.h_feedwater)
    fan_out = after & (seconds < state.event_until)
    Q_cap = jnp.where((state.event_kind == 1) & fan_out, fan_cap, pr.Q_max)
    pump_out = after & (seconds < state.event_time + state.standby_delay)
    q_cap = jnp.where(
        (state.event_kind == 2) & pump_out,
        pr.feed_trip_capacity * pr.q_steam_nominal,
        pr.q_feed_max,
    )
    return Q_cap, q_cap


def compute_next_state(
    action_raw,
    state: BoilerDrumState,
    params: BoilerDrumParams,
    key: jax.Array,
    integration_method: str = "rk4_1",
):
    """``action_raw`` is ``[fuel, feedwater]``, each in [-1, 1]. ``key`` is
    unused: the plan is drawn at reset, and nothing else moves (rule 17)."""
    del key
    pr = params
    action_raw = jnp.atleast_1d(action_raw)
    Q_fuel = convert_raw_action_to_range(
        action_raw[0], min_action=0.0, max_action=pr.Q_max
    )
    q_feed = convert_raw_action_to_range(
        action_raw[1], min_action=0.0, max_action=pr.q_feed_max
    )
    q_demand = load_plan(state, plan_seconds(state, pr, 1), pr)
    Q_cap, q_cap = actuator_caps(state, plan_seconds(state, pr), pr)
    Q_fuel = jnp.minimum(Q_fuel, Q_cap)
    q_feed = jnp.minimum(q_feed, q_cap)

    _compute_velocity = partial(
        compute_velocity, action=(Q_fuel, q_feed), q_demand=q_demand, params=pr
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.pressure, state.V_wt, state.alpha_r, state.V_sd]),
        delta_t=pr.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    pressure = jnp.clip(new_positions[0], 20.0, 200.0)
    V_wt = jnp.clip(new_positions[1], 1.0, pr.V_t - 1.0)
    alpha_r = jnp.clip(new_positions[2], 1e-4, 0.9)
    V_sd = jnp.maximum(new_positions[3], 0.0)

    level = drum_level(V_wt, alpha_r, V_sd, state.level_ref, pressure, pr)
    time_high = jnp.where(
        level >= pr.level_trip_high, state.time_high + pr.delta_t, 0.0
    )
    time_low = jnp.where(level <= pr.level_trip_low, state.time_low + pr.delta_t, 0.0)
    return (
        state.replace(
            pressure=pressure,
            V_wt=V_wt,
            alpha_r=alpha_r,
            V_sd=V_sd,
            level=level,
            q_steam=steam_draw(q_demand, pressure, pr),
            q_demand=q_demand,
            Q_fuel=Q_fuel,
            q_feed=q_feed,
            time_high=time_high,
            time_low=time_low,
            time=state.time + 1,
        ),
        None,
    )


def get_obs(state: BoilerDrumState, params: BoilerDrumParams):
    """What the control room reads (rule 16, REALITY.md sec. 1 and 2).

    ``[level, pressure, q_steam, fuel_pct, feed_pct, target_level,
    target_pressure, load_now, load_planned, s_to_ramp, fuel_cap_pct,
    feed_cap_pct]``

    Drum level, drum pressure and steam flow, the three-element measurements
    [S5, S12]; the fuel and feedwater actuators' positions; and the load plan,
    which the unit master knows ahead [S10]: the valve's load now, the load the
    plan ramps to, and the seconds until the ramp starts (0 once it has). Riser
    quality and the steam under the water surface are hidden: no instrument
    reads them. An upset is read as it strikes, not before: the plan then
    shows the runback's load, and the last two channels the firing and the
    feedwater the plant can still deliver, as the fan and pump trip alarms
    tell the control room.
    """
    seconds = plan_seconds(state, params)
    Q_cap, q_cap = actuator_caps(state, seconds, params)
    upset = (
        (state.event_kind > 0)
        & (seconds >= state.event_time)
        & (seconds < state.event_until)
    )
    planned = jnp.where(
        upset, jnp.minimum(state.load_to, runback_target(params)), state.load_to
    )
    return jnp.array(
        [
            state.level,
            state.pressure,
            state.q_steam,
            100.0 * state.Q_fuel / params.Q_max,
            100.0 * state.q_feed / params.q_feed_max,
            jnp.zeros_like(state.level),
            state.target_pressure,
            state.q_demand,
            planned,
            jnp.maximum(state.ramp_start - seconds, 0.0),
            100.0 * Q_cap / params.Q_max + 0.0 * state.level,
            100.0 * q_cap / params.q_feed_max + 0.0 * state.level,
        ]
    )


def check_is_terminal(state: BoilerDrumState, params: BoilerDrumParams, xp=jnp):
    """A level trip: past either trip level for more than ``level_trip_delay``.

    High level trips the turbine against carryover, low level trips the fuel
    against dry tubes [S12, S9]. No pressure limit is modelled: REALITY.md
    found no sourced value (rule 14).
    """
    high = state.time_high > params.level_trip_delay
    low = state.time_low > params.level_trip_delay
    terminated = xp.logical_or(high, low)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def steam_off_plan(state: BoilerDrumState, params: BoilerDrumParams, xp=jnp):
    """Kilograms of steam delivered off the load plan in one step, beyond the
    steam meter's resolution."""
    off = xp.abs(state.q_steam - state.q_demand) - params.steam_resolution
    return xp.maximum(off, 0.0) * params.delta_t


def imbalance_cost(kg, params: BoilerDrumParams):
    """GBP for ``kg`` of steam's generation at the imbalance price."""
    return kg / params.steam_rate / 1000.0 * params.imbalance_price


def compute_reward_terms(
    state: BoilerDrumState, params: BoilerDrumParams, xp=jnp, tripped=None
):
    """The score's terms, in GBP, each >= 0.

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    if tripped is None:
        tripped, _ = check_is_terminal(state, params, xp)
    terms = {"imbalance": imbalance_cost(steam_off_plan(state, params, xp), params)}
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def compute_reward(state: BoilerDrumState, params: BoilerDrumParams, xp=jnp):
    return R.total(compute_reward_terms(state, params, xp), xp)


def steady_state(params: BoilerDrumParams, p_bar=None, q_steam=None):
    """The balanced operating point at ``p_bar`` and ``q_steam``, at normal
    water level.

    Returns ``(V_wt, alpha_r, V_sd, level_ref, Q)``. At steady state the
    riser balance gives ``alpha_r h_c q_dc(alpha_r) = Q``, a scalar root, and
    the drum balance gives ``V_sd`` explicitly. ``V_wt`` is what puts the
    water surface at normal level, ``normal_level_height`` above the drum's
    reference ([S1] Appendix B).
    """
    pr = params
    p_bar = pr.p_nominal if p_bar is None else p_bar
    q_steam = pr.q_steam_nominal if q_steam is None else q_steam
    h_s = steam_enthalpy(p_bar, pr)
    h_w = water_enthalpy(p_bar, pr)
    h_c = h_s - h_w
    rho_s = steam_density(p_bar, pr)
    Q = q_steam * (h_s - pr.h_feedwater)

    def residual(a):
        av = void_fraction(a, p_bar, pr)
        return a * h_c * circulation_flow(av, p_bar, pr) - Q

    alpha_r = jnp.asarray(0.1)
    grad = jax.grad(residual)
    for _ in range(20):
        alpha_r = jnp.clip(alpha_r - residual(alpha_r) / grad(alpha_r), 1e-3, 0.9)

    T_d = pr.T_d_flow / q_steam
    V_sd = pr.V_sd0 + T_d * (pr.h_feedwater - h_w) * q_steam / (h_c * rho_s)
    av = void_fraction(alpha_r, p_bar, pr)
    level_ref = pr.A_d * pr.normal_level_height
    V_wd = level_ref - V_sd
    V_wt = V_wd + pr.V_dc + (1.0 - av) * pr.V_r
    return V_wt, alpha_r, V_sd, level_ref, Q


def circulation_ratio(params: BoilerDrumParams, p_bar=None, q_steam=None):
    """Circulation flow divided by steam flow: 5 to 15 for natural circulation."""
    pr = params
    p_bar = pr.p_nominal if p_bar is None else p_bar
    q_steam = pr.q_steam_nominal if q_steam is None else q_steam
    _, alpha_r, _, _, _ = steady_state(pr, p_bar, q_steam)
    alpha_v = void_fraction(alpha_r, p_bar, pr)
    return circulation_flow(alpha_v, p_bar, pr) / q_steam
