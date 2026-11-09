"""
LDPE autoclave: an adiabatic high-pressure reactor held at its unstable steady
state by initiator flow (charter Amendment 3).

See ``PHYSICS.md`` in this directory for the sources, the limits and the
score, and ``REALITY.md`` for the plant record they come from.

The family [L1; L2; L3]: ethylene is polymerised by free radicals at 1250 to
1950 bar and 170 to 300 C in a stirred autoclave. The walls remove little
heat, so the reactor is essentially adiabatic and the cold monomer feed is its
cooling; the temperature is held by the initiator fed into it, and "the
industrial reactor normally operates at an unstable steady state", where the
temperature either climbs or falls until polymerisation stops [L1].

Model, one perfectly mixed zone (Marini and Georgakis's perfectly mixed
autoclave [L4]), with the quasi-steady radical balance::

    dI/dt = (I_f - I) / tau - k_d(T) I
    dM/dt = (M_f - M) / tau - k_p(T) M lambda_0
    dT/dt = (T_f - T) / tau + (-dH_p) k_p(T) M lambda_0 / (rho c_p)
    lambda_0 = sqrt(2 f k_d(T) I / k_t(T))
    k = k0 exp(-(E + V_A P) / (R T))

The operator's move is the initiator's concentration in the feed.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

GAS_R = 8.314  # J/(mol K)


@struct.dataclass
class UnstableCSTRParams(EnvParams):
    # ---- Vessel and feed (the RSU autoclave [L3, Tables 3, 4]) ----
    V: float = 0.498  # m3 [L3: 498 L]
    feed_mass: float = 8.0  # kg/s [L3]
    rho: float = 499.0  # kg/m3, the mixture [L3]
    #: Ethylene's molar mass, kg/kmol: the feed's monomer concentration is
    #: rho / 28.05.
    mw: float = 28.05
    #: The cold monomer feed, 353 K [L1, p. 3].
    T_feed: float = 353.0  # K
    #: Its slow wander around that, every day: amplitude and period ours (no
    #: figure found), a fifth of the feed_temperature upset's 5 K, over 10 min.
    feed_T_wander: float = 1.0  # K
    feed_T_wander_period: float = 600.0  # s
    #: Pressure, 1600 atm [L1, p. 3], inside the family's 1250 to 1950 bar [L2].
    P: float = 1600.0 * 101325.0  # Pa

    # ---- Kinetics (Ham et al.'s autoclave set, as pooled in [L2, Table 4]) ----
    kp0: float = 1.0e6  # m3/(kmol s)
    Ep: float = 27.1e3  # J/mol
    Vp: float = -23.1e-6  # m3/mol (the table's -23.1, cm3/mol)
    kt0: float = 6.0e8  # m3/(kmol s): combination and disproportionation, 3e8 each
    Et: float = 16.5e3  # J/mol
    #: The initiator, peroxide "X" of [L2, Table 3] (supplier data): a half-life
    #: of 0.7 s at 513 K, inside the family's 0.02 to 1 s [L2, p. 21].
    kd0: float = 4.20e15  # 1/s
    Ed: float = 153.5e3  # J/mol
    #: Initiator efficiency. Ours, family: the radicals that start chains.
    f_eff: float = 0.5
    #: Ethylene's heat of polymerisation, textbook (about 3.3 MJ/kg).
    dH: float = -93.6e6  # J/kmol
    #: The supercritical mixture's heat capacity. Ours, family: about 2.5
    #: kJ/(kg K) at these conditions.
    cp: float = 2500.0  # J/(kg K)

    # ---- The move: initiator in the feed ----
    #: kmol/m3. Ours: 0 to twice what the setpoint needs (about 4.3e-4 at
    #: 513 K, a mass fraction near the 1.2e-4 the RSU study feeds [L3]).
    I_f_min: float = 0.0
    I_f_max: float = 1.0e-3

    # ---- Setpoint and trip ----
    #: A zone temperature of the Politeno reactor, 513 K [L1, p. 3].
    T_setpoint: float = 513.0  # K
    #: High temperature: past the family's operating range, 300 C [L2], ethylene
    #: decomposition runs away [L3], and the autoclave's protection vents it.
    #: The trip. Ours at the range's top: 573 K.
    T_trip: float = 573.0  # K
    #: Low temperature: polymerisation stops [L1] and the reactor is lost to the
    #: cold feed. Not a trip in the sources; it costs production (the score).

    # ---- Episode (rules 17 and 20) ----
    #: Seconds. 30 min at 1 s, about 60 residence times (ours).
    delta_t: float = 1.0
    max_steps_in_episode: int = 1800
    #: Each scenario's one event, at a second drawn in this window (ours).
    event_window: Tuple[float, float] = (120.0, 900.0)
    #: Scenario events, 0 when off (scenarios.py):
    setpoint_step: float = 0.0  # K, a grade change [L1: zones at 513 to 537 K]
    feed_T_step: float = 0.0  # K, the feed's temperature moves
    efficiency_drop: float = 0.0  # fraction of f lost (impurities), ours
    feed_flow_step: float = 0.0  # fraction of the feed flow, ours

    # ---- Score (charter rule 15) ----
    # The job is the temperature held at its setpoint, within a short time
    # [L1, p. 3]: the polymer's grade rides on it. Quadratic on 1 K (ours, a
    # zone thermocouple's useful resolution), plus the trip, priced as the lost
    # production of a vent and restart (ours: 2 h; no restart time found).
    reward_version: int = 3
    e_floor: float = 1.0  # K
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    # Above any untripped step: twice the cost of a reactor 220 K off (cold
    # feed against the setpoint's 513 K).
    failure_cost: float = 2.0 * (220.0 / 1.0) ** 2
    restart_steps: int = 7200
    rho_floor_tracking: float = 0.0
    rho_floor: float = 0.0
    floor_is_documented_minimum: bool = True
    #: Version 1 only.
    precision_floor: float = 1.0  # K


#: Units of the UnstableCSTRParams fields, for the generated environment pages.
PARAM_UNITS = {
    "V": "m^3",
    "feed_mass": "kg/s",
    "rho": "kg/m^3",
    "mw": "kg/kmol",
    "T_feed": "K",
    "feed_T_wander": "K",
    "feed_T_wander_period": "s",
    "P": "Pa",
    "kp0": "m^3/(kmol*s)",
    "Ep": "J/mol",
    "Vp": "m^3/mol",
    "kt0": "m^3/(kmol*s)",
    "Et": "J/mol",
    "kd0": "1/s",
    "Ed": "J/mol",
    "f_eff": "-",
    "dH": "J/kmol",
    "cp": "J/(kg*K)",
    "I_f_min": "kmol/m^3",
    "I_f_max": "kmol/m^3",
    "T_setpoint": "K",
    "T_trip": "K",
    "delta_t": "s",
    "event_window": "s",
    "setpoint_step": "K",
    "feed_T_step": "K",
    "efficiency_drop": "-",
    "feed_flow_step": "-",
    "reward_version": "-",
    "e_floor": "K",
    "e_tol": "K",
    "tracking_exponent": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
    "precision_floor": "K",
}


@struct.dataclass
class UnstableCSTRState(EnvState):
    I: float  # kmol/m3, initiator in the reactor, HIDDEN
    M: float  # kmol/m3, ethylene, HIDDEN
    T: float  # K, read by its thermocouple
    target_T: float  # K, the setpoint
    I_f: float  # kmol/m3, the initiator feed as it ran last step
    #: The scenario's event second (a never time for none).
    event_at: float = struct.field(default=1.0e9, kw_only=True)
    #: The feed's wander's phase at the episode's start (rad).
    feed_phase: float = struct.field(default=0.0, kw_only=True)

    actuator_held: jax.Array = actuator_field(1)


def feed_temperature(phase, params: UnstableCSTRParams, second):
    """The monomer feed's temperature at ``second`` with no event: its 353 K
    plus the wander from ``phase``."""
    p = params
    turn = jnp.where(p.feed_T_wander_period > 0.0, second / p.feed_T_wander_period, 0.0)
    return p.T_feed + p.feed_T_wander * jnp.sin(phase + 2.0 * jnp.pi * turn)


def rate_constant(k0, E, T, params: UnstableCSTRParams, V_act=0.0):
    """``k0 exp(-(E + V_act P) / (R T))``, as [L2, eq. 18] at Tref infinite."""
    return k0 * jnp.exp(-(E + V_act * params.P) / (GAS_R * T))


def polymerisation_rate(I, M, T, params: UnstableCSTRParams, f=None):
    """``k_p M lambda_0``, kmol/(m3 s), with the radicals at quasi-steady state."""
    p = params
    f = p.f_eff if f is None else f
    kd = rate_constant(p.kd0, p.Ed, T, p)
    kp = rate_constant(p.kp0, p.Ep, T, p, p.Vp)
    kt = rate_constant(p.kt0, p.Et, T, p)
    lam = jnp.sqrt(2.0 * f * kd * jnp.maximum(I, 0.0) / kt)
    return kp * M * lam


def residence(params: UnstableCSTRParams, flow_scale=1.0):
    """tau = rho V / feed mass flow, s."""
    return params.rho * params.V / (params.feed_mass * flow_scale)


def compute_velocity(position, action, params: UnstableCSTRParams, extra=None):
    """d/dt of ``[I, M, T]``; ``action`` is the initiator in the feed (kmol/m3),
    ``extra`` the feed's ``(T_f, f, flow scale)`` now."""
    p = params
    T_f, f, scale = (p.T_feed, p.f_eff, 1.0) if extra is None else extra
    I, M, T = position[0], position[1], position[2]
    tau = residence(p, scale)
    M_f = p.rho / p.mw
    kd = rate_constant(p.kd0, p.Ed, T, p)
    rp = polymerisation_rate(I, M, T, p, f)
    dI = (action - I) / tau - kd * I
    dM = (M_f - M) / tau - rp
    dT = (T_f - T) / tau + (-p.dH) * rp / (p.rho * p.cp)
    return jnp.array([dI, dM, dT]), None


def events(state: UnstableCSTRState, params: UnstableCSTRParams, second):
    """The scenario's event at ``second``: ``(setpoint, T_f, f, flow scale)``."""
    p = params
    on = second >= state.event_at
    return (
        p.T_setpoint + jnp.where(on, p.setpoint_step, 0.0),
        feed_temperature(state.feed_phase, p, second)
        + jnp.where(on, p.feed_T_step, 0.0),
        p.f_eff * (1.0 - jnp.where(on, p.efficiency_drop, 0.0)),
        1.0 + jnp.where(on, p.feed_flow_step, 0.0),
    )


def advance(x, I_f, params: UnstableCSTRParams, extra, integration_method: str):
    """``(I, M, T)`` one step on from ``x`` at the initiator feed ``I_f``
    (kmol/m3) and the feed's ``extra = (T_f, f, flow scale)``, held over the
    step. The plant's step, and the oracle's model of it.

    Each substep moves the initiator exactly, ``dI/dt = I_f / tau - (1 / tau +
    k_d) I`` being linear at the substep's temperature, then the monomer and
    temperature by RK4 at that initiator: initiator decomposition grows stiff
    near the trip (k_d about 40 /s at 573 K), where plain RK4 at a 0.25 s
    substep diverges.
    """
    p = params
    n = int(integration_method.split("_")[1]) if "_" in integration_method else 1
    h = p.delta_t / n
    tau = residence(p, extra[2])

    def substep(carry, _):
        I, M, T = carry
        a = 1.0 / tau + rate_constant(p.kd0, p.Ed, T, p)
        I_eq = I_f / (tau * a)
        I = I_eq + (I - I_eq) * jnp.exp(-a * h)

        def mt(y):
            v, _ = compute_velocity(jnp.array([I, y[0], y[1]]), I_f, p, extra=extra)
            return v[1:]

        y = jnp.array([M, T])
        k1 = mt(y)
        k2 = mt(y + 0.5 * h * k1)
        k3 = mt(y + 0.5 * h * k2)
        k4 = mt(y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return (I, y[0], y[1]), None

    (I, M, T), _ = jax.lax.scan(substep, (x[0], x[1], x[2]), None, length=n)
    return jnp.stack([I, M, T])


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    raw: float,
    state: UnstableCSTRState,
    params: UnstableCSTRParams,
    integration_method: str = "rk4_4",
):
    """``raw`` in [-1, 1] maps to the initiator feed in [I_f_min, I_f_max],
    held over the step (``advance``)."""
    p = params
    second = state.time * p.delta_t
    I_f = convert_raw_action_to_range(raw, min_action=p.I_f_min, max_action=p.I_f_max)
    _, T_f, f, scale = events(state, p, second)
    x = jnp.stack([state.I, state.M, state.T])
    I, M, T = advance(x, I_f, p, (T_f, f, scale), integration_method)
    target, *_ = events(state, p, second + p.delta_t)
    return (
        state.replace(I=I, M=M, T=T, I_f=I_f, target_T=target, time=state.time + 1),
        None,
    )


def steady_state(T, params: UnstableCSTRParams):
    """The steady state at ``T`` with the feed as nominal: ``(I, M, I_f)``,
    closed form. The energy balance fixes the conversion, the monomer balance
    the rate, the radical balance the initiator, its balance the feed."""
    p = params
    tau = residence(p)
    M_f = p.rho / p.mw
    rp = p.rho * p.cp * (T - p.T_feed) / (tau * (-p.dH))
    M = M_f - rp * tau
    kd = rate_constant(p.kd0, p.Ed, T, p)
    kp = rate_constant(p.kp0, p.Ep, T, p, p.Vp)
    kt = rate_constant(p.kt0, p.Et, T, p)
    lam = rp / (kp * M)
    I = kt * lam**2 / (2.0 * p.f_eff * kd)
    return I, M, I * (1.0 + kd * tau)


# Deliberately not jitted (see the other process tasks).
def get_obs(state: UnstableCSTRState, params: UnstableCSTRParams):
    """``[T, I_f_pct, T_setpoint]``: the zone's temperature [L1, p. 3: the
    controllers read measured segment temperatures], the initiator pump's own
    rate, and the setpoint. Conversion and the initiator in the reactor are
    hidden: no on-line measurement of them is in the sources."""
    pct = 100.0 * (state.I_f - params.I_f_min) / (params.I_f_max - params.I_f_min)
    return jnp.array([state.T, pct, state.target_T])


def check_is_terminal(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    """The high-temperature trip: decomposition's protection [L2; L3]."""
    terminated = jnp.logical_or(state.T >= params.T_trip, ~jnp.isfinite(state.T))
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``): the
    temperature held at its setpoint [L1], and the trip.

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    if tripped is None:
        tripped, _ = check_is_terminal(state, params, xp)
    terms = {
        "tracking": R.tracking_cost(
            state.target_T - state.T,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
    }
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def compute_reward_v1(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    """Version 1's log-scaled tracking, kept for reproduction."""
    return log_scaled_reward(
        xp.abs(state.target_T - state.T), params.precision_floor, 220.0, xp
    )


def compute_reward(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
