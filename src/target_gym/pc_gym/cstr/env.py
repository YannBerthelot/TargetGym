"""
Jacketed exothermic CSTR, a PARSEX-like member of its family (charter
Amendment 3).

See ``PHYSICS.md`` in this directory for the sources, the limits and the
score, and ``REALITY.md`` for the plant record they come from.

The job [S5; S6]: hold the reactor temperature at its setpoint by moving the
cooling water, while keeping heat removal ahead of heat release so the reactor
never runs away; a high-temperature interlock cuts the feed [S3].

Model (all per minute, mol and K)::

    dC_A/dt = (C_Af - C_A) / tau - k(T) C_A,         k = k0 exp(-E/R / T)
    dT/dt   = (T_f - T) / tau + (-dH) k C_A / (rho cp)
              - UA (T - T_j) / (rho cp V)
    dT_j/dt = [F_w cp_w (T_w,in - T_j) + UA (T - T_j)] / (M_j cp_w)

the PARSEX pilot's first-order reaction and recirculation cooler [S1], with
the cooler lumped as a jacket of its own holdup, and fresh cooling-water flow
F_w the operator's move [S1, p. 4]. UA is derived from S1's design point.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

#: Water's density (g/L) and heat capacity (J/(g K)), textbook.
RHO_WATER = 1000.0
CP_WATER = 4.18


@struct.dataclass
class CSTRParams(EnvParams):
    # ---- Reaction and reactor (the PARSEX pilot [S1, pp. 4-5]) ----
    k0: float = 2.0e11 * 60.0  # 1/min [S1: 2.0e11 1/s]
    EA_over_R: float = 10700.0  # K [S1]
    deltaHr: float = -2.0e5  # J/mol [S1: -200 kJ/mol]
    Caf: float = 1.5  # mol/L [S1: 1.5 kmol/m3]
    q: float = 5.4  # L/min [S1: 90 g/s of feed]
    V: float = 183.6  # L, derived: q x 34 min [S1, p. 5]
    rho: float = RHO_WATER  # g/L, the aqueous feed as water
    C: float = CP_WATER  # J/(g K)

    # ---- Cooler, lumped as a jacket (charter Amendment 3: the jacket in code)
    #: Derived from S1's design point (T 58 C, cooler 47 C, CA 0.31 mol/L, feed
    #: 24 C): the heat the balance leaves, 8.56 kW, over 11 K.
    UA: float = 46717.0  # J/(min K)
    #: Fresh cooling water, the move. The floor is S1's 40 g/s [S1, Table 1];
    #: the top is the 136 g/s S1's Figure 2 logs, since the plant ran above
    #: Table 1's 110 g/s: "around 120 g/s" at the design point [S1, p. 6].
    Fw_min: float = 40.0 * 60.0  # g/min
    Fw_max: float = 136.0 * 60.0  # g/min
    #: Fresh water inlet temperature, derived: the 8.56 kW the design point
    #: leaves (UA's row) at S1's 120 g/s with the cooler at 47 C [S1, p. 6;
    #: Tables 3, 4] is 17.1 K of rise, so water enters at 29.9 C.
    Tw_in: float = 303.08  # K
    #: The jacket's water holdup. Ours, family: a jacket holds a fraction of
    #: its vessel; 20 % of the reactor's mass, which puts its time constant at
    #: 2.2 min (derived), the scale of the 2 min jacket dead time one plant
    #: reports [S6].
    M_j: float = 36720.0  # g

    # ---- Feed (S1, Tables 3, 4: 23.1 and 24.9 C on two days) ----
    feed_T_range: Tuple[float, float] = (296.25, 298.05)  # K, drawn per episode

    # ---- Setpoint and interlock ----
    #: The normal day's reactor temperature, 58 C [S1, Tables 3, 4].
    T_setpoint: float = 331.15  # K
    #: The interlock cuts the feed on high temperature [S3, PDF p. 38]. Ours,
    #: no PARSEX figure: 75 C, 17 K above the setpoint, short of the 81 C an
    #: adiabatic rise from the feed reaches, so a runaway trips it.
    T_trip: float = 348.15  # K

    # ---- Episode (rules 17 and 20) ----
    #: Minutes. 4 h at 15 s steps: over twice the 110 min a 2 C setpoint
    #: step took to settle [S1, Fig. 2]. Ours.
    delta_t: float = 0.25
    time_unit_seconds: float = 60.0
    max_steps_in_episode: int = 960
    #: Each scenario's one event, at a minute drawn in this window (ours).
    event_window: Tuple[float, float] = (30.0, 150.0)  # min
    #: Scenario events, 0 when off (scenarios.py):
    setpoint_step: float = 0.0  # K, a setpoint move [S1, Fig. 2: 2 C]
    k0_drop: float = 0.0  # fraction lost [S1, Fig. 3: 40 %]
    #: Minutes the loss takes, linear; 0 is S1's instant step. Activity decays
    #: through operation [S1, p. 6] but no source gives a rate inside hours.
    k0_decay: float = 0.0  # min
    feed_rise: float = 0.0  # fraction of Caf [S3: feed build-up], size ours
    cooling_loss: float = 0.0  # min the cooling water fails [S2; S3]

    # ---- Score (charter rule 15) ----
    # The job is holding the temperature [S5; S6], so the score tracks it:
    # quadratic on the thermometry's resolution (ours: 0.1 K, a reactor
    # probe's), plus the interlock trip, priced as the lost production of a
    # restart (ours: 4 h of feed, seven residence times; no restart time found).
    reward_version: int = 3
    e_floor: float = 0.1  # K
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    # Above the worst untripped step: a reactor cooled to its feed, 35 K off.
    failure_cost: float = 2.0 * (35.0 / 0.1) ** 2
    restart_steps: int = 960
    rho_floor_tracking: float = 0.0
    rho_floor: float = 0.0
    floor_is_documented_minimum: bool = True
    #: Version 1 only.
    precision_floor: float = 0.1  # K


#: Units of the CSTRParams fields, for the generated environment pages.
PARAM_UNITS = {
    "k0": "1/min",
    "EA_over_R": "K",
    "deltaHr": "J/mol",
    "Caf": "mol/L",
    "q": "L/min",
    "V": "L",
    "rho": "g/L",
    "C": "J/(g*K)",
    "UA": "J/(min*K)",
    "Fw_min": "g/min",
    "Fw_max": "g/min",
    "Tw_in": "K",
    "M_j": "g",
    "feed_T_range": "K",
    "T_setpoint": "K",
    "T_trip": "K",
    "delta_t": "min",
    "time_unit_seconds": "s",
    "event_window": "min",
    "setpoint_step": "K",
    "k0_drop": "-",
    "k0_decay": "min",
    "feed_rise": "-",
    "cooling_loss": "min",
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
class CSTRState(EnvState):
    C_a: float  # mol/L, HIDDEN: no online analyser [REALITY.md]
    T: float  # K, reactor, read through its thermowell
    target_T: float  # K, the setpoint

    F_w: float  # g/min, the cooling water as it ran last step
    T_j: float = struct.field(default=320.15, kw_only=True)  # K, jacket
    #: The feed's temperature this episode (K), HIDDEN.
    T_feed: float = struct.field(default=297.15, kw_only=True)
    #: The scenario's event minute (a long-past or never time for none).
    event_at: float = struct.field(default=1.0e9, kw_only=True)

    actuator_held: jax.Array = actuator_field(1)


def rate(T, C_a, params: CSTRParams, k0=None):
    """The reaction rate k(T) C_A, mol/(L min)."""
    k0 = params.k0 if k0 is None else k0
    return k0 * jnp.exp(-params.EA_over_R / T) * C_a


def compute_velocity(position, action, params: CSTRParams, extra=None):
    """d/dt of ``[C_A, T, T_j]``; ``action`` is the water flow F_w (g/min),
    ``extra`` the feed's ``(T_f, C_Af, k0)`` now (the params' by default)."""
    p = params
    T_f, Caf, k0 = (p.feed_T_range[0], p.Caf, p.k0) if extra is None else extra
    C_a, T, T_j = position[0], position[1], position[2]
    r = rate(T, C_a, p, k0)
    tau = p.V / p.q
    dC = (Caf - C_a) / tau - r
    dT = (
        (T_f - T) / tau
        + (-p.deltaHr) * r / (p.rho * p.C)
        - p.UA * (T - T_j) / (p.rho * p.C * p.V)
    )
    dTj = (action * CP_WATER * (p.Tw_in - T_j) + p.UA * (T - T_j)) / (p.M_j * CP_WATER)
    return jnp.array([dC, dT, dTj]), None


def events(state: CSTRState, params: CSTRParams, minute):
    """The scenario's event at ``minute``: ``(setpoint, Caf, k0, cooling up)``."""
    p = params
    after = minute >= state.event_at
    lost = jnp.logical_and(after, minute < state.event_at + p.cooling_loss)
    since = minute - state.event_at
    decayed = jnp.where(
        p.k0_decay > 0.0,
        jnp.clip(since / jnp.maximum(p.k0_decay, 1e-9), 0.0, 1.0),
        jnp.where(after, 1.0, 0.0),
    )
    return (
        p.T_setpoint + jnp.where(after, p.setpoint_step, 0.0),
        p.Caf * (1.0 + jnp.where(after, p.feed_rise, 0.0)),
        p.k0 * (1.0 - p.k0_drop * decayed),
        jnp.logical_not(lost),
    )


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    F_w_raw: float,
    state: CSTRState,
    params: CSTRParams,
    integration_method: str = "rk4_1",
):
    """``F_w_raw`` in [-1, 1] maps to the cooling water in [Fw_min, Fw_max];
    during a cooling loss no water flows, whatever the command [S2; S3]."""
    p = params
    minute = state.time * p.delta_t
    F_w = convert_raw_action_to_range(F_w_raw, min_action=p.Fw_min, max_action=p.Fw_max)
    target, Caf, k0, cooling = events(state, p, minute)
    F_w = jnp.where(cooling, F_w, 0.0)
    (C_a, T, T_j), metrics = integrate_dynamics(
        positions=jnp.array([state.C_a, state.T, state.T_j]),
        delta_t=p.delta_t,
        compute_velocity=partial(
            compute_velocity, action=F_w, params=p, extra=(state.T_feed, Caf, k0)
        ),
        method=integration_method,
    )
    target_next, *_ = events(state, p, minute + p.delta_t)
    return (
        state.replace(
            C_a=C_a,
            T=T,
            T_j=T_j,
            F_w=F_w,
            target_T=target_next,
            time=state.time + 1,
        ),
        metrics,
    )


def steady_state(T, T_f, params: CSTRParams):
    """The steady state that holds the reactor at ``T`` with feed at ``T_f``:
    ``(C_A, T_j, F_w)``, closed form from the three balances."""
    p = params
    tau = p.V / p.q
    k = p.k0 * jnp.exp(-p.EA_over_R / T)
    C_a = p.Caf / (1.0 + k * tau)
    # Heat the jacket must take, J/min.
    duty = p.q * p.rho * p.C * (T_f - T) + (-p.deltaHr) * k * C_a * p.V
    T_j = T - duty / p.UA
    F_w = duty / (CP_WATER * (T_j - p.Tw_in))
    return C_a, T_j, F_w


# Deliberately not jitted (see the other process tasks).
def get_obs(state: CSTRState, params: CSTRParams):
    """``[T, T_j, F_w_pct, T_setpoint]``: the reactor's temperature probe, the
    jacket outlet temperature [S6; S8, Fig. 6], the water flow, logged on
    PARSEX [S1, Fig. 2], and the setpoint. No composition: an online analyser
    on such a reactor was not found, and Bayer sampled once a day [S3]."""
    pct = 100.0 * (state.F_w - params.Fw_min) / (params.Fw_max - params.Fw_min)
    return jnp.array([state.T, state.T_j, pct, state.target_T])


def check_is_terminal(state: CSTRState, params: CSTRParams, xp=jnp):
    """The interlock: high reactor temperature cuts the feed [S3]."""
    terminated = jnp.logical_or(state.T >= params.T_trip, ~jnp.isfinite(state.T))
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: CSTRState, params: CSTRParams, xp=jnp, tripped=None):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``):
    the temperature held at its setpoint [S5; S6], and the interlock trip.

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


def compute_reward_v1(state: CSTRState, params: CSTRParams, xp=jnp):
    """Version 1's log-scaled tracking, kept for reproduction."""
    return log_scaled_reward(
        xp.abs(state.target_T - state.T), params.precision_floor, 20.0, xp
    )


def compute_reward(state: CSTRState, params: CSTRParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
