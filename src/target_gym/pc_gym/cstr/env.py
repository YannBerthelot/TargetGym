"""
Jacketed exothermic CSTR run for what it makes, a PARSEX-like member of its
family (charter Amendment 3; version 8 by the owner's decision of 2026-11-24).

See ``PHYSICS.md`` in this directory for the sources, the limits and the
score, and ``REALITY.md`` for the plant record they come from.

The job [S13; S5; S6]: make product. Hold the reactor inside its temperature
band by the cooling water, and carry as much feed as the cooling allows, cutting
it when the cooling saturates [S13], so the reactor never runs away; a
high-temperature interlock cuts the feed [S3].

Model (all per minute, mol and K)::

    dC_A/dt = (C_Af - C_A) q / V - k(T) C_A,         k = k0 exp(-E/R / T)
    dT/dt   = (T_f - T) q / V + (-dH) k C_A / (rho cp)
              - UA (T - T_j) / (rho cp V)
    dT_j/dt = [F_w cp_w (T_w,in - T_j) + UA (T - T_j)] / (M_j cp_w)

the PARSEX pilot's first-order reaction and recirculation cooler [S1], with
the cooler lumped as a jacket of its own holdup. The operator moves the fresh
cooling water F_w [S1, p. 4] and the feed q [S1, Table 1; S13]. UA is derived
from S1's design point.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import log_scaled_reward

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
    q: float = 5.4  # L/min, the design feed [S1: 90 g/s]
    #: The feed, the second handle. The top is S1's 110 g/s [S1, Table 1]. The
    #: floor is ours: S1's range starts at 0 and no turndown was found.
    q_min: float = 2.4  # L/min, 40 g/s
    q_max: float = 6.6  # L/min, 110 g/s
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
    #: The feed sits at room temperature, so it follows the day: a cycle
    #: across S1's two days' range [S1, Tables 3, 4], its phase drawn per
    #: episode. The period is ours: room temperature cycles once a day.
    feed_T_range: Tuple[float, float] = (296.25, 298.05)  # K
    feed_T_period: float = 1440.0  # min; 0 holds the feed where it starts

    # ---- Setpoint and interlock ----
    #: The normal day's reactor temperature, 58 C [S1, Tables 3, 4].
    T_setpoint: float = 331.15  # K
    #: Product made more than this from the setpoint is off-spec. Ours: no
    #: tolerance was found (REALITY.md, third search); flat inside, by the
    #: owner's decision of 2026-11-24.
    T_band: float = 1.0  # K
    #: Minutes a setpoint move takes. Derived: S1's closed loop took about an
    #: hour to carry its 2 C move to its peak [S1, Fig. 2], and a band that
    #: jumped 2 K at once could not be kept by anything.
    setpoint_ramp: float = 60.0  # min
    #: The water valve at the reset, of its span: where [S13]'s example holds
    #: it, short of its limit [S13, pp. 436, 438].
    valve_rest: float = 0.95
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
    feed_rise: float = 0.0  # fraction of Caf [S3: feed build-up], size derived
    cooling_loss: float = 0.0  # min the cooling water fails [S2; S3]
    #: Minutes a setpoint move, a catalyst loss or a feed rise lasts before it
    #: returns: a move back, a catalyst change, the feed batch run out. Only a
    #: stream sets it (scenarios.py); in a window the event holds to the end.
    event_hold: float = 1.0e9  # min

    # ---- Score (charter B5) ----
    # The job is product [S13]: the score is the production lost against the
    # nameplate (the top feed fully converted), in fractions of it per step.
    # Product made outside the band counts for nothing, and a trip loses the
    # production of a restart (ours: 4 h, seven residence times; no restart
    # time found). One unit throughout, so no price is needed.
    reward_version: int = 3
    #: The thermometry's resolution (ours: 0.1 K, a reactor probe's): what the
    #: band's edge is read to.
    e_floor: float = 0.1  # K
    failure_cost: float = 1.0  # a step of nameplate production
    restart_steps: int = 960
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
    "q_min": "L/min",
    "q_max": "L/min",
    "V": "L",
    "rho": "g/L",
    "C": "J/(g*K)",
    "UA": "J/(min*K)",
    "Fw_min": "g/min",
    "Fw_max": "g/min",
    "Tw_in": "K",
    "M_j": "g",
    "feed_T_range": "K",
    "feed_T_period": "min",
    "T_setpoint": "K",
    "T_band": "K",
    "setpoint_ramp": "min",
    "valve_rest": "fraction of span",
    "T_trip": "K",
    "delta_t": "min",
    "time_unit_seconds": "s",
    "event_window": "min",
    "setpoint_step": "K",
    "k0_drop": "-",
    "k0_decay": "min",
    "feed_rise": "-",
    "cooling_loss": "min",
    "event_hold": "min",
    "reward_version": "-",
    "e_floor": "K",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor": "reward",
    "precision_floor": "K",
}


@struct.dataclass
class CSTRState(EnvState):
    C_a: float  # mol/L, HIDDEN: no online analyser [REALITY.md]
    T: float  # K, reactor, read through its thermowell
    target_T: float  # K, the setpoint

    F_w: float  # g/min, the cooling water as it ran last step
    q: float = struct.field(default=5.4, kw_only=True)  # L/min, the feed as it ran
    T_j: float = struct.field(default=320.15, kw_only=True)  # K, jacket
    #: The feed's temperature now (K), HIDDEN, and its daily cycle's phase
    #: at the episode's start (rad).
    T_feed: float = struct.field(default=297.15, kw_only=True)
    feed_phase: float = struct.field(default=0.0, kw_only=True)
    #: The scenario's event minute (a long-past or never time for none).
    event_at: float = struct.field(default=1.0e9, kw_only=True)
    #: When the event returns (min), and its size as it started (``event_sizes``):
    #: an event carries its own size, so it runs to its return whatever the
    #: params do meanwhile (docs/proposals/long-strata.md, S1).
    event_until: float = struct.field(default=1.0e9, kw_only=True)
    event_size: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(5), kw_only=True
    )

    actuator_held: jax.Array = actuator_field(2)


def rate(T, C_a, params: CSTRParams, k0=None):
    """The reaction rate k(T) C_A, mol/(L min)."""
    k0 = params.k0 if k0 is None else k0
    return k0 * jnp.exp(-params.EA_over_R / T) * C_a


def compute_velocity(position, action, params: CSTRParams, extra=None):
    """d/dt of ``[C_A, T, T_j]``; ``action`` is ``(F_w, q)``, the water (g/min)
    and the feed (L/min), ``extra`` the feed's ``(T_f, C_Af, k0)`` now (the
    params' by default)."""
    p = params
    T_f, Caf, k0 = (p.feed_T_range[0], p.Caf, p.k0) if extra is None else extra
    C_a, T, T_j = position[0], position[1], position[2]
    F_w, q = action[0], action[1]
    r = rate(T, C_a, p, k0)
    dC = (Caf - C_a) * q / p.V - r
    dT = (
        (T_f - T) * q / p.V
        + (-p.deltaHr) * r / (p.rho * p.C)
        - p.UA * (T - T_j) / (p.rho * p.C * p.V)
    )
    dTj = (F_w * CP_WATER * (p.Tw_in - T_j) + p.UA * (T - T_j)) / (p.M_j * CP_WATER)
    return jnp.array([dC, dT, dTj]), None


def feed_temperature(phase, params: CSTRParams, minute):
    """The feed's temperature at ``minute``: the middle of ``feed_T_range``
    plus half its width along the day's cycle, from ``phase``."""
    lo, hi = params.feed_T_range
    held = params.feed_T_period <= 0.0
    turn = jnp.where(held, 0.0, minute / jnp.where(held, 1.0, params.feed_T_period))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * jnp.sin(phase + 2.0 * jnp.pi * turn)


def event_sizes(params: CSTRParams):
    """An event's size as it starts: the setpoint move, the activity lost, its
    decay, the feed's rise and the cooling loss, from the scenario's params."""
    p = params
    return jnp.array(
        [p.setpoint_step, p.k0_drop, p.k0_decay, p.feed_rise, p.cooling_loss]
    )


def day_minute(state: CSTRState, params: CSTRParams):
    """The feed's day at the next step, in minutes, from a step count wrapped on
    the day: float32 holds a minute count exactly only to about 4 million, so the
    cycle stays exact however long the plant runs."""
    p = params
    per_day = jnp.maximum(jnp.round(p.feed_T_period / p.delta_t), 1.0)
    return (state.time % per_day.astype(jnp.int32)) * p.delta_t + p.delta_t


def target_at(state: CSTRState, params: CSTRParams, minute):
    """The setpoint at ``minute``: a move ramps over ``setpoint_ramp`` minutes
    from its event, and back the same way from its return."""
    ramp = jnp.maximum(params.setpoint_ramp, 1e-9)
    up = jnp.clip((minute - state.event_at) / ramp, 0.0, 1.0)
    down = jnp.clip((minute - state.event_until) / ramp, 0.0, 1.0)
    return params.T_setpoint + state.event_size[0] * (up - down)


def events(state: CSTRState, params: CSTRParams, minute):
    """The event at ``minute``: ``(setpoint, Caf, k0, cooling up)``, with the size
    it started with, active from ``event_at`` until ``event_until``."""
    p = params
    _, drop, decay, rise, loss = (state.event_size[i] for i in range(5))
    after = minute >= state.event_at
    on = jnp.logical_and(after, minute < state.event_until)
    lost = jnp.logical_and(after, minute < state.event_at + loss)
    since = minute - state.event_at
    decayed = jnp.where(
        decay > 0.0,
        jnp.clip(since / jnp.maximum(decay, 1e-9), 0.0, 1.0),
        jnp.where(after, 1.0, 0.0),
    )
    return (
        target_at(state, p, minute),
        p.Caf * (1.0 + jnp.where(on, rise, 0.0)),
        p.k0 * (1.0 - drop * jnp.where(on, decayed, 0.0)),
        jnp.logical_not(lost),
    )


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    raw,
    state: CSTRState,
    params: CSTRParams,
    integration_method: str = "rk4_1",
):
    """``raw`` in [-1, 1] twice: the cooling water in [Fw_min, Fw_max] and the
    feed in [q_min, q_max]. During a cooling loss no water flows, whatever the
    command [S2; S3]."""
    p = params
    minute = state.time * p.delta_t
    u = jnp.clip(jnp.reshape(raw, (2,)), -1.0, 1.0)
    F_w = p.Fw_min + 0.5 * (u[0] + 1.0) * (p.Fw_max - p.Fw_min)
    q = p.q_min + 0.5 * (u[1] + 1.0) * (p.q_max - p.q_min)
    _, Caf, k0, cooling = events(state, p, minute)
    F_w = jnp.where(cooling, F_w, 0.0)
    (C_a, T, T_j), metrics = integrate_dynamics(
        positions=jnp.array([state.C_a, state.T, state.T_j]),
        delta_t=p.delta_t,
        compute_velocity=partial(
            compute_velocity,
            action=jnp.stack([F_w, q]),
            params=p,
            extra=(state.T_feed, Caf, k0),
        ),
        method=integration_method,
    )
    return (
        state.replace(
            C_a=C_a,
            T=T,
            T_j=T_j,
            F_w=F_w,
            q=q,
            target_T=target_at(state, p, minute + p.delta_t),
            T_feed=feed_temperature(state.feed_phase, p, day_minute(state, p)),
            time=state.time + 1,
        ),
        metrics,
    )


def steady_state(T, T_f, params: CSTRParams, q=None):
    """The steady state that holds the reactor at ``T`` with feed at ``T_f``
    and ``q`` L/min of it (the design feed by default): ``(C_A, T_j, F_w)``,
    closed form from the three balances."""
    p = params
    q = p.q if q is None else q
    k = p.k0 * jnp.exp(-p.EA_over_R / T)
    C_a = p.Caf / (1.0 + k * p.V / q)
    # Heat the jacket must take, J/min.
    duty = q * p.rho * p.C * (T_f - T) + (-p.deltaHr) * k * C_a * p.V
    T_j = T - duty / p.UA
    F_w = duty / (CP_WATER * (T_j - p.Tw_in))
    return C_a, T_j, F_w


def feed_at_rest(T, T_f, F_w, params: CSTRParams):
    """The feed (L/min) whose steady state holds the reactor at ``T`` with the
    water at ``F_w``. The jacket's balance gives its temperature and duty; the
    reactor's is a quadratic in the feed, whose smaller root is the operating
    one."""
    p = params
    T_j = (F_w * CP_WATER * p.Tw_in + p.UA * T) / (F_w * CP_WATER + p.UA)
    duty = p.UA * (T - T_j)
    k = p.k0 * jnp.exp(-p.EA_over_R / T)
    a, c = p.rho * p.C * (T - T_f), k * p.V
    h = (-p.deltaHr) * k * p.V * p.Caf - a * c - duty
    # The smaller root as 2 duty c / (h + sqrt(...)), not (h - sqrt(...)) / 2a:
    # the difference of two numbers near 1e6 loses float32 digits, enough for a
    # reset run eagerly and one run compiled to part by 6e-6 of the feed.
    return 2.0 * duty * c / (h + jnp.sqrt(h * h - 4.0 * a * duty * c))


# Deliberately not jitted (see the other process tasks).
def get_obs(state: CSTRState, params: CSTRParams):
    """``[T, T_j, F_w_pct, q_pct, T_setpoint]``: the reactor's temperature
    probe, the jacket outlet temperature [S6; S8, Fig. 6], the water and feed
    flows, logged on PARSEX [S1, Fig. 2; Table 1], and the setpoint. No
    composition: an online analyser on such a reactor was not found, and Bayer
    sampled once a day [S3].

    This reads whatever ``state`` it is given. The instruments are not here:
    ``CSTR.get_obs`` hands it the measured copy of the state (``CSTR.readings``),
    with the thermowell's lag on ``T`` and the outlet's 2 min dead time on
    ``T_j``, so the agent never reads the state's ``T_j`` directly."""
    p = params
    water = 100.0 * (state.F_w - p.Fw_min) / (p.Fw_max - p.Fw_min)
    feed = 100.0 * (state.q - p.q_min) / (p.q_max - p.q_min)
    return jnp.array([state.T, state.T_j, water, feed, state.target_T])


def check_is_terminal(state: CSTRState, params: CSTRParams, xp=jnp):
    """The interlock: high reactor temperature cuts the feed [S3]."""
    terminated = jnp.logical_or(state.T >= params.T_trip, ~jnp.isfinite(state.T))
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def made(state: CSTRState, params: CSTRParams):
    """Product made now, mol/min: the feed times what it has converted."""
    _, Caf, _, _ = events(state, params, state.time * params.delta_t)
    return state.q * (Caf - state.C_a)


def compute_reward_terms(state: CSTRState, params: CSTRParams, xp=jnp, tripped=None):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``), in
    fractions of the nameplate's production per step: what was not made, what
    was made outside the band [S13; owner, 2026-11-24], and the interlock trip.

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    product = made(state, p) / (p.q_max * p.Caf)
    on_spec = xp.abs(state.T - state.target_T) <= p.T_band
    terms = {
        "lost_production": 1.0 - product,
        "off_spec": xp.where(on_spec, 0.0, product),
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


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
