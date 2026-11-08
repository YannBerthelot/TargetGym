"""
Binary distillation — Skogestad's "Column A".

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
A 40-tray binary column with a total condenser and a reboiler, 41 equilibrium
stages in all, indexed 1 = reboiler ... 41 = condenser, feed on stage 21.
Constant molar overflow, constant relative volatility, equilibrium on every
stage, no vapour holdup, and tight level loops on the reboiler and condenser.

Per-stage component balance, with vapour-liquid equilibrium
``y = alpha x / (1 + (alpha-1) x)``::

    tray i :  d(M_i x_i)/dt = L_{i+1} x_{i+1} + V y_{i-1} - L_i x_i - V y_i
    reboiler: M dx_1/dt = L_2 x_2 - V y_1 - B x_1
    condenser:M dx_N/dt = V y_{N-1} - (L + D) x_N

Tray holdup moves with the liquid through it, ``L_i = L0_i + (M_i - M0) / tau_L``,
so a reflux change takes 39 lags to reach the reboiler. Saturated liquid feed, so
steady liquid below the feed is ``L + F``, while vapour is constant throughout.
``D = V - L`` and ``B = L_2 - V`` follow from the level loops.

Why this is a hard target MDP
-----------------------------
* **Severe input interaction.** At the nominal high-purity operating point the
  steady-state gain matrix has an RGA of 36 and a condition number of 145:
  reflux and boilup push both compositions almost the same way, so the
  useful direction is a small difference between two large, nearly-cancelling
  effects. This is the textbook example of an ill-conditioned plant.
* **Slow and high-order.** 41 states with a dominant time constant of hours,
  and the composition profile inside the column is the memory.
* **Only the products are analysed.** 39 of the 41 stage compositions are
  hidden -- a real column has analysers on the product streams and
  thermocouples on a few trays, here one, on stage 15.
* **Feed composition drifts and is unmeasured**, which moves the whole profile.
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

N_STAGES = 41  # 40 trays + reboiler + condenser, minus shared indexing
N_FEED = 21  # feed stage, counting the reboiler as stage 1
TRAYS = jnp.arange(2, N_STAGES)  # stage numbers of the 39 trays, bottom first

# The tray thermocouple. Stage 15, six below the feed, counting the reboiler as
# stage 1: the bottom-section stage with the largest steady-state temperature
# gain to boilup (the maximum gain rule; 220 K per kmol/min at the nominal
# point, derived), and the location Skogestad recommends for column A
# ("Distillation control: dos and don'ts", section 6.3). Its temperature is the
# linear boiling-point curve T = x T_bL + (1 - x) T_bH, with column A's boiling
# points (S. Skogestad, distillation mailing list, 19 May 1999).
TEMPERATURE_STAGE = 15
T_BOIL_LIGHT = 341.9  # K
T_BOIL_HEAVY = 355.4  # K


@struct.dataclass
class DistillationParams(EnvParams):
    # ---- Column ----
    alpha: float = 1.5  # relative volatility
    M_tray: float = 0.5  # kmol tray holdup
    # The drums: D2's holdups "for more realistic studies" (charter Amendment
    # 3); Column A's own analysis uses 0.5 kmol in each [D1, D2].
    M_reboiler: float = 10.0  # kmol
    M_condenser: float = 32.1  # kmol
    tau_L: float = 0.063  # min, liquid lag per tray (Skogestad's taul)
    L_nominal: float = 2.706  # kmol/min, the reflux at which M_i = M_tray

    # ---- Feed ----
    F: float = 1.0  # kmol/min, nominal [D2]
    zF_nominal: float = 0.5  # mole fraction light component [D2]
    qF: float = 1.0  # saturated liquid [D2]
    # The bad days of [D1, p. 33, Fig. 18]: the feed rate steps by 20 % and its
    # composition by 0.1. Each comes once an episode, at a time drawn in
    # ``step_window`` and with a random sign; timing and sign are ours.
    feed_step: float = 0.2  # fraction of F
    zF_step: float = 0.1  # mole fraction
    step_window: Tuple[int, int] = (30, 300)  # min

    # ---- Manipulated variables (LV configuration) ----
    # Ours: no Column A flow limits were found. Wide enough to hold both specs
    # after every sourced feed step, with 5 % to spare: F 0.8 to 1.2 and zF 0.4
    # to 0.6 need L 1.95 to 2.96 and V 2.29 to 3.65 (derived,
    # test_distillation_hydraulics.py). Flooding, the real ceiling on V [D6],
    # has no sourced figure.
    L_min: float = 1.85
    L_max: float = 3.11
    V_min: float = 2.18
    V_max: float = 3.83
    # Distillate is D = V - L; clipped so the overall balance stays physical at
    # the corners of the action box, where V - L could otherwise go negative.
    D_min: float = 0.10
    D_max: float = 0.90

    # ---- Product zones (charter rules 14 and 19) ----
    # On a real splitter the product limits are control zones, a quality
    # limit, not a trip [D4, p. 5, Table 1]: below yD_spec or above xB_spec the
    # product is off spec. Column A's products, 0.99 and 0.01 [D2], are its
    # design point; the family designs a column with its impurity 3 sigma
    # inside the spec, 700 ppm against a 1,000 ppm limit [D7], so each
    # impurity's spec is the design impurity over 0.7. Ours, family member.
    yD_spec: float = 1.0 - 0.01 / 0.7
    xB_spec: float = 0.01 / 0.7

    # ---- Retired version-1 reward fields, read only by reward_version 1 ----
    precision_floor: float = 4e-4  # mole fraction, online analyser scatter
    boilup_cost_weight: float = 0.0

    # ---- Time discretization ----
    # Minutes [D1, D2]. 400 min, about two dominant time constants (194 min
    # [D1, p. 14]), holds both feed steps and the recovery after the later one
    # (about 40 min in [D1, Fig. 18]); ours. The integrator needs 16 substeps
    # at this step size for stability; see ``compute_next_state``.
    delta_t: float = 1.0
    #: ``delta_t`` is in minutes (Skogestad's model unit); seconds per unit.
    time_unit_seconds: float = 60.0
    max_steps_in_episode: int = 400

    # ---- Score (charter rule 15; REALITY.md, "The operator's job") ----
    # Keep both impurities inside their zones at the least reflux and boilup
    # [D4, p. 5]. In kmol of boilup per step: the boilup itself, and each kmol
    # of off-spec product, which is run through the column again at the
    # nominal V/F = 3.206 kmol of boilup per kmol (ours: no price was found,
    # and a second pass is the cheapest way to recover it). Inside the zones
    # purity earns nothing more; it costs the boilup it takes.
    reward_version: int = 3
    boilup_price: float = 1.0  # per kmol of boilup
    reprocess_price: float = 3.206  # kmol of boilup per kmol off spec
    # No trip (the zones are not trips), so a trip has no cost: 0, the
    # charter's convention for a plant with no trip (B5).
    failure_cost: float = 0.0
    restart_steps: int = 1
    # Read by the benchmark's time-in-band KPI, not by the reward: the
    # analysers' scatter.
    e_floor_top: float = 4e-4  # mole fraction
    e_floor_bottom: float = 4e-4


#: Units of the DistillationParams fields, for the generated environment pages.
PARAM_UNITS = {
    "alpha": "-",
    "M_tray": "kmol",
    "M_reboiler": "kmol",
    "M_condenser": "kmol",
    "tau_L": "min",
    "L_nominal": "kmol/min",
    "F": "kmol/min",
    "zF_nominal": "-",
    "qF": "-",
    "feed_step": "-",
    "zF_step": "-",
    "step_window": "min",
    "L_min": "kmol/min",
    "L_max": "kmol/min",
    "V_min": "kmol/min",
    "V_max": "kmol/min",
    "D_min": "kmol/min",
    "D_max": "kmol/min",
    "yD_spec": "-",
    "xB_spec": "-",
    "precision_floor": "-",
    "boilup_cost_weight": "-",
    "delta_t": "min",
    "time_unit_seconds": "s",
    "reward_version": "-",
    "boilup_price": "kmol",
    "reprocess_price": "kmol/kmol",
    "failure_cost": "kmol",
    "restart_steps": "steps",
    "e_floor_top": "-",
    "e_floor_bottom": "-",
}


def steady_liquid(L, params: DistillationParams):
    """Liquid leaving each tray, bottom tray first, at a steady reflux ``L``."""
    return jnp.where(TRAYS > N_FEED, L, L + params.qF * params.F)


def tray_holdup(liquid, params: DistillationParams):
    """Tray holdups from their liquid flows, Skogestad's eq. 47 with lambda = 0."""
    nominal = steady_liquid(params.L_nominal, params)
    return params.M_tray + params.tau_L * (liquid - nominal)


def _nominal_liquid():
    p = DistillationParams()
    return steady_liquid(p.L_nominal, p)


@struct.dataclass
class DistillationState(EnvState):
    x: jnp.ndarray  # (N_STAGES,) liquid composition -- interior stages HIDDEN
    zF: float  # feed composition, the unmeasured disturbance (HIDDEN)

    L: float  # reflux (manipulated)
    V: float  # boilup (manipulated)
    target_yD: float  # the product specs, constant (the zones' edges)
    target_xB: float

    actuator_held: jax.Array = actuator_field(2)
    #: Feed rate now, kmol/min (measured, FC 1 [D4]).
    F: float = struct.field(default=1.0, kw_only=True)
    #: The episode's two feed steps (HIDDEN until they come): the minute each
    #: comes, and its sign. A state built by hand has none.
    feed_step_at: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(10**9, jnp.int32), kw_only=True
    )
    feed_sign: jax.Array = struct.field(default=1.0, kw_only=True)
    zF_step_at: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(10**9, jnp.int32), kw_only=True
    )
    zF_sign: jax.Array = struct.field(default=1.0, kw_only=True)
    # Liquid leaving each tray, ordered as ``steady_liquid`` (HIDDEN).
    liquid: jax.Array = struct.field(default_factory=_nominal_liquid, kw_only=True)


def vle(x, params: DistillationParams):
    """Vapour composition in equilibrium with liquid ``x``."""
    return params.alpha * x / (1.0 + (params.alpha - 1.0) * x)


def flows(L, V, params: DistillationParams, F=None):
    """Product flows from the overall balance, kept physical by clipping D."""
    F = params.F if F is None else F
    D = jnp.clip(V - L, params.D_min, params.D_max)
    B = F - D
    return D, B


def compute_velocity(position, action, zF, params: DistillationParams, F=None):
    """Stage balances and tray hydraulics, as Skogestad's ``colamod.m``.

    ``position`` is the 41 compositions, then the 39 tray flows in the order of
    ``steady_liquid``. ``action`` is ``(L, V)`` in physical units, ``F`` the
    feed rate now (``params.F`` by default).
    """
    p = params
    F = p.F if F is None else F
    x, liquid = position[:N_STAGES], position[N_STAGES:]
    L, V = action
    D, _ = flows(L, V, p, F)
    # The reboiler's level loop passes on what arrives, so bottoms follows the
    # cascade, within the range ``flows`` keeps D in.
    B = jnp.clip(liquid[0] - V, F - p.D_max, F - p.D_min)
    y = vle(x, p)

    # Interior trays: liquid down from above, vapour up from below, and the
    # feed into the liquid on stage N_FEED. Vapour is the same throughout.
    liquid_in = jnp.append(liquid[1:], L)
    feed = TRAYS == N_FEED
    dM = liquid_in - liquid + feed * p.qF * F
    dMx = (
        liquid_in * x[2:] + V * y[:-2] - liquid * x[1:-1] - V * y[1:-1] + feed * F * zF
    )
    dx_trays = (dMx - x[1:-1] * dM) / tray_holdup(liquid, p)

    dx_reboiler = (liquid[0] * x[1] - V * y[0] - B * x[0]) / p.M_reboiler
    dx_condenser = (V * y[-2] - (L + D) * x[-1]) / p.M_condenser

    velocity = [jnp.array([dx_reboiler]), dx_trays, jnp.array([dx_condenser])]
    return jnp.concatenate(velocity + [dM / p.tau_L]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: jnp.ndarray,
    state: DistillationState,
    params: DistillationParams,
    key: jax.Array,
    integration_method: str = "rk4_16",
):
    """``action_raw`` is ``[L_raw, V_raw]`` in [-1, 1].

    The default 16 substeps are a stability requirement, not a refinement. The
    stiffest composition mode is |lambda| = 32.5 /min at the nominal point and
    37.6 /min across the flow box, and the tray flows lag at 1 / tau_L = 15.9
    /min, inside that. RK4 is stable only for h*|lambda| < 2.78, and 16 substeps
    give 2.35 at worst. Fewer substeps do not merely lose accuracy -- the
    profile saturates to 0/1 within a few steps.
    """
    p = params
    L = convert_raw_action_to_range(
        action_raw[0], min_action=p.L_min, max_action=p.L_max
    )
    V = convert_raw_action_to_range(
        action_raw[1], min_action=p.V_min, max_action=p.V_max
    )

    # The feed's two steps [D1, Fig. 18], on the episode's schedule.
    t_next = state.time + 1
    F = p.F * (1.0 + p.feed_step * state.feed_sign * (t_next > state.feed_step_at))
    zF = p.zF_nominal + p.zF_step * state.zF_sign * (t_next > state.zF_step_at)

    _compute_velocity = partial(
        compute_velocity, action=(L, V), zF=zF, params=params, F=F
    )
    new_position, _ = integrate_dynamics(
        positions=jnp.concatenate([state.x, state.liquid]),
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    # Compositions are mole fractions; clip guards the integrator, not the
    # physics (a converged solution never leaves [0, 1]).
    new_x = jnp.clip(new_position[:N_STAGES], 0.0, 1.0)
    liquid = new_position[N_STAGES:]

    return (
        state.replace(
            x=new_x, liquid=liquid, zF=zF, F=F, L=L, V=V, time=state.time + 1
        ),
        None,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def tray_temperature(state: DistillationState):
    """The stage-15 tray sensor's reading, K (``TEMPERATURE_STAGE``)."""
    x = state.x[TEMPERATURE_STAGE - 1]
    return T_BOIL_LIGHT * x + T_BOIL_HEAVY * (1.0 - x)


def get_obs(state: DistillationState, params: DistillationParams):
    """``[yD, xB, L_pct, V_pct, spec_yD, spec_xB, T_15, F]``.

    The two product analysers [D4, Fig. 1; D5], one tray temperature [D5], the
    operator's own reflux and boilup, the product specs, and the feed flow,
    which the plant meters (FC 1 [D4, p. 5]). The interior profile, which is
    the column's memory, is hidden, as is the feed composition: no feed
    analyser was found.
    """
    L_pct = (state.L - params.L_min) / (params.L_max - params.L_min)
    V_pct = (state.V - params.V_min) / (params.V_max - params.V_min)
    return jnp.array(
        [
            state.x[-1],  # yD, distillate
            state.x[0],  # xB, bottoms
            L_pct,
            V_pct,
            state.target_yD,
            state.target_xB,
            tray_temperature(state),
            state.F,
        ]
    )


def check_is_terminal(state: DistillationState, params: DistillationParams, xp=jnp):
    """No trip: the product limits are zones [D4, p. 5, Table 1], and flooding,
    the column's hydraulic limit [D6], has no sourced kind (REALITY.md)."""
    terminated = xp.zeros((), dtype=bool)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: DistillationState, params: DistillationParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0, in kmol of boilup per step
    (``DistillationParams``, "Score").

    - ``boilup``: the reboiler's duty [D4, p. 5];
    - ``off_spec``: each product outside its zone, run through again.

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    D, B = flows(state.L, state.V, p, state.F)
    off_top = xp.where(state.x[-1] < p.yD_spec, D, 0.0)
    off_bottom = xp.where(state.x[0] > p.xB_spec, B, 0.0)
    terms = {
        "boilup": R.priced_cost(state.V * p.delta_t, p.boilup_price, xp),
        "off_spec": R.priced_cost(
            (off_top + off_bottom) * p.delta_t, p.reprocess_price, xp
        ),
    }
    return R.with_trip(terms, tripped, R.trip_cost(p), xp)


def compute_reward_v1(state: DistillationState, params: DistillationParams, xp=jnp):
    """Both product purities tracked, minus reboiler duty.

    The two terms are *multiplied* rather than summed: hitting one spec while
    losing the other is not half a success, it is an off-spec column.
    """
    err_top = xp.abs(state.target_yD - state.x[-1])
    err_bot = xp.abs(state.target_xB - state.x[0])
    # Mole fractions live on [0, 1], so that is the envelope.
    top = log_scaled_reward(err_top, params.precision_floor, 1.0, xp)
    bottom = log_scaled_reward(err_bot, params.precision_floor, 1.0, xp)
    boilup = (state.V - params.V_min) / (params.V_max - params.V_min)
    return top * bottom * (1.0 - params.boilup_cost_weight * boilup)


def compute_reward(state: DistillationState, params: DistillationParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )


def separation_factor(state: DistillationState):
    """S = (yD/(1-yD)) / (xB/(1-xB)) -- the column's overall separation."""
    yD, xB = state.x[-1], state.x[0]
    return (yD / (1.0 - yD + 1e-12)) / (xB / (1.0 - xB + 1e-12) + 1e-12)
