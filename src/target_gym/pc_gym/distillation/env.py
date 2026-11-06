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

# Ornstein-Uhlenbeck feed-composition disturbance: mean-reversion rate (1/min).
# 1/theta ~ 500 min, a few dominant time constants, so the feed drifts as a
# changing operating condition rather than as noise.
FEED_OU_THETA = 2.0e-3

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
    M_drum: float = 0.5  # kmol condenser / reboiler holdup
    tau_L: float = 0.063  # min, liquid lag per tray (Skogestad's taul)
    L_nominal: float = 2.706  # kmol/min, the reflux at which M_i = M_tray

    # ---- Feed ----
    F: float = 1.0  # kmol/min
    zF_nominal: float = 0.5  # mole fraction light component
    qF: float = 1.0  # saturated liquid
    zF_noise_std: float = 0.03  # OU stationary std
    zF_min: float = 0.40
    zF_max: float = 0.60

    # ---- Manipulated variables (LV configuration) ----
    # Wide enough to hold every target in the box at reset feeds within two sigma:
    # (0.995, 0.005) needs L 3.35 and V 3.85, (0.98, 0.02) L 2.27 and V 2.73.
    L_min: float = 2.20
    L_max: float = 3.40
    V_min: float = 2.70
    V_max: float = 3.90
    # Distillate is D = V - L; clipped so the overall balance stays physical at
    # the corners of the action box, where V - L could otherwise go negative.
    D_min: float = 0.10
    D_max: float = 0.90

    # ---- Operating / termination bounds ----
    yD_floor: float = 0.90  # distillate purity lost
    xB_ceiling: float = 0.10  # bottoms purity lost

    # ---- Reward shaping ----
    # Error scale for the MPC's tracking term, in mole fraction. Not read by
    # ``compute_reward``, which normalises by the full [0, 1] envelope under a
    # log; this is the planner's quadratic surrogate scale. 0.02 is about the
    # width this column is actually operated over. A band the reward ignores
    # and the controller steers by is worth watching: the glass furnace had one
    # inherited from a deleted reward and it cost 16% against its own PID.
    tracking_band: float = 0.02
    precision_floor: float = 1e-4  # mole fraction, online analyser resolution
    # Zeroed for the 0.6 line: this phase scores setpoint tracking alone.
    # A running cost is a real part of every one of these plants, but its
    # weight against tracking accuracy is a design decision this library has
    # not earned yet, and an arbitrary one turns a tracking benchmark into a
    # multi-objective problem whose Pareto point nobody chose. The glass
    # furnace showed the cost of getting it wrong: its MPC sat 6 K cold with
    # fuel at minimum 80% of the time, because a 0.1 fuel weight against a
    # quadratic tracking surrogate made that the optimum of the objective it
    # was given. The field and the term stay, so a weight can be restored
    # once there is a defensible way to set it. See docs/roadmap.md.
    boilup_cost_weight: float = 0.0  # was 0.05; reboiler duty is the running cost

    # ---- Targets ----
    target_yD_range: Tuple[float, float] = (0.980, 0.995)
    target_xB_range: Tuple[float, float] = (0.005, 0.020)
    initial_L_range: Tuple[float, float] = (2.65, 2.76)

    # ---- Time discretization ----
    # Minutes. Dominant time constant ~194 min, so 600 steps ~ 3 tau. The
    # integrator needs 16 substeps at this step size for stability; see
    # ``compute_next_state``.
    delta_t: float = 1.0
    #: ``delta_t`` is in minutes (Skogestad's model unit); seconds per unit.
    time_unit_seconds: float = 60.0
    max_steps_in_episode: int = 200

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Two tracked compositions, one term each, summed. The shipped MPC's
    # long-run hold errors under the shipped feed-composition disturbance
    # (`scripts/measure_hold.py`, 1200 hold steps after a 582-step burn-in)
    # are top 7.6e-7 and bottom 5.2e-7 mole fraction, each its lowest of three
    # seeds (PID down to 5.7e-5 and 1.3e-4): the current oracle,
    # planning on the mean feed with a decaying step. It held
    # 1.35e-5 and 3.3e-5 before. Upper bounds on the achievable floors.
    # e_tol = 0 provisionally: the product purity specifications a column is
    # run against come from the sales contract and are to be supplied. Boilup
    # above the hold-phase rate (3.282 kmol/min, PID; MPC 3.292) is charged at
    # weight 1. A trip is charged at twice the reachable 0.095 error on each
    # product, in floor units (`failure_cost`).
    reward_version: int = 2
    # The MPC holds 7.6e-7 / 5.2e-7 in the shipped disturbance, below the 1e-4
    # composition-analyser resolution: the instrument sets both scales.
    e_floor_top: float = 1e-4  # mole fraction
    e_floor_bottom: float = 1e-4
    e_tol: float = 0.0  # provisional; purity specs to be supplied
    tracking_exponent: float = 2.0
    c_hold: float = 3.282  # boilup while holding (PID)
    running_weight: float = 1.0
    failure_cost: float = (
        2.0 * 2.0 * (0.095 / 1e-4) ** 2
    )  # reachable 0.095 on each product (0.98 -> 0.90 trip; 0.02 -> 0.10)
    #: Restart time priced into a trip (``reward.trip_cost``; 4 h at 1 min steps: column shutdown and re-establishment of the profile, provisional).
    restart_steps: int = 240
    #: Tracking cost per step at the floor, in the reward's units.
    #: The oracle's best hold: the lowest per-seed hold cost the shipped MPC
    #: demonstrated, in the reward's units (the MPC's 7.64e-7 / 5.19e-7 holds in 1e-4 units).
    rho_floor_tracking: float = (7.64e-7 / 1e-4) ** 2 + (5.19e-7 / 1e-4) ** 2
    rho_floor: float = (7.64e-7 / 1e-4) ** 2 + (5.19e-7 / 1e-4) ** 2
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False


#: Units of the DistillationParams fields, for the generated environment pages.
PARAM_UNITS = {
    "alpha": "-",
    "M_tray": "kmol",
    "M_drum": "kmol",
    "tau_L": "min",
    "L_nominal": "kmol/min",
    "F": "kmol/min",
    "zF_nominal": "-",
    "qF": "-",
    "zF_noise_std": "-",
    "zF_min": "-",
    "zF_max": "-",
    "L_min": "kmol/min",
    "L_max": "kmol/min",
    "V_min": "kmol/min",
    "V_max": "kmol/min",
    "D_min": "kmol/min",
    "D_max": "kmol/min",
    "yD_floor": "-",
    "xB_ceiling": "-",
    "tracking_band": "-",
    "precision_floor": "-",
    "boilup_cost_weight": "-",
    "delta_t": "min",
    "time_unit_seconds": "s",
    "e_floor_top": "-",
    "e_floor_bottom": "-",
    "reward_version": "-",
    "e_tol": "-",
    "tracking_exponent": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
    "c_hold": "kmol/min",
    "running_weight": "-",
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
    target_yD: float
    target_xB: float

    actuator_held: jax.Array = actuator_field(2)
    # Liquid leaving each tray, ordered as ``steady_liquid`` (HIDDEN).
    liquid: jax.Array = struct.field(default_factory=_nominal_liquid, kw_only=True)


def vle(x, params: DistillationParams):
    """Vapour composition in equilibrium with liquid ``x``."""
    return params.alpha * x / (1.0 + (params.alpha - 1.0) * x)


def flows(L, V, params: DistillationParams):
    """Product flows from the overall balance, kept physical by clipping D."""
    D = jnp.clip(V - L, params.D_min, params.D_max)
    B = params.F - D
    return D, B


def compute_velocity(position, action, zF, params: DistillationParams):
    """Stage balances and tray hydraulics, as Skogestad's ``colamod.m``.

    ``position`` is the 41 compositions, then the 39 tray flows in the order of
    ``steady_liquid``. ``action`` is ``(L, V)`` in physical units.
    """
    p = params
    x, liquid = position[:N_STAGES], position[N_STAGES:]
    L, V = action
    D, _ = flows(L, V, p)
    # The reboiler's level loop passes on what arrives, so bottoms follows the
    # cascade, within the range ``flows`` keeps D in.
    B = jnp.clip(liquid[0] - V, p.F - p.D_max, p.F - p.D_min)
    y = vle(x, p)

    # Interior trays: liquid down from above, vapour up from below, and the
    # feed into the liquid on stage N_FEED. Vapour is the same throughout.
    liquid_in = jnp.append(liquid[1:], L)
    feed = TRAYS == N_FEED
    dM = liquid_in - liquid + feed * p.qF * p.F
    dMx = (
        liquid_in * x[2:]
        + V * y[:-2]
        - liquid * x[1:-1]
        - V * y[1:-1]
        + feed * p.F * zF
    )
    dx_trays = (dMx - x[1:-1] * dM) / tray_holdup(liquid, p)

    dx_reboiler = (liquid[0] * x[1] - V * y[0] - B * x[0]) / p.M_drum
    dx_condenser = (V * y[-2] - (L + D) * x[-1]) / p.M_drum

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

    # OU feed-composition disturbance, drawn from a key folded with
    # ``state.time`` so a caller passing a constant key still gets a genuine
    # zero-mean process.
    noise = jax.random.normal(jax.random.fold_in(key, state.time))
    sigma = p.zF_noise_std * jnp.sqrt(2.0 * FEED_OU_THETA * p.delta_t)
    zF = jnp.clip(
        state.zF
        + FEED_OU_THETA * (p.zF_nominal - state.zF) * p.delta_t
        + sigma * noise,
        p.zF_min,
        p.zF_max,
    )

    _compute_velocity = partial(compute_velocity, action=(L, V), zF=zF, params=params)
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
        state.replace(x=new_x, liquid=liquid, zF=zF, L=L, V=V, time=state.time + 1),
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
    """``[yD, xB, L_pct, V_pct, target_yD, target_xB, T_15]``.

    The two product compositions are analysed and one tray temperature is
    measured, as on a real column, which has analysers on the product
    streams and thermocouples on a few trays, not an assay of all 40. The
    interior profile, which is the column's memory, is otherwise hidden, as
    is the feed composition.
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
        ]
    )


def check_is_terminal(state: DistillationState, params: DistillationParams, xp=jnp):
    yD, xB = state.x[-1], state.x[0]
    terminated = xp.logical_or(yD <= params.yD_floor, xB >= params.xB_ceiling)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: DistillationState, params: DistillationParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    p = params
    if tripped is None:
        tripped, _ = check_is_terminal(state, p, xp)
    track = R.tracking_cost(
        state.target_yD - state.x[-1], p.e_floor_top, p.e_tol, p.tracking_exponent, xp
    ) + R.tracking_cost(
        state.target_xB - state.x[0], p.e_floor_bottom, p.e_tol, p.tracking_exponent, xp
    )
    terms = {
        "tracking": track,
        "running": R.running_cost(state.V, p.c_hold, p.running_weight, xp),
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
