"""
Binary distillation — Skogestad's "Column A".

See ``PHYSICS.md`` in this directory for provenance, the sourced parameter
table and validation targets. Method: ``docs/PHYSICS_METHODOLOGY.md``.

Model
-----
A 40-tray binary column with a total condenser and a reboiler, 41 equilibrium
stages in all, indexed 1 = reboiler ... 41 = condenser, feed on stage 21.
Constant molar flows, constant relative volatility, equilibrium on every
stage, no vapour holdup.

Per-stage component balance, with vapour-liquid equilibrium
``y = alpha x / (1 + (alpha-1) x)``::

    tray i :  M dx_i/dt = L_{i+1} x_{i+1} + V y_{i-1} - L_i x_i - V y_i
    reboiler: M dx_1/dt = L_2 x_2 - V y_1 - B x_1
    condenser:M dx_N/dt = V y_{N-1} - (L + D) x_N

Saturated liquid feed, so the liquid rate below the feed is ``L + F`` while the
vapour rate is constant throughout. ``D = V - L`` and ``B = L + F - V`` follow
from the overall balance.

Why this is a hard target MDP
-----------------------------
* **Severe input interaction.** At the nominal high-purity operating point the
  steady-state gain matrix has an RGA around 50 and a condition number around
  200: reflux and boilup push both compositions almost the same way, so the
  useful direction is a small difference between two large, nearly-cancelling
  effects. This is the textbook example of an ill-conditioned plant.
* **Slow and high-order.** 41 states with a dominant time constant of hours,
  and the composition profile inside the column is the memory.
* **Only the products are measured.** 39 of the 41 stage compositions are
  hidden -- a real column has analysers on the product streams, not on every
  tray.
* **Feed composition drifts and is unmeasured**, which moves the whole profile.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

N_STAGES = 41  # 40 trays + reboiler + condenser, minus shared indexing
N_FEED = 21  # feed stage, counting the reboiler as stage 1

# Ornstein-Uhlenbeck feed-composition disturbance: mean-reversion rate (1/min).
# 1/theta ~ 500 min, a few dominant time constants, so the feed drifts as a
# changing operating condition rather than as noise.
FEED_OU_THETA = 2.0e-3


@struct.dataclass
class DistillationParams(EnvParams):
    # ---- Column ----
    alpha: float = 1.5  # relative volatility
    M_tray: float = 0.5  # kmol tray holdup
    M_drum: float = 0.5  # kmol condenser / reboiler holdup

    # ---- Feed ----
    F: float = 1.0  # kmol/min
    zF_nominal: float = 0.5  # mole fraction light component
    qF: float = 1.0  # saturated liquid
    zF_noise_std: float = 0.03  # OU stationary std
    zF_min: float = 0.40
    zF_max: float = 0.60

    # ---- Manipulated variables (LV configuration) ----
    L_min: float = 2.30
    L_max: float = 3.10
    V_min: float = 2.80
    V_max: float = 3.60
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
    # Two tracked compositions, one term each, summed. Floors are the shipped
    # MPC's long-run hold errors under the shipped feed-composition
    # disturbance (`scripts/measure_hold.py`, 1200 hold steps after a
    # 582-step burn-in): top 1.35e-5, bottom 3.3e-5 mole fraction, each the MPC's lowest of three seeds (PID down to 5.7e-5
    # and 1.3e-4). Upper bounds on the achievable floors. e_tol = 0
    # provisionally: the product purity specifications a column is run
    # against come from the sales contract and are to be supplied. Boilup
    # above the hold-phase rate (3.282 kmol/min, PID; MPC 3.292) is charged
    # at weight 1. The unit composition span costs (1 / 1.35e-5)^2 = 5.5e9 per
    # step on the top alone; a trip twice the two-end sum.
    reward_version: int = 2
    # The MPC holds 1.35e-5 / 3.3e-5 in the shipped disturbance, below the 1e-4
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
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    #: The NEA reference: the lowest per-seed hold cost the shipped MPC
    #: demonstrated, in the reward's units (the MPC's 1.35e-5 / 3.3e-5 holds in 1e-4 units).
    rho_floor_tracking: float = (1.35e-5 / 1e-4) ** 2 + (3.3e-5 / 1e-4) ** 2
    rho_floor: float = (1.35e-5 / 1e-4) ** 2 + (3.3e-5 / 1e-4) ** 2
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = False


@struct.dataclass
class DistillationState(EnvState):
    x: jnp.ndarray  # (N_STAGES,) liquid composition -- interior stages HIDDEN
    zF: float  # feed composition, the unmeasured disturbance (HIDDEN)

    L: float  # reflux (manipulated)
    V: float  # boilup (manipulated)
    target_yD: float
    target_xB: float


def vle(x, params: DistillationParams):
    """Vapour composition in equilibrium with liquid ``x``."""
    return params.alpha * x / (1.0 + (params.alpha - 1.0) * x)


def flows(L, V, params: DistillationParams):
    """Product flows from the overall balance, kept physical by clipping D."""
    D = jnp.clip(V - L, params.D_min, params.D_max)
    B = params.F - D
    return D, B


def compute_velocity(position, action, zF, params: DistillationParams):
    """Stage-wise component balances. ``action`` is ``(L, V)`` in physical units."""
    p = params
    x = position
    L, V = action
    D, B = flows(L, V, p)
    y = vle(x, p)

    # Constant molar flows: a saturated-liquid feed adds F to the liquid below
    # the feed stage, while the vapour rate is the same throughout.
    stage = jnp.arange(1, N_STAGES + 1)
    L_stage = jnp.where(stage > N_FEED, L, L + p.qF * p.F)

    # Interior trays: liquid down from above, vapour up from below.
    liquid_in = L_stage[2:] * x[2:]
    vapour_in = V * y[:-2]
    liquid_out = L_stage[1:-1] * x[1:-1]
    vapour_out = V * y[1:-1]
    dx_trays = (liquid_in + vapour_in - liquid_out - vapour_out) / p.M_tray

    dx_reboiler = (L_stage[1] * x[1] - V * y[0] - B * x[0]) / p.M_drum
    dx_condenser = (V * y[-2] - (L + D) * x[-1]) / p.M_drum

    dx = jnp.concatenate(
        [jnp.array([dx_reboiler]), dx_trays, jnp.array([dx_condenser])]
    )
    # Feed enters the liquid on stage N_FEED.
    dx = dx.at[N_FEED - 1].add(p.F * zF / p.M_tray)
    return dx, None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: jnp.ndarray,
    state: DistillationState,
    params: DistillationParams,
    key: jax.Array,
    integration_method: str = "rk4_16",
):
    """``action_raw`` is ``[L_raw, V_raw]`` in [-1, 1].

    The default 16 substeps are a stability requirement, not a refinement.
    Tray holdup M = 0.5 against flows of ~3 gives a tray time constant of
    ~0.17 min, so the fastest eigenvalue is |lambda| ~ (L+V)/M ~ 11.8 /min.
    RK4 is stable only for h*|lambda| < 2.78, and at delta_t = 1 min that
    needs h <= 0.235 min. Fewer substeps do not merely lose accuracy -- the
    profile saturates to 0/1 within a single step.
    """
    p = params
    L = convert_raw_action_to_range(
        action_raw[0], min_action=p.L_min, max_action=p.L_max
    )
    V = convert_raw_action_to_range(
        action_raw[1], min_action=p.V_min, max_action=p.V_max
    )

    # OU feed-composition disturbance, drawn from a key folded with
    # ``state.time`` so a caller passing a constant key -- which every rollout
    # helper here does -- still gets a genuine zero-mean process.
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
    new_x, _ = integrate_dynamics(
        positions=state.x,
        delta_t=p.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    # Compositions are mole fractions; clip guards the integrator, not the
    # physics (a converged solution never leaves [0, 1]).
    new_x = jnp.clip(new_x, 0.0, 1.0)

    return (
        state.replace(x=new_x, zF=zF, L=L, V=V, time=state.time + 1),
        None,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: DistillationState, params: DistillationParams):
    """``[yD, xB, L_pct, V_pct, target_yD, target_xB]``.

    Only the two product compositions are measured -- a real column has
    analysers on the product streams, not on all 40 trays. The interior
    profile, which is the column's memory, is hidden, as is the feed
    composition.
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
        ]
    )


def check_is_terminal(state: DistillationState, params: DistillationParams, xp=jnp):
    yD, xB = state.x[-1], state.x[0]
    terminated = xp.logical_or(yD <= params.yD_floor, xB >= params.xB_ceiling)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: DistillationState, params: DistillationParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    p = params
    terminated, _ = check_is_terminal(state, p, xp)
    track = R.tracking_cost(
        state.target_yD - state.x[-1], p.e_floor_top, p.e_tol, p.tracking_exponent, xp
    ) + R.tracking_cost(
        state.target_xB - state.x[0], p.e_floor_bottom, p.e_tol, p.tracking_exponent, xp
    )
    terms = {
        "tracking": track,
        "running": R.running_cost(state.V, p.c_hold, p.running_weight, xp),
    }
    return R.with_trip(terms, terminated, R.trip_cost(p), xp)


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
