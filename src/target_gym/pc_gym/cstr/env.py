from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

#: Salt for the load path's key (``step_load``).
_LOADS = 0x4C4F4144


@struct.dataclass
class CSTRParams(EnvParams):
    q: float = 100.0
    V: float = 100.0
    rho: float = 1000.0
    C: float = 0.239
    deltaHr: float = -5e4
    EA_over_R: float = 8750.0
    k0: float = 7.2e10
    UA: float = 5e4
    Ti: float = 350.0
    Caf: float = 1.0

    T_c_max: float = 302.0
    T_c_min: float = 295.0

    T_max: float = 350.0
    T_min: float = 300.0
    C_a_min: float = 0.7
    # The composition analyser's scatter. Below this the reward would be paying
    # for differences an online analyser cannot report.
    precision_floor: float = 5e-4  # mol/L
    C_a_max: float = 1.0

    # Every target holds under every load with 0.5 K of coolant to spare at each
    # stop (PHYSICS.md, section 5). (0.84, 0.91) left none at either end.
    target_CA_range: Tuple[float, float] = (0.87, 0.90)
    initial_CA_range: Tuple[float, float] = (0.8, 0.85)
    initial_T: float = 330.0
    delta_t: float = 0.25
    #: ``delta_t`` is in minutes (the PC-gym model's unit); seconds per unit.
    time_unit_seconds: float = 60.0
    # Over twice the SIMC cascade's 67-step settle (v3; 100 in v2).
    max_steps_in_episode: int = 150

    # ---- Unmeasured feed loads (``step_load``; figures ours, PHYSICS.md) ----
    #: Limits of each draw. The largest round pair that keeps the ignition fold
    #: above ``T_c_max`` at every corner, by 0.25 K.
    Ti_load: float = 1.0
    Caf_load: float = 0.02
    #: First arrival, the fixed settle of ``eval.FIXED_SETTLE``, so the whole
    #: path is scored, and the steps between arrivals (10 min).
    load_start: int = 67
    load_period: int = 40
    #: 0 switches the loads off, and the planner's copy holds the one it read.
    load_scale: float = 1.0

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # Measured with no load on the plant, before PR 9 re-measures the hold:
    # the shipped MPC held the concentration to 1e-6 mol/L after settling
    # (`scripts/measure_hold.py`), so the normalisation uses a documented
    # minimum, the online analyser's scatter. The span costs
    # (0.3 / 5e-4)^2 = 3.6e5 per step; a trip, which no state reaches, twice that.
    reward_version: int = 2
    e_floor: float = 5e-4  # mol/L, composition analyser scatter
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    failure_cost: float = 7.2e5
    #: Restart time priced into a trip (``reward.trip_cost``; 1 h at 15 s steps: cool, purge and re-feed, provisional; never charged).
    restart_steps: int = 240
    #: The oracle's best hold. Zero from the plant with no load, where exact
    #: hold is achievable and ``e_floor`` is a scale, not a floor; PR 9 redoes it.
    rho_floor_tracking: float = 0.0
    rho_floor: float = 0.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True


#: Units of the CSTRParams fields, for the generated environment pages.
PARAM_UNITS = {
    "q": "L/min",
    "V": "L",
    "rho": "g/L",
    "C": "J/(g*K)",
    "deltaHr": "J/mol",
    "EA_over_R": "K",
    "k0": "1/min",
    "UA": "J/(min*K)",
    "Ti": "K",
    "Caf": "mol/L",
    "T_c_max": "K",
    "T_c_min": "K",
    "T_max": "K",
    "T_min": "K",
    "C_a_min": "mol/L",
    "precision_floor": "mol/L",
    "C_a_max": "mol/L",
    "initial_T": "K",
    "delta_t": "min",
    "time_unit_seconds": "s",
    "Ti_load": "K",
    "Caf_load": "mol/L",
    "load_start": "steps",
    "load_period": "steps",
    "load_scale": "-",
    "e_floor": "mol/L",
    "reward_version": "-",
    "e_tol": "mol/L",
    "tracking_exponent": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
}


@struct.dataclass
class CSTRState(EnvState):
    C_a: float
    T: float
    target_CA: float

    # For rendering
    T_c: float

    actuator_held: jax.Array = actuator_field(1)
    #: The ``[Ti, Caf]`` load over the last step (``step_load``), hidden.
    load: jax.Array = struct.field(default_factory=lambda: jnp.zeros(2), kw_only=True)


def step_load(state: CSTRState, params: CSTRParams):
    """The feed's load over the step from ``state``, in K and mol/L.

    None until ``load_start``, then a uniform draw held for ``load_period``
    steps. It comes from the episode's ``sensor_key`` and the clock, so no
    action or step key moves it and a trip's restart keeps it: every policy on
    one episode key meets the same loads. At ``load_scale`` 0 the state's load
    stands, so the planner's copy (``plan_params``) keeps the one the oracle read.
    """
    block = jnp.maximum((state.time - params.load_start) // params.load_period, 0)
    key = jax.random.fold_in(jax.random.fold_in(state.sensor_key, _LOADS), block)
    draw = jax.random.uniform(key, (2,), state.load.dtype, -1.0, 1.0)
    limit = params.load_scale * jnp.array([params.Ti_load, params.Caf_load])
    live = jnp.where(state.time >= params.load_start, limit * draw, 0.0)
    return jnp.where(params.load_scale > 0, live, state.load)


def with_load(params: CSTRParams, load):
    """``params`` with the feed moved by ``load``."""
    return params.replace(Ti=params.Ti + load[0], Caf=params.Caf + load[1])


def compute_velocity(position, action, params: CSTRParams):
    T_c = action
    C_a, T = position

    rA = params.k0 * jnp.exp(-params.EA_over_R / T) * C_a

    velocity_C_a = params.q / params.V * (params.Caf - C_a) - rA
    velocity_T = (
        params.q / params.V * (params.Ti - T)
        + ((-params.deltaHr) * rA) * (1 / (params.rho * params.C))
        + params.UA * (T_c - T) * (1 / (params.rho * params.C * params.V))
    )

    return jnp.array([velocity_C_a, velocity_T]), None


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    T_c_raw: float,
    state: CSTRState,
    params: CSTRParams,
    integration_method: str = "rk4_1",
):
    """
    T_c_raw : Cooling temperature raw input
    """
    dt = params.delta_t
    T_c = convert_raw_action_to_range(
        T_c_raw, min_action=params.T_c_min, max_action=params.T_c_max
    )
    load = step_load(state, params)
    _compute_velocity = partial(
        compute_velocity, action=T_c, params=with_load(params, load)
    )

    (C_a, T), metrics = integrate_dynamics(
        positions=jnp.array([state.C_a, state.T]),
        delta_t=dt,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    return (
        state.replace(C_a=C_a, T=T, T_c=T_c, load=load, time=state.time + 1),
        metrics,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(
    state: CSTRState,
    params: CSTRParams,
):
    return jnp.array([state.C_a, state.T, state.target_CA])


def check_is_terminal(state: CSTRState, params: CSTRParams, xp=jnp):
    # No trip: T_c_max sits below the ignition fold at every load (PHYSICS.md,
    # section 2), so the reactor stays on its low branch and the 300 / 350 K and
    # 0.7 / C_a_max limits are unreachable. Kept as documentation of the envelope.
    terminated = jnp.zeros((), dtype=bool)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: CSTRState, params: CSTRParams, xp=jnp, tripped=None):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    if tripped is None:
        tripped, _ = check_is_terminal(state, params, xp)
    terms = {
        "tracking": R.tracking_cost(
            state.target_CA - state.C_a,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
    }
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


def compute_reward_v1(state: CSTRState, params: CSTRParams, xp=jnp):
    # Log-scaled: every halving of the concentration error is worth the same,
    # down to what the analyser can resolve. The previous form divided the error
    # by the whole 0.3 mol/L envelope and squared it, which is nearly flat over
    # any error a working controller produces -- checklist check 1.
    return log_scaled_reward(
        xp.abs(state.target_CA - state.C_a),
        params.precision_floor,
        params.C_a_max - params.C_a_min,
        xp,
    )


def compute_reward(state: CSTRState, params: CSTRParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
