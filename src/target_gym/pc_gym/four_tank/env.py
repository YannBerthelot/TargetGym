from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward


@struct.dataclass
class FourTankParams(EnvParams):
    # Gravitational acceleration (m/s²)
    g: float = 9.81
    # Pump split ratios (fraction of pump flow to lower tanks)
    gamma1: float = 0.2
    gamma2: float = 0.2
    # Pump gain constants (m³/s per Volt)
    k1: float = 0.00085
    k2: float = 0.00095
    # Outlet orifice areas (m²)
    a1: float = 0.0035
    a2: float = 0.003
    a3: float = 0.002
    a4: float = 0.0025
    # Tank cross-sectional areas (m²)
    A1: float = 1.0
    A2: float = 1.0
    A3: float = 1.0
    A4: float = 1.0

    # Pump voltage bounds (V)
    v_min: float = 0.0
    v_max: float = 10.0

    # Level bounds (m)
    h_min: float = 0.05
    precision_floor: float = 1e-3  # m, level transmitter resolution (1 mm)
    h_max: float = 1.5

    # Error scale for the MPC's tracking term, in metres. Not read by
    # ``compute_reward``: the reward normalises by the full span under a log,
    # which is what fixed the flat-reward problem this was introduced for. It
    # survives because the planner's quadratic surrogate needs a scale, and 5 cm
    # is a real miss on a plant whose setpoints live between 0.10 and 0.30 m.
    # See D1 in PHYSICS.md for why a band the reward ignores is worth watching.
    tracking_band: float = 0.05

    # Target level ranges for tanks 1 and 2.
    #
    # These MUST sit inside the plant's reachable envelope. With both pumps
    # saturated at v_max the steady state tops out at h1 = 0.360, h2 = 0.429,
    # so the previous (0.5, 1.0) band sat entirely above it and *no* sampled
    # target was attainable -- every episode was unwinnable, and the shared
    # effectiveness contract could not see it because the PID still beat every
    # constant action while both sat far from setpoint.
    #
    # The band below is jointly reachable: because the pumps are cross-coupled,
    # h1 and h2 are sampled independently but must be attainable *together*,
    # and every pair in this box is.
    #
    # It is also sized against the UPPER tanks, which is the subtler
    # constraint. A high h1 with a low h2 is held by a low v1 and a high v2 --
    # and v1 is what feeds tank 4. At the (0.25, 0.12) corner of a wider box
    # the steady h4 sits only 16 mm above the low-level trip, so any transient
    # dip ends the episode; one seed in twenty tripped there regardless of
    # gains. This box keeps at least 50 mm of margin on h4 at every corner.
    target_h1_range: Tuple[float, float] = (0.11, 0.19)
    target_h2_range: Tuple[float, float] = (0.14, 0.28)

    # Initial level ranges. Chosen to start inside the same envelope: the old
    # h1 upper bound of 0.4 was above the maximum sustainable level, so an
    # episode could begin at a level the plant can never hold.
    initial_h1_range: Tuple[float, float] = (0.11, 0.19)
    initial_h2_range: Tuple[float, float] = (0.14, 0.26)
    initial_h3_range: Tuple[float, float] = (0.25, 0.45)
    # Raised from 0.09: the target box is sized to keep 50 mm of margin on h4
    # above the 0.05 m trip, and starting at 0.09 gave only 40, so an episode
    # could begin closer to the low-level trip than any commanded steady state
    # ever gets.
    initial_h4_range: Tuple[float, float] = (0.10, 0.16)

    delta_t: float = 1.0
    max_steps_in_episode: int = 500

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    # No disturbance and fixed targets: the shipped MPC holds both levels to
    # 1e-5 m after settling (`scripts/measure_hold.py`), so the floor is
    # effectively zero and the normalisation uses the level transmitter's
    # 1 mm resolution. One term per tank, summed. The span costs
    # (1.45 / 1e-3)^2 = 2.1e6 per tank per step; overflow or dry-out twice the
    # two-tank maximum.
    reward_version: int = 2
    e_floor: float = 1e-3  # m, level transmitter resolution, both tanks
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    failure_cost: float = 8.4e6
    #: Steps the plant is down after a trip before it restarts (10 min at 1 s steps: refill after an overflow or dry-out, provisional).
    restart_steps: int = 600
    #: Tracking cost per step at the floor, in the reward's units; the NEA floor.
    rho_floor_tracking: float = 2.0
    rho_floor: float = 2.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True


@struct.dataclass
class FourTankState(EnvState):
    h1: float  # Level of tank 1 (controlled)
    h2: float  # Level of tank 2 (controlled)
    h3: float  # Level of tank 3 (upper, feeds tank 1)
    h4: float  # Level of tank 4 (upper, feeds tank 2)
    target_h1: float
    target_h2: float

    # For rendering
    v1: float
    v2: float
    #: Steps of downtime left after a trip (``base.failure_kernel``); 0 when healthy.
    downtime: int = 0


def compute_velocity(position, action, params: FourTankParams):
    h1, h2, h3, h4 = position[0], position[1], position[2], position[3]
    v1, v2 = action[0], action[1]

    sqrt_h1 = _safe_sqrt(h1)
    sqrt_h2 = _safe_sqrt(h2)
    sqrt_h3 = _safe_sqrt(h3)
    sqrt_h4 = _safe_sqrt(h4)

    dh1dt = (
        -(params.a1 / params.A1) * jnp.sqrt(2 * params.g) * sqrt_h1
        + (params.a3 / params.A1) * jnp.sqrt(2 * params.g) * sqrt_h3
        + (params.gamma1 * params.k1 / params.A1) * v1
    )
    dh2dt = (
        -(params.a2 / params.A2) * jnp.sqrt(2 * params.g) * sqrt_h2
        + (params.a4 / params.A2) * jnp.sqrt(2 * params.g) * sqrt_h4
        + (params.gamma2 * params.k2 / params.A2) * v2
    )
    dh3dt = (
        -(params.a3 / params.A3) * jnp.sqrt(2 * params.g) * sqrt_h3
        + ((1 - params.gamma2) * params.k2 / params.A3) * v2
    )
    dh4dt = (
        -(params.a4 / params.A4) * jnp.sqrt(2 * params.g) * sqrt_h4
        + ((1 - params.gamma1) * params.k1 / params.A4) * v1
    )

    return jnp.array([dh1dt, dh2dt, dh3dt, dh4dt]), None


def _safe_sqrt(x):
    """``sqrt(max(x, 0))`` with a finite derivative at zero.

    Torricelli outflow goes as the square root of the level, and a tank can sit
    empty. Written as ``jnp.sqrt(jnp.maximum(h, 0.0))`` the forward value is
    right, but the reverse-mode derivative is not: d(sqrt)/dx is infinite at
    zero, and the clamp contributes a zero, so the product is NaN. That NaN then
    poisons the whole gradient -- which is why gradient-based PID tuning on this
    plant returned NaN gains while its loss evaluated perfectly well.

    The nested ``where`` is the standard remedy: the inner one keeps the value
    passed to ``sqrt`` away from zero along the differentiated path, and the
    outer one restores the exact value. Forward results are unchanged.
    """
    positive = x > 0.0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, x, 1.0)), 0.0)


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    action_raw: jnp.ndarray,
    state: FourTankState,
    params: FourTankParams,
    integration_method: str = "rk4_1",
):
    v1 = convert_raw_action_to_range(action_raw[0], params.v_min, params.v_max)
    v2 = convert_raw_action_to_range(action_raw[1], params.v_min, params.v_max)
    action = jnp.array([v1, v2])

    _compute_velocity = partial(compute_velocity, action=action, params=params)
    (h1, h2, h3, h4), metrics = integrate_dynamics(
        positions=jnp.array([state.h1, state.h2, state.h3, state.h4]),
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    return (
        state.replace(h1=h1, h2=h2, h3=h3, h4=h4, v1=v1, v2=v2, time=state.time + 1),
        metrics,
    )


# Deliberately not jitted. It was decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, which keys the compilation
# cache on the params object: a fresh ``Params(...)`` -- what every sweep, tuner
# and MPC builds -- was a cache miss and a full recompile, measured at ~1600x the
# cost of a cached call. Callers that want it fused already jit ``step_env``,
# which traces this inline.
def get_obs(state: FourTankState, params: FourTankParams):
    return jnp.array(
        [state.h1, state.h2, state.h3, state.h4, state.target_h1, state.target_h2]
    )


def check_is_terminal(state: FourTankState, params: FourTankParams, xp=jnp):
    h_vals = jnp.array([state.h1, state.h2, state.h3, state.h4])
    terminated = jnp.any(jnp.logical_or(h_vals <= params.h_min, h_vals >= params.h_max))
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: FourTankState, params: FourTankParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    terminated, _ = check_is_terminal(state, params, xp)
    track = R.tracking_cost(
        state.target_h1 - state.h1,
        params.e_floor,
        params.e_tol,
        params.tracking_exponent,
        xp,
    ) + R.tracking_cost(
        state.target_h2 - state.h2,
        params.e_floor,
        params.e_tol,
        params.tracking_exponent,
        xp,
    )
    terms = {
        "tracking": track,
    }
    return R.with_downtime(
        terms, R.is_down(terminated, state, xp), params.failure_cost, xp
    )


def compute_reward_v1(state: FourTankState, params: FourTankParams, xp=jnp):
    """Mean of the two level-tracking scores.

    Tracking is log-scaled (``utils.log_scaled_reward``): every halving of the
    error is worth the same increment, from the plant's operating envelope down
    to ``precision_floor``, the finest error its instrument can resolve. Below
    that the reward stops paying, because further "improvement" is noise.

    This replaced a clipped squared band, which was flat -- exactly zero, no
    gradient -- for any error outside the band, and which stopped
    discriminating just where a good controller operates. See
    docs/reward-shaping.md.

    """
    span = params.h_max - params.h_min
    r1 = log_scaled_reward(
        xp.abs(state.target_h1 - state.h1), params.precision_floor, span, xp
    )
    r2 = log_scaled_reward(
        xp.abs(state.target_h2 - state.h2), params.precision_floor, span, xp
    )
    return (r1 + r2) / 2.0


def compute_reward(state: FourTankState, params: FourTankParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
