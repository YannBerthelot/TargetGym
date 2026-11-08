from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

from target_gym import reward as R
from target_gym.base import EnvParams, EnvState, actuator_field
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward


@struct.dataclass
class FourTankParams(EnvParams):
    # Johansson's quadruple-tank rig at Lund, its non-minimum phase setting P+
    # [S1, p. 2046], in SI (REALITY.md, "The plant"; PHYSICS.md section 2).
    g: float = 9.81  # m/s^2 [S1: 981 cm/s^2]
    # Valve split, set by hand before a run and not moved during it [S3, p. 2].
    gamma1: float = 0.43
    gamma2: float = 0.34
    # Pump gains, P+ (m^3/s per V) [S1: 3.14, 3.29 cm^3/(V s)]
    k1: float = 3.14e-6
    k2: float = 3.29e-6
    # Outlet holes (m^2) [S1: 0.071, 0.057 cm^2]
    a1: float = 0.071e-4
    a2: float = 0.057e-4
    a3: float = 0.071e-4
    a4: float = 0.057e-4
    # Tank cross-sections (m^2) [S1: 28, 32 cm^2]
    A1: float = 28e-4
    A2: float = 32e-4
    A3: float = 28e-4
    A4: float = 32e-4

    # Pump voltage (V). The Lund maximum is not found; ours, 10 V, which puts
    # the 3.15 V operating point at a third of the range (KTH runs at half of
    # 15 V [S4, pp. 5-6]).
    v_min: float = 0.0
    v_max: float = 10.0
    #: The P+ operating point, both pumps [S1, p. 2046].
    v_nominal: float = 3.15

    # Levels. The floor is an empty tank, h >= 0 [S2, p. 121], a physical
    # limit, not a trip. The tank is full at 25 cm on the KTH rig [S4, p. 15]
    # (the Lund height is not found); past it the water overflows back to the
    # reservoir. No overflow trip or interlock was found, so neither is one.
    h_max: float = 0.25  # m
    precision_floor: float = 1e-3  # m, ours: the level reading's step

    # ---- Episode (charter rule 17) ----
    # Established operation: the rig at its P+ steady state, setpoints on it,
    # then setpoint steps of about 5 percentage points under automatic control
    # [S4, p. 9; S1, Figs. 5-6], here 1.25 cm of a 25 cm tank. Ours: three
    # steps 1000 s apart, the NMP response's settling scale [S1, Fig. 6], the
    # first away from the point, the second back, the third on the other tank.
    setpoint_step: float = 0.0125  # m
    setpoint_times: Tuple[int, int, int] = (300, 1300, 2300)  # steps
    # Disturbances the rigs use [S4, Exercise 4.1.1, p. 9; S5, pp. 1-2]. Sizes
    # and timing not found; ours:
    #: an extra outlet opened in one upper tank, as a fraction of its own
    #: outlet's area, for ``leak_duration`` steps at a random time;
    leak_fraction: float = 0.3
    leak_duration: int = 300
    #: a cup of water poured into one lower tank at a random time (m^3).
    cup_volume: float = 1.0e-4

    delta_t: float = 1.0
    max_steps_in_episode: int = 3300

    # ---- Reward (charter rule 15) ----
    # The rig's job is tracking: hold the two lower levels at their setpoints
    # [S1, p. 2046]. No band was found, so the cost is quadratic in the error,
    # one term per tank, on the level reading's 1 mm step (ours).
    reward_version: int = 3
    e_floor: float = 1e-3  # m
    e_tol: float = 0.0
    tracking_exponent: float = 2.0
    # The rig cannot trip (no interlock found), so this is never charged; it is
    # sized above any reachable tracking cost, both errors at a full tank.
    failure_cost: float = 2.0 * (0.25 / 1e-3) ** 2
    restart_steps: int = 1
    #: The oracle's best hold: not measured on this rig yet.
    rho_floor_tracking: float = 0.0
    rho_floor: float = 0.0
    #: True where e_floor is a resolution, not a measured or certified floor.
    floor_is_documented_minimum: bool = True


#: Units of the FourTankParams fields, for the generated environment pages.
PARAM_UNITS = {
    "g": "m/s^2",
    "gamma1": "-",
    "gamma2": "-",
    "k1": "m^3/(V*s)",
    "k2": "m^3/(V*s)",
    "a1": "m^2",
    "a2": "m^2",
    "a3": "m^2",
    "a4": "m^2",
    "A1": "m^2",
    "A2": "m^2",
    "A3": "m^2",
    "A4": "m^2",
    "v_min": "V",
    "v_max": "V",
    "v_nominal": "V",
    "h_max": "m",
    "precision_floor": "m",
    "setpoint_step": "m",
    "setpoint_times": "steps",
    "leak_fraction": "-",
    "leak_duration": "steps",
    "cup_volume": "m^3",
    "e_floor": "m",
    "reward_version": "-",
    "e_tol": "m",
    "tracking_exponent": "-",
    "failure_cost": "reward",
    "restart_steps": "steps",
    "rho_floor_tracking": "reward",
    "rho_floor": "reward",
}


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

    #: The episode's schedule, drawn at reset (hidden): the setpoint moves as
    #: ``(tank, sign)`` per entry of ``setpoint_times``, the start step and
    #: upper tank (0 for tank 3, 1 for tank 4) of the extra outlet, and the
    #: step and lower tank of the cup.
    #: A state built by hand has no moves, no outlet and no cup in its episode.
    sp_tank: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((), jnp.int32), kw_only=True
    )
    sp_sign: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros(3), kw_only=True
    )
    leak_start: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(-(10**9), jnp.int32), kw_only=True
    )
    leak_tank: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((), jnp.int32), kw_only=True
    )
    cup_time: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(-1, jnp.int32), kw_only=True
    )
    cup_tank: jax.Array = struct.field(
        default_factory=lambda: jnp.zeros((), jnp.int32), kw_only=True
    )
    actuator_held: jax.Array = actuator_field(2)


def steady_levels(volts, params: FourTankParams):
    """The four levels both pumps at ``volts`` hold, from the mass balances
    [S1, eq. (1)]: each outflow a_i sqrt(2 g h_i) equals its inflow."""
    p = params
    q3 = (1.0 - p.gamma2) * p.k2 * volts
    q4 = (1.0 - p.gamma1) * p.k1 * volts
    q1 = p.gamma1 * p.k1 * volts + q3
    q2 = p.gamma2 * p.k2 * volts + q4
    level = lambda q, a: (q / a) ** 2 / (2.0 * p.g)  # noqa: E731
    return level(q1, p.a1), level(q2, p.a2), level(q3, p.a3), level(q4, p.a4)


def setpoint_ranges(params: FourTankParams):
    """The setpoints an episode can ask for: the operating point's lower
    levels, one ``setpoint_step`` either side."""
    h1, h2, _, _ = steady_levels(params.v_nominal, params)
    s = params.setpoint_step
    return (h1 - s, h1 + s), (h2 - s, h2 + s)


def setpoint_move(state: FourTankState, params: FourTankParams, time):
    """The move the schedule makes to the two setpoints at step ``time``: the
    first away from the point, the second back, the third on the other tank."""
    times = jnp.asarray(params.setpoint_times)
    step = params.setpoint_step * state.sp_sign
    first = jnp.where(time == times[0], step[0], 0.0) - jnp.where(
        time == times[1], step[0], 0.0
    )
    third = jnp.where(time == times[2], step[2], 0.0)
    delta = jnp.zeros(2)
    delta = delta.at[state.sp_tank].add(first)
    return delta.at[1 - state.sp_tank].add(third)


def compute_velocity(
    position, action, params: FourTankParams, load=(0.0, 0.0), leak=(0.0, 0.0)
):
    """The four mass balances [S1, eq. (1)]. ``load`` is an unmeasured flow
    into tanks 1, 2 (the cup), ``leak`` the area of an extra outlet open in
    tanks 3, 4 [S4, Exercise 4.1.1]."""
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
        + load[0] / params.A1
    )
    dh2dt = (
        -(params.a2 / params.A2) * jnp.sqrt(2 * params.g) * sqrt_h2
        + (params.a4 / params.A2) * jnp.sqrt(2 * params.g) * sqrt_h4
        + (params.gamma2 * params.k2 / params.A2) * v2
        + load[1] / params.A2
    )
    dh3dt = (
        -((params.a3 + leak[0]) / params.A3) * jnp.sqrt(2 * params.g) * sqrt_h3
        + ((1 - params.gamma2) * params.k2 / params.A3) * v2
    )
    dh4dt = (
        -((params.a4 + leak[1]) / params.A4) * jnp.sqrt(2 * params.g) * sqrt_h4
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

    t = state.time
    cup = jnp.where(
        t == state.cup_time, params.cup_volume / params.delta_t, 0.0
    ) * jnp.array([state.cup_tank == 0, state.cup_tank == 1])
    open_ = (t >= state.leak_start) & (t < state.leak_start + params.leak_duration)
    leak = jnp.where(open_, params.leak_fraction, 0.0) * jnp.array(
        [
            (state.leak_tank == 0) * params.a3,
            (state.leak_tank == 1) * params.a4,
        ]
    )
    _compute_velocity = partial(
        compute_velocity, action=action, params=params, load=cup, leak=leak
    )
    (h1, h2, h3, h4), metrics = integrate_dynamics(
        positions=jnp.array([state.h1, state.h2, state.h3, state.h4]),
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )
    # A full tank overflows back to the reservoir [S4, p. 15]: the level holds
    # at the rim. An empty tank stays empty.
    h1, h2, h3, h4 = (jnp.clip(h, 0.0, params.h_max) for h in (h1, h2, h3, h4))
    target = jnp.stack([state.target_h1, state.target_h2]) + setpoint_move(
        state, params, t + 1
    )
    return (
        state.replace(
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            v1=v1,
            v2=v2,
            target_h1=target[0],
            target_h2=target[1],
            time=state.time + 1,
        ),
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
    """No trip: the rig has none [S2, p. 121; S4, p. 15]. An empty tank is a
    nonlinearity and a full one overflows, both inside the dynamics."""
    terminated = jnp.zeros((), dtype=bool)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(
    state: FourTankState, params: FourTankParams, xp=jnp, tripped=None
):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``).

    ``tripped`` overrides the trip check on ``state``, for a measured copy,
    whose noise must not fire or hide a trip (``base.measured_reward``).
    """
    if tripped is None:
        tripped, _ = check_is_terminal(state, params, xp)
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
    return R.with_trip(terms, tripped, R.trip_cost(params), xp)


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
    span = params.h_max
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
