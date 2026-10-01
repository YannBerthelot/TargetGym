"""Unstable CSTR: cstr's reactor held on its open-loop unstable middle branch.

See ``PHYSICS.md`` in this directory for provenance, the parameter table and
the validation targets.

The two balances and the ten physical parameter values are the shipped
``cstr``'s, imported from ``target_gym.pc_gym.cstr.env`` and never retyped, so
an edit there moves this task too (its spec declares that file in
``fingerprint_sources``). This module adds a first-order jacket lag between
the coolant command and the jacket temperature the balances see, a drifting
feed temperature, a stratified schedule of six concentration targets on the
middle branch, where every target is a saddle, and a high-temperature trip.

State (three ODEs, one stochastic disturbance, the schedule)
------------------------------------------------------------
``C_a``            reactant concentration (mol/L), measured
``T``              reactor temperature (K), measured
``T_j``            jacket temperature (K), measured; lags the command by ``tau_j``
``Ti_dev``         feed-temperature drift (K), an Ornstein-Uhlenbeck process, hidden
``target_levels``  the six levels of the episode; only the live one is observed
``block_clock``    steps since the schedule started, hidden
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial as partial

import target_gym.pc_gym.cstr.env as cstr_env
from target_gym import reward as R
from target_gym.base import EnvParams, EnvState
from target_gym.integration import integrate_dynamics
from target_gym.utils import convert_raw_action_to_range, log_scaled_reward

#: Blocks per episode: the first level and five switches.
N_BLOCKS = 6

# The shipped cstr's defaults, read once. Every inherited value below is
# written ``_CSTR.<field>``.
_CSTR = cstr_env.CSTRParams()

# The step, in minutes (3 s). ``delta_t`` and ``restart_steps`` both use it.
_DT = 0.05


@struct.dataclass
class UnstableCSTRParams(EnvParams):
    # ---- Reactor: cstr's ten values (read) ----
    q: float = _CSTR.q  # L/min
    V: float = _CSTR.V  # L
    rho: float = _CSTR.rho  # g/L
    C: float = _CSTR.C  # J/(g K); deviation D1
    deltaHr: float = _CSTR.deltaHr  # J/mol
    EA_over_R: float = _CSTR.EA_over_R  # K
    k0: float = _CSTR.k0  # 1/min
    UA: float = _CSTR.UA  # J/(min K)
    Ti: float = _CSTR.Ti  # K, feed temperature before the drift
    Caf: float = _CSTR.Caf  # mol/L
    # Composition analyser resolution (read, cstr's). Version 1 only.
    precision_floor: float = _CSTR.precision_floor  # mol/L
    #: ``delta_t`` is in minutes, as in cstr; seconds per unit.
    time_unit_seconds: float = 60.0

    # ---- Jacket and interlocks ----
    # Lower bound read from PC-gym's CSTR action space; the upper bound is
    # ours, and spans both folds with at least 4.6 K headroom at |Ti_dev| 6 K
    # (derived).
    T_c_min: float = 290.0  # K
    T_c_max: float = 310.0  # K
    #: Jacket time constant. TUNED: not sourced, justified by PHYSICS.md's
    #: jacket-lag sensitivity table.
    tau_j: float = 0.1  # min (6 s)
    #: High-high interlock (ours), 4.49 K above the extinction fold and
    #: 12.17 K above the hottest target (derived).
    T_trip: float = 365.0  # K
    #: Validity guard (ours). Unreachable; it trips the non-physical proposals
    #: of the off-envelope sweep in conformance check 5.
    T_valid_min: float = 250.0  # K

    # ---- Feed-temperature drift (ours) ----
    Ti_sigma: float = 2.0  # K, stationary standard deviation
    Ti_tau: float = 10.0  # min, correlation time
    # Clip on the initial draw only, in standard deviations: with it every
    # reset is recoverable (PHYSICS.md, task design). The running drift is
    # not clipped.
    Ti_dev_reset_clip: float = 3.0

    # ---- Schedule and reset (ours) ----
    # The middle branch, 5.89 K from the ignition fold and 7.68 K from the
    # extinction fold (derived).
    target_CA_range: Tuple[float, float] = (0.45, 0.65)  # mol/L
    # A fixed multiset, permuted per episode, so every episode has the same
    # total squared move, 0.035 (mol/L)^2 (derived).
    switch_sizes: Tuple[float, ...] = (0.05, 0.05, 0.10, 0.10, 0.10)  # mol/L
    # 10 min, provisional: 3 x the settle after the worst switch of the
    # cascade on DEFAULT_GAINS (derived, scripts/unstable_cstr_numbers.py
    # --section settle).
    block_steps: int = 200
    initial_CA_offset: float = 0.01  # mol/L, reset box half-width
    initial_T_offset: float = 1.5  # K, reset box half-width
    delta_t: float = _DT  # min (3 s)
    max_steps_in_episode: int = 1200  # N_BLOCKS * block_steps, 60 min

    # ---- Reward (docs/reward-shaping.md; version 2) ----
    reward_version: int = 2
    # The analyser resolution, provisional. scripts/measure_hold.py records the
    # MPC's hold from minute 6 of each block to the switch, less the steps in
    # which it already moves toward the next level (target_gym.eval finds
    # them), and e_floor is raised only if that hold is coarser. The v2
    # ranking and the PID tuning do not depend on it: every cost term scales by
    # 1/e_floor^2.
    e_floor: float = 1e-4  # mol/L
    e_tol: float = 0.0  # no product specification supplied (provisional)
    tracking_exponent: float = 2.0
    #: The largest |target - C_a| an untripped state reaches (derived, closed
    #: form): max(Caf - 0.45, 0.65 - C_a at 365 K on the steady-state curve)
    #: = max(0.55, 0.386). The v1 envelope and the base of ``failure_cost``.
    CA_error_max: float = 0.55  # mol/L
    failure_cost: float = 2.0 * (0.55 / 1e-4) ** 2  # twice the largest tracking cost
    #: Restart time priced into a trip (``reward.trip_cost``): cstr's
    #: provisional 1 h restart (240 steps of 15 s) at 3 s steps.
    restart_steps: int = round(_CSTR.restart_steps * _CSTR.delta_t / _DT)
    #: The NEA reference, (e_hold_min / e_floor) ** 2. The MPC holds C_a to
    #: 6.29e-6 mol/L (measured, the lowest per-seed hold scripts/measure_hold.py
    #: recorded), under the analyser resolution, so e_floor stays at the
    #: resolution and the reference is that finer hold in floor-widths, squared.
    rho_floor_tracking: float = (6.29e-6 / 1e-4) ** 2
    rho_floor: float = (6.29e-6 / 1e-4) ** 2  # no running cost: rho_floor_tracking
    #: False: the plant is disturbed (docs/reward-shaping.md).
    floor_is_documented_minimum: bool = False


@struct.dataclass
class UnstableCSTRState(EnvState):
    C_a: float
    T: float
    #: Jacket temperature: the lagged coolant the balances see.
    T_j: float
    #: Feed-temperature drift, added to ``Ti``. Hidden.
    Ti_dev: float
    #: The episode's six levels. Only the live one is observed.
    target_levels: jnp.ndarray
    #: Steps since the schedule started. Hidden. ``reset_env`` sets it to 0, so
    #: a trip's fresh draw restarts the schedule while ``time`` runs on.
    block_clock: int


def steady_temperature(level, params: UnstableCSTRParams, xp=jnp):
    """T*(L), the reactor temperature at which ``C_a = L`` is steady.

    From the concentration balance alone, (q/V)(Caf - L) = k(T) L, so it does
    not depend on the feed temperature or its drift.
    """
    qV = params.q / params.V
    return params.EA_over_R / (
        xp.log(params.k0) - xp.log(qV * (params.Caf - level) / level)
    )


def steady_coolant(level, Ti_dev, params: UnstableCSTRParams, xp=jnp):
    """Tc*(L, dTi), the jacket temperature that makes (L, T*(L)) an equilibrium.

    Tc* = T* + ((q/V)(T* - Ti - dTi) - A (q/V)(Caf - L)) / beta, with
    A = -deltaHr / (rho C) and beta = UA / (rho C V).
    """
    qV = params.q / params.V
    A = -params.deltaHr / (params.rho * params.C)
    beta = params.UA / (params.rho * params.C * params.V)
    T = steady_temperature(level, params, xp)
    return T + (qV * (T - params.Ti - Ti_dev) - A * qV * (params.Caf - level)) / beta


def live_target(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    """The level the state is scored against and observes.

    The clamp holds the last level past block six. ``block_clock`` is traced
    under ``jit``, so the clamp uses ``xp.minimum``.
    """
    block = xp.minimum(state.block_clock // params.block_steps, N_BLOCKS - 1)
    return state.target_levels[block]


def draw_schedule(key, params: UnstableCSTRParams) -> jnp.ndarray:
    """The six levels of one episode, shape ``(N_BLOCKS,)``.

    The first is uniform in the band. Each switch takes the next size from a
    permutation of ``switch_sizes`` and a fair-coin sign, flipped when the
    move would leave the band. With sizes at most half the band one sign is
    always legal, so the sizes are exactly the multiset, in a random order.
    """
    start_key, order_key, sign_key = jax.random.split(key, 3)
    lo, hi = params.target_CA_range
    first = jax.random.uniform(start_key, minval=lo, maxval=hi)
    sizes = jax.random.permutation(order_key, jnp.asarray(params.switch_sizes))
    signs = jnp.where(jax.random.bernoulli(sign_key, 0.5, sizes.shape), 1.0, -1.0)

    def switch(level, size_and_sign):
        move = size_and_sign[0] * size_and_sign[1]
        leaves = (level + move < lo) | (level + move > hi)
        new = jnp.clip(level + jnp.where(leaves, -move, move), lo, hi)
        return new, new

    _, later = jax.lax.scan(switch, first, jnp.stack([sizes, signs], axis=1))
    return jnp.concatenate([first[None], later])


def compute_velocity(position, action, Ti_dev, params: UnstableCSTRParams):
    """dC_a/dt, dT/dt and dT_j/dt.

    The two balances are cstr's own, with the jacket temperature ``T_j`` in the
    coolant slot and the drifted feed ``Ti + Ti_dev``. The jacket follows the
    command ``action`` through a first-order lag.
    """
    T_j = position[2]
    balances = cstr_env.compute_velocity(
        position[:2], T_j, params.replace(Ti=params.Ti + Ti_dev)
    )[0]
    return (
        jnp.array([balances[0], balances[1], (action - T_j) / params.tau_j]),
        None,
    )


@partial(jax.jit, static_argnames=["integration_method"])
def compute_next_state(
    T_c_raw: float,
    state: UnstableCSTRState,
    params: UnstableCSTRParams,
    key: jax.Array,
    integration_method: str = "rk4_1",
):
    """One step: the command, the three ODEs, the drift, then the clocks.

    T_c_raw : coolant command, raw in [-1, 1] (clipped); 0 gives 300 K.
    """
    T_c = convert_raw_action_to_range(
        T_c_raw, min_action=params.T_c_min, max_action=params.T_c_max
    )
    # The jacket lag is the third state inside the same RK4 step. The drift is
    # held at its start-of-step value, the value the MPC reads.
    _compute_velocity = partial(
        compute_velocity, action=T_c, Ti_dev=state.Ti_dev, params=params
    )
    new_positions, _ = integrate_dynamics(
        positions=jnp.array([state.C_a, state.T, state.T_j]),
        delta_t=params.delta_t,
        compute_velocity=_compute_velocity,
        method=integration_method,
    )

    # Exact Ornstein-Uhlenbeck update. The innovation is drawn from a key
    # folded with the pre-increment ``time``, as glass_furnace and battery do,
    # so a caller passing a constant key still gets a zero-mean process.
    a = jnp.exp(-params.delta_t / params.Ti_tau)
    xi = jax.random.normal(jax.random.fold_in(key, state.time))
    new_Ti_dev = a * state.Ti_dev + params.Ti_sigma * jnp.sqrt(1.0 - a**2) * xi

    # The live target is derived from ``block_clock``, so advancing it is the
    # whole schedule update.
    return (
        state.replace(
            C_a=new_positions[0],
            T=new_positions[1],
            T_j=new_positions[2],
            Ti_dev=new_Ti_dev,
            block_clock=state.block_clock + 1,
            time=state.time + 1,
        ),
        None,
    )


# Deliberately not jitted (the shipped envs' reason). Decorated with
# ``@partial(jax.jit, static_argnames=["params"])``, it would key the
# compilation cache on the params object, so a fresh ``Params(...)``, which
# every sweep, tuner and MPC builds, would be a cache miss and a full recompile,
# measured at ~1600x the cost of a cached call. Callers that want it fused
# already jit ``step_env``, which traces this inline.
def get_obs(state: UnstableCSTRState, params: UnstableCSTRParams):
    """``[C_a, T, T_j, live target]``. The drift, the later levels and the
    block clock are hidden."""
    return jnp.array([state.C_a, state.T, state.T_j, live_target(state, params)])


def check_is_terminal(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    # One-sided: the extinguished branch (T about 310 to 330 K) is charged as
    # tracking error and is always recoverable. The lower bound and the sign of
    # C_a are a validity guard no reachable state meets. A NaN fails every
    # comparison, so a non-finite proposal trips.
    inside = (
        (state.T < params.T_trip) & (state.T > params.T_valid_min) & (state.C_a >= 0.0)
    )
    terminated = xp.logical_not(inside)
    truncated = state.time >= params.max_steps_in_episode
    return terminated, truncated


def compute_reward_terms(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    """The reward's additive cost terms, each >= 0 (``target_gym.reward``)."""
    terminated, _ = check_is_terminal(state, params, xp)
    terms = {
        "tracking": R.tracking_cost(
            live_target(state, params, xp) - state.C_a,
            params.e_floor,
            params.e_tol,
            params.tracking_exponent,
            xp,
        ),
    }
    return R.with_trip(terms, terminated, R.trip_cost(params), xp)


def compute_reward_v1(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    """Version 1 (ours): log-scaled tracking, 1 at zero error and 0 at
    ``CA_error_max``, and ``-restart_steps`` on a tripped step.

    With 0 on the tripped step, a policy that trips and restarts warm scored
    above every safe one (PHYSICS.md, version-1 reward). Charging the downtime
    at version 1's best per-step score, as version 2 prices it, restores the
    order. Version 1 is therefore not bounded in [0, 1] on this task.
    """
    terminated, _ = check_is_terminal(state, params, xp)
    tracking = log_scaled_reward(
        xp.abs(live_target(state, params, xp) - state.C_a),
        params.precision_floor,
        params.CA_error_max,
        xp,
    )
    return xp.where(terminated, -1.0 * params.restart_steps, tracking)


def compute_reward(state: UnstableCSTRState, params: UnstableCSTRParams, xp=jnp):
    return R.select(
        params.reward_version,
        compute_reward_v1(state, params, xp),
        R.total(compute_reward_terms(state, params, xp), xp),
        xp,
    )
