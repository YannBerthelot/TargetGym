"""Controllers for the unstable CSTR, and its point-of-no-return diagnostic.

See ``PHYSICS.md`` in this directory, section 7. Everything here is outside the
version stamp and inside the baseline fingerprint: the env never imports this
module at load time, so editing a controller or the diagnostic never moves
``unstable_cstr-v2``.

The cascade PID holds the saddle through its inner loop on T. No P or PI loop
on C_a alone can stabilise the plant (PHYSICS.md, section 2), so the outer PI
on C_a only sets the temperature the inner loop holds. Its factory refuses
gains whose linearised closed loop does not decay fast enough at every target
(``check_cascade_gains``), which is how the tuner's search is kept away from a
controller that chatters between the coolant bounds.

The point of no return is the lowest reactor temperature from which full
cooling still reaches the trip. ``point_of_no_return`` computes it exactly on
the env's own step; ``pnr_line`` is a straight line in C_a fitted below it at
the hottest feed drift, which the PID's setpoint ceiling and the MPC's soft
bound read. ``raw_from_coolant`` and ``coolant_from_raw`` convert between the
raw action and the coolant command in K.

The MPC is a CasADi/IPOPT NMPC on orthogonal collocation, which keeps the
unstable mode from blowing up a propagated rollout. It reads the true state,
the drift and the whole schedule, keeps T under the point-of-no-return line
less ``MPC_PNR_MARGIN_K`` through a soft constraint, and hands any step whose
solve fails to the cascade PID, whose memory it keeps tracked to the actions
it applies.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import enable_x64
from jax.tree_util import Partial as partial
from scipy.linalg import solve_discrete_are

from target_gym.experts import pid as _pid_mod
from target_gym.experts.mpc import CasadiMPC
from target_gym.pc_gym.unstable_cstr.env import (
    N_BLOCKS,
    UnstableCSTRParams,
    UnstableCSTRState,
    check_is_terminal,
    compute_next_state,
    get_obs,
    steady_coolant,
    steady_temperature,
)

# ---------------------------------------------------------------------------
# Point of no return
# ---------------------------------------------------------------------------

#: The point-of-no-return line, T = PNR_LINE[0] + PNR_LINE[1] (C_a -
#: PNR_LINE_C_A): K at ``PNR_LINE_C_A``, and K per mol/L. Derived: of the lines
#: at or below the exact point of no return at a +6 K feed drift (3 standard
#: deviations) on a 0.005 mol/L grid over C_a 0.40 to 0.70, the one with the
#: largest mean there, its slope rounded to 0.1 and its intercept rounded down
#: to 0.1 K (scripts/unstable_cstr_numbers.py --section pnr). The PID's
#: setpoint ceiling and the MPC's soft bound read it, so it is frozen with the
#: physics, and changing it re-opens the tuning and every recorded baseline.
PNR_LINE = (354.9, -40.7)
#: The concentration at which ``PNR_LINE[0]`` is given (mol/L), the band's hot end.
PNR_LINE_C_A = 0.45

# Lower end of the bisection (K, ours). ``point_of_no_return`` returns NaN
# where full cooling trips even from here.
_PNR_LOW = 325.0


def pnr_line(C_a):
    """The point-of-no-return line at concentration ``C_a``, in K.

    It lies at or below the exact point of no return (jacket at the steady
    coolant) across C_a 0.40 to 0.70 at every drift within 3 standard
    deviations (scripts/unstable_cstr_numbers.py --section pnr prints the gaps).
    Plain arithmetic, so the MPC reads it on CasADi symbols too.
    """
    return PNR_LINE[0] + PNR_LINE[1] * (C_a - PNR_LINE_C_A)


def raw_from_coolant(T_c, params: UnstableCSTRParams):
    """The raw action that commands coolant temperature ``T_c`` (K).

    The inverse of ``coolant_from_raw``. A ``T_c`` outside [T_c_min, T_c_max]
    gives a raw action outside [-1, 1], which the env clips.
    """
    return 2.0 * (T_c - params.T_c_min) / (params.T_c_max - params.T_c_min) - 1.0


def coolant_from_raw(u_raw, params: UnstableCSTRParams):
    """The coolant command (K) of raw action ``u_raw`` in [-1, 1].

    The linear map of ``convert_raw_action_to_range`` without its clip, so it
    also takes CasADi symbols. Clip ``u_raw`` first where it may leave
    [-1, 1], as ``compute_next_state`` does.
    """
    return params.T_c_min + 0.5 * (u_raw + 1.0) * (params.T_c_max - params.T_c_min)


@partial(jax.jit, static_argnames=["horizon_steps"])
def trips_under_full_cooling(
    C_a, T, T_j, Ti_dev, params: UnstableCSTRParams, horizon_steps: int = 200
):
    """True where holding the coolant at ``T_c_min`` from ``(C_a, T, T_j)``
    still trips within ``horizon_steps``.

    The env's own step (``compute_next_state``, the jacket lag included) with
    the drift frozen at ``Ti_dev``, and the env's trip check at every step
    end. Scalar inputs; vmap for arrays.
    """
    state = UnstableCSTRState(
        time=0,
        C_a=C_a,
        T=T,
        T_j=T_j,
        Ti_dev=Ti_dev,
        target_levels=jnp.zeros(N_BLOCKS),
        block_clock=0,
    )
    key = jax.random.PRNGKey(0)

    def body(carry, _):
        s, tripped = carry
        s = compute_next_state(-1.0, s, params, key)[0].replace(Ti_dev=Ti_dev)
        return (s, tripped | check_is_terminal(s, params)[0]), None

    carry, _ = jax.lax.scan(
        body,
        (state, check_is_terminal(state, params)[0]),
        None,
        length=horizon_steps,
    )
    return carry[1]


@partial(jax.jit, static_argnames=["horizon_steps", "iters"])
def point_of_no_return(
    C_a, T_j, Ti_dev, params: UnstableCSTRParams, horizon_steps=200, iters=24
):
    """The lowest T from which full cooling still reaches ``T_trip`` (K).

    Holding the coolant command at ``T_c_min`` through the jacket lag, with
    the drift frozen at ``Ti_dev`` (``trips_under_full_cooling``). Bisection
    on [325 K, ``T_trip``]; 24 iterations resolve it to 40 K / 2^24 = 2.4e-6 K,
    and a point that trips even from 325 K gives NaN. Scalar inputs; vmap for
    arrays, or call ``point_of_no_return_batch``. Used by the tests and the
    numbers script, never by ``step_env``.
    """
    one = jnp.ones_like(jnp.asarray(T_j))

    def bisect(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        trips = trips_under_full_cooling(C_a, mid, T_j, Ti_dev, params, horizon_steps)
        return jnp.where(trips, lo, mid), jnp.where(trips, mid, hi)

    low = _PNR_LOW * one
    bracket = jax.lax.fori_loop(0, iters, bisect, (low, params.T_trip * one))
    outside = trips_under_full_cooling(C_a, low, T_j, Ti_dev, params, horizon_steps)
    return jnp.where(outside, jnp.nan, bracket[1])


def point_of_no_return_batch(C_a, T_j, Ti_dev, params: UnstableCSTRParams):
    """``point_of_no_return`` at every point of broadcastable arrays, in float64.

    Returns a NumPy array of the broadcast shape (at least 1-D), NaN where full
    cooling trips even from the bisection's lower end.
    """
    args = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, float)) for x in (C_a, T_j, Ti_dev))
    )
    with enable_x64():
        f = jax.vmap(point_of_no_return, in_axes=(0, 0, 0, None))
        flat = f(*(jnp.asarray(x.ravel(), jnp.float64) for x in args), params)
    return np.asarray(flat).reshape(args[0].shape)


# ---------------------------------------------------------------------------
# Cascade PID
# ---------------------------------------------------------------------------

#: The tuner's starting point and the factory's fallback (ours). Every tuned
#: gain is nonzero: the coordinate search multiplies gains, so a zero would
#: never move.
DEFAULT_GAINS: dict = {
    "Kp_T": 7.0,  # K of coolant per K of T error; inner proportional
    "Kd_T": 0.4,  # K of coolant per K/min; inner derivative, first difference
    "Kc_Ca": 100.0,  # K of setpoint per mol/L; outer proportional
    "Ti_Ca": 0.5,  # min; outer integral time
    "sp_rate": 3.0,  # K/min; setpoint rate limit, 0.15 K per step
    "sp_below": 8.0,  # K; setpoint floor T* - 8 K
    "sp_above": 2.0,  # K; setpoint ceiling T* + 2 K
    "sp_line_margin": 0.3,  # K; setpoint ceiling pnr_line(L) - 0.3 K
    "Tc_bias": 300.0,  # K; inner-loop coolant bias
}
#: The gains the tuner searches.
TUNED_GAINS = ("Kp_T", "Kd_T", "Kc_Ca", "Ti_Ca")
#: The gains it holds. Their right value is set by the distance to the point
#: of no return, which the return sees only once a trip happens.
HELD_GAINS = ("sp_rate", "sp_below", "sp_above", "sp_line_margin", "Tc_bias")


class CascadeState(NamedTuple):
    """The cascade's memory. Read by attribute, never unpacked."""

    #: Integral of the C_a error (mol/L min).
    integral: float
    #: Reactor temperature at the previous step (K), for the derivative.
    T_prev: float
    #: Temperature setpoint at the previous step (K), for the rate limit.
    T_sp_prev: float


def setpoint_window(level, gains, params: UnstableCSTRParams, xp=np):
    """The clip on the temperature setpoint at target ``level``, in K.

    From T*(L) - ``sp_below`` to the lower of T*(L) + ``sp_above`` and
    ``pnr_line(L) - sp_line_margin``. The line is read at the target level,
    so the ceiling is constant within a block and adds no nonlinearity to the
    loop the guard linearises. With ``DEFAULT_GAINS`` the line binds for
    targets below about 0.46 mol/L (derived, --section pnr).
    """
    T_ff = steady_temperature(level, params, xp)
    ceiling = xp.minimum(
        T_ff + gains["sp_above"], pnr_line(level) - gains["sp_line_margin"]
    )
    return T_ff - gains["sp_below"], ceiling


def cascade_init(obs, gains, params: UnstableCSTRParams, xp=np) -> CascadeState:
    """Fresh memory for observation ``obs``: no integral, no derivative kick on
    the first step, and the setpoint starting at the measured T.

    Starting the setpoint at the measured T keeps the rate limit in force from
    the first step. On the catches from the extinguished branch that
    scripts/unstable_cstr_numbers.py --pid-validation runs, T starts 17 to
    43 K below T* (derived). A setpoint started at T* sends the coolant to its
    upper bound until T arrives, and with ``DEFAULT_GAINS`` the reactor then
    overshoots to the trip on 42 of those 50 catches; started at T, none trips
    (measured, --pid-validation runs both starts). A warm reset starts within
    1.5 K of T*, so there the two starts differ by at most 1.5 K.
    """
    T = obs[1]
    return CascadeState(integral=xp.zeros_like(T), T_prev=T, T_sp_prev=T)


def cascade_step(gains, cs: CascadeState, obs, params: UnstableCSTRParams, xp=np):
    """One update of the cascade. Returns ``(u_raw, new memory)``.

    ``obs`` is the env's ``[C_a, T, T_j, target]``. With ``xp=jnp`` it traces,
    so it scans and vmaps inside the env's own rollouts.
    """
    C_a, T, level = obs[0], obs[1], obs[3]
    dt = params.delta_t
    e = level - C_a

    # Outer PI on C_a around the steady-state feedforward. A C_a below its
    # target means the reaction runs too fast, so the setpoint goes down.
    T_ff = steady_temperature(level, params, xp)
    T_sp_raw = T_ff - gains["Kc_Ca"] * e - gains["Kc_Ca"] / gains["Ti_Ca"] * cs.integral
    lo, hi = setpoint_window(level, gains, params, xp)
    ramp = gains["sp_rate"] * dt
    T_sp = xp.clip(xp.clip(T_sp_raw, lo, hi), cs.T_sp_prev - ramp, cs.T_sp_prev + ramp)

    # Inner PD on T, the derivative on the measurement.
    T_c_raw = (
        gains["Tc_bias"]
        + gains["Kp_T"] * (T_sp - T)
        - gains["Kd_T"] * (T - cs.T_prev) / dt
    )
    T_c = xp.clip(T_c_raw, params.T_c_min, params.T_c_max)

    # Conditional integration: the integral moves only while neither the
    # coolant nor the setpoint is limited.
    free = (T_c == T_c_raw) & (T_sp == T_sp_raw)
    new = CascadeState(
        integral=cs.integral + xp.where(free, e * dt, 0.0),
        T_prev=T,
        T_sp_prev=T_sp,
    )
    return raw_from_coolant(T_c, params), new


def cascade_track(
    gains, cs: CascadeState, obs, u_raw, params: UnstableCSTRParams, xp=np
) -> CascadeState:
    """Bumpless transfer: the memory that would have produced the applied
    action ``u_raw`` at this observation.

    The setpoint is the one the inner PD turns into that coolant, clipped to
    the setpoint window, and the integral the one at which the outer PI gives
    that setpoint. A controller that takes over on the next step starts from
    the coolant last applied.
    """
    C_a, T, level = obs[0], obs[1], obs[3]
    T_c = coolant_from_raw(xp.clip(u_raw, -1.0, 1.0), params)
    lo, hi = setpoint_window(level, gains, params, xp)
    derivative = gains["Kd_T"] * (T - cs.T_prev) / params.delta_t
    T_sp = xp.clip(T + (T_c - gains["Tc_bias"] + derivative) / gains["Kp_T"], lo, hi)
    T_ff = steady_temperature(level, params, xp)
    integral = (
        (T_ff - gains["Kc_Ca"] * (level - C_a) - T_sp) * gains["Ti_Ca"] / gains["Kc_Ca"]
    )
    return CascadeState(integral=integral, T_prev=T, T_sp_prev=T_sp)


class UnstableCSTRCascadePID:
    """Cascade PID for the unstable CSTR: PI on C_a sets T, PD on T sets the coolant.

    Observation layout (``env.get_obs``)::

        [C_a, T, T_j, target]

    The action is raw in [-1, 1]. The memory persists through a trip (the
    suite's convention), and after a restart the derivative sees the jump in
    T once. ``track`` is what the MPC calls on the steps it controls, so that
    a fallback to this controller starts from the coolant last applied.
    """

    def __init__(self, gains=None, params: UnstableCSTRParams | None = None):
        gains = dict(DEFAULT_GAINS if gains is None else gains)
        unknown = sorted(set(gains) - set(DEFAULT_GAINS))
        missing = sorted(set(DEFAULT_GAINS) - set(gains))
        if unknown or missing:
            raise ValueError(
                f"cascade gains: unknown {unknown}, missing {missing}; "
                f"expected exactly {sorted(DEFAULT_GAINS)}"
            )
        self.gains = {k: float(v) for k, v in gains.items()}
        self.params = UnstableCSTRParams() if params is None else params
        self.reset()

    def reset(self):
        """Clear the memory. The next ``step`` or ``track`` starts it afresh."""
        self._cs = None

    def _memory(self, obs) -> CascadeState:
        if self._cs is None:
            self._cs = cascade_init(obs, self.gains, self.params)
        return self._cs

    def step(self, obs) -> np.ndarray:
        obs = np.asarray(obs, float)
        u_raw, self._cs = cascade_step(self.gains, self._memory(obs), obs, self.params)
        return np.array([float(u_raw)])

    def track(self, obs, u_raw) -> None:
        obs = np.asarray(obs, float)
        self._cs = cascade_track(
            self.gains,
            self._memory(obs),
            obs,
            float(np.asarray(u_raw, float).reshape(())),
            self.params,
        )

    __call__ = step


def load_gains() -> dict:
    """``DEFAULT_GAINS`` overlaid with the numeric entries stored under
    ``"unstable_cstr"`` in ``data/pid_gains.json`` (its ``"note"`` is skipped).

    Reads ``target_gym.experts.pid._load_gains()`` at call time, so a tuner
    that patches ``_gains_cache`` is seen.
    """
    stored = _pid_mod._load_gains().get("unstable_cstr", {})
    return {
        **DEFAULT_GAINS,
        **{
            k: float(v)
            for k, v in stored.items()
            if not isinstance(v, (str, dict, list))
        },
    }


def make_unstable_cstr_pid() -> UnstableCSTRCascadePID:
    """Registry factory: the stored gains, refused unless the guard passes."""
    gains = load_gains()
    check_cascade_gains(gains)
    return UnstableCSTRCascadePID(gains)


# ---------------------------------------------------------------------------
# Stability guard
# ---------------------------------------------------------------------------

#: Slowest closed-loop decay the factory accepts, /min (ours). ``DEFAULT_GAINS``
#: decay at 1.73 /min or faster at every target (derived, ``closed_loop_rates``).
GUARD_MIN_DECAY = 0.5
#: Targets the guard checks (mol/L): the band's ends and three points between.
GUARD_LEVELS = (0.45, 0.50, 0.55, 0.60, 0.65)


class UnstableCascadeGains(ValueError):
    """Cascade gains the stability guard refuses."""


def _closed_loop_equilibrium(level, Ti_dev, gains, params: UnstableCSTRParams):
    """``[C_a, T, T_j, integral, T_prev, T_sp_prev]`` at which the closed loop
    rests on target ``level``: the plant on (L, T*, Tc*) and the memory that
    commands Tc* there. The rate limit and the clip are inactive. Gains may be
    arrays; the last axis is the six entries."""
    T_star = steady_temperature(level, params, np)
    T_c = steady_coolant(level, Ti_dev, params, np)
    T_sp = T_star + (T_c - gains["Tc_bias"]) / gains["Kp_T"]
    integral = -(T_sp - T_star) * gains["Ti_Ca"] / gains["Kc_Ca"]
    return np.stack(
        np.broadcast_arrays(level, T_star, T_c, integral, T_star, T_sp), axis=-1
    )


def _closed_loop_jacobian(z, level, Ti_dev, gains, params: UnstableCSTRParams):
    """Jacobian of one closed-loop step (the cascade, then the env's
    ``compute_next_state``) with respect to ``z``, at ``z``."""

    def one_step(x):
        state = UnstableCSTRState(
            time=0,
            C_a=x[0],
            T=x[1],
            T_j=x[2],
            Ti_dev=Ti_dev,
            target_levels=jnp.full(N_BLOCKS, level),
            block_clock=0,
        )
        cs = CascadeState(integral=x[3], T_prev=x[4], T_sp_prev=x[5])
        u_raw, cs = cascade_step(gains, cs, get_obs(state, params), params, xp=jnp)
        new = compute_next_state(u_raw, state, params, jax.random.PRNGKey(0))[0]
        return jnp.stack(
            [new.C_a, new.T, new.T_j, cs.integral, cs.T_prev, cs.T_sp_prev]
        )

    return jax.jacfwd(one_step)(z)


_closed_loop_jacobians = jax.jit(
    jax.vmap(_closed_loop_jacobian, in_axes=(0, 0, None, None, None))
)


def closed_loop_rates(
    gains,
    params: UnstableCSTRParams | None = None,
    levels=GUARD_LEVELS,
    Ti_dev: float = 0.0,
) -> np.ndarray:
    """Per target, the closed loop's slowest rate, log(max |eig J|) / delta_t
    in /min: negative decays.

    J is the Jacobian of one closed-loop step at the equilibrium for that
    target and drift, from ``jax.jacfwd`` through ``cascade_step`` and the
    env's own ``compute_next_state`` (drift noise off, drift held), so no
    dynamics are restated. Always computed in float64, so the verdict does
    not depend on the caller's precision. The drift enters the balances
    additively, so the rates do not depend on ``Ti_dev``.
    """
    params = (UnstableCSTRParams() if params is None else params).replace(Ti_sigma=0.0)
    gains = {k: float(gains[k]) for k in DEFAULT_GAINS}
    with enable_x64():
        z = np.stack(
            [_closed_loop_equilibrium(L, Ti_dev, gains, params) for L in levels]
        )
        J = np.asarray(
            _closed_loop_jacobians(
                jnp.asarray(z),
                jnp.asarray(levels, dtype=jnp.float64),
                jnp.float64(Ti_dev),
                gains,
                params,
            )
        )
    radius = np.abs(np.linalg.eigvals(J)).max(axis=-1)
    return np.log(radius) / params.delta_t


def setpoint_clip_margins(
    gains,
    params: UnstableCSTRParams | None = None,
    Ti_devs=(-6.0, 0.0, 6.0),
    levels=GUARD_LEVELS,
) -> np.ndarray:
    """Distance (K) of the equilibrium setpoint T* + (Tc* - Tc_bias) / Kp_T
    from the nearer edge of the setpoint window, per target (rows) and drift
    (columns). Negative means the cascade cannot reach that equilibrium."""
    params = UnstableCSTRParams() if params is None else params
    L = np.asarray(levels, float)[:, None]
    T_star = steady_temperature(L, params, np)
    T_c = steady_coolant(L, np.asarray(Ti_devs, float)[None, :], params, np)
    T_sp = T_star + (T_c - gains["Tc_bias"]) / gains["Kp_T"]
    lo, hi = setpoint_window(L, gains, params, np)
    return np.minimum(T_sp - lo, hi - T_sp)


@functools.lru_cache(maxsize=256)
def _guard_problems(gain_items, params):
    gains = dict(gain_items)
    rates = closed_loop_rates(gains, params)
    margins = setpoint_clip_margins(gains, params)
    problems = [
        f"closed loop at C_a {L:.2f} has rate {r:+.3f} /min, "
        f"slower than -{GUARD_MIN_DECAY} /min"
        for L, r in zip(GUARD_LEVELS, rates)
        if not r <= -GUARD_MIN_DECAY
    ]
    problems += [
        f"equilibrium setpoint at C_a {L:.2f} is {-m:.3f} K outside the "
        "setpoint window"
        for L, row in zip(GUARD_LEVELS, margins)
        for m in row
        if not m > 0.0
    ]
    return tuple(problems)


def check_cascade_gains(gains, params: UnstableCSTRParams | None = None) -> None:
    """Raise ``UnstableCascadeGains`` unless the closed loop decays at
    ``GUARD_MIN_DECAY`` or faster at every target and each equilibrium
    setpoint lies inside the setpoint window at a drift of -6, 0 and +6 K.

    Cached on the gain values, so the tuner pays the compile once per process.
    """
    problems = _guard_problems(
        tuple((k, float(gains[k])) for k in DEFAULT_GAINS), params
    )
    if problems:
        raise UnstableCascadeGains("; ".join(problems))


# ---------------------------------------------------------------------------
# CasADi NMPC
# ---------------------------------------------------------------------------

#: The MPC's horizon, in steps of ``delta_t`` (ours), 1.6 min. From 0.01 K
#: inside the point of no return, full cooling turns the excursion at most 21
#: steps later, at a +6 K drift (derived, scripts/unstable_cstr_numbers.py
#: --section pnr). 32 steps is 1.52 times that, so a plan that recovers from
#: near the edge runs past the peak.
MPC_HORIZON = 32
#: How far below the point-of-no-return line the MPC's soft bound on T sits
#: (K, ours).
MPC_PNR_MARGIN_K = 1.0
#: The C_a error that costs 1 per stage (mol/L, ours). It only conditions the
#: NLP: the stage cost is the v2 tracking term times (e_floor / error_scale)^2.
MPC_ERROR_SCALE = 0.01
#: Weight on the squared move of the raw action between stages (ours).
MPC_MOVE_WEIGHT = 1e-3
#: Cost per K of excess over the soft bound, per control interval (ours).
MPC_PNR_PENALTY = 1e4
#: The largest move, in K of coolant, that the fallback PID's first action may
#: make from the coolant last applied (ours, provisional).
#: test_mpc_falls_back_to_the_pid asserts it, and
#: scripts/unstable_cstr_numbers.py --mpc-gate measures that move on every
#: step of its episodes and prints it against this bound.
FALLBACK_STEP_BOUND_K = 0.5
#: The time-varying parameters of each stage: the target, the feed drift, and
#: the equilibrium on that target at that drift, which the terminal cost reads.
MPC_TVP = ("target", "Ti_dev", "C_a_eq", "T_eq", "T_j_eq")


def _casadi_rhs(params: UnstableCSTRParams, C_a, T, T_j, u_raw, Ti_dev, exp):
    """dC_a/dt, dT/dt and dT_j/dt for the MPC's model.

    The env's ``compute_velocity`` runs on ``jnp`` and CasADi cannot trace it,
    so this is the one place the balances are written a second time, term for
    term in the env's order: cstr's two balances with the jacket temperature
    as the coolant and the drifted feed ``Ti + Ti_dev``, and the jacket lag
    driven by the raw command. test_mpc_model_is_the_env_velocity and
    test_mpc_predicts_one_step_like_the_plant hold it to the env.
    """
    p = params
    T_c = coolant_from_raw(u_raw, p)
    rA = p.k0 * exp(-p.EA_over_R / T) * C_a
    return [
        p.q / p.V * (p.Caf - C_a) - rA,
        p.q / p.V * ((p.Ti + Ti_dev) - T)
        + ((-p.deltaHr) * rA) * (1 / (p.rho * p.C))
        + p.UA * (T_j - T) * (1 / (p.rho * p.C * p.V)),
        (T_c - T_j) / p.tau_j,
    ]


def one_step_linearisation(
    params: UnstableCSTRParams, level: float = PNR_LINE_C_A, Ti_dev: float = 0.0
):
    """``(A, B)``: the Jacobians of one env step with respect to (C_a, T, T_j)
    and the raw action, at the equilibrium on target ``level`` and drift
    ``Ti_dev``.

    From ``jax.jacfwd`` through the env's own ``compute_next_state`` in
    float64, with the drift held, so no dynamics are restated.
    """
    params = params.replace(Ti_sigma=0.0)
    T_star = float(steady_temperature(level, params, np))
    T_c = float(steady_coolant(level, Ti_dev, params, np))
    with enable_x64():

        def step(x, u_raw):
            state = UnstableCSTRState(
                time=0,
                C_a=x[0],
                T=x[1],
                T_j=x[2],
                Ti_dev=Ti_dev,
                target_levels=jnp.full(N_BLOCKS, level),
                block_clock=0,
            )
            new = compute_next_state(u_raw, state, params, jax.random.PRNGKey(0))[0]
            return jnp.stack([new.C_a, new.T, new.T_j])

        x = jnp.array([level, T_star, T_c], jnp.float64)
        u = jnp.float64(raw_from_coolant(T_c, params))
        A, B = jax.jacfwd(step, argnums=(0, 1))(x, u)
    return np.asarray(A), np.asarray(B).reshape(3, 1)


def terminal_weight(
    params: UnstableCSTRParams,
    error_scale: float = MPC_ERROR_SCALE,
    move_weight: float = MPC_MOVE_WEIGHT,
    level: float = PNR_LINE_C_A,
) -> np.ndarray:
    """The MPC's terminal weight P, a 3 x 3 matrix on (C_a, T, T_j).

    The solution of the discrete algebraic Riccati equation for the env's
    one-step linearisation at C_a ``level`` (0.45 by default, the target whose
    unstable mode is fastest, the lag included), with the stage weights
    Q = diag(1 / error_scale^2, 1e-6, 1e-6) and R = move_weight: the cost of
    the rest of the episode under the LQR there, so the horizon's end does not
    look free to an unstable plant.
    """
    A, B = one_step_linearisation(params, level)
    Q = np.diag([1.0 / error_scale**2, 1e-6, 1e-6])
    return solve_discrete_are(A, B, Q, np.array([[move_weight]]))


class UnstableCSTRCasadiMPC(CasadiMPC):
    """CasADi/IPOPT NMPC for the unstable CSTR, falling back to the cascade PID.

    States : [C_a, T, T_j]   Input: u_raw in [-1, 1] -> T_c in [T_c_min, T_c_max]
    Stage k: tvp ``MPC_TVP``, the level the env scores the state k steps ahead
             against, the drift's mean path, and the equilibrium they define

    Collocation (radau, degree 3) makes the model a set of constraints, so the
    unstable mode cannot blow up a propagated rollout, which is what defeats a
    single-shooting planner here. The stage cost is the v2 tracking term,
    rescaled; the terminal cost is ``terminal_weight`` around the equilibrium
    on the last stage's target; T stays under ``pnr_line(C_a) -
    pnr_margin_K`` through a soft constraint checked at every collocation
    point, so a state that starts past it still has a solution.

    Like every MPC in the suite it is an oracle ceiling. It reads the true
    C_a, T, T_j and feed drift, and the whole schedule with its clock. The
    drift path is the drift's conditional mean, a^k Ti_dev, which is the env's
    own update with ``Ti_sigma`` zeroed (``plan_params``).

    A step whose solve fails, or reports success with a non-finite action, is
    counted in ``solve_failures`` and handed to the cascade PID. The PID's
    memory is tracked to every action the MPC applies (``cascade_track``), so
    its first action starts from the coolant last applied. Holding the last
    action, as ``CasadiMPC`` does, can trip this plant.
    """

    SCALING = {"_x": {"C_a": 0.5, "T": 350.0, "T_j": 300.0}}

    def __init__(
        self,
        env,
        params: UnstableCSTRParams,
        horizon: int = MPC_HORIZON,
        mpc_dt: float | None = None,
        pnr_margin_K: float = MPC_PNR_MARGIN_K,
        error_scale: float = MPC_ERROR_SCALE,
        move_weight: float = MPC_MOVE_WEIGHT,
        pnr_penalty: float = MPC_PNR_PENALTY,
        fallback: UnstableCSTRCascadePID | None = None,
    ):
        # Everything _build_mpc reads, before CasadiMPC.__init__ calls it.
        self.pnr_margin_K = float(pnr_margin_K)
        self.error_scale = float(error_scale)
        self.move_weight = float(move_weight)
        self.pnr_penalty = float(pnr_penalty)
        self._pid = (
            UnstableCSTRCascadePID(load_gains(), params)
            if fallback is None
            else fallback
        )
        self._preview: dict = {}
        self._last_clock: int | None = None
        super().__init__(env, params, horizon=horizon, mpc_dt=mpc_dt)

    def _build_mpc(self):
        import casadi
        import do_mpc

        p = self.params
        model = do_mpc.model.Model("continuous")
        C_a = model.set_variable("_x", "C_a")
        T = model.set_variable("_x", "T")
        T_j = model.set_variable("_x", "T_j")
        u_raw = model.set_variable("_u", "u_raw")
        tvp = {name: model.set_variable("_tvp", name) for name in MPC_TVP}
        rhs = _casadi_rhs(p, C_a, T, T_j, u_raw, tvp["Ti_dev"], casadi.exp)
        for name, expr in zip(("C_a", "T", "T_j"), rhs):
            model.set_rhs(name, expr)
        model.setup()

        mpc = do_mpc.controller.MPC(model)
        mpc.set_param(
            n_horizon=self.horizon,
            t_step=self.mpc_dt,
            n_robust=0,
            store_full_solution=False,
            state_discretization="collocation",
            collocation_type="radau",
            collocation_deg=3,
            collocation_ni=1,
            # The bound is checked at every collocation point, the interval
            # ends included. The measured state is given, so it is left out.
            nl_cons_check_colloc_points=True,
        )

        stage = ((tvp["target"] - C_a) / self.error_scale) ** 2
        P = terminal_weight(
            p.replace(delta_t=self.mpc_dt), self.error_scale, self.move_weight
        )
        dx = casadi.vertcat(C_a - tvp["C_a_eq"], T - tvp["T_eq"], T_j - tvp["T_j_eq"])
        mpc.set_objective(lterm=stage, mterm=casadi.bilin(casadi.DM(P), dx, dx))
        mpc.set_rterm(u_raw=self.move_weight)
        mpc.bounds["lower", "_u", "u_raw"] = -1.0
        mpc.bounds["upper", "_u", "u_raw"] = 1.0

        mpc.set_nl_cons(
            "pnr",
            T - (pnr_line(C_a) - self.pnr_margin_K),
            ub=0.0,
            soft_constraint=True,
            penalty_term_cons=self.pnr_penalty,
        )

        self._preview = {name: np.zeros(self.horizon + 1) for name in MPC_TVP}
        tvp_tpl = mpc.get_tvp_template()

        def tvp_fun(_t):
            for name, values in self._preview.items():
                for k, value in enumerate(values):
                    tvp_tpl["_tvp", k, name] = float(value)
            return tvp_tpl

        mpc.set_tvp_fun(tvp_fun)
        self._apply_scaling(mpc)
        mpc.set_param(nlpsol_opts=self._quiet_ipopt())
        mpc.setup()

        # The model's right-hand side as a function, for the tests.
        x = casadi.SX.sym("x", 3)
        u = casadi.SX.sym("u_raw")
        d = casadi.SX.sym("Ti_dev")
        self.rhs_function = casadi.Function(
            "rhs",
            [x, u, d],
            [casadi.vertcat(*_casadi_rhs(p, x[0], x[1], x[2], u, d, casadi.exp))],
            ["x", "u_raw", "Ti_dev"],
            ["rhs"],
        )
        return mpc

    def preview(self, state) -> dict:
        """The NLP's time-varying parameters at ``state``: for each name in
        ``MPC_TVP``, an array over the stages k = 0..horizon.

        Stage k is the state k control intervals ahead. Its target is the
        level the env scores that state against,
        ``target_levels[min((block_clock + k) // block_steps, 5)]`` at
        ``mpc_dt = delta_t``, and its drift is a^k Ti_dev with
        a = exp(-mpc_dt / Ti_tau), held over the interval as the env holds it
        over a step.
        """
        p = self.params
        k = np.arange(self.horizon + 1)
        ahead = np.floor(k * (self.mpc_dt / p.delta_t) + 1e-9).astype(int)
        block = np.minimum(
            (int(state.block_clock) + ahead) // p.block_steps, N_BLOCKS - 1
        )
        target = np.asarray(state.target_levels, float)[block]
        drift = float(state.Ti_dev) * np.exp(-self.mpc_dt / p.Ti_tau) ** k
        return {
            "target": target,
            "Ti_dev": drift,
            "C_a_eq": target,
            "T_eq": steady_temperature(target, p, np),
            "T_j_eq": steady_coolant(target, drift, p, np),
        }

    def _update_setpoint(self, state):
        self._preview = self.preview(state)

    def _extract_x0(self, state):
        return np.array([float(state.C_a), float(state.T), float(state.T_j)])

    def _initialise(self, x0) -> None:
        """Cold start: every stage at the measured state, every move at the
        command that holds the jacket where it is."""
        m = self._mpc
        m.x0 = x0
        m.u0 = np.array([np.clip(raw_from_coolant(x0[2], self.params), -1.0, 1.0)])
        m.set_initial_guess()
        self._initialized = True

    def step(self, obs, state):
        """The raw action for ``state``; ``obs`` is what the fallback PID reads.

        A full override of ``CasadiMPC.step``, as the reactor's and the pH
        plant's are: the base clips after its fallback test, and ``np.clip``
        passes a NaN through, so a solve that reports success with a
        non-finite action would reach the plant. Here that counts as a failed
        solve. A trip restarts the schedule, and a falling block clock starts
        the warm start afresh from the new state.
        """
        self._update_setpoint(state)
        x0 = self._extract_x0(state)
        clock = int(state.block_clock)
        if not self._initialized or (
            self._last_clock is not None and clock < self._last_clock
        ):
            self._initialise(x0)
        self._last_clock = clock
        guess = self._save_guess()
        failures = self.solve_failures
        u = np.array(self._mpc.make_step(x0), dtype=float).flatten()
        finite = bool(np.all(np.isfinite(u)))
        ok = self._record_solve() and finite
        if not finite:
            # Counted once: _record_solve has already counted a solve that
            # reported failure.
            if self.solve_failures == failures:
                self.solve_failures += 1
            self.last_return_status = "non-finite action"
        if ok:
            self._pid.track(obs, np.clip(u, -1.0, 1.0))
        else:
            u = self._fallback(guess, u, obs)
        self._last_u = u
        return float(np.clip(u, -1.0, 1.0)[0])

    def _fallback(self, guess, u, obs):
        """Restore the last good warm start and hand this step to the cascade
        PID, stepped on ``obs``.

        The restore is ``CasadiMPC._fallback``'s; the action it would hold is
        not used. do-mpc keeps the failed iterate's first move as the previous
        move, which the next solve's move penalty reads, so it is replaced by
        the action applied.
        """
        super()._fallback(guess, u)
        u = np.asarray(self._pid.step(obs), dtype=float)
        self._mpc.u0 = u
        return u

    def reset(self):
        """Clear the warm start, do-mpc's stored history and the PID's memory."""
        super().reset()
        self._mpc.reset_history()
        self._pid.reset()
        self._last_clock = None


def make_unstable_cstr_mpc(env, params, **kwargs) -> UnstableCSTRCasadiMPC:
    """Registry factory: the CasADi/IPOPT NMPC, falling back to the cascade PID.

    The fallback is built from the stored gains and held to the PID factory's
    stability guard, so gains the guard refuses stop this factory too.
    ``kwargs`` go to ``UnstableCSTRCasadiMPC`` (``horizon``, ``mpc_dt``,
    ``pnr_margin_K``, ``error_scale``, ``move_weight``, ``pnr_penalty``); any
    other name raises ``TypeError``, which ``test_mpc_baselines._cheap_mpc``
    relies on.
    """
    gains = load_gains()
    check_cascade_gains(gains)
    return UnstableCSTRCasadiMPC(
        env, params, fallback=UnstableCSTRCascadePID(gains, params), **kwargs
    )
