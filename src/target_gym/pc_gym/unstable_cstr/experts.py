"""The unstable CSTR's expert, and its point-of-no-return diagnostic. The
MPC is in ``oracle.py``.

See ``PHYSICS.md`` in this directory, section 7. Everything here is outside the
version stamp and inside the expert fingerprint: the env never imports this
module at load time, so editing a controller or the diagnostic never moves
``unstable_cstr-v2``. The oracle is guided by the cascade PID and bounded by
``pnr_line``, so this module is in the oracle fingerprint too.

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
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.tree_util import Partial as partial

from target_gym.experts import pid as _pid_mod
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

try:
    # Newer JAX; 0.10 removed the experimental one. The declared range starts
    # at JAX 0.6, which has only that.
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

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
    Plain arithmetic, so it takes NumPy and JAX arrays alike.
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

    The linear map of ``convert_raw_action_to_range`` without its clip. Clip
    ``u_raw`` first where it may leave [-1, 1], as ``compute_next_state``
    does.
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
    "Kd_T": 0.4,  # K of coolant per K/min; inner derivative, filtered
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
    #: Filtered reactor temperature (K), the derivative's memory
    #: (``pid.filtered_derivative``).
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

    # Inner PD on T, the derivative on the filtered measurement.
    rate, T_filtered = _pid_mod.filtered_derivative(
        T, cs.T_prev, gains["Kp_T"], gains["Kd_T"], dt
    )
    T_c_raw = gains["Tc_bias"] + gains["Kp_T"] * (T_sp - T) - gains["Kd_T"] * rate
    T_c = xp.clip(T_c_raw, params.T_c_min, params.T_c_max)

    # Conditional integration: the integral moves only while neither the
    # coolant nor the setpoint is limited.
    free = (T_c == T_c_raw) & (T_sp == T_sp_raw)
    new = CascadeState(
        integral=cs.integral + xp.where(free, e * dt, 0.0),
        T_prev=T_filtered,
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
    rate, T_filtered = _pid_mod.filtered_derivative(
        T, cs.T_prev, gains["Kp_T"], gains["Kd_T"], params.delta_t
    )
    derivative = gains["Kd_T"] * rate
    T_sp = xp.clip(T + (T_c - gains["Tc_bias"] + derivative) / gains["Kp_T"], lo, hi)
    T_ff = steady_temperature(level, params, xp)
    integral = (
        (T_ff - gains["Kc_Ca"] * (level - C_a) - T_sp) * gains["Ti_Ca"] / gains["Kc_Ca"]
    )
    return CascadeState(integral=integral, T_prev=T_filtered, T_sp_prev=T_sp)


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

    Reads ``target_gym.experts.pid.load_gains()`` at call time, so a tuner
    that patches ``_gains_cache`` is seen.
    """
    stored = _pid_mod.load_gains().get("unstable_cstr", {})
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


@struct.dataclass
class UnstableCSTRPIDParams:
    """The cascade's gains (the ``DEFAULT_GAINS`` keys) and the plant it reads."""

    Kp_T: float
    Kd_T: float
    Kc_Ca: float
    Ti_Ca: float
    sp_rate: float
    sp_below: float
    sp_above: float
    sp_line_margin: float
    Tc_bias: float
    plant: UnstableCSTRParams


@struct.dataclass
class UnstableCSTRPIDState:
    """``CascadeState`` plus whether it has been started from an observation."""

    integral: float
    T_prev: float
    T_sp_prev: float
    started: bool


def unstable_cstr_pid_step(params, state, obs, xp=jnp):
    """``UnstableCSTRCascadePID.step`` as a pure step: the first call starts
    the memory from ``obs`` (``cascade_init``), as the class does lazily."""
    obs = xp.asarray(obs, float)
    gains = {k: getattr(params, k) for k in DEFAULT_GAINS}
    fresh = cascade_init(obs, gains, params.plant, xp)
    cs = CascadeState(
        integral=xp.where(state.started, state.integral, fresh.integral),
        T_prev=xp.where(state.started, state.T_prev, fresh.T_prev),
        T_sp_prev=xp.where(state.started, state.T_sp_prev, fresh.T_sp_prev),
    )
    u_raw, cs = cascade_step(gains, cs, obs, params.plant, xp)
    return xp.reshape(u_raw, (1,)), UnstableCSTRPIDState(
        integral=cs.integral, T_prev=cs.T_prev, T_sp_prev=cs.T_sp_prev, started=True
    )


def make_unstable_cstr_pid_controller() -> _pid_mod.Controller:
    """``make_unstable_cstr_pid`` as a ``Controller``: the same stored gains,
    the same guard and the same checks on the gain names."""
    cascade = make_unstable_cstr_pid()
    return _pid_mod.Controller(
        UnstableCSTRPIDParams(**cascade.gains, plant=cascade.params),
        UnstableCSTRPIDState(integral=0.0, T_prev=0.0, T_sp_prev=0.0, started=False),
        unstable_cstr_pid_step,
    )


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
