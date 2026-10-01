"""Every derived number in unstable_cstr's PHYSICS.md, from the env's own code.

Run with ``uv run python scripts/unstable_cstr_numbers.py [--section NAME ...]``.
With no flags it prints the fast sections, in under two minutes. The heavy
tables have their own flags:

    --sensitivity      jacket-lag sensitivity (PHYSICS.md section 6)
    --v1               the version-1 reward and its exploit (section 8)
    --pid-validation   the cascade's 300-episode validation (section 7,
                       test_pid_holds_every_target runs 130 of them)
    --mpc-gate         the MPC's 110-episode gate, 100 random schedules and
                       10 alternating (section 7); --workers N runs its
                       episodes in N processes

Every number goes through the task's own code: the closed forms,
``compute_velocity``, ``compute_next_state``, ``step_env``,
``point_of_no_return`` and ``cascade_step(xp=jnp)`` under ``vmap`` and
``scan``. No dynamics are restated, so the numbers describe the env as it
ships. Roots, folds, eigenvalues and the point of no return are computed in
float64 (``jax.experimental.enable_x64``); every closed-loop table runs in
float32, as the env does. Each table says which. The env cannot take a zero
jacket lag (``tau_j`` divides), so the 0 s rows set the jacket to the command
at the start of each step and make ``tau_j`` infinite, which holds it there
through the step. That is the lag-free plant, stepped by the same
``compute_next_state``.
"""

import argparse
import concurrent.futures as cf
import functools
import time
import warnings

# Ahead of the imports below: importing target_gym pulls in do-mpc, which warns
# about optional features this package does not use.
warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import enable_x64  # noqa: E402
from scipy.linalg import expm  # noqa: E402
from scipy.optimize import brentq, linprog  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import target_gym.pc_gym.cstr.env as cstr_env  # noqa: E402
from target_gym import registry  # noqa: E402
from target_gym.experts.mpc import plan_params  # noqa: E402
from target_gym.pc_gym.unstable_cstr.env import (  # noqa: E402
    N_BLOCKS,
    UnstableCSTRParams,
    UnstableCSTRState,
    check_is_terminal,
    compute_next_state,
    compute_velocity,
    get_obs,
    live_target,
    steady_coolant,
    steady_temperature,
)
from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR  # noqa: E402
from target_gym.pc_gym.unstable_cstr.experts import (  # noqa: E402
    DEFAULT_GAINS,
    FALLBACK_STEP_BOUND_K,
    GUARD_LEVELS,
    GUARD_MIN_DECAY,
    HELD_GAINS,
    MPC_HORIZON,
    MPC_PNR_MARGIN_K,
    PNR_LINE,
    PNR_LINE_C_A,
    TUNED_GAINS,
    CascadeState,
    _closed_loop_equilibrium,
    cascade_init,
    cascade_step,
    closed_loop_rates,
    coolant_from_raw,
    load_gains,
    pnr_line,
    point_of_no_return_batch,
    raw_from_coolant,
    setpoint_clip_margins,
    setpoint_window,
    trips_under_full_cooling,
)
from target_gym.utils import convert_raw_action_to_range  # noqa: E402

P = UnstableCSTRParams()
ENV = UnstableCSTR()
KEY = jax.random.PRNGKey(0)
#: The five targets every table uses, spanning the band (mol/L).
TARGETS = GUARD_LEVELS
#: Feed drifts the tables use (K): 0 and 3 standard deviations either way.
DRIFTS = (0.0, -6.0, 6.0)
#: Seeds the shipped tuner scores on (scripts/tune_pid.py, TUNERS row).
TUNE_SEEDS = 24
#: The shipped tuner's search (scripts/tune_pid.py, _SEARCH_FACTORS).
SEARCH_FACTORS = (0.5, 0.7, 1.4, 2.0)
#: Water's heat capacity (J/(g K), read), for deviation D1.
C_WATER = 4.184
#: This script's hold window, which its tables score holds on: minutes 6 to
#: 8.4 of each block, 120 <= block_clock % 200 < 168. It ends where the MPC's
#: horizon reaches the next switch, so no step in it can be the MPC moving
#: toward the next level. The floor comes from scripts/measure_hold.py, which
#: finds those steps from the errors instead (target_gym.eval.anticipations).
HOLD_STEPS = 80


def in_hold(block_clock):
    """Whether a state at this block clock is in this script's hold window."""
    pos = np.asarray(block_clock) % P.block_steps
    return (pos >= P.block_steps - HOLD_STEPS) & (pos < P.block_steps - MPC_HORIZON)


def say(s=""):
    print(s, flush=True)


def title(name, precision):
    say("=" * 78)
    say(f"{name}   [{precision}]")


def _lag_params(lag_s, params=P):
    """Params at a jacket lag of ``lag_s`` seconds, and whether it is the
    lag-free plant (module docstring)."""
    if lag_s == 0:
        return params.replace(tau_j=jnp.inf), True
    return params.replace(tau_j=lag_s / 60.0), False


def _states(C_a, T, T_j, Ti_dev, levels=None, dtype=None):
    """A batch of states from broadcastable arrays; ``levels`` defaults to
    ``C_a`` held for the whole schedule."""
    C_a, T, T_j, Ti_dev = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, float)) for x in (C_a, T, T_j, Ti_dev))
    )
    n = C_a.size
    if levels is None:
        levels = np.repeat(C_a[:, None], N_BLOCKS, axis=1)
    levels = np.broadcast_to(np.asarray(levels, float), (n, N_BLOCKS))
    dtype = dtype or jnp.zeros(()).dtype  # float64 inside enable_x64
    f = functools.partial(jnp.asarray, dtype=dtype)
    return UnstableCSTRState(
        time=jnp.zeros(n, jnp.int32),
        C_a=f(C_a),
        T=f(T),
        T_j=f(T_j),
        Ti_dev=f(Ti_dev),
        target_levels=f(levels),
        block_clock=jnp.zeros(n, jnp.int32),
    )


def _equilibrium_states(levels, Ti_dev, params=P, dtype=None):
    """The plant resting on each level: (L, T*(L), Tc*(L, dTi))."""
    L = np.asarray(levels, float)
    return _states(
        L,
        steady_temperature(L, params, np),
        steady_coolant(L, Ti_dev, params, np),
        Ti_dev,
        dtype=dtype,
    )


def _jacobian(C_a, T, T_j, Ti_dev=0.0, params=P):
    """d(velocity)/d(C_a, T, T_j) of the env's ``compute_velocity``, float64.
    Batched over the leading axis of the inputs."""
    x = np.stack(np.broadcast_arrays(*(np.atleast_1d(v) for v in (C_a, T, T_j))), -1)
    d = np.broadcast_to(np.asarray(Ti_dev, float), x.shape[:1])
    with enable_x64():
        # The command is held at the jacket temperature: a state at rest.
        J = jax.vmap(
            jax.jacfwd(lambda y, u, dti: compute_velocity(y, u, dti, params)[0])
        )(*(jnp.asarray(v, jnp.float64) for v in (x, x[:, 2], d)))
    return np.asarray(J)


def _eig(J):
    """Eigenvalues sorted by real part."""
    w = np.linalg.eigvals(J)
    return np.take_along_axis(w, np.argsort(w.real, axis=-1), -1)


def _roots(T_c, Ti_dev=0.0, params=P):
    """Every steady state at jacket temperature ``T_c``, as levels L with
    T = T*(L): the roots of Tc*(L, dTi) = T_c."""
    grid = np.linspace(1e-4, params.Caf - 1e-4, 20001)
    f = steady_coolant(grid, Ti_dev, params, np) - T_c
    return [
        brentq(
            lambda L: steady_coolant(L, Ti_dev, params, np) - T_c,
            grid[i],
            grid[i + 1],
            xtol=1e-15,
            rtol=1e-15,
        )
        for i in np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
    ]


def _low_state(T_j, Ti_dev=0.0, params=P):
    """The extinguished steady state at jacket temperature ``T_j`` and drift
    ``Ti_dev``, as a level: the root of Tc*(L, dTi) = T_j between the
    ignition fold and Caf."""
    return brentq(
        lambda L: steady_coolant(L, Ti_dev, params, np) - T_j,
        _folds(params)[0][0],
        params.Caf - 1e-9,
        xtol=1e-14,
    )


def _kind(J2):
    tr, det = np.trace(J2), np.linalg.det(J2)
    if det < 0:
        return "saddle"
    shape = "focus" if tr**2 < 4 * det else "node"
    return f"{'stable' if tr < 0 else 'unstable'} {shape}"


@functools.lru_cache(maxsize=None)
def _folds(params=P):
    """The two folds of the steady-state curve, as levels: extrema of Tc*(L).
    Returns ((L, T, Tc) ignition, (L, T, Tc) extinction)."""
    with enable_x64():
        slope = jax.jit(jax.grad(lambda L: steady_coolant(L, 0.0, params, jnp)))
        grid = np.linspace(0.01, 0.99, 981)
        g = np.array([float(slope(L)) for L in grid])
        roots = [
            brentq(lambda L: float(slope(L)), grid[i], grid[i + 1], xtol=1e-14)
            for i in np.flatnonzero(np.sign(g[:-1]) != np.sign(g[1:]))
        ]
    out = [
        (
            L,
            float(steady_temperature(L, params, np)),
            float(steady_coolant(L, 0.0, params, np)),
        )
        for L in roots
    ]
    return tuple(sorted(out, key=lambda r: r[1]))


@functools.partial(jax.jit, static_argnames=("n",))
def _hold_action(states, raw, params, n):
    """``n`` steps of ``compute_next_state`` from each state at a constant raw
    action, the drift held and no trip handling. Returns the C_a, T and T_j
    paths and the trip check of each proposal, each shaped (batch, n)."""

    def one(state, u):
        def body(s, _):
            s2 = compute_next_state(u, s, params, KEY)[0].replace(Ti_dev=s.Ti_dev)
            return s2, (s2.C_a, s2.T, s2.T_j, check_is_terminal(s2, params)[0])

        return jax.lax.scan(body, state, None, length=n)[1]

    return jax.vmap(one)(states, raw)


def _first(mask, axis=-1):
    """Index of the first True along ``axis`` (1-based step count), or 0."""
    return np.where(mask.any(axis), np.argmax(mask, axis) + 1, 0)


@functools.partial(jax.jit, static_argnames=("iters",))
def _pnr_along_jit(C_a, T, dC_a, T_j, Ti_dev, params, iters=24):
    def bisect(_, bracket):
        lo, hi = bracket
        mid = 0.5 * (lo + hi)
        trips = trips_under_full_cooling(C_a + mid * dC_a, T + mid, T_j, Ti_dev, params)
        return jnp.where(trips, lo, mid), jnp.where(trips, mid, hi)

    one = jnp.ones_like(T)
    return jax.lax.fori_loop(0, iters, bisect, (0.0 * one, 40.0 * one))[1]


def _pnr_along(C_a, T, dC_a, T_j, Ti_dev, params=P):
    """The smallest offset s (K of T) such that full cooling from
    (C_a + s dC_a, T + s, T_j) still trips: the point of no return along a
    direction, float64."""
    args = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, float)) for x in (C_a, T, dC_a, T_j, Ti_dev))
    )
    with enable_x64():
        f = jax.vmap(_pnr_along_jit, in_axes=(0, 0, 0, 0, 0, None))
        return np.asarray(f(*(jnp.asarray(x, jnp.float64) for x in args), params))


def _unstable_direction(L, params=P):
    """dC_a/dT along the unstable eigenvector at target L (the jacket
    component is zero: the lag row is (0, 0, -1/tau_j))."""
    J2 = _jacobian(
        L,
        steady_temperature(L, params, np),
        steady_coolant(L, 0.0, params, np),
        params=params,
    )[0, :2, :2]
    w, v = np.linalg.eig(J2)
    vec = v[:, np.argmax(w.real)].real
    return vec[0] / vec[1]


# ---------------------------------------------------------------------------
# Closed-loop machinery (float32, as the env ships)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _runner(
    n_steps,
    cascade=True,
    hold_drift=False,
    lag0=False,
    record=False,
    sp_at_T_star=False,
    integration_method="rk4_1",
):
    """A jitted batch of episodes on the env's own ``step_env`` (trips and
    fresh restarts included), each with its own start state, rollout key and
    controller: cascade gains, or a constant raw action ``{"raw": u}``.

    Returns running sums per episode (version-1 and version-2 rewards, trips,
    steps on the extinguished branch, steps at a coolant bound, |e| on
    untripped steps) and, with ``record``, per-step arrays (batch, n_steps):
    C_a, T, T_j, Ti_dev, e = target - C_a, tripped, block clock, raw action.
    ``sp_at_T_star`` starts the setpoint at T*(target) instead of the measured
    T that ``cascade_init`` uses, for comparison.
    """
    env = UnstableCSTR(integration_method=integration_method)

    def one(state, key, ctrl, params, ext_T):
        run_params = params.replace(tau_j=jnp.inf) if lag0 else params
        v1_params = run_params.replace(reward_version=1)
        obs0 = get_obs(state, params)
        if cascade:
            cs = cascade_init(obs0, ctrl, params, xp=jnp)
            if sp_at_T_star:
                cs = cs._replace(T_sp_prev=steady_temperature(obs0[3], params, jnp))
        else:
            cs = CascadeState(obs0[1] * 0.0, obs0[1] * 0.0, obs0[1] * 0.0)
        drift = state.Ti_dev
        zero = jnp.zeros((), jnp.float32)
        count = jnp.zeros((), jnp.int32)
        acc0 = dict(v1=zero, v2=zero, trips=count, ext=count, sat=count, abs_e=zero)

        def body(carry, _):
            s, cs, acc = carry
            obs = get_obs(s, params)
            if cascade:
                u, cs = cascade_step(ctrl, cs, obs, params, xp=jnp)
            else:
                u = ctrl["raw"]
            if lag0:
                s = s.replace(
                    T_j=convert_raw_action_to_range(u, params.T_c_min, params.T_c_max)
                )
            _, s2, r2, _, info = env.step_env(key, s, u, run_params)
            r1 = env.step_env(key, s, u, v1_params)[2]
            if hold_drift:
                s2 = s2.replace(Ti_dev=drift)
            tripped = info["tripped"]
            ok = jnp.logical_not(tripped)
            e = live_target(s2, params) - s2.C_a
            acc = dict(
                v1=acc["v1"] + r1,
                v2=acc["v2"] + r2,
                trips=acc["trips"] + tripped.astype(jnp.int32),
                ext=acc["ext"] + (ok & (s2.T < ext_T)).astype(jnp.int32),
                sat=acc["sat"] + (jnp.abs(u) >= 1.0).astype(jnp.int32),
                abs_e=acc["abs_e"] + jnp.where(ok, jnp.abs(e), 0.0),
            )
            out = None
            if record:
                out = (s2.C_a, s2.T, s2.T_j, s2.Ti_dev, e, tripped, s2.block_clock, u)
            return (s2, cs, acc), out

        (_, _, acc), out = jax.lax.scan(body, (state, cs, acc0), None, length=n_steps)
        return acc, out

    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None, None)))


REC_FIELDS = ("C_a", "T", "T_j", "Ti_dev", "e", "tripped", "block_clock", "u")


def run_episodes(states, keys, ctrl, params=P, n_steps=None, record=False, **options):
    """Episodes on the env: see ``_runner``. ``ctrl`` is a gains dict or
    ``{"raw": u}``, scalars or arrays over the batch. Returns (summary,
    record), numpy arrays."""
    n = int(np.asarray(states.C_a).shape[0])
    n_steps = int(params.max_steps_in_episode) if n_steps is None else n_steps
    cascade = "raw" not in ctrl
    names = DEFAULT_GAINS if cascade else ("raw",)
    ctrl = {k: jnp.broadcast_to(jnp.asarray(ctrl[k], jnp.float32), (n,)) for k in names}
    ext_T = jnp.float32(_folds()[0][1])
    acc, out = _runner(n_steps, cascade=cascade, record=record, **options)(
        states, keys, ctrl, params, ext_T
    )
    summary = {k: np.asarray(v) for k, v in acc.items()}
    rec = None if out is None else dict(zip(REC_FIELDS, (np.asarray(x) for x in out)))
    return summary, rec


def reset_batch(seeds, params=P):
    """``reset_env`` for each seed, with the seed's key as the rollout key
    (as ``runners.rollout`` does)."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, jnp.uint32))
    states = jax.vmap(lambda k: ENV.reset_env(k, params)[1])(keys)
    return states, keys


def recentre(states, level, params=P, Ti_dev=None):
    """The reset draws moved onto a constant ``level`` schedule (or a given
    one), keeping each draw's offsets from its own first level."""
    n = states.C_a.shape[0]
    levels = np.broadcast_to(np.asarray(level, float), (n, N_BLOCKS))
    first = np.asarray(states.target_levels)[:, 0]
    L0 = levels[:, 0]
    d = (
        np.asarray(states.Ti_dev)
        if Ti_dev is None
        else np.full(L0.shape, float(Ti_dev))
    )
    return _states(
        L0 + (np.asarray(states.C_a) - first),
        steady_temperature(L0, params, np)
        + (np.asarray(states.T) - steady_temperature(first, params, np)),
        steady_coolant(L0, d, params, np),
        d,
        levels=levels,
        dtype=jnp.float32,
    )


def episode_margins(rec, params=P, k=8, lag0=False):
    """Per episode, the smallest exact margin to the point of no return,
    PNR(C_a, T_j, Ti_dev) - T, over untripped steps: the ``k`` steps closest
    to the line are refined by bisection (float64). Also returns the
    smallest line margin, pnr_line(C_a) - T. At 0 s lag the jacket is at the
    full-cooling command at once."""
    line = np.where(rec["tripped"], np.inf, pnr_line(rec["C_a"]) - rec["T"])
    idx = np.argsort(line, axis=1)[:, :k]
    rows = np.arange(line.shape[0])[:, None]
    pick = {f: rec[f][rows, idx].ravel() for f in ("C_a", "T", "T_j", "Ti_dev")}
    T_j = np.full(pick["T_j"].shape, params.T_c_min) if lag0 else pick["T_j"]
    exact = (
        point_of_no_return_batch(pick["C_a"], T_j, pick["Ti_dev"], params) - pick["T"]
    )
    exact = np.where(np.isfinite(line[rows, idx].ravel()), exact, np.inf)
    return exact.reshape(idx.shape).min(axis=1), line.min(axis=1)


def summarize(summary, n_steps):
    trips = summary["trips"]
    return dict(
        trip_share=float((trips > 0).mean()),
        trips=float(trips.mean()),
        ext_share=float((summary["ext"] > 0).mean()),
        v1=float(summary["v1"].mean() / n_steps),
        v2=float(-summary["v2"].mean() / n_steps),
        sat=float(summary["sat"].mean() / n_steps),
    )


def _gains_str(g):
    return ", ".join(f"{k} {g[k]:.4g}" for k in TUNED_GAINS)


# ---------------------------------------------------------------------------
# Fast sections
# ---------------------------------------------------------------------------


def section_params():
    title("params: the ten values, derived groups, restart, drift, envelope", "float64")
    shipped = cstr_env.CSTRParams()
    names = ("q", "V", "rho", "C", "deltaHr", "EA_over_R", "k0", "UA", "Ti", "Caf")
    say(
        "  read from cstr's CSTRParams: "
        + ", ".join(f"{n} {getattr(shipped, n):g}" for n in names)
    )
    same = all(
        getattr(P, n) == getattr(shipped, n) for n in (*names, "precision_floor")
    )
    say(f"  UnstableCSTRParams carries the same ten and precision_floor: {same}")
    qV = P.q / P.V
    A = -P.deltaHr / (P.rho * P.C)
    beta = P.UA / (P.rho * P.C * P.V)
    gamma = P.EA_over_R / P.Ti
    say(
        f"  A = -deltaHr/(rho C) = {A:.3f} K L/mol; adiabatic rise A Caf = "
        f"{A * P.Caf:.3f} K"
    )
    say(
        f"  beta = UA/(rho C V) = {beta:.5f} /min; q/V = {qV:g} /min; residence time "
        f"V/q = {P.V / P.q:.2f} min"
    )
    say(
        "  Uppal-Ray-Poore groups: gamma = EA_over_R/Ti = "
        f"{gamma:.3f}, B = gamma A Caf/Ti = {gamma * A * P.Caf / P.Ti:.3f}, "
        f"beta = UA/(q rho C) = {P.UA / (P.q * P.rho * P.C):.4f}, "
        f"Da = k0 exp(-gamma) V/q = {P.k0 * np.exp(-gamma) * P.V / P.q:.5f}"
    )
    restart = shipped.restart_steps * shipped.delta_t / P.delta_t
    say(
        f"  restart: cstr's {shipped.restart_steps} steps x {shipped.delta_t} min = "
        f"{restart:.0f} steps at {P.delta_t * 60:.0f} s; params.restart_steps = "
        f"{P.restart_steps}"
    )
    a = np.exp(-P.delta_t / P.Ti_tau)
    say(
        f"  drift: a = exp(-dt/Ti_tau) = {a:.5f}; innovation sd = "
        f"{P.Ti_sigma * np.sqrt(1 - a**2):.4f} K "
        f"per step; reset clip +-{P.Ti_dev_reset_clip * P.Ti_sigma:.1f} K"
    )
    L_trip = brentq(lambda L: steady_temperature(L, P, np) - P.T_trip, 1e-6, 0.99)
    lo, hi = P.target_CA_range
    env_max = max(P.Caf - lo, hi - L_trip)
    say(
        f"  CA_error_max = max(Caf - {lo}, {hi} - C_a at {P.T_trip:.0f} K on the "
        "steady curve "
        f"({L_trip:.4f})) = max({P.Caf - lo:.3f}, {hi - L_trip:.3f}) = {env_max:.3f}; "
        f"params: {P.CA_error_max}"
    )
    fc = 2.0 * (P.CA_error_max / P.e_floor) ** 2
    say(
        f"  failure_cost = 2 (CA_error_max/e_floor)^2 = {fc:.4g} (params "
        f"{P.failure_cost:.4g}); "
        f"trip cost = restart_steps x failure_cost = {P.restart_steps * fc:.4g}"
    )
    say(
        f"  episode: {P.max_steps_in_episode} steps = {N_BLOCKS} x {P.block_steps} = "
        f"{P.max_steps_in_episode * P.delta_t:.0f} min"
    )


def section_steady():
    title("steady: roots, types, folds, Hopf, limit cycle, anchors", "float64")
    ign, ext = _folds()
    say(
        f"  folds (jacket held): ignition T {ign[1]:.3f} K at Tc {ign[2]:.3f} K (C_a "
        f"{ign[0]:.4f}); "
        f"extinction T {ext[1]:.3f} K at Tc {ext[2]:.3f} K (C_a {ext[0]:.4f})"
    )
    say(f"  three steady states for Tc in ({ext[2]:.3f}, {ign[2]:.3f}) K")
    say("  Tc (K)  states: C_a (mol/L), T (K), type, trace (/min)")
    for T_c in (295.0, 300.0, 302.0, 305.0, 310.0):
        rows = []
        for L in _roots(T_c):
            T = float(steady_temperature(L, P, np))
            J2 = _jacobian(L, T, T_c)[0, :2, :2]
            rows.append(f"({L:.6f}, {T:.4f}, {_kind(J2)}, {np.trace(J2):+.3f})")
        say(f"  {T_c:5.1f}   " + "; ".join(rows))

    def trace_at(L):
        return np.trace(
            _jacobian(L, steady_temperature(L, P, np), steady_coolant(L, 0.0, P, np))[
                0, :2, :2
            ]
        )

    grid = np.linspace(0.005, ext[0] - 1e-4, 400)
    tr = np.array([trace_at(L) for L in grid])
    i = np.flatnonzero(np.sign(tr[:-1]) != np.sign(tr[1:]))[0]
    L_h = brentq(trace_at, grid[i], grid[i + 1], xtol=1e-12)
    say(
        f"  Hopf on the upper branch: Tc {float(steady_coolant(L_h, 0.0, P, np)):.3f} "
        "K, "
        f"T {float(steady_temperature(L_h, P, np)):.3f} K, C_a {L_h:.5f}"
    )

    L_up = min(_roots(305.0))
    T_up = float(steady_temperature(L_up, P, np))
    with enable_x64():
        paths = _hold_action(
            _states(L_up, T_up + 0.5, 305.0, 0.0),
            jnp.full(1, raw_from_coolant(305.0, P)),
            P,
            6000,
        )
    T_path = np.asarray(paths[1])[0, 2000:]
    say(
        f"  at a 305 K coolant with the trip disabled (compute_next_state, 300 min): "
        f"T cycles between {T_path.min():.1f} and {T_path.max():.1f} K"
    )

    L_mid = sorted(_roots(300.0))[1]
    T_mid = float(steady_temperature(L_mid, P, np))
    J3 = _jacobian(L_mid, T_mid, 300.0)[0]
    say(
        f"  nominal middle state at 300 K: C_a {L_mid:.6f}, T {T_mid:.4f} K; "
        "eigenvalues "
        f"{', '.join(f'{w.real:+.4f}' for w in _eig(J3[:2, :2]))} /min without the "
        "lag, "
        f"{', '.join(f'{w.real:+.4f}' for w in _eig(J3))} with it"
    )
    with enable_x64():
        v = compute_velocity(jnp.array([0.5, 350.0, 300.0]), 300.0, 0.0, P)[0]
    say(
        f"  textbook point (0.5, 350 K) at 300 K: dC_a/dt {float(v[0]):+.3e} mol/(L "
        f"min), dT/dt {float(v[1]):+.3e} K/min"
    )

    L_low = max(_roots(300.0))
    say(
        f"  APMonitor low state at 300 K (read 0.87725294608097, 324.475443431599): "
        f"{L_low:.14f}, {float(steady_temperature(L_low, P, np)):.12f}"
    )
    for T_c, C_pub, T_pub in ((299.709, 0.483, 350.970), (299.413, 0.465, 352.000)):
        L = sorted(_roots(T_c))[1]
        T = float(steady_temperature(L, P, np))
        det = np.linalg.det(_jacobian(L, T, T_c)[0, :2, :2])
        say(
            f"  Decardi-Nelson and Liu middle state at {T_c} K (read {C_pub}, "
            f"{T_pub}): "
            f"{L:.5f}, {T:.3f} K; det J {det:+.3f}"
        )


def section_targets():
    title("targets: per-target table, loop coefficients, headroom, signs", "float64")
    say(
        "  L     T* (K)    Tc*(0)   Tc*(-6)  Tc*(+6)  eig no lag (/min)   eig with "
        "lag (/min)          1/lam+ (s)  lam+ tau_j  det-tr/tau_j"
    )
    for L in TARGETS:
        T = float(steady_temperature(L, P, np))
        Tc = {d: float(steady_coolant(L, d, P, np)) for d in DRIFTS}
        J3 = _jacobian(L, T, Tc[0.0])[0]
        J2 = J3[:2, :2]
        e2, e3 = _eig(J2).real, _eig(J3).real
        lam = e2[-1]
        coef = np.linalg.det(J2) - np.trace(J2) / P.tau_j
        B = np.array([0.0, 0.0, 1.0 / P.tau_j])
        C = np.array([1.0, 0.0, 0.0])
        checks = []
        for k in (10.0, 100.0):
            p_loop = np.poly(J3 - k * np.outer(B, C))
            aug = np.zeros((4, 4))
            aug[:3, :3] = J3 - k * np.outer(B, C)
            aug[:3, 3] = -k * B
            aug[3, :3] = C
            checks += [p_loop[2], np.poly(aug)[2]]
        assert np.allclose(checks, coef, rtol=1e-6, atol=1e-6), (checks, coef)
        say(
            f"  {L:.2f}  {T:8.3f}  {Tc[0.0]:7.3f}  {Tc[-6.0]:7.3f}  {Tc[6.0]:7.3f}  "
            f"({e2[0]:+.3f}, {e2[1]:+.3f})    ({e3[0]:+.3f}, {e3[1]:+.3f}, "
            f"{e3[2]:+.3f})    "
            f"{60 / lam:6.1f}      {lam * P.tau_j:.3f}      {coef:+.2f}"
        )
    say(
        "  det - tr/tau_j is the s coefficient of every P loop on C_a and the s^2 "
        "coefficient of every"
    )
    say(
        "  PI loop on C_a (checked on the closed-loop polynomials at two gains): "
        "negative, so neither stabilises"
    )
    band = np.linspace(*P.target_CA_range, 2001)
    for d in (0.0, 6.0, -6.0):
        Tc = steady_coolant(band, d, P, np)
        room = np.minimum(Tc - P.T_c_min, P.T_c_max - Tc).min()
        say(
            f"  dTi {d:+.0f} K: Tc* {Tc.min():.3f} to {Tc.max():.3f} K; coolant "
            f"headroom {room:.3f} K"
        )
    with enable_x64():
        grad = jax.vmap(jax.grad(lambda L: steady_coolant(L, 0.0, P, jnp)))(
            jnp.asarray(band)
        )
        states = _equilibrium_states(np.array(TARGETS), 0.0, dtype=jnp.float64)
        raw = jnp.asarray(raw_from_coolant(np.asarray(states.T_j) + 1.0, P))
        after = _hold_action(states, raw, P, 2)[0]
    say(
        f"  static sign: dTc*/dL over the band {float(grad.min()):+.2f} to "
        f"{float(grad.max()):+.2f} K per mol/L"
    )
    dC = np.asarray(after) - np.asarray(TARGETS)[:, None]
    say(
        "  short-term sign: C_a change 1 and 2 steps after a +1 K coolant step at "
        "rest: "
        + ", ".join(f"{L:.2f}: {a:+.1e}, {b:+.1e}" for L, (a, b) in zip(TARGETS, dC))
    )


def section_pnr():
    title(
        "pnr: point of no return, reset corners, line refit, boundary to peak",
        "float64",
    )
    L = np.array(TARGETS)
    T_star = steady_temperature(L, P, np)
    dCa_dT = np.array([_unstable_direction(x) for x in L])
    say(
        "  pure T: C_a = L, jacket at Tc*(L, dTi) (the command at 0 s). Eigenvector: "
        "along the unstable"
    )
    say(
        "  direction, dC_a/dT = "
        + ", ".join(f"{x:+.4f}" for x in dCa_dT)
        + f" mol/(L K). Corner: (L + {P.initial_CA_offset:g}, T* + "
        + f"{P.initial_T_offset:g} K)."
    )
    say("  L     lag   dTi   T*       T_pnr     PNR-T*   along eigvec   corner margin")
    for lag in (6, 0):
        params, lag0 = _lag_params(lag)
        for d in DRIFTS:
            T_j = np.full(L.shape, P.T_c_min) if lag0 else steady_coolant(L, d, P, np)
            pnr = point_of_no_return_batch(L, T_j, d, params)
            along = _pnr_along(L, T_star, dCa_dT, T_j, d, params)
            corner = point_of_no_return_batch(
                L + P.initial_CA_offset, T_j, d, params
            ) - (T_star + P.initial_T_offset)
            for i in range(L.size):
                say(
                    f"  {L[i]:.2f}  {lag:2d} s  {d:+3.0f}  {T_star[i]:7.2f}  "
                    f"{pnr[i]:8.3f}   {pnr[i] - T_star[i]:6.3f}   "
                    f"{along[i]:8.3f}       {corner[i]:+7.3f}"
                )
    Tj45 = float(steady_coolant(0.45, 0.0, P, np))
    ca = np.array([0.45, 0.47, 0.50, 0.55])
    say(
        "  PNR with C_a off its target (jacket at Tc*(0.45), dTi 0, 6 s): "
        + ", ".join(
            f"C_a {c:.2f}: {p:.2f} K"
            for c, p in zip(ca, point_of_no_return_batch(ca, Tj45, 0.0, P))
        )
    )

    lo_band = P.target_CA_range[0]
    T0 = float(steady_temperature(lo_band, P, np)) + P.initial_T_offset

    def corner(d):
        return (
            float(
                point_of_no_return_batch(
                    lo_band + P.initial_CA_offset,
                    steady_coolant(lo_band, d, P, np),
                    d,
                    P,
                )[0]
            )
            - T0
        )

    cross = brentq(corner, 0.0, 15.0, xtol=1e-4)
    say(
        f"  the worst reset corner (C_a {lo_band + P.initial_CA_offset:.2f}, "
        f"T*({lo_band:.2f}) + {P.initial_T_offset:g} K) passes the PNR at "
        f"dTi {cross:+.2f} K ({cross / P.Ti_sigma:.2f} sd); the reset clip is "
        f"{P.Ti_dev_reset_clip * P.Ti_sigma:.1f} K"
    )

    # The line: the supporting line from below of the +6 K curve.
    grid = np.round(np.arange(0.40, 0.70 + 1e-9, 0.005), 3)
    curves = {
        d: point_of_no_return_batch(grid, steady_coolant(grid, d, P, np), d, P)
        for d in DRIFTS
    }
    x = grid - PNR_LINE_C_A
    fit = linprog(
        c=[-x.size, -x.sum()],
        A_ub=np.stack([np.ones_like(x), x], 1),
        b_ub=curves[6.0],
        bounds=[(None, None)] * 2,
        method="highs",
    ).x
    slope = round(float(fit[1]), 1)
    intercept = np.floor(np.min(curves[6.0] - slope * x) * 10.0) / 10.0
    say(
        f"  line refit at dTi +6 K on C_a 0.40 to 0.70 (0.005 grid): supporting line "
        f"{fit[0]:.3f} {fit[1]:+.3f} (C_a - {PNR_LINE_C_A}); slope rounded, intercept "
        "to the 0.1 K below: "
        f"({intercept:.1f}, {slope:.1f})"
    )
    say(
        f"  frozen PNR_LINE {PNR_LINE}: "
        + (
            "equals the refit"
            if PNR_LINE == (intercept, slope)
            else "DIFFERS from the refit"
        )
    )
    on_targets = np.isin(grid, L)
    for d in DRIFTS:
        gap = curves[d] - pnr_line(grid)
        say(
            f"  dTi {d:+3.0f}: PNR - line, smallest {gap.min():.3f} K at C_a "
            f"{grid[np.argmin(gap)]:.3f}; at the targets "
            + ", ".join(f"{g:.3f}" for g in gap[on_targets])
        )
    band = np.linspace(*P.target_CA_range, 2001)
    T_b = steady_temperature(band, P, np)
    headroom = pnr_line(band) - MPC_PNR_MARGIN_K - T_b
    say(
        f"  MPC bound pnr_line(L) - {MPC_PNR_MARGIN_K:g} K is {headroom.min():.3f} K "
        f"above T*(L) at worst (L {band[np.argmin(headroom)]:.3f})"
    )
    ceiling = setpoint_window(band, DEFAULT_GAINS, P)[1]
    binds = band[ceiling < T_b + DEFAULT_GAINS["sp_above"] - 1e-12]
    say(
        f"  PID ceiling at 0.45: {ceiling[0]:.3f} K = T* + {ceiling[0] - T_b[0]:.3f} "
        "K; the line binds for L up to "
        f"{binds.max() if binds.size else float('nan'):.3f}"
    )

    # Boundary to peak: from just inside the PNR, how long full cooling takes
    # to turn the excursion.
    say(
        "  boundary to peak (start 0.01 K below the PNR, full cooling, 6 s), against "
        f"the {MPC_HORIZON}-step horizon:"
    )
    for d in (0.0, 6.0):
        T_j = steady_coolant(L, d, P, np)
        start = point_of_no_return_batch(L, T_j, d, P) - 0.01
        with enable_x64():
            T_path = np.asarray(
                _hold_action(_states(L, start, T_j, d), jnp.full(L.size, -1.0), P, 200)[
                    1
                ]
            )
        peak = np.argmax(T_path, axis=1) + 1
        say(
            f"    dTi {d:+.0f}: "
            + ", ".join(
                f"{x:.2f}: {k} steps ({k * P.delta_t:.2f} min, peak "
                f"{T_path[i].max():.2f} K)"
                for i, (x, k) in enumerate(zip(L, peak))
            )
            + f"; horizon / longest = {MPC_HORIZON / peak.max():.2f}"
        )


def _corners():
    """135 starts: five targets x the reset box's 3 x 3 grid x dTi -6, 0, +6."""
    rows = [
        (
            L + a * P.initial_CA_offset,
            float(steady_temperature(L, P, np)) + b * P.initial_T_offset,
            float(steady_coolant(L, d, P, np)),
            d,
        )
        for L in TARGETS
        for a in (-1, 0, 1)
        for b in (-1, 0, 1)
        for d in (-6.0, 0.0, 6.0)
    ]
    return np.array(rows).T


def section_reach():
    title(
        "reach: full heating and cooling from the reset box, zero action, extinction",
        "float32",
    )
    C_a, T, T_j, d = _corners()
    states = _states(C_a, T, T_j, d, dtype=jnp.float32)
    trips = np.asarray(_hold_action(states, jnp.full(C_a.size, 1.0), P, 60)[3])
    first = _first(trips)
    say(
        f"  full heating from {C_a.size} corners: trips {int((first > 0).sum())}; "
        f"steps to trip {first.min()} to "
        f"{first.max()} (median {np.median(first):.0f}), "
        f"{first.min() * P.delta_t * 60:.0f} to {first.max() * P.delta_t * 60:.0f} s"
    )
    per_target = first.reshape(len(TARGETS), -1)
    say(
        "    by target: "
        + ", ".join(
            f"{L:.2f}: {k.min()} to {k.max()}" for L, k in zip(TARGETS, per_target)
        )
        + " steps"
    )
    cool = _hold_action(states, jnp.full(C_a.size, -1.0), P, 1200)
    say(
        "  full cooling for 1200 steps: trips "
        f"{int(np.asarray(cool[3]).any(1).sum())}; peak T "
        f"{float(np.asarray(cool[1]).max()):.2f} K"
    )

    L = np.repeat(np.array(TARGETS), 2)
    off = np.tile([-0.5, 0.5], len(TARGETS))
    T0 = steady_temperature(L, P, np) + off
    zero = _hold_action(
        _states(L, T0, steady_coolant(L, 0.0, P, np), 0.0, dtype=jnp.float32),
        jnp.zeros(L.size),
        P,
        1200,
    )
    Tz, tz = np.asarray(zero[1]), np.asarray(zero[3])
    leave = _first(np.abs(Tz - steady_temperature(L, P, np)[:, None]) > 2.0)
    say(
        "  zero action (300 K) from T* +- 0.5 K: steps to leave +-2 K, then the "
        "outcome within 60 min"
    )
    outcome = [
        f"trip at {k}" if k else f"extinguished, T {T_end:.1f} K"
        for k, T_end in zip(_first(tz), Tz[:, -1])
    ]
    say(
        "    "
        + "; ".join(
            f"{x:.2f}{o:+.1f}: {k}, {w}" for x, o, k, w in zip(L, off, leave, outcome)
        )
    )

    L_low = max(_roots(300.0))
    heat = _hold_action(
        _states(L_low, steady_temperature(L_low, P, np), 300.0, 0.0, dtype=jnp.float32),
        jnp.ones(1),
        P,
        60,
    )
    Th = np.asarray(heat[1])[0]
    say(
        f"  from the low state at 300 K ({float(steady_temperature(L_low, P, np)):.2f} "
        "K), full heating: "
        f"340 K after {_first(Th >= 340.0)} steps, trip after "
        f"{_first(np.asarray(heat[3])[0])} steps"
    )


def section_integrator():
    title(
        "integrator: RK4 stability and accuracy",
        "float64 for eigenvalues, float32 and float64 for steps",
    )
    C_a, T, T_j, d = _corners()
    first64 = None
    # Full heating from the 135 reset-box starts, and from the extinguished
    # branch (jacket 290 to 300 K, dTi -6, 0, +6): every state before the
    # trip, where the plant can be.
    low = [
        (_low_state(tj, dd), tj, dd)
        for dd in (-6.0, 0.0, 6.0)
        for tj in np.arange(290.0, 300.5, 1.0)
    ]
    L_low, Tj_low, d_low = np.array(low).T
    starts = {
        "the 135 reset-box starts": (C_a, T, T_j, d),
        f"{L_low.size} extinguished starts": (
            L_low,
            steady_temperature(L_low, P, np),
            Tj_low,
            d_low,
        ),
    }
    for name, (c0, t0, j0, d0) in starts.items():
        with enable_x64():
            paths = _hold_action(_states(c0, t0, j0, d0), jnp.ones(c0.size), P, 80)
            Cp, Tp, Jp, tp = (np.asarray(x) for x in paths)
        first = _first(tp)
        first64 = first if first64 is None else first64
        before = np.arange(80)[None, :] < (first[:, None] - 1)
        pts = np.concatenate(
            [
                np.stack([c0, t0, j0], 1),
                np.stack([Cp[before], Tp[before], Jp[before]], 1),
            ]
        )
        lam = np.abs(np.linalg.eigvals(_jacobian(*pts.T))).max(1)
        i = int(np.argmax(lam))
        say(
            f"  largest |lambda| dt under full heating from {name}, before the trip: "
            f"{lam[i] * P.delta_t:.3f} (C_a {pts[i, 0]:.3f}, T {pts[i, 1]:.2f} K); "
            "RK4's real-axis limit is 2.785"
        )
    Cg, Tg = np.meshgrid(
        np.linspace(0.0, P.Caf, 201), np.linspace(280.0, P.T_trip, 171), indexing="ij"
    )
    Jg = _jacobian(Cg.ravel(), Tg.ravel(), 300.0)
    lg = np.abs(np.linalg.eigvals(Jg)).max(1)
    i = int(np.argmax(lg))
    say(
        f"  largest |lambda| dt anywhere with 0 <= C_a <= Caf and T <= {P.T_trip:.0f} "
        f"K: {lg[i] * P.delta_t:.3f} "
        f"(C_a {Cg.ravel()[i]:.2f}, T {Tg.ravel()[i]:.1f} K)"
    )
    first32 = _first(
        np.asarray(
            _hold_action(
                _states(C_a, T, T_j, d, dtype=jnp.float32), jnp.ones(C_a.size), P, 80
            )[3]
        )
    )
    say(
        "  steps to trip under full heating, float32 against float64: "
        f"{int((first32 != first64).sum())} of "
        f"{C_a.size} corners differ (largest difference "
        f"{int(np.abs(first32 - first64).max())} steps)"
    )

    gains = load_gains()
    states, keys = reset_batch([0])
    out = {
        method: run_episodes(
            states, keys, gains, record=True, integration_method=method
        )[1]
        for method in ("rk4_1", "rk4_16")
    }
    dT = np.abs(out["rk4_1"]["T"] - out["rk4_16"]["T"]).max()
    dC = np.abs(out["rk4_1"]["C_a"] - out["rk4_16"]["C_a"]).max()
    say(
        "  shipped cascade, seed 0, 1200 steps: rk4_1 against rk4_16, max |dT| "
        f"{dT:.2e} K, max |dC_a| {dC:.2e} mol/L"
    )
    L_low = max(_roots(300.0))
    low = _states(
        L_low, steady_temperature(L_low, P, np), 300.0, 0.0, dtype=jnp.float32
    )
    ign = {}
    for method in ("rk4_1", "rk4_16"):

        def run(s, method=method):
            def body(s, _):
                s2 = compute_next_state(
                    jnp.float32(1.0), s, P, KEY, integration_method=method
                )[0]
                return s2, (s2.T, check_is_terminal(s2, P)[0])

            return jax.lax.scan(body, s, None, length=40)[1]

        ign[method] = [np.asarray(x) for x in jax.vmap(run)(low)]
    k1, k16 = _first(ign["rk4_1"][1][0]), _first(ign["rk4_16"][1][0])
    upto = min(k1, k16) - 1
    say(
        f"  ignition under full heating from the low state at 300 K: trip at step {k1} "
        f"(rk4_1) and {k16} (rk4_16); "
        "max |dT| before it "
        f"{np.abs(ign['rk4_1'][0][0, :upto] - ign['rk4_16'][0][0, :upto]).max():.3e} K"
    )


def section_invariance():
    title(
        "invariance: drift, lag, and the conformance checks 5 and 7 replayed",
        "float32 unless stated",
    )
    worst = 0.0
    for L in TARGETS:
        T = float(steady_temperature(L, P, np))
        J0 = _jacobian(L, T, steady_coolant(L, 0.0, P, np), 0.0)[0]
        for d in (-6.0, 6.0):
            worst = max(
                worst,
                np.abs(_jacobian(L, T, steady_coolant(L, d, P, np), d)[0] - J0).max(),
            )
    say(
        "  plant Jacobian at each target's equilibrium, dTi -6 and +6 against 0: "
        f"largest difference {worst:.1e} (float64)"
    )
    for name, gains in (
        ("DEFAULT_GAINS", DEFAULT_GAINS),
        ("shipped gains", load_gains()),
    ):
        r = np.array([closed_loop_rates(gains, Ti_dev=d) for d in (-6.0, 0.0, 6.0)])
        say(
            f"  closed-loop rates, {name}: "
            + ", ".join(f"{x:+.3f}" for x in r[1])
            + " /min; largest change with the drift "
            + f"{np.abs(r - r[1]).max():.1e} (float64)"
        )

    # Check 5 (tests/test_env_conformance.py::test_regimes_join_smoothly), replayed.
    key = jax.random.PRNGKey(0)
    _, state = ENV.reset_env(key, P)
    fields = ["C_a", "T", "T_j", "Ti_dev"]
    action = jnp.zeros((1,), jnp.float32)
    worst, where = 0.0, None
    for f in fields:
        base = float(getattr(state, f))
        span = abs(base) if abs(base) > 1e-6 else 1.0
        xs = jnp.linspace(base - 0.75 * span, base + 0.75 * span, 121)
        eps = 1e-3 * span

        def one(x, _f=f):
            s2 = ENV.step_env(key, state.replace(**{_f: x}), action, P)[1]
            return jnp.stack([getattr(s2, g) for g in fields])

        jac = jax.jit(jax.vmap(jax.jacfwd(one)))
        Jm, Jp = jac(xs - eps), jac(xs + eps)
        num, den = jnp.abs(Jp - Jm), jnp.abs(Jp) + jnp.abs(Jm)
        peak = jnp.max(den, axis=0, keepdims=True)
        both = jnp.minimum(jnp.abs(Jp), jnp.abs(Jm)) > 0.05 * jnp.maximum(peak, 1e-30)
        jump = np.nan_to_num(
            np.asarray(jnp.where(both & (den > 0), num / jnp.maximum(den, 1e-30), 0.0))
        )
        if jump.max() > worst:
            worst = float(jump.max())
            to = fields[int(np.argmax(jump.max(axis=0)))]
            at = float(xs[int(np.argmax(jump.max(axis=1)))])
            where = f"{f} -> {to} at {f} = {at:.4g}"
    say(
        f"  check 5: largest one-sided Jacobian jump {worst:.3f} ({where}); the limit "
        "is 0.5"
    )

    # Check 7 (test_plant_does_not_accelerate_without_input), replayed.
    key = jax.random.PRNGKey(0)
    _, state = ENV.reset_env(key, P)
    step = jax.jit(ENV.step_env)
    traj, trips = [[float(getattr(state, f)) for f in fields]], 0
    for _ in range(min(int(P.max_steps_in_episode), 600)):
        key, sub = jax.random.split(key)
        _, state, _, _, info = step(sub, state, action, P)
        trips += int(info["tripped"])
        traj.append([float(getattr(state, f)) for f in fields])
    tr = np.array(traj)
    inc = np.abs(np.diff(tr, axis=0))
    fifth = max(len(inc) // 5, 1)
    early, late = inc[:fifth].mean(0), inc[-fifth:].mean(0)
    noise = 32.0 * np.finfo(np.float32).eps * np.maximum(np.abs(tr).mean(0), 1.0)
    ratio = np.where(
        early > np.maximum(noise, 1e-12), late / np.maximum(early, 1e-12), 0.0
    )
    say(
        f"  check 7 (600 unforced steps, {trips} trips): late/early increment ratio "
        + ", ".join(f"{f} {r:.2f}" for f, r in zip(fields, ratio))
        + "; the limit is 8, and the task is allowlisted"
    )


def section_deviations():
    title("deviations: D1 and the energy invariant", "float64")
    A = -P.deltaHr / (P.rho * P.C)
    A_water = -P.deltaHr / (P.rho * C_WATER)
    say(
        f"  D1: adiabatic rise A Caf = {A * P.Caf:.1f} K with C = {P.C} J/(g K); "
        f"{A_water * P.Caf:.1f} K for "
        f"an aqueous feed (C = {C_WATER} J/(g K), read)"
    )
    batch = P.replace(UA=0.0, q=0.0)
    with enable_x64():
        paths = _hold_action(_states(0.1, 340.0, 300.0, 0.0), jnp.zeros(1), batch, 400)
    C_a, T = (np.asarray(x)[0] for x in paths[:2])
    E = T + A * C_a
    k = _first(C_a <= 0.001)
    say(
        "  adiabatic batch (UA 0, q 0) from C_a 0.1 mol/L, 340 K: 99 % conversion "
        f"after {k} steps, T {T[k - 1]:.2f} K; "
        f"T + A C_a moves by at most {np.abs(E[:k] - (340.0 + A * 0.1)).max():.1e} K"
    )


def section_disturbance():
    title("disturbance: the feed drift and its effect on C_a", "float32 unless stated")
    seeds = np.arange(64)
    states, keys = reset_batch(seeds)
    _, rec = run_episodes(states, keys, {"raw": -1.0}, record=True)
    x = np.concatenate([np.asarray(states.Ti_dev)[:, None], rec["Ti_dev"]], axis=1)
    per_seed = x.mean(1)
    lag1 = float((x[:, 1:] * x[:, :-1]).sum() / (x[:, :-1] ** 2).sum())
    say(
        "  64 seeds x 1200 steps at full cooling (constant key, "
        f"{int(rec['tripped'].sum())} trips): mean {x.mean():+.3f} K "
        f"(3 standard errors {3 * per_seed.std(ddof=1) / np.sqrt(len(seeds)):.2f} K), "
        f"RMS {np.sqrt((x**2).mean()):.3f} K, "
        f"lag-1 {lag1:.5f} (exp(-dt/Ti_tau) = {np.exp(-P.delta_t / P.Ti_tau):.5f})"
    )
    a = np.exp(-P.delta_t / P.Ti_tau)
    sd = P.Ti_sigma * np.sqrt(1.0 - a**2)
    with enable_x64():
        base = _equilibrium_states(np.array(TARGETS), 0.0, dtype=jnp.float64)
        bumped = base.replace(Ti_dev=base.Ti_dev + sd)
        raw = jnp.asarray(raw_from_coolant(np.asarray(base.T_j), P))
        ref = np.asarray(_hold_action(base, raw, P, 20)[0])
        moved = np.asarray(_hold_action(bumped, raw, P, 20)[0])
    dC = moved - ref
    say(
        f"  one innovation ({sd:.3f} K, held), coolant held at Tc* (float64): C_a "
        "moves by "
        + ", ".join(
            f"{L:.2f}: {dC[i, 0]:+.1e} / {dC[i, 4]:+.1e}" for i, L in enumerate(TARGETS)
        )
        + " mol/L after 1 / 5 steps"
    )
    # Under the shipped cascade: the same start with the drift held at 0 and
    # at one innovation, so the controller's own start-up cancels.
    L = np.array(TARGETS)
    both = _equilibrium_states(np.concatenate([L, L]), 0.0)
    both = both.replace(Ti_dev=jnp.asarray(np.repeat([0.0, sd], L.size), jnp.float32))
    keys = jnp.tile(KEY[None], (2 * L.size, 1))
    _, rec = run_episodes(
        both, keys, load_gains(), n_steps=400, record=True, hold_drift=True
    )
    dC = np.abs(rec["C_a"][L.size :] - rec["C_a"][: L.size]).max(1)
    say(
        "  the same innovation under the shipped cascade (held, 20 min): largest C_a "
        "difference " + ", ".join(f"{x:.2f}: {c:.1e}" for x, c in zip(L, dC)) + " mol/L"
    )


def switch_response(gains, pairs=None, drifts=(-6.0, 0.0, 6.0), n_steps=400):
    """Each switch from rest on its first level, the switch at step
    ``block_steps``, drift held. Returns rows (from, to, dTi, settle steps
    (2 % band), closure steps (10 % band), peak T, trips); steps count from
    the first step scored against the new level."""
    pairs = pairs or [
        (a, b)
        for a in TARGETS
        for b in TARGETS
        if np.round(abs(a - b), 2) in (0.05, 0.10)
    ]
    rows = [(a, b, d) for a, b in pairs for d in drifts]
    frm = np.array([r[0] for r in rows])
    to = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])
    levels = np.concatenate([frm[:, None], np.repeat(to[:, None], N_BLOCKS - 1, 1)], 1)
    states = _states(
        frm,
        steady_temperature(frm, P, np),
        steady_coolant(frm, d, P, np),
        d,
        levels=levels,
        dtype=jnp.float32,
    )
    keys = jnp.tile(KEY[None], (len(rows), 1))
    _, rec = run_episodes(
        states, keys, gains, n_steps=n_steps, record=True, hold_drift=True
    )
    after = np.abs(rec["e"][:, P.block_steps - 1 :])
    size = np.abs(to - frm)[:, None]
    outside = after > 0.02 * size
    settle = np.where(
        outside.any(1), outside.shape[1] - np.argmax(outside[:, ::-1], 1), 0
    )
    closure = _first(after <= 0.10 * size)
    peak = rec["T"][:, P.block_steps - 1 :].max(1)
    return list(zip(frm, to, d, settle, closure, peak, rec["tripped"].sum(1)))


def section_settle():
    title(
        "settle: block length from the settle on DEFAULT_GAINS, closure against the "
        "horizon",
        "float32",
    )
    for name, gains in (
        ("DEFAULT_GAINS", DEFAULT_GAINS),
        ("shipped (load_gains)", load_gains()),
    ):
        rows = switch_response(gains)
        worst = max(rows, key=lambda r: r[3])
        say(f"  {name}: {_gains_str(gains)}")
        say(
            "    switch       settle to 2 % of the move (steps, by dTi -6/0/+6)   90 "
            "% closure (steps)   peak T (K)"
        )
        for i in range(0, len(rows), 3):
            grp = rows[i : i + 3]
            say(
                f"    {grp[0][0]:.2f} -> {grp[0][1]:.2f}   "
                + "/".join(str(int(r[3])) for r in grp).ljust(20)
                + "                             "
                + "/".join(str(int(r[4])) for r in grp).ljust(14)
                + "        "
                + f"{max(r[5] for r in grp):.2f}"
                + ("  TRIPS" if any(r[6] for r in grp) else "")
            )
        settle_min = worst[3] * P.delta_t
        block_min = max(int(np.ceil(3.0 * settle_min - 1e-9)), 6)
        say(
            f"    worst settle {int(worst[3])} steps = {settle_min:.2f} min "
            f"({worst[0]:.2f} -> {worst[1]:.2f}); "
            f"3 x settle = {3 * settle_min:.2f} min -> {block_min} min blocks = "
            f"{round(block_min / P.delta_t)} steps "
            f"(params.block_steps {P.block_steps})"
        )
        worst_close = max(rows, key=lambda r: r[4])
        say(
            f"    longest 90 % closure {int(worst_close[4])} steps = "
            f"{worst_close[4] * P.delta_t:.2f} min, against the "
            f"MPC's {MPC_HORIZON}-step ({MPC_HORIZON * P.delta_t:.1f} min) horizon"
        )


FAST = {
    "params": section_params,
    "steady": section_steady,
    "targets": section_targets,
    "pnr": section_pnr,
    "reach": section_reach,
    "integrator": section_integrator,
    "invariance": section_invariance,
    "deviations": section_deviations,
    "disturbance": section_disturbance,
    "settle": section_settle,
}


# ---------------------------------------------------------------------------
# Heavy sections
# ---------------------------------------------------------------------------


def pid_validation(n_random=200, gains=None):
    """The shipped cascade's validation. test_pid_holds_every_target runs
    the same three groups, with 64 random and 16 alternating episodes."""
    gains = gains or load_gains()
    title(
        f"pid-validation: the shipped cascade on {n_random + 100} episodes",
        "float32 episodes, float64 PNR",
    )
    say(
        f"  gains: {_gains_str(gains)}; held: "
        + ", ".join(f"{k} {gains[k]:g}" for k in HELD_GAINS)
    )
    say(
        "  closed-loop rates per target: "
        + ", ".join(
            f"{L:.2f}: {r:+.2f}" for L, r in zip(TARGETS, closed_loop_rates(gains))
        )
        + f" /min (guard: <= -{GUARD_MIN_DECAY}); smallest setpoint-window margin "
        + f"{setpoint_clip_margins(gains).min():.2f} K"
    )
    n = P.max_steps_in_episode
    groups = {}
    states, keys = reset_batch(np.arange(1000, 1000 + n_random))
    groups["random schedules, drift on"] = (states, keys, False)
    alt_seeds = np.arange(2000, 2050)
    alt, alt_keys = reset_batch(alt_seeds)
    alt = recentre(alt, np.tile([0.55, 0.45], 3), Ti_dev=6.0)
    groups["0.55/0.45 alternating, dTi +6 K held"] = (alt, alt_keys, True)
    i = np.arange(50)
    L = np.where(i % 2 == 0, 0.45, 0.65)
    d = np.array([-6.0, 0.0, 6.0])[i % 3]
    T_j0 = 290.0 + (i // 6)
    C0 = np.array([_low_state(tj, dd) for tj, dd in zip(T_j0, d)])
    catch = _states(
        C0,
        steady_temperature(C0, P, np),
        T_j0,
        d,
        levels=np.repeat(L[:, None], N_BLOCKS, 1),
        dtype=jnp.float32,
    )
    catch_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(3000 + i, jnp.uint32))
    groups["catches from extinction, dTi -6/0/+6 held"] = (catch, catch_keys, True)

    worst = np.inf
    for name, (s, k, hold) in groups.items():
        summary, rec = run_episodes(s, k, gains, record=True, hold_drift=hold)
        exact, line = episode_margins(rec)
        ok = ~rec["tripped"]
        e = np.abs(rec["e"])
        captured = np.where((e < 0.01) & ok, np.arange(n)[None, :], n).min(1)
        later = (
            (np.arange(n)[None, :] > captured[:, None])
            & ok
            & (rec["T"] < _folds()[0][1])
        )
        above = (rec["T"] > pnr_line(rec["C_a"])) & ok
        say(
            f"  {name} ({len(exact)} episodes): trips {int(summary['trips'].sum())}; "
            "re-extinctions after capture "
            f"{int(later.any(1).sum())}; smallest exact PNR margin {exact.min():.3f} K "
            f"(p1 {np.percentile(exact, 1):.3f}, "
            f"median {np.median(exact):.3f}); smallest line margin {line.min():.3f} K; "
            f"steps above the line {int(above.sum())}"
        )
        # A state is scored against level[block_clock // block_steps].
        held = ok & in_hold(rec["block_clock"])
        per_episode = np.array(
            [row[m].mean() if m.any() else np.nan for row, m in zip(e, held)]
        )
        say(
            f"    |e| in the hold window (minutes 6 to 8.4 of each block): mean {e[held].mean():.2e}, "
            "per-episode means "
            f"{np.nanmin(per_episode):.2e} to {np.nanmax(per_episode):.2e} mol/L; v1 "
            f"{summary['v1'].mean() / n:.4f}, "
            f"v2 cost {-summary['v2'].mean() / n:.4g} per step"
        )
        worst = min(worst, exact.min())
    summary = run_episodes(
        catch, catch_keys, gains, hold_drift=True, sp_at_T_star=True
    )[0]
    say(
        "  the same catches with the setpoint started at T*(target): "
        f"trips {int(summary['trips'].sum())} in "
        f"{int((summary['trips'] > 0).sum())} of 50 episodes"
    )
    # Decision rule: a smallest exact margin under 0.5 K raises GUARD_MIN_DECAY
    # to 1.0 /min, followed by one re-tune.
    if worst < 0.5:
        say(
            f"  decision rule: smallest exact margin {worst:.3f} K is under 0.5 K, so "
            "raise GUARD_MIN_DECAY to 1.0 /min and re-tune once"
        )
    else:
        say(
            f"  decision rule: smallest exact margin {worst:.3f} K >= 0.5 K, so "
            f"GUARD_MIN_DECAY stays {GUARD_MIN_DECAY} /min"
        )


def v1_table(n_seeds=128):
    title(
        "v1: version 1 against version 2 on 12 policies and a constant scan, "
        f"{n_seeds} seeds",
        "float32",
    )
    n = P.max_steps_in_episode
    seeds = np.arange(4000, 4000 + n_seeds)
    states, keys = reset_batch(seeds)
    policies = [
        ("(a) full heating, 310 K", {"raw": 1.0}),
        ("(b) full cooling, 290 K", {"raw": -1.0}),
        ("(c) cascade on DEFAULT_GAINS", DEFAULT_GAINS),
        ("(c') shipped cascade", load_gains()),
    ]
    for L in TARGETS:
        tc = float(steady_coolant(L, 0.0, P, np))
        policies.append(
            (
                f"(d) constant Tc*({L:.2f}) = {tc:.2f} K",
                {"raw": raw_from_coolant(tc, P)},
            )
        )
    policies.append(("(e) zero action, 300 K", {"raw": 0.0}))
    res = {}
    lump = float(P.restart_steps)
    say(
        f"  {'policy':42s} {'v1, 0 on a trip':>16s} {'trips/ep':>9s} "
        f"{'v2 cost/step':>13s} {'v1 as shipped':>14s}"
    )
    for name, ctrl in policies:
        s = run_episodes(states, keys, ctrl)[0]
        res[name] = dict(
            v1_unfixed=float((s["v1"] + lump * s["trips"]).mean() / n),
            trips=float(s["trips"].mean()),
            v2=float(-s["v2"].mean() / n),
            v1=float(s["v1"].mean() / n),
        )
        r = res[name]
        say(
            f"  {name:42s} {r['v1_unfixed']:16.4f} {r['trips']:9.2f} {r['v2']:13.4g} "
            f"{r['v1']:14.4f}"
        )
    cs = np.arange(290.0, 310.001, 0.25)
    tiled = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (cs.size,) + (1,) * (x.ndim - 1)), states
    )
    s = run_episodes(
        tiled,
        jnp.tile(keys, (cs.size, 1)),
        {"raw": np.repeat(raw_from_coolant(cs, P), n_seeds)},
    )[0]
    v1d = ((s["v1"] + lump * s["trips"]) / n).reshape(cs.size, n_seeds).mean(1)
    trips = s["trips"].reshape(cs.size, n_seeds).mean(1)
    v2 = (-s["v2"] / n).reshape(cs.size, n_seeds).mean(1)
    v1 = (s["v1"] / n).reshape(cs.size, n_seeds).mean(1)
    best = int(np.argmax(v1d))
    safe = np.flatnonzero(trips == 0)
    say(
        "  constant scan 290 to 310 K by 0.25 K: best under v1 with 0 on a trip "
        f"{cs[best]:.2f} K (v1 {v1d[best]:.4f}, "
        f"trips {trips[best]:.2f}, v2 {v2[best]:.4g})"
    )
    res[f"best constant ({cs[best]:.2f} K)"] = dict(
        v1_unfixed=v1d[best], trips=trips[best], v2=v2[best], v1=v1[best]
    )
    if safe.size:
        j = safe[np.argmax(v1d[safe])]
        say(f"  best trip-free constant {cs[j]:.2f} K: v1 {v1d[j]:.4f}, v2 {v2[j]:.4g}")
        res[f"best trip-free constant ({cs[j]:.2f} K)"] = dict(
            v1_unfixed=v1d[j], trips=0.0, v2=v2[j], v1=v1[j]
        )
    names = list(res)
    tripping = [k for k in names if res[k]["trips"] > 0]
    safe_p = [k for k in names if res[k]["trips"] == 0]
    pairs = [
        (a, b)
        for a in tripping
        for b in safe_p
        if res[a]["v1_unfixed"] > res[b]["v1_unfixed"]
    ]
    say(
        "  tripping policies above a trip-free one under v1 with 0 on a trip: "
        f"{len(pairs)} pairs"
    )
    lump_star = max(
        (
            (res[a]["v1_unfixed"] - res[b]["v1_unfixed"]) * n / res[a]["trips"]
            for a, b in pairs
        ),
        default=0.0,
    )
    say(
        f"  break-even: a tripped step must score below -{lump_star:.1f} to reverse "
        f"every pair; the shipped v1 charges -{lump:.0f}"
    )
    still = [(a, b) for a in tripping for b in safe_p if res[a]["v1"] > res[b]["v1"]]
    say(
        "  tripping policies still above a trip-free one under the shipped v1: "
        f"{len(still)}"
    )
    v2s = [-res[k]["v2"] for k in names]
    say(
        f"  Spearman rank correlation with v2 over {len(names)} policies: v1 with 0 on "
        "a trip "
        f"{spearmanr([res[k]['v1_unfixed'] for k in names], v2s).correlation:.3f}, "
        "shipped v1 "
        f"{spearmanr([res[k]['v1'] for k in names], v2s).correlation:.3f} (zero action "
        "and Tc*(0.50) are one policy)"
    )


def _grid_rates(gain_sets, params, lag0, levels=TARGETS):
    """Closed-loop rates (/min) for many gain sets at once, rows = gain sets,
    columns = levels, float64: the Jacobian of one closed-loop step at each
    equilibrium (``experts.closed_loop_rates``, batched, and at 0 s lag)."""
    gains = {k: np.asarray(gain_sets[k], float) for k in DEFAULT_GAINS}
    params = params.replace(Ti_sigma=0.0)
    out = []
    with enable_x64():
        for L in levels:
            z = _closed_loop_equilibrium(L, 0.0, gains, params)
            J = np.asarray(
                _cl_jacobian(lag0)(
                    jnp.asarray(z),
                    jnp.float64(L),
                    {k: jnp.asarray(v) for k, v in gains.items()},
                    params,
                )
            )
            out.append(np.log(np.abs(np.linalg.eigvals(J)).max(-1)) / params.delta_t)
    return np.stack(out, 1)


@functools.lru_cache(maxsize=None)
def _cl_jacobian(lag0):
    """Jacobians of one closed-loop step, batched over gain sets: the
    cascade, then ``compute_next_state`` (at 0 s, the jacket set to the
    command). The same map ``experts.closed_loop_rates`` linearises."""

    def one_step(z, level, g, params):
        state = UnstableCSTRState(
            time=0,
            C_a=z[0],
            T=z[1],
            T_j=z[2],
            Ti_dev=0.0 * z[0],
            target_levels=jnp.full(N_BLOCKS, level),
            block_clock=0,
        )
        cs = CascadeState(integral=z[3], T_prev=z[4], T_sp_prev=z[5])
        u, cs = cascade_step(g, cs, get_obs(state, params), params, xp=jnp)
        if lag0:
            state = state.replace(
                T_j=convert_raw_action_to_range(u, params.T_c_min, params.T_c_max)
            )
            params = params.replace(tau_j=jnp.inf)
        new = compute_next_state(u, state, params, KEY)[0]
        return jnp.stack(
            [new.C_a, new.T, new.T_j, cs.integral, cs.T_prev, cs.T_sp_prev]
        )

    return jax.jit(jax.vmap(jax.jacfwd(one_step), in_axes=(0, None, 0, None)))


def _tune(params, lag0, start, guard=True, seeds=None, passes=2):
    """The shipped tuner's coordinate descent (scripts/tune_pid.py,
    _tune_aircraft_search: each tuned gain times 0.5, 0.7, 1.4 and 2.0, two
    passes, greedy) on the mean version-2 return, the held gains fixed. With
    ``guard`` a candidate counts only if the factory's guard would pass it at
    this lag; while the best set does not, a candidate that raises its
    slowest rate is taken instead (the shipped tuner scores a refused start
    -inf and stops there). Returns the gains, the start's (return, slowest
    rate, guard verdict), the number of evaluations and of seeds."""
    seeds = TUNE_SEEDS if seeds is None else seeds
    states, keys = reset_batch(np.arange(seeds))

    def evaluate(cands):
        batch = {k: np.repeat([c[k] for c in cands], seeds) for k in DEFAULT_GAINS}
        tiled = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (len(cands),) + (1,) * (x.ndim - 1)), states
        )
        s = run_episodes(
            tiled, jnp.tile(keys, (len(cands), 1)), batch, params=params, lag0=lag0
        )[0]
        rets = s["v2"].reshape(len(cands), seeds).mean(1)
        rates = _grid_rates(
            {k: [c[k] for c in cands] for k in DEFAULT_GAINS}, params, lag0
        ).max(1)
        ok = np.array([(setpoint_clip_margins(c, params) > 0).all() for c in cands]) & (
            rates <= -GUARD_MIN_DECAY
        )
        return rets, rates, ok

    def rank(ret, rate, ok):
        return (0, -ret) if (ok or not guard) else (1, rate)

    best = dict(start)
    r, rt, ok = evaluate([best])
    best_key, n_eval = rank(r[0], rt[0], ok[0]), 1
    first = (float(r[0]), float(rt[0]), bool(ok[0]))
    for _ in range(passes):
        for g in TUNED_GAINS:
            cands = [{**best, g: best[g] * f} for f in SEARCH_FACTORS]
            r, rt, ok = evaluate(cands)
            n_eval += len(cands)
            for c, a, b, o in zip(cands, r, rt, ok):
                if rank(a, b, o) < best_key:
                    best, best_key = c, rank(a, b, o)
    return best, first, n_eval, seeds


def _hold_and_alt(params, lag0, gains, n_hold):
    """One-hour holds at each target and 0.55/0.45 alternation, drift on."""
    states, keys = reset_batch(np.arange(5000, 5000 + n_hold), params)
    hold = {}
    for L in TARGETS:
        s = recentre(states, L, params)
        summary, rec = run_episodes(
            s, keys, gains, params=params, record=True, lag0=lag0
        )
        e = np.abs(rec["e"][:, 240:])[~rec["tripped"][:, 240:]]
        hold[L] = dict(
            trip=float((summary["trips"] > 0).mean()),
            ext=float((summary["ext"] > 0).mean()),
            mae=float(e.mean()) if e.size else np.nan,
        )
    s = recentre(states, np.tile([0.55, 0.45], 3), params)
    summary = run_episodes(s, keys, gains, params=params, lag0=lag0)[0]
    return hold, dict(
        trip=float((summary["trips"] > 0).mean()),
        ext=float((summary["ext"] > 0).mean()),
    )


#: Jacket lags of the sensitivity table (s).
LAGS = (0, 3, 6, 12, 20, 30)
#: The cascade gain grid of the sensitivity table. The outer gain reaches
#: down to 6.25 K per mol/L: with 25 as its lowest value the best P-only inner
#: loop at 6 s looked unstable, an artefact of the grid.
GAIN_GRID = dict(
    Kp_T=np.geomspace(0.25, 40.0, 28),
    Kd_T=np.concatenate([[0.0], np.geomspace(0.02, 4.0, 23)]),
    Kc_Ca=np.array([6.25, 12.5, 25.0, 50.0, 100.0, 200.0, 400.0]),
    Ti_Ca=np.array([0.25, 0.5, 1.0, 2.0]),
)


def sensitivity(n_val=None, lags=LAGS, grid=GAIN_GRID, n_resets=20000):
    title(
        "sensitivity: the jacket lag at " + ", ".join(str(x) for x in lags) + " s",
        "float32 episodes, float64 PNR and rates",
    )
    mesh = np.meshgrid(*grid.values(), indexing="ij")
    grid_sets = {
        **{k: np.full(mesh[0].size, v) for k, v in DEFAULT_GAINS.items()},
        **{k: m.ravel() for k, m in zip(grid, mesh)},
    }
    say(
        f"  gain grid: {mesh[0].size} cascade sets (Kp_T x Kd_T x Kc_Ca x Ti_Ca = "
        + " x ".join(str(v.size) for v in grid.values())
        + "), Kd_T = 0 is the P-only inner loop"
    )
    L45 = TARGETS[0]
    lam = float(
        _eig(
            _jacobian(
                L45, steady_temperature(L45, P, np), steady_coolant(L45, 0.0, P, np)
            )[0, :2, :2]
        ).real[-1]
    )
    for lag in lags:
        t0 = time.time()
        params, lag0 = _lag_params(lag)
        n_rows = n_val or (1024 if lag >= 12 else 256)
        say("-" * 78)
        say(
            f"  LAG {lag} s: lam+ tau_j at 0.45 = {lam * lag / 60.0:.3f} (lam+ "
            f"{lam:.3f} /min does not depend on the lag)"
        )
        L = np.array(TARGETS)
        T_star = steady_temperature(L, P, np)
        T_j = np.full(L.shape, P.T_c_min) if lag0 else steady_coolant(L, 0.0, P, np)
        pnr0 = point_of_no_return_batch(L, T_j, 0.0, params) - T_star
        T_j6 = P.T_c_min if lag0 else float(steady_coolant(L45, 6.0, P, np))
        pnr6 = float(point_of_no_return_batch(L45, T_j6, 6.0, params)[0]) - T_star[0]
        corner = {
            d: float(
                point_of_no_return_batch(
                    L45 + P.initial_CA_offset,
                    P.T_c_min if lag0 else steady_coolant(L45, d, P, np),
                    d,
                    params,
                )[0]
            )
            - (T_star[0] + P.initial_T_offset)
            for d in (0.0, 6.0)
        }
        say(
            "    PNR - T* (pure T, dTi 0): "
            + ", ".join(f"{x:.2f}: {p:.2f}" for x, p in zip(L, pnr0))
            + f" K; at 0.45 and dTi +6: {pnr6:.2f} K; reset corner at 0.45: "
            + f"{corner[0.0]:+.2f} / {corner[6.0]:+.2f} K (dTi 0 / +6)"
        )
        states, _ = reset_batch(np.arange(n_resets), params)
        C0, T0, d0 = (np.asarray(x) for x in (states.C_a, states.T, states.Ti_dev))
        Tj0 = np.full(C0.shape, P.T_c_min) if lag0 else np.asarray(states.T_j)
        with enable_x64():
            doomed = np.asarray(
                jax.vmap(trips_under_full_cooling, in_axes=(0, 0, 0, 0, None))(
                    *(jnp.asarray(x, jnp.float64) for x in (C0, T0, Tj0, d0)), params
                )
            )
        say(
            f"    resets past the PNR ({n_resets} reset_env draws, drift clipped at 3 "
            f"sd): {doomed.mean() * 100:.3f} %"
        )
        J3 = _jacobian(L45, T_star[0], steady_coolant(L45, 0.0, P, np), params=params)[
            0
        ]
        if lag0:
            # The balances' jacket column is the coolant's when there is no lag.
            A, B, C = J3[:2, :2], J3[:2, 2], np.array([0.0, 1.0])
        else:
            A, B, C = (
                J3,
                np.array([0.0, 0.0, 1.0 / params.tau_j]),
                np.array([0.0, 1.0, 0.0]),
            )
        m = A.shape[0]
        M = np.zeros((m + 1, m + 1))
        M[:m, :m], M[:m, m] = A, B
        Md = expm(M * P.delta_t)
        k = np.arange(0.0, 40.0, 0.01)
        rad = np.abs(
            np.linalg.eigvals(
                Md[None, :m, :m] - k[:, None, None] * np.outer(Md[:m, m], C)[None]
            )
        ).max(1)
        window = (k[rad < 1][0], k[rad < 1][-1]) if (rad < 1).any() else None
        ends = np.array([0.45, 0.65])
        rest = _equilibrium_states(ends, 0.0, P, dtype=jnp.float32)
        if lag0:
            rest = rest.replace(T_j=jnp.full(2, P.T_c_max, jnp.float32))
        steps = _first(np.asarray(_hold_action(rest, jnp.ones(2), params, 400)[3]))
        say(
            "    P-only on T stabilising gain window at 0.45 (ZOH, K coolant per K): "
            f"{window}; "
            f"full heating trips from rest at 0.45 / 0.65 in {steps[0]} / {steps[1]} "
            "steps"
        )
        rates = _grid_rates(grid_sets, params, lag0)
        best = rates.min(0)
        p_only = grid_sets["Kd_T"] == 0.0
        ponly = rates[p_only].min(0) if p_only.any() else np.full(len(L), np.nan)
        say(
            "    best closed-loop rate on the grid (/min; best P-only inner loop in "
            "brackets): "
            + ", ".join(
                f"{x:.2f}: {b:+.2f} ({p:+.2f})" for x, b, p in zip(L, best, ponly)
            )
        )
        val_states, val_keys = reset_batch(np.arange(2000, 2000 + n_rows), params)
        default = summarize(
            run_episodes(val_states, val_keys, DEFAULT_GAINS, params=params, lag0=lag0)[
                0
            ],
            P.max_steps_in_episode,
        )
        dr = _grid_rates({k: [v] for k, v in DEFAULT_GAINS.items()}, params, lag0)[0]
        say(
            f"    DEFAULT_GAINS, {n_rows} episodes: {default['trip_share'] * 100:.1f} "
            f"% with a trip, {default['trips']:.3f} trips/ep, "
            f"{default['ext_share'] * 100:.1f} % extinguished; rate at 0.45 "
            f"{dr[0]:+.2f} /min"
        )
        variants = [("guarded", dict(DEFAULT_GAINS), True)]
        if lag == 6:
            variants.append(("plain (no guard)", dict(DEFAULT_GAINS), False))
        if lag == 12:
            variants.append(
                ("guarded, 1.5 K/min ramp", {**DEFAULT_GAINS, "sp_rate": 1.5}, True)
            )
        for label, start, guard in variants:
            tuned, first, n_eval, n_seeds = _tune(params, lag0, start, guard=guard)
            summary, rec = run_episodes(
                val_states, val_keys, tuned, params=params, lag0=lag0, record=True
            )
            s = summarize(summary, P.max_steps_in_episode)
            tr = _grid_rates({k: [v] for k, v in tuned.items()}, params, lag0)[0]
            exact = episode_margins(rec, params, k=6, lag0=lag0)[0]
            clean = exact[summary["trips"] == 0]
            e, tgt = np.abs(rec["e"]), rec["C_a"] + rec["e"]
            # A state is scored against level[block_clock // block_steps].
            ok = ~rec["tripped"]
            per = []
            for x in L:
                near = ok & (np.abs(tgt - x) <= 0.025 + 1e-6)
                late = near & in_hold(rec["block_clock"])
                per.append(
                    (
                        e[near].mean() if near.any() else np.nan,
                        e[late].mean() if late.any() else np.nan,
                    )
                )
            say(
                f"    [{label}] {n_eval} evaluations x {n_seeds} seeds; start "
                f"return {first[0]:.4g} (slowest rate {first[1]:+.2f}, "
                f"{'passes' if first[2] else 'fails'} the guard) -> {_gains_str(tuned)}"
            )
            say(
                f"    [{label}] {n_rows} episodes: {s['trip_share'] * 100:.1f} % with "
                f"a trip, {s['trips']:.3f} trips/ep, "
                f"{s['ext_share'] * 100:.1f} % extinguished, v2 {s['v2']:.4g}/step, v1 "
                f"{s['v1']:.4f}/step, "
                f"coolant at a bound {s['sat'] * 100:.1f} % of steps"
            )
            say(
                f"    [{label}] rates "
                + ", ".join(f"{x:.2f}: {r:+.2f}" for x, r in zip(L, tr))
                + " /min; smallest PNR margin over trip-free episodes "
                + f"{clean.min() if clean.size else float('nan'):.2f} K"
            )
            say(
                f"    [{label}] mean |e| per target, whole block / hold window (mol/L): "
                + "; ".join(f"{x:.2f}: {a:.2e} / {b:.2e}" for x, (a, b) in zip(L, per))
            )
            if guard:
                hold, alt = _hold_and_alt(params, lag0, tuned, n_rows)
                say(
                    f"    [{label}] 1 h holds ({n_rows} per target): "
                    + "; ".join(
                        f"{x:.2f}: trip {h['trip'] * 100:.1f} %, ext "
                        f"{h['ext'] * 100:.1f} %, |e| {h['mae']:.1e}"
                        for x, h in hold.items()
                    )
                    + f"; 0.55/0.45 alternation: trip {alt['trip'] * 100:.1f} %, "
                    + f"ext {alt['ext'] * 100:.1f} %"
                )
                if label == "guarded":
                    lost = [x for x, b in zip(L, best) if b >= 0.0] + [
                        x for x, h in hold.items() if h["trip"] > 0 or h["ext"] > 0
                    ]
                    say(f"    not held trip-free: {sorted(set(lost)) or 'none'}")
        say(f"    lag {lag} s done in {time.time() - t0:.0f} s")


#: Episodes in the gate's stress row: the 0.55/0.45 alternation with the drift
#: held at +6 K (3 standard deviations), the hot-going switches at the hottest
#: feed, where the MPC's soft bound sits closest to the targets.
GATE_STRESS = 10
#: Gate groups: rollout seeds from, and a label.
GATE_GROUPS = {
    "random": (1000, "random schedules, drift on"),
    "alternating": (2000, "0.55/0.45 alternating, dTi +6 K held"),
}


def _gate_starts(group, seeds):
    """Start states and rollout keys of gate episodes, as ``pid_validation``
    builds them, and whether the drift is held at its start value."""
    states, keys = reset_batch(seeds)
    if group == "alternating":
        return recentre(states, np.tile([0.55, 0.45], 3), Ti_dev=6.0), keys, True
    return states, keys, False


@functools.lru_cache(maxsize=None)
def _jit_step():
    return jax.jit(ENV.step_env)


def _gate_episode(job):
    """One MPC episode of the gate, for the pool.

    The registry's factory on ``plan_params``, as the suite builds the MPC,
    stepped on the env's own ``step_env`` with the seed's key throughout (as
    ``runners.rollout`` does). Before each step it also asks the fallback PID,
    without stepping it, for the action it would take over with, and records
    that coolant's distance from the coolant last applied: the size of a
    fallback's first move, on every step, whether or not a solve failed.
    """
    group, seed = job
    spec = registry.get("unstable_cstr")
    states, keys, hold = _gate_starts(group, [seed])
    state = jax.tree_util.tree_map(lambda x: x[0], states)
    key = keys[0]
    drift = state.Ti_dev
    mpc = spec.make_mpc(ENV, plan_params(spec, P))
    mpc.reset()

    counts = {"fallbacks": 0, "non_finite": 0}
    fallback = mpc._fallback

    def counted(guess, u, obs):
        counts["fallbacks"] += 1
        counts["non_finite"] += int(not np.all(np.isfinite(u)))
        return fallback(guess, u, obs)

    mpc._fallback = counted
    step = _jit_step()
    n = int(P.max_steps_in_episode)
    rec = {f: [] for f in REC_FIELDS}
    seconds = np.zeros(n)
    transfer = np.full(n, np.nan)
    v2 = 0.0
    last = None
    for t in range(n):
        obs = ENV.get_obs(state, P)
        pid = mpc._pid
        if last is not None and pid._cs is not None:
            u_pid = cascade_step(
                pid.gains, pid._cs, np.asarray(obs, float), pid.params
            )[0]
            transfer[t] = coolant_from_raw(float(u_pid), P) - coolant_from_raw(last, P)
        t0 = time.perf_counter()
        u = mpc.step(obs, state)
        seconds[t] = time.perf_counter() - t0
        _, state, reward, _, info = step(key, state, jnp.asarray([u]), P)
        if hold:
            state = state.replace(Ti_dev=drift)
        v2 += float(reward)
        last = u
        for f, v in zip(
            REC_FIELDS,
            (
                state.C_a,
                state.T,
                state.T_j,
                state.Ti_dev,
                live_target(state, P) - state.C_a,
                info["tripped"],
                state.block_clock,
                u,
            ),
        ):
            rec[f].append(np.asarray(v))
    return {
        "group": group,
        "seed": seed,
        "rec": {f: np.asarray(v) for f, v in rec.items()},
        "seconds": seconds,
        "transfer": transfer,
        "v2": v2,
        "report": mpc.solver_report(),
        "capped": mpc.solve_capped,
        **counts,
    }


def mpc_gate(n_episodes=100, workers=1):
    """The MPC's gate: zero trips over ``n_episodes`` random episodes and the
    stress row. Also prints solve times, fallbacks (non-finite actions among
    them), the smallest point-of-no-return margin, and the fallback's first
    move against ``FALLBACK_STEP_BOUND_K``, the bound
    test_mpc_falls_back_to_the_pid asserts."""
    spec = registry.get("unstable_cstr")
    probe = spec.make_mpc(ENV, plan_params(spec, P))
    n = int(P.max_steps_in_episode)
    title(
        f"mpc-gate: the MPC on {n_episodes} random episodes and {GATE_STRESS} "
        "alternating ones",
        "float32 episodes, float64 PNR",
    )
    say(
        f"  horizon {probe.horizon} steps ({probe.horizon * probe.mpc_dt:.2f} min); "
        f"soft bound pnr_line(C_a) - {probe.pnr_margin_K:g} K at "
        f"{probe.pnr_penalty:g} per K and interval; error scale "
        f"{probe.error_scale:g} mol/L; move weight {probe.move_weight:g}"
    )
    say(f"  fallback PID: {_gains_str(probe._pid.gains)}")
    jobs = [
        (group, GATE_GROUPS[group][0] + i)
        for group, count in (("random", n_episodes), ("alternating", GATE_STRESS))
        for i in range(count)
    ]
    t0 = time.time()
    results = []
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_gate_episode, job) for job in jobs]
            for fut in cf.as_completed(futures):
                results.append(fut.result())
                if len(results) % 10 == 0:
                    say(
                        f"    {len(results)}/{len(jobs)} episodes, {time.time() - t0:.0f} s"
                    )
    else:
        for job in jobs:
            results.append(_gate_episode(job))
            if len(results) % 10 == 0:
                say(
                    f"    {len(results)}/{len(jobs)} episodes, {time.time() - t0:.0f} s"
                )
    results.sort(key=lambda r: (r["group"] != "random", r["seed"]))

    gains = load_gains()
    all_trips = 0
    for group, (_, label) in GATE_GROUPS.items():
        rs = [r for r in results if r["group"] == group]
        if not rs:
            continue
        seeds = np.array([r["seed"] for r in rs])
        rec = {f: np.stack([r["rec"][f] for r in rs]) for f in REC_FIELDS}
        exact, line = episode_margins(rec)
        ok = ~rec["tripped"]
        trips = rec["tripped"].sum(1)
        all_trips += int(trips.sum())
        over = np.where(
            ok, rec["T"] - (pnr_line(rec["C_a"]) - probe.pnr_margin_K), -np.inf
        )
        say(
            f"  {label} (seeds {seeds.min()} to {seeds.max()}, {len(rs)} episodes): "
            f"trips {int(trips.sum())} in {int((trips > 0).sum())} episodes; "
            f"fallback steps {sum(r['fallbacks'] for r in rs)} "
            f"(non-finite actions {sum(r['non_finite'] for r in rs)}); capped solves "
            f"{sum(r['capped'] for r in rs)}; mean IPOPT iterations "
            f"{np.mean([r['report']['solver_mean_iters'] for r in rs]):.1f}"
        )
        worst = int(np.argmin(exact))
        say(
            f"    smallest exact PNR margin {exact.min():.3f} K (seed {seeds[worst]}; "
            f"p1 {np.percentile(exact, 1):.3f}, median {np.median(exact):.3f}); "
            f"smallest line margin {line.min():.3f} K; steps above the soft bound "
            f"{int((over > 0).sum())} (largest excess {max(over.max(), 0.0):.3f} K)"
        )
        e = np.abs(rec["e"])
        held = ok & in_hold(rec["block_clock"])
        per_episode = np.array(
            [row[m].mean() if m.any() else np.nan for row, m in zip(e, held)]
        )
        states, keys, hold = _gate_starts(group, seeds)
        pid = run_episodes(states, keys, gains, hold_drift=hold)[0]
        mpc_cost = -np.mean([r["v2"] for r in rs]) / n
        pid_cost = -pid["v2"].mean() / n
        say(
            f"    |e| in the hold window (minutes 6 to 8.4 of each block): mean {e[held].mean():.2e}, "
            f"per-episode means {np.nanmin(per_episode):.2e} to "
            f"{np.nanmax(per_episode):.2e} mol/L; v2 cost per step {mpc_cost:.4g} "
            f"against the shipped PID's {pid_cost:.4g} on the same starts "
            f"(ratio {mpc_cost / pid_cost:.3f}; PID trips {int(pid['trips'].sum())})"
        )

    seconds = np.concatenate([r["seconds"] for r in results]) * 1e3
    say(
        f"  solve time per step (ms, {seconds.size} steps, wall clock on this "
        f"machine under its current load): p50 {np.percentile(seconds, 50):.1f}, "
        f"p90 {np.percentile(seconds, 90):.1f}, p99 {np.percentile(seconds, 99):.1f}, "
        f"max {seconds.max():.0f}, mean {seconds.mean():.1f}; "
        f"{seconds.sum() / 1e3 / len(results):.0f} s of solves per episode; "
        f"run {time.time() - t0:.0f} s on {workers} worker(s)"
    )
    transfer = np.abs(np.concatenate([r["transfer"] for r in results]))
    settled = np.concatenate(
        [
            np.concatenate(
                [
                    [False],
                    in_hold(r["rec"]["block_clock"])[:-1],
                ]
            )
            for r in results
        ]
    )
    finite = np.isfinite(transfer)
    say(
        "  a fallback's first move, |PID coolant - coolant last applied| on every "
        f"step: p50 {np.percentile(transfer[finite], 50):.3f} K, p99 "
        f"{np.percentile(transfer[finite], 99):.3f} K, max {transfer[finite].max():.3f} "
        "K; after a step in the hold window max "
        f"{transfer[finite & settled].max():.3f} K (FALLBACK_STEP_BOUND_K, "
        f"the bound test_mpc_falls_back_to_the_pid asserts, is "
        f"{FALLBACK_STEP_BOUND_K:g} K)"
    )
    if all_trips == 0:
        say(f"  gate: PASSES, zero trips in {len(results)} episodes")
    else:
        say(
            f"  gate: FAILS, {all_trips} trips; the remedies to try, in order: "
            "horizon 45, then 60, then a larger pnr_penalty"
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--section",
        action="append",
        choices=list(FAST),
        help="fast section(s) to print; default all",
    )
    ap.add_argument(
        "--sensitivity", action="store_true", help="the jacket-lag table (heavy)"
    )
    ap.add_argument(
        "--v1", action="store_true", help="the version-1 exploit table (heavy)"
    )
    ap.add_argument(
        "--pid-validation",
        action="store_true",
        help="the cascade's 300-episode validation (heavy)",
    )
    ap.add_argument(
        "--mpc-gate", action="store_true", help="the MPC's 110-episode gate (heavy)"
    )
    ap.add_argument("--workers", type=int, default=1, help="processes for --mpc-gate")
    ap.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="episodes per row of a heavy table (each has its default)",
    )
    args = ap.parse_args()
    t0 = time.time()
    heavy = args.sensitivity or args.v1 or args.pid_validation or args.mpc_gate
    for name in args.section or ([] if heavy else list(FAST)):
        FAST[name]()
    if args.pid_validation:
        pid_validation(args.seeds or 200)
    if args.v1:
        v1_table(args.seeds or 128)
    if args.sensitivity:
        sensitivity(args.seeds)
    if args.mpc_gate:
        mpc_gate(args.seeds or 100, args.workers)
    say("=" * 78)
    say(f"runtime {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
