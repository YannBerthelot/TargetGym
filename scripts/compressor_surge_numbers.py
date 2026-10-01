"""Every derived number in compressor_surge's PHYSICS.md, from the env's own code.

Run with ``uv run python scripts/compressor_surge_numbers.py [--section NAME ...]``.
With no flags it prints the fast sections, in under a minute.
``--section mpc-width`` runs only when named: it builds the NMPC at three
clip smoothings, solves the first 12 steps of four episodes (seeds 1000 to
1003) with each, and solves each episode's first step from the cold guess
with both of the NMPC's IPOPT instances, under two minutes on one core
(PHYSICS.md section 7). The heavy tables have their own flags:

    --hardness         the hardness table (PHYSICS.md section 5)
    --sensitivity      the sensitivity table and the restart break-evens
                       (section 6)
    --v1               the version-1 reward and its exploit, with the two rare
                       trippers on 5120 episodes each (section 8)
    --pid-validation   the pair's 4096-episode validation (section 7;
                       test_the_pair_holds_the_validation_seeds runs 1024 of
                       them)
    --mpc-gate         the NMPC's acceptance run (section 7): the
                       surge-margin budget, then each candidate margin on 128
                       episodes and the stress battery, against the pair and
                       G10, with solve times. Each candidate runs 128
                       episodes of 1200 NMPC steps and 16 battery runs of 410;
                       --workers N runs them in N processes, --mpc-horizon
                       and --mpc-margins change what it runs, and --gate-out
                       names where the per-step arrays go.

``--seeds N`` sets the episodes per row of a heavy table. Heavy tables use
seeds from 1000 up, apart from the recording seeds 0 to 9 and the tuner's 0
to 23.

Every number goes through the task's own code: the closed forms,
``compute_velocity``, ``compute_next_state``, ``step_env``,
``equilibrium_phi``, the action map and its inverse (``commands_from_raw``,
``raw_from_commands``) and ``pair_step(xp=jnp)`` under ``vmap`` and
``scan``. No dynamics are restated here, so the numbers describe the env as
it ships; ``--section mpc-width`` and ``--mpc-gate`` run the NMPC's own
CasADi step from ``experts.py``, which test_mpc_model_is_the_env_step and
test_mpc_predicts_one_step_like_the_plant hold to the env. Closed forms,
roots and Jacobians are computed in float64
(``jax.experimental.enable_x64``), and so are single trajectories unless
their line says float32, as the integrator's rk4_10 against rk4_40
comparison does. Closed-loop tables run in float32, as the env does; the
guard section also repeats its battery in float64 to compare the two. Each
section's title gives its precision.

The heuristic policies (a fixed recycle, the margin-blind chaser, the guarded
valve-as-actuator) are built by ``policy`` and stepped by ``_act``, their one
copy. tests/compressor_surge/test_compressor_surge_env.py loads this file and
runs the same two functions, so the tests and the tables run the same
policies. They share a speed PI with kp_speed 0.08 of rated speed per kPa
and ki_speed 0.04 per kPa s (``HEURISTIC_SPEED``, the speed loop of the
pair's ``DEFAULT_GAINS``), and the guarded heuristic uses the pair's
anti-surge logic at its own line, gains and override.
"""

import argparse
import concurrent.futures as cf
import contextlib
import functools
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings

# Ahead of the imports below: importing target_gym pulls in do-mpc, which warns
# about optional features this package does not use.
warnings.filterwarnings("ignore")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import enable_x64  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.stats import beta, chi2, spearmanr  # noqa: E402

import target_gym.compressor_surge.experts as experts  # noqa: E402
from target_gym import reward as R  # noqa: E402
from target_gym.compressor_surge.env import (  # noqa: E402
    KPA,
    N_DEMAND_BLOCKS,
    N_SETPOINT_BLOCKS,
    CompressorSurgeParams,
    CompressorSurgeState,
    characteristic,
    compute_next_state,
    compute_reward_terms,
    compute_reward_v1,
    compute_velocity,
    consumer_flow,
    demand_opening,
    equilibrium_phi,
    get_obs,
    greitzer_B,
    helmholtz_omega,
    live_target,
    recycle_flow,
    recycle_power,
    sound_speed,
    suction_density,
    surge_constant,
    surge_flow_per_speed,
)
from target_gym.compressor_surge.env_jax import CompressorSurge  # noqa: E402
from target_gym.compressor_surge.experts import (  # noqa: E402
    BATTERY,
    BATTERY_LEAD,
    BATTERY_START_CLOCK,
    BATTERY_STEPS,
    DEFAULT_GAINS,
    GUARD_MAX_TRAVEL,
    GUARD_MIN_DECAY,
    GUARD_MIN_MARGIN,
    GUARD_POINTS,
    GUARD_TRAVEL_STEPS,
    MPC_CLIP_SMOOTHING,
    MPC_ENERGY_WEIGHT,
    MPC_ERROR_SCALE_KPA,
    MPC_HORIZON,
    MPC_MOVE_WEIGHT,
    MPC_POWER_REF,
    MPC_SOLVE_BUDGET_S,
    MPC_STATES,
    MPC_SURGE_MARGIN,
    MPC_TERMINAL_WEIGHT,
    _state_at,
    battery_report,
    closed_loop_decay,
    commands_from_raw,
    load_gains,
    make_compressor_surge_mpc,
    pair_init,
    pair_step,
    raw_from_commands,
    settled_point,
    stage_openings,
)
from target_gym.experts.mpc import IPOPT_MAX_ITER  # noqa: E402

P = CompressorSurgeParams()
ENV = CompressorSurge()
RHO = float(suction_density(P))
A01 = float(sound_speed(P, np))
TWO_W = 2.0 * P.W
#: Speed PI of the heuristic policies (ours), the pair's default speed loop.
HEURISTIC_SPEED = {"kp_speed": 0.08, "ki_speed": 0.04}
#: The margin-blind chaser's valve gain, opening per kPa over the setpoint (ours).
CHASER_GAIN = 0.2
#: The guarded heuristic's valve bias and gain (ours): x = 0.2 + 0.4 (dp - p_ref)
#: in kPa, never below the anti-surge PI at its line.
GUARDED_BIAS, GUARDED_GAIN = 0.2, 0.4
#: Where heavy tables start their seeds.
HEAVY_SEED0 = 1000
#: ISA sea-level table values (read, U.S. Standard Atmosphere 1976).
ISA_DENSITY, ISA_SOUND = 1.2250, 340.294
#: Reference accuracy of the header-pressure transmitter, fraction of span
#: (read: Yokogawa EJA530E and Siemens SITRANS P320 data sheets), and the
#: calibrated span (ours), kPa.
TRANSMITTER_ACCURACY, TRANSMITTER_SPAN_KPA = 0.055e-2, 50.0


def say(s=""):
    print(s, flush=True)


def title(name, precision):
    say("=" * 78)
    say(f"{name}   [{precision}]")


def flow_scale(N=1.0, params=P):
    """m_c per unit Phi at speed N, rho01 A_c U_r N (kg/s)."""
    return float(suction_density(params)) * params.A_c * params.U_r * N


def head_scale(N=1.0, params=P):
    """dp per unit Psi at speed N, rho01 (U_r N)^2 (Pa)."""
    return float(suction_density(params)) * (params.U_r * N) ** 2


def psi(phi, params=P):
    return float(characteristic(phi, params, np))


def right_branch_phi(psi_value, params=P):
    """The right-branch Phi (>= 2W) with Psi_c(Phi) = ``psi_value``."""
    return brentq(lambda f: psi(f, params) - psi_value, TWO_W, 0.8316, xtol=1e-15)


def equilibrium(N, x, u, params=P, lo=TWO_W):
    """The equilibrium (Phi, m_c, dp) with Phi in [``lo``, 0.8316] at fixed
    speed, recycle and opening, by bisection on the env's own flow balance,
    float64. By default the right branch; raises ``ValueError`` where the
    balance has no root in the bracket."""
    flow, head = flow_scale(N, params), head_scale(N, params)

    def residual(phi):
        dp = head * psi(phi, params)
        return (
            flow * phi
            - float(consumer_flow(dp, u, params, np))
            - float(recycle_flow(dp, x, params, np))
        )

    phi = brentq(residual, lo, 0.8316, xtol=1e-15)
    return phi, flow * phi, head * psi(phi, params)


def line_flow(dp, margin, params=P):
    """Compressor flow on the line Phi = 2W (1 + margin) at header pressure
    ``dp``: the speed that gives ``dp`` there, times the flow per speed."""
    phi = TWO_W * (1.0 + margin)
    U = np.sqrt(dp / (float(suction_density(params)) * psi(phi, params)))
    return float(suction_density(params)) * params.A_c * U * phi


def state_of(
    m_c,
    dp,
    N,
    x,
    u,
    *,
    N_ramp=None,
    dev=0.0,
    setpoints=24.0e3,
    levels=None,
    block_clock=0,
    time=0,
    dtype=None,
):
    """One state (use inside ``enable_x64`` for float64); ``levels`` default to
    a constant opening ``u``."""
    dtype = dtype or jnp.zeros(()).dtype
    f = functools.partial(jnp.asarray, dtype=dtype)
    return CompressorSurgeState(
        time=jnp.asarray(time, jnp.int32),
        m_c=f(m_c),
        dp=f(dp),
        N=f(N),
        N_ramp=f(N if N_ramp is None else N_ramp),
        x=f(x),
        demand_dev=f(dev),
        phi_min=f(m_c / flow_scale(N)),
        setpoint_levels=f(jnp.broadcast_to(setpoints, (N_SETPOINT_BLOCKS,))),
        demand_levels=f(jnp.full(N_DEMAND_BLOCKS, u) if levels is None else levels),
        block_clock=jnp.asarray(block_clock, jnp.int32),
    )


@functools.partial(jax.jit, static_argnames=["n", "method"])
def _rollout(state, action, params, n, method="rk4_10"):
    """``n`` steps of ``compute_next_state`` (no trip, no restart) at a constant
    raw action and key. Returns the stacked states."""
    key = jax.random.PRNGKey(0)

    def body(s, _):
        s2 = compute_next_state(action, s, params, key, integration_method=method)[0]
        return s2, s2

    return jax.lax.scan(body, state, None, length=n)[1]


@functools.partial(jax.jit, static_argnames=["n"])
def substep_phi(state, action, params, n):
    """Phi at each of the ten substeps of ``n`` env steps from ``state`` at a
    constant raw action, shape (n, 10), with the deviation's noise off.

    The env's ``compute_next_state`` at ``delta_t / 10`` with one substep is
    one of its own substeps (a rate-limit update and one RK4 step). With the
    schedule clock scaled by ten, the deviation held within each step and
    updated between steps as the env updates it, the fine run reproduces the
    env's substeps (``--section integrator`` prints the match)."""
    fine = params.replace(
        delta_t=params.delta_t / 10,
        demand_block_steps=10 * params.demand_block_steps,
        setpoint_block_steps=10 * params.setpoint_block_steps,
        demand_theta=0.0,
        demand_sigma=0.0,
    )
    a = jnp.exp(-params.demand_theta * params.delta_t)
    key = jax.random.PRNGKey(0)
    scale = suction_density(params) * params.A_c * params.U_r

    def outer(s, _):
        def inner(ss, _):
            ss = compute_next_state(action, ss, fine, key, integration_method="rk4_1")[
                0
            ]
            return ss, ss.m_c / (scale * ss.N)

        s, phis = jax.lax.scan(inner, s, None, length=10)
        return s.replace(demand_dev=a * s.demand_dev), phis

    start = state.replace(block_clock=10 * state.block_clock)
    return jax.lax.scan(outer, start, None, length=n)[1]


def velocity_jacobian(m_c, dp, N, x, u, params=P):
    """d(dm_c/dt, d dp/dt)/d(m_c, dp) from ``compute_velocity``, float64,
    broadcast over array inputs. Returns (..., 2, 2)."""
    shape = np.broadcast_shapes(*(np.shape(v) for v in (m_c, dp, N, x, u)))
    flat = [
        np.broadcast_to(np.asarray(v, float), shape).ravel() for v in (m_c, dp, N, x, u)
    ]
    with enable_x64():

        def one(m, p, n, xx, uu):
            def f(y):
                v, _ = compute_velocity(
                    jnp.stack([y[0], y[1], n, jnp.zeros_like(n)]),
                    None,
                    n,
                    xx,
                    0.0,
                    jnp.full(N_DEMAND_BLOCKS, uu),
                    params,
                )
                return v[:2]

            return jax.jacfwd(f)(jnp.stack([m, p]))

        J = jax.jit(jax.vmap(one))(*(jnp.asarray(v, jnp.float64) for v in flat))
    return np.asarray(J).reshape(shape + (2, 2))


def step_eigenvalues(m_c, dp, N, x, u, params=P):
    """ln(mu) / dt for the (m_c, dp) block of the one-step Jacobian of
    ``compute_next_state``, the commands holding N and x, float64."""
    key = jax.random.PRNGKey(0)
    with enable_x64():
        state = state_of(m_c, dp, N, x, u)
        action = jnp.asarray(raw_from_commands(N, x, params), jnp.float64)

        def f(y):
            s = compute_next_state(
                action, state.replace(m_c=y[0], dp=y[1]), params, key
            )[0]
            return jnp.stack([s.m_c, s.dp])

        J = np.asarray(jax.jacfwd(f)(jnp.array([m_c, dp], jnp.float64)))
    return np.log(np.linalg.eigvals(J).astype(complex)) / params.delta_t


def upper(lams):
    return lams[np.argmax(lams.imag)]


# ---------------------------------------------------------------------------
# Closed-loop machinery (float32, as the env ships)
# ---------------------------------------------------------------------------

MODES = {"pair": 0, "fixed": 1, "chaser": 2, "guarded": 3, "constant": 4, "random": 5}


def policy(kind, bias=0.0, line=None, raw=(0.0, 0.0), gains=None):
    """A controller spec for the runner.

    ``pair``: the pair at ``gains`` (``load_gains()`` by default).
    ``fixed``: the heuristic speed PI with the recycle held at ``bias``.
    ``chaser``: the speed PI with x = bias + 0.2 (dp - setpoint) in kPa, blind
    to the margin. ``guarded``: x = 0.2 + 0.4 (dp - setpoint), never below the
    anti-surge PI (3, 0.3) at control line ``line`` with its override at
    0.4 ``line``. ``constant``: the raw action ``raw``. ``random``: uniform
    raw actions, drawn per step from a stream apart from the env's.
    """
    g = dict(load_gains() if gains is None else gains)
    kpx = 0.0
    if kind in ("fixed", "chaser", "guarded"):
        g.update(HEURISTIC_SPEED)
    if kind == "chaser":
        kpx = CHASER_GAIN
    if kind == "guarded":
        g.update(
            kp_surge=3.0,
            ki_surge=0.3,
            surge_line=line,
            override_line=0.4 * line,
            override_hold=2.0,
            reset_band=0.05,
        )
        bias, kpx = GUARDED_BIAS, GUARDED_GAIN
    return {
        "mode": np.int32(MODES[kind]),
        "bias": np.float32(bias),
        "kpx": np.float32(kpx),
        "raw": np.asarray(raw, np.float32),
        "gains": {k: np.float32(g[k]) for k in DEFAULT_GAINS},
    }


def _act(spec, ps, obs, key, params):
    """One step of the controller ``spec`` (``policy``): its raw action and
    the pair's next memory. Every heuristic takes the pair's speed command
    and sets its own recycle opening; the valve half of the action map is
    the env's, through ``commands_from_raw`` and ``raw_from_commands``."""
    u_pair, ps = pair_step(spec["gains"], ps, obs, params, xp=jnp)
    N_pair, x_pair = commands_from_raw(u_pair, params, jnp)
    x_chase = jnp.clip(spec["bias"] + spec["kpx"] * (obs[0] - obs[6]), 0.0, 1.0)
    mode = spec["mode"]
    x = jnp.select(
        [mode == 1, mode == 2, mode == 3],
        [spec["bias"], x_chase, jnp.maximum(x_chase, x_pair)],
        x_pair,
    )
    # The speed half stays the pair's raw command, unchanged.
    u = jnp.stack([u_pair[0], raw_from_commands(N_pair, x, params, jnp)[1]])
    u = jnp.where(mode == 4, spec["raw"], u)
    u = jnp.where(mode == 5, jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0), u)
    return u, ps


REC = ("m_c", "dp", "N", "x", "opening", "phi_min", "dev", "e", "tripped")


@functools.lru_cache(maxsize=None)
def _runner(n_steps, record=False):
    """A jitted batch of episodes on the env's own ``step_env`` (trips and
    fresh restarts included, under a constant rollout key as the suite's
    rollouts run), each with its own start state, key and controller spec.

    Per episode it returns sums over untripped steps (squared error in kPa^2,
    |error|, running cost, recycle power, valve travel, version 1), the trip count, the
    smallest margin phi_min / 2W - 1 and steps under the pair's override
    line. With ``record``, per-step arrays of ``REC``."""

    def one(state, key, spec, params):
        policy_key = jax.random.fold_in(key, 0x5EED)
        ps = pair_init(get_obs(state, params), xp=jnp)
        f32 = jnp.zeros((), jnp.float32)
        acc0 = dict(
            sq_e=f32,
            abs_e=f32,
            max_e=f32,
            run=f32,
            power=f32,
            v1=f32,
            travel=f32,
            trips=jnp.zeros((), jnp.int32),
            ok=jnp.zeros((), jnp.int32),
            below=jnp.zeros((), jnp.int32),
            min_margin=jnp.full((), jnp.inf, jnp.float32),
        )

        def body(carry, _):
            s, ps, acc = carry
            obs = get_obs(s, params)
            u, ps = _act(spec, ps, obs, jax.random.fold_in(policy_key, s.time), params)
            _, s2, _, _, info = ENV.step_env(key, s, u, params)
            tripped = info["tripped"]
            ok = jnp.logical_not(tripped)
            # On an untripped step the continuing state is the scored one.
            e = (live_target(s2, params) - s2.dp) / KPA
            terms = compute_reward_terms(s2, params)
            margin = s2.phi_min / TWO_W - 1.0
            v1 = jnp.where(
                tripped, -1.0 * params.restart_steps, compute_reward_v1(s2, params)
            )
            okf = ok.astype(jnp.float32)
            acc = dict(
                sq_e=acc["sq_e"] + okf * e**2,
                abs_e=acc["abs_e"] + okf * jnp.abs(e),
                max_e=jnp.maximum(acc["max_e"], okf * jnp.abs(e)),
                run=acc["run"] + okf * terms["running"],
                power=acc["power"] + okf * recycle_power(s2, params),
                v1=acc["v1"] + v1,
                travel=acc["travel"] + okf * jnp.abs(s2.x - s.x),
                trips=acc["trips"] + tripped.astype(jnp.int32),
                ok=acc["ok"] + ok.astype(jnp.int32),
                below=acc["below"] + (ok & (margin < 0.075)).astype(jnp.int32),
                min_margin=jnp.where(
                    ok, jnp.minimum(acc["min_margin"], margin), acc["min_margin"]
                ),
            )
            out = None
            if record:
                opening = demand_opening(
                    s2.demand_levels,
                    s2.block_clock * params.delta_t,
                    s2.demand_dev,
                    params,
                )
                out = (
                    s2.m_c,
                    s2.dp,
                    s2.N,
                    s2.x,
                    opening,
                    s2.phi_min,
                    s2.demand_dev,
                    e,
                    tripped,
                )
            return (s2, ps, acc), out

        (_, _, acc), out = jax.lax.scan(body, (state, ps, acc0), None, length=n_steps)
        return acc, out

    return jax.jit(jax.vmap(one, in_axes=(0, 0, 0, None)))


@functools.lru_cache(maxsize=None)
def _resetter():
    return jax.jit(jax.vmap(lambda k, p: ENV.reset_env(k, p)[1], in_axes=(0, None)))


def reset_batch(seeds, params=P):
    """``reset_env`` for each seed, with the seed's key as the rollout key (as
    ``runners.rollout`` does)."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(np.asarray(seeds), jnp.uint32))
    return _resetter()(keys, params), keys


def run(spec, seeds, params=P, n_steps=None, record=False, chunk=1024, states=None):
    """Episodes of one controller spec over ``seeds``. Returns (summary,
    record) as numpy arrays over the episodes."""
    n_steps = int(params.max_steps_in_episode) if n_steps is None else n_steps
    seeds = np.asarray(seeds)
    summaries, records = [], []
    for i in range(0, len(seeds), chunk):
        part = seeds[i : i + chunk]
        s0, keys = reset_batch(part, params)
        if states is not None:
            s0 = jax.tree_util.tree_map(lambda a: a[i : i + chunk], states)
        spec_b = jax.tree_util.tree_map(
            lambda a: jnp.broadcast_to(jnp.asarray(a), (len(part),) + np.shape(a)), spec
        )
        acc, out = _runner(n_steps, record)(s0, keys, spec_b, params)
        summaries.append({k: np.asarray(v) for k, v in acc.items()})
        if record:
            records.append(dict(zip(REC, (np.asarray(v) for v in out))))
    summary = {k: np.concatenate([s[k] for s in summaries]) for k in summaries[0]}
    rec = None
    if record:
        rec = {k: np.concatenate([r[k] for r in records]) for k in REC}
    return summary, rec


def costs(summary, params=P):
    """Per episode, the version-2 cost (tracking + running + trips at the trip
    cost), recomputed in float64 from the sums, so a different ``e_floor``
    needs no re-run."""
    tracking = summary["sq_e"].astype(float) / params.e_floor**2
    return tracking + summary["run"] + summary["trips"] * R.trip_cost(params)


def one_sided_bound(k, n, level=0.95):
    """Clopper-Pearson upper bound on a per-episode rate after k in n."""
    return float(beta.ppf(level, k + 1, n - k)) if k < n else 1.0


def poisson_interval(k, n, level=0.95):
    """Exact two-sided interval on a rate per episode after k events in n."""
    a = 1.0 - level
    lo = 0.0 if k == 0 else chi2.ppf(a / 2, 2 * k) / 2
    hi = chi2.ppf(1 - a / 2, 2 * k + 2) / 2
    return lo / n, hi / n


# ---------------------------------------------------------------------------
# Fast sections
# ---------------------------------------------------------------------------


def section_params():
    title("params: constants, derived groups, envelope, costs", "float64")
    say(
        f"  rho01 = P01 / (R T01) = {RHO:.5f} kg/m^3 (ISA {ISA_DENSITY}, "
        f"{RHO / ISA_DENSITY - 1:+.1e} relative); a01 = {A01:.3f} m/s (ISA "
        f"{ISA_SOUND}, {A01 / ISA_SOUND - 1:+.1e})"
    )
    w = float(helmholtz_omega(P, np))
    say(
        f"  omega_H = {w:.4f} rad/s, f_H = {w / (2 * math.pi):.4f} Hz, period "
        f"{2 * math.pi / w:.4f} s = {2 * math.pi / w / P.delta_t:.2f} steps"
    )
    say(
        "  B at 70 / 87.5 / 100 / 105 %: "
        + " / ".join(
            f"{float(greitzer_B(N, P, np)):.3f}" for N in (0.70, 0.875, 1.0, 1.05)
        )
    )
    peak = P.psi_c0 + 2 * P.H
    say(
        f"  peak head coefficient {peak:.2f}; rated peak {head_scale() * peak / KPA:.3f} kPa, "
        f"pressure ratio {(P.P01 + head_scale() * peak) / P.P01:.4f}; 105 % peak "
        f"{head_scale(P.N_max) * peak / KPA:.3f} kPa"
    )
    K = float(surge_constant(P))
    say(
        f"  surge line dp = K m^2, K = {K:.3f} Pa/(kg/s)^2; surge flow at rated "
        f"{float(surge_flow_per_speed(P)):.4f} kg/s"
    )
    say(
        "  surge flow at 20 / 24 / 28 kPa: "
        + " / ".join(f"{math.sqrt(p / K):.3f}" for p in (20e3, 24e3, 28e3))
        + " kg/s; on the 15 % control line: "
        + " / ".join(f"{line_flow(p, 0.15):.3f}" for p in (20e3, 24e3, 28e3))
        + " kg/s"
    )
    y = P.phi_join / P.W - 1.0
    slope = 1.5 * P.H / P.W * (1 - y * y)
    zero_head = P.phi_join - psi(P.phi_join) / slope
    say(
        f"  tangent line at phi_join {P.phi_join}: slope {slope:.4f}, zero head at "
        f"Phi {zero_head:.4f}"
    )
    phi_lo, _, dp_lo = equilibrium(P.N_min, 1.0, 1.0)
    top = P.p_ref_range[1] / KPA
    say(
        f"  lowest untripped pressure: {dp_lo / KPA:.3f} kPa (70 %, recycle and "
        f"consumers fully open, Phi {phi_lo:.4f}); dp_error_max = max({top:.0f} - "
        f"{dp_lo / KPA:.3f}, {head_scale(P.N_max) * peak / KPA:.3f} - "
        f"{P.p_ref_range[0] / KPA:.0f}) = {top - dp_lo / KPA:.3f} kPa; params "
        f"{P.dp_error_max}"
    )
    fc = 2.0 * (P.dp_error_max / P.e_floor) ** 2
    say(
        f"  failure_cost = 2 (dp_error_max / e_floor)^2 = {fc:.5g} per step at "
        f"e_floor {P.e_floor} kPa (params {P.failure_cost:.5g}); trip cost = "
        f"restart_steps x failure_cost = {P.restart_steps} x {fc:.5g} = "
        f"{R.trip_cost(P):.5g}"
    )
    floor = TRANSMITTER_ACCURACY * TRANSMITTER_SPAN_KPA
    say(
        f"  precision_floor = {TRANSMITTER_ACCURACY * 100:.3f} % of a 0 to "
        f"{TRANSMITTER_SPAN_KPA:.0f} kPa span = {floor:.4f} kPa (params "
        f"{P.precision_floor}); one step at 1 kPa error costs "
        f"{(1.0 / P.e_floor) ** 2:.0f}"
    )
    a = math.exp(-P.demand_theta * P.delta_t)
    say(
        f"  deviation: a = exp(-theta dt) = {a:.6f}; innovation sd = "
        f"{P.demand_sigma * math.sqrt(1 - a * a):.5f} of opening; reset clip "
        f"+-{P.demand_dev_reset_clip * P.demand_sigma:.2f}"
    )
    ratio = P.k_r * math.sqrt(K)
    say(
        f"  recycle at full opening passes k_r sqrt(K) = {ratio:.3f} x the surge flow "
        f"at any pressure; it carries the surge flow alone from an opening of "
        f"{1 / ratio:.4f}"
    )
    for label, phi, N in (
        ("rated peak", TWO_W, 1.0),
        ("Phi 0.777 at 105 %", 0.777, 1.05),
    ):
        mach = phi * P.U_r * N / A01
        say(f"  duct Mach at the {label}: {mach:.3f}")
    say(
        f"  episode: {P.max_steps_in_episode} steps = {N_SETPOINT_BLOCKS} x "
        f"{P.setpoint_block_steps} = {N_DEMAND_BLOCKS} x {P.demand_block_steps} = "
        f"{P.max_steps_in_episode * P.delta_t:.0f} s"
    )


def _t63(state, action, n=600, params=P):
    """Time for dp to cover 63.2 % of its change after ``action`` from
    ``state``, deviation off, float64, linear interpolation between steps."""
    with enable_x64():
        dp = np.asarray(_rollout(state, jnp.asarray(action, jnp.float64), params, n).dp)
    dp = np.concatenate([[float(state.dp)], dp])
    frac = (dp - dp[0]) / (dp[-1] - dp[0])
    k = int(np.argmax(frac >= 1 - math.exp(-1)))
    t = (k - 1 + (0.632 - frac[k - 1]) / (frac[k] - frac[k - 1])) * params.delta_t
    return t, dp[-1]


def section_steady():
    title("steady: feasibility, forced recycle, equilibria, t63", "float64")
    for dp in (20e3, 28e3):
        phi = right_branch_phi(dp / head_scale(P.N_max))
        say(
            f"  at {dp / KPA:.0f} kPa: 105 % delivers {flow_scale(P.N_max) * phi:.3f} "
            f"kg/s (Phi {phi:.4f}); the consumers take "
            f"{float(consumer_flow(dp, 1.0, P, np)):.3f} kg/s at full opening"
        )
    rng = np.random.default_rng(0)
    dp = rng.uniform(*P.p_ref_range, 100_000)
    u = rng.uniform(*P.demand_range, 100_000)
    m_d = np.asarray(consumer_flow(dp, u, P, np))
    shares = []
    for margin in (0.0, 0.05, 0.10, 0.15):
        m_line = np.vectorize(lambda p, s=margin: line_flow(p, s))(dp[:20000])
        shares.append(
            (m_d[:20000] < m_line).mean()
            if margin
            else (m_d < np.sqrt(dp / float(surge_constant(P)))).mean()
        )
    say(
        "  blocks forcing recycle at margins 0 / 5 / 10 / 15 %: "
        + " / ".join(f"{s:.3f}" for s in shares)
        + " (1e5 draws at 0, 2e4 at the others)"
    )
    grid_dp = np.linspace(*P.p_ref_range, 41)[:, None]
    grid_u = np.linspace(*P.demand_range, 41)[None, :]
    m_d = np.asarray(consumer_flow(grid_dp, grid_u, P, np))
    per_x = np.asarray(recycle_flow(grid_dp, 1.0, P, np))
    phi_max = np.vectorize(lambda p: right_branch_phi(p / head_scale(P.N_max)))(grid_dp)
    x_max = (flow_scale(P.N_max) * phi_max - m_d) / per_x
    row = []
    for margin in (0.0, 0.05, 0.10, 0.15):
        m_line = np.vectorize(lambda p, s=margin: line_flow(p, s))(grid_dp)
        row.append(np.maximum((m_line - m_d) / per_x, 0.0).max())
    say(
        "  fixed-recycle band: the smallest opening that keeps every block off "
        "the line at margins 0 / 5 / 10 / 15 %: "
        + " / ".join(f"{v:.3f}" for v in row)
        + f"; the largest at which 105 % holds every block: {x_max.min():.3f}"
    )
    speeds = []
    for p in (20e3, 28e3):
        for uu in (0.35, 0.95, 1.0):
            m = float(consumer_flow(p, uu, P, np))
            if m < math.sqrt(p / float(surge_constant(P))):
                speeds.append(math.sqrt(p / (head_scale() * (P.psi_c0 + 2 * P.H))))
            else:
                c = RHO * P.A_c**2 * p / m**2
                phi = brentq(lambda f: psi(f) / f**2 - c, TWO_W, 0.8316, xtol=1e-15)
                speeds.append(m / (flow_scale() * phi))
    say(
        f"  speed each schedule corner needs (surge line where recycle is forced): "
        f"{min(speeds) * 100:.1f} to {max(speeds) * 100:.1f} %"
    )
    openings = np.linspace(0.2, 1.0, 81)
    zero = np.array([equilibrium(0.875, 0.5, uu) for uu in openings])
    say(
        f"  zero action (87.5 %, recycle 0.5), openings 0.2 to 1.0: Phi "
        f"{zero[:, 0].min():.4f} to {zero[:, 0].max():.4f}, dp "
        f"{zero[:, 2].min() / KPA:.2f} to {zero[:, 2].max() / KPA:.2f} kPa"
    )
    reset = np.array([equilibrium(P.N_reset, P.x_reset, uu) for uu in openings])
    lowest = P.demand_range[0] - P.demand_dev_reset_clip * P.demand_sigma
    first = np.array(
        [equilibrium(P.N_reset, P.x_reset, uu) for uu in np.linspace(lowest, 1.0, 72)]
    )
    say(
        f"  reset (100 %, recycle 0.35), openings 0.2 to 1.0: Phi "
        f"{reset[:, 0].min():.4f} to {reset[:, 0].max():.4f}, dp "
        f"{reset[:, 2].min() / KPA:.2f} to {reset[:, 2].max() / KPA:.2f} kPa; over "
        f"the drawable first openings {lowest:.2f} to 1.0: Phi {first[:, 0].min():.4f} "
        f"to {first[:, 0].max():.4f}"
    )
    say(
        f"  dead-head speed (shutoff head psi_c0 rho U^2 = static head): "
        f"{math.sqrt(P.dp_out / (head_scale() * P.psi_c0)) * 100:.2f} %"
    )
    for margin, name in ((0.0, "surge line"), (0.15, "control line")):
        phi = TWO_W * (1 + margin)
        N20, N28 = (math.sqrt(p / (head_scale() * psi(phi))) for p in (20e3, 28e3))
        say(
            f"  on the {name}: speed {N20 * 100:.2f} % at 20 kPa, {N28 * 100:.2f} % at "
            f"28 kPa; a 20 to 28 kPa move needs {(N28 - N20) * 100:.2f} % of rated, "
            f"{(N28 - N20) / P.drive_rate:.1f} s at the drive rate plus the {P.tau_N:.0f} s lag"
        )
    quiet = P.replace(demand_sigma=0.0)
    _, m0, dp0 = equilibrium(0.875, 0.5, 0.9)
    with enable_x64():
        base = state_of(m0, dp0, 0.875, 0.5, 0.9)
    rows = []
    for label, N, x in (
        ("speed 87.5 to 88.5 %", 0.885, 0.5),
        ("speed 87.5 to 105 %", 1.05, 0.5),
        ("recycle 0.5 to 0.45", 0.875, 0.45),
        ("recycle 0.5 to 1.0", 0.875, 1.0),
        ("recycle 0.5 to 0.0", 0.875, 0.0),
    ):
        t, _ = _t63(base, raw_from_commands(N, x, P), params=quiet)
        rows.append(f"{label} {t:.2f} s")
    say("  t63 of dp from the zero-action point at opening 0.9: " + "; ".join(rows))


def section_linear():
    title("linear: Helmholtz pair, damping, boundaries, Fig. 2.10, Hopf", "float64")
    w = float(helmholtz_omega(P, np))
    m = flow_scale() * TWO_W
    dp = head_scale() * (P.psi_c0 + 2 * P.H)
    u_peak = m / float(consumer_flow(dp, 1.0, P, np))
    with enable_x64():
        g = float(jax.grad(lambda p: consumer_flow(p, u_peak, P))(jnp.float64(dp)))
    re = -(A01**2 / P.V_p) * g / 2.0
    lam = upper(step_eigenvalues(m, dp, 1.0, 0.0, u_peak))
    say(
        f"  on the peak (rated, recycle shut, opening {u_peak:.4f}): one-step "
        f"eigenvalue {lam.real:.4f} +- {lam.imag:.4f}i /s, modulus {abs(lam):.4f} "
        f"against omega_H {w:.4f}; closed-form real part {re:.4f}"
    )

    def pair_at(p, uu, x=0.0):
        mm = float(consumer_flow(p, uu, P, np)) + float(recycle_flow(p, x, P, np))
        c = RHO * P.A_c**2 * p / mm**2
        phi = brentq(lambda f: psi(f) / f**2 - c, TWO_W, 0.8316, xtol=1e-15)
        N = mm / (flow_scale() * phi)
        la = upper(np.linalg.eigvals(velocity_jacobian(mm, p, N, x, uu)))
        return -la.real / abs(la), la.imag / (2 * math.pi), phi, N

    zeta, f, phi, N = pair_at(24e3, 0.8)
    say(
        f"  off-peak, 24 kPa, demand 0.8, recycle shut: Phi {phi:.4f}, speed "
        f"{N * 100:.2f} %, zeta {zeta:.3f}, damped {f:.3f} Hz"
    )
    margins = np.linspace(0.10, 0.0, 41)
    phis = TWO_W * (1 + margins)
    psis = np.asarray(characteristic(phis, P, np))
    mm = np.sqrt(24e3 * RHO * P.A_c**2 * phis**2 / psis)
    NN = mm / (flow_scale() * phis)
    uu = mm / float(consumer_flow(24e3, 1.0, P, np))
    lams = np.linalg.eigvals(velocity_jacobian(mm, 24e3, NN, 0.0, uu))
    up = lams[np.arange(len(lams)), np.argmax(lams.imag, axis=1)]
    z = -up.real / np.abs(up)
    say(
        f"  damping at 24 kPa, recycle shut: zeta {z[margins.round(4) == 0.08][0]:.3f} "
        f"at 8 %, {z[-1]:.3f} on the line, monotone over 41 margins: "
        f"{bool(np.all(np.diff(z) < 0))}"
    )

    Ns = np.linspace(P.N_min, P.N_max, 15)
    xs = np.linspace(0.0, 1.0, 21)
    phi_g = np.linspace(0.45, 0.5, 2001)
    Ng, Xg, Pg = np.meshgrid(Ns, xs, phi_g, indexing="ij")
    dpg = RHO * (P.U_r * Ng) ** 2 * np.asarray(characteristic(Pg, P, np))
    mg = RHO * P.A_c * P.U_r * Ng * Pg
    ug = (mg - np.asarray(recycle_flow(dpg, Xg, P, np))) / np.asarray(
        consumer_flow(dpg, 1.0, P, np)
    )
    valid = (ug >= 0.2) & (ug <= 1.0)
    reaches = valid[..., -1]
    J = velocity_jacobian(
        mg[reaches], dpg[reaches], Ng[reaches], Xg[reaches], ug[reaches]
    )
    trace = J[..., 0, 0] + J[..., 1, 1]
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    slivers, static = [], 0
    for tr, de, ok, (i, k) in zip(trace, det, valid[reaches], np.argwhere(reaches)):
        start = np.flatnonzero(~ok).max() + 1 if (~ok).any() else 0
        unstable = np.flatnonzero(tr[start:] >= 0.0)
        if unstable.size:
            slivers.append((1.0 - phi_g[start + unstable.max()] / TWO_W, Ns[i], xs[k]))
        static += bool((de[start:] <= 0.0).any())
    slivers = np.array(slivers)
    lo, hi = slivers[:, 0].argmin(), slivers[:, 0].argmax()
    say(
        f"  action box (15 speeds x 21 openings): {int(reaches.sum())} families reach "
        f"the peak with opening in [0.2, 1]; trace on the peak at most "
        f"{trace[:, -1].max():.3f} /s; {len(slivers)} meet a Hopf point, the stable "
        f"sliver {slivers[lo, 0] * 100:.2f} % ({slivers[lo, 1] * 100:.0f} %, recycle "
        f"{slivers[lo, 2]:.2f}) to {slivers[hi, 0] * 100:.2f} % ({slivers[hi, 1] * 100:.0f} %, "
        f"recycle {slivers[hi, 2]:.2f}); static boundaries within 10 % of the peak: {static}"
    )

    scale = P.L_c / P.U_r  # s per unit of Gravdahl's s = U t / L_c at rated speed
    for gain in (0.65, 0.61):
        p0 = P.replace(dp_out=0.0, demand_sigma=0.0)
        uu = gain * P.A_c * math.sqrt(RHO) / P.k_d
        fl, hd = flow_scale(1.0, p0), head_scale(1.0, p0)
        phi = brentq(
            lambda f: fl * f - float(consumer_flow(hd * psi(f, p0), uu, p0, np)),
            0.3,
            0.8,
            xtol=1e-15,
        )
        la = upper(step_eigenvalues(fl * phi, hd * psi(phi, p0), 1.0, 0.0, uu, p0))
        say(
            f"  Gravdahl throttle gain {gain} (opening {uu:.4f}, no static head): "
            f"equilibrium Phi {phi:.4f}, Psi {psi(phi, p0):.4f}; eigenvalue "
            f"{la.real:.3f} +- {la.imag:.3f}i /s = {la.real * scale:+.4f} +- "
            f"{la.imag * scale:.3f}i per unit s"
        )

    # Hopf criticality at rated speed, recycle shut: along the family of
    # openings, the Hopf point left of the peak, then runs just past it.
    phis = np.linspace(TWO_W, 0.47, 3001)
    dps = head_scale() * np.asarray(characteristic(phis, P, np))
    ms = flow_scale() * phis
    us = ms / np.asarray(consumer_flow(dps, 1.0, P, np))
    Jh = velocity_jacobian(ms, dps, 1.0, 0.0, us)
    tr = Jh[..., 0, 0] + Jh[..., 1, 1]
    k = int(np.argmax(tr >= 0.0))
    phi_h = phis[k]
    say(
        f"  Hopf at rated, recycle shut: Phi {phi_h:.4f} ({(1 - phi_h / TWO_W) * 100:.2f} % "
        f"left of the peak), opening {us[k]:.4f}"
    )
    quiet = P.replace(demand_sigma=0.0)
    cycle = None
    for past in (0.001, 0.003, 0.01):
        phi_e = phi_h * (1 - past)
        dpe = head_scale() * psi(phi_e)
        ue = flow_scale() * phi_e / float(consumer_flow(dpe, 1.0, P, np))
        with enable_x64():
            s0 = state_of(flow_scale() * phi_e * 1.001, dpe, 1.0, 0.0, ue)
            states = _rollout(
                s0,
                jnp.asarray(raw_from_commands(1.0, 0.0, P), jnp.float64),
                quiet,
                1200,
            )
        phi_t = np.asarray(states.m_c)[-200:] / flow_scale()
        if cycle is None:
            cycle = (float(states.m_c[-1]), float(states.dp[-1]))
        say(
            f"    {past * 100:.1f} % past it, from 0.1 % above the equilibrium flow: after "
            f"120 s Phi swings {phi_t.min():.4f} to {phi_t.max():.4f}"
        )
    # Hysteresis: started on that deep-surge cycle at openings where the
    # equilibrium is stable.
    for right in (0.003, 0.01, 0.03, 0.10):
        phi_e = phi_h * (1 + right)
        dpe = head_scale() * psi(phi_e)
        ue = flow_scale() * phi_e / float(consumer_flow(dpe, 1.0, P, np))
        with enable_x64():
            s0 = state_of(cycle[0], cycle[1], 1.0, 0.0, ue)
            states = _rollout(
                s0,
                jnp.asarray(raw_from_commands(1.0, 0.0, P), jnp.float64),
                quiet,
                1200,
            )
        phi_t = np.asarray(states.m_c)[-200:] / flow_scale()
        say(
            f"    {right * 100:.1f} % right of it (Phi {phi_e:.4f}, a stable equilibrium), "
            f"started on that cycle: after 120 s Phi swings {phi_t.min():.4f} to {phi_t.max():.4f}"
        )


def section_anchor():
    title(
        "anchor: Gravdahl Fig. 2.8, deep surge in the task's configuration", "float64"
    )
    p0 = P.replace(dp_out=0.0, demand_sigma=0.0, delta_t=0.005)
    uu = 0.61 * P.A_c * math.sqrt(RHO) / P.k_d
    with enable_x64():
        s0 = state_of(0.6 * flow_scale(), 0.6 * head_scale(), 1.0, 0.0, uu)
        states = _rollout(
            s0, jnp.asarray(raw_from_commands(1.0, 0.0, P), jnp.float64), p0, 440
        )
    phi = np.concatenate([[0.6], np.asarray(states.m_c) / flow_scale()])
    ps = np.concatenate([[0.6], np.asarray(states.dp) / head_scale()])
    down = np.flatnonzero((phi[:-1] >= 0.2) & (phi[1:] < 0.2)) * p0.delta_t
    say(
        f"  Fig. 2.8 (B {float(greitzer_B(1.0, P, np)):.3f}, gain 0.61, opening {uu:.4f}, "
        f"from Phi = Psi = 0.6, 2.2 s at 5 ms steps): Phi {phi.min():.3f} to "
        f"{phi.max():.3f}, Psi {ps.min():.3f} to {ps.max():.3f}; flow collapses at "
        + ", ".join(f"{t:.2f} s (s = {t * P.U_r / P.L_c:.1f})" for t in down)
    )
    fine = P.replace(demand_sigma=0.0, delta_t=0.01)
    w = float(helmholtz_omega(P, np))
    for N in (1.0, 0.70):
        phi0, m0, dp0 = equilibrium(N, P.x_reset, 0.2, lo=0.3)
        with enable_x64():
            s0 = state_of(m0, dp0, N, P.x_reset, 0.2)
            states = _rollout(
                s0,
                jnp.asarray(raw_from_commands(N, 0.0, P), jnp.float64),
                fine,
                6000,
                "rk4_1",
            )
        m = np.asarray(states.m_c)
        tail = m[-3000:]
        mid = 0.5 * (tail.max() + tail.min())
        ups = np.flatnonzero((tail[:-1] < mid) & (tail[1:] >= mid))
        period = np.diff(ups).mean() * fine.delta_t if len(ups) > 2 else float("nan")
        say(
            f"  {N * 100:.0f} %, from the equilibrium at recycle 0.35 (Phi {phi0:.4f}), "
            "recycle closing to shut, opening 0.2, trip off, "
            f"10 ms samples: first 5 s m_c {m[:500].min():.2f} to {m[:500].max():.2f} kg/s; "
            f"last 30 s {tail.min():.2f} to {tail.max():.2f} kg/s, period {period:.3f} s = "
            f"{period * w / (2 * math.pi):.2f} Helmholtz periods; dp over the last 30 s "
            f"{np.asarray(states.dp)[-3000:].min() / KPA:.2f} to "
            f"{np.asarray(states.dp)[-3000:].max() / KPA:.2f} kPa (shutoff head "
            f"{head_scale(N) * P.psi_c0 / KPA:.2f}, static head {P.dp_out / KPA:.0f})"
        )


def section_reach():
    title("reach: zero action, full travel, random actions, reset margins", "float32")
    seeds = np.arange(20)
    for label, raw in (("zero action", (0.0, 0.0)), ("full travel high", (1.0, 1.0))):
        summ, rec = run(policy("constant", raw=raw), seeds, record=True)
        phi = rec["m_c"] / (flow_scale() * rec["N"])
        say(
            f"  {label}, 20 schedules x 1200 steps: {int(summ['trips'].sum())} trips; "
            f"smallest phi_min {rec['phi_min'].min():.4f}; Phi at most {phi.max():.4f}; "
            f"dp {rec['dp'].min() / KPA:.2f} to {rec['dp'].max() / KPA:.2f} kPa; all "
            f"finite: {bool(all(np.isfinite(v).all() for v in rec.values()))}"
        )
    firsts = []
    for level in (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95, 1.0):
        p = P.replace(demand_range=(level, level))
        _, rec = run(policy("constant", raw=(-1.0, -1.0)), [0], params=p, record=True)
        t = rec["tripped"][0]
        firsts.append(f"{level:.2f}: {int(np.argmax(t)) + 1 if t.any() else 'never'}")
    say(
        "  full travel low (70 %, shut) from the PRNGKey(0) reset, first trip by level: "
        + ", ".join(firsts)
    )
    summ, rec = run(policy("constant", raw=(-1.0, -1.0)), seeds, record=True)
    first = np.array([int(np.argmax(t)) + 1 for t in rec["tripped"]])
    say(
        f"  full travel low, 20 schedules: trips in {int((summ['trips'] > 0).sum())} of 20, "
        f"{summ['trips'].mean():.1f} per episode; first trip at step {first.min()} to "
        f"{first.max()}, median {np.median(first):.1f}; highest dp before a trip "
        f"{rec['dp'][~rec['tripped']].max() / KPA:.2f} kPa"
    )
    summ, _ = run(policy("random"), np.arange(64))
    say(
        f"  uniform random actions: trips {int(summ['trips'].sum())}, in "
        f"{int((summ['trips'] > 0).sum())} of 64 episodes; smallest margin "
        f"{summ['min_margin'].min():.4f}"
    )
    # Zero action's lowest Phi: the reset at the lowest drawable first opening,
    # then a drop straight to the clip floor 0.2.
    lowest = P.demand_range[0] - P.demand_dev_reset_clip * P.demand_sigma
    phi_r, m0, dp0 = equilibrium(P.N_reset, P.x_reset, lowest)
    quiet = P.replace(demand_sigma=0.0)
    with enable_x64():
        s0 = state_of(m0, dp0, P.N_reset, P.x_reset, 0.2)
        states = _rollout(s0, jnp.zeros(2, jnp.float64), quiet, 300)
    say(
        f"  zero action's lowest Phi: the reset at opening {lowest:.2f} sits at "
        f"{phi_r:.4f}; a drop to 0.2 straight after reaches "
        f"{float(np.min(states.phi_min)):.4f} (soft minimum; float64); the zero-action "
        f"equilibrium at 0.2 is {equilibrium(0.875, 0.5, 0.2)[0]:.4f}"
    )
    for label, raw in (("zero action", (0.0, 0.0)), ("full travel low", (-1.0, -1.0))):
        _, rec = run(
            policy("constant", raw=raw), np.arange(256), n_steps=1, record=True
        )
        say(
            f"  first step after 256 resets, {label}: smallest phi_min "
            f"{rec['phi_min'].min():.4f}, trips {int(rec['tripped'].sum())}"
        )


def section_integrator():
    title(
        "integrator: stiffness, convergence, soft minimum, reset Newton",
        "float64 / float32",
    )
    # Stiffest eigenvalue over envelope equilibria and along full travel high.
    pts = []
    for N in np.linspace(P.N_min, P.N_max, 8):
        for x in np.linspace(0.0, 1.0, 6):
            for uu in np.linspace(0.2, 1.0, 9):
                try:
                    phi, m, dp = equilibrium(N, x, uu)
                except ValueError:
                    continue
                pts.append((m, dp, N, x, uu))
    pts = np.array(pts)
    lam_eq = np.linalg.eigvals(velocity_jacobian(*pts.T))
    _, rec = run(policy("constant", raw=(1.0, 1.0)), np.arange(20), record=True)
    sel = (slice(None), slice(None, None, 5))
    lam_run = np.linalg.eigvals(
        velocity_jacobian(
            *(rec[k][sel].ravel() for k in ("m_c", "dp", "N", "x", "opening"))
        )
    )
    h = P.delta_t / 10
    for label, lam in (
        (f"{len(pts)} right-branch equilibria over the action box", lam_eq),
        ("full travel high, 20 runs", lam_run),
    ):
        z = h * lam.ravel()
        amp = np.abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)
        say(
            f"  {label}: largest |lambda| {np.abs(lam).max():.1f} /s, h |lambda| "
            f"{np.abs(z).max():.3f} (real-axis limit 2.785); largest RK4 amplification "
            f"{amp.max():.4f}"
        )
    # 10 against 40 substeps over a 5 s ramp ending 2 % right of the line, and
    # the soft minimum against the substeps' hard minimum.
    p = P.replace(demand_sigma=0.0)
    dp0 = 24.0e3
    m0 = float(consumer_flow(dp0, 0.9, p, np))
    c = RHO * P.A_c**2 * dp0 / m0**2
    phi0 = brentq(lambda f: psi(f) / f**2 - c, 0.5, 0.8316)
    N = m0 / (flow_scale() * phi0)
    phi_end = 1.02 * TWO_W
    dp_end = head_scale(N) * psi(phi_end)
    u_end = flow_scale(N) * phi_end / float(consumer_flow(dp_end, 1.0, P, np))
    levels = np.array([0.9] + [u_end] * 5)
    s0 = state_of(
        m0, dp0, N, 0.0, 0.9, levels=levels, block_clock=190, dtype=jnp.float32
    )
    action = jnp.asarray(raw_from_commands(N, 0.0, P), jnp.float32)
    a = _rollout(s0, action, p, 80, "rk4_10")
    b = _rollout(s0, action, p, 80, "rk4_40")
    phi_a, phi_b = (
        np.asarray(s.m_c) / (flow_scale() * np.asarray(s.N)) for s in (a, b)
    )
    say(
        f"  rk4_10 against rk4_40, 8 s run through a 5 s ramp to 2 % from the line "
        f"(speed {N * 100:.1f} %, opening 0.9 to {u_end:.3f}, float32): dp "
        f"{np.abs(np.asarray(a.dp) - np.asarray(b.dp)).max() / np.asarray(b.dp).max():.1e} "
        f"relative, step-end Phi minimum {abs(phi_a.min() / phi_b.min() - 1):.1e} relative "
        f"({phi_b.min():.4f})"
    )
    # Substep Phi from ``substep_phi``, against the env's own step.
    with enable_x64():
        s64 = state_of(m0, dp0, N, 0.0, 0.9, levels=levels, block_clock=190)
        act64 = jnp.asarray(raw_from_commands(N, 0.0, P), jnp.float64)
        coarse = _rollout(s64, act64, p, 80, "rk4_10")
        sub = np.asarray(substep_phi(s64, act64, p, 80))
    gap = sub.min(axis=1) - np.asarray(coarse.phi_min)
    say(
        f"  soft minimum against the substeps' hard minimum over that run: "
        f"hard - soft in [{gap.min():.4e}, {gap.max():.4e}] against the bound "
        f"T ln 10 = {P.soft_min_temperature * math.log(10):.4e}; the fine replay's "
        f"step-end states match to {np.abs(sub[:, -1] - np.asarray(coarse.m_c) / (flow_scale() * np.asarray(coarse.N))).max():.1e}"
    )
    # The reset's Newton count.
    openings = np.linspace(0.2, 1.0, 81)
    root = np.array([equilibrium(P.N_reset, P.x_reset, uu)[0] for uu in openings])
    rows = []
    for iters in range(1, 8):
        with enable_x64():
            e64 = np.abs(
                np.asarray(
                    equilibrium_phi(
                        P.N_reset, P.x_reset, jnp.asarray(openings), P, iters
                    )
                )
                / root
                - 1
            ).max()
        e32 = np.abs(
            np.asarray(
                equilibrium_phi(
                    P.N_reset, P.x_reset, jnp.asarray(openings, jnp.float32), P, iters
                ),
                float,
            )
            / root
            - 1
        ).max()
        rows.append(f"{iters}: {e64:.1e} / {e32:.1e}")
    say(
        "  reset Newton from Phi 0.70, largest relative error over openings 0.2 to 1.0 (float64 / float32) by iterations: "
        + "; ".join(rows)
    )


def section_invariance():
    title("invariance: check 5 replay, check 7 ratios, fan laws", "float32 / float64")
    key = jax.random.PRNGKey(0)
    _, state = ENV.reset_env(key, P)
    fields = [
        f
        for f in state.__dataclass_fields__
        if f != "time"
        and np.ndim(getattr(state, f)) == 0
        and np.issubdtype(np.asarray(getattr(state, f)).dtype, np.floating)
    ]
    action = jnp.zeros(2)
    worst, where = 0.0, None

    def one(x, i):
        s = state.replace(
            **{f: jnp.where(i == k, x, getattr(state, f)) for k, f in enumerate(fields)}
        )
        _, s2, _, _, _ = ENV.step_env(key, s, action, P)
        return jnp.stack([getattr(s2, g) for g in fields])

    nxt_fn = jax.jit(jax.vmap(one, in_axes=(0, None)))
    jac = jax.jit(jax.vmap(jax.jacfwd(one), in_axes=(0, None)))
    nonfinite = total = 0
    for i_field, f in enumerate(fields):
        base = float(getattr(state, f))
        span = abs(base) if abs(base) > 1e-6 else 1.0
        xs = jnp.linspace(base - 0.75 * span, base + 0.75 * span, 121)
        eps = 1e-3 * span
        nxt = np.asarray(nxt_fn(xs, i_field))
        nonfinite += int((~np.isfinite(nxt)).any(axis=1).sum())
        total += nxt.shape[0]
        Jm, Jp = jac(xs - eps, i_field), jac(xs + eps, i_field)
        num, den = jnp.abs(Jp - Jm), jnp.abs(Jp) + jnp.abs(Jm)
        peak = jnp.max(den, axis=0, keepdims=True)
        both = jnp.minimum(jnp.abs(Jp), jnp.abs(Jm)) > 0.05 * jnp.maximum(peak, 1e-30)
        jump = np.nan_to_num(
            np.asarray(jnp.where(both & (den > 0), num / jnp.maximum(den, 1e-30), 0.0))
        )
        if jump.size and jump.max() > worst:
            worst = float(jump.max())
            k = int(np.argmax(jump.max(axis=0)))
            i = int(np.argmax(jump.max(axis=1)))
            where = f"{f} -> {fields[k]} at {f} = {float(xs[i]):.4g}"
    say(
        f"  check 5 (PRNGKey(0) reset, zero action, fields {', '.join(fields)}): "
        f"{nonfinite} of {total} next states non-finite; largest one-sided Jacobian "
        f"jump {worst:.3f} ({where}) against the limit 0.5"
    )
    step = jax.jit(ENV.step_env)
    traj = [np.array([float(getattr(state, f)) for f in fields])]
    k7 = key
    for _ in range(600):
        k7, sub = jax.random.split(k7)
        _, state, _, _, _ = step(sub, state, action, P)
        traj.append(np.array([float(getattr(state, f)) for f in fields]))
    tr = np.stack(traj)
    inc = np.abs(np.diff(tr, axis=0))
    fifth = len(inc) // 5
    early, late = inc[:fifth].mean(axis=0), inc[-fifth:].mean(axis=0)
    noise = 32.0 * np.finfo(np.float32).eps * np.maximum(np.abs(tr).mean(axis=0), 1.0)
    measurable = early > np.maximum(noise, 1e-12)
    ratio = np.where(measurable, late / np.maximum(early, 1e-12), 0.0)
    say(
        "  check 7 (600 zero-action steps, late / early mean increment): "
        + ", ".join(f"{f} {r:.2f}" for f, r in zip(fields, ratio))
        + f"; worst {ratio.max():.2f} against the limit 8"
    )
    p = P.replace(dp_out=0.0)
    rows = []
    with enable_x64():
        ref = None
        for N in (1.0, 0.70, 0.875, 1.05):
            phi = float(equilibrium_phi(N, 0.0, 0.6, p, 30, xp=np))
            m = flow_scale(N, p) * phi
            dp = head_scale(N, p) * psi(phi, p)
            if ref is None:
                ref = (m, dp)
                continue
            rows.append(
                f"{N * 100:.1f} %: flow / N {abs(m / N / ref[0] - 1):.1e}, dp / N^2 {abs(dp / N**2 / ref[1] - 1):.1e}"
            )
    say(
        "  fan laws (no static head, recycle shut, opening 0.6), deviation from rated: "
        + "; ".join(rows)
    )


def section_deviations():
    title("deviations: D1 to D5 figures", "float64")
    phi = 0.98 * TWO_W
    dp = head_scale(0.70) * psi(phi)
    m = flow_scale(0.70) * phi
    uu = m / float(consumer_flow(dp, 1.0, P, np))
    with enable_x64():
        states = _rollout(
            state_of(m, dp, 0.70, 0.0, uu),
            jnp.asarray(raw_from_commands(0.70, 0.0, P), jnp.float64),
            P.replace(demand_sigma=0.0),
            100,
        )
    say(
        f"  D1: at 70 %, recycle shut, 2 % left of the peak (opening {uu:.4f}), dp after "
        f"10 s {float(states.dp[-1]):.1f} Pa against the cubic's {dp:.1f} Pa"
    )
    mach = 0.777 * P.U_r * P.N_max / A01
    drop = 1 - (1 + 0.5 * (P.gamma - 1) * mach**2) ** (-1 / (P.gamma - 1))
    say(
        f"  D2: duct Mach {mach:.3f} at Phi 0.777, 105 %; isentropic density {drop * 100:.1f} % below suction"
    )
    pr = (P.P01 + head_scale() * (P.psi_c0 + 2 * P.H)) / P.P01
    T2 = P.T01 * pr ** ((P.gamma - 1) / P.gamma)
    say(
        f"  D3: rated peak pressure ratio {pr:.4f}; isentropic discharge {T2:.2f} K "
        f"(+{T2 - P.T01:.2f} K); speed of sound and f_H {(math.sqrt(T2 / P.T01) - 1) * 100:.2f} % "
        f"higher than at suction"
    )
    phi5 = 1.05 * TWO_W
    dp5 = head_scale() * psi(phi5)
    m5 = flow_scale() * phi5
    u5 = m5 / float(consumer_flow(dp5, 1.0, P, np))
    with enable_x64():
        s0 = state_of(0.99 * TWO_W * flow_scale(), dp5, 1.0, 0.0, u5)
        states = _rollout(
            s0,
            jnp.asarray(raw_from_commands(1.0, 0.0, P), jnp.float64),
            P.replace(demand_sigma=0.0),
            10,
        )
    phi_t = np.asarray(states.m_c) / flow_scale()
    say(
        f"  D4: from 5 % right of the line with the flow knocked to 1 % left of it "
        f"(rated, recycle shut): step-end Phi {phi_t[0]:.4f} after one step, first "
        f"phi_min {float(states.phi_min[0]):.4f} (trips: {bool(states.phi_min[0] < TWO_W)})"
    )
    leak = P.k_d * math.sqrt(P.check_width * math.log(2.0))
    say(f"  D5: leak at zero drop and full opening k_d sqrt(s ln 2) = {leak:.4f} kg/s")


def section_disturbance():
    title(
        "disturbance: OU statistics, the margin a seen innovation costs",
        "float32 / float64",
    )
    a = math.exp(-P.demand_theta * P.delta_t)
    sd = P.demand_sigma * math.sqrt(1 - a * a)
    _, rec = run(policy("constant", raw=(0.0, 0.0)), np.arange(64), record=True)
    dev = rec["dev"].astype(float)
    means = dev.mean(axis=1)
    lag1 = (dev[:, 1:] * dev[:, :-1]).sum() / (dev[:, :-1] ** 2).sum()
    say(
        f"  deviation under zero action, 64 seeds x 1200 steps: mean {means.mean():+.5f} "
        f"(SE {means.std(ddof=1) / 8:.5f}), RMS {np.sqrt((dev**2).mean()):.4f} (law "
        f"{P.demand_sigma}), lag-1 {lag1:.4f} (law {a:.4f}); innovation sd {sd:.5f}"
    )
    quiet = P.replace(demand_sigma=0.0)
    for margin in (0.02, 0.03, 0.05):
        phi = TWO_W * (1 + margin)
        pts = []
        for p in (20e3, 24e3, 28e3):
            N = math.sqrt(p / (head_scale() * psi(phi)))
            if not P.N_min <= N <= P.N_max:
                continue
            m = flow_scale(N) * phi
            for uu in np.arange(0.20, 0.951, 0.05):
                x = (m - float(consumer_flow(p, uu, P, np))) / float(
                    recycle_flow(p, 1.0, P, np)
                )
                if 0.0 <= x <= 1.0:
                    pts.append((m, p, N, x, uu))
        pts = np.array(pts)
        out = {}
        with enable_x64():
            for zsd in (0.0, 1.0, 4.0, 5.0):
                for mode in ("hold", "react"):
                    if zsd == 0.0 and mode == "react":
                        continue
                    xc = pts[:, 3] if mode == "hold" else np.ones(len(pts))
                    phis = jax.vmap(
                        lambda m, p, N, x, uu, xc: substep_phi(
                            state_of(m, p, N, x, uu, dev=-zsd * sd),
                            raw_from_commands(N, xc, P, jnp),
                            quiet,
                            10,
                        )
                    )(*(jnp.asarray(pts[:, i]) for i in range(5)), jnp.asarray(xc))
                    out[(zsd, mode)] = np.asarray(phis).min(axis=2) / TWO_W - 1
        plan = out[(0.0, "hold")]
        cells = []
        for zsd in (1.0, 4.0, 5.0):
            for mode in ("hold", "react"):
                lost1 = (plan[:, 0] - out[(zsd, mode)][:, 0]).max() * 100
                lost10 = (plan.min(axis=1) - out[(zsd, mode)].min(axis=1)).max() * 100
                cells.append(f"{zsd:.0f} sd {mode} {lost1:.3f} / {lost10:.3f}")
        say(
            f"  margin {margin:.0%} ({len(pts)} points), margin points lost on the substeps' "
            "hard minimum, first step / 10 steps: " + "; ".join(cells)
        )


def _battery_trace(gains, dtype, perturb=0.0, n_steps=BATTERY_STEPS):
    """The guard's battery runs, recorded: per scenario and step the margin
    phi_min / 2W - 1 and the valve command, with the start pressures scaled
    by (1 + perturb) and ``n_steps`` steps."""
    quiet = P.replace(demand_sigma=0.0)
    g = {k: jnp.asarray(float(gains[k]), dtype) for k in DEFAULT_GAINS}

    def one(z0, setpoints, levels):
        state = _state_at(z0, setpoints, levels, BATTERY_START_CLOCK, quiet)
        ps = pair_init(get_obs(state, quiet), xp=jnp)

        def body(carry, _):
            s, ps = carry
            u, ps = pair_step(g, ps, get_obs(s, quiet), quiet, xp=jnp)
            s = compute_next_state(u, s, quiet, jax.random.PRNGKey(0))[0]
            return (s, ps), (s.phi_min / TWO_W - 1.0, 0.5 * (u[1] + 1.0))

        return jax.lax.scan(body, (state, ps), None, length=n_steps)[1]

    z0 = np.stack([settled_point(p0, u0, gains, P) for p0, _, u0, _ in BATTERY])
    z0[:, 1] *= 1.0 + perturb
    sp = np.array([[p0, p0, p1, p1] for p0, p1, _, _ in BATTERY])
    lv = np.array([[u0] * 3 + [u1] * 3 for _, _, u0, u1 in BATTERY])
    margin, xcmd = jax.jit(jax.vmap(one))(
        *(jnp.asarray(v, dtype) for v in (z0, sp, lv))
    )
    return np.asarray(margin, float), np.asarray(xcmd, float)


GUARD_SETS = {
    "DEFAULT_GAINS (0.08, 0.04, 2, 3)": {},
    "anti-surge (3, 3)": {"kp_surge": 3.0, "ki_surge": 3.0},
    "anti-surge (3, 0.3)": {"kp_surge": 3.0, "ki_surge": 0.3},
    "slow pair (0.01, 0.005, 1.5, 3)": {
        "kp_speed": 0.01,
        "ki_speed": 0.005,
        "kp_surge": 1.5,
        "ki_surge": 3.0,
    },
    "anti-surge (0.5, 0.05)": {"kp_surge": 0.5, "ki_surge": 0.05},
    "anti-surge (1, 1)": {"kp_surge": 1.0, "ki_surge": 1.0},
    "anti-surge (12, 1.2)": {"kp_surge": 12.0, "ki_surge": 1.2},
    "speed (0.3, 0.3)": {"kp_speed": 0.3, "ki_speed": 0.3},
}


def section_guard():
    title(
        "guard: decay, battery, the frozen margin, rounding, cold cost",
        "float64 Jacobians / float32 battery",
    )
    say("  settled points [m_c, dp, N, x] at the guard points and battery starts:")
    for p, uu in GUARD_POINTS + tuple((p0, u0) for p0, _, u0, _ in BATTERY[:3]):
        z = settled_point(p, uu, DEFAULT_GAINS, P)
        say(
            f"    {p / KPA:.0f} kPa, opening {uu:.2f}: m_c {z[0]:.3f}, N {z[2] * 100:.2f} %, x {z[4]:.4f}, margin {z[0] / (float(surge_flow_per_speed(P)) * z[2]) - 1:.4f}"
        )
    for name, change in GUARD_SETS.items():
        g = {**DEFAULT_GAINS, **change}
        d = closed_loop_decay(g)
        m, t, tr = battery_report(g)
        failed = [
            check
            for check, ok in (
                ("decay", (d >= GUARD_MIN_DECAY).all()),
                ("trip", not t.any()),
                ("margin", (m >= GUARD_MIN_MARGIN).all()),
                ("travel", (tr <= GUARD_MAX_TRAVEL).all()),
            )
            if not ok
        ]
        verdict = "refused on " + ", ".join(failed) if failed else "passes"
        say(
            f"  {name}: decay "
            + " ".join(f"{v:.3f}" for v in d)
            + " /s; battery "
            + " ".join(f"{v:.4f}" for v in m)
            + f", min {m.min():.4f}{', trips' if t.any() else ''}; valve-command "
            f"travel over the last {GUARD_TRAVEL_STEPS} steps, max {tr.max():.4f}: "
            f"{verdict}"
        )
    m = battery_report(DEFAULT_GAINS)[0]
    frozen = max(math.floor((m.min() - 0.015) * 1000) / 1000, 0.075)
    say(
        f"  GUARD_MIN_MARGIN: battery minimum of DEFAULT_GAINS {m.min():.4f} less 0.015, "
        f"rounded down to 0.001 and never below 0.075: {frozen:.3f} (experts.py "
        f"{GUARD_MIN_MARGIN}; {'matches' if abs(frozen - GUARD_MIN_MARGIN) < 1e-12 else 'DIFFERS'})"
    )
    m32 = _battery_trace(DEFAULT_GAINS, jnp.float32)[0]
    with enable_x64():
        m64 = _battery_trace(DEFAULT_GAINS, jnp.float64)[0]
    spread = [
        _battery_trace(DEFAULT_GAINS, jnp.float32, eps)[0][:, BATTERY_LEAD:].min(axis=1)
        for eps in (1e-7, -1e-7, 1e-6, -1e-6, 1e-5, -1e-5)
    ]
    spread = np.array(spread)
    say(
        "  battery minimum per scenario, float32 / float64: "
        + " ".join(
            f"{a:.4f}/{b:.4f}"
            for a, b in zip(
                m32[:, BATTERY_LEAD:].min(axis=1), m64[:, BATTERY_LEAD:].min(axis=1)
            )
        )
    )
    say(
        f"  under start pressures moved by 1e-7 to 1e-5 relative (float32), the battery "
        f"minimum ranges {spread.min(axis=1).min():.4f} to {spread.min(axis=1).max():.4f}; "
        f"per scenario the widest spread is {np.ptp(spread, axis=0).max():.4f}"
    )
    last = slice(BATTERY_STEPS - 100, BATTERY_STEPS)
    for name, change in (
        ("DEFAULT_GAINS", {}),
        ("anti-surge (3, 3)", GUARD_SETS["anti-surge (3, 3)"]),
        ("anti-surge (3, 0.3)", GUARD_SETS["anti-surge (3, 0.3)"]),
    ):
        g = {**DEFAULT_GAINS, **change}
        a32 = _battery_trace(g, jnp.float32)[1]
        with enable_x64():
            a64 = _battery_trace(g, jnp.float64)[1]
        say(
            f"  {name}: valve-command travel over the last 10 s of each run (30 to 40 s "
            "after the move), float32 / float64: "
            + " ".join(
                f"{np.abs(np.diff(a[last])).sum():.2f}/{np.abs(np.diff(b[last])).sum():.2f}"
                for a, b in zip(a32, a64)
            )
        )
    # How long the ringing lasts: 120 s of each battery run, the time after
    # the move from which the valve command travels under 0.01 per 10 s.
    for dtype, label in ((jnp.float32, "float32"), (jnp.float64, "float64")):
        with enable_x64() if dtype == jnp.float64 else contextlib.nullcontext():
            xs = _battery_trace(DEFAULT_GAINS, dtype, n_steps=BATTERY_LEAD + 1200)[1]
        settle = []
        for x in xs:
            travel = np.abs(np.diff(x[BATTERY_LEAD:]))
            window = np.convolve(travel, np.ones(100), "valid")
            quiet = np.flatnonzero(window >= 0.01)
            settle.append(0.0 if quiet.size == 0 else (quiet.max() + 101) * P.delta_t)
        say(
            f"  DEFAULT_GAINS, {label}: seconds after the move until the valve command "
            "travels under 0.01 per 10 s, per run (120 s runs): "
            + " ".join(f"{t:.0f}" for t in settle)
        )
    code = (
        "import time, warnings; warnings.filterwarnings('ignore'); t0 = time.time(); "
        "from target_gym.compressor_surge.experts import DEFAULT_GAINS, check_pair_gains; "
        "t1 = time.time(); check_pair_gains(DEFAULT_GAINS); t2 = time.time(); "
        "check_pair_gains({**DEFAULT_GAINS, 'kp_speed': 0.081}); t3 = time.time(); "
        "print(f'{t1 - t0:.2f} {t2 - t1:.2f} {t3 - t2:.2f}')"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu"}
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    try:
        imp, cold, warm = (float(v) for v in out.stdout.split())
        say(
            f"  cold cost in a fresh process (OMP_NUM_THREADS=1): import {imp:.2f} s, "
            f"check_pair_gains {cold:.2f} s (budget 15 s), a second gain set {warm:.2f} s; "
            f"load average {os.getloadavg()[0]:.1f}"
        )
    except ValueError:
        say(f"  cold cost: the subprocess failed: {out.stderr.strip()[-400:]}")


FAST = {
    "params": section_params,
    "steady": section_steady,
    "linear": section_linear,
    "anchor": section_anchor,
    "reach": section_reach,
    "integrator": section_integrator,
    "invariance": section_invariance,
    "deviations": section_deviations,
    "disturbance": section_disturbance,
    "guard": section_guard,
}


# ---------------------------------------------------------------------------
# Heavy tables
# ---------------------------------------------------------------------------


def _row(name, summ, params=P, n_steps=None):
    n_steps = n_steps or params.max_steps_in_episode
    c = costs(summ, params)
    trips = summ["trips"]
    return dict(
        name=name,
        trips=float(trips.mean()),
        share=float((trips > 0).mean()),
        cost=float(c.mean()),
        cost_notrip=float((c - trips * R.trip_cost(params)).mean()),
        min_margin=float(summ["min_margin"].min()),
        power_kW=float((summ["power"] / np.maximum(summ["ok"], 1)).mean() / 1e3),
        energy_share=float(summ["run"].sum() / max(c.sum(), 1e-30)),
        v1_zero=float(((summ["v1"] + params.restart_steps * trips) / n_steps).mean()),
        v1=float((summ["v1"] / n_steps).mean()),
        travel=float(summ["travel"].mean()),
        n=len(trips),
        k=int((trips > 0).sum()),
    )


def pid_validation(n=4096):
    gains = load_gains()
    seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + n)
    title(
        f"pid-validation: the pair on {n} episodes, seeds {seeds[0]} to {seeds[-1]}",
        "float32",
    )
    say("  gains: " + ", ".join(f"{k} {gains[k]:g}" for k in DEFAULT_GAINS))
    summ, _ = run(policy("pair", gains=gains), seeds)
    r = _row("pair", summ)
    mm = summ["min_margin"]
    q = np.quantile(mm, [0.0, 0.001, 0.01, 0.05, 0.5])
    say(
        f"  trips {int(summ['trips'].sum())}, in {r['k']} of {n} episodes (one-sided "
        f"95 % bound {one_sided_bound(r['k'], n):.1e} per episode); smallest margin "
        f"per episode: min {q[0]:.4f}, 0.1 % {q[1]:.4f}, 1 % {q[2]:.4f}, 5 % "
        f"{q[3]:.4f}, median {q[4]:.4f}"
    )
    say(
        f"  episodes whose margin went under the override line 0.075: "
        f"{int((summ['below'] > 0).sum())}; cost per episode {r['cost']:.4g} at e_floor "
        f"{P.e_floor} kPa; mean recycle power {r['power_kW']:.1f} kW; mean |e| "
        f"{(summ['abs_e'] / np.maximum(summ['ok'], 1)).mean():.3f} kPa; valve travel "
        f"{r['travel']:.2f} per episode"
    )
    first = summ["min_margin"][:1024]
    say(
        f"  the first 1024, the seeds of test_the_pair_holds_the_validation_seeds: trips "
        f"{int(summ['trips'][:1024].sum())}, "
        f"smallest margin {first.min():.4f}"
    )
    fails = r["k"] > 0 or q[0] < 0.075
    say(
        "  decision rule (PHYSICS.md section 7): "
        + (
            "FAILS: raise GUARD_MIN_MARGIN by 0.01 and re-tune once; if it fails again, "
            "ship DEFAULT_GAINS and say so"
            if fails
            else "passes: no trip, margin above the override line"
        )
    )


def _hardness_policies():
    return {
        "recycle shut, speed PI": policy("fixed", 0.0),
        "fixed recycle 0.10": policy("fixed", 0.10),
        "fixed recycle 0.20": policy("fixed", 0.20),
        "fixed recycle 0.25": policy("fixed", 0.25),
        "fixed recycle 0.30": policy("fixed", 0.30),
        "fixed recycle 0.50": policy("fixed", 0.50),
        "chaser, bias 0.2": policy("chaser", 0.2),
        "chaser, bias 0.3": policy("chaser", 0.3),
        "chaser, bias 0.35": policy("chaser", 0.35),
        "chaser, bias 0.4": policy("chaser", 0.4),
        "chaser, bias 0.45": policy("chaser", 0.45),
        "guarded, 5 % line (G5)": policy("guarded", line=0.05),
        "guarded, 10 % line (G10)": policy("guarded", line=0.10),
        "the pair": policy("pair"),
        "zero action": policy("constant"),
        "random actions": policy("random"),
    }


def hardness(n=256):
    seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + n)
    title(
        f"hardness: {n} episodes per policy, seeds {seeds[0]} to {seeds[-1]}", "float32"
    )
    rows = {
        name: _row(name, run(spec, seeds)[0])
        for name, spec in _hardness_policies().items()
    }
    g10 = rows["guarded, 10 % line (G10)"]["cost"]
    say(
        f"  {'policy':28s} {'trips/ep':>9s} {'eps trip':>9s} {'cost/ep':>10s} {'/ G10':>8s} "
        f"{'min margin':>11s} {'recycle kW':>11s} {'energy':>7s} {'valve travel':>13s}"
    )
    for r in rows.values():
        say(
            f"  {r['name']:28s} {r['trips']:9.3f} {r['share'] * 100:8.1f}% {r['cost']:10.3e} "
            f"{r['cost'] / g10:8.2f} {r['min_margin']:11.4f} {r['power_kW']:11.1f} "
            f"{r['energy_share'] * 100:6.2f}% {r['travel']:13.2f}"
        )
    best = min((r for r in rows.values() if r["k"] == 0), key=lambda r: r["cost"])
    say(
        f"  best trip-free policy: {best['name']}, {best['cost']:.4g} per episode; recycle "
        f"energy is {best['energy_share'] * 100:.2f} % of its cost; a trip costs "
        f"{R.trip_cost(P) / best['cost']:.3g} of its episodes"
    )


SENSITIVITY_ROWS = {
    "as shipped": {},
    "drive 1.66 %/s": {"drive_rate": 0.0166},
    "drive 10.4 %/s": {"drive_rate": 0.104},
    "valve close 33 %/s (3 s full close)": {"valve_close_rate": 1.0 / 3.0},
    "valve open 100 %/s": {"valve_open_rate": 1.0},
    "deviation sd 0.01": {"demand_sigma": 0.01},
    "deviation sd 0.04": {"demand_sigma": 0.04},
    "demand ramp 2 s": {"demand_ramp_s": 2.0},
    "demand ramp 10 s": {"demand_ramp_s": 10.0},
    "constant setpoint 24 kPa": {"p_ref_range": (24.0e3, 24.0e3)},
    "setpoint range 22 to 26 kPa": {"p_ref_range": (22.0e3, 26.0e3)},
    "V_p 0.5x": {"V_p": 7.5},
    "V_p 2x": {"V_p": 30.0},
    "L_c 0.5x": {"L_c": 2.0},
    "L_c 2x": {"L_c": 8.0},
}


def sensitivity(n=256):
    seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + n)
    title(
        f"sensitivity: {n} episodes per cell, seeds {seeds[0]} to {seeds[-1]}",
        "float32",
    )
    specs = {
        "shut": policy("fixed", 0.0),
        "chaser": policy("chaser", 0.3),
        "fixed": policy("fixed", 0.30),
        "pair": policy("pair"),
        "G10": policy("guarded", line=0.10),
    }
    for row, change in SENSITIVITY_ROWS.items():
        p = P.replace(**change)
        r = {k: _row(k, run(spec, seeds, params=p)[0], p) for k, spec in specs.items()}
        g = r["G10"]["cost"]
        B = float(greitzer_B(1.0, p, np))
        fH = float(helmholtz_omega(p, np)) / (2 * math.pi)
        say(
            f"  {row:36s} (f_H {fH:.2f} Hz, B {B:.2f}): shut {r['shut']['trips']:.2f} trips/ep | "
            f"chaser 0.3 trips in {r['chaser']['share'] * 100:.1f} % | fixed 0.30 "
            f"{r['fixed']['cost'] / g:.2f} x G10 (trips in {r['fixed']['share'] * 100:.1f} %) | pair "
            f"{r['pair']['cost'] / g:.2f} x G10, {r['pair']['trips'] * n:.0f} trips, min margin "
            f"{r['pair']['min_margin']:.4f} | G10 {g:.3e}, {r['G10']['trips'] * n:.0f} trips, "
            f"min margin {r['G10']['min_margin']:.4f}"
        )
    _restart_break_evens(seeds)


def _v1_policies():
    return {
        "recycle shut, speed PI": policy("fixed", 0.0),
        "fixed recycle 0.10": policy("fixed", 0.10),
        "fixed recycle 0.20": policy("fixed", 0.20),
        "fixed recycle 0.25": policy("fixed", 0.25),
        "chaser, bias 0.2": policy("chaser", 0.2),
        "chaser, bias 0.3": policy("chaser", 0.3),
        "full travel low": policy("constant", raw=(-1.0, -1.0)),
        "fixed recycle 0.30": policy("fixed", 0.30),
        "fixed recycle 0.50": policy("fixed", 0.50),
        "zero action": policy("constant"),
        "the pair": policy("pair"),
        "guarded, 10 % line (G10)": policy("guarded", line=0.10),
    }


def _restart_break_evens(seeds):
    """The smallest ``restart_steps`` at which every safe policy of the v1 set
    ranks above every tripping one, under v2 and under v1."""
    n = P.max_steps_in_episode
    rows = [_row(k, run(spec, seeds)[0]) for k, spec in _v1_policies().items()]
    safe = [r for r in rows if r["k"] == 0]
    trip = [r for r in rows if r["k"] > 0]
    fc = P.failure_cost
    need_v2 = max(
        (
            (s["cost"] - t["cost_notrip"]) / (t["trips"] * fc)
            for t in trip
            for s in safe
        ),
        default=0.0,
    )
    need_v1 = max(
        ((t["v1_zero"] - s["v1"]) * n / t["trips"] for t in trip for s in safe),
        default=0.0,
    )
    say(
        f"  restart break-evens over the v1 set on {len(seeds)} seeds: v2 orders every "
        f"safe policy above every tripping one from {max(need_v2, 0):.0f} steps, v1 from "
        f"{max(need_v1, 0):.0f} steps; shipped {P.restart_steps}"
    )


def v1_table(n=128, n_rare=5120):
    seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + n)
    steps = P.max_steps_in_episode
    title(
        f"v1: the exploit on {n} seeds ({seeds[0]} up), rare trippers on {n_rare}",
        "float32",
    )
    rows = [_row(k, run(spec, seeds)[0]) for k, spec in _v1_policies().items()]
    say(
        f"  {'policy':28s} {'v1, 0 on trip':>14s} {'v1 shipped':>11s} {'trips/ep':>9s} {'v2 cost/step':>13s}"
    )
    for r in rows:
        say(
            f"  {r['name']:28s} {r['v1_zero']:14.4f} {r['v1']:11.3f} {r['trips']:9.2f} {r['cost'] / steps:13.4e}"
        )
    v2 = np.array([-r["cost"] for r in rows])
    for key, label in (("v1_zero", "0 on a trip"), ("v1", "as shipped")):
        rho = spearmanr([r[key] for r in rows], v2).correlation
        say(f"  Spearman(v1 with {label}, v2) over {len(rows)} policies: {rho:+.3f}")
    safe = [r for r in rows if r["k"] == 0]
    trip = [r for r in rows if r["k"] > 0]
    wrong = sum(1 for t in trip for s in safe if t["v1_zero"] > s["v1"])
    need = max(
        (
            (t["v1_zero"] - s["v1"]) * steps / t["trips"]
            for t in trip
            for s in safe
            if t["v1_zero"] > s["v1"]
        ),
        default=0.0,
    )
    say(
        f"  with 0 on a trip, {wrong} of {len(trip) * len(safe)} (tripping, safe) pairs rank "
        f"the tripping policy higher; a tripped step must score below -{need:.0f} to reverse "
        f"them all"
    )
    fixed = next(r for r in rows if r["name"] == "fixed recycle 0.30")
    best = min(safe, key=lambda r: r["cost"])
    say(
        f"  v1 prices a trip at {P.restart_steps / steps:.2f} episodes of its best score; v2 at "
        f"{R.trip_cost(P) / best['cost']:.3g} episodes of the best trip-free policy ({best['name']})"
    )
    rare_seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + n_rare)
    for name, spec in (
        ("guarded, 5 % line (G5)", policy("guarded", line=0.05)),
        ("chaser, bias 0.4", policy("chaser", 0.4)),
    ):
        r = _row(name, run(spec, rare_seeds)[0])
        rate = r["trips"]
        lo, hi = poisson_interval(int(round(rate * n_rare)), n_rare)
        v1_ship = r["v1_zero"] - rate * P.restart_steps / steps
        charge = [
            (r["v1_zero"] - fixed["v1"]) * steps / q if q > 0 else float("inf")
            for q in (rate, hi, lo)
        ]
        v2_rate = (fixed["cost"] - r["cost_notrip"]) / R.trip_cost(P)
        match = (
            (r["v1_zero"] - fixed["v1"]) * steps / v2_rate
            if v2_rate > 0
            else float("nan")
        )
        say(
            f"  {name}: {rate * n_rare:.0f} trips in {n_rare} ({rate:.2e} per episode, 95 % "
            f"{lo:.1e} to {hi:.1e}); v1 per step {r['v1_zero']:.4f} with 0 on a trip, "
            f"{v1_ship:.4f} as shipped, against fixed 0.30's {fixed['v1']:.4f}; v2 "
            f"{r['cost']:.3e} per episode against {fixed['cost']:.3e}"
        )
        say(
            f"    v1 charge that ranks it below fixed 0.30: {charge[0]:.3g} ({charge[1]:.3g} "
            f"to {charge[2]:.3g}); v2 stops separating it from fixed 0.30 at {v2_rate:.2e} "
            f"trips per episode, and a v1 charge of {match:.3g} gives v1 that break-even"
        )


# ---------------------------------------------------------------------------
# The NMPC's acceptance run (--mpc-gate)
# ---------------------------------------------------------------------------

#: The params ``--mpc-gate`` builds the NMPC with: ``plan_params(spec, P)``,
#: since the spec's ``noise_fields`` is ``("demand_sigma",)``.
PLANNED = P.replace(demand_sigma=0.0)
#: One innovation of the deviation, in opening (derived): sigma sqrt(1 - a^2)
#: with a = exp(-theta dt).
INNOVATION_SD = P.demand_sigma * math.sqrt(
    1.0 - math.exp(-2.0 * P.demand_theta * P.delta_t)
)
#: The innovation the budget's part 1 applies, in innovation sd (ours).
BUDGET_INNOVATION_SD = 5.0
#: Part 2 of the budget, the model gap on the step's phi_min, as a margin:
#: test_mpc_predicts_one_step_like_the_plant's bound of 1e-4 on Phi over 2W.
MODEL_GAP_MARGIN = 1e-4 / TWO_W
#: The surge margins ``--mpc-gate`` may ship (ours).
GATE_MARGINS = (0.02, 0.03, 0.05)
#: Every how many steps part 1 re-solves at a perturbed deviation (ours).
BUDGET_SAMPLE_EVERY = 10
#: Episodes of the budget pass, seeds from HEAVY_SEED0 (ours).
BUDGET_EPISODES = 16
#: Ten innovations of -1 sd, one after each of the ten steps that end on
#: schedule clocks 600 to 609, the first ten of the battery's demand ramp.
BATTERY_INNOVATIONS = 10

_GATE_MPCS: dict = {}


def _gate_mpc(margin, horizon):
    """The NMPC at ``margin`` and ``horizon``, built once per process."""
    if (margin, horizon) not in _GATE_MPCS:
        _GATE_MPCS[(margin, horizon)] = make_compressor_surge_mpc(
            ENV, PLANNED, horizon=horizon, surge_margin=margin
        )
    return _GATE_MPCS[(margin, horizon)]


@functools.lru_cache(maxsize=None)
def _gate_steps():
    """The env's ``step_env`` and ``compute_next_state``, jitted once per
    process. The second gives a step's proposal, whose ``phi_min`` is the
    step's margin even on a step that trips."""
    return (
        jax.jit(ENV.step_env),
        jax.jit(lambda u, s, p: compute_next_state(u, s, p, jax.random.PRNGKey(0))[0]),
    )


def _counters(mpc):
    """The NMPC's running counters: solves, IPOPT iterations, failed solves
    as the base counts them (capped ones included), capped solves, steps
    with the surge slack active, steps handed to the pair and capped solves
    applied."""
    return np.array(
        [
            mpc.solve_calls,
            mpc.solve_iters,
            mpc.solve_failures,
            mpc.solve_capped,
            mpc.slack_steps,
            mpc.fallback_steps,
            mpc.capped_steps,
        ]
    )


def _snapshot(mpc):
    """Everything a what-if solve changes, so it can be undone."""
    m = mpc._mpc
    keep = (
        "_last_clock",
        "_initialized",
        "_last_u",
        "solve_calls",
        "solve_iters",
        "solve_failures",
        "solve_capped",
        "last_return_status",
        "slack_steps",
        "fallback_steps",
        "capped_steps",
        "last_planned_margin",
        "last_slack",
        "_needs_cold",
    )
    return {
        "guess": mpc._save_guess(),
        "u0": np.array(m.u0.cat, float).ravel(),
        "pair": mpc._pid._ps,
        "attrs": {k: getattr(mpc, k) for k in keep},
    }


def _restore(mpc, snap):
    m = mpc._mpc
    import casadi

    # As in CompressorSurgeMPC._fallback: do-mpc needs a DM here, or the next
    # cold step fails in set_initial_guess.
    m.opt_x_num.master = casadi.DM(snap["guess"]["opt_x"])
    for attr, key in (("lam_g_num", "lam_g"), ("lam_x_num", "lam_x")):
        if key in snap["guess"]:
            setattr(m, attr, snap["guess"][key])
    m.u0 = snap["u0"]
    mpc._pid._ps = snap["pair"]
    for k, v in snap["attrs"].items():
        setattr(mpc, k, v)


def _innovation_margin(mpc, state, proposal):
    """Part 1 of the budget at ``state``: the margin of the step the NMPC
    takes after re-solving with the deviation moved by -5 innovation sd, from
    the warm start the unperturbed solve uses. Undone afterwards."""
    snap = _snapshot(mpc)
    perturbed = state.replace(
        demand_dev=state.demand_dev - BUDGET_INNOVATION_SD * INNOVATION_SD
    )
    u = mpc.step(ENV.get_obs(perturbed, P), perturbed)
    margin = float(proposal(jnp.asarray(u), perturbed, P).phi_min) / TWO_W - 1.0
    _restore(mpc, snap)
    return margin


GATE_REC = (
    "e",
    "power",
    "margin",
    "planned",
    "tripped",
    "seconds",
    "fallback",
    "capped",
    "slack",
)


def _gate_episode(job):
    """One NMPC episode of ``--mpc-gate``, for the pool: ``(margin, horizon, seed,
    sample_every)``. The seed's key resets the env and drives every step, as
    ``run`` does for the heuristics, so the rows compare on the same draws.

    Per step it records the header-pressure error (kPa) and recycle power of
    the scored state, the step's margin phi_min / 2W - 1 (from the proposal,
    so a tripped step has one too), the NMPC's planned first-step margin, the
    trip flag, the wall-clock time of ``mpc.step``, whether the step fell back
    to the pair, whether it applied a solve an IPOPT cap stopped, and whether
    the plan used the surge slack. With
    ``sample_every``, every that-many steps it also measures part 1 of the
    budget: the realised margin less the margin after a seen innovation of
    -5 sd and the NMPC's re-solve."""
    margin, horizon, seed, sample_every = job
    mpc = _gate_mpc(margin, horizon)
    mpc.reset()
    step, proposal = _gate_steps()
    key = jax.random.PRNGKey(seed)
    obs, state = ENV.reset_env(key, P)
    n = int(P.max_steps_in_episode)
    rec = {name: np.zeros(n) for name in GATE_REC}
    losses = []
    start = _counters(mpc)
    for t in range(n):
        what_if = None
        if sample_every and t % sample_every == sample_every - 1:
            what_if = _innovation_margin(mpc, state, proposal)
        before = _counters(mpc)
        t0 = time.perf_counter()
        u = mpc.step(obs, state)
        rec["seconds"][t] = time.perf_counter() - t0
        after = _counters(mpc)
        rec["fallback"][t] = after[5] > before[5]
        rec["capped"][t] = after[6] > before[6]
        rec["slack"][t] = after[4] > before[4]
        rec["planned"][t] = mpc.last_planned_margin
        rec["margin"][t] = (
            float(proposal(jnp.asarray(u), state, P).phi_min) / TWO_W - 1.0
        )
        if what_if is not None:
            losses.append(rec["margin"][t] - what_if)
        obs, new, _, _, info = step(key, state, jnp.asarray(u), P)
        scored = info["last_state"] if not bool(info["tripped"]) else None
        rec["tripped"][t] = bool(info["tripped"])
        if scored is not None:
            rec["e"][t] = float(live_target(scored, P) - scored.dp) / KPA
            rec["power"][t] = float(recycle_power(scored, P))
        state = new
    return {
        "margin_plan": margin,
        "seed": seed,
        "rec": rec,
        "losses": np.array(losses),
        "counters": _counters(mpc) - start,
    }


def _gate_battery(job):
    """One run of the NMPC's stress battery: ``(margin, horizon, scenario,
    mode)``. From the scenario's settled point (``settled_point`` with the
    pair's gains) at clock ``BATTERY_START_CLOCK``, 410 steps of the env's
    step with ``demand_sigma`` 0, as the guard's battery runs, and no restart.
    ``mode`` "dev": the deviation starts at -4 stationary sd and decays by the
    env's own update. ``mode`` "innov": the deviation starts at 0, and after
    each of the ten steps that end on clocks 600 to 609 it moves by -1
    innovation sd, written between steps where the env's update puts it."""
    margin, horizon, scenario, mode = job
    mpc = _gate_mpc(margin, horizon)
    mpc.reset()
    _, proposal = _gate_steps()
    quiet = P.replace(demand_sigma=0.0)
    p0, p1, u0, u1 = BATTERY[scenario]
    z0 = settled_point(p0, u0, load_gains(), P)
    state = _state_at(
        jnp.asarray(z0, jnp.float32),
        jnp.asarray([p0, p0, p1, p1], jnp.float32),
        jnp.asarray([u0] * 3 + [u1] * 3, jnp.float32),
        BATTERY_START_CLOCK,
        quiet,
    )
    if mode == "dev":
        state = state.replace(demand_dev=jnp.float32(-4.0 * P.demand_sigma))
    start = _counters(mpc)
    margins, tripped = np.zeros(BATTERY_STEPS), np.zeros(BATTERY_STEPS, bool)
    for t in range(BATTERY_STEPS):
        u = mpc.step(ENV.get_obs(state, quiet), state)
        state = proposal(jnp.asarray(u), state, quiet)
        margins[t] = float(state.phi_min) / TWO_W - 1.0
        tripped[t] = float(state.phi_min) < TWO_W
        if (
            mode == "innov"
            and BATTERY_LEAD - 1 <= t < BATTERY_LEAD - 1 + BATTERY_INNOVATIONS
        ):
            state = state.replace(
                demand_dev=state.demand_dev - jnp.float32(INNOVATION_SD)
            )
    return {
        "margin_plan": margin,
        "scenario": scenario,
        "mode": mode,
        "margins": margins,
        "tripped": tripped,
        "counters": _counters(mpc) - start,
    }


def _pool_map(fn, jobs, workers, label):
    """``fn`` over ``jobs`` on ``workers`` processes, with progress lines."""
    t0 = time.time()
    out = []
    if workers > 1:
        # The workers inherit this, so IPOPT's linear algebra does not spawn
        # threads on top of the processes.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn, job) for job in jobs]
            for fut in cf.as_completed(futures):
                out.append(fut.result())
                if len(out) % 8 == 0 or len(out) == len(jobs):
                    say(
                        f"    {label}: {len(out)}/{len(jobs)}, {time.time() - t0:.0f} s"
                    )
    else:
        for job in jobs:
            out.append(fn(job))
            if len(out) % 8 == 0 or len(out) == len(jobs):
                say(f"    {label}: {len(out)}/{len(jobs)}, {time.time() - t0:.0f} s")
    return out


def _nmpc_summary(results, params=P):
    """The heuristics' ``run`` summary fields for NMPC episodes, from the
    recorded arrays, so ``costs`` prices both the same way."""
    ok = np.stack([1.0 - r["rec"]["tripped"] for r in results])
    e = np.stack([r["rec"]["e"] for r in results])
    power = np.stack([r["rec"]["power"] for r in results])
    running = np.asarray(
        R.running_cost(power, params.c_hold, params.running_weight, np)
    )
    margin = np.stack([r["rec"]["margin"] for r in results])
    return {
        "sq_e": (ok * e**2).sum(1),
        "run": (ok * running).sum(1),
        "trips": np.stack([r["rec"]["tripped"] for r in results]).sum(1),
        "min_margin": np.where(ok > 0, margin, np.inf).min(1),
        "power": (ok * power).sum(1),
        "ok": ok.sum(1),
    }


def mpc_gate(n=128, workers=1, horizon=None, margins=None, out_dir=None):
    """The NMPC's acceptance run (PHYSICS.md section 7): the surge-margin
    budget, then each candidate margin at or above its floor on ``n``
    episodes and the stress battery, against the pair and G10 on the same
    seeds, with solve times. Prints the verdicts of the acceptance rules and
    saves every per-step array."""
    horizon = MPC_HORIZON if horizon is None else horizon
    out_dir = os.path.abspath(
        out_dir or os.path.join(tempfile.gettempdir(), "compressor_surge_mpc_gate")
    )
    os.makedirs(out_dir, exist_ok=True)
    budget_seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + BUDGET_EPISODES)
    seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + n)
    title(
        f"mpc-gate: the NMPC at horizon {horizon} ({horizon * P.delta_t:.1f} s); "
        f"budget on seeds {budget_seeds[0]} to {budget_seeds[-1]}, candidates on "
        f"{n} episodes, seeds {seeds[0]} to {seeds[-1]}, and the 16-run stress battery",
        "float32 env, float64 NLP",
    )
    say(f"  outputs: {out_dir}")
    say(
        f"  objective constants: error scale {MPC_ERROR_SCALE_KPA} kPa, energy weight "
        f"{MPC_ENERGY_WEIGHT:.4g} above {MPC_POWER_REF / 1e3:.1f} kW, terminal weight "
        f"{MPC_TERMINAL_WEIGHT:g}, move weight {MPC_MOVE_WEIGHT:g}; the fallback pair's gains "
        + ", ".join(f"{k} {v:g}" for k, v in load_gains().items())
    )

    # Part 1 and part 3 of the budget, at the shipped margin.
    say(
        f"  budget: part 1 re-solves every {BUDGET_SAMPLE_EVERY}th step with the "
        f"deviation moved by -{BUDGET_INNOVATION_SD:g} innovation sd "
        f"({BUDGET_INNOVATION_SD * INNOVATION_SD:.4f} of opening), at the shipped margin "
        f"{MPC_SURGE_MARGIN}"
    )
    budget = _pool_map(
        _gate_episode,
        [
            (MPC_SURGE_MARGIN, horizon, int(s), BUDGET_SAMPLE_EVERY)
            for s in budget_seeds
        ],
        workers,
        "budget episodes",
    )
    losses = np.concatenate([r["losses"] for r in budget])
    worst = float(losses.max())
    floor_raw = 2.0 * (worst + MODEL_GAP_MARGIN)
    floor = next((m for m in GATE_MARGINS if m >= floor_raw), None)
    say(
        f"  part 1, seen-innovation loss over {losses.size} samples: max {worst * 100:.3f} "
        f"points, 99 % {np.percentile(losses, 99) * 100:.3f}, median "
        f"{np.median(losses) * 100:.3f}; part 2, model gap {MODEL_GAP_MARGIN * 100:.3f} "
        f"points (the bound of test_mpc_predicts_one_step_like_the_plant); floor "
        f"2 x (part 1 + part 2) = {floor_raw * 100:.3f} points"
        f" -> {'none of ' + str(GATE_MARGINS) if floor is None else floor}"
    )
    np.savez(os.path.join(out_dir, "budget.npz"), losses=losses)
    if floor is None:
        say("  FAILS: the budget's floor is above every candidate margin")
        return 1
    candidates = [m for m in (margins or GATE_MARGINS) if m >= floor]

    pair = run(policy("pair"), seeds)[0]
    g10 = run(policy("guarded", line=0.10), seeds)[0]
    pair_cost, g10_cost = costs(pair).mean(), costs(g10).mean()
    say(
        f"  on the same seeds: the pair {pair_cost:.4g} per episode ({int(pair['trips'].sum())} "
        f"trips, smallest margin {pair['min_margin'].min():.4f}); G10 {g10_cost:.4g} "
        f"({int(g10['trips'].sum())} trips, smallest margin {g10['min_margin'].min():.4f})"
    )

    verdicts = {}
    all_seconds = []
    for margin in candidates:
        results = sorted(
            _pool_map(
                _gate_episode,
                [(margin, horizon, int(s), 0) for s in seeds],
                workers,
                f"margin {margin} episodes",
            ),
            key=lambda r: r["seed"],
        )
        battery = _pool_map(
            _gate_battery,
            [
                (margin, horizon, i, mode)
                for i in range(len(BATTERY))
                for mode in ("dev", "innov")
            ],
            workers,
            f"margin {margin} battery",
        )
        summ = _nmpc_summary(results)
        cost = costs(summ)
        rec = {name: np.stack([r["rec"][name] for r in results]) for name in GATE_REC}
        np.savez(
            os.path.join(out_dir, f"episodes_margin_{margin}.npz"),
            seeds=seeds,
            **rec,
        )
        np.savez(
            os.path.join(out_dir, f"battery_margin_{margin}.npz"),
            margins=np.stack([b["margins"] for b in battery]),
            tripped=np.stack([b["tripped"] for b in battery]),
            scenario=np.array([b["scenario"] for b in battery]),
            mode=np.array([b["mode"] for b in battery]),
        )
        counters = np.sum([r["counters"] for r in results], axis=0)
        mm = summ["min_margin"]
        q = np.quantile(mm, [0.0, 0.01, 0.05, 0.5])
        shortfall = rec["planned"] - rec["margin"]
        shortfall = shortfall[np.isfinite(shortfall) & (rec["tripped"] == 0)]
        k = int((summ["trips"] > 0).sum())
        seconds = rec["seconds"].ravel()
        all_seconds.append(seconds)
        b_margin = min(b["margins"][BATTERY_LEAD:].min() for b in battery)
        b_trips = sum(int(b["tripped"].any()) for b in battery)
        b_counters = np.sum([b["counters"] for b in battery], axis=0)
        checks = {
            "no trip": k == 0,
            "smallest margin >= half the plan": q[0] >= 0.5 * margin,
            "residual shortfall < a quarter of the plan": shortfall.max()
            < 0.25 * margin,
            "battery: no trip": b_trips == 0,
            "battery: margin >= half the plan": b_margin >= 0.5 * margin,
            "battery: slack never active": b_counters[4] == 0,
            "cost below G10": cost.mean() < g10_cost,
        }
        verdicts[margin] = (all(checks.values()), cost.mean(), checks)
        say(f"  margin {margin}:")
        say(
            f"    episodes: trips {int(summ['trips'].sum())}, in {k} of {n} episodes "
            f"(one-sided 95 % bound {one_sided_bound(k, n):.1e} per episode); smallest "
            f"margin per episode: min {q[0]:.4f}, 1 % {q[1]:.4f}, 5 % {q[2]:.4f}, "
            f"median {q[3]:.4f}"
        )
        say(
            f"    prediction residual, realised less planned first-step margin: min "
            f"{-shortfall.max() * 100:.4f} points, max {-shortfall.min() * 100:.4f}; "
            f"fallbacks to the pair {int(counters[5])}; capped solves applied "
            f"{int(counters[6])}; steps with the surge slack active {int(counters[4])}; "
            f"mean IPOPT iterations {counters[1] / max(counters[0], 1):.1f}"
        )
        say(
            f"    cost {cost.mean():.4g} per episode at e_floor {P.e_floor} kPa and c_hold "
            f"{P.c_hold / 1e3:.1f} kW ({cost.mean() / pair_cost:.3f} x the pair, "
            f"{cost.mean() / g10_cost:.3f} x G10); energy {summ['run'].sum() / max(cost.sum(), 1e-30) * 100:.2f} "
            f"% of it; mean recycle power {(summ['power'] / summ['ok']).mean() / 1e3:.1f} kW; "
            f"mean |e| {np.abs(rec['e'][rec['tripped'] == 0]).mean():.4f} kPa"
        )
        say(
            f"    stress battery (16 runs): trips {b_trips}, smallest margin after the move "
            f"{b_margin:.4f}, slack-active steps {int(b_counters[4])}, fallbacks to the "
            f"pair {int(b_counters[5])}, capped solves applied {int(b_counters[6])}"
        )
        say(
            f"    solve time per step (ms): p50 {np.percentile(seconds, 50) * 1e3:.0f}, p90 "
            f"{np.percentile(seconds, 90) * 1e3:.0f}, p99 {np.percentile(seconds, 99) * 1e3:.0f}, "
            f"max {seconds.max() * 1e3:.0f} (budget p50 {MPC_SOLVE_BUDGET_S[0] * 1e3:.0f}, p99 "
            f"{MPC_SOLVE_BUDGET_S[1] * 1e3:.0f}; load average {os.getloadavg()[0]:.1f})"
        )
        say(
            "    "
            + "; ".join(
                f"{name}: {'yes' if ok else 'NO'}" for name, ok in checks.items()
            )
        )

    seconds = np.concatenate(all_seconds)
    within = (
        np.percentile(seconds, 50) <= MPC_SOLVE_BUDGET_S[0]
        and np.percentile(seconds, 99) <= MPC_SOLVE_BUDGET_S[1]
    )
    passing = [m for m, (ok, _, _) in verdicts.items() if ok]
    if not passing:
        say(
            "  no candidate passes. If the battery used the slack, tighten the later "
            "stages first; if none beats G10, re-run with --mpc-horizon 80 once, then take "
            "it to the owner as mpc_degraded"
        )
        return 1
    cheapest = min(passing, key=lambda m: verdicts[m][1])
    wider = [
        m
        for m in passing
        if m > cheapest and verdicts[m][1] <= 1.01 * verdicts[cheapest][1]
    ]
    ship = max(wider) if wider else cheapest
    say(
        f"  ship surge_margin {ship} (passing: {passing}; the cheapest is {cheapest}, and a "
        f"wider one within 1 % of its cost is preferred); solve times "
        f"{'within' if within else 'OVER'} budget"
        + ("" if within else ": horizon 40, then move blocking (PHYSICS.md section 7)")
    )
    return 0


# ---------------------------------------------------------------------------
# The NMPC's clip smoothing and cold start (--section mpc-width, on request)
# ---------------------------------------------------------------------------

#: The clip smoothings ``--section mpc-width`` compares, as fractions of one
#: substep's actuator travel: the exact clip, 0.01 and the shipped width.
MPC_WIDTHS = (0.0, 0.01, MPC_CLIP_SMOOTHING)
#: Episodes the section runs, seeds from ``HEAVY_SEED0`` (ours).
MPC_WIDTH_EPISODES = 4
#: Steps of each episode it solves at each width (ours).
MPC_WIDTH_STEPS = 12
#: Commands per actuator in the grid over one step's reach (ours).
MPC_WIDTH_GRID = 41


@contextlib.contextmanager
def _clip_smoothing(width):
    """``experts.MPC_CLIP_SMOOTHING`` set to ``width`` for an NMPC built
    inside, and restored after. ``CompressorSurgeMPC._build_mpc`` reads it
    when it writes the NLP and the model's step function."""
    shipped = experts.MPC_CLIP_SMOOTHING
    experts.MPC_CLIP_SMOOTHING = width
    try:
        yield
    finally:
        experts.MPC_CLIP_SMOOTHING = shipped


def _reach_grid(state, n=MPC_WIDTH_GRID, params=P):
    """An ``n`` x ``n`` grid of commands [N_cmd, x_cmd] spanning one step's
    travel of both actuators from ``state``, inside the command bounds, as
    the NLP's reach constraint allows. Shape (2, n * n)."""
    dt = params.delta_t
    N_ramp, x = float(state.N_ramp), float(state.x)
    N_cmd = np.linspace(
        max(params.N_min, N_ramp - params.drive_rate * dt),
        min(params.N_max, N_ramp + params.drive_rate * dt),
        n,
    )
    x_cmd = np.linspace(
        max(0.0, x - params.valve_close_rate * dt),
        min(1.0, x + params.valve_open_rate * dt),
        n,
    )
    NN, XX = np.meshgrid(N_cmd, x_cmd, indexing="ij")
    return np.stack([NN.ravel(), XX.ravel()])


def _width_episode(mpc, seed, step):
    """The first ``MPC_WIDTH_STEPS`` steps of ``seed``'s episode under
    ``mpc`` on the float32 env, as ``--mpc-gate`` runs them. Returns the
    IPOPT iterations and solve seconds per step, the solves the iteration
    cap stopped, the fallbacks, the trips and the states solved from."""
    mpc.reset()
    key = jax.random.PRNGKey(seed)
    obs, state = ENV.reset_env(key, P)
    iters, seconds, states = [], [], []
    capped = fallbacks = trips = 0
    for _ in range(MPC_WIDTH_STEPS):
        states.append(state)
        before = _counters(mpc)
        t0 = time.perf_counter()
        u = mpc.step(obs, state)
        seconds.append(time.perf_counter() - t0)
        d = _counters(mpc) - before
        iters.append(int(d[1]))
        capped += int(d[3])
        fallbacks += int(d[5])
        obs, state, _, _, info = step(key, state, jnp.asarray(u), P)
        trips += int(bool(info["tripped"]))
    return iters, seconds, capped, fallbacks, trips, states


def section_mpc_width():
    """The NMPC's clip smoothing and cold start (PHYSICS.md section 7).

    1. At each width of ``MPC_WIDTHS`` the NMPC is built once and runs the
       first ``MPC_WIDTH_STEPS`` steps of ``MPC_WIDTH_EPISODES`` episodes
       (seeds from ``HEAVY_SEED0``) on the float32 env: per episode the
       solves the iteration cap stopped, and per width the mean IPOPT
       iterations, the median solve time, the fallbacks and the trips.
    2. The one-step gap of each rounding against the exact clip: the model's
       next header pressure and its ten substep Phis, over a grid of commands
       spanning the reach of both actuators, at every state the shipped
       width solved from in part 1. Float64 (CasADi).
    3. The cold start: each episode's first step, from the guess a cold start
       builds, solved by the IPOPT instance for warm starts (warm-start
       options, adaptive barrier) and by the one for cold starts (IPOPT's
       defaults).

    The cold guess is built from the state and the preview alone
    (``CompressorSurgeMPC._initialise``), so a first step's iteration count
    does not depend on what the controller solved before.

    Builds the NMPC three times; the latest run took 51.2 s."""
    seeds = np.arange(HEAVY_SEED0, HEAVY_SEED0 + MPC_WIDTH_EPISODES)
    title(
        f"mpc-width: the NMPC's clip smoothing on the first {MPC_WIDTH_STEPS} steps "
        f"of seeds {seeds[0]} to {seeds[-1]}, its one-step gap, the cold start",
        "float32 env, float64 NLP",
    )
    step, _ = _gate_steps()
    shipped, visited = None, []
    for width in MPC_WIDTHS:
        with _clip_smoothing(width):
            t0 = time.perf_counter()
            mpc = make_compressor_surge_mpc(ENV, PLANNED)
            build = time.perf_counter() - t0
            if width not in mpc._rhs_cache:
                raise RuntimeError(
                    f"the NMPC was not built at clip smoothing {width}: "
                    "experts.MPC_CLIP_SMOOTHING is no longer what _build_mpc reads"
                )
            runs = [_width_episode(mpc, int(seed), step) for seed in seeds]
        iters = np.concatenate([r[0] for r in runs])
        seconds = np.concatenate([r[1] for r in runs])
        capped = [r[2] for r in runs]
        first = sum(int(r[0][0] >= IPOPT_MAX_ITER) for r in runs)
        later = np.array([r[0][1:] for r in runs], float).mean()
        say(
            f"  width {width:g}: steps stopped by the {IPOPT_MAX_ITER}-iteration cap, "
            f"per episode, {' '.join(str(c) for c in capped)} ({sum(capped)} of "
            f"{iters.size}, of which first steps {first}); mean IPOPT iterations "
            f"{iters.mean():.1f}, {later:.1f} after the first step; median solve {np.median(seconds) * 1e3:.0f} ms; fallbacks "
            f"{sum(r[3] for r in runs)}; trips {sum(r[4] for r in runs)}; build "
            f"{build:.0f} s"
        )
        if width == MPC_CLIP_SMOOTHING:
            shipped = mpc
            visited = [s for r in runs for s in r[5]]

    exact = shipped.rhs_function(0.0)
    for width in MPC_WIDTHS[1:]:
        rounded = shipped.rhs_function(width)
        dp_gap = phi_gap = 0.0
        for s in visited:
            u = _reach_grid(s)
            n = u.shape[1]
            x = np.array([float(getattr(s, name)) for name in MPC_STATES])
            opening = stage_openings(
                s.demand_levels, int(s.block_clock), float(s.demand_dev), P
            )
            args = (
                np.repeat(x[:, None], n, axis=1),
                u,
                np.repeat(opening[:, None], n, axis=1),
            )
            a_next, a_phi = (np.asarray(v) for v in exact.map(n)(*args))
            b_next, b_phi = (np.asarray(v) for v in rounded.map(n)(*args))
            dp_gap = max(dp_gap, float(np.abs(b_next[1] - a_next[1]).max()))
            phi_gap = max(phi_gap, float(np.abs(b_phi - a_phi).max()))
        say(
            f"  one-step gap of width {width:g} against the exact clip, over a "
            f"{MPC_WIDTH_GRID} x {MPC_WIDTH_GRID} grid of commands spanning both "
            f"actuators' reach at each of the {len(visited)} states above: next "
            f"header pressure at most {dp_gap:.3g} Pa, a substep's Phi at most "
            f"{phi_gap:.2g}"
        )

    for label, options in (
        ("warm", "warm-start options, adaptive barrier"),
        ("cold", "IPOPT's defaults"),
    ):
        results = []
        for seed in seeds:
            _, s0 = ENV.reset_env(jax.random.PRNGKey(int(seed)), P)
            shipped.reset()
            shipped._update_setpoint(s0)
            x0 = shipped._extract_x0(s0)
            shipped._initialise(x0)
            m = shipped._mpc
            m.S = shipped._solvers[label]
            m.data.init_storage()
            m.make_step(x0)
            stats = m.solver_stats
            results.append(f"{int(stats['iter_count'])} ({stats['return_status']})")
        say(
            f"  cold start, each episode's first step from the cold guess, solved by "
            f"the {label}-start instance ({options}): iterations per episode "
            + ", ".join(results)
        )


#: Sections that run only when named with --section: they build the NMPC.
ON_REQUEST = {"mpc-width": section_mpc_width}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--section",
        action="append",
        choices=list(FAST) + list(ON_REQUEST),
        help="section(s) to print; default every fast one (mpc-width runs only "
        "when named)",
    )
    ap.add_argument(
        "--hardness", action="store_true", help="the hardness table (heavy)"
    )
    ap.add_argument(
        "--sensitivity", action="store_true", help="the sensitivity table (heavy)"
    )
    ap.add_argument(
        "--v1", action="store_true", help="the version-1 exploit table (heavy)"
    )
    ap.add_argument(
        "--pid-validation", action="store_true", help="the pair's validation (heavy)"
    )
    ap.add_argument(
        "--mpc-gate", action="store_true", help="the NMPC's acceptance run (heavy)"
    )
    ap.add_argument("--workers", type=int, default=1, help="processes for --mpc-gate")
    ap.add_argument(
        "--mpc-horizon",
        type=int,
        default=None,
        help="the NMPC's horizon in --mpc-gate (default the shipped one)",
    )
    ap.add_argument(
        "--mpc-margins",
        type=float,
        nargs="+",
        default=None,
        help="candidate surge margins for --mpc-gate (default 0.02 0.03 0.05)",
    )
    ap.add_argument(
        "--gate-out",
        default=None,
        help="directory for --mpc-gate's saved arrays (default a folder in the "
        "system's temporary directory; the absolute path is printed)",
    )
    ap.add_argument(
        "--seeds", type=int, default=None, help="episodes per row of a heavy table"
    )
    ap.add_argument(
        "--rare-seeds", type=int, default=5120, help="episodes per rare tripper in --v1"
    )
    args = ap.parse_args()
    t0 = time.time()
    heavy = (
        args.hardness
        or args.sensitivity
        or args.v1
        or args.pid_validation
        or args.mpc_gate
    )
    status = 0
    for name in args.section or ([] if heavy else list(FAST)):
        {**FAST, **ON_REQUEST}[name]()
    if args.pid_validation:
        pid_validation(args.seeds or 4096)
    if args.hardness:
        hardness(args.seeds or 256)
    if args.sensitivity:
        sensitivity(args.seeds or 256)
    if args.v1:
        v1_table(args.seeds or 128, args.rare_seeds)
    if args.mpc_gate:
        status = mpc_gate(
            args.seeds or 128,
            args.workers,
            args.mpc_horizon,
            args.mpc_margins,
            args.gate_out,
        )
    say("=" * 78)
    say(f"runtime {time.time() - t0:.1f} s")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
