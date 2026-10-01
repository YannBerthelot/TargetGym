"""The unstable CSTR's controllers.

Enforces ``src/target_gym/pc_gym/unstable_cstr/PHYSICS.md``, section 7: the
cascade PID's gains and where they are read from, the stability guard its
factory applies, the held setpoint limits, the anti-windup, the agreement
between the functional core that scripts/unstable_cstr_numbers.py
--pid-validation runs and the stateful controller the suite runs, and the
PID's hold on scheduled episodes. Then the CasADi MPC: its model against the
env's, its fallback to the PID, what it plans on, its preview of the
schedule, its terminal weight, the factory's keywords, and the floor its
recorded hold sets.
"""

import copy
import functools
import json
import pathlib

import casadi
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.optimize import brentq

import target_gym.experts.pid as pid_mod
from target_gym import registry
from target_gym.experts.mpc import plan_params
from target_gym.pc_gym.unstable_cstr.env import (
    N_BLOCKS,
    UnstableCSTRParams,
    UnstableCSTRState,
    compute_next_state,
    compute_velocity,
    get_obs,
    live_target,
    steady_coolant,
    steady_temperature,
)
from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR
from target_gym.pc_gym.unstable_cstr.experts import (
    DEFAULT_GAINS,
    FALLBACK_STEP_BOUND_K,
    GUARD_LEVELS,
    GUARD_MIN_DECAY,
    HELD_GAINS,
    MPC_ERROR_SCALE,
    MPC_HORIZON,
    MPC_MOVE_WEIGHT,
    MPC_TVP,
    TUNED_GAINS,
    CascadeState,
    UnstableCascadeGains,
    UnstableCSTRCasadiMPC,
    UnstableCSTRCascadePID,
    cascade_init,
    cascade_step,
    check_cascade_gains,
    closed_loop_rates,
    coolant_from_raw,
    load_gains,
    make_unstable_cstr_mpc,
    make_unstable_cstr_pid,
    one_step_linearisation,
    pnr_line,
    point_of_no_return_batch,
    raw_from_coolant,
    setpoint_clip_margins,
    setpoint_window,
    terminal_weight,
)
from target_gym.utils import convert_raw_action_to_range

#: The inner-loop moves of the tuner's unguarded search from DEFAULT_GAINS at
#: the shipped 6 s lag, Kp_T x 2 and Kd_T x 0.7, on DEFAULT_GAINS' outer loop.
#: The loop grows at +1.07 /min at 0.45 (derived, closed_loop_rates). The whole
#: search, the "plain (no guard)" row of scripts/unstable_cstr_numbers.py
#: --sensitivity, also moves the outer loop and ends at Kp_T 14, Kd_T 0.28,
#: Kc_Ca 70, Ti_Ca 0.35 (measured), a loop that grows at +0.44 /min at 0.45
#: (derived).
PLAIN_6S = {**DEFAULT_GAINS, "Kp_T": 7.0 * 2.0, "Kd_T": 0.4 * 0.7}
#: Kp_T x 1.4 and Kd_T x 0.7 on DEFAULT_GAINS. The loop decays at 0.11 /min
#: at 0.45 (derived, closed_loop_rates), slower than GUARD_MIN_DECAY.
MARGINAL_6S = {**DEFAULT_GAINS, "Kp_T": 7.0 * 1.4, "Kd_T": 0.4 * 0.7}
#: Reactor temperature at the ignition fold (K), asserted by
#: test_unstable_cstr_physics.py::test_multiplicity_window_and_folds. Below it
#: the reactor is on its extinguished branch.
IGNITION_FOLD_T = 335.654
#: test_pid_holds_every_target's smallest accepted exact margin to the point
#: of no return (K, ours).
PID_MIN_PNR_MARGIN_K = 0.25
#: How far test_pid_holds_every_target lets T pass ``pnr_line(C_a)`` (K,
#: ours), for float32 rounding and a small overshoot of the setpoint ceiling,
#: which sits ``sp_line_margin`` below the line at the target level.
PID_LINE_TOL_K = 0.1
#: The recorded measurements test_floor_is_the_recorded_mpc_hold reads.
DATA = pathlib.Path(registry.__file__).parent / "data"


@pytest.fixture(scope="module")
def env():
    return UnstableCSTR()


@pytest.fixture(scope="module")
def params():
    return UnstableCSTRParams()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


@pytest.fixture(scope="module")
def planned(params):
    """The params the suite builds the MPC with (``plan_params``)."""
    return plan_params(registry.get("unstable_cstr"), params)


@pytest.fixture(scope="module")
def mpc(env, planned):
    """The MPC at its shipped horizon. Tests that step it build their own."""
    return make_unstable_cstr_mpc(env, planned)


def _obs(C_a, T, level, T_j=300.0):
    return np.array([C_a, T, T_j, level])


def _run_class(pid, env, params, step, key, n, levels=None):
    """``n`` steps of the stateful PID on the env, from ``reset_env(key)``
    (with its levels replaced when given). Returns the observations the PID
    saw, its actions and the setpoint it held after each step."""
    obs, state = env.reset_env(key, params)
    if levels is not None:
        state = state.replace(target_levels=jnp.asarray(levels, jnp.float32))
        obs = env.get_obs(state, params)
    seen, actions, setpoints = [], [], []
    for _ in range(n):
        seen.append(np.asarray(obs, float))
        action = pid(obs)
        actions.append(float(action[0]))
        setpoints.append(float(pid._cs.T_sp_prev))
        obs, state, _, _, _ = step(key, state, jnp.asarray(action), params)
    return np.array(seen), np.array(actions), np.array(setpoints)


# ---------------------------------------------------------------------------
# Gains, where they come from, and the guard
# ---------------------------------------------------------------------------


def test_default_gains_are_finite_and_tuned_ones_nonzero():
    """The coordinate search multiplies gains, so a tuned gain that
    starts at zero never moves. The tuned and held names split the gains."""
    assert all(np.isfinite(v) for v in DEFAULT_GAINS.values())
    assert all(DEFAULT_GAINS[k] != 0.0 for k in TUNED_GAINS)
    assert set(TUNED_GAINS) | set(HELD_GAINS) == set(DEFAULT_GAINS)
    assert not set(TUNED_GAINS) & set(HELD_GAINS)


def test_gains_are_read_under_the_task_key(monkeypatch):
    """The factory reads ``pid._gains_cache["unstable_cstr"]`` at call
    time (the tuner patches it), skips the ``note``, and nothing else in
    ``pid_gains.json`` starts with the task's name, since
    ``baseline_fingerprint`` collects gains by that prefix."""
    stored = pid_mod._load_gains()
    # On target, 0.2 K above T*: the inner loop alone acts, through Kp_T.
    obs = _obs(0.5, 350.2, 0.5, 300.0)
    shipped = make_unstable_cstr_pid()

    patched = {**load_gains(), "Kp_T": 8.0, "note": "patched by the test"}
    monkeypatch.setattr(pid_mod, "_gains_cache", {**stored, "unstable_cstr": patched})
    built = make_unstable_cstr_pid()
    assert built.gains["Kp_T"] == 8.0
    assert "note" not in built.gains
    assert built.step(obs)[0] != shipped.step(obs)[0]

    monkeypatch.undo()
    assert make_unstable_cstr_pid().gains == shipped.gains

    on_disk = json.loads(pid_mod._GAINS_FILE.read_text())
    ours = {k for k in on_disk if k.startswith("unstable_cstr")}
    assert ours <= {"unstable_cstr"}
    for key in ours:
        numeric = {k for k, v in on_disk[key].items() if not isinstance(v, str)}
        assert numeric <= set(DEFAULT_GAINS), numeric - set(DEFAULT_GAINS)


def test_shipped_gains_pass_the_guard():
    """The factory must build the shipped controller: its closed loop
    decays at ``GUARD_MIN_DECAY`` or faster at every target, and every
    equilibrium setpoint sits inside the setpoint window at a drift of -6, 0
    and +6 K."""
    gains = load_gains()
    assert np.all(closed_loop_rates(gains) <= -GUARD_MIN_DECAY)
    assert np.all(setpoint_clip_margins(gains) > 0.0)
    check_cascade_gains(gains)
    assert isinstance(make_unstable_cstr_pid(), UnstableCSTRCascadePID)


def test_guard_refuses_unstable_and_marginal_gains(monkeypatch):
    """``PLAIN_6S``, which grows, and ``MARGINAL_6S``, which decays too
    slowly, are refused, and so is a setpoint window too narrow to hold every
    target. A refused entry in ``pid_gains.json`` stops the factory. Checking
    one drift is enough, since the drift enters the balances additively and
    the rates at -6, 0 and +6 K, each linearised at its own equilibrium,
    agree."""
    assert closed_loop_rates(PLAIN_6S)[0] == pytest.approx(1.07, abs=0.01)
    assert closed_loop_rates(MARGINAL_6S)[0] == pytest.approx(-0.11, abs=0.01)
    np.testing.assert_allclose(
        closed_loop_rates(DEFAULT_GAINS),
        [-1.73, -1.84, -1.98, -2.19, -2.76],
        atol=0.01,
    )
    for gains in (PLAIN_6S, MARGINAL_6S, {**DEFAULT_GAINS, "sp_above": 0.5}):
        with pytest.raises(UnstableCascadeGains):
            check_cascade_gains(gains)

    with enable_x64():
        rates = [closed_loop_rates(load_gains(), Ti_dev=d) for d in (-6.0, 0.0, 6.0)]
    np.testing.assert_allclose(rates[0], rates[1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(rates[2], rates[1], rtol=0.0, atol=1e-6)

    monkeypatch.setattr(
        pid_mod,
        "_gains_cache",
        {**pid_mod._load_gains(), "unstable_cstr": MARGINAL_6S},
    )
    with pytest.raises(UnstableCascadeGains):
        make_unstable_cstr_pid()


# ---------------------------------------------------------------------------
# The held limits, anti-windup, reset, the functional core and the PID's hold
# on scheduled episodes
# ---------------------------------------------------------------------------


def test_setpoint_is_rate_limited_and_clipped(env, params, step):
    """Through a schedule with 0.1 mol/L switches both ways, the
    setpoint moves at most ``sp_rate`` per step (0.15 K) and stays inside
    [T* - 8, min(T* + 2, pnr_line(L) - 0.3)] of the level in force. The rate
    limit acts after the clip, so after a switch the setpoint ramps into the
    new window at the full rate; it never moves away from it. The ceiling is
    at least 0.3 K below the line across the band, and at the hot end it is
    the line that binds."""
    gains = DEFAULT_GAINS
    lo_band, hi_band = params.target_CA_range
    band = np.linspace(lo_band, hi_band, 201)
    ceiling = setpoint_window(band, gains, params)[1]
    assert np.all(ceiling <= pnr_line(band) - gains["sp_line_margin"] + 1e-9)
    assert np.all(
        ceiling <= steady_temperature(band, params, np) + gains["sp_above"] + 1e-9
    )

    levels = [0.55, 0.45, 0.55, 0.65, 0.55, 0.45]
    seen, _, T_sp = _run_class(
        UnstableCSTRCascadePID(gains, params),
        env,
        params,
        step,
        jax.random.PRNGKey(3),
        1200,
        levels,
    )
    ramp = gains["sp_rate"] * params.delta_t
    # The setpoint starts at the first observed T (``cascade_init``).
    previous = np.concatenate([[seen[0, 1]], T_sp[:-1]])
    moves = T_sp - previous
    assert np.abs(moves).max() <= ramp + 1e-9
    assert np.isclose(np.abs(moves), ramp).sum() > 20

    lo, hi = setpoint_window(seen[:, 3], gains, params)
    inside = (T_sp >= lo - 1e-9) & (T_sp <= hi + 1e-9)
    ramping_down = (T_sp > hi) & np.isclose(moves, -ramp)
    ramping_up = (T_sp < lo) & np.isclose(moves, ramp)
    assert np.all(inside | ramping_down | ramping_up)
    # The cold-going switches start above the new window.
    assert np.any(ramping_down)

    # At the hot end the ceiling is the line's: a large excess of C_a, with
    # the setpoint already near the top, stops exactly there. The floor
    # holds the same way against a large deficit.
    level = 0.45
    lo, hi = (float(x) for x in setpoint_window(level, gains, params))
    assert hi == pytest.approx(float(pnr_line(level)) - gains["sp_line_margin"])
    assert hi < float(steady_temperature(level, params, np)) + gains["sp_above"]
    for C_a, T_sp_prev, edge in (
        (level + 0.05, hi - 0.05, hi),
        (level - 0.10, lo + 0.05, lo),
    ):
        cs = CascadeState(integral=0.0, T_prev=T_sp_prev, T_sp_prev=T_sp_prev)
        new = cascade_step(gains, cs, _obs(C_a, T_sp_prev, level), params)[1]
        assert float(new.T_sp_prev) == pytest.approx(edge, abs=1e-12)


def test_integrator_freezes_while_limited(params):
    """Conditional integration: the integral moves by e dt when neither
    the coolant nor the setpoint is limited, and not at all when the coolant
    saturates, the setpoint is rate limited, or it is clipped to its window."""
    gains = DEFAULT_GAINS
    level = 0.55
    T_ff = float(steady_temperature(level, params, np))
    e = 0.001
    C_a = level - e
    T_sp_raw = T_ff - gains["Kc_Ca"] * e  # with a zero integral
    zero = 0.0

    def integral_after(obs, T_sp_prev, T_prev=None):
        cs = CascadeState(
            integral=zero,
            T_prev=obs[1] if T_prev is None else T_prev,
            T_sp_prev=T_sp_prev,
        )
        return float(cascade_step(gains, cs, obs, params)[1].integral)

    # Free: the plant at the setpoint, no ramp.
    assert integral_after(_obs(C_a, T_sp_raw, level), T_sp_raw) == pytest.approx(
        e * params.delta_t, rel=1e-12
    )
    # The coolant at its lower bound: T is 5 K above the setpoint.
    assert integral_after(_obs(C_a, T_sp_raw + 5.0, level), T_sp_raw) == 0.0
    # The coolant at its upper bound: T is 5 K below the setpoint.
    assert integral_after(_obs(C_a, T_sp_raw - 5.0, level), T_sp_raw) == 0.0
    # The setpoint rate limited: it was 1 K above where the PI puts it.
    assert integral_after(_obs(C_a, T_sp_raw + 1.0, level), T_sp_raw + 1.0) == 0.0
    # The setpoint clipped: a 0.05 mol/L excess of C_a asks for T* + 5 K.
    hi = float(setpoint_window(level, gains, params)[1])
    assert integral_after(_obs(level + 0.05, hi, level), hi) == 0.0


def test_reset_clears_the_controller(env, params, step):
    """After ``reset()`` the controller repeats its action sequence
    exactly, and the first step after a reset has no derivative kick and no
    setpoint jump: the setpoint starts at the measured T and ramps from
    there, however far T is from T*(target)."""
    pid = UnstableCSTRCascadePID()
    key = jax.random.PRNGKey(5)
    first = _run_class(pid, env, params, step, key, 80)[1]
    pid.reset()
    second = _run_class(pid, env, params, step, key, 80)[1]
    np.testing.assert_array_equal(first, second)

    # The first action is the same whatever the derivative gain: T_prev
    # starts at the observed T. Stepping on another observation first and
    # resetting leaves no trace of it.
    obs = _obs(0.54, 348.3, 0.55, 301.0)
    stiff = UnstableCSTRCascadePID({**DEFAULT_GAINS, "Kd_T": 4.0})
    assert stiff.step(obs)[0] == UnstableCSTRCascadePID().step(obs)[0]
    pid.step(_obs(0.60, 344.0, 0.55, 299.0))
    pid.reset()
    assert pid.step(obs)[0] == UnstableCSTRCascadePID().step(obs)[0]

    # A catch from the extinguished branch: T is 30 K below T*(0.45).
    catch = UnstableCSTRCascadePID()
    catch.step(_obs(0.93, 322.9, 0.45, 300.0))
    ramp = DEFAULT_GAINS["sp_rate"] * params.delta_t
    assert float(catch._cs.T_sp_prev) == pytest.approx(322.9 + ramp, abs=1e-9)


def test_functional_core_matches_the_stateful_pid(env, params, step):
    """``cascade_step(xp=jnp)`` under ``lax.scan`` gives the stateful
    controller's actions along a whole episode: exactly (1e-5) on the
    observations the controller saw, and within float32 rounding when it
    closes the loop on the env itself, as scripts/unstable_cstr_numbers.py
    --pid-validation and test_pid_holds_every_target run it."""
    key = jax.random.PRNGKey(0)
    gains = DEFAULT_GAINS
    seen, actions, _ = _run_class(
        UnstableCSTRCascadePID(gains, params), env, params, step, key, 1200
    )

    def replay(obs_seq):
        def body(cs, obs):
            u_raw, cs = cascade_step(gains, cs, obs, params, xp=jnp)
            return cs, u_raw

        cs0 = cascade_init(obs_seq[0], gains, params, xp=jnp)
        return np.asarray(jax.lax.scan(body, cs0, obs_seq)[1])

    with enable_x64():
        np.testing.assert_allclose(
            replay(jnp.asarray(seen, jnp.float64)), actions, rtol=0.0, atol=1e-5
        )

    def closed(carry, _):
        state, cs = carry
        obs = get_obs(state, params)
        u_raw, cs = cascade_step(gains, cs, obs, params, xp=jnp)
        state = env.step_env(key, state, u_raw, params)[1]
        return (state, cs), obs

    _, state0 = env.reset_env(key, params)
    cs0 = cascade_init(get_obs(state0, params), gains, params, xp=jnp)
    _, obs_seq = jax.lax.scan(closed, (state0, cs0), None, length=1200)
    obs_seq = np.asarray(obs_seq)
    assert np.abs(obs_seq[:, 0] - seen[:, 0]).max() <= params.e_floor
    assert np.abs(obs_seq[:, 1] - seen[:, 1]).max() <= 1e-2
    # The same six levels were tracked, so the schedule was not restarted.
    np.testing.assert_array_equal(obs_seq[:, 3], seen[:, 3])


def _catch_level(T_j, Ti_dev, params):
    """The extinguished steady state at jacket temperature ``T_j`` and drift
    ``Ti_dev``, as a level: the root of Tc*(L, dTi) = T_j past the ignition
    fold (C_a above about 0.74 mol/L)."""
    return brentq(
        lambda L: steady_coolant(L, Ti_dev, params, np) - T_j, 0.75, 0.999, xtol=1e-14
    )


def _scheduled_starts(env, params, n_random=64, n_alternating=16):
    """The start states, rollout keys and drift holds of the episodes
    test_pid_holds_every_target runs, built as scripts/unstable_cstr_numbers.py
    --pid-validation builds its three groups, with fewer random and
    alternating seeds.

    Random schedules: ``reset_env`` under seeds 1000 on, the drift free. The
    0.55/0.45 alternation: ``reset_env`` under seeds 2000 on, each draw moved
    onto that schedule with its offsets from its own first level kept, the
    drift held at +6 K. The 50 catches: catch i starts on the extinguished
    state at a jacket of 290 + i // 6 K and a drift of -6, 0 or +6 K (i % 3),
    held, with a constant 0.45 or 0.65 schedule (i % 2), under key 3000 + i."""

    def resets(seeds):
        keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, jnp.uint32))
        return jax.vmap(lambda k: env.reset_env(k, params)[1])(keys), keys

    f32 = functools.partial(jnp.asarray, dtype=jnp.float32)
    scheduled, scheduled_keys = resets(np.arange(1000, 1000 + n_random))

    drawn, alternating_keys = resets(np.arange(2000, 2000 + n_alternating))
    first = np.asarray(drawn.target_levels, float)[:, 0]
    level, drift = 0.55, 6.0
    alternating = drawn.replace(
        C_a=f32(level + np.asarray(drawn.C_a, float) - first),
        T=f32(
            steady_temperature(level, params, np)
            + np.asarray(drawn.T, float)
            - steady_temperature(first, params, np)
        ),
        T_j=f32(np.full(n_alternating, steady_coolant(level, drift, params, np))),
        Ti_dev=f32(np.full(n_alternating, drift)),
        target_levels=f32(np.tile([0.55, 0.45], (n_alternating, N_BLOCKS // 2))),
    )

    i = np.arange(50)
    catch_level = np.where(i % 2 == 0, 0.45, 0.65)
    catch_drift = np.array([-6.0, 0.0, 6.0])[i % 3]
    catch_T_j = 290.0 + i // 6
    C_a = np.array(
        [_catch_level(tj, d, params) for tj, d in zip(catch_T_j, catch_drift)]
    )
    catches = UnstableCSTRState(
        time=jnp.zeros(i.size, jnp.int32),
        C_a=f32(C_a),
        T=f32(steady_temperature(C_a, params, np)),
        T_j=f32(catch_T_j),
        Ti_dev=f32(catch_drift),
        target_levels=f32(np.repeat(catch_level[:, None], N_BLOCKS, axis=1)),
        block_clock=jnp.zeros(i.size, jnp.int32),
    )
    catch_keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(3000 + i, jnp.uint32))

    states = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x), scheduled, alternating, catches
    )
    keys = jnp.concatenate([scheduled_keys, alternating_keys, catch_keys])
    hold = jnp.asarray([False] * n_random + [True] * (n_alternating + i.size))
    return states, keys, hold


def _scheduled_episodes(env, params, gains, states, keys, hold):
    """One episode per start on ``step_env`` under the cascade's functional
    core (test_functional_core_matches_the_stateful_pid holds it to the
    stateful PID), with the drift held at its start value where ``hold`` is
    set. Returns the continuing C_a, T, T_j, drift, tracking error and trip
    flag of every step, each (episodes, steps)."""

    def one(state, key, held):
        drift = state.Ti_dev
        cs = cascade_init(get_obs(state, params), gains, params, xp=jnp)

        def body(carry, _):
            s, cs = carry
            u, cs = cascade_step(gains, cs, get_obs(s, params), params, xp=jnp)
            _, s2, _, _, info = env.step_env(key, s, jnp.reshape(u, (1,)), params)
            s2 = s2.replace(Ti_dev=jnp.where(held, drift, s2.Ti_dev))
            e = live_target(s2, params) - s2.C_a
            return (s2, cs), (s2.C_a, s2.T, s2.T_j, s2.Ti_dev, e, info["tripped"])

        n = params.max_steps_in_episode
        return jax.lax.scan(body, (state, cs), None, length=n)[1]

    out = jax.jit(jax.vmap(one))(states, keys, hold)
    return tuple(np.asarray(x) for x in out)


@pytest.mark.slow
def test_pid_holds_every_target(env, params):
    """The PID the registry builds (``make_pid``, so the tuned gains stored
    in ``src/target_gym/data/pid_gains.json``) on 130 scheduled episodes, 64
    random schedules with the drift on, 16 of the 0.55/0.45 alternation with
    the drift held at +6 K, and 50 catches from the extinguished branch at
    both ends of the band (``_scheduled_starts``).

    No episode trips. Every episode brings C_a within 0.01 mol/L of its
    target, and after that T never falls below the ignition fold, so the
    reactor never re-extinguishes. At the eight steps of each episode closest
    to the point-of-no-return line, the exact point of no return is at least
    ``PID_MIN_PNR_MARGIN_K`` above T, and on every step T is at most
    ``PID_LINE_TOL_K`` above ``pnr_line(C_a)``. On the 300 episodes of
    scripts/unstable_cstr_numbers.py --pid-validation, of which these are a
    subset, the smallest exact margin is 1.336 K and T stays 0.333 K or more
    below the line (measured)."""
    pid = registry.get("unstable_cstr").make_pid()
    assert isinstance(pid, UnstableCSTRCascadePID)
    states, keys, hold = _scheduled_starts(env, params)
    C_a, T, T_j, Ti_dev, e, tripped = _scheduled_episodes(
        env, params, pid.gains, states, keys, hold
    )
    n_episodes, n = C_a.shape
    assert n_episodes == 130

    assert not tripped.any(), np.flatnonzero(tripped.any(axis=1))
    captured = np.where(np.abs(e) < 0.01, np.arange(n)[None, :], n).min(axis=1)
    assert np.all(captured < n), np.flatnonzero(captured == n)
    later = np.arange(n)[None, :] > captured[:, None]
    re_extinguished = (later & (T < IGNITION_FOLD_T)).any(axis=1)
    assert not re_extinguished.any(), np.flatnonzero(re_extinguished)

    line = pnr_line(C_a) - T
    assert line.min() >= -PID_LINE_TOL_K, line.min()
    nearest = np.argsort(line, axis=1)[:, :8]
    rows = np.arange(n_episodes)[:, None]
    exact = (
        point_of_no_return_batch(
            C_a[rows, nearest], T_j[rows, nearest], Ti_dev[rows, nearest], params
        )
        - T[rows, nearest]
    )
    assert exact.min() >= PID_MIN_PNR_MARGIN_K, exact.min()


def test_guard_levels_span_the_band(params):
    """The guard checks both ends of the target band."""
    assert (min(GUARD_LEVELS), max(GUARD_LEVELS)) == params.target_CA_range


# ---------------------------------------------------------------------------
# The CasADi MPC and the floor its hold sets
# ---------------------------------------------------------------------------


def _pid_states(env, params, step, key, n):
    """``n`` steps of the shipped cascade on the env from ``reset_env(key)``.
    Returns the states it acted on and its raw actions."""
    pid = UnstableCSTRCascadePID(load_gains(), params)
    _, state = env.reset_env(key, params)
    states, actions = [], []
    for _ in range(n):
        action = pid(env.get_obs(state, params))
        states.append(state)
        actions.append(float(action[0]))
        state = step(key, state, jnp.asarray(action), params)[1]
    return states, np.array(actions)


def test_mpc_model_is_the_env_velocity(mpc):
    """The MPC's model is the one piece of physics written twice: the
    env's velocity is ``jnp`` code, which CasADi cannot trace. At 1000 random
    points, over and beyond the reachable set, its right-hand side equals the
    env's ``compute_velocity`` to 1e-12 relative, in float64."""
    p = mpc.params
    rng = np.random.default_rng(0)
    n = 1000
    x = np.stack(
        [
            rng.uniform(0.0, 1.0, n),
            rng.uniform(280.0, 400.0, n),
            rng.uniform(285.0, 315.0, n),
        ]
    )
    u = rng.uniform(-1.0, 1.0, n)
    d = rng.uniform(-8.0, 8.0, n)
    ours = np.asarray(mpc.rhs_function.map(n)(x, u[None, :], d[None, :]))
    with enable_x64():
        T_c = convert_raw_action_to_range(jnp.asarray(u), p.T_c_min, p.T_c_max)
        theirs = jax.vmap(lambda y, a, dd: compute_velocity(y, a, dd, p)[0])(
            jnp.asarray(x.T), T_c, jnp.asarray(d)
        )
    np.testing.assert_allclose(ours, np.asarray(theirs).T, rtol=1e-12, atol=0.0)


def test_mpc_predicts_one_step_like_the_plant(env, params, planned, step, mpc):
    """Model-review check 13. On 120 states of a shipped-PID episode
    with the drift on, one control interval of the MPC's model, integrated to
    1e-12 with the drift held as the env holds it over a step, lands where
    ``step_env`` under the planner's params does: the mean signed C_a error is
    under a tenth of ``e_floor`` and none exceeds ``e_floor``. A one-signed
    error would settle the MPC off its target."""
    states, actions = _pid_states(env, params, step, jax.random.PRNGKey(7), 1200)
    x = casadi.SX.sym("x", 3)
    u = casadi.SX.sym("u_raw")
    d = casadi.SX.sym("Ti_dev")
    interval = casadi.integrator(
        "interval",
        "cvodes",
        {"x": x, "p": casadi.vertcat(u, d), "ode": mpc.rhs_function(x, u, d)},
        0.0,
        mpc.mpc_dt,
        {"abstol": 1e-12, "reltol": 1e-12},
    )
    key = jax.random.PRNGKey(0)
    errors = []
    for state, action in list(zip(states, actions))[::10]:
        x0 = [float(state.C_a), float(state.T), float(state.T_j)]
        model = interval(x0=x0, p=[action, float(state.Ti_dev)])["xf"]
        plant = step(key, state, jnp.asarray([action]), planned)[1]
        errors.append(float(model[0]) - float(plant.C_a))
    errors = np.array(errors)
    assert errors.size == 120
    assert abs(errors.mean()) <= 0.1 * planned.e_floor, errors.mean()
    assert np.abs(errors).max() <= planned.e_floor, np.abs(errors).max()


def test_mpc_falls_back_to_the_pid(env, params, planned, step, monkeypatch):
    """A step whose solve fails is handed to the cascade PID: the action
    is the one the PID gives for that observation, the warm start is the one
    from before the failed solve, the failure is counted, and the next solve
    is clean. A solve that reports success with a NaN action is treated the
    same. Because the PID's memory tracks every action the MPC applies, its
    first action after 20 MPC steps starts within ``FALLBACK_STEP_BOUND_K``
    (0.5 K) of the coolant last applied, where a PID started cold jumps by
    more than 1 K. Measured, the step is 0.10 K here and the cold start's
    3.9 K. Gains the guard refuses stop the factory."""
    mpc = make_unstable_cstr_mpc(env, planned)
    key = jax.random.PRNGKey(2)
    _, state = env.reset_env(key, params)
    # At rest on the hot end with a 6 K hot feed, the equilibrium coolant is
    # 296.3 K, 3.7 K below the PID's 300 K bias, so a cold start would show.
    level = params.target_CA_range[0]
    state = state.replace(
        C_a=jnp.float32(level),
        T=jnp.float32(steady_temperature(level, params, np)),
        T_j=jnp.float32(steady_coolant(level, 6.0, params, np)),
        Ti_dev=jnp.float32(6.0),
        target_levels=jnp.full(state.target_levels.shape, level, jnp.float32),
    )
    for _ in range(20):
        action = mpc.step(env.get_obs(state, params), state)
        state = step(key, state, jnp.asarray([action]), params)[1]
    assert mpc.solve_failures == 0
    last_coolant = coolant_from_raw(action, planned)

    solve = mpc._mpc.make_step

    def failed_solve(x0):
        u = solve(x0)
        m = mpc._mpc
        m.opt_x_num.master = np.full(np.asarray(m.opt_x_num.master).shape, np.nan)
        m.u0 = np.array([np.nan])
        m.solver_stats = {
            **m.solver_stats,
            "success": False,
            "return_status": "Restoration_Failed",
        }
        return u

    def non_finite_solve(x0):
        u = solve(x0)
        mpc._mpc.u0 = np.array([np.nan])
        return np.full_like(u, np.nan)

    for n, fault, status in (
        (1, failed_solve, "Restoration_Failed"),
        (2, non_finite_solve, "non-finite action"),
    ):
        monkeypatch.setattr(mpc._mpc, "make_step", fault)
        obs = env.get_obs(state, params)
        warm = np.array(mpc._mpc.opt_x_num.master)
        expected = float(np.clip(copy.deepcopy(mpc._pid).step(obs)[0], -1.0, 1.0))
        cold = float(UnstableCSTRCascadePID(load_gains(), planned).step(obs)[0])
        action = mpc.step(obs, state)
        assert np.isfinite(action)
        assert action == expected
        np.testing.assert_array_equal(np.array(mpc._mpc.opt_x_num.master), warm)
        assert mpc.solve_failures == n
        assert mpc.last_return_status == status
        # The next solve's move penalty reads the action applied.
        assert float(mpc._mpc.u0.cat) == action
        if n == 1:
            first_step = abs(coolant_from_raw(action, planned) - last_coolant)
            assert first_step <= FALLBACK_STEP_BOUND_K
            assert abs(coolant_from_raw(cold, planned) - last_coolant) > 1.0
        state = step(key, state, jnp.asarray([action]), params)[1]

        monkeypatch.undo()
        action = mpc.step(env.get_obs(state, params), state)
        assert np.isfinite(action)
        assert mpc.solve_failures == n
        state = step(key, state, jnp.asarray([action]), params)[1]

    monkeypatch.setattr(
        pid_mod,
        "_gains_cache",
        {**pid_mod._load_gains(), "unstable_cstr": MARGINAL_6S},
    )
    with pytest.raises(UnstableCascadeGains):
        make_unstable_cstr_mpc(env, planned, horizon=5)


def test_mpc_plans_the_drift_mean(env, params, planned, mpc):
    """The suite builds the MPC on ``plan_params``, which zeroes the
    drift's noise and nothing else. The NLP never reads ``Ti_sigma``, so the
    first action is the same under either params: a regression guard for
    ``scripts/audit_mpc_horizons.py``, which skips ``plan_params`` and must
    still audit the shipped controller. The drift path the MPC plans on is the
    conditional mean a^k Ti_dev, the env's own update with the noise zeroed."""
    assert registry.get("unstable_cstr").noise_fields == ("Ti_sigma",)
    assert planned.Ti_sigma == 0.0
    assert planned == params.replace(Ti_sigma=0.0)

    _, state = env.reset_env(jax.random.PRNGKey(4), params)
    state = state.replace(Ti_dev=jnp.float32(4.0))
    obs = env.get_obs(state, params)
    first = [
        make_unstable_cstr_mpc(env, p, horizon=5).step(obs, state)
        for p in (params, planned)
    ]
    assert first[0] == first[1]

    drift = mpc.preview(state)["Ti_dev"]
    a = np.exp(-planned.delta_t / planned.Ti_tau)
    np.testing.assert_allclose(
        drift, 4.0 * a ** np.arange(MPC_HORIZON + 1), rtol=1e-12, atol=0.0
    )
    path, s = [], state
    for _ in range(MPC_HORIZON + 1):
        path.append(float(s.Ti_dev))
        s = compute_next_state(0.0, s, planned, jax.random.PRNGKey(0))[0]
    np.testing.assert_allclose(drift, path, rtol=1e-5)


def test_mpc_preview_matches_the_env(env, params, planned, mpc):
    """Stage k of the NLP is the state k steps ahead, and its target is
    the level the env scores that state against: inside a block, across a
    block boundary, and past the sixth block, where the last level holds.
    What the NLP receives through its tvp function is that preview."""
    _, state = env.reset_env(jax.random.PRNGKey(6), params)
    levels = np.asarray(state.target_levels, float)
    key = jax.random.PRNGKey(0)
    # (block clock, block of the first stage, block of the last stage). The
    # last case's stages run past the end of the sixth block.
    for clock, first, last in (
        (0, 0, 0),
        (params.block_steps - 10, 0, 1),
        (params.max_steps_in_episode - 15, 5, 5),
    ):
        s = start = state.replace(block_clock=jnp.int32(clock))
        expected = []
        for _ in range(mpc.horizon + 1):
            expected.append(float(live_target(s, planned)))
            s = compute_next_state(0.0, s, planned, key)[0]
        preview = mpc.preview(start)
        np.testing.assert_array_equal(preview["target"], expected)
        assert (expected[0], expected[-1]) == (levels[first], levels[last])

        mpc._update_setpoint(start)
        received = mpc._mpc.tvp_fun(0.0)
        for name in MPC_TVP:
            np.testing.assert_array_equal(
                [float(received["_tvp", k, name]) for k in range(mpc.horizon + 1)],
                preview[name],
            )


def test_terminal_weight_is_positive_definite(env, params, mpc):
    """The terminal weight solves the Riccati equation of the env's own
    one-step linearisation at C_a 0.45, where the open loop is unstable. It
    is finite and positive definite, and the LQR it implies is stable. The
    equilibrium the terminal cost is centred on is one of the MPC's model at
    the last stage's drift."""
    p = mpc.params
    A, B = one_step_linearisation(p)
    assert np.abs(np.linalg.eigvals(A)).max() > 1.0

    P = terminal_weight(p)
    assert np.all(np.isfinite(P))
    np.testing.assert_allclose(P, P.T, rtol=1e-9, atol=0.0)
    assert np.linalg.eigvalsh(P).min() > 0.0
    R = np.array([[MPC_MOVE_WEIGHT]])
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1.0
    Q = np.diag([1.0 / MPC_ERROR_SCALE**2, 1e-6, 1e-6])
    residual = A.T @ P @ A - A.T @ P @ B @ K + Q - P
    assert np.abs(residual).max() <= 1e-8 * np.abs(P).max()

    _, state = env.reset_env(jax.random.PRNGKey(8), params)
    state = state.replace(Ti_dev=jnp.float32(-4.0))
    preview = mpc.preview(state)
    x_eq = [preview[name][-1] for name in ("C_a_eq", "T_eq", "T_j_eq")]
    u_eq = raw_from_coolant(x_eq[2], p)
    rhs = np.asarray(mpc.rhs_function(x_eq, u_eq, preview["Ti_dev"][-1])).ravel()
    np.testing.assert_allclose(rhs, 0.0, atol=1e-9)
    assert x_eq[0] == preview["target"][-1]


def test_mpc_factory_takes_the_cheap_horizon(env, planned, mpc):
    """``tests/experts/test_mpc_baselines._cheap_mpc`` tries the gradient
    planners' ``n_iter`` first and needs a ``TypeError`` to fall through to
    ``horizon=5`` alone, through the registry's factory."""
    assert isinstance(mpc, UnstableCSTRCasadiMPC)
    assert mpc.horizon == MPC_HORIZON
    assert mpc.mpc_dt == planned.delta_t
    spec = registry.get("unstable_cstr")
    for kwargs in (
        {"horizon": 5, "n_iter": 2, "n_tail": 0},
        {"horizon": 5, "n_iter": 2},
    ):
        with pytest.raises(TypeError):
            spec.make_mpc(env, planned, **kwargs)
    assert spec.make_mpc(env, planned, horizon=5).horizon == 5
    assert isinstance(env.make_mpc(planned, horizon=5), UnstableCSTRCasadiMPC)


def test_floor_is_the_recorded_mpc_hold():
    """``e_floor`` and the NEA reference come from the MPC's recorded
    hold. With e_hold_min the smallest per-seed hold error that
    scripts/measure_hold.py recorded for the MPC (from minute 6 of each block
    to the switch, less the steps in which the MPC already moves toward the
    next level, which target_gym.eval.anticipations finds), ``e_floor`` is
    the larger of the analyser resolution (``precision_floor``, 1e-4 mol/L)
    and e_hold_min, ``rho_floor_tracking`` and ``rho_floor`` are both
    (e_hold_min / e_floor)^2, since there is no running cost, and the MPC's
    hold cost in the recorded protocol row is at least 0.98 of that
    reference. Each equality holds to 1 %."""
    spec = registry.get("unstable_cstr")
    p = spec.make_test_params()
    holds = json.loads((DATA / "hold_measurements.json").read_text())
    if "unstable_cstr" not in holds:
        pytest.fail(
            "no recorded MPC hold for unstable_cstr. Run `uv run python "
            "scripts/measure_hold.py --envs unstable_cstr` and commit "
            "src/target_gym/data/hold_measurements.json."
        )
    rows = json.loads((DATA / "protocol_results.json").read_text())
    if "unstable_cstr" not in rows:
        pytest.fail(
            "no recorded protocol row for unstable_cstr. Run `uv run python "
            "scripts/evaluate_baselines.py --envs unstable_cstr` and commit "
            "src/target_gym/data/protocol_results.json."
        )

    per_seed = holds["unstable_cstr"]["mpc"]["e_hold_per_seed"]
    e_hold_min = min(float(seed[0]) for seed in per_seed)
    assert p.e_floor == pytest.approx(max(p.precision_floor, e_hold_min), rel=0.01)
    reference = (e_hold_min / p.e_floor) ** 2
    assert p.rho_floor_tracking == pytest.approx(reference, rel=0.01)
    assert p.rho_floor == pytest.approx(p.rho_floor_tracking, rel=0.01)
    hold_tracking = rows["unstable_cstr"]["mpc"]["hold_tracking"]
    assert hold_tracking >= 0.98 * p.rho_floor_tracking, (
        hold_tracking,
        p.rho_floor_tracking,
    )
