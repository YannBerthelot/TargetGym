"""Environment behaviour of the unstable CSTR.

Enforces the task design in ``src/target_gym/pc_gym/unstable_cstr/PHYSICS.md``:
the observation, the action and the jacket lag, the stratified schedule and
where each state is scored, where resets sit against the point of no return,
the feed drift, the trip and its fresh restart through ``base.failure_kernel``,
both reward versions and the order version 1 gives, where full heating and full
cooling take the plant, the error envelope and the regime of validity, and
integrator convergence. Deviations D2, D3 and D5 are strict xfails at the end.
Runs in float32, as the env ships, except where a test states otherwise.
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.optimize import brentq

import target_gym.pc_gym.cstr.env as cstr_env
from target_gym import reward as R
from target_gym.pc_gym.unstable_cstr.env import (
    N_BLOCKS,
    UnstableCSTRParams,
    UnstableCSTRState,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    compute_velocity,
    get_obs,
    live_target,
    steady_coolant,
    steady_temperature,
)
from target_gym.pc_gym.unstable_cstr.env_jax import UnstableCSTR
from target_gym.pc_gym.unstable_cstr.experts import (
    cascade_init,
    cascade_step,
    load_gains,
    point_of_no_return_batch,
    raw_from_coolant,
)

TARGETS = (0.45, 0.50, 0.55, 0.60, 0.65)
#: Open-loop growth rate at each target (/min), asserted by
#: test_unstable_cstr_physics.py::test_every_target_is_a_saddle.
LAMBDA_PLUS = (3.158, 2.834, 2.422, 1.940, 1.391)
#: Reactor temperature at the ignition fold (K), asserted by
#: test_unstable_cstr_physics.py::test_multiplicity_window_and_folds.
IGNITION_FOLD_T = 335.654


@pytest.fixture(scope="module")
def env():
    return UnstableCSTR()


@pytest.fixture(scope="module")
def params():
    return UnstableCSTRParams()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


def _settled(params, level, Ti_dev=0.0, T_offset=0.0, levels=None, block_clock=0):
    """A float32 state on the equilibrium the env holds at ``level``."""
    f32 = lambda x: jnp.asarray(x, jnp.float32)  # noqa: E731
    return UnstableCSTRState(
        time=jnp.asarray(0, jnp.int32),
        C_a=f32(level),
        T=f32(steady_temperature(level, params, np) + T_offset),
        T_j=f32(steady_coolant(level, Ti_dev, params, np)),
        Ti_dev=f32(Ti_dev),
        target_levels=f32(np.full(N_BLOCKS, level) if levels is None else levels),
        block_clock=jnp.asarray(block_clock, jnp.int32),
    )


def _rollout(env, params, state, action, key, n):
    """``n`` steps under a constant key and a constant raw action. Returns the
    continuing states (stacked), the rewards and the trip flags."""

    def body(s, _):
        _, s2, r, _, info = env.step_env(key, s, action, params)
        return s2, (s2, r, info["tripped"])

    _, out = jax.lax.scan(body, state, None, length=n)
    return out


def _corners(params):
    """135 starts: the five targets, the reset box's 3 x 3 grid (corners,
    edge midpoints and centre) and a drift of -6, 0 and +6 K, with the jacket
    at Tc*(L, dTi). Float32 states, each on a constant schedule at its target."""
    rows = np.array(
        [
            (
                L + a * params.initial_CA_offset,
                float(steady_temperature(L, params, np)) + b * params.initial_T_offset,
                float(steady_coolant(L, d, params, np)),
                d,
                L,
            )
            for L in TARGETS
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for d in (-6.0, 0.0, 6.0)
        ]
    )
    f32 = functools.partial(jnp.asarray, dtype=jnp.float32)
    return UnstableCSTRState(
        time=jnp.zeros(len(rows), jnp.int32),
        C_a=f32(rows[:, 0]),
        T=f32(rows[:, 1]),
        T_j=f32(rows[:, 2]),
        Ti_dev=f32(rows[:, 3]),
        target_levels=f32(np.repeat(rows[:, 4:5], N_BLOCKS, axis=1)),
        block_clock=jnp.zeros(len(rows), jnp.int32),
    )


def _pinned_rollout(env, params, states, raw, n):
    """``n`` steps of ``step_env`` from each state at a constant raw action,
    with the drift pinned at the state's own value before every step, under a
    constant key. Returns T of the continuing states and the trip flags, each
    (batch, n)."""
    key = jax.random.PRNGKey(0)
    action = jnp.array([raw], jnp.float32)

    def one(state):
        def body(s, _):
            s = s.replace(Ti_dev=state.Ti_dev)
            _, s2, _, _, info = env.step_env(key, s, action, params)
            return s2, (s2.T, info["tripped"])

        return jax.lax.scan(body, state, None, length=n)[1]

    T, tripped = jax.jit(jax.vmap(one))(states)
    return np.asarray(T), np.asarray(tripped)


def _bang_bang(env, params, keys, pins, n):
    """One episode of random bang-bang coolant per key, from ``reset_env``: the
    command is full heating or full cooling and switches at random, at a rate
    drawn per episode between 2 and 30 % per step (holds of about 10 s to
    2.5 min). The drift is pinned before every step to the episode's entry
    of ``pins``, or left free where that entry is NaN. Returns the physics'
    proposal of every step (C_a, T, T_j), before any restart, and the trip
    flags, each (batch, n)."""

    def episode(key, pin):
        _, state = env.reset_env(key, params)
        flip_key, rate_key, sign_key = jax.random.split(jax.random.fold_in(key, 1), 3)
        rate = jax.random.uniform(rate_key, minval=0.02, maxval=0.3)
        flips = jax.random.bernoulli(flip_key, rate, (n,))
        first = jnp.where(jax.random.bernoulli(sign_key), 1.0, -1.0)
        actions = first * jnp.where(jnp.cumsum(flips) % 2 == 0, 1.0, -1.0)

        def body(s, u):
            s = s.replace(Ti_dev=jnp.where(jnp.isnan(pin), s.Ti_dev, pin))
            proposal = compute_next_state(u, s, params, key)[0]
            _, s2, _, _, info = env.step_env(key, s, jnp.reshape(u, (1,)), params)
            return s2, (proposal.C_a, proposal.T, proposal.T_j, info["tripped"])

        return jax.lax.scan(body, state, actions)[1]

    out = jax.jit(jax.vmap(episode))(keys, jnp.asarray(pins, jnp.float32))
    return tuple(np.asarray(x) for x in out)


def _closed_loop(env, params, key, raw=None):
    """One episode from ``reset_env(key)`` under that constant key, as the
    suite's rollouts run: the shipped cascade PID (its functional core, the
    one the numbers script runs), or a constant raw action. Returns the
    continuing C_a and T, the reward and the trip flag of every step."""
    gains = load_gains()
    obs, state = env.reset_env(key, params)
    cs = cascade_init(obs, gains, params, xp=jnp)

    def body(carry, _):
        s, cs = carry
        if raw is None:
            u, cs = cascade_step(gains, cs, get_obs(s, params), params, xp=jnp)
        else:
            u = jnp.float32(raw)
        _, s2, r, _, info = env.step_env(key, s, jnp.reshape(u, (1,)), params)
        return (s2, cs), (s2.C_a, s2.T, r, info["tripped"])

    return jax.lax.scan(body, (state, cs), None, length=params.max_steps_in_episode)[1]


def _extinguished(params, T_j=300.0):
    """The extinguished steady state at jacket temperature ``T_j`` with no
    drift, as a level: the one root of Tc*(L) = T_j past the ignition fold
    (C_a above about 0.74 mol/L)."""
    return brentq(
        lambda L: steady_coolant(L, 0.0, params, np) - T_j, 0.75, 0.99, xtol=1e-14
    )


def _full_heating_from_extinction(params, integration_method="rk4_1", n=40):
    """Full heating from the extinguished state at a 300 K jacket, drift off.
    Returns T of the continuing states and the trip flags."""
    env = UnstableCSTR(integration_method=integration_method)
    p = params.replace(Ti_sigma=0.0)
    state = _settled(p, _extinguished(p))
    states, _, tripped = _rollout(
        env, p, state, jnp.array([1.0]), jax.random.PRNGKey(0), n
    )
    return np.asarray(states.T), np.asarray(tripped)


# ---------------------------------------------------------------------------
# 1. Layout, action and jacket
# ---------------------------------------------------------------------------


def test_layout(env, params):
    """Eleven state entries besides ``time``: four scalars, six levels and the
    block clock. The observation is ``[C_a, T, T_j, live target]``."""
    obs, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert list(UnstableCSTRState.__dataclass_fields__) == [
        "time",
        "C_a",
        "T",
        "T_j",
        "Ti_dev",
        "target_levels",
        "block_clock",
    ]
    for name in ("C_a", "T", "T_j", "Ti_dev"):
        leaf = getattr(state, name)
        assert leaf.shape == () and leaf.dtype == jnp.float32, name
    assert state.target_levels.shape == (N_BLOCKS,)
    assert state.target_levels.dtype == jnp.float32
    assert state.block_clock.shape == () and state.block_clock.dtype == jnp.int32
    entries = sum(
        int(np.size(getattr(state, f)))
        for f in UnstableCSTRState.__dataclass_fields__
        if f != "time"
    )
    assert entries == 11

    expected = [state.C_a, state.T, state.T_j, live_target(state, params)]
    np.testing.assert_array_equal(np.asarray(obs), np.asarray(expected))
    assert obs.shape == env.obs_shape == env.observation_space(params).shape == (4,)
    assert env.action_space(params).shape == (1,)
    assert env.obs_value_index == 0 and env.obs_target_index == 3
    assert params.max_steps_in_episode == N_BLOCKS * params.block_steps


def test_action_maps_to_the_coolant_range(params):
    """Raw -1, 0 and +1 settle the jacket at 290, 300 and 310 K, and a raw
    action beyond [-1, 1] is clipped."""
    key = jax.random.PRNGKey(0)
    for raw, T_c in (
        (-1.0, 290.0),
        (0.0, 300.0),
        (1.0, 310.0),
        (-3.0, 290.0),
        (3.0, 310.0),
    ):
        state = _settled(params, 0.55)
        for _ in range(30):  # 15 jacket time constants
            state, _ = compute_next_state(jnp.float32(raw), state, params, key)
        assert float(state.T_j) == pytest.approx(T_c, abs=1e-3), raw


def test_jacket_lags_the_command(params):
    """After a 5 K command step the jacket covers about 63 % of it in one
    ``tau_j`` (two steps) and approaches it without overshoot."""
    key = jax.random.PRNGKey(0)
    state = _settled(params, 0.55).replace(T_j=jnp.float32(300.0))
    T_j = [300.0]
    for _ in range(40):
        state, _ = compute_next_state(
            jnp.float32(raw_from_coolant(305.0, params)), state, params, key
        )
        T_j.append(float(state.T_j))
    T_j = np.array(T_j)
    steps_per_tau = round(params.tau_j / params.delta_t)
    assert steps_per_tau == 2
    assert (T_j[steps_per_tau] - 300.0) / 5.0 == pytest.approx(
        1.0 - np.exp(-1.0), abs=0.01
    )
    assert np.all(np.diff(T_j) >= 0.0)
    assert np.all(T_j <= 305.0 + 1e-4)


# ---------------------------------------------------------------------------
# 2. Schedule and reset
# ---------------------------------------------------------------------------


def test_reset_draws_a_stratified_schedule(env, params):
    """Over 256 seeds every level lies in the band, the five moves are the
    switch multiset in some order (every order appears), and so every episode
    has the same total squared move."""
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    _, states = jax.jit(jax.vmap(lambda k: env.reset_env(k, params)))(keys)
    levels = np.asarray(states.target_levels, float)
    lo, hi = params.target_CA_range
    assert np.all(levels >= lo - 1e-6) and np.all(levels <= hi + 1e-6)

    moves = np.diff(levels, axis=1)
    sizes = np.abs(moves)
    np.testing.assert_allclose(
        np.sort(sizes, axis=1),
        np.broadcast_to(np.sort(params.switch_sizes), sizes.shape),
        atol=1e-6,
    )
    np.testing.assert_allclose((moves**2).sum(axis=1), 0.035, atol=1e-6)

    orders = {tuple(np.round(row, 2)) for row in sizes}
    assert orders == set(itertools.permutations(params.switch_sizes))
    assert np.any(moves > 0) and np.any(moves < 0)
    assert levels[:, 0].min() < lo + 0.02 and levels[:, 0].max() > hi - 0.02

    assert np.all(np.asarray(states.block_clock) == 0)
    assert np.all(np.asarray(states.time) == 0)


def test_reset_is_warm_with_a_clipped_drift(env, params):
    """Every start is within the reset box around the first level, with the
    jacket already at the steady coolant for the drawn drift, and the initial
    drift is the stationary law clipped at 3 standard deviations (6 K)."""
    keys = jax.random.split(jax.random.PRNGKey(1), 4096)
    _, s = jax.jit(jax.vmap(lambda k: env.reset_env(k, params)))(keys)
    first = np.asarray(s.target_levels[:, 0], float)
    Ti_dev = np.asarray(s.Ti_dev, float)
    assert np.all(np.abs(np.asarray(s.C_a) - first) <= params.initial_CA_offset + 1e-6)
    T_star = steady_temperature(first, params, np)
    assert np.all(np.abs(np.asarray(s.T) - T_star) <= params.initial_T_offset + 1e-4)
    np.testing.assert_allclose(
        np.asarray(s.T_j), steady_coolant(first, Ti_dev, params, np), atol=1e-3
    )

    clip = params.Ti_dev_reset_clip * params.Ti_sigma
    assert clip == 6.0
    assert np.all(np.abs(Ti_dev) <= clip)
    # About 0.27 % of draws pass 3 sd, so the clip binds on some of these.
    assert np.isclose(np.abs(Ti_dev).max(), clip)
    assert np.std(Ti_dev) == pytest.approx(params.Ti_sigma, rel=0.05)
    assert abs(np.mean(Ti_dev)) < 0.1


def test_every_reset_is_inside_the_point_of_no_return(env, params):
    """Full cooling recovers every start. The worst corner of the reset box,
    C_a 0.01 mol/L and T 1.5 K above the hottest target's equilibrium with
    the drift at its 6 K clip, is 0.51 K inside the exact point of no return
    (derived, scripts/unstable_cstr_numbers.py --section pnr). That corner
    would cross it only at a drift between +9.0 and +9.5 K, past the clip.
    Every other corner of the 135, and every one of 512 drawn resets, has a
    wider margin."""
    lo = params.target_CA_range[0]
    C_a = lo + params.initial_CA_offset
    T = float(steady_temperature(lo, params, np)) + params.initial_T_offset

    def corner_margin(Ti_dev):
        T_j = steady_coolant(lo, Ti_dev, params, np)
        return float(point_of_no_return_batch(C_a, T_j, Ti_dev, params)[0]) - T

    clip = params.Ti_dev_reset_clip * params.Ti_sigma
    worst = corner_margin(clip)
    assert worst >= 0.4  # 0.511
    assert corner_margin(9.0) > 0.0 > corner_margin(9.5)

    corners = _corners(params)
    margins = point_of_no_return_batch(
        corners.C_a, corners.T_j, corners.Ti_dev, params
    ) - np.asarray(corners.T)
    assert margins.min() == pytest.approx(worst, abs=1e-4)

    keys = jax.random.split(jax.random.PRNGKey(7), 512)
    _, s = jax.jit(jax.vmap(lambda k: env.reset_env(k, params)))(keys)
    drawn = point_of_no_return_batch(s.C_a, s.T_j, s.Ti_dev, params) - np.asarray(
        s.T, float
    )
    assert drawn.min() >= worst - 1e-3


def test_live_target_follows_the_block_clock(env, params, step):
    """The live target is ``level[min(clock // block_steps, 5)]``, the last
    level holds past the sixth block, and the step that enters a block is
    scored against that block's level, which its observation already shows.
    The observation reads ``block_steps`` from the params the step is given,
    as the reward does, so a shorter block moves both."""
    levels = np.array([0.45, 0.50, 0.55, 0.60, 0.65, 0.60], np.float32)
    state = _settled(params, 0.45, levels=levels)
    short = params.replace(block_steps=50, max_steps_in_episode=300, Ti_sigma=0.0)
    for p in (params, short):
        b = p.block_steps
        for clock in (0, b - 1, b, 2 * b, 5 * b - 1, 5 * b, 6 * b - 1, 6 * b, 5000):
            s = state.replace(block_clock=jnp.asarray(clock, jnp.int32))
            expected = levels[min(clock // b, N_BLOCKS - 1)]
            assert float(live_target(s, p)) == expected, (b, clock)

    for p in (params.replace(Ti_sigma=0.0), short):
        before = state.replace(block_clock=jnp.asarray(p.block_steps - 1, jnp.int32))
        action = jnp.array([raw_from_coolant(float(before.T_j), p)])
        obs, after, reward, _, info = step(jax.random.PRNGKey(0), before, action, p)
        assert not bool(info["tripped"])
        assert int(after.block_clock) == p.block_steps
        assert float(obs[3]) == levels[1], p.block_steps
        assert float(reward) == pytest.approx(
            -(((levels[1] - float(after.C_a)) / p.e_floor) ** 2), rel=1e-5
        )


# ---------------------------------------------------------------------------
# 3. Feed drift
# ---------------------------------------------------------------------------


def test_feed_drift_is_a_zero_mean_ou_process(env, params):
    """Under a constant rollout key, as every shipped rollout helper passes,
    the drift is a zero-mean AR(1) with coefficient exp(-delta_t / Ti_tau) and
    stationary sd ``Ti_sigma``: no ratchet from a repeated innovation. Driven
    at full cooling, which never trips (a trip would redraw the drift)."""
    n_seeds, n_steps = 64, params.max_steps_in_episode
    action = jnp.array([-1.0])

    def episode(key):
        _, state = env.reset_env(key, params)
        states, _, tripped = _rollout(env, params, state, action, key, n_steps)
        return jnp.concatenate([state.Ti_dev[None], states.Ti_dev]), tripped

    keys = jax.random.split(jax.random.PRNGKey(2), n_seeds)
    x, tripped = jax.jit(jax.vmap(episode))(keys)
    x = np.asarray(x, float)
    assert not np.any(np.asarray(tripped))

    # The per-seed means are independent, so their spread carries the
    # autocorrelation into the standard error.
    means = x.mean(axis=1)
    se = means.std(ddof=1) / np.sqrt(n_seeds)
    assert abs(means.mean()) < 3.0 * se
    assert np.sqrt((x**2).mean()) == pytest.approx(params.Ti_sigma, abs=0.22)
    # Pooled regression of x[t+1] on x[t] through the origin.
    a = np.exp(-params.delta_t / params.Ti_tau)
    assert a == pytest.approx(0.99501, abs=1e-5)
    lag1 = (x[:, :-1] * x[:, 1:]).sum() / (x[:, :-1] ** 2).sum()
    assert lag1 == pytest.approx(a, abs=0.002)

    # Without innovations the drift decays as a^k from its initial value.
    quiet = params.replace(Ti_sigma=0.0)
    state = _settled(quiet, 0.55, Ti_dev=4.0)
    states, _, tripped = _rollout(env, quiet, state, action, keys[0], 200)
    assert not np.any(np.asarray(tripped))
    np.testing.assert_allclose(
        np.asarray(states.Ti_dev), 4.0 * a ** np.arange(1, 201), rtol=1e-4
    )


# ---------------------------------------------------------------------------
# 4. Trip and restart
# ---------------------------------------------------------------------------


def test_trip_is_one_sided_with_a_validity_guard(params, step):
    """High temperature trips at 365 K. There is no low trip: an extinguished
    state continues. T at or below 250 K, a negative C_a and a NaN trip as a
    validity guard. One step from 400 K trips."""
    base = _settled(params, 0.55)
    tripping = [
        base.replace(T=jnp.float32(365.0)),
        base.replace(T=jnp.float32(380.0)),
        base.replace(T=jnp.float32(250.0)),
        base.replace(T=jnp.float32(200.0)),
        base.replace(C_a=jnp.float32(-1e-3)),
        base.replace(T=jnp.float32(np.nan)),
        base.replace(C_a=jnp.float32(np.nan)),
    ]
    continuing = [
        base,
        base.replace(T=jnp.float32(364.9)),
        base.replace(C_a=jnp.float32(0.9), T=jnp.float32(320.0)),  # extinguished
        base.replace(C_a=jnp.float32(0.0)),
    ]
    for s in tripping:
        assert bool(check_is_terminal(s, params)[0]), (float(s.C_a), float(s.T))
    for s in continuing:
        assert not bool(check_is_terminal(s, params)[0]), (float(s.C_a), float(s.T))

    hot = base.replace(T=jnp.float32(400.0))
    _, _, _, terminated, info = step(
        jax.random.PRNGKey(0), hot, jnp.array([-1.0]), params
    )
    assert bool(info["tripped"])
    assert not bool(terminated)


def test_a_trip_restarts_fresh_on_the_same_clock(env, params, step):
    """Full heating trips. The step is charged ``restart_steps x failure_cost``
    in version 2 and exactly ``-restart_steps`` in version 1, and the window
    continues from the state ``reset_env`` draws from the step's key (new
    schedule, new drift, block clock 0), with ``time`` running on."""
    key = jax.random.PRNGKey(3)
    v1 = params.replace(reward_version=1)
    _, state = env.reset_env(key, params)
    heat = jnp.array([1.0])
    for n in range(1, 200):
        _, new, r2, _, info = step(key, state, heat, params)
        _, _, r1, _, _ = step(key, state, heat, v1)
        if bool(info["tripped"]):
            break
        state = new
    else:
        pytest.fail("full heating did not trip within 200 steps")

    # Equal to float32 rounding: the step's copy of the reset is compiled
    # inside the step, where XLA may fuse its arithmetic differently.
    fresh = env.reset_env(key, params)[1].replace(time=new.time)
    for name in UnstableCSTRState.__dataclass_fields__:
        np.testing.assert_allclose(
            np.asarray(getattr(new, name)),
            np.asarray(getattr(fresh, name)),
            rtol=1e-6,
            err_msg=name,
        )
    assert int(new.time) == n
    assert int(new.block_clock) == 0
    # 7.26e10 is not a float32; float32 steps by 8192 there.
    assert float(r2) == pytest.approx(
        -params.restart_steps * params.failure_cost, rel=1e-6
    )
    assert float(r1) == -1200.0


def test_restart_time_is_cstrs_hour(params):
    """cstr's provisional one-hour restart, 240 steps of 15 s, is 1200 steps
    at 3 s, and a tripped step costs twice the largest reachable tracking
    cost for each of them. The largest error is the closed form behind
    ``CA_error_max``: the feed concentration above the lowest target, or the
    highest target above the steady C_a at the trip temperature."""
    shipped = cstr_env.CSTRParams()
    assert params.restart_steps == 1200
    assert params.restart_steps == pytest.approx(
        shipped.restart_steps * shipped.delta_t / params.delta_t
    )
    assert params.failure_cost == 2.0 * (params.CA_error_max / params.e_floor) ** 2
    assert R.trip_cost(params) == pytest.approx(7.26e10)

    qV = params.q / params.V
    k_trip = params.k0 * np.exp(-params.EA_over_R / params.T_trip)
    lo, hi = params.target_CA_range
    closed_form = max(params.Caf - lo, hi - qV * params.Caf / (qV + k_trip))
    assert params.CA_error_max == pytest.approx(closed_form, abs=1e-9)
    assert hi - qV * params.Caf / (qV + k_trip) == pytest.approx(0.386, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. Rewards
# ---------------------------------------------------------------------------


def _exact_state(params, C_a, T=350.0, level=0.5):
    """A float64 NumPy state, for scoring the reward exactly with ``xp=np``."""
    return UnstableCSTRState(
        time=0,
        C_a=C_a,
        T=T,
        T_j=300.0,
        Ti_dev=0.0,
        target_levels=np.full(N_BLOCKS, level),
        block_clock=0,
    )


def test_v2_reward_units(params):
    """Version 2 is minus the floor-normalised squared error: 0 at the target,
    -1 one ``e_floor`` away. Its only terms are tracking and the trip; the
    coolant is a setting with no tariff, so there is no running cost."""
    terms = compute_reward_terms(_exact_state(params, 0.5), params, np)
    assert set(terms) == {"tracking", "failure"}
    assert float(compute_reward(_exact_state(params, 0.5), params, np)) == 0.0
    for sign in (-1.0, 1.0):
        s = _exact_state(params, 0.5 + sign * params.e_floor)
        assert float(compute_reward(s, params, np)) == pytest.approx(-1.0, rel=1e-9)

    tripped = _exact_state(params, 0.3, T=366.0)
    terms = compute_reward_terms(tripped, params, np)
    assert float(terms["tracking"]) == 0.0
    assert float(terms["failure"]) == R.trip_cost(params)
    assert float(compute_reward(tripped, params, np)) == -R.trip_cost(params)


def test_v1_is_log_scaled_and_charges_downtime(params):
    """Version 1 is 1 at zero error and 0 at ``CA_error_max``, pays an equal
    increment for each halving of the error above the floor, and is exactly
    ``-restart_steps`` on a tripped step."""
    v1 = params.replace(reward_version=1)

    def score(error, T=350.0):
        return float(compute_reward(_exact_state(v1, 0.5 - error, T=T), v1, np))

    assert score(0.0) == 1.0
    assert score(-v1.CA_error_max) == pytest.approx(0.0, abs=1e-12)
    errors = [0.4, 0.2, 0.1, 0.05, 0.025]
    gains = np.diff([score(e) for e in errors])
    assert np.all(gains > 0.0)
    np.testing.assert_allclose(gains, gains[0], rtol=0.01)
    assert all(0.0 <= score(e) <= 1.0 for e in errors)

    assert score(0.0, T=366.0) == -1.0 * v1.restart_steps == -1200.0
    assert score(0.3, T=366.0) == -1200.0


@pytest.mark.slow
def test_v1_ranks_tripping_below_safe(env, params):
    """The exploit version 1 is fixed against, measured on 32 seeds. Full
    heating and a constant 305.75 K coolant trip over and over, and with 0 on
    the tripped step each restart put the plant back beside its target for
    free, so both outscored a trip-free constant, 295.25 K. With
    ``-restart_steps`` on the tripped step both rank below it, and the
    shipped PID ranks above it. The constants come from a 0.25 K scan of
    constant coolants under the unfixed version 1, which
    scripts/unstable_cstr_numbers.py --v1 runs on the env."""
    v1 = params.replace(reward_version=1)
    n = v1.max_steps_in_episode
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(32))

    def scores(raw=None):
        _, _, r, tripped = jax.jit(
            jax.vmap(lambda k: _closed_loop(env, v1, k, raw=raw))
        )(keys)
        return np.asarray(r, float).sum(axis=1) / n, np.asarray(tripped).sum(axis=1)

    heat, heat_trips = scores(1.0)
    hot, hot_trips = scores(raw_from_coolant(305.75, v1))
    safe, safe_trips = scores(raw_from_coolant(295.25, v1))
    pid, pid_trips = scores()

    assert np.all(heat_trips > 0) and np.all(hot_trips > 0)
    assert not safe_trips.any() and not pid_trips.any()
    # With 0 on the tripped step instead of -restart_steps, the exploit.
    unfixed = [
        x + v1.restart_steps * k / n for x, k in ((heat, heat_trips), (hot, hot_trips))
    ]
    assert all(u.mean() > safe.mean() for u in unfixed)
    # As shipped: every tripping policy below the safe constant, below the PID.
    assert heat.mean() < safe.mean() and hot.mean() < safe.mean()
    assert safe.mean() < pid.mean()


# ---------------------------------------------------------------------------
# 6. Instability, measured by stepping the env
# ---------------------------------------------------------------------------


def test_unforced_error_grows_at_the_saddle_rate(env, params):
    """With the coolant held at Tc* and the drift off, a 0.02 K start along the
    unstable eigenvector grows at lambda+ at every target (within 5 % over 10
    steps). This asserts the check 7 allowlist entry by stepping the env,
    independently of any formula."""
    p = params.replace(Ti_sigma=0.0)
    key = jax.random.PRNGKey(0)
    n = 10
    for L, lam in zip(TARGETS, LAMBDA_PLUS):
        reference = _settled(p, L)
        with enable_x64():
            J = jax.jacfwd(lambda x: compute_velocity(x, 0.0, 0.0, p)[0][:2])(
                jnp.array(
                    [L, steady_temperature(L, p, np), steady_coolant(L, 0.0, p, np)]
                )
            )[:, :2]
        w, v = np.linalg.eig(np.asarray(J))
        direction = v[:, np.argmax(w.real)].real
        direction = direction / direction[1]
        kicked = reference.replace(
            C_a=reference.C_a + jnp.float32(0.02 * direction[0]),
            T=reference.T + jnp.float32(0.02),
        )
        action = jnp.array([raw_from_coolant(float(reference.T_j), p)])
        held, _, tripped_ref = _rollout(env, p, reference, action, key, n)
        grown, _, tripped = _rollout(env, p, kicked, action, key, n)
        assert not np.any(np.asarray(tripped_ref)) and not np.any(np.asarray(tripped))
        gap = float(grown.T[-1]) - float(held.T[-1])
        start = float(kicked.T) - float(reference.T)
        rate = np.log(gap / start) / (n * p.delta_t)
        assert rate == pytest.approx(lam, rel=0.05), L


def test_zero_action_leaves_the_middle_state(env, params):
    """Raw 0 (a 300 K jacket) from each target +-0.5 K: the reactor leaves a
    +-2 K band within 40 steps, and either trips or ends extinguished."""
    p = params.replace(Ti_sigma=0.0)
    key = jax.random.PRNGKey(0)
    for L in TARGETS:
        T_star = float(steady_temperature(L, p, np))
        for offset in (-0.5, 0.5):
            state = _settled(p, L, T_offset=offset)
            states, _, tripped = _rollout(env, p, state, jnp.array([0.0]), key, 200)
            tripped = np.asarray(tripped)
            left = tripped | (np.abs(np.asarray(states.T) - T_star) > 2.0)
            assert np.argmax(left) < 40 and left.any(), (L, offset)
            extinguished = (
                float(states.T[-1]) < IGNITION_FOLD_T and float(states.C_a[-1]) > 0.8
            )
            assert tripped.any() or extinguished, (L, offset)


# ---------------------------------------------------------------------------
# 7. Where full heating, full cooling and extinction take the plant
# ---------------------------------------------------------------------------


def test_full_heating_trips_from_every_corner(env, params):
    """Conformance check 8's fact for this task (docs/model-review-checklist.md):
    the coolant moves the plant, and fast. Full heating trips it from each of
    135 starts (the five targets, the reset box's 3 x 3 grid, and a drift of
    -6, 0 and +6 K, pinned) in 5 to 20 steps, median 9, which is 15 to 60 s
    (derived, scripts/unstable_cstr_numbers.py --section reach)."""
    _, tripped = _pinned_rollout(env, params, _corners(params), 1.0, 25)
    assert tripped.any(axis=1).all()
    first = np.argmax(tripped, axis=1) + 1
    assert first.min() >= 5 and first.max() <= 20
    assert np.median(first) == 9


@pytest.mark.slow
def test_full_cooling_never_trips(env, params):
    """Full cooling held for a whole episode from the same 135 starts never
    trips: every start is inside the point of no return. The hottest it gets
    is 355.28 K (derived, --section reach), from the hottest target's corner
    at a +6 K drift, and after an hour every start is extinguished, below the
    ignition fold (``IGNITION_FOLD_T``)."""
    T, tripped = _pinned_rollout(
        env, params, _corners(params), -1.0, params.max_steps_in_episode
    )
    assert not tripped.any()
    assert T.max() < 356.0
    assert np.all(T[:, -1] < IGNITION_FOLD_T)


def test_extinction_is_recoverable(params):
    """The failure is one-sided. The extinguished branch is not a trip, and
    full heating brings the plant back from it quickly: from the low state at
    a 300 K jacket (324.48 K), T reaches 340 K after 18 steps, and held there
    the plant trips at step 25 (derived, --section reach). A controller that
    re-ignites the reactor has 21 s between the two."""
    p = params.replace(Ti_sigma=0.0)
    low = _settled(p, _extinguished(p))
    assert float(low.T) == pytest.approx(324.475, abs=1e-3)
    assert not bool(check_is_terminal(low, p)[0])

    T, tripped = _full_heating_from_extinction(params)
    assert tripped.any()
    trip = int(np.argmax(tripped)) + 1
    ignited = int(np.argmax(T >= 340.0)) + 1
    assert ignited < trip
    assert ignited == pytest.approx(18, abs=1)
    assert trip == pytest.approx(25, abs=1)


# ---------------------------------------------------------------------------
# 8. Error envelope, regime of validity and integrator convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_error_envelope_is_certified(env, params):
    """``CA_error_max`` (0.55 mol/L) rests on a closed form, and this is its
    non-trivial half. While T stays below the trip, C_a cannot fall below the
    steady C_a at 365 K, 0.2636 mol/L (derived): below it the feed brings in
    reactant faster than the reaction at 365 K burns it. So C_a is never more
    than 0.65 - 0.2636 = 0.386 mol/L below its level. The other half, never
    more than 0.55 above it, follows from C_a <= Caf and is not asserted.

    Checked on 512 episodes of random bang-bang coolant, whose drift is
    pinned at +8 K for a third, at -8 K for a third and free for the rest,
    since the process alone rarely reaches 4 standard deviations. They take
    T above 364.9 K without tripping, and the lowest C_a on an untripped step
    is 0.334 (measured)."""
    qV = params.q / params.V
    floor = (
        qV * params.Caf / (qV + params.k0 * np.exp(-params.EA_over_R / params.T_trip))
    )
    assert params.target_CA_range[1] - floor <= params.CA_error_max

    n_seeds = 512
    keys = jax.random.split(jax.random.PRNGKey(8), n_seeds)
    pins = np.full(n_seeds, np.nan)
    pins[: n_seeds // 3] = 8.0
    pins[n_seeds // 3 : 2 * (n_seeds // 3)] = -8.0
    C_a, T, _, tripped = _bang_bang(
        env, params, keys, pins, params.max_steps_in_episode
    )
    kept = ~tripped
    assert tripped.any() and T[kept].max() > params.T_trip - 0.1
    assert C_a[kept].min() >= floor


@pytest.mark.slow
def test_states_stay_physical_on_the_reachable_set(env, params):
    """Wherever random coolant takes the plant with the drift within 3
    standard deviations (64 episodes of random bang-bang, the drift pinned at
    -6 K, at +6 K or in between), every step's proposal is physical. T stays
    above the coldest thing it exchanges heat with, the 290 K jacket (the
    feed is at 344 K or more), 0 < C_a < Caf, the jacket stays inside the
    actuator's range, and every trip is a high-temperature trip: the
    validity guard (T <= 250 K, C_a < 0, or a non-finite state) never fires.
    The lowest T reached is 310.0 K and C_a spans 0.33 to 0.96 mol/L
    (measured)."""
    n_seeds = 64
    keys = jax.random.split(jax.random.PRNGKey(9), n_seeds)
    pins = np.random.default_rng(9).uniform(-6.0, 6.0, n_seeds)
    pins[:11], pins[11:22] = -6.0, 6.0
    C_a, T, T_j, tripped = _bang_bang(
        env, params, keys, pins, params.max_steps_in_episode
    )

    assert np.isfinite(C_a).all() and np.isfinite(T).all() and np.isfinite(T_j).all()
    assert T.min() >= params.T_c_min - 0.1
    assert C_a.min() > 0.0 and C_a.max() < params.Caf
    assert T_j.min() >= params.T_c_min - 1e-3 and T_j.max() <= params.T_c_max + 1e-3
    assert tripped.any()
    assert np.all(T[tripped] >= params.T_trip)
    assert not np.any((T <= params.T_valid_min) | (C_a < 0.0))


@pytest.mark.slow
def test_rk4_1_matches_rk4_16(params):
    """One RK4 step per 3 s env step is converged. Over a whole episode of the
    shipped cascade (seed 0), 16 substeps move T by at most 9.16e-4 K and C_a
    by 2.74e-6 mol/L, under 3 % of the analyser's resolution (derived,
    scripts/unstable_cstr_numbers.py --section integrator). Full heating from
    the extinguished state, the fastest transient the plant has before a
    trip, differs by 7.0e-4 K and trips on the same step (derived, same
    section)."""
    key = jax.random.PRNGKey(0)
    runs = {
        method: jax.jit(
            functools.partial(
                _closed_loop, UnstableCSTR(integration_method=method), params
            )
        )(key)
        for method in ("rk4_1", "rk4_16")
    }
    (C_a1, T1, _, tripped1), (C_a16, T16, _, tripped16) = (
        tuple(np.asarray(x) for x in runs[m]) for m in ("rk4_1", "rk4_16")
    )
    assert not tripped1.any() and not tripped16.any()
    assert np.abs(T1 - T16).max() <= 1e-3
    assert np.abs(C_a1 - C_a16).max() <= 1e-5

    T1, tripped1 = _full_heating_from_extinction(params, "rk4_1")
    T16, tripped16 = _full_heating_from_extinction(params, "rk4_16")
    trip = int(np.argmax(tripped1))
    assert tripped1.any() and int(np.argmax(tripped16)) == trip
    assert np.abs(T1[:trip] - T16[:trip]).max() <= 5e-3


# ---------------------------------------------------------------------------
# 9. JAX transforms
# ---------------------------------------------------------------------------


def test_step_is_jittable_and_vmappable(env, params, step):
    """``step_env`` gives the same step under ``jit``, and reset and step batch
    over seeds under ``vmap`` and run under ``lax.scan``."""
    key = jax.random.PRNGKey(4)
    _, state = env.reset_env(key, params)
    action = jnp.array([0.1])
    obs_e, s_e, r_e, _, _ = env.step_env(key, state, action, params)
    obs_j, s_j, r_j, _, _ = step(key, state, action, params)
    np.testing.assert_allclose(np.asarray(obs_j), np.asarray(obs_e), rtol=1e-6)
    assert float(r_j) == pytest.approx(float(r_e), rel=1e-4, abs=1e-4)
    assert int(s_j.block_clock) == int(s_e.block_clock) == 1

    n = 8
    keys = jax.random.split(key, n)
    obs, states = jax.vmap(lambda k: env.reset_env(k, params))(keys)
    assert obs.shape == (n, 4)
    obs2, states2, rewards, terminated, info = jax.vmap(
        lambda k, s: env.step_env(k, s, action, params)
    )(keys, states)
    assert obs2.shape == (n, 4) and rewards.shape == (n,) and terminated.shape == (n,)
    assert info["tripped"].shape == (n,)
    assert states2.target_levels.shape == (n, N_BLOCKS)
    assert np.all(np.isfinite(np.asarray(rewards)))

    scanned, rewards, _ = _rollout(env, params, state, action, key, 50)
    assert rewards.shape == (50,)
    assert np.all(np.isfinite(np.asarray(rewards)))
    assert int(scanned.time[-1]) == 50


# ---------------------------------------------------------------------------
# 10. Known deviations (PHYSICS.md section 9), each a strict xfail: the test
#     states what a real plant does, and fails the suite if the model starts
#     doing it without PHYSICS.md and this marker being updated.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Deviation D2: no jacket energy balance; the jacket follows the command "
        "through a fixed lag, whatever heat it removes."
    ),
)
def test_jacket_heats_in_a_runaway(params, step):
    """D2. A real jacket warms while the reactor runs away inside it, since
    the coolant can carry off only so much heat. Here the jacket has no
    energy balance: with the command held at its temperature, T_j stays put
    while T climbs from 2 K above T* at the hottest target to within 0.2 K
    of the trip (measured)."""
    p = params.replace(Ti_sigma=0.0)
    state = _settled(p, TARGETS[0], T_offset=2.0)
    action = jnp.array([raw_from_coolant(float(state.T_j), p)])
    T, T_j = [float(state.T)], [float(state.T_j)]
    for _ in range(40):
        _, state, _, _, info = step(jax.random.PRNGKey(0), state, action, p)
        if bool(info["tripped"]):
            break
        T.append(float(state.T))
        T_j.append(float(state.T_j))
    if T[-1] - T[0] < 5.0:
        pytest.fail(f"no runaway to test the jacket on: T {T[0]:.2f} to {T[-1]:.2f} K")
    assert T_j[-1] - T_j[0] > 0.1


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D3: C_a is observed at once; the analyser has no dead time.",
)
def test_analyser_has_dead_time(params, step):
    """D3. An online analyser reports a sample a minute or more after it was
    drawn, so a change in C_a reaches the observation only after that dead
    time (20 steps). Here it arrives at once: two plants 0.01 mol/L apart
    in C_a already give different observations after the first step."""
    p = params.replace(Ti_sigma=0.0)
    a = _settled(p, 0.55)
    b = a.replace(C_a=a.C_a + jnp.float32(0.01))
    action = jnp.array([raw_from_coolant(float(a.T_j), p)])
    key = jax.random.PRNGKey(0)
    seen = []
    for _ in range(19):  # the first 19 observations predate a 1 min dead time
        obs_a, a, _, _, _ = step(key, a, action, p)
        obs_b, b, _, _, _ = step(key, b, action, p)
        seen.append(abs(float(obs_b[0]) - float(obs_a[0])))
    assert max(seen) < 1e-6


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Deviation D5: observations are exact; there is no measurement noise.",
)
def test_observations_are_noisy(params, step):
    """D5. Two readings of one plant differ by the instruments' noise. Here
    the same state stepped with the same action under two keys gives the
    same observation to the last bit: the key only draws the hidden drift's
    next innovation."""
    state = _settled(params, 0.55)
    action = jnp.array([raw_from_coolant(float(state.T_j), params)])
    obs_1 = step(jax.random.PRNGKey(1), state, action, params)[0]
    obs_2 = step(jax.random.PRNGKey(2), state, action, params)[0]
    assert np.any(np.asarray(obs_1) != np.asarray(obs_2))
