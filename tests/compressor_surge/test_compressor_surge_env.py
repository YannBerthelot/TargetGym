"""Environment behaviour of the compressor.

Enforces the task design in ``src/target_gym/compressor_surge/PHYSICS.md``:
the layout, the action mapping and the actuators, the speed lag and its
composition with the drive's rate limit, both schedules and where each state
is scored, the RK4 stages' demand times, the demand deviation, the trip on
the substep minimum and its fresh restart through ``base.failure_kernel``,
both reward versions and the order version 1 gives, where zero action and
full travel take the plant, the error envelope, and integrator convergence.
Runs in float32, as the env ships, except where a test states otherwise.

The env is built directly from its class. Three tests run the pair from
``experts.py`` at ``DEFAULT_GAINS`` (the tuner's starting point): the
observed margin against ``pair_init``, the pair's place in the version-1
order, and a pair episode in float32. The heuristic policies come from
scripts/compressor_surge_numbers.py, which holds their only copy.
"""

import functools
import importlib.util
import math
import pathlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.optimize import brentq

from target_gym import reward as R
from target_gym.compressor_surge.env import (
    KPA,
    N_DEMAND_BLOCKS,
    N_SETPOINT_BLOCKS,
    CompressorSurgeParams,
    CompressorSurgeState,
    characteristic,
    check_is_terminal,
    compute_next_state,
    compute_reward,
    compute_reward_terms,
    compute_velocity,
    consumer_flow,
    equilibrium_phi,
    get_obs,
    live_target,
    recycle_power,
    scheduled_demand,
    sound_speed,
    suction_density,
    surge_constant,
    surge_flow_per_speed,
)
from target_gym.compressor_surge.env_jax import CompressorSurge
from target_gym.compressor_surge.experts import (
    DEFAULT_GAINS,
    pair_init,
    raw_from_commands,
    surge_margin_from_obs,
)


@pytest.fixture(scope="module")
def env():
    return CompressorSurge()


@pytest.fixture(scope="module")
def params():
    return CompressorSurgeParams()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flow_scale(params, N=1.0):
    return suction_density(params) * params.A_c * params.U_r * N


def _head_scale(params, N=1.0):
    return suction_density(params) * (params.U_r * N) ** 2


def _action(params, N, x, dtype=jnp.float32):
    """The raw action that commands speed N and recycle x."""
    return jnp.asarray(raw_from_commands(N, x, params), dtype)


def _state(
    params,
    m_c,
    dp,
    N,
    x,
    u,
    *,
    N_ramp=None,
    dev=0.0,
    setpoints=None,
    levels=None,
    block_clock=0,
    time=0,
    dtype=jnp.float32,
):
    """A state at (m_c, dp, N, x), the drive setpoint at N unless given, a
    constant consumer opening ``u`` unless ``levels`` are given, and 24 kPa
    setpoints unless given. Build it inside ``enable_x64`` for float64."""
    f = functools.partial(jnp.asarray, dtype=dtype)
    return CompressorSurgeState(
        time=jnp.asarray(time, jnp.int32),
        m_c=f(m_c),
        dp=f(dp),
        N=f(N),
        N_ramp=f(N if N_ramp is None else N_ramp),
        x=f(x),
        demand_dev=f(dev),
        phi_min=f(m_c / _flow_scale(params, N)),
        setpoint_levels=f(
            np.full(N_SETPOINT_BLOCKS, 24.0e3) if setpoints is None else setpoints
        ),
        demand_levels=f(np.full(N_DEMAND_BLOCKS, u) if levels is None else levels),
        block_clock=jnp.asarray(block_clock, jnp.int32),
    )


def _equilibrium_state(params, N, x, u, dtype=jnp.float32, **kw):
    """The state on the right-branch equilibrium at (N, x, u)."""
    with enable_x64():
        phi = float(equilibrium_phi(N, x, u, params, 30, xp=np))
    m = _flow_scale(params, N) * phi
    dp = _head_scale(params, N) * float(characteristic(phi, params, np))
    return _state(params, m, dp, N, x, u, dtype=dtype, **kw)


def _rollout(env, params, state, action, key, n):
    """``n`` steps of ``step_env`` under a constant key and action. Returns
    the continuing states (stacked), the rewards and the trip flags."""

    def body(s, _):
        _, s2, r, _, info = env.step_env(key, s, action, params)
        return s2, (s2, r, info["tripped"])

    return jax.lax.scan(body, state, None, length=n)[1]


@functools.lru_cache(maxsize=None)
def _numbers():
    """scripts/compressor_surge_numbers.py, loaded once. The script silences
    every warning when it loads, so it loads inside ``catch_warnings``."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "scripts"
        / "compressor_surge_numbers.py"
    )
    spec = importlib.util.spec_from_file_location("compressor_surge_numbers", path)
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


def _policy(kind, bias=0.0):
    """Test policies on the observation, built by the numbers script's
    ``policy`` and run by its ``_act``: ``"fixed"`` holds the recycle at
    ``bias`` under the pair's speed PI at ``HEURISTIC_SPEED`` (``bias`` 0 is
    the recycle shut); ``"chaser"`` uses the recycle as a pressure actuator
    blind to the margin, x = bias + ``CHASER_GAIN`` (dp - setpoint) in kPa,
    under the same speed PI; ``"zero"`` is raw 0 on both actuators; ``"pair"``
    is the pair at ``DEFAULT_GAINS`` (the tuner's starting point). Each
    returns ``(init(obs), step(memory, obs, params))``, the memory being the
    pair's."""
    numbers = _numbers()
    spec = numbers.policy(
        "constant" if kind == "zero" else kind, bias, gains=DEFAULT_GAINS
    )
    # Read only by the script's random policy, which no test here runs.
    key = jax.random.PRNGKey(0)

    def init(obs):
        return pair_init(obs, xp=jnp)

    def step(ps, obs, params):
        return numbers._act(spec, ps, obs, key, params)

    return init, step


def _episodes(env, params, keys, kind, bias=0.0):
    """One whole episode per key under that constant key, as the suite's
    rollouts run. Controller memory persists through a trip. Returns the
    rewards and trip flags, each (seeds, steps)."""
    init, policy = _policy(kind, bias)

    def episode(key):
        obs, state = env.reset_env(key, params)

        def body(carry, _):
            s, obs, c = carry
            a, c = policy(c, obs, params)
            obs2, s2, r, _, info = env.step_env(key, s, a, params)
            return (s2, obs2, c), (r, info["tripped"])

        return jax.lax.scan(
            body, (state, obs, init(obs)), None, length=params.max_steps_in_episode
        )[1]

    r, tripped = jax.jit(jax.vmap(episode))(keys)
    return np.asarray(r, float), np.asarray(tripped)


# ---------------------------------------------------------------------------
# 1. Layout, action and actuators
# ---------------------------------------------------------------------------


def test_layout(env, params):
    """Ten state fields besides ``time``: seven scalars, four setpoint levels,
    six demand levels (17 floats) and the int block clock. The observation is
    ``[dp, m_c, m_d, N, N_ramp, x, live setpoint]`` with pressures in kPa and
    the speeds and valve in percent; the tracked channel is 0 and its target
    6. Both schedules fill the episode exactly."""
    obs, state = env.reset_env(jax.random.PRNGKey(0), params)
    assert list(CompressorSurgeState.__dataclass_fields__) == [
        "time",
        "m_c",
        "dp",
        "N",
        "N_ramp",
        "x",
        "demand_dev",
        "phi_min",
        "setpoint_levels",
        "demand_levels",
        "block_clock",
    ]
    for name in ("m_c", "dp", "N", "N_ramp", "x", "demand_dev", "phi_min"):
        leaf = getattr(state, name)
        assert leaf.shape == () and leaf.dtype == jnp.float32, name
    assert state.setpoint_levels.shape == (N_SETPOINT_BLOCKS,) == (4,)
    assert state.demand_levels.shape == (N_DEMAND_BLOCKS,) == (6,)
    assert state.setpoint_levels.dtype == state.demand_levels.dtype == jnp.float32
    assert state.block_clock.shape == () and state.block_clock.dtype == jnp.int32
    floats = sum(
        int(np.size(getattr(state, f)))
        for f in CompressorSurgeState.__dataclass_fields__
        if f not in ("time", "block_clock")
    )
    assert floats == 17

    opening = float(state.demand_levels[0] + state.demand_dev)
    expected = [
        float(state.dp) / KPA,
        float(state.m_c),
        float(consumer_flow(state.dp, opening, params)),
        100.0 * float(state.N),
        100.0 * float(state.N_ramp),
        100.0 * float(state.x),
        float(live_target(state, params)) / KPA,
    ]
    np.testing.assert_allclose(np.asarray(obs), expected, rtol=1e-6)
    assert obs.shape == env.obs_shape == env.observation_space(params).shape == (7,)
    assert env.action_space(params).shape == (2,)
    assert env.obs_value_index == 0 and env.obs_target_index == 6
    assert (
        N_SETPOINT_BLOCKS * params.setpoint_block_steps
        == N_DEMAND_BLOCKS * params.demand_block_steps
        == params.max_steps_in_episode
        == 1200
    )
    assert env.integration_method == "rk4_10"
    CompressorSurge(integration_method="rk4_40")
    for bad in ("rk4", "euler_10", "rk4_0", "rk2_10"):
        with pytest.raises(ValueError):
            CompressorSurge(integration_method=bad)


def test_actions_map_to_commands(params):
    """Raw -1, 0 and +1 command 70, 87.5 and 105 % speed and a recycle
    opening of 0, 0.5 and 1; a raw action beyond [-1, 1] is clipped. Each
    command is reached from within one step's reach and shows in the
    observation's percent channels."""
    key = jax.random.PRNGKey(0)
    base = _equilibrium_state(params, 0.9, 0.3, 0.6)
    for raw, N_cmd, x_cmd in (
        (-1.0, 0.70, 0.0),
        (0.0, 0.875, 0.5),
        (1.0, 1.05, 1.0),
        (-3.0, 0.70, 0.0),
        (3.0, 1.05, 1.0),
    ):
        state = base.replace(
            N_ramp=jnp.float32(N_cmd - 0.001 if N_cmd > 0.8 else N_cmd + 0.001),
            x=jnp.float32(x_cmd - 0.02 if x_cmd > 0.5 else x_cmd + 0.005),
        )
        s, _ = compute_next_state(jnp.array([raw, raw]), state, params, key)
        assert float(s.N_ramp) == pytest.approx(N_cmd, abs=1e-6), raw
        assert float(s.x) == pytest.approx(x_cmd, abs=1e-6), raw
        obs = get_obs(s, params)
        assert float(obs[4]) == pytest.approx(100.0 * N_cmd, abs=1e-4)
        assert float(obs[5]) == pytest.approx(100.0 * x_cmd, abs=1e-4)


def test_observed_margin_matches_the_state(env, params):
    """The speed and valve channels are in percent and the state holds
    fractions. Over a pair episode the margin the pair reads from the
    observation, ``surge_margin_from_obs``, equals m_c / (surge flow at
    rated x N) - 1 from the state, and ``pair_init``'s integrators equal the
    state's drive setpoint and valve, to float32 rounding (1e-6 relative)."""
    key = jax.random.PRNGKey(3)
    init, policy = _policy("pair")

    def body(carry, _):
        s, obs, c = carry
        a, c = policy(c, obs, params)
        obs2, s2, _, _, _ = env.step_env(key, s, a, params)
        return (s2, obs2, c), (obs2, s2)

    obs0, state0 = env.reset_env(key, params)
    obs, states = jax.jit(
        lambda s, o: jax.lax.scan(
            body, (s, o, init(o)), None, length=params.max_steps_in_episode
        )[1]
    )(state0, obs0)
    obs = np.asarray(obs, float)
    flow = float(surge_flow_per_speed(params))
    from_state = np.asarray(states.m_c, float) / (flow * np.asarray(states.N, float))
    np.testing.assert_allclose(
        1.0 + surge_margin_from_obs(obs.T, params), from_state, rtol=1e-6
    )
    ps = pair_init(obs.T)
    np.testing.assert_allclose(ps.I_N, np.asarray(states.N_ramp, float), rtol=1e-6)
    np.testing.assert_allclose(ps.I_x, np.asarray(states.x, float), rtol=1e-6)
    # The episode moved both actuators over their working range.
    assert np.ptp(np.asarray(states.x)) > 0.2
    assert np.ptp(np.asarray(states.N)) > 0.05


def test_actuators_are_rate_limited_per_substep(params):
    """In one 0.1 s step the drive setpoint moves at most 0.003 of rated
    (3 %/s), the recycle valve at most 0.05 opening (full open in 2 s) and
    0.01 closing (full close in 10 s). A command within reach is reached
    inside the step. The limit applies per 10 ms substep, so the speed after
    one step of a full drive ramp is the lag's response to a ten-stair
    staircase, not to a jump (float64)."""
    key = jax.random.PRNGKey(0)
    with enable_x64():
        base = _equilibrium_state(params, 0.9, 0.5, 0.6, dtype=jnp.float64)
        for N_cmd, x_cmd, N_end, x_end in (
            (1.05, 1.0, 0.903, 0.55),  # both at their limits, rising
            (0.70, 0.0, 0.897, 0.49),  # falling, the valve closing slower
            (0.9015, 0.52, 0.9015, 0.52),  # within reach: reached
            (0.8985, 0.495, 0.8985, 0.495),
        ):
            action = _action(params, N_cmd, x_cmd, jnp.float64)
            s, _ = compute_next_state(action, base, params, key)
            assert float(s.N_ramp) == pytest.approx(N_end, abs=1e-12), N_cmd
            assert float(s.x) == pytest.approx(x_end, abs=1e-12), x_cmd

        # Staircase: +0.0003 at the start of each substep j = 0..9, then the
        # lag for the rest of the step.
        action = _action(params, 1.05, 0.5, jnp.float64)
        s, _ = compute_next_state(action, base, params, key)
        h, rate = params.delta_t / 10, params.drive_rate
        stairs = sum(
            rate * h * (1.0 - math.exp(-(params.delta_t - j * h) / params.tau_N))
            for j in range(10)
        )
        jump = rate * params.delta_t * (1.0 - math.exp(-params.delta_t / params.tau_N))
        assert float(s.N) - 0.9 == pytest.approx(stairs, rel=1e-6)
        assert abs(stairs - jump) > 0.4 * jump


def test_speed_lags_the_drive(params):
    """(a) With the drive setpoint already at the command, the speed at each
    step boundary is cmd + (N0 - cmd) exp(-t / tau_N) to 1e-5, so 63.2 % of a
    move at 1.0 s. (b) Behind a rate-limited move of duration T_r = move /
    drive_rate, the 63 % time is tau_N + tau_N ln((tau_N / T_r)(exp(T_r /
    tau_N) - 1)) (derived) within 0.01 s: 1.05 s for a 0.003 move and 1.54 s
    for 0.03. The per-substep staircase shifts it by about h / 2 (float64)."""
    key = jax.random.PRNGKey(0)
    tau = params.tau_N
    with enable_x64():
        base = _equilibrium_state(
            params.replace(demand_sigma=0.0), 0.9, 0.5, 0.6, dtype=jnp.float64
        )

        # (a) The lag alone.
        cmd = 0.95
        s0 = base.replace(N_ramp=jnp.float64(cmd))
        action = _action(params, cmd, 0.5, jnp.float64)
        states = _rollout_next(params, s0, action, key, 30)
        t = np.arange(1, 31) * params.delta_t
        np.testing.assert_allclose(
            np.asarray(states.N), cmd + (0.9 - cmd) * np.exp(-t / tau), atol=1e-5
        )
        assert (float(states.N[9]) - 0.9) / (cmd - 0.9) == pytest.approx(
            1.0 - math.exp(-1.0), abs=1e-5
        )

        # (b) Behind the rate limit.
        for move, expected in ((0.003, 1.05), (0.03, 1.54)):
            T_r = move / params.drive_rate
            closed = tau + tau * math.log((tau / T_r) * (math.exp(T_r / tau) - 1.0))
            assert closed == pytest.approx(expected, abs=0.005)
            action = _action(params, 0.9 + move, 0.5, jnp.float64)
            states = _rollout_next(params, base, action, key, 40)
            frac = np.concatenate([[0.0], (np.asarray(states.N) - 0.9) / move])
            t = np.arange(len(frac)) * params.delta_t
            k = int(np.argmax(frac >= 1.0 - math.exp(-1.0)))
            t63 = (
                t[k - 1]
                + (1.0 - math.exp(-1.0) - frac[k - 1])
                / (frac[k] - frac[k - 1])
                * params.delta_t
            )
            assert t63 == pytest.approx(closed, abs=0.01), move


def _rollout_next(params, state, action, key, n):
    """``n`` steps of ``compute_next_state`` (no trip), stacked states."""

    def body(s, _):
        s2 = compute_next_state(action, s, params, key)[0]
        return s2, s2

    return jax.lax.scan(body, state, None, length=n)[1]


# ---------------------------------------------------------------------------
# 2. Schedules
# ---------------------------------------------------------------------------


def test_schedule_draw(env, params):
    """Over 256 seeds every level lies in its range. The live setpoint changes
    only at 30, 60 and 90 s. The scheduled demand is level 0 until 20 s and
    then moves only during the 5 s ramps that start at 20, 40, 60, 80 and
    100 s, linearly, reaching each block's level at the ramp's end."""
    keys = jax.random.split(jax.random.PRNGKey(0), 256)
    _, states = jax.jit(jax.vmap(lambda k: env.reset_env(k, params)))(keys)
    sp = np.asarray(states.setpoint_levels, float)
    dm = np.asarray(states.demand_levels, float)
    assert sp.min() >= 20.0e3 - 1e-2 and sp.max() <= 28.0e3 + 1e-2
    assert dm.min() >= 0.35 - 1e-6 and dm.max() <= 0.95 + 1e-6
    assert sp.min() < 20.2e3 and sp.max() > 27.8e3
    assert np.all(np.asarray(states.block_clock) == 0)

    clocks = jnp.arange(params.max_steps_in_episode)
    targets = jax.vmap(
        lambda s: jax.vmap(lambda c: live_target(s.replace(block_clock=c), params))(
            clocks
        )
    )(states)
    changes = (
        np.flatnonzero(np.any(np.diff(np.asarray(targets), axis=1) != 0, axis=0)) + 1
    )
    assert list(changes) == [300, 600, 900]

    tau = np.linspace(0.0, 125.0, 12501)  # 10 ms grid
    d = np.asarray(
        jax.vmap(lambda L: scheduled_demand(L, jnp.asarray(tau), params))(
            states.demand_levels
        )
    )
    np.testing.assert_allclose(
        d[:, tau < 20.0], dm[:, :1] * np.ones((1, (tau < 20.0).sum())), atol=1e-6
    )
    ramping = np.zeros_like(tau, bool)
    for k in range(1, N_DEMAND_BLOCKS):
        start = 20.0 * k
        window = (tau >= start) & (tau <= start + 5.0)
        ramping |= window
        w = np.clip((tau[window] - start) / 5.0, 0.0, 1.0)
        np.testing.assert_allclose(
            d[:, window],
            dm[:, k - 1 : k] + (dm[:, k : k + 1] - dm[:, k - 1 : k]) * w,
            atol=1e-5,
        )
        at_end = (tau >= start + 5.0) & (tau < start + 20.0)
        np.testing.assert_allclose(
            d[:, at_end], np.repeat(dm[:, k : k + 1], at_end.sum(), 1), atol=1e-5
        )
    moving = np.any(np.abs(np.diff(d, axis=1)) > 1e-7, axis=0)
    assert np.all(ramping[1:][moving] | ramping[:-1][moving])


def test_live_target_follows_the_block_clock(params, step):
    """The live setpoint is ``level[min(clock // 300, 3)]`` and holds past the
    fourth block. The step that enters a block is scored against that
    block's level, which its observation already shows. Both read
    ``setpoint_block_steps`` from the params the step is given."""
    levels = np.array([20.0e3, 22.0e3, 26.0e3, 28.0e3], np.float32)
    quiet = params.replace(demand_sigma=0.0)
    short = quiet.replace(setpoint_block_steps=50, max_steps_in_episode=200)
    state = _equilibrium_state(quiet, 0.9, 0.0, 0.8, setpoints=levels)
    for p in (params, short):
        b = p.setpoint_block_steps
        for clock in (0, b - 1, b, 2 * b, 3 * b - 1, 3 * b, 4 * b - 1, 4 * b, 5000):
            s = state.replace(block_clock=jnp.asarray(clock, jnp.int32))
            assert float(live_target(s, p)) == levels[min(clock // b, 3)], (b, clock)

    for p in (quiet, short):
        before = state.replace(
            block_clock=jnp.asarray(p.setpoint_block_steps - 1, jnp.int32)
        )
        obs, after, reward, _, info = step(
            jax.random.PRNGKey(0), before, _action(p, 0.9, 0.0), p
        )
        assert not bool(info["tripped"])
        assert int(after.block_clock) == p.setpoint_block_steps
        assert float(obs[6]) == pytest.approx(levels[1] / KPA)
        # The recycle is shut, so the running term is 0 and the reward is the
        # tracking cost against the new level.
        assert float(recycle_power(after, p)) == 0.0
        assert float(reward) == pytest.approx(
            -(((levels[1] - float(after.dp)) / KPA / p.e_floor) ** 2), rel=1e-5
        )


def test_demand_uses_stage_times(params):
    """``tau`` rides in the integrated position with dtau/dt = 1, so each RK4
    stage reads the demand ramp at its own time: during a ramp the velocity
    at tau differs from the velocity at the substep's start by exactly the
    consumer flow of the ramp's progress, and one step with 10 substeps
    matches 400 substeps to 1e-3 Pa (float64). A demand frozen at each
    substep's start would make that error first order in h."""
    key = jax.random.PRNGKey(0)
    levels = np.array([0.9, 0.4, 0.4, 0.4, 0.4, 0.4])
    with enable_x64():
        p = params.replace(demand_sigma=0.0)
        state = _equilibrium_state(
            p, 0.95, 0.3, 0.9, levels=levels, block_clock=220, dtype=jnp.float64
        )
        tau0 = 22.0  # 2 s into the ramp from 0.9 to 0.4 that starts at 20 s
        pos = jnp.array([state.m_c, state.dp, state.N, tau0])
        v0, _ = compute_velocity(
            pos, None, state.N_ramp, state.x, 0.0, state.demand_levels, p
        )
        v1, _ = compute_velocity(
            pos.at[3].add(0.005),
            None,
            state.N_ramp,
            state.x,
            0.0,
            state.demand_levels,
            p,
        )
        assert float(v0[3]) == 1.0
        du = (0.4 - 0.9) * 0.005 / 5.0
        a2 = float(sound_speed(p, np)) ** 2
        expected = -(a2 / p.V_p) * float(consumer_flow(state.dp, du, p))
        assert float(v1[1] - v0[1]) == pytest.approx(expected, rel=1e-9)

        action = _action(p, 0.95, 0.3, jnp.float64)
        coarse = compute_next_state(action, state, p, key)[0]
        fine = compute_next_state(action, state, p, key, integration_method="rk4_400")[
            0
        ]
        assert abs(float(coarse.dp) - float(fine.dp)) < 1e-3
        assert abs(float(fine.dp) - float(state.dp)) > 100.0  # the ramp moved it


# ---------------------------------------------------------------------------
# 3. Demand deviation
# ---------------------------------------------------------------------------


def test_demand_deviation_is_a_zero_mean_ou_process(env, params):
    """Under a constant rollout key, as every shipped rollout helper passes,
    the deviation is a zero-mean AR(1) with coefficient exp(-theta dt) =
    0.9802 and stationary sd ``demand_sigma``: no ratchet from a repeated
    innovation. 64 seeds of 1200 steps at zero action, which never trips (a
    trip would redraw the deviation). Mean within 3 standard errors (from the
    per-seed means), RMS within 10 %, pooled lag-1 coefficient within 0.002;
    under the exact law these bands hold with probability 0.997, 0.9999 and
    0.995 (derived: effective samples N (1 - a) / (1 + a), and the AR(1)
    coefficient's standard error sqrt((1 - a^2) / N))."""
    n_seeds, n_steps = 64, params.max_steps_in_episode
    action = jnp.zeros(2)

    def episode(key):
        _, state = env.reset_env(key, params)
        states, _, tripped = _rollout(env, params, state, action, key, n_steps)
        return jnp.concatenate([state.demand_dev[None], states.demand_dev]), tripped

    keys = jax.random.split(jax.random.PRNGKey(2), n_seeds)
    x, tripped = jax.jit(jax.vmap(episode))(keys)
    x = np.asarray(x, float)
    assert not np.asarray(tripped).any()

    means = x.mean(axis=1)
    se = means.std(ddof=1) / np.sqrt(n_seeds)
    assert abs(means.mean()) < 3.0 * se
    assert np.sqrt((x**2).mean()) == pytest.approx(params.demand_sigma, rel=0.10)
    a = math.exp(-params.demand_theta * params.delta_t)
    assert a == pytest.approx(0.980199, abs=1e-6)
    lag1 = (x[:, :-1] * x[:, 1:]).sum() / (x[:, :-1] ** 2).sum()
    assert lag1 == pytest.approx(a, abs=0.002)
    # Innovation sd: sigma sqrt(1 - a^2) = 0.00396 of opening (derived).
    assert params.demand_sigma * math.sqrt(1.0 - a**2) == pytest.approx(
        0.00396, abs=1e-5
    )

    # Without innovations the deviation decays as a^k from its initial value.
    quiet = params.replace(demand_sigma=0.0)
    state = _equilibrium_state(quiet, 0.9, 0.5, 0.6, dev=0.05)
    states, _, _ = _rollout(env, quiet, state, action, keys[0], 100)
    np.testing.assert_allclose(
        np.asarray(states.demand_dev), 0.05 * a ** np.arange(1, 101), rtol=1e-4
    )


# ---------------------------------------------------------------------------
# 4. Trip and restart
# ---------------------------------------------------------------------------


def test_trip_reads_the_substep_minimum(params, step):
    """A swing that dips below the line between two control samples trips.
    From m_c 12.466 kg/s and dp 33.065 kPa at rated speed with the recycle
    shut and an opening whose equilibrium is 12 % right of the line, Phi is
    0.5088 at the start of the step and 0.5085 at its end, and below 0.491 in
    the middle (derived: 0.05 s before the bottom of a Helmholtz swing that
    reaches 0.490, by integrating the env's velocity backwards). The hard
    minimum is read here from ten one-substep calls; the step's soft minimum
    lies at most T ln 10 = 2.3e-3 below it, and at or below it."""
    p = params.replace(demand_sigma=0.0)
    flow, head = _flow_scale(p), _head_scale(p)
    phi_eq = 0.56
    u = (
        flow
        * phi_eq
        / float(consumer_flow(head * float(characteristic(phi_eq, p, np)), 1.0, p, np))
    )
    state = _state(p, 12.466, 33065.4, 1.0, 0.0, u)
    action = _action(p, 1.0, 0.0)
    key = jax.random.PRNGKey(0)

    s1, _ = compute_next_state(action, state, p, key)
    one_substep = p.replace(delta_t=p.delta_t / 10)
    s, phis = state, []
    for _ in range(10):
        s, _ = compute_next_state(
            action, s, one_substep, key, integration_method="rk4_1"
        )
        phis.append(float(s.m_c) / flow)
    hard = min(phis)
    assert float(state.m_c) / flow > 0.5 and float(s1.m_c) / flow > 0.5
    assert phis[-1] == pytest.approx(float(s1.m_c) / flow, abs=1e-6)
    assert hard < 0.491
    gap = hard - float(s1.phi_min)
    assert -1e-6 <= gap <= p.soft_min_temperature * math.log(10) + 1e-6

    _, after, _, terminated, info = step(key, state, action, p)
    assert bool(info["tripped"]) and not bool(terminated)

    # The rule itself: at the line and above it the state continues; below it,
    # or NaN, it trips.
    for phi_min, trips in (
        (0.5, False),
        (0.5001, False),
        (0.4999, True),
        (np.nan, True),
    ):
        s = state.replace(phi_min=jnp.float32(phi_min))
        assert bool(check_is_terminal(s, p)[0]) == trips, phi_min


def test_a_trip_restarts_fresh_on_the_same_clock(env, params, step):
    """Full travel low (70 %, recycle shut) trips. The step is charged
    ``restart_steps x failure_cost`` in version 2 and exactly
    ``-restart_steps`` in version 1, and the window continues from the state
    ``reset_env`` draws from the step's key (new schedule, new deviation,
    block clock 0), with ``time`` running on."""
    key = jax.random.PRNGKey(3)
    v1 = params.replace(reward_version=1)
    _, state = env.reset_env(key, params)
    low = jnp.array([-1.0, -1.0])
    for n in range(1, 400):
        _, new, r2, terminated, info = step(key, state, low, params)
        _, _, r1, _, _ = step(key, state, low, v1)
        if bool(info["tripped"]):
            break
        state = new
    else:
        pytest.fail("full travel low did not trip within 400 steps")

    assert not bool(terminated)
    # Equal to float32 rounding: the step's copy of the reset is compiled
    # inside the step, where XLA may fuse its arithmetic differently.
    fresh = env.reset_env(key, params)[1].replace(time=new.time)
    for name in CompressorSurgeState.__dataclass_fields__:
        np.testing.assert_allclose(
            np.asarray(getattr(new, name)),
            np.asarray(getattr(fresh, name)),
            rtol=1e-6,
            err_msg=name,
        )
    assert int(new.time) == n
    assert int(new.block_clock) == 0
    assert float(r2) == pytest.approx(
        -params.restart_steps * params.failure_cost, rel=1e-6
    )
    assert float(r1) == -9000.0


def test_restart_and_failure_cost_are_derived(params):
    """``dp_error_max`` is the 28 kPa top level against the lowest pressure
    an untripped plant holds, 7.32 kPa at 70 % with the recycle and the
    consumers fully open (derived); the other side, the 105 % peak 35.66 kPa
    against 20 kPa, is smaller. ``failure_cost`` is twice the largest
    tracking cost, 2 (20.68 / 0.0275)^2 = 1.131e6, and a trip costs 9000 of
    them, 1.018e10."""
    with enable_x64():
        phi = float(equilibrium_phi(params.N_min, 1.0, 1.0, params, 30, xp=np))
    dp_low = (
        _head_scale(params, params.N_min) * float(characteristic(phi, params, np)) / KPA
    )
    assert dp_low == pytest.approx(7.32, abs=0.005)
    peak = _head_scale(params, params.N_max) * (params.psi_c0 + 2 * params.H) / KPA
    assert peak == pytest.approx(35.66, abs=0.005)
    assert params.dp_error_max == pytest.approx(
        params.p_ref_range[1] / KPA - dp_low, abs=0.005
    )
    assert peak - params.p_ref_range[0] / KPA < params.dp_error_max

    assert params.failure_cost == 2.0 * (params.dp_error_max / params.e_floor) ** 2
    assert params.failure_cost == pytest.approx(1.131e6, rel=1e-4)
    assert params.restart_steps == 9000
    assert R.trip_cost(params) == pytest.approx(1.0179e10, rel=1e-4)


@pytest.mark.slow
def test_the_error_envelope_is_certified(env, params):
    """``dp_error_max`` (20.68 kPa) rests on the steady closed form; this is
    its dynamic check. 512 episodes of random bang-bang on both actuators,
    each switching at a rate drawn per episode between 2 and 30 % per step,
    with the deviation pinned at +4 sd (+0.08) for a third, at -4 sd for a
    third and free for the rest: on every untripped step |setpoint - dp| is
    at most 20.68 kPa, the top level less the lowest pressure an untripped
    plant holds (7.32 kPa, derived, scripts/compressor_surge_numbers.py
    --section params)."""
    n_seeds = 512
    keys = jax.random.split(jax.random.PRNGKey(8), n_seeds)
    pins = np.full(n_seeds, np.nan)
    pins[: n_seeds // 3] = 0.08
    pins[n_seeds // 3 : 2 * (n_seeds // 3)] = -0.08

    def episode(key, pin):
        _, state = env.reset_env(key, params)
        flip_key, rate_key, sign_key = jax.random.split(jax.random.fold_in(key, 1), 3)
        rate = jax.random.uniform(rate_key, (2,), minval=0.02, maxval=0.3)
        flips = jax.random.bernoulli(flip_key, rate, (params.max_steps_in_episode, 2))
        first = jnp.where(jax.random.bernoulli(sign_key, 0.5, (2,)), 1.0, -1.0)
        actions = first * jnp.where(jnp.cumsum(flips, axis=0) % 2 == 0, 1.0, -1.0)

        def body(s, a):
            s = s.replace(demand_dev=jnp.where(jnp.isnan(pin), s.demand_dev, pin))
            _, s2, _, _, info = env.step_env(key, s, a, params)
            return s2, (live_target(s2, params) - s2.dp, info["tripped"])

        return jax.lax.scan(body, state, actions)[1]

    err, tripped = jax.jit(jax.vmap(episode))(keys, jnp.asarray(pins, jnp.float32))
    err, tripped = np.asarray(err) / KPA, np.asarray(tripped)
    assert tripped.any()
    assert np.abs(err[~tripped]).max() <= params.dp_error_max


# ---------------------------------------------------------------------------
# 5. Rewards
# ---------------------------------------------------------------------------


def _exact_state(params, dp, x=0.0, phi_min=0.6, target=24.0e3):
    """A float64 NumPy state, for scoring the reward exactly with ``xp=np``."""
    return CompressorSurgeState(
        time=0,
        m_c=15.0,
        dp=dp,
        N=1.0,
        N_ramp=1.0,
        x=x,
        demand_dev=0.0,
        phi_min=phi_min,
        setpoint_levels=np.full(N_SETPOINT_BLOCKS, target),
        demand_levels=np.full(N_DEMAND_BLOCKS, 0.6),
        block_clock=0,
    )


def test_v2_reward_units(params):
    """Version 2 is minus the floor-normalised squared error plus the
    recycle power above ``c_hold``: tracking is 0 at the setpoint and 1 one
    ``e_floor`` (0.0275 kPa) away; running is 0 at or below ``c_hold`` and 1 at
    twice it. A tripped state pays the trip cost alone."""
    dp = 24.0e3
    per_x = float(recycle_power(_exact_state(params, dp, x=1.0), params, np))
    x_hold = params.c_hold / per_x

    terms = compute_reward_terms(_exact_state(params, dp), params, np)
    assert set(terms) == {"tracking", "running", "failure"}
    assert float(compute_reward(_exact_state(params, dp), params, np)) == 0.0
    for sign in (-1.0, 1.0):
        s = _exact_state(params, dp + sign * params.e_floor * KPA)
        assert float(compute_reward_terms(s, params, np)["tracking"]) == pytest.approx(
            1.0, rel=1e-9
        )
    for scale, running in ((0.5, 0.0), (1.0, 0.0), (2.0, 1.0), (3.0, 2.0)):
        s = _exact_state(params, dp, x=scale * x_hold)
        assert float(recycle_power(s, params, np)) == pytest.approx(
            scale * params.c_hold, rel=1e-12
        )
        assert float(compute_reward_terms(s, params, np)["running"]) == pytest.approx(
            running, abs=1e-9
        )
        assert float(compute_reward(s, params, np)) == pytest.approx(-running, abs=1e-9)

    tripped = _exact_state(params, 20.0e3, x=1.0, phi_min=0.49)
    terms = compute_reward_terms(tripped, params, np)
    assert float(terms["tracking"]) == 0.0 and float(terms["running"]) == 0.0
    assert float(terms["failure"]) == R.trip_cost(params)
    assert float(compute_reward(tripped, params, np)) == -R.trip_cost(params)


def test_v1_is_log_scaled_and_charges_downtime(params):
    """Version 1 is 1 at zero error and 0 at ``dp_error_max``, pays an equal
    increment for each halving of the error above the floor, ignores the
    recycle, and is exactly ``-restart_steps`` on a tripped step."""
    v1 = params.replace(reward_version=1)

    def score(error_kpa, **kw):
        return float(
            compute_reward(_exact_state(v1, 24.0e3 - error_kpa * KPA, **kw), v1, np)
        )

    assert score(0.0) == 1.0
    assert score(0.0, x=1.0) == 1.0
    assert score(v1.dp_error_max) == pytest.approx(0.0, abs=1e-12)
    errors = [16.0, 8.0, 4.0, 2.0]
    gains = np.diff([score(e) for e in errors])
    assert np.all(gains > 0.0)
    np.testing.assert_allclose(gains, gains[0], rtol=0.02)
    assert all(0.0 <= score(e) <= 1.0 for e in errors)

    assert score(0.0, phi_min=0.49) == -1.0 * v1.restart_steps == -9000.0
    assert score(5.0, phi_min=0.49) == -9000.0


@pytest.mark.slow
def test_v1_ranks_tripping_below_safe(env, params):
    """The exploit version 1 is fixed against, on 32 seeds, with policies that
    trip in most episodes: the recycle shut under a pressure PI on speed, a
    fixed recycle of 0.10, and the margin-blind chaser at bias 0.2 (the
    numbers script's policies, ``_policy``; PHYSICS.md, section 8). Under
    the constant rollout key each restart replays the episode's own first
    block, so a policy that trips early trips again and again. With 0 on the
    tripped step the chaser outscores a fixed recycle of 0.30, which never
    trips. With ``-restart_steps`` on the tripped step all three rank below
    the fixed 0.30, zero action and the pair at ``DEFAULT_GAINS`` (the
    tuner's starting point), which never trips and scores above the fixed
    0.30."""
    v1 = params.replace(reward_version=1)
    n = v1.max_steps_in_episode
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(32))

    def scores(kind, bias=0.0):
        r, tripped = _episodes(env, v1, keys, kind, bias)
        return r.sum(axis=1) / n, tripped.sum(axis=1)

    shut, shut_trips = scores("fixed", 0.0)
    tenth, tenth_trips = scores("fixed", 0.10)
    chaser, chaser_trips = scores("chaser", 0.2)
    safe, safe_trips = scores("fixed", 0.30)
    zero, zero_trips = scores("zero")
    pair, pair_trips = scores("pair")

    assert not safe_trips.any() and not zero_trips.any() and not pair_trips.any()
    for trips in (shut_trips, tenth_trips, chaser_trips):
        assert trips.sum() >= 32
    unfixed = chaser + v1.restart_steps * chaser_trips / n
    assert unfixed.mean() > safe.mean()
    for tripping in (shut, tenth, chaser):
        assert tripping.mean() < zero.mean() < safe.mean() < pair.mean()


# ---------------------------------------------------------------------------
# 6. Where zero action and full travel take the plant
# ---------------------------------------------------------------------------


def _phi(states, params):
    return np.asarray(states.m_c) / _flow_scale(params, np.asarray(states.N))


def _travel(env, params, action, seeds, n=None):
    """Whole episodes from ``reset_env`` at a constant raw action, one per
    seed. Returns the continuing states and the trip flags."""
    n = params.max_steps_in_episode if n is None else n

    def episode(key):
        _, state = env.reset_env(key, params)
        states, _, tripped = _rollout(env, params, state, action, key, n)
        return states, tripped

    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds))
    states, tripped = jax.jit(jax.vmap(episode))(keys)
    return states, np.asarray(tripped)


def test_zero_action_never_trips(env, params):
    """Raw 0 (87.5 % speed, recycle half open) over 20 schedules of 1200
    steps: no trip, and the smallest phi_min is at least 0.55. Zero action's
    lowest Phi is the reset's (derived: 0.581 at the lowest drawable first
    opening, 0.29, and 0.568 for a drop from there to the clip floor 0.2);
    its own equilibrium at opening 0.2 is 0.632. Measured here: 0.610
    (scripts/compressor_surge_numbers.py --section reach prints all four)."""
    states, tripped = _travel(env, params, jnp.zeros(2), range(20))
    assert not tripped.any()
    assert float(np.min(states.phi_min)) >= 0.55
    # Why: the recycle alone carries the surge flow at any pressure from an
    # opening of 1 / (k_r sqrt(K)) = 0.454 (derived), under zero action's 0.5.
    x_alone = 1.0 / (params.k_r * math.sqrt(float(surge_constant(params))))
    assert x_alone == pytest.approx(0.454, abs=1e-3)


@pytest.mark.slow
def test_full_travel_low_trips(env, params):
    """Conformance check 8's fact: full travel low (70 %, recycle shut) trips
    the plant from the PRNGKey(0) reset at every demand level from 0.20 to
    0.95 within 400 steps, sooner at lower demand (3 steps at 0.20, 117 at
    0.95); at full opening it never does. Over 20 drawn schedules it trips in
    every one, first between 14 and 153 steps (measured, both by
    scripts/compressor_surge_numbers.py --section reach)."""
    low = jnp.array([-1.0, -1.0])
    first = []
    for level in (0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95):
        p = params.replace(demand_range=(level, level))
        _, tripped = _travel(env, p, low, [0], 400)
        assert tripped.any(), level
        first.append(int(np.argmax(tripped[0])) + 1)
    assert first == sorted(first)
    assert first[0] <= 5 and first[-1] <= 400
    _, tripped = _travel(env, params.replace(demand_range=(1.0, 1.0)), low, [0], 400)
    assert not tripped.any()

    _, tripped = _travel(env, params, low, range(20), 400)
    assert tripped.any(axis=1).all()


@pytest.mark.slow
def test_full_travel_high_never_trips(env, params):
    """Full travel high (105 %, recycle open) over 20 schedules of 1200
    steps: no trip, and Phi stays at or below 0.777, under the join of the
    cubic and its tangent line (0.78)."""
    states, tripped = _travel(env, params, jnp.array([1.0, 1.0]), range(20))
    assert not tripped.any()
    phi = _phi(states, params)
    assert phi.max() <= 0.7775
    assert phi.max() < params.phi_join


@pytest.mark.slow
def test_setpoint_chasing_without_the_margin_trips(env, params):
    """The hardness claim on the env, 64 seeds: a pressure PI on speed with
    the recycle shut trips in every episode, and the margin-blind chaser at
    bias 0.3, which uses the recycle for pressure and ignores the margin,
    trips in at least one. On 256 other seeds (1000 to 1255) the two trip in
    100 % and 11.3 % of episodes (measured, scripts/compressor_surge_numbers.py
    --hardness)."""
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(64))
    _, shut = _episodes(env, params, keys, "fixed", 0.0)
    _, chaser = _episodes(env, params, keys, "chaser", 0.3)
    assert shut.any(axis=1).all()
    assert chaser.any()


# ---------------------------------------------------------------------------
# 7. Integrator convergence
# ---------------------------------------------------------------------------


def test_integrator_is_converged(params):
    """``rk4_10`` against ``CompressorSurge(integration_method="rk4_40")``
    over a 5 s demand ramp that ends 2 % right of the line: at 24 kPa and
    fixed speed with the recycle shut, the opening falls from 0.9 to the one
    whose equilibrium is at Phi 0.51, starting 1 s into the run, and the run
    lasts 8 s (speed 88.7 %, final opening 0.744). The header pressure
    agrees to 0.1 % and the smallest Phi at the step boundaries to 0.2 %
    (float32, as shipped; measured 4.6e-7 and 1.2e-7, float32 rounding, by
    scripts/compressor_surge_numbers.py --section integrator). The soft
    minimum is not compared: its offset grows with the number of
    substeps."""
    p = params.replace(demand_sigma=0.0)
    rho = float(suction_density(p))
    dp0 = 24.0e3
    m0 = float(consumer_flow(dp0, 0.9, p, np))
    # The speed that puts the compressor at (m0, dp0) on the right branch.
    c = rho * p.A_c**2 * dp0 / m0**2
    phi0 = brentq(lambda f: float(characteristic(f, p, np)) / f**2 - c, 0.5, 0.8316)
    N = m0 / (_flow_scale(p) * phi0)
    phi_end = 1.02 * 2.0 * p.W
    dp_end = _head_scale(p, N) * float(characteristic(phi_end, p, np))
    u_end = _flow_scale(p, N) * phi_end / float(consumer_flow(dp_end, 1.0, p, np))
    levels = np.array([0.9, u_end, u_end, u_end, u_end, u_end])
    state = _state(p, m0, dp0, N, 0.0, 0.9, levels=levels, block_clock=190)
    action = _action(p, N, 0.0)
    key = jax.random.PRNGKey(0)

    runs = {}
    for method in ("rk4_10", "rk4_40"):
        env = CompressorSurge(integration_method=method)
        states, _, tripped = _rollout(env, p, state, action, key, 80)
        assert not np.asarray(tripped).any(), method
        runs[method] = states
    a, b = runs["rk4_10"], runs["rk4_40"]
    assert np.abs(np.asarray(a.dp) - np.asarray(b.dp)).max() <= 1e-3 * float(
        np.max(b.dp)
    )
    phi_a, phi_b = _phi(a, p).min(), _phi(b, p).min()
    assert phi_a == pytest.approx(phi_b, rel=2e-3)
    assert phi_b < 0.52  # the run does come within 4 % of the line


def test_float32_episode_is_finite(env, params):
    """A whole pair episode in float32, as the env ships: every state leaf,
    observation and reward is finite, and the pair does not trip."""
    key = jax.random.PRNGKey(11)
    init, policy = _policy("pair")

    def body(carry, _):
        s, obs, c = carry
        a, c = policy(c, obs, params)
        obs2, s2, r, _, info = env.step_env(key, s, a, params)
        return (s2, obs2, c), (s2, obs2, r, info["tripped"])

    obs0, state0 = env.reset_env(key, params)
    states, obs, rewards, tripped = jax.jit(
        lambda s, o: jax.lax.scan(
            body, (s, o, init(o)), None, length=params.max_steps_in_episode
        )[1]
    )(state0, obs0)
    for name, leaf in states.__dict__.items():
        assert np.all(np.isfinite(np.asarray(leaf))), name
    assert np.asarray(obs).dtype == np.float32
    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.isfinite(np.asarray(rewards)))
    assert not np.asarray(tripped).any()
