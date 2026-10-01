"""The compressor's controllers.

Enforces ``src/target_gym/compressor_surge/PHYSICS.md``, section 7: the
pair's gains and where they are read from, the guard its factory applies, the
override, the anti-windup of both loops, the reset, and the agreement between
the functional core that scripts/compressor_surge_numbers.py runs and the
stateful controller the suite runs. Then the NMPC: its model against the
env's step, its one-step prediction against the plant, its fallback to the
pair, its routing and its guess for cold solves, and its preview of both
schedules. Last, the pair on the first 1024 seeds of its validation, and the
floor that scripts/measure_hold.py records from the NMPC's hold.

The env is built directly from its class, and the planner's params are the
env's with ``demand_sigma`` zeroed, which is what ``plan_params`` gives for
the registered spec, whose ``noise_fields`` is ``("demand_sigma",)``.
"""

import copy
import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

import target_gym.experts.pid as pid_mod
from target_gym.compressor_surge.env import (
    KPA,
    CompressorSurgeParams,
    CompressorSurgeState,
    compute_next_state,
    demand_opening,
    get_obs,
    live_target,
    suction_density,
    surge_flow_per_speed,
)
from target_gym.compressor_surge.env_jax import CompressorSurge
from target_gym.compressor_surge.experts import (
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
    HELD_GAINS,
    MPC_HORIZON,
    MPC_STATES,
    MPC_SUBSTEPS,
    MPC_TERMINAL_WEIGHT,
    SURGE_CONTROL_LINE,
    TUNED_GAINS,
    CompressorSurgeMPC,
    CompressorSurgePair,
    PairState,
    UnsafePairGains,
    _state_at,
    battery_report,
    check_pair_gains,
    closed_loop_decay,
    commands_from_raw,
    load_gains,
    make_compressor_surge_mpc,
    make_compressor_surge_pid,
    pair_init,
    pair_step,
    raw_from_commands,
    settled_point,
    soft_min,
    stage_openings,
    surge_margin_from_obs,
)

#: Gain sets the guard must refuse, each with the other loop at
#: ``DEFAULT_GAINS``: an anti-surge loop whose linearised slow mode decays at
#: 0.083 /s, under ``GUARD_MIN_DECAY``; a weak one; one that chatters; a
#: proportional gain at which the valve settles into a limit cycle against
#: its closing limit, which only the travel check sees; a gentler loop that
#: decays fast enough but lets the battery too close to the line, which only
#: the margin check sees; and an aggressive speed loop.
REFUSED = {
    "anti-surge (3, 0.3)": {"kp_surge": 3.0, "ki_surge": 0.3},
    "anti-surge (0.5, 0.05)": {"kp_surge": 0.5, "ki_surge": 0.05},
    "anti-surge (12, 1.2)": {"kp_surge": 12.0, "ki_surge": 1.2},
    "anti-surge (3, 3)": {"kp_surge": 3.0, "ki_surge": 3.0},
    "anti-surge (1, 1)": {"kp_surge": 1.0, "ki_surge": 1.0},
    "speed (0.3, 0.3)": {"kp_speed": 0.3, "ki_speed": 0.3},
}
#: Where scripts/measure_hold.py and scripts/evaluate_baselines.py record.
DATA = pathlib.Path(pid_mod.__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def env():
    return CompressorSurge()


@pytest.fixture(scope="module")
def params():
    return CompressorSurgeParams()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


@pytest.fixture(scope="module")
def planned(params):
    """The params the suite will build the NMPC with: ``plan_params`` zeroes
    the spec's ``noise_fields``, ``("demand_sigma",)``."""
    return params.replace(demand_sigma=0.0)


@pytest.fixture(scope="module")
def mpc(env, planned):
    """The NMPC at its shipped horizon. Tests that step it build their own."""
    return make_compressor_surge_mpc(env, planned)


def _obs(params, dp_kpa, margin, N=1.0, x=0.3, setpoint_kpa=24.0, N_ramp=None):
    """An observation at header pressure ``dp_kpa``, flow-coefficient margin
    ``margin`` at speed ``N`` (fraction of rated), recycle ``x`` and setpoint
    ``setpoint_kpa``. The delivered-flow channel is not read by the pair."""
    m_c = surge_flow_per_speed(params) * N * (1.0 + margin)
    N_ramp = N if N_ramp is None else N_ramp
    return np.array(
        [dp_kpa, m_c, 10.0, 100.0 * N, 100.0 * N_ramp, 100.0 * x, setpoint_kpa]
    )


def _run_class(pair, env, params, step, key, n):
    """``n`` steps of the stateful pair on the env from ``reset_env(key)``
    under that constant key. Returns the observations it saw and its
    actions."""
    obs, state = env.reset_env(key, params)
    seen, actions = [], []
    for _ in range(n):
        seen.append(np.asarray(obs, float))
        action = pair(obs)
        actions.append(action)
        obs, state, _, _, _ = step(key, state, jnp.asarray(action), params)
    return np.array(seen), np.array(actions)


# ---------------------------------------------------------------------------
# Gains, where they come from, and the guard
# ---------------------------------------------------------------------------


def test_default_gains_are_finite_and_tuned_ones_nonzero():
    """The coordinate search multiplies gains, so a tuned gain that starts
    at zero never moves. The tuned and held names split the gains, and the
    control line rendering draws is the pair's."""
    assert all(np.isfinite(v) for v in DEFAULT_GAINS.values())
    assert all(DEFAULT_GAINS[k] != 0.0 for k in TUNED_GAINS)
    assert set(TUNED_GAINS) | set(HELD_GAINS) == set(DEFAULT_GAINS)
    assert not set(TUNED_GAINS) & set(HELD_GAINS)
    assert SURGE_CONTROL_LINE == DEFAULT_GAINS["surge_line"] == 0.15
    assert DEFAULT_GAINS["override_line"] == 0.5 * DEFAULT_GAINS["surge_line"]


def test_gains_are_read_under_the_task_key(monkeypatch, params):
    """The factory reads ``pid._gains_cache["compressor_surge"]`` at call
    time (the tuner patches it), skips the ``note``, and nothing else in
    ``pid_gains.json`` starts with the task's name, since
    ``baseline_fingerprint`` collects gains by that prefix."""
    stored = pid_mod._load_gains()
    # 1 kPa under the setpoint on the control line at 90 % speed: the speed
    # loop acts through kp_speed.
    obs = _obs(params, 23.0, 0.15, N=0.9)
    shipped = make_compressor_surge_pid()

    patched = {**load_gains(), "kp_speed": 0.09, "note": "patched by the test"}
    monkeypatch.setattr(
        pid_mod, "_gains_cache", {**stored, "compressor_surge": patched}
    )
    built = make_compressor_surge_pid()
    assert built.gains["kp_speed"] == 0.09
    assert "note" not in built.gains
    assert built.step(obs)[0] != shipped.step(obs)[0]

    monkeypatch.undo()
    assert make_compressor_surge_pid().gains == shipped.gains

    on_disk = json.loads(pid_mod._GAINS_FILE.read_text())
    ours = {k for k in on_disk if k.startswith("compressor_surge")}
    assert ours <= {"compressor_surge"}
    for key in ours:
        numeric = {k for k, v in on_disk[key].items() if not isinstance(v, str)}
        assert numeric <= set(DEFAULT_GAINS), numeric - set(DEFAULT_GAINS)


def test_shipped_gains_pass_the_guard(params):
    """The factory must build the shipped pair: its closed loop decays at
    ``GUARD_MIN_DECAY`` or faster at the five control-line points, and every
    battery run never trips, keeps at least ``GUARD_MIN_MARGIN`` and ends
    with its valve command travelling at most ``GUARD_MAX_TRAVEL`` over the
    last ``GUARD_TRAVEL_STEPS`` steps. The guard's points are fixed points of
    the closed loop, so its Jacobians are taken where the loop rests."""
    gains = load_gains()
    decay = closed_loop_decay(gains)
    margins, tripped, travel = battery_report(gains)
    assert np.all(decay >= GUARD_MIN_DECAY), decay
    assert np.all(margins >= GUARD_MIN_MARGIN), margins
    assert not tripped.any()
    assert np.all(travel <= GUARD_MAX_TRAVEL), travel
    # The travel window is each run's last 10 s, 30 to 40 s after the move.
    dt = params.delta_t
    assert GUARD_TRAVEL_STEPS * dt == pytest.approx(10.0)
    assert (BATTERY_STEPS - BATTERY_LEAD - GUARD_TRAVEL_STEPS) * dt == pytest.approx(
        30.0
    )
    check_pair_gains(gains)
    assert isinstance(make_compressor_surge_pid(), CompressorSurgePair)

    # The settled points: on the control line where the valve is open,
    # further right with the valve shut, and the pair's commands hold them.
    for p_ref, u in GUARD_POINTS + tuple((p0, u0) for p0, _, u0, _ in BATTERY):
        z = settled_point(p_ref, u, gains, params)
        state_obs = np.array(
            [z[1] / KPA, z[0], 10.0, 100 * z[2], 100 * z[3], 100 * z[4], p_ref / KPA]
        )
        margin = surge_margin_from_obs(state_obs, params)
        if z[4] > 0.0:
            assert margin == pytest.approx(gains["surge_line"], abs=1e-9)
        else:
            assert margin > gains["surge_line"]
        u_raw, _ = pair_step(gains, pair_init(state_obs), state_obs, params)
        N_cmd, x_cmd = commands_from_raw(u_raw, params)
        assert N_cmd == pytest.approx(z[2], abs=1e-12)
        assert x_cmd == pytest.approx(z[4], abs=1e-12)


def test_guard_refuses_known_bad_gains(monkeypatch):
    """The six gain sets of ``REFUSED`` are refused, and so is a refused
    entry in ``pid_gains.json``. The anti-surge loop at (3, 0.3) keeps a mode
    decaying at 0.083 /s, under ``GUARD_MIN_DECAY``; the weak loop
    (0.5, 0.05) decays at 0.045 /s and lets the battery within 0.0618 of the
    line; the chattering (12, 1.2) and the aggressive speed loop grow at the
    control line. At (3, 3) the linearised loop decays as fast as
    ``DEFAULT_GAINS``, but after the battery's demand drops the valve command
    swings faster than the valve closes and the valve ratchets in a limit
    cycle, which only the travel check refuses. At (1, 1) the loop decays as
    fast too, and the battery comes within 0.0838 of the line, which only the
    margin check refuses. ``DEFAULT_GAINS`` decay at 0.451 to 0.461 /s, and
    ``GUARD_MIN_MARGIN`` is frozen from their battery minimum, 0.1133, less
    0.015 and rounded down to 0.001 (derived,
    scripts/compressor_surge_numbers.py --section guard)."""
    np.testing.assert_allclose(
        closed_loop_decay(DEFAULT_GAINS),
        [0.451, 0.456, 0.460, 0.457, 0.461],
        atol=2e-3,
    )
    margins, _, travel = battery_report(DEFAULT_GAINS)
    assert margins.min() == pytest.approx(0.1133, abs=1e-4)
    assert GUARD_MIN_MARGIN == max(
        math.floor((margins.min() - 0.015) * 1000) / 1000, 0.075
    )
    assert travel.max() <= 0.1 * GUARD_MAX_TRAVEL, travel
    slow = closed_loop_decay({**DEFAULT_GAINS, **REFUSED["anti-surge (3, 0.3)"]})
    np.testing.assert_allclose(slow, 0.083, atol=2e-3)
    assert np.all(slow < GUARD_MIN_DECAY)
    weak = {**DEFAULT_GAINS, **REFUSED["anti-surge (0.5, 0.05)"]}
    assert np.all(closed_loop_decay(weak) < GUARD_MIN_DECAY)
    assert battery_report(weak)[0].min() < GUARD_MIN_MARGIN
    for name in ("anti-surge (12, 1.2)", "speed (0.3, 0.3)"):
        assert closed_loop_decay({**DEFAULT_GAINS, **REFUSED[name]}).min() < 0.0
    ringing = {**DEFAULT_GAINS, **REFUSED["anti-surge (3, 3)"]}
    assert np.all(closed_loop_decay(ringing) >= GUARD_MIN_DECAY)
    assert battery_report(ringing)[2].max() > 10.0 * GUARD_MAX_TRAVEL
    with pytest.raises(UnsafePairGains, match="limit cycle"):
        check_pair_gains(ringing)
    gentle = {**DEFAULT_GAINS, **REFUSED["anti-surge (1, 1)"]}
    assert np.all(closed_loop_decay(gentle) >= GUARD_MIN_DECAY)
    margins, tripped, travel = battery_report(gentle)
    assert margins.min() == pytest.approx(0.0838, abs=1e-4)
    assert not tripped.any() and np.all(travel <= GUARD_MAX_TRAVEL)
    with pytest.raises(UnsafePairGains, match="surge line"):
        check_pair_gains(gentle)

    for name, change in REFUSED.items():
        with pytest.raises(UnsafePairGains):
            check_pair_gains({**DEFAULT_GAINS, **change})

    monkeypatch.setattr(
        pid_mod,
        "_gains_cache",
        {
            **pid_mod._load_gains(),
            "compressor_surge": {**DEFAULT_GAINS, **REFUSED["anti-surge (3, 0.3)"]},
        },
    )
    with pytest.raises(UnsafePairGains):
        make_compressor_surge_pid()


# ---------------------------------------------------------------------------
# The override, anti-windup, reset and the functional core
# ---------------------------------------------------------------------------


def test_override_opens_fully_and_holds(params):
    """A margin below the 7.5 % override line commands full opening, held
    for 2 s (20 steps, the triggering one included) after the margin
    recovers; then the anti-surge PI resumes. A second dip during the hold
    restarts it. Just above the override line the PI acts."""
    gains = DEFAULT_GAINS
    hold_steps = round(gains["override_hold"] / params.delta_t)
    assert hold_steps == 20

    def valve_commands(margins):
        ps = pair_init(_obs(params, 24.0, margins[0]))
        out = []
        for m in margins:
            obs = _obs(params, 24.0, m)
            pi_only = pair_step(gains, ps._replace(timer=0.0), obs, params)[0]
            u_raw, ps = pair_step(gains, ps, obs, params)
            out.append(
                (
                    commands_from_raw(u_raw, params)[1],
                    commands_from_raw(pi_only, params)[1],
                )
            )
        return np.array(out)

    cmds = valve_commands([0.05] + [0.15] * 30)
    assert np.all(cmds[:hold_steps, 0] == 1.0)
    np.testing.assert_array_equal(cmds[hold_steps:, 0], cmds[hold_steps:, 1])
    assert np.all(cmds[hold_steps:, 0] < 1.0)

    cmds = valve_commands([0.05] + [0.15] * 10 + [0.07] + [0.15] * 30)
    assert np.all(cmds[: 11 + hold_steps, 0] == 1.0)
    assert cmds[11 + hold_steps, 0] < 1.0

    cmds = valve_commands([gains["override_line"] + 1e-6] * 3)
    np.testing.assert_array_equal(cmds[:, 0], cmds[:, 1])
    assert np.all(cmds[:, 0] < 1.0)


def test_integrators_do_not_wind_up(params):
    """External reset: while the valve lags its command (the observed
    valve stuck at 0.20 with the margin 5 points under the line), the
    anti-surge integral is clipped to within ``reset_band`` of the observed
    valve before each use, so the command settles at 0.20 + 0.05 + kp_surge
    es instead of growing. Conditional integration: neither integral moves
    while its command is saturated in the direction of its error."""
    gains = DEFAULT_GAINS
    band, dt = gains["reset_band"], params.delta_t
    es = 0.05
    obs = _obs(params, 24.0, gains["surge_line"] - es, x=0.20)
    ps = pair_init(obs)
    for _ in range(200):
        u_raw, new = pair_step(gains, ps, obs, params)
        used = np.clip(ps.I_x, 0.20 - band, 0.20 + band)
        assert abs(used - 0.20) <= band + 1e-12
        assert float(new.I_x) == pytest.approx(used + gains["ki_surge"] * es * dt)
        ps = new
    x_cmd = commands_from_raw(u_raw, params)[1]
    assert x_cmd == pytest.approx(0.20 + band + gains["kp_surge"] * es, abs=1e-9)

    def after(ps, obs):
        return pair_step(gains, ps, obs, params)[1]

    # Speed at its upper limit with the pressure low, and at its lower limit
    # with the pressure high: the speed integral holds.
    for I_N, dp in ((1.10, 20.0), (0.60, 28.0)):
        ps = PairState(I_N=I_N, I_x=0.3, timer=0.0)
        assert float(after(ps, _obs(params, dp, 0.15)).I_N) == I_N
    # Inside the limits it integrates ki_speed e dt.
    ps = PairState(I_N=0.9, I_x=0.3, timer=0.0)
    assert float(after(ps, _obs(params, 23.0, 0.15)).I_N) == pytest.approx(
        0.9 + gains["ki_speed"] * 1.0 * dt
    )
    # Valve command below 0 with the margin above the line, and above 1 with
    # it below: the anti-surge integral holds (after the reset clip).
    for x, margin in ((0.0, 0.40), (1.0, 0.09)):
        ps = PairState(I_N=1.0, I_x=x, timer=0.0)
        new = after(ps, _obs(params, 24.0, margin, x=x))
        assert float(new.I_x) == x


def test_reset_clears_the_pair(env, params, step):
    """After ``reset()`` the pair repeats its action sequence exactly.
    The first speed command continues the observed drive setpoint by the
    proportional term, and the first valve command is ``x + kp_surge es``
    from ``pair_init``'s integral, or full opening below the override line."""
    pair = CompressorSurgePair()
    key = jax.random.PRNGKey(5)
    first = _run_class(pair, env, params, step, key, 80)[1]
    pair.reset()
    second = _run_class(pair, env, params, step, key, 80)[1]
    np.testing.assert_array_equal(first, second)

    gains = DEFAULT_GAINS
    obs = _obs(params, 22.5, 0.20, N=0.88, N_ramp=0.90, x=0.25, setpoint_kpa=24.0)
    pair.step(_obs(params, 26.0, 0.30, N=0.85, x=0.4))
    pair.reset()
    N_cmd, x_cmd = commands_from_raw(pair.step(obs), params)
    assert N_cmd == pytest.approx(0.90 + gains["kp_speed"] * 1.5, abs=1e-12)
    es = gains["surge_line"] - surge_margin_from_obs(obs, params)
    assert x_cmd == pytest.approx(0.25 + gains["kp_surge"] * es, abs=1e-12)

    pair.reset()
    below = _obs(params, 24.0, 0.07, x=0.25)
    assert commands_from_raw(pair.step(below), params)[1] == 1.0


def test_functional_core_matches_the_class(env, params, step):
    """``pair_step(xp=jnp)`` under ``lax.scan`` gives the stateful pair's
    actions along a whole episode, to 1e-5 on the observations the pair saw.

    Closing the loop on the env itself, as scripts/compressor_surge_numbers.py
    runs it, the float32 core and the float64 class track the same four
    setpoints with no trip or restart, and their header pressures stay within
    0.01 kPa of each other at every step. That holds because no limit cycle
    amplifies their rounding differences (``GUARD_MAX_TRAVEL``)."""
    key = jax.random.PRNGKey(0)
    gains = DEFAULT_GAINS
    seen, actions = _run_class(
        CompressorSurgePair(gains, params), env, params, step, key, 1200
    )

    def replay(obs_seq):
        def body(ps, obs):
            u_raw, ps = pair_step(gains, ps, obs, params, xp=jnp)
            return ps, u_raw

        ps0 = pair_init(obs_seq[0], xp=jnp)
        return np.asarray(jax.lax.scan(body, ps0, obs_seq)[1])

    with enable_x64():
        np.testing.assert_allclose(
            replay(jnp.asarray(seen, jnp.float64)), actions, rtol=0.0, atol=1e-5
        )

    def closed(carry, _):
        state, ps = carry
        obs = get_obs(state, params)
        u_raw, ps = pair_step(gains, ps, obs, params, xp=jnp)
        state = env.step_env(key, state, u_raw, params)[1]
        return (state, ps), obs

    _, state0 = env.reset_env(key, params)
    ps0 = pair_init(get_obs(state0, params), xp=jnp)
    _, obs_seq = jax.lax.scan(closed, (state0, ps0), None, length=1200)
    obs_seq = np.asarray(obs_seq)
    # The same four setpoints were tracked, so neither run restarted.
    np.testing.assert_array_equal(obs_seq[:, 6], seen[:, 6])
    gap = np.abs(obs_seq[:, 0] - seen[:, 0])
    assert gap.max() <= 0.01, (gap.max(), int(gap.argmax()))


# ---------------------------------------------------------------------------
# The NMPC
# ---------------------------------------------------------------------------

#: Typical size of each model state (m_c, dp, N, N_ramp, x), for the absolute
#: part of test_mpc_model_is_the_env_step's tolerance where a state passes
#: through zero.
STATE_SCALE = np.array([20.0, 3.0e4, 1.0, 1.0, 1.0])


def _values(state):
    """The model's five physical states of an env state, as floats."""
    return np.array([float(getattr(state, name)) for name in MPC_STATES])


def _within_reach(command, state, params):
    """``command`` ([N_cmd, x_cmd]) moved onto one step's actuator travel from
    ``state``. The env's per-substep rate limits treat a command beyond that
    travel exactly as one at it, and the NLP only plans commands within it."""
    dt = params.delta_t
    N_ramp, x = float(state.N_ramp), float(state.x)
    return np.array(
        [
            np.clip(
                command[0],
                N_ramp - params.drive_rate * dt,
                N_ramp + params.drive_rate * dt,
            ),
            np.clip(
                command[1],
                x - params.valve_close_rate * dt,
                x + params.valve_open_rate * dt,
            ),
        ]
    )


def _pair_states(env, params, step, key, n):
    """``n`` steps of the shipped pair on the env from ``reset_env(key)``.
    Returns the states it acted on and its raw actions."""
    pair = CompressorSurgePair(load_gains(), params)
    obs, state = env.reset_env(key, params)
    states, actions = [], []
    for _ in range(n):
        action = pair(obs)
        states.append(state)
        actions.append(action)
        obs, state, _, _, _ = step(key, state, jnp.asarray(action), params)
    return states, np.array(actions)


def test_mpc_model_is_the_env_step(mpc, params):
    """Model-review check 12. The NMPC's step is the one piece of physics
    written twice: the env's is ``jnp`` code, which CasADi cannot trace. With
    the exact clip, at 1000 random states, schedules and clocks (ramps and the
    stretch past the episode's end included), deviations of up to 3 sd and
    commands within one step's reach, the model's next state and the env's
    soft minimum of its substep Phis equal ``compute_next_state``'s to 1e-6
    relative, in float64. test_mpc_predicts_one_step_like_the_plant bounds
    the shipped smoothing's own gap."""
    p = params
    rng = np.random.default_rng(0)
    n = 1000
    flow = float(suction_density(p)) * p.A_c * p.U_r
    dt = p.delta_t
    N = rng.uniform(0.72, 1.05, n)
    m_c = flow * N * rng.uniform(0.45, 0.80, n)
    dp = rng.uniform(5.0e3, 35.0e3, n)
    N_ramp = rng.uniform(p.N_min, p.N_max, n)
    x = rng.uniform(0.0, 1.0, n)
    dev = rng.uniform(-0.06, 0.06, n)
    clock = rng.integers(0, 1300, n)
    levels = rng.uniform(0.35, 0.95, (n, 6))
    N_cmd = np.clip(
        N_ramp + rng.uniform(-1.0, 1.0, n) * p.drive_rate * dt, p.N_min, p.N_max
    )
    x_cmd = np.clip(
        x + rng.uniform(-p.valve_close_rate * dt, p.valve_open_rate * dt, n), 0.0, 1.0
    )
    openings = np.stack(
        [stage_openings(levels[i], clock[i], dev[i], p) for i in range(n)]
    )
    x_next, phis = mpc.rhs_function(0.0).map(n)(
        np.stack([m_c, dp, N, N_ramp, x]), np.stack([N_cmd, x_cmd]), openings.T
    )
    ours = np.asarray(x_next).T
    ours_min = soft_min(np.asarray(phis).T, p.soft_min_temperature)

    raw = raw_from_commands(N_cmd, x_cmd, p).T
    with enable_x64():

        def one(m_c, dp, N, N_ramp, x, dev, clock, levels, raw):
            state = CompressorSurgeState(
                time=jnp.int32(0),
                m_c=m_c,
                dp=dp,
                N=N,
                N_ramp=N_ramp,
                x=x,
                demand_dev=dev,
                phi_min=jnp.zeros_like(m_c),
                setpoint_levels=jnp.full(4, 24.0e3),
                demand_levels=levels,
                block_clock=clock,
            )
            new = compute_next_state(raw, state, p, jax.random.PRNGKey(0))[0]
            return jnp.stack([new.m_c, new.dp, new.N, new.N_ramp, new.x, new.phi_min])

        theirs = np.asarray(
            jax.jit(jax.vmap(one))(
                *(jnp.asarray(v, jnp.float64) for v in (m_c, dp, N, N_ramp, x, dev)),
                jnp.asarray(clock, jnp.int32),
                jnp.asarray(levels, jnp.float64),
                jnp.asarray(raw, jnp.float64),
            )
        )
    gap = np.abs(ours - theirs[:, :5])
    assert np.all(gap <= 1e-6 * np.abs(theirs[:, :5]) + 1e-9 * STATE_SCALE), gap.max(0)
    np.testing.assert_allclose(ours_min, theirs[:, 5], rtol=1e-6, atol=0.0)
    # The draw covered the demand ramps and the stretch past the last block.
    ramp = (clock % 200 < 50) & (clock >= 200) & (clock < 1200)
    assert ramp.sum() > 100 and (clock >= 1200).sum() > 30


def test_mpc_predicts_one_step_like_the_plant(env, params, planned, step, mpc):
    """Model-review check 13. One step of the NMPC's model, with the
    shipped smoothing of the substep rate limits and the deviation the step
    integrates with, lands where the env's float32 step under the planner's
    params does. On 118 states of a pair episode with the deviation on, and
    two built from one of them, one knocked to 2 % from the surge line with
    the valve closing at full rate and one pushed far down the right branch
    (Phi 0.76) with both actuators opening at full rate: the header pressure
    within 0.05 ``e_floor`` and the step's ``phi_min`` within 1e-4.

    The env step is ``compute_next_state`` rather than ``step_env``, since a
    built state may trip, and ``step_env`` would then return the restart."""
    p = params
    states, actions = _pair_states(env, p, step, jax.random.PRNGKey(7), 1200)
    picked = list(zip(states, actions))[::10][:118]
    base = picked[60][0]
    flow = float(suction_density(p)) * p.A_c * p.U_r * float(base.N)
    dt = p.delta_t
    picked += [
        (
            base.replace(
                m_c=jnp.float32(surge_flow_per_speed(p) * float(base.N) * 1.02)
            ),
            raw_from_commands(
                float(base.N_ramp), float(base.x) - p.valve_close_rate * dt, p
            ),
        ),
        (
            base.replace(m_c=jnp.float32(flow * 0.76)),
            raw_from_commands(
                float(base.N_ramp) + p.drive_rate * dt,
                float(base.x) + p.valve_open_rate * dt,
                p,
            ),
        ),
    ]
    assert len(picked) == 120
    f = mpc.rhs_function()
    plant = jax.jit(
        lambda raw, s: compute_next_state(raw, s, planned, jax.random.PRNGKey(0))[0]
    )
    dp_gap, phi_gap = [], []
    for state, raw in picked:
        command = _within_reach(commands_from_raw(raw, p), state, p)
        opening = stage_openings(
            state.demand_levels, int(state.block_clock), float(state.demand_dev), p
        )
        x_next, phis = f(_values(state), command, opening)
        new = plant(jnp.asarray(raw_from_commands(*command, p), jnp.float32), state)
        dp_gap.append(float(x_next[1]) - float(new.dp))
        phi_gap.append(
            soft_min(np.asarray(phis).ravel(), p.soft_min_temperature)
            - float(new.phi_min)
        )
    dp_gap, phi_gap = np.abs(dp_gap), np.abs(phi_gap)
    assert dp_gap.max() <= 0.05 * planned.e_floor * KPA, dp_gap.max()
    assert phi_gap.max() <= 1e-4, phi_gap.max()


def test_mpc_falls_back_to_the_pair(env, params, planned, step, monkeypatch):
    """A step whose solve fails is handed to the pair: the action is
    exactly the one a ``CompressorSurgePair`` tracked through the same steps
    returns, the warm start is the one the failed solve began from, the
    failure and the fallback are counted, the next solve's move penalty reads
    the commands applied, and the next solve is clean. A solve that reports
    success with a NaN action is treated the same. A solve that the
    iteration cap stopped is applied and counted in ``capped_steps``;
    ``fallback_steps`` does not count it.

    After 20 NMPC steps that end near the NMPC's plan margin (below the
    pair's 7.5 % override line), the pair's first command opens the recycle
    fully, the override's response and the jump PHYSICS.md accepts, while
    its speed command continues the last applied one within one step's drive
    travel.
    Gains the guard refuses stop the factory."""
    gains = load_gains()
    mpc = make_compressor_surge_mpc(env, planned)
    shadow = CompressorSurgePair(gains, planned)
    key = jax.random.PRNGKey(2)
    _, state = env.reset_env(key, params)
    # Settled 4 % from the surge line at 28 kPa with the consumers at 0.35:
    # recycle is forced there and its power is above MPC_POWER_REF, so the
    # NMPC moves the plant to its plan margin, MPC_SURGE_MARGIN, and holds it
    # there.
    z = settled_point(28.0e3, 0.35, {**gains, "surge_line": 0.04}, params)
    flow = float(suction_density(params)) * params.A_c * params.U_r
    f32 = jnp.float32
    state = state.replace(
        m_c=f32(z[0]),
        dp=f32(z[1]),
        N=f32(z[2]),
        N_ramp=f32(z[3]),
        x=f32(z[4]),
        demand_dev=f32(0.0),
        phi_min=f32(z[0] / (flow * z[2])),
        setpoint_levels=jnp.full(4, 28.0e3, f32),
        demand_levels=jnp.full(6, 0.35, f32),
    )
    margins = []
    for _ in range(20):
        obs = env.get_obs(state, params)
        action = mpc.step(obs, state)
        shadow.track(obs, action)
        state = step(key, state, jnp.asarray(action), params)[1]
        margins.append(float(state.phi_min) / (2.0 * params.W) - 1.0)
    assert mpc.solve_failures == 0
    assert margins[-1] < gains["override_line"], margins
    last = commands_from_raw(action, planned)

    solve = mpc._mpc.make_step
    began = {}

    def failed_solve(x0):
        m = mpc._mpc
        began["guess"] = np.array(m.opt_x_num.master)
        u = solve(x0)
        m.opt_x_num.master = np.full(np.asarray(m.opt_x_num.master).shape, np.nan)
        m.solver_stats = {
            **m.solver_stats,
            "success": False,
            "return_status": "Restoration_Failed",
        }
        return u

    def non_finite_solve(x0):
        began["guess"] = np.array(mpc._mpc.opt_x_num.master)
        return np.full_like(solve(x0), np.nan)

    for n, fault, status in (
        (1, failed_solve, "Restoration_Failed"),
        (2, non_finite_solve, "non-finite action"),
    ):
        monkeypatch.setattr(mpc._mpc, "make_step", fault)
        obs = env.get_obs(state, params)
        expected = copy.deepcopy(shadow).step(obs)
        action = mpc.step(obs, state)
        np.testing.assert_array_equal(action, expected)
        np.testing.assert_array_equal(
            np.array(mpc._mpc.opt_x_num.master), began["guess"]
        )
        assert mpc.solve_failures == mpc.fallback_steps == n
        assert mpc.capped_steps == 0
        assert mpc.last_return_status == status
        np.testing.assert_array_equal(
            np.array(mpc._mpc.u0.cat).ravel(), commands_from_raw(action, planned)
        )
        if n == 1:
            assert (
                surge_margin_from_obs(np.asarray(obs), params) < gains["override_line"]
            )
            N_cmd, x_cmd = commands_from_raw(action, planned)
            assert x_cmd == 1.0
            assert abs(N_cmd - last[0]) <= params.drive_rate * params.delta_t
        shadow.step(obs)
        state = step(key, state, jnp.asarray(action), params)[1]

        monkeypatch.undo()
        obs = env.get_obs(state, params)
        action = mpc.step(obs, state)
        shadow.track(obs, action)
        assert np.all(np.isfinite(action))
        assert mpc.solve_failures == mpc.fallback_steps == n
        state = step(key, state, jnp.asarray(action), params)[1]

    # A solve that the iteration cap stopped is applied. The base counts it as
    # a failed and a capped solve; it is not a fallback.
    def capped_solve(x0):
        u = solve(x0)
        began["u"] = np.array(u, float).ravel()
        mpc._mpc.solver_stats = {
            **mpc._mpc.solver_stats,
            "success": False,
            "return_status": "Maximum_Iterations_Exceeded",
        }
        return u

    monkeypatch.setattr(mpc._mpc, "make_step", capped_solve)
    action = mpc.step(env.get_obs(state, params), state)
    np.testing.assert_array_equal(
        action, np.clip(raw_from_commands(*began["u"], planned), -1.0, 1.0)
    )
    assert mpc.solve_failures == 3 and mpc.solve_capped == 1
    assert mpc.fallback_steps == 2 and mpc.capped_steps == 1
    monkeypatch.undo()

    monkeypatch.setattr(
        pid_mod,
        "_gains_cache",
        {
            **pid_mod._load_gains(),
            "compressor_surge": {**DEFAULT_GAINS, **REFUSED["anti-surge (3, 0.3)"]},
        },
    )
    with pytest.raises(UnsafePairGains):
        make_compressor_surge_mpc(env, planned, horizon=5)


def test_mpc_retries_a_failed_cold_step_cold(env, params, planned, step, monkeypatch):
    """A cold step (the first, or the first after a trip) goes to the IPOPT
    instance with the default barrier. When its solve fails, the fallback
    restores the cold rollout, which carries no multipliers, so the next step
    goes to the cold instance again instead of warm-starting from it. After a
    clean cold solve the steps are warm, and a failed warm step leaves the
    next one warm, from its own restored plan. A short horizon keeps the
    solves cheap; the routing does not depend on it."""
    mpc = make_compressor_surge_mpc(env, planned, horizon=10)
    key = jax.random.PRNGKey(4)
    _, state = env.reset_env(key, params)
    solve = mpc._mpc.make_step
    used, fail = [], {"now": False}

    def recorded(x0):
        used.append("cold" if mpc._mpc.S is mpc._solvers["cold"] else "warm")
        u = solve(x0)
        return np.full_like(u, np.nan) if fail["now"] else u

    monkeypatch.setattr(mpc._mpc, "make_step", recorded)
    for failing in (True, True, False, False, True, False):
        fail["now"] = failing
        obs = env.get_obs(state, params)
        action = mpc.step(obs, state)
        assert np.all(np.isfinite(action))
        state = step(key, state, jnp.asarray(action), params)[1]
    assert used == ["cold", "cold", "cold", "warm", "warm", "warm"]
    assert mpc.solve_failures == mpc.fallback_steps == 3

    mpc.reset()
    obs = env.get_obs(state, params)
    mpc.step(obs, state)
    assert used[-1] == "cold"


def test_mpc_cold_start_forgets_the_last_plan(env, params, planned, step, monkeypatch):
    """A cold start builds its guess from the state and the preview alone:
    the model's rollout, the commands that hold the actuators, zero surge
    slacks and no multipliers. do-mpc's ``set_initial_guess`` resets only
    the states and inputs, so ``_initialise`` resets the slacks itself.

    After five steps of an episode, with the last plan's slacks then moved
    to 0.01, ``reset()`` and a step from the episode's first state start
    IPOPT from the decision vector and multipliers of the controller's first
    step, and IPOPT takes as many iterations. A short horizon keeps the
    solves cheap."""
    import casadi

    mpc = make_compressor_surge_mpc(env, planned, horizon=10)
    key = jax.random.PRNGKey(4)
    obs0, state0 = env.reset_env(key, params)
    solve = mpc._mpc.make_step
    starts = []

    def recorded(x0):
        m = mpc._mpc
        start = {
            "x": np.array(m.opt_x_num.master, float).ravel(),
            "lam_x": np.array(m.lam_x_num, float).ravel(),
            "lam_g": np.array(m.lam_g_num, float).ravel(),
        }
        u = solve(x0)
        start["iters"] = int(m.solver_stats["iter_count"])
        starts.append(start)
        return u

    monkeypatch.setattr(mpc._mpc, "make_step", recorded)
    obs, state = obs0, state0
    for _ in range(5):
        action = mpc.step(obs, state)
        obs, state, _, _, _ = step(key, state, jnp.asarray(action), params)
    m = mpc._mpc
    v = np.array(m.opt_x_num.master, float).ravel()
    v[mpc._eps_index] = 0.01
    m.opt_x_num.master = casadi.DM(v)

    mpc.reset()
    mpc.step(obs0, state0)
    first, again = starts[0], starts[-1]
    assert np.all(first["x"][mpc._eps_index] == 0.0)
    for name in ("x", "lam_x", "lam_g"):
        np.testing.assert_array_equal(again[name], first[name], err_msg=name)
    assert again["iters"] == first["iters"]


def test_mpc_preview_matches_the_env(env, params, planned, mpc):
    """Stage k of the NLP is the state k steps ahead. Its setpoint is the
    level the env scores that state against, and its openings are what the
    env's RK4 reads during the step from it: the demand at the step's stage
    times, from the env's own clock and deviation k steps on. Checked inside
    a block, across a demand ramp, across a setpoint step, and past the
    episode's end, where the last levels hold; the deviation starts at 1.5 sd.
    What the NLP receives through its tvp function is that preview, and its
    stage weights put the terminal cost on the last planned state."""
    _, state = env.reset_env(jax.random.PRNGKey(6), params)
    state = state.replace(
        demand_dev=jnp.float32(0.03),
        setpoint_levels=jnp.array([21.0e3, 27.0e3, 23.0e3, 25.0e3], jnp.float32),
        demand_levels=jnp.array([0.45, 0.85, 0.40, 0.70, 0.55, 0.90], jnp.float32),
    )
    setpoints = np.asarray(state.setpoint_levels, float)
    levels = np.asarray(state.demand_levels, float)
    H = mpc.horizon
    assert H == MPC_HORIZON
    h = params.delta_t / MPC_SUBSTEPS
    offsets = 0.5 * h * np.arange(2 * MPC_SUBSTEPS + 1)
    advance = jax.jit(
        lambda s: compute_next_state(jnp.zeros(2), s, planned, jax.random.PRNGKey(0))[0]
    )
    # (block clock, setpoint block of the first and of the last planned
    # state). From 190 the stages cross the whole ramp into the second
    # demand block (clock 200 to 250), from 270 the setpoint step at 300, and
    # from 1180 the end of the episode.
    for clock, first, last in ((0, 0, 0), (190, 0, 0), (270, 0, 1), (1180, 3, 3)):
        s = start = state.replace(block_clock=jnp.int32(clock))
        p_ref, opening = [], []
        for _ in range(H + 2):
            p_ref.append(float(live_target(s, planned)))
            tau = jnp.asarray(
                float(s.block_clock) * params.delta_t + offsets, jnp.float32
            )
            opening.append(
                np.asarray(demand_opening(s.demand_levels, tau, s.demand_dev, planned))
            )
            s = advance(s)
        preview = mpc.preview(start)
        np.testing.assert_array_equal(preview["p_ref"], p_ref)
        assert (p_ref[0], p_ref[H]) == (setpoints[first], setpoints[last])
        np.testing.assert_allclose(preview["opening"], opening, rtol=0.0, atol=1e-6)
        if clock == 190:
            # The ramp is in the preview: the openings move by most of the
            # step between the two levels.
            spread = np.ptp(preview["opening"])
            assert spread > 0.9 * abs(levels[1] - levels[0]), spread

        mpc._update_setpoint(start)
        received = mpc._mpc.tvp_fun(0.0)
        for name in ("opening", "p_ref", "w_track", "w_energy"):
            got = np.array(
                [
                    np.asarray(received["_tvp", k, name], float).ravel()
                    for k in range(H + 2)
                ]
            )
            np.testing.assert_array_equal(
                got.reshape(np.shape(preview[name])), preview[name]
            )

    np.testing.assert_array_equal(
        preview["w_track"], [1.0] * H + [MPC_TERMINAL_WEIGHT, 0.0]
    )
    np.testing.assert_array_equal(preview["w_energy"], [1.0] * H + [0.0, 0.0])
    assert isinstance(mpc, CompressorSurgeMPC)


def test_mpc_cold_start_avoids_a_surging_guess(env, params, planned):
    """A cold start with the recycle shut and a demand drop inside the
    horizon, the stress battery's first scenario with the demand deviation at
    -4 stationary sd. Holding both actuators where they are runs that guess
    into surge, and IPOPT started there converges to a plan that surges.
    With that guess the acceptance run's stress battery tripped in 12 of its
    16 runs at both candidate margins (measured,
    scripts/compressor_surge_numbers.py --mpc-gate, before the cold start
    took the pair's rollout). The cold start must begin outside the surge
    region, so the first plan uses no surge slack."""
    p0, p1, u0, u1 = BATTERY[0]
    quiet = params.replace(demand_sigma=0.0)
    z0 = settled_point(p0, u0, load_gains(), params)
    state = _state_at(
        jnp.asarray(z0, jnp.float32),
        jnp.asarray([p0, p0, p1, p1], jnp.float32),
        jnp.asarray([u0] * 3 + [u1] * 3, jnp.float32),
        BATTERY_START_CLOCK,
        quiet,
    )
    state = state.replace(demand_dev=jnp.float32(-4.0 * params.demand_sigma))
    mpc = make_compressor_surge_mpc(env, planned)
    mpc.reset()
    mpc.step(np.asarray(env.get_obs(state, quiet)), state)
    v = np.array(mpc._mpc.opt_x_num.master, float).ravel()
    assert v[mpc._eps_index][1:].max() < 1e-6


# ---------------------------------------------------------------------------
# The pair on the validation seeds, and the floor
# ---------------------------------------------------------------------------


def _numbers_script():
    """scripts/compressor_surge_numbers.py, loaded as a module. It silences
    every warning when it loads, so it loads inside ``catch_warnings``."""
    import importlib.util
    import warnings

    path = pathlib.Path(__file__).resolve().parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "compressor_surge_numbers", path / "compressor_surge_numbers.py"
    )
    module = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        spec.loader.exec_module(module)
    return module


@pytest.mark.slow
def test_the_pair_holds_the_validation_seeds():
    """The shipped pair, with the gains the registry's factory reads, on the
    first 1024 of the validation's seeds: no trip, and the smallest surge
    margin of every episode stays above the override line. The same episodes
    as ``scripts/compressor_surge_numbers.py --pid-validation``, which runs
    4096 and prints the margin quantiles. With the shipped gains it measured
    no trip in 4096 and a smallest margin of 0.0877 over all of them, 0.0878
    over the first 1024, against the override line's 0.075."""
    numbers = _numbers_script()
    gains = numbers.load_gains()
    seeds = np.arange(numbers.HEAVY_SEED0, numbers.HEAVY_SEED0 + 1024)
    summ, _ = numbers.run(numbers.policy("pair", gains=gains), seeds)
    assert int(summ["trips"].sum()) == 0
    assert float(summ["min_margin"].min()) > gains["override_line"]


def test_floor_is_the_recorded_mpc_hold(params):
    """``e_floor``, ``c_hold`` and the NEA references come from the NMPC's
    recorded hold. With e_hold_min the smallest per-seed hold error that
    scripts/measure_hold.py recorded for the NMPC, ``e_floor`` is the larger
    of the transmitter's accuracy (``precision_floor``, 0.0275 kPa) and
    e_hold_min. ``c_hold`` is the NMPC's mean hold-phase recycle power, which
    the script records in W, as ``recycle_power`` returns it.
    ``rho_floor_tracking`` is (e_hold_min / e_floor)^2, and ``rho_floor``
    equals it, as on the suite's other plants with a running cost: recycle
    power at ``c_hold`` costs nothing, and ``c_hold`` is the NMPC's own. The
    NMPC's hold tracking cost in the recorded protocol row is at least 0.98
    of that reference. Each equality holds to 1 %."""
    holds = json.loads((DATA / "hold_measurements.json").read_text())
    if "compressor_surge" not in holds:
        pytest.fail(
            "no recorded NMPC hold for compressor_surge. Run `uv run python "
            "scripts/measure_hold.py --envs compressor_surge`, set e_floor, "
            "c_hold, rho_floor_tracking and rho_floor from it, and commit "
            "src/target_gym/data/hold_measurements.json."
        )
    rows = json.loads((DATA / "protocol_results.json").read_text())
    if "compressor_surge" not in rows:
        pytest.fail(
            "no recorded protocol row for compressor_surge. Run `uv run python "
            "scripts/evaluate_baselines.py --envs compressor_surge` and commit "
            "src/target_gym/data/protocol_results.json."
        )

    mpc = holds["compressor_surge"]["mpc"]
    e_hold_min = min(float(seed[0]) for seed in mpc["e_hold_per_seed"])
    p = params
    assert p.e_floor == pytest.approx(max(p.precision_floor, e_hold_min), rel=0.01)
    assert p.c_hold == pytest.approx(float(mpc["c_hold"]), rel=0.01)
    reference = (e_hold_min / p.e_floor) ** 2
    assert p.rho_floor_tracking == pytest.approx(reference, rel=0.01)
    assert p.rho_floor == pytest.approx(p.rho_floor_tracking, rel=0.01)
    hold_tracking = rows["compressor_surge"]["mpc"]["hold_tracking"]
    assert hold_tracking >= 0.98 * p.rho_floor_tracking, (
        hold_tracking,
        p.rho_floor_tracking,
    )
