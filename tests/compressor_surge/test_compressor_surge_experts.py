"""The compressor's controllers.

Enforces ``src/target_gym/compressor_surge/PHYSICS.md``, section 7: the
pair's gains and where they are read from, the guard its factory applies, the
override, the anti-windup of both loops, the reset, and the agreement between
the functional core that scripts/compressor_surge_numbers.py runs and the
stateful controller the suite runs. Last, the pair on the first 1024 seeds
of its validation, and the floor that scripts/measure_hold.py records from
the MPC's hold. The oracle, on the shared solver, has its own tests
(tests/oracles/test_process_oracles.py).

The env is built directly from its class.
"""

import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import target_gym.experts.pid as pid_mod
from target_gym import records
from target_gym.compressor_surge.env import (
    KPA,
    CompressorSurgeParams,
    get_obs,
    surge_flow_per_speed,
)
from target_gym.compressor_surge.env_jax import CompressorSurge
from target_gym.compressor_surge.experts import (
    BATTERY,
    BATTERY_LEAD,
    BATTERY_STEPS,
    DEFAULT_GAINS,
    GUARD_MAX_TRAVEL,
    GUARD_MIN_DECAY,
    GUARD_MIN_MARGIN,
    GUARD_POINTS,
    GUARD_TRAVEL_STEPS,
    HELD_GAINS,
    TUNED_GAINS,
    CompressorSurgePair,
    PairState,
    UnsafePairGains,
    battery_report,
    check_pair_gains,
    closed_loop_decay,
    commands_from_raw,
    load_gains,
    make_compressor_surge_pid,
    pair_init,
    pair_step,
    settled_point,
    surge_margin_from_obs,
)

try:
    from jax import enable_x64
except ImportError:  # JAX 0.6; 0.10 removed the experimental name
    from jax.experimental import enable_x64

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


@pytest.fixture(scope="module")
def env():
    return CompressorSurge()


@pytest.fixture(scope="module")
def params():
    return CompressorSurgeParams()


@pytest.fixture(scope="module")
def step(env):
    return jax.jit(env.step_env)


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
    at zero never moves, and it moves every gain it is not told to hold, so
    a new gain has to be named tuned or held. The override sits at half the
    control margin, where rendering draws its alarm."""
    assert all(np.isfinite(v) for v in DEFAULT_GAINS.values())
    assert all(DEFAULT_GAINS[k] != 0.0 for k in TUNED_GAINS)
    assert set(TUNED_GAINS) | set(HELD_GAINS) == set(DEFAULT_GAINS)
    assert not set(TUNED_GAINS) & set(HELD_GAINS)
    assert DEFAULT_GAINS["override_line"] == 0.5 * DEFAULT_GAINS["surge_line"]


def test_gains_are_read_under_the_task_key(monkeypatch, params):
    """The factory reads ``pid._gains_cache["compressor_surge"]`` at call
    time (the tuner patches it), skips the ``note``, and nothing else in
    ``pid_gains.json`` starts with the task's name, since
    ``expert_fingerprint`` collects gains by that prefix."""
    stored = pid_mod.load_gains()
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
            **pid_mod.load_gains(),
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
    # The closed loop tests rounding, which a noisy run would amplify.
    params = params.replace(sensor_noise_scale=0.0)
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
    # The same four setpoints were tracked, so neither run restarted. Ramps are
    # computed, so the two compilations can differ by an ulp.
    np.testing.assert_allclose(obs_seq[:, 6], seen[:, 6], rtol=1e-6)
    gap = np.abs(obs_seq[:, 0] - seen[:, 0])
    assert gap.max() <= 0.01, (gap.max(), int(gap.argmax()))


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


@pytest.mark.large
def test_the_pair_holds_the_validation_seeds():
    """The shipped pair, with the gains the registry's factory reads, on the
    first 1024 of the validation's seeds: no trip, and the smallest surge
    margin of every episode stays above the override line. The same episodes
    as ``scripts/compressor_surge_numbers.py --pid-validation``, which runs
    4096 and prints the margin quantiles. With the shipped gains it measured
    no trip in 4096 and a smallest margin of 0.0951 over all of them, 0.0954
    over the first 1024, against the override line's 0.075."""
    numbers = _numbers_script()
    gains = numbers.load_gains()
    seeds = np.arange(numbers.HEAVY_SEED0, numbers.HEAVY_SEED0 + 1024)
    summ, _ = numbers.run(numbers.policy("pair", gains=gains), seeds)
    assert int(summ["trips"].sum()) == 0
    assert float(summ["min_margin"].min()) > gains["override_line"]


def test_floor_is_the_recorded_mpc_hold(params):
    """``e_floor``, ``c_hold`` and the oracle's best holds come from the NMPC's
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
    holds = records.load("hold_measurements")
    if "compressor_surge" not in holds:
        pytest.fail(
            "no recorded NMPC hold for compressor_surge. Run `uv run python "
            "scripts/measure_hold.py --envs compressor_surge`, set e_floor, "
            "c_hold, rho_floor_tracking and rho_floor from it, and commit "
            "src/target_gym/data/hold_measurements/compressor_surge.json."
        )
    rows = records.load("protocol_results")
    if "compressor_surge" not in rows:
        pytest.fail(
            "no recorded protocol row for compressor_surge. Run `uv run python "
            "scripts/evaluate_baselines.py --envs compressor_surge` and commit "
            "src/target_gym/data/protocol_results/compressor_surge.json."
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
