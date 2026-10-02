"""The 2D aircraft oracle: the capture on the shipped planner, the hold with
the throttle on the PID's airspeed loop.

The shared gradient planner never moved its throttle after its first plan
(the oracle audit, 2026-10). On ``plane`` and ``plane_sine`` the shipped
planner still flies the capture of the initial altitude offset, and once the
aircraft has settled a second planner takes over: the throttle is the
cascaded PID's airspeed loop, in the plant and, as a JAX port, inside the
planner's rollout, and the planner optimises the elevator alone with a step
decaying from 0.05 to 0.002 over 100 iterations. ``plane_energy`` keeps the
shipped planner, and its altitude floor was reset from an earlier oracle's
4.55 m hold to 1.20 m (``plane_energy-v3``). The settings live in the
package's ``experts.py``, which is in the three tasks' baseline fingerprint,
and each task's oracle reads its PID gains from a key that fingerprint
collects. The planner's objective is the reward over a fixed scale per task,
so the NEA reference its own holds set does not change it.
"""

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym import registry
from target_gym.experts import pid as pid_mod
from target_gym.experts.mpc import plan_params
from target_gym.plane.env import PlaneParams
from target_gym.plane.experts import (
    PlaneHandoverMPC,
    PlaneMPC,
    _bumpless_integral,
    _speed_loop,
    _speed_loop_gains,
    cascaded_pid_factory,
    make_plane_mpc,
    task_settings,
)

TASKS = ["plane", "plane_sine", "plane_energy"]
HANDOVER = {"plane": True, "plane_sine": True, "plane_energy": False}
HOLDS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "target_gym"
    / "data"
    / "hold_measurements.json"
)


def _oracle(name, **kwargs):
    spec = registry.get(name)
    env = spec.make_env()
    params = spec.make_test_params()
    return spec.make_mpc(env, plan_params(spec, params), **kwargs)


def _numeric(entry):
    return {k: v for k, v in entry.items() if not isinstance(v, (str, dict, list))}


def _check_shipped(planner):
    """The planner the tasks had before the oracle audit."""
    assert isinstance(planner, PlaneMPC)
    assert planner.speed_loop is False and planner._pid is None
    assert (planner.horizon, planner.n_tail, planner.n_iter) == (20, 0, 50)
    assert planner.lr == pytest.approx(0.05)
    assert planner.lr_end is None
    assert planner.guide_plan_fn is planner.initial_plan_fn is not None


@pytest.mark.parametrize("name", TASKS)
def test_the_registry_builds_each_task_with_its_settings(name):
    oracle = _oracle(name)
    if not HANDOVER[name]:
        _check_shipped(oracle)
        return
    assert isinstance(oracle, PlaneHandoverMPC)
    _check_shipped(oracle.capture)
    hold = oracle.hold
    assert hold.speed_loop is True and hold._pid is not None
    assert (hold.horizon, hold.n_tail, hold.n_iter) == (20, 0, 100)
    assert hold.lr == pytest.approx(0.05)
    assert hold.lr_end == pytest.approx(0.002)
    assert hold.guide_plan_fn is hold.initial_plan_fn is not None
    assert (oracle.e_on, oracle.settle_steps, oracle.e_off) == (3.0, 5, 20.0)


def test_the_objective_scale_is_fixed_per_task():
    """The planner's objective is the reward over a fixed scale, the one each
    oracle was measured with, not over ``rho_floor_tracking``: that is the NEA
    reference, set from this oracle's own holds, and reading it made each
    re-measured floor change the oracle that measured it."""
    scales = {0: 0.84**2, 3: 1.0, 1: 1.0}
    for pattern, scale in scales.items():
        settings = task_settings(PlaneParams(target_pattern=pattern))
        assert settings["objective_scale"] == pytest.approx(scale)
    spec = registry.get("plane_energy")
    env = spec.make_env()
    params = plan_params(spec, spec.make_test_params())
    plans = jnp.asarray(
        np.random.default_rng(0).uniform(-1.0, 1.0, (2, 3, 2)), jnp.float32
    )
    _, state = env.reset_env(jax.random.PRNGKey(0), params)
    values = []
    for rho in (params.rho_floor_tracking, 50.0):
        oracle = make_plane_mpc(
            env, params.replace(rho_floor_tracking=rho), horizon=3, n_iter=1
        )
        values.append([float(oracle._jit_rollout(plan, state)) for plan in plans])
    assert values[0] == values[1]


def test_other_patterns_and_version_1_keep_the_shipped_planner():
    for pattern in (2, 4):  # ramp and chirp: no registered task
        settings = task_settings(PlaneParams(target_pattern=pattern))
        assert settings["speed_loop"] is False
        assert settings["handover"] is False
        assert settings["lr_end"] is None
        # The scale they planned with before the oracle audit, the then
        # default rho_floor_tracking.
        assert settings["objective_scale"] == pytest.approx(0.84**2)
    env = registry.get("plane").make_env()
    v1 = make_plane_mpc(env, PlaneParams(reward_version=1))
    assert isinstance(v1, PlaneMPC)
    assert v1.speed_loop is False
    assert (v1.horizon, v1.n_tail, v1.n_iter, v1.lr_end) == (30, 60, 50, None)
    assert v1.guide_plan_fn is None


def test_explicit_settings_override_the_table():
    env = registry.get("plane").make_env()
    oracle = make_plane_mpc(env, PlaneParams(), n_iter=3, lr_end=None, speed_loop=False)
    assert isinstance(oracle, PlaneMPC)
    assert (oracle.n_iter, oracle.lr_end, oracle.speed_loop) == (3, None, False)
    loop = make_plane_mpc(env, PlaneParams(), n_iter=3, handover=False)
    assert isinstance(loop, PlaneMPC)
    assert (loop.n_iter, loop.speed_loop) == (3, True)
    both = make_plane_mpc(env, PlaneParams(), n_iter=3)
    assert both.hold.n_iter == 3 and both.capture.n_iter == 50
    with pytest.raises(ValueError, match="speed_loop"):
        make_plane_mpc(env, PlaneParams(), speed_loop=False, handover=True)


@pytest.mark.parametrize("name", TASKS)
def test_each_oracle_reads_gains_its_fingerprint_collects(name):
    """``baseline_fingerprint`` hashes the gain keys that start with the task's
    name, so the oracle reads one of those. The copies equal
    ``plane_cascaded``, which the PID baseline reads, so a retune of it has to
    update them too, and that stales these tasks' records as it should."""
    spec = registry.get(name)
    key = task_settings(spec.make_test_params())["gains_key"]
    assert key.startswith(spec.name)
    on_disk = json.loads(pid_mod._GAINS_FILE.read_text())
    assert _numeric(on_disk[key]) == _numeric(on_disk["plane_cascaded"])
    oracle = _oracle(name)
    shipped = vars(pid_mod.make_plane_cascaded_pid())
    for planner in (oracle.capture, oracle.hold) if HANDOVER[name] else (oracle,):
        assert vars(planner.initial_plan_fn.make_pid()) == shipped
    if HANDOVER[name]:
        live = vars(oracle.hold._pid)
        assert {k: v for k, v in live.items() if not k.startswith("_")} == {
            k: v for k, v in shipped.items() if not k.startswith("_")
        }


def test_the_tuner_writes_the_copies_with_plane_cascaded(monkeypatch):
    """scripts/tune_pid.py writes ``plane_cascaded``'s values under every key
    an oracle reads a copy from, keeping each copy's own note, so a retune
    keeps the test above passing without a hand edit."""
    import importlib.util
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))  # the script prepends src
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tune_pid.py"
    loader = importlib.util.spec_from_file_location("_tune_pid_under_test", path)
    tune = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(tune)

    read = {
        task_settings(registry.get(n).make_test_params())["gains_key"] for n in TASKS
    }
    copies = set(read) - {"plane_cascaded"}
    assert set(tune.GAINS_COPIES["plane_cascaded"]) == copies
    on_disk = json.loads(pid_mod._GAINS_FILE.read_text())
    gains = {k: dict(v) for k, v in on_disk.items()}
    gains["plane_cascaded"] = {
        **gains["plane_cascaded"],
        "Kp_alt": 0.5,
        "note": "a retune",
    }
    tune._write_copies(gains, "plane_cascaded")
    for key in copies:
        assert _numeric(gains[key]) == _numeric(gains["plane_cascaded"])
        assert gains[key]["Kp_alt"] == 0.5
        assert gains[key]["note"] == on_disk[key]["note"]


def test_a_missing_gains_key_raises():
    with pytest.raises(KeyError, match="plane_nowhere"):
        cascaded_pid_factory("plane_nowhere")()


def test_the_jax_speed_loop_is_the_pids_throttle_branch():
    """The port inside the rollout steps the integrator and the anti-windup
    exactly as the live PID does, through saturation both ways."""
    pid = pid_mod.make_plane_cascaded_pid()
    pid.reset()
    gains = _speed_loop_gains(pid)
    speeds = np.concatenate(
        [
            np.linspace(230.0, 200.0, 20),  # slow: full throttle
            np.full(10, 290.0),  # fast: idle
            230.0 + 3.0 * np.sin(np.arange(30) / 3.0),
        ]
    ).astype(np.float32)
    integ = jnp.float32(0.0)
    step = jax.jit(_speed_loop, static_argnums=2)
    obs = np.zeros(10, dtype=np.float32)
    for v in speeds:
        obs[0] = v
        expected = float(pid.step(obs)[0])
        throttle, integ = step(integ, jnp.float32(v), gains)
        assert float(throttle) == pytest.approx(expected, abs=1e-6)
        assert float(integ) == pytest.approx(float(pid._speed_integral), abs=1e-3)


def test_the_bumpless_integral_starts_cold_at_full_throttle():
    """Climbing out 30 m/s slow, the cold loop asks for full throttle, which
    is what the capture planner was flying: the integrator starts at zero.
    Seeding it to match a command a hair inside full throttle would make the
    integral term cancel the proportional one, and with the loop's small
    integral gain the airspeed would take hundreds of steps to recover."""
    pid = pid_mod.make_plane_cascaded_pid()
    gains = _speed_loop_gains(pid)
    assert _bumpless_integral(0.9986, 200.0, gains) == 0.0
    assert _bumpless_integral(-0.9986, 290.0, gains) == 0.0


def test_the_bumpless_integral_matches_a_throttle_off_its_stops():
    """4 m/s fast at 0.03 throttle, the cold loop would cut to about -0.44;
    seeded, its first command lands within the tolerance of 0.03, from below,
    and the step after it moves on the proportional term alone."""
    pid = pid_mod.make_plane_cascaded_pid()
    gains = _speed_loop_gains(pid)
    obs = np.zeros(10, dtype=np.float32)
    obs[0] = 234.2
    pid.reset()
    cold = float(pid.step(obs)[0])
    assert cold < -0.4
    pid.reset()
    pid._speed_integral = _bumpless_integral(0.03, float(obs[0]), gains, tol=0.01)
    first = float(pid.step(obs)[0])
    assert first == pytest.approx(0.02, abs=1e-5)


class _Stub:
    """Stands in for a ``PlaneMPC``: a constant throttle, no planning."""

    def __init__(self, speed_loop, throttle):
        self.speed_loop, self.throttle = speed_loop, throttle
        self.env = self.params = None
        self.horizon = 3
        self._pid = pid_mod.make_plane_cascaded_pid() if speed_loop else None
        self.reset()

    def reset(self):
        self._actions = jnp.zeros((3, 2))
        self._fresh, self.steps = True, 0
        if self._pid is not None:
            self._pid.reset()

    def step(self, obs, state):
        self.steps += 1
        self._fresh = False
        self._actions = jnp.full((3, 2), self.throttle)
        return np.array([self.throttle, 0.1], dtype=np.float32)


def _state(t, error, x_dot=200.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        time=t, target_altitude=1000.0, z=1000.0 - error, x_dot=x_dot
    )


def test_the_handover_waits_for_a_settled_aircraft_and_falls_back():
    """The capture planner flies until five consecutive steps within 3 m,
    the hold planner from the fifth, warm-started from the capture plan, and
    the capture planner again once the error passes 20 m."""
    capture, hold = _Stub(False, 0.9986), _Stub(True, 0.4)
    oracle = PlaneHandoverMPC(capture, hold)
    errors = [50.0, 2.0, 2.0, 2.5, 2.0, 2.0, 1.0, 25.0, 2.0]
    for t, error in enumerate(errors):
        action = oracle.step(None, _state(t, error))
        if t == 4:
            assert oracle.mode == "capture" and action[0] == np.float32(0.9986)
        if t == 5:
            assert oracle.mode == "hold" and action[0] == np.float32(0.4)
            assert hold._pid._speed_integral == 0.0  # both at full throttle
    assert oracle.handovers == [(5, "hold"), (7, "capture")]
    assert (capture.steps, hold.steps) == (7, 2)
    assert oracle.mode == "capture"
    oracle.reset()
    assert oracle.mode == "capture" and oracle.handovers == []
    assert jnp.all(oracle._actions == 0.0)


def test_the_capture_resumes_from_the_hold_plan_at_the_applied_throttle():
    capture, hold = _Stub(False, 0.9986), _Stub(True, 0.4)
    oracle = PlaneHandoverMPC(capture, hold, settle_steps=1)
    oracle.step(None, _state(0, 50.0))
    oracle.step(None, _state(1, 1.0))
    assert oracle.mode == "hold"
    hold._actions = jnp.asarray([[0.0, 0.3], [0.0, 0.2], [0.0, 0.1]])
    oracle._to_capture(_state(2, 30.0))
    assert np.allclose(np.asarray(capture._actions)[:, 0], 0.4)
    assert np.allclose(np.asarray(capture._actions)[:, 1], [0.3, 0.2, 0.1])
    assert capture._fresh is False


@pytest.mark.slow
def test_the_plant_gets_the_pids_throttle_and_the_plan_moves_the_elevator():
    """One planner step: the plant gets the live PID's throttle, the plan's
    throttle column has no gradient, and ``reset`` clears the integrator.
    About 30 s, nearly all of it compiling the planner."""
    spec = registry.get("plane")
    env = spec.make_env()
    params = plan_params(spec, spec.make_test_params())
    oracle = make_plane_mpc(env, params, horizon=3, n_iter=3, handover=False)
    obs, state = env.reset_env(jax.random.PRNGKey(0), spec.make_test_params())
    pid = pid_mod.make_plane_cascaded_pid()
    pid.reset()
    action = oracle.step(obs, state)
    assert action.shape == (2,)
    assert action[0] == np.float32(pid.step(np.asarray(obs))[0])
    assert oracle._pid._speed_integral == pid._speed_integral
    plan = jnp.zeros((3, 2), dtype=jnp.float32)
    grad = jax.grad(lambda a: oracle._rollout(a, (state, jnp.float32(0.0))))(plan)
    assert np.all(np.asarray(grad[:, 0]) == 0.0)
    assert np.any(np.asarray(grad[:, 1]) != 0.0)
    oracle.reset()
    assert oracle._pid._speed_integral == 0.0
    assert jnp.all(oracle._actions == 0.0)


@pytest.mark.parametrize("name", TASKS)
def test_the_floors(name):
    """``plane``'s floor is the 1 m altimeter resolution. ``plane_sine``'s
    1.26 m is an earlier oracle's hold, kept: the current oracle holds below
    the 1 m resolution, which is within 1.5x of it. ``plane_energy``'s was
    reset from 4.55 m, an earlier oracle's hold 3.8x the current one, to
    1.20 m, the current oracle's best hold, which changes the reward
    (``plane_energy-v3``). The failure cost is twice the altitude envelope's
    cost in floor units."""
    spec = registry.get(name)
    p = spec.make_test_params()
    expected = {"plane": (1.0, 2), "plane_sine": (1.26, 2), "plane_energy": (1.2, 3)}
    assert (p.e_floor, spec.version) == pytest.approx(expected[name])
    assert p.failure_cost == pytest.approx(2.0 * (12192.0 / p.e_floor) ** 2)


@pytest.mark.parametrize("name", TASKS)
def test_the_nea_reference_is_the_recorded_hold(name):
    """rho_floor is the oracle's lowest per-seed altitude hold in floor units
    (``scripts/measure_hold.py``)."""
    p = registry.get(name).make_test_params()
    (hold,) = json.loads(HOLDS.read_text())[name]["mpc"]["e_hold_min"]
    assert p.rho_floor == p.rho_floor_tracking
    assert p.rho_floor == pytest.approx((hold / p.e_floor) ** 2, rel=0.02)


@pytest.mark.slow
def test_the_throttle_holds_cruise_on_plane():
    """On protocol seed 1 the planner-driven throttle froze at 0.70 and the
    airspeed sat 27.2 m/s off cruise on average after the protocol's
    burn-in, with the altitude 0.99 m off (oracle audit, 2026-10). The
    shipped planner still flies the capture, and hands over at step 15; from
    there the throttle follows the airspeed on the PID's loop. About two
    minutes."""
    spec = registry.get("plane")
    env = spec.make_env()
    params = spec.make_test_params()
    oracle = spec.make_mpc(env, plan_params(spec, params))
    oracle.reset()
    key = jax.random.PRNGKey(1)
    obs, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)
    speed_err, alt_err, throttle = [], [], []
    for t in range(int(params.max_steps_in_episode)):
        action = np.asarray(oracle.step(obs, state))
        obs, state, _, _, info = step(key, state, action, params)
        assert not bool(info["tripped"])
        if t >= 140:  # the protocol's burn-in on this task
            speed = float(np.hypot(state.x_dot, state.z_dot))
            speed_err.append(abs(speed - float(params.target_speed)))
            alt_err.append(abs(float(state.target_altitude - state.z)))
            throttle.append(float(action[0]))
    assert oracle.handovers == [(15, "hold")]
    assert np.ptp(throttle) > 0.01
    assert np.mean(speed_err) < 10.0
    assert np.mean(alt_err) < 0.8
