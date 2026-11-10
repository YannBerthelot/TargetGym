"""A test run that trips the plant is refused or cut, never read across the restart.

Contract, from ``base.failure_kernel``: under the continuing default a trip
restarts the plant inside the same trajectory and reports it only as
``info["tripped"]``. A tuner, bump test or diagnostic that keeps the states and
drops ``info`` reads the restart as the plant's response (the reactor's boron
bump of 2026-11-10, whose fitted gain flipped sign). So every such reader either
raises ``TrippedDuringTest``, stops at the trip, or scores the tripped run as
refused (infinite cost); none returns a reading taken after a restart.

The trips are made on purpose by narrowing a real plant's envelope: the first
order plant (gain 1, input range [-2, 2]) with ``x_max`` 0.5, so any input past
0.5 settles outside it; the aircraft with ``min_alt`` above where they fly.
"""

import importlib
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.identification import Plant, TrippedDuringTest
from target_gym.registry import REGISTRY

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

#: The first order plant's envelope, narrowed so a raw input of 0.25 or more
#: (u = 0.5, the steady x) trips it.
NARROW = {"x_max": 0.5}


def _script(name):
    return importlib.import_module(name)


class _Narrowed:
    """``spec`` with its test params narrowed by ``limits``: the same plant,
    a smaller envelope."""

    def __init__(self, spec, **limits):
        self._spec, self._limits = spec, limits

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def make_test_params(self, **overrides):
        return self._spec.make_test_params(**{**self._limits, **overrides})


def _manual(raw):
    from target_gym.experts.pid import Controller

    return Controller(None, 0.0, lambda _p, s, obs, xp=jnp: (xp.full((1,), raw), s))


# --- target_gym.identification.Plant (the recipes' and the ladder's bump tests)


def test_a_settle_that_trips_is_refused():
    spec = REGISTRY["first_order"]
    p = spec.make_test_params(**NARROW)
    with pytest.raises(TrippedDuringTest):
        Plant(spec, p, [lambda o, s, q: o[0]], [(None, 0.5)], controller=_manual(1.0))


def test_a_bump_that_trips_is_cut_at_the_trip():
    # Settled at x = 0 (raw 0), then a raw step of 0.5 drives x to 1, past 0.5.
    spec = REGISTRY["first_order"]
    p = spec.make_test_params(**NARROW)
    plant = Plant(
        spec, p, [lambda o, s, q: o[0]], [(None, 0.5)], controller=_manual(0.0)
    )
    inst = _script("tune_process_experts").Instrument.of(spec.make_env(), p, "x")
    bump = plant.bump(0, 0, 0, 0.5, 60, inst)
    assert bump.trip is not None and bump.trip[1] == 0
    assert len(bump.dy) < 60
    # Every reading kept is from before the trip, so inside the envelope.
    assert np.all(bump.dy + np.mean(bump.base) < 0.5)


# --- scripts/tune_industrial_experts.py


def test_an_industrial_step_that_trips_is_refused(monkeypatch):
    industrial = _script("tune_industrial_experts")
    monkeypatch.setitem(
        industrial.REGISTRY, "first_order", _Narrowed(REGISTRY["first_order"], **NARROW)
    )

    class Hold:
        channel = 0

        def reset(self):
            pass

        def __call__(self, expert, obs, t, d):
            return np.array([d], np.float32)

    with pytest.raises(TrippedDuringTest):
        industrial.step_response("first_order", 100, Hold(), 20, 20, 1.0)


# --- target_gym.experts.relay_autotune (scripts/tune_pid.py)


def _relay_plant():
    from target_gym import FirstOrderParams, FirstOrderSystem

    env = FirstOrderSystem(integration_method="rk4_1")

    def reset_fn(key, p, t):
        _, state = env.reset_env(key, p)
        return state.replace(target_x=t, x=t)

    return env, FirstOrderParams(**NARROW), reset_fn


def test_a_relay_experiment_that_trips_is_refused():
    from target_gym.experts.relay_autotune import relay_experiment

    env, params, reset_fn = _relay_plant()
    # Biased at raw 0.2 (x settles at 0.4) with a relay of 0.4: the high side
    # drives x towards 1.2, past 0.5, before the error can switch it back.
    with pytest.warns(UserWarning, match="tripped"):
        result = relay_experiment(
            env, params, reset_fn, 0, 1, 0.45, relay_amplitude=0.4,
            max_steps=2000, action_bias=0.2,
        )  # fmt: skip
    assert result["success"] is False and result["Ku"] is None
    assert result["tripped"] is True


def test_a_bias_search_that_trips_is_refused():
    from target_gym.experts.relay_autotune import relay_experiment

    env, params, reset_fn = _relay_plant()
    # No bias given: the search settles the plant at raw +1 (x towards 2).
    with pytest.raises(TrippedDuringTest):
        relay_experiment(env, params, reset_fn, 0, 1, 0.0, max_steps=200)


# --- scripts/measure_time_constants.py


def test_an_open_loop_step_stops_at_the_trip():
    measure = _script("measure_time_constants")
    spec = REGISTRY["first_order"]
    env, params = spec.make_env(), spec.make_test_params(**NARROW)
    # Raw 0.4 is u = 0.8: x climbs past 0.5 well inside the episode.
    ys = measure.step_test(env, params, 0.4)
    assert len(ys) < int(params.max_steps_in_episode) + 1
    assert np.all(ys < 0.5)


# --- scripts/audit_mpc_horizons.py


def test_a_horizon_trace_that_trips_counts_as_a_crash():
    audit = _script("audit_mpc_horizons")
    spec = REGISTRY["first_order"]
    env, p = spec.make_env(), spec.make_test_params(**NARROW)
    key = jax.random.PRNGKey(0)
    _, st = env.reset_env(key, p)
    step = jax.jit(env.step_env)
    _, crashed = audit.trace(env, p, step, key, st, lambda o: 1.0, (0,), (1,))
    assert crashed


# --- scripts/measure_hold.py


def test_a_hold_whose_every_seed_trips_is_refused(monkeypatch):
    hold = _script("measure_hold")
    # Every first order target is drawn in [0.5, 1.5] (``target_x_range``), so
    # the shipped PID drives each seed through the narrowed limit.
    narrow = _Narrowed(REGISTRY["first_order"], **NARROW)
    monkeypatch.setitem(hold.REGISTRY, "first_order", narrow)
    with pytest.raises(TrippedDuringTest):
        hold.measure("first_order", 2, ("pid",))


# --- scripts/tune_aircraft_guidance.py


def _aircraft():
    spec = importlib.util.spec_from_file_location(
        "_tune_aircraft_guidance", ROOT / "scripts" / "tune_aircraft_guidance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_guidance_rollout_that_trips_scores_infinite_from_the_trip():
    tune = _aircraft()
    params = REGISTRY["plane"].make_test_params(min_alt=12_000.0)  # below the floor
    _, track = tune.rollout(
        "plane", tune.P2.plane_tecs_step, tune.P2.PlaneTECSParams(), [100], params
    )
    assert np.isinf(track[0]).all()


def test_a_flight_test_that_trips_is_refused(monkeypatch):
    tune = _aircraft()
    narrow = _Narrowed(REGISTRY["plane3d_heading"], min_alt=10_000.0)
    monkeypatch.setitem(tune.REGISTRY, "plane3d_heading", narrow)
    with pytest.raises(TrippedDuringTest):
        tune.flight_test()


# --- scripts/record_credibility.py (the null policies' feedforward table)


def test_a_target_cycle_with_a_trip_gives_no_settled_point():
    credibility = _script("record_credibility")
    # One episode, two cycles of 8 steps (targets 1 then 2, actions 10 then
    # 20); the second cycle trips on its 3rd step, so only the first settles.
    target = np.r_[np.full(8, 1.0), np.full(8, 2.0)][None, :, None]
    action = np.r_[np.full(8, 10.0), np.full(8, 20.0)][None, :, None]
    change = np.zeros((1, 16), bool)
    change[0, 8] = True
    tripped = np.zeros((1, 16), bool)
    tripped[0, 10] = True
    rollout = {
        "length": np.array([16]),
        "target_change": change,
        "target": target,
        "action": action,
        "tripped": tripped,
    }
    targets, actions = credibility.settled_points(rollout)
    np.testing.assert_array_equal(targets, [[1.0]])
    np.testing.assert_array_equal(actions, [[10.0]])
