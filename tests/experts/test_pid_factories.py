"""Every shipped PID controller is built, and driven on its own environment.

Each registry entry names one ``make_<env>_pid_controller`` in its package's
``experts.py``, returning a ``Controller``. ``test_registry_rules`` checks that
its jitted form acts like the scored host form; this file checks the things
that comparison cannot see: constants that are present and finite, an action of
the shape the environment takes, and a ``reset()`` that really clears state.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from target_gym.experts.pid import Controller
from target_gym.registry import REGISTRY

WITH_PID = sorted(name for name, spec in REGISTRY.items() if spec.has_pid)


def _parameters(obj) -> list[float]:
    """Every numeric constant in a params struct, nested loops included."""
    fields = getattr(obj, "__dataclass_fields__", None)
    if not fields:
        return []
    found = []
    for field in fields:
        value = getattr(obj, field)
        if isinstance(value, (bool, str)) or value is None:
            continue
        if isinstance(value, (int, float, np.ndarray, np.number)):
            found.extend(np.asarray(value, dtype=float).ravel().tolist())
        else:
            found.extend(_parameters(value))
    return found


def _first_obs(name):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.make_test_params(max_steps_in_episode=20)
    obs, _state = env.reset_env(jax.random.PRNGKey(0), params)
    return env, params, obs


@pytest.mark.parametrize("name", WITH_PID)
def test_expert_builds_with_finite_parameters(name):
    """Not every expert is a PID (the reactor runs Westinghouse's rod speed
    program), so the check is on every numeric constant it carries."""
    ctrl = REGISTRY[name].make_pid_controller()
    assert isinstance(ctrl, Controller), f"{name}: got {type(ctrl).__name__}"
    values = _parameters(ctrl.params)
    assert values, f"{name}: no constants found on {type(ctrl.params).__name__}"
    assert all(np.isfinite(v) for v in values), f"{name}: non-finite constant"


@pytest.mark.parametrize("name", WITH_PID)
def test_pid_acts_on_its_environment(name):
    """A controller must produce a usable action from a real observation.

    This is what catches an observation-layout change: the controller still
    builds, still has the right gains, and indexes a slot that has moved.
    """
    env, params, obs = _first_obs(name)
    pid = REGISTRY[name].make_pid()
    action = np.atleast_1d(np.asarray(pid(obs), dtype=float))

    assert np.all(np.isfinite(action)), f"{name}: non-finite action"
    expected = env.action_space(params).shape or (1,)
    assert (
        action.shape == expected
    ), f"{name}: produced {action.shape}, environment takes {expected}"


@pytest.mark.parametrize("name", WITH_PID)
def test_pid_is_deterministic_across_resets(name):
    """reset() must return the controller to its initial state.

    A PID that carries integral windup across an episode boundary is a
    controller whose first action depends on what happened last episode.
    """
    _env, _params, obs = _first_obs(name)
    pid = REGISTRY[name].make_pid()

    first = np.asarray(pid(obs), dtype=float)
    for _ in range(3):
        pid(obs)
    pid.reset()
    again = np.asarray(pid(obs), dtype=float)

    np.testing.assert_allclose(
        first,
        again,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"{name}: reset() did not clear controller state",
    )


def test_the_tuned_gains_have_public_names():
    """Contract: ``load_gains()`` is the tuned-gains file as stored, and
    ``gain(task, name, default)`` is that task's stored value as a float, or
    the default when the file has none. Expected values come from the file
    itself (``provenance.GAINS_PATH``), not from the loader."""
    import json

    from target_gym.experts.pid import gain, load_gains
    from target_gym.provenance import GAINS_PATH

    stored = json.loads(GAINS_PATH.read_text())
    assert load_gains() == stored
    assert gain("cstr", "Kp", 0.0) == float(stored["cstr"]["Kp"])
    assert gain("cstr", "no_such_gain", 1.5) == 1.5
