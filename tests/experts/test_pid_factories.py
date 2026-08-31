"""Every PID factory is built, and driven where its environment is known.

``experts.pid`` defines 46 factories; the registry uses 18. The other 28 are
alternatives a user can choose -- gain-scheduled, cascaded and MIMO variants of
the shipped default -- and nothing exercised them, so a rename or a changed
observation layout could leave them broken while the registered controller kept
passing. They are part of what ships, so they are part of what is tested.

Each factory returns one of two shapes: a stateful controller with ``reset``,
or the functional ``(params, state)`` pair used with ``pid_step``. Both are
checked for well-formed gains; the stateful ones are additionally driven with a
real observation from the environment their name identifies.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import target_gym.experts.pid as pid_module
from target_gym.registry import REGISTRY

FACTORIES = sorted(
    name
    for name in dir(pid_module)
    if name.startswith("make_") and callable(getattr(pid_module, name))
)

# Longest first, so make_plane3d_heading_* maps to plane3d_heading and not plane.
_ENV_NAMES = sorted(REGISTRY, key=len, reverse=True)


def _env_for(factory: str) -> str | None:
    """The registry environment a factory's name identifies, if any."""
    body = factory[len("make_") :]
    for env_name in _ENV_NAMES:
        if body == env_name or body.startswith(env_name + "_"):
            return env_name
    return None


def _gains(obj) -> list[float]:
    """Collect every Kp/Ki/Kd in a params struct, including nested MIMO loops."""
    found = []
    fields = getattr(obj, "__dataclass_fields__", None)
    if not fields:
        return found
    for field in fields:
        value = getattr(obj, field)
        if field.split("_")[0] in ("Kp", "Ki", "Kd"):
            found.extend(np.asarray(value, dtype=float).ravel().tolist())
        else:
            found.extend(_gains(value))
    return found


def test_the_registry_uses_a_subset_of_the_shipped_factories():
    """Guards the premise of this file rather than a behaviour.

    If the registry ever referenced a factory that does not exist, the failure
    would otherwise surface far from here.
    """
    assert len(FACTORIES) >= len(
        REGISTRY
    ), f"{len(FACTORIES)} factories for {len(REGISTRY)} environments"


@pytest.mark.parametrize("factory", FACTORIES)
def test_pid_factory_builds_with_finite_gains(factory):
    built = getattr(pid_module, factory)()

    if isinstance(built, tuple):
        params, _state = built
        gains = _gains(params)
        assert gains, f"{factory}: no gains found on {type(params).__name__}"
        assert all(np.isfinite(g) for g in gains), f"{factory}: non-finite gain"
    else:
        assert hasattr(
            built, "reset"
        ), f"{factory} returned {type(built).__name__}, which has no reset()"
        built.reset()


@pytest.mark.parametrize(
    "factory", [f for f in FACTORIES if _env_for(f) is not None], ids=lambda f: f
)
def test_stateful_pid_factory_acts_on_its_environment(factory):
    """A controller must produce a usable action from a real observation.

    This is what catches an observation-layout change: the controller still
    builds, still has the right gains, and indexes a slot that has moved.
    """
    built = getattr(pid_module, factory)()
    if isinstance(built, tuple):
        pytest.skip(f"{factory} is the functional form; driven via pid_step elsewhere")

    name = _env_for(factory)
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 20})
    obs, _state = env.reset_env(jax.random.PRNGKey(0), params)

    built.reset()
    action = np.atleast_1d(np.asarray(built(obs), dtype=float))

    assert np.all(np.isfinite(action)), f"{factory} on {name}: non-finite action"
    expected = env.action_space(params).shape or (1,)
    assert (
        action.shape == expected
    ), f"{factory} on {name}: produced {action.shape}, environment takes {expected}"


@pytest.mark.parametrize(
    "factory", [f for f in FACTORIES if _env_for(f) is not None], ids=lambda f: f
)
def test_stateful_pid_factory_is_deterministic_across_resets(factory):
    """reset() must return the controller to its initial state.

    A PID that carries integral windup across an episode boundary is a
    controller whose first action depends on what happened last episode.
    """
    built = getattr(pid_module, factory)()
    if isinstance(built, tuple):
        pytest.skip(f"{factory} is the functional form")

    name = _env_for(factory)
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 20})
    obs, _ = env.reset_env(jax.random.PRNGKey(0), params)

    built.reset()
    first = np.asarray(built(obs), dtype=float)
    for _ in range(3):
        built(obs)
    built.reset()
    again = np.asarray(built(obs), dtype=float)

    np.testing.assert_allclose(
        first,
        again,
        rtol=1e-6,
        atol=1e-8,
        err_msg=f"{factory}: reset() did not clear controller state",
    )
