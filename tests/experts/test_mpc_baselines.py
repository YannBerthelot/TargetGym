"""Every registered MPC is built and driven.

Before this, four of the sixteen MPC baselines were ever constructed by the
suite. The other twelve -- the reactor, the furnace, the kiln, the drum, the
battery and the rest -- could have stopped building, or started returning
actions the environment cannot accept, and nothing would have said so. An MPC
is one half of what a learned policy is measured against, so a silently broken
one quietly moves the bar.

These construct each MPC, drive it in closed loop for a few steps, and check
the contract the environment relies on: an action of the right shape, inside
the action space, finite, and leaving the plant in a finite state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from target_gym.registry import REGISTRY

MPC_ENVS = [name for name, spec in REGISTRY.items() if spec.has_mpc]
CLOSED_LOOP_STEPS = 3


def _bounds(space, shape):
    low = np.broadcast_to(np.asarray(space.low, dtype=float), shape)
    high = np.broadcast_to(np.asarray(space.high, dtype=float), shape)
    return low, high


@pytest.mark.parametrize("name", MPC_ENVS)
def test_registered_mpc_builds_and_controls(name):
    spec = REGISTRY[name]
    env = spec.make_env()
    params = spec.params_cls(**{**spec.test_params, "max_steps_in_episode": 20})

    mpc = spec.make_mpc(env, params)
    mpc.reset()

    space = env.action_space(params)
    expected = space.shape or (1,)

    key = jax.random.PRNGKey(0)
    _, state = env.reset_env(key, params)
    step = jax.jit(env.step_env)

    for i in range(CLOSED_LOOP_STEPS):
        obs = env.get_obs(state, params)
        action = np.atleast_1d(np.asarray(mpc.step(obs, state), dtype=float))

        assert (
            action.shape == expected
        ), f"{name}: MPC returned {action.shape}, action space is {expected}"
        assert np.all(np.isfinite(action)), f"{name}: non-finite action at step {i}"
        low, high = _bounds(space, action.shape)
        assert np.all(action >= low - 1e-6) and np.all(
            action <= high + 1e-6
        ), f"{name}: action {action} outside [{low}, {high}] at step {i}"

        _, state, reward, terminated, _ = step(key, state, jnp.asarray(action), params)
        assert np.isfinite(float(reward)), f"{name}: non-finite reward at step {i}"
        if bool(terminated):
            break

    for leaf in jax.tree_util.tree_leaves(state):
        assert np.all(
            np.isfinite(np.asarray(leaf))
        ), f"{name}: MPC drove the plant to a non-finite state"


def test_every_environment_without_an_mpc_says_why():
    """A missing baseline must be documented, not silent."""
    undocumented = [
        name
        for name, spec in REGISTRY.items()
        if not spec.has_mpc and not spec.baselines_note
    ]
    assert (
        not undocumented
    ), f"environments with no MPC and no baselines_note: {undocumented}"
