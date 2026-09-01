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


# ─── The quality contract ───────────────────────────────
#
# Everything above checks that an MPC *runs*. Nothing checked that it controls,
# and for a long time several did not: the wind turbine returned -0.02 against
# the PID's 393, the battery -61, the glass furnace -23, and the 2D aircraft flew
# into the ground. All of them passed the tests above, because emitting finite,
# in-bounds actions is exactly what a controller that has given up does.
#
# That gap mattered more than an ordinary baseline bug. The MPC is what a learned
# policy is measured against, so "RL beat MPC" on any of those environments was a
# statement about a broken baseline, not about RL.
#
# This is deliberately a tripwire, not a benchmark. It runs few seeds and each
# environment's own short ``test_params`` episode, so its numbers are not the
# published ones -- those live in docs/baselines.md and were measured over ten
# seeds. What it has to catch is gross regression, and the tolerance is set from
# the real defects: as a fraction of the PID's return those cost -100% (turbine),
# -88% (four-tank), -39% (battery) and -18% (furnace), while the environments
# that legitimately sit level with their PID are within 2.4%. A 15% band
# separates the two populations with room on both sides.
#
# The seed count is low because MPC rollouts are expensive, and that is a real
# limitation: measuring on two seeds produced three wrong conclusions during the
# work that motivated this file, so the parity cases here are checked loosely on
# purpose and the strict comparisons are left to the ten-seed table.

QUALITY_SEEDS = 3
PID_SHORTFALL_TOLERANCE = 0.15


@pytest.mark.slow
@pytest.mark.parametrize("name", MPC_ENVS)
def test_mpc_controls_at_least_as_well_as_the_pid(name):
    """The MPC is presented as an upper bound; hold it to that.

    Two failures in one test, deliberately. They are distinct symptoms -- ending
    the plant, and simply scoring less -- but the MPC rollout is the whole cost
    of this file, and splitting them doubled the slow suite's runtime for no
    extra coverage. A module-level cache would not fix it either: under xdist the
    two tests for one environment can land on different workers.

    Termination is checked first because it is the sharper signal. A controller
    that trips the turbine or flies the aircraft into the ground can still
    average acceptably across seeds, and averaging is exactly what hid it.
    """
    from target_gym.runners.runners import mpc_policy, pid_policy, rollout

    spec = REGISTRY[name]
    if not spec.has_pid:
        pytest.skip(f"{name}: {spec.baselines_note}")
    if spec.mpc_degraded:
        pytest.xfail(f"{name}: {spec.mpc_degraded}")

    params = spec.make_test_params()
    env = spec.make_env()
    horizon = int(params.max_steps_in_episode)
    pid, mpc = [], []
    for seed in range(QUALITY_SEEDS):
        _, _, r = rollout(spec, params, pid_policy(spec), seed)
        pid.append(float(np.sum(r)))
        _, _, r = rollout(spec, params, mpc_policy(spec, env, params), seed)
        mpc.append(float(np.sum(r)))
        assert len(r) >= horizon, (
            f"{name}: MPC ended the episode at step {len(r)} of {horizon} on "
            f"seed {seed} -- the plant reached a terminal state. Terminal "
            f"conditions are reported through a boolean, so a reward penalty "
            f"behind ``where(terminated, ...)`` gives the planner the cost of a "
            f"crash but no gradient away from the boundary; a differentiable "
            f"barrier on the approach is what works (see make_wind_turbine_mpc)."
        )

    pid, mpc = np.array(pid), np.array(mpc)
    # Scale the allowance by the PID's own magnitude, so this reads the same way
    # for a return of 99 and one of 1100.
    allowance = PID_SHORTFALL_TOLERANCE * abs(pid.mean())
    assert mpc.mean() >= pid.mean() - allowance, (
        f"{name}: MPC {mpc.mean():.2f} vs PID {pid.mean():.2f} "
        f"({(mpc.mean() - pid.mean()) / abs(pid.mean()):+.1%}, allowance "
        f"{PID_SHORTFALL_TOLERANCE:.0%}). Per seed: MPC {np.round(mpc, 1).tolist()}, "
        f"PID {np.round(pid, 1).tolist()}. An MPC that has stopped controlling "
        f"still emits finite in-bounds actions, so check its objective before its "
        f"gains: every instance of this so far was an objective the planner could "
        f"not descend (a clipped plateau) or one that did not share the reward's "
        f"minimiser (a mis-scaled or non-monotone term), never a tuning problem."
    )
