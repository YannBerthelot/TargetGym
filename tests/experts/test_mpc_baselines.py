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
# What this catches, measured by reverting each fix and re-running:
#
#   wind turbine surrogate removed  -> caught (terminates at step 20 of 400)
#   battery surrogate removed       -> caught (-27.5% of the PID)
#   glass furnace scale reverted    -> NOT caught
#
# The furnace case is the honest limit. That bug costs -17.6% of the PID over ten
# seeds but only -4.4% over the five this can afford, against +3.7% when fixed --
# an eight-point gap that no sane tolerance separates. Tightening the threshold
# until it caught this one bug would be fitting the test to a known answer and
# would make it flaky. A subtle objective error is below this contract's
# resolution; the ten-seed table in docs/baselines.md is what finds those.
#
# The tolerance is set from the defects rather than chosen. As a fraction of the
# PID's return the gross failures cost -100% (turbine), -88% (four tank) and
# -27% (battery), while every environment that legitimately sits level with its
# PID is within 4% once fixed. 10% separates those two populations with room on
# both sides.
#
# Five seeds, not two or three. Two seeds hides the battery entirely -- its MPC
# scored 277 on seed 0 and 65 on seed 1, so the average of the two looks healthy
# while the ten-seed truth is -61. That is the same trap that produced three
# wrong conclusions in the work this file came from, and it applies to the
# tripwire, not only to the ranking.
#
# Episodes are capped at 250 steps to bound the cost. The floor is set by the
# four-tank, whose tracking error takes ~198 steps to close: capped at 100 it
# fails not because the MPC is broken but because the episode ends before the
# controller's advantage exists.
#
# Cost: about nine minutes of the slow job's thirteen. It runs only there --
# CI's matrix build uses -m "not slow" -- so the fast job is untouched.

QUALITY_SEEDS = 5
CONTRACT_STEPS = 250
PID_SHORTFALL_TOLERANCE = 0.10


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

    horizon = min(int(spec.make_test_params().max_steps_in_episode), CONTRACT_STEPS)
    params = spec.make_test_params(max_steps_in_episode=horizon)
    env = spec.make_env()
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
