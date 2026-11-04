"""A short PPO run learns on one task of each family.

The conformance battery shows every task is well formed, and "the PID beats
the best constant action" shows each can be solved. Neither shows that a
learner can use what a task gives it, which is what the library is for.

Contract, from the owner's direction of 2026-11-04: on the scored reward and
configuration, a short PPO run ends up better than the best constant action on
seeds it never trained on. That is the bar the shipped PID meets in
``tests/test_env_conformance.py``, so no threshold is tuned per task.

It fails when the action stops reaching the plant or the reward stops following
it, on any of these tasks, and every shape and sign test would still pass. What
it says about the observation depends on the task. Hiding the target fails it
on ``cstr``, where a constant is a fair baseline. It does not on ``plane`` or
``battery``, where a constant crashes the aircraft or drains the pack, so
staying alive already beats it.

The tasks are one per registry group. The agent's settings are the same on
each, and a typical run has to clear the bar, so one unlucky seed is not a bug.

Ajax needs JAX 0.10 or later, which ``uv.lock`` gives Python 3.14 alone, so
these run in CI's learning job and skip without the ``learning`` group.
"""

import itertools

import numpy as np
import pytest

import target_gym as tg
from target_gym.benchmark import run_policy, task_info

TASKS = ("plane", "cstr", "glass_furnace", "battery")
TRAINING_SEEDS = (0, 1, 2, 3, 4)
HELD_OUT_SEEDS = tuple(range(200, 210))
CONSTANT_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
# Two policies with the same return differ in the last bits of a float32 sum, so
# a tie must not read as a win. This is rounding, not a bar: a typical run saves
# about 90 to 98 % of the constant's cost.
ROUNDING = 1e-3


class Trained:
    """An Ajax policy as a ``benchmark.Policy``."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, key, obs, task):
        return self.policy.reset(task.n_episodes)

    def act(self, carry, obs, t):
        return self.policy.act(obs, carry)


class Constants:
    """One constant action per episode, as a ``benchmark.Policy``."""

    def __init__(self, actions):
        self.actions = actions

    def reset(self, key, obs, task):
        return None

    def act(self, carry, obs, t):
        return self.actions, carry


def _returns(name, policy, seeds):
    env, params = tg.make(name)
    task = task_info(name, env, params, len(seeds))
    return run_policy(policy, env, params, task, seeds=seeds)["reward"].sum(axis=1)


def _best_constant(name):
    """The best constant action's mean return on the held-out seeds, each
    candidate paired on the same episodes."""
    env, params = tg.make(name)
    grid = np.array(
        list(
            itertools.product(CONSTANT_LEVELS, repeat=env.action_space(params).shape[0])
        )
    )
    actions = np.repeat(grid, len(HELD_OUT_SEEDS), axis=0)
    returns = _returns(name, Constants(actions), HELD_OUT_SEEDS * len(grid))
    return returns.reshape(len(grid), len(HELD_OUT_SEEDS)).mean(axis=1).max()


@pytest.mark.learning
@pytest.mark.parametrize("name", TASKS)
def test_a_short_ppo_run_beats_the_best_constant_action(name):
    ppo = pytest.importorskip("ajax").PPO
    env, params = tg.make(name)
    agent = ppo(
        env,
        env_params=params,
        n_envs=16,
        n_steps=64,
        batch_size=256,
        n_epochs=4,
        normalize_observations=True,
        normalize_rewards=True,
    )
    state, _ = agent.train(seed=list(TRAINING_SEEDS), n_timesteps=16 * 64 * 200)

    trained = np.median(
        [
            _returns(
                name, Trained(agent.policy(state, seed_index=i)), HELD_OUT_SEEDS
            ).mean()
            for i in range(len(TRAINING_SEEDS))
        ]
    )
    best_constant = _best_constant(name)
    assert trained > best_constant + ROUNDING * abs(best_constant), (
        f"{name}: a typical trained policy returns {trained:.4g}, no better than "
        f"the best constant action's {best_constant:.4g}. The agent could not "
        "use the observation, or the reward does not follow from the action."
    )
