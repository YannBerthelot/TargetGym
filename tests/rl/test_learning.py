"""The learning check (charter J7): a short PPO run on one task of each family
and on every scored task, reported beside the best constant action and the
shipped expert.

The conformance battery shows every task is well formed, and "the PID beats
the best constant action" shows each can be solved. This check shows what a
learner with fixed settings makes of a task, on seeds it never trained on.

Contract, from charter J7 and the owner's decision of 2026-11-27: the check is
reported, never gating. Each task's run records its five trained policies' mean
returns, the best constant action's and the shipped expert's, on the same seeds
and reward. ``TG_LEARNING_REPORT`` names a folder for one JSON file per task,
which CI's learning job tables in its summary. The check fails only on what is
a bug whatever the learner: a return that is not finite, which a plant gives
when it turns NaN under an action inside its box. It gated on beating the best
constant from 2026-11-04 until then.

The tasks are one per registry group and the scored tasks (charter J7; the
outside review's L4, 2026-11-27). The agent's settings are the same on each.

The runs are ``target_gym.learned.train``'s, the runner of the v1 record's PPO
line, at its budget and settings: training seeds 0 to 4 on ``tg.make``'s
configuration and ``info["measured_reward"]``, the reward the protocol trains a
learner on (docs/rl-protocol.md, section 2: the owner's measurement-noise
decision 3 of 2026-11-05). They are read on the true reward, as the score is,
on validation seeds, never the scoring seeds 200 to 209 (review M6).

Ajax needs JAX 0.10 or later, which ``uv.lock`` gives Python 3.14 alone, so
these run in CI's learning job and skip without the ``learning`` group.
"""

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

import target_gym as tg
from target_gym.benchmark import Unsupported, run_policy, shipped_policy, task_info
from target_gym.charter import FLAGSHIP, VALIDATION_SEEDS
from target_gym.learned import train

TASKS = tuple(dict.fromkeys(("plane", "cstr", "glass_furnace", "battery", *FLAGSHIP)))
RUNS = 5
HELD_OUT_SEEDS = VALIDATION_SEEDS[:10]
CONSTANT_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)


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
    """The best constant action and its mean return on the held-out seeds, each
    candidate paired on the same episodes. A NaN return wins the argmax, so the
    report names the constant that gave it."""
    env, params = tg.make(name)
    grid = np.array(
        list(
            itertools.product(CONSTANT_LEVELS, repeat=env.action_space(params).shape[0])
        )
    )
    actions = np.repeat(grid, len(HELD_OUT_SEEDS), axis=0)
    returns = _returns(name, Constants(actions), HELD_OUT_SEEDS * len(grid))
    means = returns.reshape(len(grid), len(HELD_OUT_SEEDS)).mean(axis=1)
    return grid[np.argmax(means)].tolist(), float(means.max())


def _expert(name):
    """The shipped expert's mean return on the held-out seeds, None where the
    task ships none."""
    try:
        return float(_returns(name, shipped_policy("pid"), HELD_OUT_SEEDS).mean())
    except Unsupported:
        return None


@pytest.mark.learning
@pytest.mark.parametrize("name", TASKS)
def test_a_short_ppo_run_is_reported_beside_the_constant_and_the_expert(name):
    pytest.importorskip("ajax")
    policies, run = train(name, "PPO", runs=RUNS)
    runs = [
        float(_returns(name, Trained(p.policy), HELD_OUT_SEEDS).mean())
        for p in policies
    ]
    action, constant = _best_constant(name)
    expert = _expert(name)
    report = {
        "task": name,
        "ajax": run["version"],
        "timesteps": run["steps"],
        "training_seconds": run["training_seconds"],
        "seeds": [HELD_OUT_SEEDS[0], HELD_OUT_SEEDS[-1]],
        "runs": runs,
        "median": float(np.median(runs)),
        "best_constant": {"action": action, "return": constant},
        "expert": expert,
    }
    if folder := os.environ.get("TG_LEARNING_REPORT"):
        Path(folder).mkdir(parents=True, exist_ok=True)
        (Path(folder) / f"{name}.json").write_text(json.dumps(report))

    returns = [*runs, constant] + ([] if expert is None else [expert])
    assert np.isfinite(returns).all(), (
        f"{name}: a return is not finite: {report}. A plant that turns NaN under "
        "an action inside its box is a bug, whatever the learner."
    )
