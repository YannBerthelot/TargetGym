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

The agent trains on ``info["measured_reward"]``, the reward the protocol trains a
learner on (docs/rl-protocol.md, section 2: the owner's measurement-noise
decision 3 of 2026-11-05), and is read on the true reward, as the score is. It is
read on validation seeds, never the scoring seeds 200 to 209 (review M6).

Ajax needs JAX 0.10 or later, which ``uv.lock`` gives Python 3.14 alone, so
these run in CI's learning job and skip without the ``learning`` group.
"""

import importlib.metadata
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest
from gymnax.wrappers.purerl import GymnaxWrapper

import target_gym as tg
from target_gym.benchmark import Unsupported, run_policy, shipped_policy, task_info
from target_gym.charter import FLAGSHIP, VALIDATION_SEEDS

TASKS = tuple(dict.fromkeys(("plane", "cstr", "glass_furnace", "battery", *FLAGSHIP)))
TRAINING_SEEDS = (0, 1, 2, 3, 4)
HELD_OUT_SEEDS = VALIDATION_SEEDS[:10]
CONSTANT_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
TIMESTEPS = 16 * 64 * 200


def _measured(out):
    """A step's outputs, ``(obs, state, reward, ..., info)``, with the measured
    reward in the reward's place, whatever flags the layer below returns."""
    *head, info = out
    head[2] = info["measured_reward"]
    return (*head, info)


class MeasuredReward(GymnaxWrapper):
    """The env with ``info["measured_reward"]`` as its reward. Ajax slides its own
    auto-reset under this layer, and copies it before ``_env`` is set."""

    def __getattr__(self, name):
        if name == "_env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._env, name)

    def step_env(self, key, state, action, params):
        return _measured(self._env.step_env(key, state, action, params))

    def step(self, key, state, action, params=None):
        return _measured(self._env.step(key, state, action, params))


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
    ajax = pytest.importorskip("ajax")
    env, params = tg.make(name)
    agent = ajax.PPO(
        MeasuredReward(env),
        env_params=params,
        n_envs=16,
        n_steps=64,
        batch_size=256,
        n_epochs=4,
        normalize_observations=True,
        normalize_rewards=True,
    )
    state, _ = agent.train(seed=list(TRAINING_SEEDS), n_timesteps=TIMESTEPS)
    runs = [
        float(
            _returns(
                name, Trained(agent.policy(state, seed_index=i)), HELD_OUT_SEEDS
            ).mean()
        )
        for i in range(len(TRAINING_SEEDS))
    ]
    action, constant = _best_constant(name)
    expert = _expert(name)
    report = {
        "task": name,
        "ajax": importlib.metadata.version("ajax-rl"),
        "timesteps": TIMESTEPS,
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
