"""PPO and SAC at one cheap budget: the learned baselines of the task protocol's
item 4.

Ajax's PPO with the learning check's settings (``tests/rl/test_learning.py``)
and Ajax's SAC with its shipped ones, each trained on 30 seeds at once for
204,800 steps on the task's scored configuration (``tg.make``), on the measured
reward (docs/rl-protocol.md, section 2). No setting is tuned and no checkpoint
chosen: each run's last policy is scored as it ended, so no evaluation seed is
spent. The numbers are reported whatever they are; they are not a pass mark.

    policies, run = train("ph_neutralization", "PPO")
    result = tg.score("ph_neutralization", policies)   # the interval covers runs

Ajax is the opt-in ``learning`` group (Python 3.14), imported only when a run
starts, so installing TargetGym never brings in an RL library.
"""

from __future__ import annotations

import os
import platform
import time

import jax
import jax.numpy as jnp

__all__ = ["BUDGET", "RUNS", "SETTINGS", "MeasuredReward", "Run", "train"]

BUDGET = 204_800
RUNS = 30
#: Only the budget is ours; the rest is the learning check's for PPO and
#: Ajax's own defaults for SAC.
SETTINGS = {
    "PPO": {
        "n_envs": 16,
        "n_steps": 64,
        "batch_size": 256,
        "n_epochs": 4,
        "normalize_observations": True,
        "normalize_rewards": True,
    },
    "SAC": {},
}


def _measured(out):
    *head, info = out
    head[2] = info["measured_reward"]
    return (*head, info)


def MeasuredReward(env):
    """``env`` with ``info["measured_reward"]`` as its reward, the one a
    learner trains on; the score reads the true one."""
    from gymnax.wrappers.purerl import GymnaxWrapper

    class Measured(GymnaxWrapper):
        def __getattr__(self, name):
            # Ajax slides its auto-reset under this layer, and copies it
            # before ``_env`` is set.
            if name == "_env" or name.startswith("__"):
                raise AttributeError(name)
            return getattr(self._env, name)

        def step_env(self, key, state, action, params):
            return _measured(self._env.step_env(key, state, action, params))

        def step(self, key, state, action, params=None):
            return _measured(self._env.step(key, state, action, params))

    return Measured(env)


class Run:
    """One trained run as a policy in the experts' form, which ``tg.score``
    compiles: ``memory = run.reset(n)``, ``action, memory = run.step(memory,
    obs)`` for one episode, vmapped over episodes."""

    def __init__(self, policy):
        self.policy = policy

    def reset(self, n_envs: int | None = None):
        carry = self.policy.reset(1)
        if n_envs is None:
            return carry
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (n_envs, *x.shape)), carry)

    def step(self, memory, obs):
        action, memory = self.policy.act(jnp.asarray(obs)[None], memory)
        return action[0], memory


def _slim(state):
    """The trained state without what only training reads, the replay buffer
    and the last rollout, so a policy compiled on it stays small."""
    collector = state.collector_state
    return state.replace(
        collector_state=collector.replace(buffer_state=None, rollout=None)
    )


def train(task: str, algorithm: str, runs: int = RUNS, steps: int = BUDGET):
    """``(policies, run)``: ``runs`` trained policies of ``algorithm`` (``"PPO"``
    or ``"SAC"``) on ``task``, and what the run was, its training wall-clock and
    the machine it ran on included."""
    import ajax

    import target_gym as tg

    env, params = tg.make(task)
    settings = SETTINGS[algorithm]
    agent = getattr(ajax, algorithm)(MeasuredReward(env), env_params=params, **settings)
    start = time.perf_counter()
    state, _ = agent.train(seed=list(range(runs)), n_timesteps=steps)
    jax.block_until_ready(jax.tree.leaves(state))
    seconds = time.perf_counter() - start
    slim = _slim(state)
    policies = [Run(agent.policy(slim, seed_index=i)) for i in range(runs)]
    return policies, {
        "algorithm": algorithm,
        "library": "ajax",
        "version": ajax.__version__,
        "runs": runs,
        "steps": steps,
        "settings": settings,
        "training_seconds": round(seconds, 1),
        "machine": {
            "cpu": platform.processor() or platform.machine(),
            "cores": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else os.cpu_count()
            ),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
        },
    }
